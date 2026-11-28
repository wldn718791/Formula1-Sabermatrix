"""
Pole2Stat v3 시즌별 아카이브 + 증분 업데이트 스크립트 (2006~2026 현대 F1 21시즌)

6단계 랩타임 기반 동적 상태공간 엔진(`f1_statespace_v3.py`)을 실행해 두 가지 CSV를 생성합니다:
- `data/f1_war_rankings.csv` : 현재 폼 평가 (반감기 1년 지수 감쇠 + 상태공간)
- `data/f1_war_career.csv`   : 커리어 평가 (전 시즌 동등 가중 상태공간)
- `data/model_meta.json`     : 분산 분해·잔차 σ 등 메타 정보
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from f1_statespace_v3 import build_v3_rankings, generate_v3_demo
from fastf1_enricher import enrich_jolpica_with_fastf1
from jolpica_f1_loader import fetch_recent_seasons_df
from run_f1_sabermetrics_pipeline import attach_salary_efficiency


def _load_or_fetch_season(
    season: int,
    seasons_dir: Path,
    is_active: bool,
    skip_fastf1: bool,
    cache_dir: str,
    force_refresh: bool,
) -> pd.DataFrame:
    season_path = seasons_dir / f"raw_{season}.csv"

    if not is_active and season_path.exists() and not force_refresh:
        df = pd.read_csv(season_path)
        if "race_pace_sec_adv" in df.columns:
            return df

    print(f"[수집] {season} 시즌 Jolpica 퀄리(Q1/Q2/Q3) + 레이스 평균 랩타임 수집 중...")
    df = fetch_recent_seasons_df([season], include_qualifying=True)
    if df.empty:
        return pd.read_csv(season_path) if season_path.exists() else pd.DataFrame()

    if not skip_fastf1 and season >= 2018:
        df = enrich_jolpica_with_fastf1(
            df, cache_dir=cache_dir, use_lap_pace_as_target=True, include_qualifying=True
        )

    df.to_csv(season_path, index=False)
    print(f"[저장] {season} 시즌 {len(df)}행 -> {season_path}")
    return df


def _attach_driver_bio(rankings: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    """API에서 추출한 풀네임·국적·시즌 등번호를 driver_id 기준으로 결합합니다."""
    name_map, nat_map, num_map = {}, {}, {}
    if "driver_fullname" in combined.columns:
        for did, grp in combined.dropna(subset=["driver_fullname"]).groupby("driver_id"):
            vals = [str(v).strip() for v in grp["driver_fullname"] if str(v).strip()]
            if vals:
                name_map[str(did)] = vals[-1]
    if "driver_nationality" in combined.columns:
        for did, grp in combined.dropna(subset=["driver_nationality"]).groupby("driver_id"):
            vals = [str(v).strip() for v in grp["driver_nationality"] if str(v).strip()]
            if vals:
                nat_map[str(did)] = vals[-1]
    if "car_number" in combined.columns:
        for (s, did), grp in combined.dropna(subset=["car_number"]).groupby(["season", "driver_id"]):
            nums = [str(v).strip().removesuffix(".0") for v in grp["car_number"] if str(v).strip() not in {"", "nan", "None"}]
            if nums:
                num_map[(int(s), str(did))] = pd.Series(nums).mode().iloc[0]

    out = rankings.copy()
    out["driver_fullname"] = out["driver_id"].map(lambda d: name_map.get(str(d), ""))
    out["driver_nationality"] = out["driver_id"].map(lambda d: nat_map.get(str(d), ""))
    out["car_number"] = [num_map.get((int(r.season), str(r.driver_id)), "") for r in out.itertuples(index=False)]

    cols = [
        "season", "driver_id", "driver", "car_number", "driver_fullname", "driver_nationality",
        "team", "car_tier", "starts", "counting_races",
        "delta_q", "delta_r", "pace_sec_adv", "pace_se_sec", "car_sec_adv", "is_uncertain",
        "f1_par", "f1_war", "f1_war_ci_low", "f1_war_ci_high",
        "dn_par", "dn_war", "dn_war_ci_low", "dn_war_ci_high",
    ]
    return out[cols]


def update_incremental(
    start_year: int = 2006,
    active_season: int = 2026,
    data_dir: str = "data",
    cache_dir: str = "./fastf1_cache",
    skip_fastf1: bool = True,
    force_refresh_archive: bool = False,
    use_demo: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(data_dir)
    seasons_dir = out_dir / "seasons"
    seasons_dir.mkdir(parents=True, exist_ok=True)

    rankings_path = out_dir / "f1_war_rankings.csv"
    career_path = out_dir / "f1_war_career.csv"
    salaries_path = out_dir / "salaries.csv"

    if use_demo:
        combined = generate_v3_demo(n_races_per_season=20, seed=42)
        combined["driver_fullname"] = combined["driver"]
        combined["driver_nationality"] = ""
        combined["car_number"] = ""
    else:
        frames = []
        all_years = [active_season] + list(range(active_season - 1, start_year - 1, -1))
        for yr in all_years:
            try:
                sdf = _load_or_fetch_season(
                    season=yr,
                    seasons_dir=seasons_dir,
                    is_active=(yr == active_season),
                    skip_fastf1=skip_fastf1,
                    cache_dir=cache_dir,
                    force_refresh=force_refresh_archive,
                )
                if not sdf.empty:
                    frames.append(sdf)
            except Exception as exc:
                print(f"[경고] {yr} 시즌 수집 중단({exc}) -> 기존 확보 시즌으로 계속 진행")
                if "429" in str(exc) or "rate" in str(exc).lower():
                    for rem_yr in range(yr - 1, start_year - 1, -1):
                        p = seasons_dir / f"raw_{rem_yr}.csv"
                        if p.exists():
                            frames.append(pd.read_csv(p))
                    break
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if combined.empty:
        raise RuntimeError("수집된 랩타임 데이터가 없어 v3 랭킹을 계산할 수 없습니다.")

    if "driver_id" not in combined.columns or combined["driver_id"].isna().all():
        combined["driver_id"] = combined["driver"].astype(str)
    combined["driver_id"] = combined["driver_id"].fillna(combined["driver"]).astype(str)

    # 1) 현재 폼 평가 (반감기 1년 지수 감쇠 + 동적 상태공간)
    cur_rank, cur_meta = build_v3_rankings(combined, half_life_years=1.0, n_sims=2000, seed=42)
    cur_rank = _attach_driver_bio(cur_rank, combined)

    # 2) 커리어 평가 (전 시즌 동등 가중 동적 상태공간)
    car_rank, car_meta = build_v3_rankings(combined, half_life_years=None, n_sims=2000, seed=42)
    car_rank = _attach_driver_bio(car_rank, combined)

    if salaries_path.exists():
        sal_df = pd.read_csv(salaries_path)
        cur_rank = attach_salary_efficiency(cur_rank, sal_df)
        car_rank = attach_salary_efficiency(car_rank, sal_df)

    cur_rank.to_csv(rankings_path, index=False)
    car_rank.to_csv(career_path, index=False)

    seasons_done = sorted(int(x) for x in cur_rank["season"].unique())
    meta_payload = {
        "engine": "v3 Dynamic State-Space Lap-Time Model",
        "half_life_years": 1.0,
        "car_variance_share_pct": cur_meta["car_variance_share_pct"],
        "driver_variance_share_pct": cur_meta["driver_variance_share_pct"],
        "residual_sigma_sec": cur_meta["residual_sigma_sec"],
        "n_races": int(combined["race_id"].nunique()),
        "seasons_range": f"{seasons_done[0]}–{seasons_done[-1]}",
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Pole2Stat v3 업데이트 완료 (랩타임 동적 상태공간 엔진) ===")
    print(f"분산 분해: 레이스카 {cur_meta['car_variance_share_pct']}% | 드라이버 {cur_meta['driver_variance_share_pct']}% | 잔차 σ={cur_meta['residual_sigma_sec']}초/랩")
    print(f"저장 완료: {rankings_path} (현재 폼) & {career_path} (커리어 평가) [{seasons_done[0]}~{seasons_done[-1]}]")
    return cur_rank, car_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Pole2Stat v3 랩타임 동적 상태공간 업데이트")
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--active-season", type=int, default=2026)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--cache-dir", type=str, default="./fastf1_cache")
    parser.add_argument("--use-fastf1", action="store_true")
    parser.add_argument("--force-refresh-archive", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    update_incremental(
        start_year=args.start_year,
        active_season=args.active_season,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        skip_fastf1=not args.use_fastf1,
        force_refresh_archive=args.force_refresh_archive,
        use_demo=args.demo,
    )


if __name__ == "__main__":
    main()
