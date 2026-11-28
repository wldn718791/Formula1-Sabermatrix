"""
Pole2Stat v2 시즌별 아카이브 + 증분 업데이트 스크립트 (2006~2026 현대 F1 21시즌)

- 확정 시즌(2006~2025): 시즌마다 `data/seasons/raw_YYYY.csv`로 개별 저장해 재수집을 방지합니다.
  중간에 API 호출 한도(429)가 걸려도 이미 받은 시즌은 보존되며, 다음 실행 시 남은 시즌만 이어서 받습니다.
- 현재 시즌(2026): 매주 최신 경기 결과와 예선 Q1/Q2/Q3를 반영해 `data/seasons/raw_2026.csv`를 갱신합니다.
- Plackett-Luce 순위 로짓 + Gumbel-Max 반사실적 시뮬레이션 + 95% 신뢰구간 + 드라이버 풀네임/국적을
  `data/f1_war_rankings.csv`에 출력합니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from f1_multilevel_prototype import preprocess_f1_dataframe
import json
from f1_plackett_luce_v2 import compute_bootstrap_ci, generate_plackett_luce_demo, tune_lambdas_temporal_cv
from fastf1_enricher import enrich_jolpica_with_fastf1
from jolpica_f1_loader import fetch_recent_seasons_df
from run_f1_sabermetrics_pipeline import _fit_best_multilevel, attach_salary_efficiency


def _assign_tier_label_for_ui(rankings: pd.DataFrame) -> pd.Series:
    """UI 필터링 전용 티어 라벨(모델 계산 배율에는 사용하지 않음)."""
    tiers = pd.Series("중상위권", index=rankings.index)
    for _, grp in rankings.groupby("season"):
        q_hi = float(grp["car_logit"].quantile(0.70))
        q_lo = float(grp["car_logit"].quantile(0.30))
        tiers.loc[grp.index[grp["car_logit"] >= q_hi]] = "우승권"
        tiers.loc[grp.index[grp["car_logit"] <= q_lo]] = "하위권"
    return tiers


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
        return pd.read_csv(season_path)

    print(f"[수집] {season} 시즌 Jolpica 결승 + 예선(Q1/Q2/Q3) 수집 중...")
    df = fetch_recent_seasons_df([season], include_qualifying=True)
    if df.empty:
        return pd.read_csv(season_path) if season_path.exists() else pd.DataFrame()

    # FastF1은 2018년 이후 시즌만 공식 지원
    if not skip_fastf1 and season >= 2018:
        if is_active and season_path.exists() and not force_refresh:
            existing = pd.read_csv(season_path)
            done = set(existing["race_id"].unique()) if not existing.empty else set()
            new_rows = df[~df["race_id"].isin(done)].copy()
            if new_rows.empty and not existing.empty:
                return existing
            enriched_new = enrich_jolpica_with_fastf1(
                new_rows, cache_dir=cache_dir, use_lap_pace_as_target=True, include_qualifying=True
            )
            df = pd.concat([existing, enriched_new], ignore_index=True) if not existing.empty else enriched_new
        else:
            df = enrich_jolpica_with_fastf1(
                df, cache_dir=cache_dir, use_lap_pace_as_target=True, include_qualifying=True
            )

    df.to_csv(season_path, index=False)
    print(f"[저장] {season} 시즌 {len(df)}행 -> {season_path}")
    return df


def update_incremental(
    start_year: int = 2006,
    active_season: int = 2026,
    data_dir: str = "data",
    cache_dir: str = "./fastf1_cache",
    skip_fastf1: bool = True,
    force_refresh_archive: bool = False,
    n_bootstrap: int = 25,
    use_demo: bool = False,
) -> pd.DataFrame:
    out_dir = Path(data_dir)
    seasons_dir = out_dir / "seasons"
    seasons_dir.mkdir(parents=True, exist_ok=True)

    rankings_path = out_dir / "f1_war_rankings.csv"
    salaries_path = out_dir / "salaries.csv"

    if use_demo:
        raw_df = generate_plackett_luce_demo(n_races_per_season=20, seed=42)
        rng = np.random.default_rng(77)
        raw_df["is_wet"] = 0
        raw_df["quali_pace_pct"] = (11 - raw_df["finish_position"]) * 0.08 + rng.normal(0, 0.08, len(raw_df))
        raw_df["driver_fullname"] = raw_df["driver"]
        raw_df["driver_nationality"] = ""
        frames = [raw_df]
    else:
        frames = []
        # 최신 시즌부터 역순으로 수집해 중간에 한도가 걸려도 최근 시즌은 우선 확보
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
                print(f"[경고] {yr} 시즌 수집 중단({exc}) -> 이미 확보된 시즌으로 랭킹 생성 계속")
                if "429" in str(exc) or "rate" in str(exc).lower():
                    # 남은 로컬 캐시 파일이 있으면 함께 로드
                    for rem_yr in range(yr - 1, start_year - 1, -1):
                        p = seasons_dir / f"raw_{rem_yr}.csv"
                        if p.exists():
                            frames.append(pd.read_csv(p))
                    break

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        raise RuntimeError("수집된 레이스 데이터가 없어 랭킹을 계산할 수 없습니다.")

    if "driver_id" not in combined.columns or combined["driver_id"].isna().all():
        combined["driver_id"] = combined["driver"].astype(str)
    combined["driver_id"] = combined["driver_id"].fillna(combined["driver"]).astype(str)

    # 드라이버 고유 ID(driver_id)별 풀네임·국적·시즌 등번호 매핑 (동명이인/동일 3글자 코드 충돌 완전 차단)
    name_map: dict[str, str] = {}
    nat_map: dict[str, str] = {}
    num_map: dict[tuple[int, str], str] = {}
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

    # 1) 시계열 교차 검증으로 최적 λ 자동 탐색 후 Plackett-Luce + 95% 부트스트랩 실행
    clean_df = combined[combined.get("mechanical_dnf", 0) == 0].dropna(subset=["finish_position"]).copy()
    cv_meta = tune_lambdas_temporal_cv(clean_df)
    print(
        f"[시계열 CV 최적 λ] λ_d={cv_meta['lambda_d']}, λ_dy={cv_meta['lambda_dy']}, "
        f"λ_c={cv_meta['lambda_c']} (OOS Log-Loss={cv_meta['val_log_loss']})"
    )
    rankings, pl_fit = compute_bootstrap_ci(
        clean_df,
        raw_df=combined,
        n_bootstrap=n_bootstrap,
        lambda_d=cv_meta["lambda_d"],
        lambda_dy=cv_meta["lambda_dy"],
        lambda_c=cv_meta["lambda_c"],
    )

    # 2) 퀄리 ΔQ 다층 모형 병합 (driver_id 기준)
    if "quali_pace_pct" in combined.columns and combined["quali_pace_pct"].notna().sum() >= 20:
        q_work = combined.dropna(subset=["quali_pace_pct"]).copy()
        q_work["driver"] = q_work["driver_id"]
        q_work["pace_score"] = q_work["quali_pace_pct"]
        q_work["mechanical_dnf"] = 0
        if "is_wet" not in q_work.columns:
            q_work["is_wet"] = 0
        q_clean = preprocess_f1_dataframe(q_work)
        q_ds, _, _, _, _ = _fit_best_multilevel(q_clean)
        q_repl = {int(s): float(np.quantile(g["est_driver_season_skill"], 0.20)) for s, g in q_ds.groupby("season")}
        q_map = {
            (int(r.season), str(r.driver)): round(float(r.est_driver_season_skill) - q_repl[int(r.season)], 3)
            for r in q_ds.itertuples(index=False)
        }
        rankings["delta_q"] = [q_map.get((int(r.season), str(r.driver_id)), np.nan) for r in rankings.itertuples(index=False)]
    else:
        rankings["delta_q"] = np.nan

    rankings["delta_r"] = rankings["skill_logit"]
    rankings["car_tier"] = _assign_tier_label_for_ui(rankings)
    rankings["driver_fullname"] = rankings["driver_id"].map(lambda d: name_map.get(str(d), ""))
    rankings["driver_nationality"] = rankings["driver_id"].map(lambda d: nat_map.get(str(d), ""))
    rankings["car_number"] = [num_map.get((int(r.season), str(r.driver_id)), "") for r in rankings.itertuples(index=False)]

    ordered_cols = [
        "season", "driver_id", "driver", "car_number", "driver_fullname", "driver_nationality",
        "team", "car_tier", "starts", "counting_races",
        "delta_q", "delta_r", "f1_par", "f1_war", "f1_war_ci_low", "f1_war_ci_high",
        "dn_par", "dn_war", "dn_war_ci_low", "dn_war_ci_high",
    ]
    rankings = rankings[ordered_cols]

    if salaries_path.exists():
        rankings = attach_salary_efficiency(rankings, pd.read_csv(salaries_path))

    rankings.to_csv(rankings_path, index=False)
    seasons_done = sorted(int(x) for x in rankings["season"].unique())
    meta_payload = {
        "lambda_d": cv_meta["lambda_d"],
        "lambda_dy": cv_meta["lambda_dy"],
        "lambda_c": cv_meta["lambda_c"],
        "val_log_loss": cv_meta["val_log_loss"],
        "car_variance_share_pct": round(pl_fit.car_variance_share_pct, 1),
        "driver_variance_share_pct": round(pl_fit.driver_variance_share_pct, 1),
        "n_races": int(clean_df["race_id"].nunique()),
        "n_bootstrap": int(n_bootstrap),
        "seasons_range": f"{seasons_done[0]}–{seasons_done[-1]}",
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== Pole2Stat v2 업데이트 완료 (2006~2026 현대 F1 아카이브) ===")
    print(f"분산 분해: 레이스카 {pl_fit.car_variance_share_pct:.1f}% | 드라이버 {pl_fit.driver_variance_share_pct:.1f}%")
    print(f"저장 경로: {rankings_path} ({len(rankings)}행, {len(seasons_done)}시즌: {seasons_done[0]}~{seasons_done[-1]})")
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="Pole2Stat v2 2006~2026 시즌별 아카이브 업데이트")
    parser.add_argument("--start-year", type=int, default=2006, help="시작 아카이브 연도 (기본값: 2006)")
    parser.add_argument("--active-season", type=int, default=2026, help="매주 갱신할 현재 시즌 (기본값: 2026)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--cache-dir", type=str, default="./fastf1_cache")
    parser.add_argument("--use-fastf1", action="store_true", help="2018년 이후 시즌에 FastF1 정밀 랩타임 병합 사용")
    parser.add_argument("--force-refresh-archive", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=25)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    update_incremental(
        start_year=args.start_year,
        active_season=args.active_season,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        skip_fastf1=not args.use_fastf1,
        force_refresh_archive=args.force_refresh_archive,
        n_bootstrap=args.n_bootstrap,
        use_demo=args.demo,
    )


if __name__ == "__main__":
    main()
