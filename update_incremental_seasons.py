"""
F1-WAR v2 증분 업데이트 스크립트 (2021~2025 확정 아카이브 + 2026 진행 시즌)

Plackett-Luce 순위 로짓 모형 + 반사실적 몬테카를로 시뮬레이션 + 95% 부트스트랩 신뢰구간
(`f1_plackett_luce_v2.py`)으로 F1-PAR, F1-WAR, dnPAR, dnWAR 및 95% CI를 산출하고
`data/f1_war_rankings.csv`를 갱신합니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from f1_multilevel_prototype import preprocess_f1_dataframe
from f1_plackett_luce_v2 import compute_bootstrap_ci, generate_plackett_luce_demo
from fastf1_enricher import enrich_jolpica_with_fastf1
from jolpica_f1_loader import fetch_recent_seasons_df
from run_f1_sabermetrics_pipeline import _fit_best_multilevel, attach_salary_efficiency


def _fetch_and_enrich(seasons: list[int], skip_fastf1: bool, cache_dir: str) -> pd.DataFrame:
    df = fetch_recent_seasons_df(seasons, include_qualifying=True)
    if df.empty or skip_fastf1:
        return df
    return enrich_jolpica_with_fastf1(
        df,
        cache_dir=cache_dir,
        use_lap_pace_as_target=True,
        include_qualifying=True,
    )


def _assign_tier_label_for_ui(rankings: pd.DataFrame) -> pd.Series:
    """UI 필터링 전용 티어 라벨(모델 계산 배율에는 전혀 사용하지 않음)."""
    tiers = pd.Series("중상위권", index=rankings.index)
    for _, grp in rankings.groupby("season"):
        q_hi = float(grp["car_logit"].quantile(0.70))
        q_lo = float(grp["car_logit"].quantile(0.30))
        tiers.loc[grp.index[grp["car_logit"] >= q_hi]] = "우승권"
        tiers.loc[grp.index[grp["car_logit"] <= q_lo]] = "하위권"
    return tiers


def update_incremental(
    archive_seasons: list[int],
    active_season: int,
    data_dir: str = "data",
    cache_dir: str = "./fastf1_cache",
    skip_fastf1: bool = False,
    force_refresh_archive: bool = False,
    n_bootstrap: int = 35,
    use_demo: bool = False,
) -> pd.DataFrame:
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_tag = "_".join(str(s) for s in sorted(archive_seasons))
    archive_path = out_dir / f"raw_archive_{archive_tag}.csv"
    live_path = out_dir / f"raw_live_{active_season}.csv"
    rankings_path = out_dir / "f1_war_rankings.csv"
    salaries_path = out_dir / "salaries.csv"

    if use_demo:
        raw_df = generate_plackett_luce_demo(n_races_per_season=20, seed=42)
        rng = np.random.default_rng(77)
        raw_df["is_wet"] = 0
        raw_df["quali_pace_pct"] = (11 - raw_df["finish_position"]) * 0.08 + rng.normal(0, 0.08, len(raw_df))
        archive_df = raw_df[raw_df["season"] < active_season].copy()
        live_df = raw_df[raw_df["season"] == active_season].copy()
    else:
        if archive_path.exists() and not force_refresh_archive:
            print(f"[아카이브 로드] 기존 파일 재사용: {archive_path}")
            archive_df = pd.read_csv(archive_path)
        else:
            print(f"[아카이브 수집] 확정 시즌 수집 시작: {archive_seasons}")
            archive_df = _fetch_and_enrich(archive_seasons, skip_fastf1=skip_fastf1, cache_dir=cache_dir)
            if not archive_df.empty:
                archive_df.to_csv(archive_path, index=False)

        print(f"[진행 시즌 수집] {active_season} 시즌 최신 결과 확인 중...")
        live_jolpica = fetch_recent_seasons_df([active_season], include_qualifying=True)

        if live_jolpica.empty:
            live_df = pd.DataFrame()
        elif skip_fastf1:
            live_df = live_jolpica
            live_df.to_csv(live_path, index=False)
        else:
            existing_live = pd.read_csv(live_path) if live_path.exists() else pd.DataFrame()
            done_races = set(existing_live["race_id"].unique()) if not existing_live.empty else set()
            new_rows = live_jolpica[~live_jolpica["race_id"].isin(done_races)].copy()
            if new_rows.empty and not existing_live.empty:
                live_df = existing_live
            else:
                enriched_new = enrich_jolpica_with_fastf1(
                    new_rows, cache_dir=cache_dir, use_lap_pace_as_target=True, include_qualifying=True
                )
                live_df = pd.concat([existing_live, enriched_new], ignore_index=True) if not existing_live.empty else enriched_new
                live_df.to_csv(live_path, index=False)

    combined = pd.concat([df for df in (archive_df, live_df) if not df.empty], ignore_index=True)
    if combined.empty:
        raise RuntimeError("수집된 레이스 데이터가 없어 랭킹을 계산할 수 없습니다.")

    # 1) v2 Plackett-Luce 순위 로짓 + 반사실적 시뮬레이션 + 95% 부트스트랩 신뢰구간
    clean_df = combined[combined.get("mechanical_dnf", 0) == 0].dropna(subset=["finish_position"]).copy()
    rankings, pl_fit = compute_bootstrap_ci(clean_df, raw_df=combined, n_bootstrap=n_bootstrap)

    # 2) 예선 ΔQ는 Jolpica/FastF1 quali_pace_pct가 있으면 별도 다층 모형으로 병합
    if "quali_pace_pct" in combined.columns and combined["quali_pace_pct"].notna().sum() >= 20:
        q_work = combined.dropna(subset=["quali_pace_pct"]).copy()
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
        rankings["delta_q"] = [q_map.get((int(r.season), str(r.driver)), np.nan) for r in rankings.itertuples(index=False)]
    else:
        rankings["delta_q"] = np.nan

    # 대시보드 호환 컬럼 정리 (delta_r = Plackett-Luce 대체 대비 로짓 우위)
    rankings["delta_r"] = rankings["skill_logit"]
    rankings["car_tier"] = _assign_tier_label_for_ui(rankings)

    ordered_cols = [
        "season", "driver", "team", "car_tier", "starts", "counting_races",
        "delta_q", "delta_r", "f1_par", "f1_war", "f1_war_ci_low", "f1_war_ci_high",
        "dn_par", "dn_war", "dn_war_ci_low", "dn_war_ci_high",
    ]
    rankings = rankings[ordered_cols]

    if salaries_path.exists():
        rankings = attach_salary_efficiency(rankings, pd.read_csv(salaries_path))

    rankings.to_csv(rankings_path, index=False)
    print("=== F1-WAR v2 증분 업데이트 완료 (Plackett-Luce + 95% CI) ===")
    print(f"분산 분해: 머신 {pl_fit.car_variance_share_pct:.1f}% | 드라이버 {pl_fit.driver_variance_share_pct:.1f}%")
    print(f"저장 경로: {rankings_path} ({len(rankings)}행, 시즌 {sorted(rankings['season'].unique())})")
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="F1-WAR v2 증분 업데이트 스크립트")
    parser.add_argument("--archive-seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--active-season", type=int, default=2026)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--cache-dir", type=str, default="./fastf1_cache")
    parser.add_argument("--skip-fastf1", action="store_true")
    parser.add_argument("--force-refresh-archive", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=35)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    update_incremental(
        archive_seasons=args.archive_seasons,
        active_season=args.active_season,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        skip_fastf1=args.skip_fastf1,
        force_refresh_archive=args.force_refresh_archive,
        n_bootstrap=args.n_bootstrap,
        use_demo=args.demo,
    )


if __name__ == "__main__":
    main()
