"""
2024~2025 확정 시즌 아카이브 + 2026 진행 시즌 증분 업데이트 스크립트

GitHub Actions에서 매주 실행하도록 설계되었습니다.
- 확정 시즌(기본 2024, 2025): 최초 1회만 수집해 `data/raw_archive_2024_2025.csv`에 고정 저장
- 현재 시즌(기본 2026): 매주 Jolpica + FastF1에서 최신 라운드만 갱신해 `data/raw_live_2026.csv`에 저장
- 두 데이터를 합쳐 교차 분류 다층 모형(예선 ΔQ + 레이스 ΔR)을 적합하고
  웹 대시보드(index.html)가 자동 로드하는 `data/f1_war_rankings.csv`를 갱신합니다.

실행 예시:
    python update_incremental_seasons.py --demo
    python update_incremental_seasons.py --archive-seasons 2024 2025 --active-season 2026
    python update_incremental_seasons.py --archive-seasons 2024 2025 --active-season 2026 --skip-fastf1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from f1_multilevel_prototype import generate_synthetic_f1_data, preprocess_f1_dataframe
from fastf1_enricher import enrich_jolpica_with_fastf1
from jolpica_f1_loader import fetch_recent_seasons_df
from run_f1_sabermetrics_pipeline import (
    _fit_best_multilevel,
    attach_salary_efficiency,
    compute_seasonal_par_rankings,
)


def _fetch_and_enrich(
    seasons: list[int],
    skip_fastf1: bool,
    cache_dir: str,
) -> pd.DataFrame:
    df = fetch_recent_seasons_df(seasons)
    if df.empty or skip_fastf1:
        return df
    return enrich_jolpica_with_fastf1(
        df,
        cache_dir=cache_dir,
        use_lap_pace_as_target=True,
        include_qualifying=True,
    )


def update_incremental(
    archive_seasons: list[int],
    active_season: int,
    data_dir: str = "data",
    cache_dir: str = "./fastf1_cache",
    skip_fastf1: bool = False,
    force_refresh_archive: bool = False,
    use_demo: bool = False,
) -> pd.DataFrame:
    """확정 아카이브와 현재 진행 시즌 데이터를 결합해 최종 랭킹 CSV를 갱신합니다."""
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive_tag = "_".join(str(s) for s in sorted(archive_seasons))
    archive_path = out_dir / f"raw_archive_{archive_tag}.csv"
    live_path = out_dir / f"raw_live_{active_season}.csv"
    rankings_path = out_dir / "f1_war_rankings.csv"
    salaries_path = out_dir / "salaries.csv"

    if use_demo:
        rng = np.random.default_rng(101)
        raw_df, _, _ = generate_synthetic_f1_data(n_races_per_season=20, seed=42)
        raw_df["status_category"] = np.where(raw_df["mechanical_dnf"] == 1, "mechanical_dnf", "finished")
        q_bonus = raw_df["driver"].map({"LEC": 0.14, "RUS": 0.08, "PER": -0.10}).fillna(0.0)
        raw_df["quali_pace_pct"] = raw_df["pace_score"] + q_bonus + rng.normal(0.0, 0.12, len(raw_df))
        # 데모 시즌을 2024, 2025, 2026으로 매핑해 아카이브+진행 시즌 흐름을 그대로 검증
        raw_df["season"] = raw_df["season"] + 2
        raw_df["team_year"] = raw_df["team"].astype(str) + "_" + raw_df["season"].astype(str)
        archive_df = raw_df[raw_df["season"].isin(archive_seasons)].copy()
        live_df = raw_df[raw_df["season"] == active_season].copy()
        archive_df.to_csv(archive_path, index=False)
        live_df.to_csv(live_path, index=False)
    else:
        # 1) 확정 시즌 아카이브: 파일이 없을 때만 1회 수집
        if archive_path.exists() and not force_refresh_archive:
            print(f"[아카이브 로드] 기존 파일 재사용: {archive_path}")
            archive_df = pd.read_csv(archive_path)
        else:
            print(f"[아카이브 수집] 확정 시즌 수집 시작: {archive_seasons}")
            archive_df = _fetch_and_enrich(archive_seasons, skip_fastf1=skip_fastf1, cache_dir=cache_dir)
            if not archive_df.empty:
                archive_df.to_csv(archive_path, index=False)
                print(f"[아카이브 저장] {len(archive_df)}행 -> {archive_path}")

        # 2) 현재 진행 시즌 증분 수집 (신규 라운드만 FastF1 추가 로드)
        print(f"[진행 시즌 수집] {active_season} 시즌 최신 결과 확인 중...")
        live_jolpica = fetch_recent_seasons_df([active_season])

        if live_jolpica.empty:
            print(f"[안내] {active_season} 시즌의 완료된 경기 데이터가 아직 없습니다.")
            live_df = pd.DataFrame()
        elif skip_fastf1:
            live_df = live_jolpica
            live_df.to_csv(live_path, index=False)
        else:
            existing_live = pd.read_csv(live_path) if live_path.exists() else pd.DataFrame()
            done_races = set(existing_live["race_id"].unique()) if not existing_live.empty else set()
            new_rows = live_jolpica[~live_jolpica["race_id"].isin(done_races)].copy()

            if new_rows.empty and not existing_live.empty:
                print(f"[진행 시즌] 신규 완료 라운드 없음 -> 기존 {live_path} 유지")
                live_df = existing_live
            else:
                print(f"[진행 시즌] 신규 {new_rows['race_id'].nunique()}개 라운드 FastF1 보정 수행")
                enriched_new = enrich_jolpica_with_fastf1(
                    new_rows,
                    cache_dir=cache_dir,
                    use_lap_pace_as_target=True,
                    include_qualifying=True,
                )
                live_df = (
                    pd.concat([existing_live, enriched_new], ignore_index=True)
                    if not existing_live.empty
                    else enriched_new
                )
                live_df.to_csv(live_path, index=False)

    combined = pd.concat([df for df in (archive_df, live_df) if not df.empty], ignore_index=True)
    if combined.empty:
        raise RuntimeError("수집된 레이스 데이터가 없어 랭킹을 계산할 수 없습니다.")

    # 3) 레이스(ΔR) + 예선(ΔQ) 교차 분류 다층 모형 적합
    race_clean = preprocess_f1_dataframe(combined).dropna(subset=["target_centered"]).copy()
    race_driver_df, car_df, car_share, drv_share, model_name = _fit_best_multilevel(race_clean)

    quali_driver_df = None
    if "quali_pace_pct" in combined.columns and combined["quali_pace_pct"].notna().sum() >= 20:
        quali_work = combined.dropna(subset=["quali_pace_pct"]).copy()
        quali_work["pace_score"] = quali_work["quali_pace_pct"]
        quali_work["mechanical_dnf"] = 0
        quali_clean = preprocess_f1_dataframe(quali_work)
        quali_driver_df, _, _, _, _ = _fit_best_multilevel(quali_clean)

    rankings = compute_seasonal_par_rankings(
        clean_df=race_clean,
        race_driver_df=race_driver_df,
        car_df=car_df,
        quali_driver_df=quali_driver_df,
    )

    # data/salaries.csv가 존재하면 자동으로 연봉 가성비 보조 컬럼 결합
    if salaries_path.exists():
        rankings = attach_salary_efficiency(rankings, pd.read_csv(salaries_path))

    rankings.to_csv(rankings_path, index=False)
    print(f"=== 증분 업데이트 완료 ({model_name}) ===")
    print(f"분산 분해: 머신 {car_share:.1f}% | 드라이버 {drv_share:.1f}%")
    print(f"저장 경로: {rankings_path} ({len(rankings)}행, 시즌 {sorted(rankings['season'].unique())})")
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 아카이브 + 현재 시즌 증분 업데이트 스크립트")
    parser.add_argument("--archive-seasons", type=int, nargs="+", default=[2024, 2025], help="고정 아카이브 시즌")
    parser.add_argument("--active-season", type=int, default=2026, help="매주 증분 갱신할 현재 시즌")
    parser.add_argument("--data-dir", type=str, default="data", help="데이터 저장 폴더")
    parser.add_argument("--cache-dir", type=str, default="./fastf1_cache", help="FastF1 로컬 캐시 폴더")
    parser.add_argument("--skip-fastf1", action="store_true", help="FastF1 없이 Jolpica 결과만 사용")
    parser.add_argument("--force-refresh-archive", action="store_true", help="아카이브 CSV 강제 재수집")
    parser.add_argument("--demo", action="store_true", help="오프라인 합성 데이터로 증분 업데이트 검증")
    args = parser.parse_args()

    update_incremental(
        archive_seasons=args.archive_seasons,
        active_season=args.active_season,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        skip_fastf1=args.skip_fastf1,
        force_refresh_archive=args.force_refresh_archive,
        use_demo=args.demo,
    )


if __name__ == "__main__":
    main()
