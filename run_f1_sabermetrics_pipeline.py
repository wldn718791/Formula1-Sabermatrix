"""
F1 세이버메트릭스 통합 파이프라인 (예선 ΔQ + 레이스 ΔR -> 다층 모형 -> F1-PAR / dnPAR)

같은 폴더에 있는 세 모듈을 연결해 시즌별 드라이버 F1-PAR, F1-WAR, 차량 중립 가치(dnPAR)
및 선택적 연봉 가성비 보조 컬럼을 일괄 계산합니다.
  - jolpica_f1_loader.py       : Jolpica F1 API 레이스 결과·리타이어 상태 수집
  - fastf1_enricher.py         : FastF1 예선(ΔQ)·레이스 클린 에어(ΔR)·우천 여부 병합
  - f1_multilevel_prototype.py : 교차 분류 다층 모형(드라이버-머신 분리)

실행 예시:
    python run_f1_sabermetrics_pipeline.py --demo
    python run_f1_sabermetrics_pipeline.py --seasons 2024 2025 2026 --salaries-csv salaries.csv
    python run_f1_sabermetrics_pipeline.py --seasons 2024 2025 2026 --skip-fastf1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from f1_multilevel_prototype import (
    fit_numpy_blup_cross_classified,
    fit_sklearn_cross_classified,
    fit_statsmodels_cross_classified,
    generate_synthetic_f1_data,
    preprocess_f1_dataframe,
)
from fastf1_enricher import enrich_jolpica_with_fastf1
from jolpica_f1_loader import fetch_recent_seasons_df

TIER_PARAMS = {
    "우승권": {"marginal_pts_per_pos": 3.4, "base_pts_per_race": 15.5},
    "중상위권": {"marginal_pts_per_pos": 2.1, "base_pts_per_race": 7.0},
    "하위권": {"marginal_pts_per_pos": 0.45, "base_pts_per_race": 0.4},
}
STANDARD_NEUTRAL_M = TIER_PARAMS["중상위권"]["marginal_pts_per_pos"]
STANDARD_NEUTRAL_BASE = TIER_PARAMS["중상위권"]["base_pts_per_race"]
POINTS_PER_WIN = 25.0


def _fit_best_multilevel(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float, str]:
    """driver + driver_year(시즌별 폼) + team_year 3중 랜덤 효과를 RidgeCV -> NumPy BLUP 순으로 적합합니다."""
    sk_res = fit_sklearn_cross_classified(df)
    if sk_res is not None:
        _, car_df, summary = sk_res
        return (
            summary["driver_season_df"],
            car_df,
            float(summary["car_variance_share_pct"]),
            float(summary["driver_variance_share_pct"]),
            "scikit-learn RidgeCV (시즌별 driver_year 포함)",
        )

    _, car_df, summary = fit_numpy_blup_cross_classified(df)
    return (
        summary["driver_season_df"],
        car_df,
        float(summary["car_variance_share_pct"]),
        float(summary["driver_variance_share_pct"]),
        "NumPy BLUP (시즌별 driver_year 포함)",
    )


def assign_car_tiers(car_df: pd.DataFrame) -> dict[str, str]:
    """추정된 팀-연도별 머신 성능을 시즌 내 분위 기준으로 우승권/중상위권/하위권으로 분류합니다."""
    tier_map: dict[str, str] = {}
    work = car_df.copy()
    work["season"] = work["team_year"].str.rsplit("_", n=1).str[-1]

    for _, grp in work.groupby("season"):
        q_high = float(grp["est_car_perf"].quantile(0.70))
        q_low = float(grp["est_car_perf"].quantile(0.30))
        for row in grp.itertuples(index=False):
            if row.est_car_perf >= q_high:
                tier_map[row.team_year] = "우승권"
            elif row.est_car_perf <= q_low:
                tier_map[row.team_year] = "하위권"
            else:
                tier_map[row.team_year] = "중상위권"
    return tier_map


def compute_seasonal_par_rankings(
    clean_df: pd.DataFrame,
    race_driver_df: pd.DataFrame,
    car_df: pd.DataFrame,
    quali_driver_df: pd.DataFrame | None = None,
    raw_df: pd.DataFrame | None = None,
    weight_quali: float = 0.35,
    weight_race: float = 0.65,
    pos_gain_per_unit_quali: float = 5.5,
    pos_gain_per_unit_race: float = 8.5,
    repl_error_rate_per_24: float = 2.0,
) -> pd.DataFrame:
    """예선(ΔQ)과 레이스(ΔR) 다층 모형 추정치를 가중 합산해 F1-PAR / F1-WAR / dnPAR을 산출합니다."""
    # 시즌별 실력(u_driver + u_driver_year)과 시즌별 대체 수준(해당 시즌 하위 20%)을 분리 매핑
    if "est_driver_season_skill" in race_driver_df.columns:
        race_skill_map = {
            (int(r.season), str(r.driver)): float(r.est_driver_season_skill)
            for r in race_driver_df.itertuples(index=False)
        }
        repl_race_by_season = {
            int(s): float(np.quantile(g["est_driver_season_skill"], 0.20))
            for s, g in race_driver_df.groupby("season")
        }
    else:
        base_map = dict(zip(race_driver_df["driver"], race_driver_df["est_driver_skill"]))
        race_skill_map = {(int(s), str(d)): float(base_map[d]) for s in clean_df["season"].unique() for d in base_map}
        q20 = float(np.quantile(race_driver_df["est_driver_skill"], 0.20))
        repl_race_by_season = {int(s): q20 for s in clean_df["season"].unique()}

    has_quali = quali_driver_df is not None and not quali_driver_df.empty
    if has_quali:
        if "est_driver_season_skill" in quali_driver_df.columns:
            quali_skill_map = {
                (int(r.season), str(r.driver)): float(r.est_driver_season_skill)
                for r in quali_driver_df.itertuples(index=False)
            }
            repl_quali_by_season = {
                int(s): float(np.quantile(g["est_driver_season_skill"], 0.20))
                for s, g in quali_driver_df.groupby("season")
            }
        else:
            q_base = dict(zip(quali_driver_df["driver"], quali_driver_df["est_driver_skill"]))
            quali_skill_map = {(int(s), str(d)): float(q_base[d]) for s in clean_df["season"].unique() for d in q_base}
            qq20 = float(np.quantile(quali_driver_df["est_driver_skill"], 0.20))
            repl_quali_by_season = {int(s): qq20 for s in clean_df["season"].unique()}
    else:
        quali_skill_map = {}
        repl_quali_by_season = {}

    tier_map = assign_car_tiers(car_df)
    work = clean_df.copy()
    if "status_category" not in work.columns:
        work["status_category"] = "finished"

    # raw_df가 주어지면 리타이어 포함 실제 총 출전 경기 수를 별도 집계
    total_starts_map: dict[tuple[int, str], int] = {}
    if raw_df is not None and not raw_df.empty:
        total_starts_map = {
            (int(k[0]), str(k[1])): int(v)
            for k, v in raw_df.groupby(["season", "driver"]).size().items()
        }

    rows: list[dict[str, object]] = []
    for (season, driver), grp in work.groupby(["season", "driver"]):
        counting_races = len(grp)
        total_starts = total_starts_map.get((int(season), str(driver)), counting_races)
        main_team_year = str(grp["team_year"].mode().iloc[0])
        main_team = str(grp["team"].mode().iloc[0]) if "team" in grp.columns else main_team_year
        car_tier = tier_map.get(main_team_year, "중상위권")
        params = TIER_PARAMS[car_tier]

        repl_r = repl_race_by_season.get(int(season), 0.0)
        race_delta = race_skill_map.get((int(season), str(driver)), repl_r) - repl_r
        if has_quali:
            repl_q = repl_quali_by_season.get(int(season), 0.0)
            quali_delta = quali_skill_map.get((int(season), str(driver)), repl_q) - repl_q
            pos_gain_per_race = (
                weight_quali * quali_delta * pos_gain_per_unit_quali
                + weight_race * race_delta * pos_gain_per_unit_race
            )
        else:
            quali_delta = np.nan
            pos_gain_per_race = race_delta * pos_gain_per_unit_race

        incidents = int((grp["status_category"] == "driver_incident").sum())
        expected_repl_incidents = repl_error_rate_per_24 * (counting_races / 24.0)
        err_diff = expected_repl_incidents - incidents

        # (1) 실제 머신 기준 F1-PAR & F1-WAR (기계적 DNF 제외 유효 경기 기준)
        pace_par = pos_gain_per_race * params["marginal_pts_per_pos"] * counting_races
        exp_race_pts = np.clip(
            params["base_pts_per_race"] + pos_gain_per_race * params["marginal_pts_per_pos"],
            0.5,
            25.0,
        )
        err_par = err_diff * exp_race_pts * 0.85
        f1_par = float(pace_par + err_par)
        f1_war = f1_par / POINTS_PER_WIN

        # (2) 차량 중립 기준 dnPAR (기계적 DNF 제외 유효 경기 기준)
        dn_pace_par = pos_gain_per_race * STANDARD_NEUTRAL_M * counting_races
        dn_exp_race_pts = np.clip(
            STANDARD_NEUTRAL_BASE + pos_gain_per_race * STANDARD_NEUTRAL_M,
            0.5,
            25.0,
        )
        dn_err_par = err_diff * dn_exp_race_pts * 0.85
        dn_par = float(dn_pace_par + dn_err_par)

        row_dict: dict[str, object] = {
            "season": int(season),
            "driver": driver,
            "team": main_team,
            "car_tier": car_tier,
            "starts": total_starts,
            "counting_races": counting_races,
        }
        if has_quali:
            row_dict["delta_q"] = round(float(quali_delta), 3)
        row_dict["delta_r"] = round(float(race_delta), 3)
        row_dict["f1_par"] = round(f1_par, 1)
        row_dict["f1_war"] = round(f1_war, 2)
        row_dict["dn_par"] = round(dn_par, 1)
        rows.append(row_dict)

    return (
        pd.DataFrame(rows)
        .sort_values(["season", "dn_par"], ascending=[False, False])
        .reset_index(drop=True)
    )


def attach_salary_efficiency(
    rankings: pd.DataFrame,
    salaries_df: pd.DataFrame,
) -> pd.DataFrame:
    """선택적 연봉 데이터를 조인해 $/F1-WAR와 $/dnPAR 참고용 보조 컬럼을 추가합니다."""
    required = {"season", "driver", "salary_musd"}
    missing = required - set(salaries_df.columns)
    if missing:
        raise ValueError(f"연봉 CSV에 필수 컬럼이 누락되었습니다: {sorted(missing)}")

    sal = salaries_df[["season", "driver", "salary_musd"]].copy()
    sal["season"] = sal["season"].astype(int)
    sal["driver"] = sal["driver"].astype(str).str.upper()
    sal["salary_musd"] = pd.to_numeric(sal["salary_musd"], errors="coerce")

    merged = rankings.copy()
    merged["driver"] = merged["driver"].astype(str).str.upper()
    merged = merged.merge(sal, on=["season", "driver"], how="left")

    merged["musd_per_f1_war"] = np.where(
        (merged["f1_war"] > 0) & merged["salary_musd"].notna(),
        (merged["salary_musd"] / merged["f1_war"]).round(2),
        np.nan,
    )
    merged["kusd_per_dn_par"] = np.where(
        (merged["dn_par"] > 0) & merged["salary_musd"].notna(),
        ((merged["salary_musd"] * 1000.0) / merged["dn_par"]).round(1),
        np.nan,
    )
    return merged


def run_pipeline(
    seasons: list[int],
    use_demo: bool = False,
    skip_fastf1: bool = False,
    cache_dir: str = "./fastf1_cache",
    salaries_csv: str | None = None,
    output_csv: str | None = None,
    weight_quali: float = 0.35,
    weight_race: float = 0.65,
) -> pd.DataFrame:
    """데이터 수집 -> 예선·레이스 다층 모형 적합 -> F1-PAR/dnPAR 산출을 일괄 실행합니다."""
    if use_demo:
        rng = np.random.default_rng(101)
        raw_df, _, _ = generate_synthetic_f1_data(n_races_per_season=20, seed=42)
        raw_df["status_category"] = np.where(raw_df["mechanical_dnf"] == 1, "mechanical_dnf", "finished")
        # 예선 스페셜리스트 특성(LEC, RUS 소폭 가산)을 포함한 합성 예선 격차 추가
        q_bonus = raw_df["driver"].map({"LEC": 0.14, "RUS": 0.08, "PER": -0.10}).fillna(0.0)
        raw_df["quali_pace_pct"] = raw_df["pace_score"] + q_bonus + rng.normal(0.0, 0.12, len(raw_df))
        source_label = "합성 데모 데이터 (예선 ΔQ + 레이스 ΔR)"
    else:
        raw_df = fetch_recent_seasons_df(seasons)
        if not skip_fastf1:
            raw_df = enrich_jolpica_with_fastf1(
                raw_df,
                cache_dir=cache_dir,
                use_lap_pace_as_target=True,
                include_qualifying=True,
            )
            source_label = f"Jolpica + FastF1 예선·레이스 ({seasons})"
        else:
            source_label = f"Jolpica 전용 ({seasons})"

    # 1) 레이스 페이스(ΔR) 다층 모형 적합
    race_clean = preprocess_f1_dataframe(raw_df).dropna(subset=["target_centered"]).copy()
    race_driver_df, car_df, car_share, drv_share, model_name = _fit_best_multilevel(race_clean)

    # 2) 예선 페이스(ΔQ) 데이터가 있으면 별도 다층 모형 적합 (기계적 DNF 무관하게 전 경기 활용)
    quali_driver_df = None
    if "quali_pace_pct" in raw_df.columns and raw_df["quali_pace_pct"].notna().sum() >= 20:
        quali_work = raw_df.dropna(subset=["quali_pace_pct"]).copy()
        quali_work["pace_score"] = quali_work["quali_pace_pct"]
        quali_work["mechanical_dnf"] = 0
        quali_clean = preprocess_f1_dataframe(quali_work)
        quali_driver_df, _, _, _, _ = _fit_best_multilevel(quali_clean)

    rankings = compute_seasonal_par_rankings(
        clean_df=race_clean,
        race_driver_df=race_driver_df,
        car_df=car_df,
        quali_driver_df=quali_driver_df,
        raw_df=raw_df,
        weight_quali=weight_quali,
        weight_race=weight_race,
    )

    if salaries_csv:
        salaries_df = pd.read_csv(salaries_csv)
        rankings = attach_salary_efficiency(rankings, salaries_df)

    print("=== F1 세이버메트릭스 파이프라인 완료 ===")
    print(f"데이터 소스 : {source_label} (유효 표본 {len(race_clean)}건)")
    print(f"적용 모형   : {model_name}")
    print(f"분산 분해   : 머신 {car_share:.1f}% | 드라이버 {drv_share:.1f}%")
    if quali_driver_df is not None:
        print(f"가중치 조합 : 예선(ΔQ) {weight_quali:.0%} + 레이스(ΔR) {weight_race:.0%}")
    if salaries_csv:
        print(f"연봉 보조   : 적용됨 ({salaries_csv})")
    print()

    latest_season = int(rankings["season"].max())
    print(f"[{latest_season} 시즌 드라이버 F1-PAR / F1-WAR / dnPAR 랭킹]")
    print(
        rankings[rankings["season"] == latest_season].to_string(
            index=False,
            na_rep="-",
            float_format=lambda x: f"{x:+.2f}",
        )
    )

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        rankings.to_csv(output_csv, index=False)
        print(f"\n전체 시즌 랭킹 저장 완료: {output_csv}")

    return rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="F1-PAR / dnPAR 통합 파이프라인 (예선 ΔQ + 레이스 ΔR)")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026], help="분석할 시즌 목록")
    parser.add_argument("--demo", action="store_true", help="네트워크 없이 합성 데이터로 전체 파이프라인 실행")
    parser.add_argument("--skip-fastf1", action="store_true", help="FastF1 병합 없이 Jolpica 결과만 사용")
    parser.add_argument("--cache-dir", type=str, default="./fastf1_cache", help="FastF1 로컬 캐시 디렉터리")
    parser.add_argument("--salaries-csv", type=str, default=None, help="선택적 추정 연봉 CSV 경로")
    parser.add_argument("--w-quali", type=float, default=0.35, help="예선 페이스(ΔQ) 가중치 (기본값: 0.35)")
    parser.add_argument("--w-race", type=float, default=0.65, help="레이스 페이스(ΔR) 가중치 (기본값: 0.65)")
    parser.add_argument("--output", type=str, default="f1_war_rankings.csv", help="결과 CSV 저장 경로")
    args = parser.parse_args()

    run_pipeline(
        seasons=args.seasons,
        use_demo=args.demo,
        skip_fastf1=args.skip_fastf1,
        cache_dir=args.cache_dir,
        salaries_csv=args.salaries_csv,
        output_csv=None if args.demo else args.output,
        weight_quali=args.w_quali,
        weight_race=args.w_race,
    )


if __name__ == "__main__":
    main()
