"""
FastF1 기반 우천 여부(is_wet), 예선(Q1/Q2/Q3) 페이스 격차(ΔQ) 및
연료·타이어 보정 클린 에어 레이스 페이스(ΔR) 추출·병합기

FastF1(https://github.com/theOehrly/Fast-F1)에서 예선(Q) 및 레이스(R) 세션을
불러와 다음 전처리를 수행한 뒤, `jolpica_f1_loader.py`가 만든 DataFrame과
(`race_id`, `driver`) 기준으로 병합합니다.

1. 예선(Q1/Q2/Q3): 세션 간 트랙 에볼루션(Q1->Q2->Q3)을 보정한 베스트 랩타임 격차(quali_pace_pct)
2. 레이스(R): 그린 플래그·피트랩 제외·앞차 2초 이상 클린 에어 필터링 + 연료·타이어 OLS 보정
3. 날씨: 강우 센서(weather_data['Rainfall']) 및 웨트 타이어 기록으로 is_wet 판별

로컬 실행 예시:
    python fastf1_enricher.py --jolpica-csv f1_recent_3seasons.csv --output f1_enriched.csv
    python fastf1_enricher.py --self-test   # FastF1 없이 오프라인 로직 단위 검증
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. 우천 여부(is_wet) 판별
# ---------------------------------------------------------------------------
def detect_wet_race(
    laps_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
) -> int:
    """강우 센서(Rainfall) 또는 인터미디어트/웨트 타이어 사용 여부로 is_wet(0/1)을 판별합니다."""
    if weather_df is not None and "Rainfall" in weather_df.columns:
        if bool(weather_df["Rainfall"].fillna(False).astype(bool).any()):
            return 1

    if "Compound" in laps_df.columns:
        wet_compounds = {"INTERMEDIATE", "WET"}
        used = set(laps_df["Compound"].dropna().astype(str).str.upper().unique())
        if used & wet_compounds:
            return 1

    return 0


# ---------------------------------------------------------------------------
# 2. 예선(Q1/Q2/Q3) 트랙 에볼루션 보정 베스트 랩타임 격차(ΔQ) 산출
# ---------------------------------------------------------------------------
def _to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def compute_quali_pace_gap(quali_results_df: pd.DataFrame) -> pd.DataFrame:
    """Q1/Q2/Q3 세션 결과에서 트랙 에볼루션을 보정한 예선 페이스 우위(%)를 산출합니다.

    Q1 탈락 드라이버와 Q3 진출 드라이버 간 노면 진화(Track Evolution) 편향을 줄이기 위해
    공통 참가 드라이버의 중앙값 시간 변화(Q2-Q1, Q3-Q2)를 추정해 모든 기록을 Q3 기준
    등가 랩타임(`adj_quali_sec`)으로 맞춘 뒤, 그리드 중앙값 대비 우위(%)를 계산합니다.
    """
    if quali_results_df.empty:
        return pd.DataFrame(columns=["driver", "best_quali_sec", "adj_quali_sec", "quali_pace_pct"])

    df = quali_results_df.copy()
    driver_col = "Abbreviation" if "Abbreviation" in df.columns else "driver"
    df["driver"] = df[driver_col].astype(str).str.upper()

    for col in ("Q1", "Q2", "Q3"):
        df[f"{col}_sec"] = _to_seconds(df[col]) if col in df.columns else np.nan

    # 세션 간 트랙 에볼루션 중앙값 추정 (음수 = 노면이 빨라짐)
    both_12 = df[["Q1_sec", "Q2_sec"]].dropna()
    evo_12 = float((both_12["Q2_sec"] - both_12["Q1_sec"]).median()) if not both_12.empty else 0.0

    both_23 = df[["Q2_sec", "Q3_sec"]].dropna()
    evo_23 = float((both_23["Q3_sec"] - both_23["Q2_sec"]).median()) if not both_23.empty else 0.0

    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        q1, q2, q3 = row.Q1_sec, row.Q2_sec, row.Q3_sec
        valid_times = [t for t in (q1, q2, q3) if pd.notna(t)]
        if not valid_times:
            continue

        best_raw = float(min(valid_times))
        # 각 드라이버가 도달한 최심 세션을 Q3 기준 노면으로 보정
        if pd.notna(q3):
            adj_sec = float(min(q3, q2 + evo_23 if pd.notna(q2) else q3))
        elif pd.notna(q2):
            adj_sec = float(q2 + evo_23)
        else:
            adj_sec = float(q1 + evo_12 + evo_23)

        rows.append(
            {
                "driver": row.driver,
                "best_quali_sec": round(best_raw, 4),
                "adj_quali_sec": round(adj_sec, 4),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["driver", "best_quali_sec", "adj_quali_sec", "quali_pace_pct"])

    field_median = float(out["adj_quali_sec"].median())
    # 양수일수록 예선 페이스가 빠름
    out["quali_pace_pct"] = (
        (field_median - out["adj_quali_sec"]) / field_median * 100.0
    ).round(4)
    return out.sort_values("quali_pace_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. 레이스 클린 에어(Clean Air) 랩 필터링
# ---------------------------------------------------------------------------
def filter_clean_air_laps(
    laps_df: pd.DataFrame,
    min_gap_ahead_sec: float = 2.0,
    quicklap_threshold: float = 1.07,
) -> pd.DataFrame:
    """SC/VSC·피트랩·1랩·더티 에어(앞차 2초 미만)·이상치 랩을 제거합니다."""
    df = laps_df.copy()
    df["lap_sec"] = _to_seconds(df["LapTime"])
    df["session_sec"] = _to_seconds(df["Time"])

    mask = df["lap_sec"].notna() & df["session_sec"].notna() & (df["LapNumber"] > 1)
    if "PitInTime" in df.columns:
        mask &= df["PitInTime"].isna()
    if "PitOutTime" in df.columns:
        mask &= df["PitOutTime"].isna()
    if "TrackStatus" in df.columns:
        mask &= df["TrackStatus"].astype(str) == "1"

    df = df.loc[mask].copy()
    if df.empty:
        return df

    df = df.sort_values("session_sec").reset_index(drop=True)
    df["gap_ahead_sec"] = df["session_sec"].diff().fillna(99.0)
    df = df[df["gap_ahead_sec"] >= min_gap_ahead_sec].copy()

    if not df.empty:
        fastest = df["lap_sec"].min()
        df = df[df["lap_sec"] <= fastest * quicklap_threshold].copy()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. 연료 하중 + 타이어 컴파운드/열화 OLS 보정 및 레이스 페이스(ΔR) 집계
# ---------------------------------------------------------------------------
def compute_fuel_tyre_corrected_pace(
    clean_laps: pd.DataFrame,
    min_laps_per_driver: int = 5,
) -> pd.DataFrame:
    """레이스 내 클린 에어 랩에 OLS 회귀를 적합해 연료·타이어 효과를 제거합니다."""
    if clean_laps.empty:
        return pd.DataFrame(
            columns=["driver", "clean_air_laps", "median_corrected_sec", "clean_air_pace_pct"]
        )

    work = clean_laps.copy()
    work["Compound"] = work["Compound"].fillna("UNKNOWN").astype(str).str.upper()
    work["TyreLife"] = pd.to_numeric(work["TyreLife"], errors="coerce").fillna(1.0)
    work["LapNumber"] = pd.to_numeric(work["LapNumber"], errors="coerce")

    ref_lap = float(work["LapNumber"].median())
    work["lap_centered"] = work["LapNumber"] - ref_lap
    work["tyre_age"] = work["TyreLife"] - 1.0

    compounds = sorted(work["Compound"].unique())
    drivers = sorted(work["Driver"].unique())

    cols_env: list[np.ndarray] = [work["lap_centered"].to_numpy(dtype=float)]
    for comp in compounds[1:]:
        cols_env.append((work["Compound"] == comp).to_numpy(dtype=float))
    for comp in compounds:
        cols_env.append(
            ((work["Compound"] == comp).to_numpy(dtype=float))
            * work["tyre_age"].to_numpy(dtype=float)
        )

    X_env = np.column_stack(cols_env)
    X_drv = (
        np.column_stack([(work["Driver"] == drv).to_numpy(dtype=float) for drv in drivers[1:]])
        if len(drivers) > 1
        else np.empty((len(work), 0))
    )
    X = np.column_stack([np.ones(len(work)), X_env, X_drv])
    y = work["lap_sec"].to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    env_effect = X_env @ beta[1 : 1 + X_env.shape[1]]
    work["corrected_lap_sec"] = work["lap_sec"] - env_effect

    agg = (
        work.groupby("Driver", as_index=False)
        .agg(
            clean_air_laps=("corrected_lap_sec", "count"),
            median_corrected_sec=("corrected_lap_sec", "median"),
        )
        .rename(columns={"Driver": "driver"})
    )
    agg = agg[agg["clean_air_laps"] >= min_laps_per_driver].copy()
    if agg.empty:
        return pd.DataFrame(
            columns=["driver", "clean_air_laps", "median_corrected_sec", "clean_air_pace_pct"]
        )

    race_median = float(agg["median_corrected_sec"].median())
    agg["clean_air_pace_pct"] = (
        (race_median - agg["median_corrected_sec"]) / race_median * 100.0
    ).round(4)
    return agg.sort_values("clean_air_pace_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. FastF1 예선(Q) + 레이스(R) 로드 및 Jolpica DataFrame 병합
# ---------------------------------------------------------------------------
def extract_fastf1_quali_features(
    season: int,
    round_num: int,
    cache_dir: str = "./fastf1_cache",
) -> pd.DataFrame:
    """단일 그랑프리의 FastF1 예선 세션(Q)에서 트랙 에볼루션 보정 예선 격차(ΔQ)를 추출합니다."""
    import fastf1

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    q_session = fastf1.get_session(season, round_num, "Q")
    q_session.load(laps=False, telemetry=False, weather=False, messages=False)

    quali_df = compute_quali_pace_gap(q_session.results)
    quali_df["race_id"] = f"{season}_R{round_num:02d}"
    return quali_df


def extract_fastf1_race_features(
    season: int,
    round_num: int,
    cache_dir: str = "./fastf1_cache",
) -> pd.DataFrame:
    """단일 그랑프리의 FastF1 레이스 세션(R)에서 우천 여부와 보정 레이스 페이스를 추출합니다."""
    import fastf1

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(season, round_num, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    is_wet = detect_wet_race(session.laps, session.weather_data)
    clean_laps = filter_clean_air_laps(session.laps)
    pace_df = compute_fuel_tyre_corrected_pace(clean_laps)
    pace_df["race_id"] = f"{season}_R{round_num:02d}"
    pace_df["is_wet_f1"] = is_wet
    return pace_df


def enrich_jolpica_with_fastf1(
    jolpica_df: pd.DataFrame,
    cache_dir: str = "./fastf1_cache",
    use_lap_pace_as_target: bool = True,
    include_qualifying: bool = True,
) -> pd.DataFrame:
    """Jolpica DataFrame의 모든 (season, round)에 FastF1 레이스·예선 특징을 병합합니다."""
    race_frames: list[pd.DataFrame] = []
    quali_frames: list[pd.DataFrame] = []

    unique_races = (
        jolpica_df[["season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .itertuples(index=False)
    )

    for season, round_num in unique_races:
        try:
            race_frames.append(extract_fastf1_race_features(int(season), int(round_num), cache_dir))
        except Exception as exc:
            print(f"[경고] {season} R{round_num:02d} 레이스 로드 건너뜀: {exc}")

        if include_qualifying:
            try:
                quali_frames.append(extract_fastf1_quali_features(int(season), int(round_num), cache_dir))
            except Exception as exc:
                print(f"[경고] {season} R{round_num:02d} 예선 로드 건너뜀: {exc}")

    fastf1_race = pd.concat(race_frames, ignore_index=True) if race_frames else pd.DataFrame()
    fastf1_quali = pd.concat(quali_frames, ignore_index=True) if quali_frames else pd.DataFrame()

    return merge_fastf1_features(
        jolpica_df=jolpica_df,
        fastf1_df=fastf1_race,
        quali_df=fastf1_quali if not fastf1_quali.empty else None,
        use_lap_pace_as_target=use_lap_pace_as_target,
    )


def merge_fastf1_features(
    jolpica_df: pd.DataFrame,
    fastf1_df: pd.DataFrame,
    quali_df: pd.DataFrame | None = None,
    use_lap_pace_as_target: bool = True,
) -> pd.DataFrame:
    """추출된 FastF1 레이스·예선 특징을 Jolpica DataFrame에 병합합니다."""
    merged = jolpica_df.copy()

    if not fastf1_df.empty:
        merged = merged.merge(
            fastf1_df[
                ["race_id", "driver", "is_wet_f1", "clean_air_laps", "median_corrected_sec", "clean_air_pace_pct"]
            ],
            on=["race_id", "driver"],
            how="left",
        )
        race_wet_map = fastf1_df.groupby("race_id")["is_wet_f1"].max().to_dict()
        merged["is_wet"] = merged["race_id"].map(race_wet_map).fillna(merged.get("is_wet", 0)).astype(int)
        merged = merged.drop(columns=["is_wet_f1"])

        if use_lap_pace_as_target:
            if "pace_score" in merged.columns and "position_pace_score" not in merged.columns:
                merged["position_pace_score"] = merged["pace_score"]
            merged["pace_score"] = merged["clean_air_pace_pct"]

    if quali_df is not None and not quali_df.empty:
        merged = merged.merge(
            quali_df[["race_id", "driver", "best_quali_sec", "adj_quali_sec", "quali_pace_pct"]],
            on=["race_id", "driver"],
            how="left",
        )

    return merged


# ---------------------------------------------------------------------------
# 6. 오프라인 단위 테스트 (예선 트랙 에볼루션 보정 + 레이스 보정 병합 검증)
# ---------------------------------------------------------------------------
def run_self_test() -> None:
    """합성 Q1/Q2/Q3 결과 및 레이스 랩 데이터로 전체 보정·병합 파이프라인을 검증합니다."""
    rng = np.random.default_rng(7)
    drivers_true_sec = {"VER": 91.50, "LEC": 91.85, "HAM": 92.10, "ALO": 92.40}

    # 1) 합성 예선 데이터 (Q1->Q2 -0.40s, Q2->Q3 -0.30s 트랙 에볼루션 반영)
    quali_sample = pd.DataFrame(
        [
            {"Abbreviation": "VER", "Q1": 89.80, "Q2": 89.40, "Q3": 89.10},
            {"Abbreviation": "LEC", "Q1": 89.95, "Q2": 89.55, "Q3": 89.25},
            {"Abbreviation": "HAM", "Q1": 90.20, "Q2": 89.80, "Q3": np.nan},  # Q2 탈락
            {"Abbreviation": "ALO", "Q1": 90.45, "Q2": np.nan, "Q3": np.nan},  # Q1 탈락
        ]
    )
    quali_df = compute_quali_pace_gap(quali_sample)
    quali_df["race_id"] = "2024_R01"

    # 2) 합성 레이스 랩 데이터
    lap_rows = []
    cum_time = {d: float(idx * 4.0) for idx, d in enumerate(drivers_true_sec)}
    for lap in range(1, 31):
        for drv, base_sec in drivers_true_sec.items():
            compound = "SOFT" if lap <= 12 else "MEDIUM"
            tyre_life = lap if lap <= 12 else lap - 12
            fuel_delta = -0.06 * (lap - 15)
            comp_delta = -0.35 if compound == "SOFT" else 0.0
            deg_delta = (0.09 if compound == "SOFT" else 0.04) * (tyre_life - 1)
            lap_sec = base_sec + fuel_delta + comp_delta + deg_delta + rng.normal(0, 0.05)

            cum_time[drv] += lap_sec
            lap_rows.append(
                {
                    "Driver": drv,
                    "LapNumber": lap,
                    "LapTime": pd.to_timedelta(lap_sec, unit="s"),
                    "Time": pd.to_timedelta(cum_time[drv], unit="s"),
                    "Compound": compound,
                    "TyreLife": tyre_life,
                    "TrackStatus": "4" if lap == 15 else "1",
                    "PitInTime": pd.NaT if lap != 12 else pd.to_timedelta(cum_time[drv], unit="s"),
                    "PitOutTime": pd.NaT if lap != 13 else pd.to_timedelta(cum_time[drv] - 20, unit="s"),
                }
            )

    laps_df = pd.DataFrame(lap_rows)
    weather_df = pd.DataFrame({"Rainfall": [False, False, True, False]})

    is_wet = detect_wet_race(laps_df, weather_df)
    clean_laps = filter_clean_air_laps(laps_df, min_gap_ahead_sec=2.0)
    race_df = compute_fuel_tyre_corrected_pace(clean_laps)
    race_df["race_id"] = "2024_R01"
    race_df["is_wet_f1"] = is_wet

    jolpica_sample = pd.DataFrame(
        [
            {"season": 2024, "round": 1, "race_id": "2024_R01", "driver": d, "team_year": f"T_{d}_2024", "mechanical_dnf": 0, "is_wet": 0, "pace_score": 0.5}
            for d in ["VER", "LEC", "HAM", "ALO"]
        ]
    )
    merged = merge_fastf1_features(jolpica_sample, race_df, quali_df=quali_df, use_lap_pace_as_target=True)

    assert is_wet == 1
    assert list(merged.sort_values("quali_pace_pct", ascending=False)["driver"]) == ["VER", "LEC", "HAM", "ALO"]
    assert list(merged.sort_values("pace_score", ascending=False)["driver"]) == ["VER", "LEC", "HAM", "ALO"]
    print("오프라인 단위 테스트 통과 (예선 ΔQ + 레이스 ΔR)!")
    print(
        merged[
            ["race_id", "driver", "is_wet", "adj_quali_sec", "quali_pace_pct", "pace_score"]
        ].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FastF1 예선(ΔQ)·레이스(ΔR)·우천 병합기")
    parser.add_argument("--jolpica-csv", type=str, help="jolpica_f1_loader.py가 생성한 CSV 경로")
    parser.add_argument("--output", type=str, default="f1_enriched.csv", help="병합 결과 저장 경로")
    parser.add_argument("--cache-dir", type=str, default="./fastf1_cache", help="FastF1 로컬 캐시 폴더")
    parser.add_argument("--skip-quali", action="store_true", help="예선(Q) 세션 수집 생략")
    parser.add_argument("--self-test", action="store_true", help="오프라인 단위 테스트 실행")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.jolpica_csv:
        parser.error("--jolpica-csv 또는 --self-test 옵션을 지정해 주세요.")

    jolpica_df = pd.read_csv(args.jolpica_csv)
    enriched = enrich_jolpica_with_fastf1(
        jolpica_df,
        cache_dir=args.cache_dir,
        include_qualifying=not args.skip_quali,
    )
    enriched.to_csv(args.output, index=False)
    print(f"병합 완료: {len(enriched)}행 저장 ({args.output})")


if __name__ == "__main__":
    main()
