"""
Jolpica F1 API 레이스 결과 + 예선(Q1/Q2/Q3) 수집 및 다층 모형 입력용 DataFrame 생성기

Ergast API의 공식 후속인 Jolpica F1 API(https://api.jolpi.ca/ergast/f1)에서
결승 결과(`/results.json`)와 예선 Q1/Q2/Q3 랩타임(`/qualifying.json`)을 함께 수집해,
FastF1을 생략(`--skip-fastf1`)한 상태에서도 트랙 에볼루션이 보정된 예선 페이스 격차
(`quali_pace_pct`)와 기계적 DNF 분류(`mechanical_dnf`)를 모두 제공합니다.

로컬 실행 예시:
    python jolpica_f1_loader.py --seasons 2024 2025 2026 --output f1_recent_3seasons.csv
    python jolpica_f1_loader.py --self-test   # 오프라인 파싱·예선 보정·분류 단위 검증
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import pandas as pd

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"

FINISHED_PATTERN = re.compile(r"^(Finished|\+\d+\s+Laps?|Lapped)$", re.IGNORECASE)

DRIVER_INCIDENT_KEYWORDS = (
    "accident",
    "collision",
    "spun off",
    "damage",
    "disqualified",
    "track limits",
)

MECHANICAL_KEYWORDS = (
    "engine",
    "gearbox",
    "transmission",
    "hydraulics",
    "brake",
    "brakes",
    "power unit",
    "suspension",
    "electrical",
    "electronics",
    "overheating",
    "cooling",
    "water",
    "oil",
    "fuel",
    "clutch",
    "throttle",
    "steering",
    "radiator",
    "exhaust",
    "battery",
    "ers",
    "turbo",
    "mechanical",
    "drivetrain",
    "driveshaft",
    "differential",
    "puncture",
    "tyre",
    "wheel",
    "vibrations",
    "undertray",
    "floor",
    "wing",
    "retired",
    "withdrew",
    "technical",
    "power loss",
    "ignition",
    "alternator",
    "sensor",
)


def classify_finish_status(status: str) -> str:
    """레이스 status 문자열을 finished / driver_incident / mechanical_dnf / other_dnf로 분류합니다."""
    text = (status or "").strip()
    if FINISHED_PATTERN.match(text):
        return "finished"

    lowered = text.lower()
    if any(kw in lowered for kw in DRIVER_INCIDENT_KEYWORDS):
        return "driver_incident"
    if any(kw in lowered for kw in MECHANICAL_KEYWORDS):
        return "mechanical_dnf"
    return "other_dnf"


def parse_lap_time_str(time_str: str | None) -> float:
    """'1:29.179' 또는 '89.179' 형식의 예선 랩타임 문자열을 초(float)로 변환합니다."""
    if not time_str or not isinstance(time_str, str):
        return np.nan
    text = time_str.strip()
    if not text:
        return np.nan
    try:
        if ":" in text:
            mins, secs = text.split(":", 1)
            return float(mins) * 60.0 + float(secs)
        return float(text)
    except ValueError:
        return np.nan


def _fetch_json_with_retry(
    url: str,
    timeout: int = 20,
    max_retries: int = 4,
    base_delay: float = 1.0,
) -> dict[str, Any]:
    """Jolpica API 레이트 리밋(429) 및 일시적 오류에 대응하는 HTTP GET 헬퍼입니다."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "F1SabermetricsPrototype/1.0 (Python urllib)"},
    )
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries - 1:
                retry_after = exc.headers.get("Retry-After")
                wait_sec = float(retry_after) if retry_after else base_delay * (2**attempt)
                time.sleep(wait_sec)
                continue
            raise
        except urllib.error.URLError:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
                continue
            raise
    raise RuntimeError(f"요청 실패: {url}")


def _fetch_paginated_races(
    season: int,
    endpoint: str,
    list_key: str,
    page_limit: int = 100,
    pause_seconds: float = 0.45,
) -> list[dict[str, Any]]:
    """단일 시즌의 results 또는 qualifying 엔드포인트를 페이지네이션으로 모두 수집합니다."""
    races_by_round: dict[str, dict[str, Any]] = {}
    offset = 0

    while True:
        params = urllib.parse.urlencode({"limit": page_limit, "offset": offset})
        url = f"{JOLPICA_BASE_URL}/{season}/{endpoint}.json?{params}"
        payload = _fetch_json_with_retry(url)

        mr_data = payload.get("MRData", {})
        total = int(mr_data.get("total", 0))
        batch_races = mr_data.get("RaceTable", {}).get("Races", [])

        for race in batch_races:
            rnd = str(race.get("round", ""))
            if rnd not in races_by_round:
                races_by_round[rnd] = {
                    **race,
                    list_key: list(race.get(list_key, [])),
                }
            else:
                races_by_round[rnd][list_key].extend(race.get(list_key, []))

        offset += page_limit
        if offset >= total or not batch_races:
            break
        time.sleep(pause_seconds)

    return [races_by_round[k] for k in sorted(races_by_round, key=lambda x: int(x))]


def fetch_season_races(
    season: int,
    page_limit: int = 100,
    pause_seconds: float = 0.45,
) -> list[dict[str, Any]]:
    """단일 시즌의 모든 그랑프리 결승 결과를 수집합니다."""
    return _fetch_paginated_races(season, "results", "Results", page_limit, pause_seconds)


def fetch_season_qualifying(
    season: int,
    page_limit: int = 100,
    pause_seconds: float = 0.45,
) -> list[dict[str, Any]]:
    """단일 시즌의 모든 그랑프리 예선(Q1/Q2/Q3) 결과를 수집합니다."""
    return _fetch_paginated_races(season, "qualifying", "QualifyingResults", page_limit, pause_seconds)


def qualifying_races_to_df(quali_races: list[dict[str, Any]]) -> pd.DataFrame:
    """Jolpica 예선 JSON에서 Q1->Q2->Q3 트랙 에볼루션을 보정한 `quali_pace_pct`를 계산합니다."""
    all_rows: list[dict[str, Any]] = []

    for race in quali_races:
        season = int(race["season"])
        round_num = int(race["round"])
        race_id = f"{season}_R{round_num:02d}"
        q_results = race.get("QualifyingResults", [])
        if not q_results:
            continue

        parsed = []
        for item in q_results:
            drv = item.get("Driver", {})
            driver_code = drv.get("code") or drv.get("driverId", "UNK")
            driver_id = drv.get("driverId") or driver_code
            parsed.append(
                {
                    "driver_id": driver_id,
                    "driver": driver_code,
                    "q1": parse_lap_time_str(item.get("Q1")),
                    "q2": parse_lap_time_str(item.get("Q2")),
                    "q3": parse_lap_time_str(item.get("Q3")),
                }
            )

        qdf = pd.DataFrame(parsed)
        both_12 = qdf[["q1", "q2"]].dropna()
        evo_12 = float((both_12["q2"] - both_12["q1"]).median()) if not both_12.empty else 0.0
        both_23 = qdf[["q2", "q3"]].dropna()
        evo_23 = float((both_23["q3"] - both_23["q2"]).median()) if not both_23.empty else 0.0

        race_rows = []
        for row in qdf.itertuples(index=False):
            valid = [t for t in (row.q1, row.q2, row.q3) if pd.notna(t)]
            if not valid:
                continue
            best_raw = float(min(valid))
            if pd.notna(row.q3):
                adj_sec = float(min(row.q3, row.q2 + evo_23 if pd.notna(row.q2) else row.q3))
            elif pd.notna(row.q2):
                adj_sec = float(row.q2 + evo_23)
            else:
                adj_sec = float(row.q1 + evo_12 + evo_23)
            race_rows.append(
                {
                    "race_id": race_id,
                    "driver_id": row.driver_id,
                    "driver": row.driver,
                    "best_quali_sec": round(best_raw, 4),
                    "adj_quali_sec": round(adj_sec, 4),
                }
            )

        if not race_rows:
            continue
        rdf = pd.DataFrame(race_rows)
        med = float(rdf["adj_quali_sec"].median())
        rdf["quali_pace_pct"] = ((med - rdf["adj_quali_sec"]) / med * 100.0).round(4)
        all_rows.extend(rdf.to_dict(orient="records"))

    return pd.DataFrame(all_rows)


def races_to_multilevel_df(
    races: list[dict[str, Any]],
    quali_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Jolpica RaceTable JSON(+선택적 예선 DataFrame)을 다층 모형 입력용 DataFrame으로 변환합니다."""
    rows: list[dict[str, Any]] = []

    for race in races:
        season = int(race["season"])
        round_num = int(race["round"])
        race_name = race.get("raceName", "")
        circuit_id = race.get("Circuit", {}).get("circuitId", "")
        results = race.get("Results", [])
        n_entrants = len(results)

        for res in results:
            driver_info = res.get("Driver", {})
            constructor_info = res.get("Constructor", {})

            driver_code = driver_info.get("code") or driver_info.get("driverId", "UNK")
            team_id = constructor_info.get("constructorId", "unknown")
            status_raw = res.get("status", "")
            status_cat = classify_finish_status(status_raw)

            raw_pos = res.get("position")
            finish_pos = int(raw_pos) if str(raw_pos).isdigit() else n_entrants
            grid_pos = int(res.get("grid", 0) or 0)
            points = float(res.get("points", 0.0) or 0.0)

            outperformed_score = (
                (n_entrants - finish_pos) / (n_entrants - 1) if n_entrants > 1 else 0.5
            )

            # Jolpica Time.millis / laps 로 실제 완주 평균 랩타임(초/랩) 산출 (리드랩 완주자 기준)
            laps_done = int(res.get("laps", 0) or 0)
            time_obj = res.get("Time") or {}
            millis_raw = time_obj.get("millis")
            avg_lap_sec = (
                (float(millis_raw) / 1000.0) / laps_done
                if (millis_raw is not None and str(millis_raw).isdigit() and laps_done > 0)
                else np.nan
            )

            rows.append(
                {
                    "season": season,
                    "round": round_num,
                    "race_id": f"{season}_R{round_num:02d}",
                    "race_name": race_name,
                    "circuit_id": circuit_id,
                    "driver": driver_code,
                    "driver_id": driver_info.get("driverId") or driver_code,
                    "car_number": str(res.get("number", "")).strip(),
                    "driver_fullname": f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}".strip(),
                    "driver_nationality": driver_info.get("nationality", ""),
                    "team": team_id,
                    "team_year": f"{team_id}_{season}",
                    "grid": grid_pos,
                    "finish_position": finish_pos,
                    "points": points,
                    "status_raw": status_raw,
                    "status_category": status_cat,
                    "mechanical_dnf": int(status_cat == "mechanical_dnf"),
                    "laps_done": laps_done,
                    "avg_lap_sec": avg_lap_sec,
                    "is_wet": 0,
                    "pace_score": round(outperformed_score, 4),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 경기별 완주 평균 랩타임 중앙값 대비 초/랩 우위(race_pace_sec_adv, 양수=빠름) 계산
    # 주회 지연(+1 Lap 등) 완주자는 우승자 평균 랩타임 × (우승 랩수 / 완주 랩수)로 근사해 정보 손실 방지
    df["race_pace_sec_adv"] = np.nan
    for rid, grp in df.groupby("race_id"):
        valid_avg = grp["avg_lap_sec"].dropna()
        if valid_avg.empty:
            continue
        leader_laps = int(grp["laps_done"].max()) if "laps_done" in grp.columns else 0
        leader_avg = float(valid_avg.min())
        est_avg = grp["avg_lap_sec"].copy()
        for idx, row in grp.iterrows():
            if pd.isna(est_avg.loc[idx]) and row["status_category"] == "finished" and row["laps_done"] > 0 and leader_laps > 0:
                est_avg.loc[idx] = leader_avg * (leader_laps / float(row["laps_done"]))
        med_sec = float(est_avg.dropna().median()) if est_avg.notna().any() else np.nan
        if pd.notna(med_sec):
            # 양수 = 경기 중앙값보다 초/랩 더 빠름
            df.loc[grp.index, "race_pace_sec_adv"] = (med_sec - est_avg).clip(-3.5, 3.5).round(4)

    if quali_df is not None and not quali_df.empty:
        merge_key = "driver_id" if "driver_id" in quali_df.columns else "driver"
        df = df.merge(
            quali_df[["race_id", merge_key, "best_quali_sec", "adj_quali_sec", "quali_pace_pct"]],
            on=["race_id", merge_key],
            how="left",
        )
    return df


def fetch_recent_seasons_df(
    seasons: list[int],
    include_qualifying: bool = True,
) -> pd.DataFrame:
    """여러 시즌의 결승 결과와 예선(Q1/Q2/Q3) 페이스 격차를 함께 수집해 결합합니다."""
    all_races: list[dict[str, Any]] = []
    all_quali: list[dict[str, Any]] = []

    for season in seasons:
        all_races.extend(fetch_season_races(season))
        if include_qualifying:
            try:
                all_quali.extend(fetch_season_qualifying(season))
            except Exception as exc:
                print(f"[경고] {season} 시즌 Jolpica 예선 수집 건너뜀: {exc}")

    quali_df = qualifying_races_to_df(all_quali) if all_quali else None
    return races_to_multilevel_df(all_races, quali_df=quali_df)


def run_self_test() -> None:
    """네트워크 없이 Jolpica 결승 + 예선(Q1/Q2/Q3) 파싱과 분류를 검증합니다."""
    sample_races = [
        {
            "season": "2024",
            "round": "1",
            "raceName": "Bahrain Grand Prix",
            "Circuit": {"circuitId": "bahrain"},
            "Results": [
                {
                    "position": "1", "points": "26", "grid": "1", "status": "Finished",
                    "Driver": {"driverId": "max_verstappen", "code": "VER"},
                    "Constructor": {"constructorId": "red_bull"},
                },
                {
                    "position": "2", "points": "18", "grid": "2", "status": "+1 Lap",
                    "Driver": {"driverId": "leclerc", "code": "LEC"},
                    "Constructor": {"constructorId": "ferrari"},
                },
                {
                    "position": "3", "points": "15", "grid": "5", "status": "Collision",
                    "Driver": {"driverId": "hamilton", "code": "HAM"},
                    "Constructor": {"constructorId": "mercedes"},
                },
                {
                    "position": None, "points": "0", "grid": "7", "status": "Power Unit",
                    "Driver": {"driverId": "alonso", "code": "ALO"},
                    "Constructor": {"constructorId": "aston_martin"},
                },
            ],
        }
    ]
    sample_quali = [
        {
            "season": "2024",
            "round": "1",
            "QualifyingResults": [
                {"Driver": {"driverId": "max_verstappen", "code": "VER"}, "Q1": "1:30.000", "Q2": "1:29.600", "Q3": "1:29.200"},
                {"Driver": {"driverId": "leclerc", "code": "LEC"}, "Q1": "1:30.200", "Q2": "1:29.800", "Q3": "1:29.400"},
                {"Driver": {"driverId": "hamilton", "code": "HAM"}, "Q1": "1:30.500", "Q2": "1:30.100"},
                {"Driver": {"driverId": "alonso", "code": "ALO"}, "Q1": "1:30.800"},
            ],
        }
    ]

    qdf = qualifying_races_to_df(sample_quali)
    df = races_to_multilevel_df(sample_races, quali_df=qdf)
    assert len(df) == 4
    assert df["quali_pace_pct"].notna().all()
    assert list(df.sort_values("quali_pace_pct", ascending=False)["driver"]) == ["VER", "LEC", "HAM", "ALO"]
    assert df.loc[df["driver"] == "ALO", "mechanical_dnf"].item() == 1
    print("오프라인 단위 테스트 통과 (결승 + Jolpica 예선 ΔQ)!")
    print(
        df[
            ["race_id", "driver", "finish_position", "status_category", "mechanical_dnf", "quali_pace_pct", "pace_score"]
        ].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Jolpica F1 API 결승+예선 데이터 수집기")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026], help="수집할 시즌 연도 목록")
    parser.add_argument("--output", type=str, default="f1_multilevel_input.csv", help="저장할 CSV 파일 경로")
    parser.add_argument("--self-test", action="store_true", help="오프라인 단위 테스트 실행")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    df = fetch_recent_seasons_df(args.seasons, include_qualifying=True)
    df.to_csv(args.output, index=False)
    print(f"수집 완료: {len(df)}행 저장 ({args.output})")


if __name__ == "__main__":
    main()
