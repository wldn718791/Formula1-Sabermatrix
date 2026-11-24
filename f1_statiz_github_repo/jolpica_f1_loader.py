"""
Jolpica F1 API 레이스 결과 수집 및 다층 모형 입력용 DataFrame 생성기

Ergast API의 공식 후속인 Jolpica F1 API(https://api.jolpi.ca/ergast/f1)에서
최근 3시즌의 그랑프리 결승 결과와 리타이어 상태(status)를 페이지네이션으로
수집한 뒤, 기계적 고장(mechanical_dnf)과 드라이버 귀책 사고를 분류하여
`f1_multilevel_prototype.py`에 바로 넣을 수 있는 DataFrame으로 변환합니다.

로컬 실행 예시:
    python jolpica_f1_loader.py --seasons 2024 2025 2026 --output f1_recent_3seasons.csv
    python jolpica_f1_loader.py --self-test   # 오프라인 파싱·분류 로직 단위 검증
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

import pandas as pd

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"

# Jolpica/Ergast status 문자열 분류 사전
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
    """레이스 status 문자열을 finished / driver_incident / mechanical_dnf / other로 분류합니다."""
    text = (status or "").strip()
    if FINISHED_PATTERN.match(text):
        return "finished"

    lowered = text.lower()
    if any(kw in lowered for kw in DRIVER_INCIDENT_KEYWORDS):
        return "driver_incident"
    if any(kw in lowered for kw in MECHANICAL_KEYWORDS):
        return "mechanical_dnf"
    return "other_dnf"


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


def fetch_season_races(
    season: int,
    page_limit: int = 100,
    pause_seconds: float = 0.35,
) -> list[dict[str, Any]]:
    """단일 시즌의 모든 그랑프리 결과를 limit/offset 페이지네이션으로 끝까지 수집합니다."""
    races_by_round: dict[str, dict[str, Any]] = {}
    offset = 0

    while True:
        params = urllib.parse.urlencode({"limit": page_limit, "offset": offset})
        url = f"{JOLPICA_BASE_URL}/{season}/results.json?{params}"
        payload = _fetch_json_with_retry(url)

        mr_data = payload.get("MRData", {})
        total = int(mr_data.get("total", 0))
        batch_races = mr_data.get("RaceTable", {}).get("Races", [])

        # 같은 라운드의 Results가 페이지 경계에서 분할될 수 있으므로 라운드별로 병합
        for race in batch_races:
            rnd = str(race.get("round", ""))
            if rnd not in races_by_round:
                races_by_round[rnd] = {
                    **race,
                    "Results": list(race.get("Results", [])),
                }
            else:
                races_by_round[rnd]["Results"].extend(race.get("Results", []))

        offset += page_limit
        if offset >= total or not batch_races:
            break
        time.sleep(pause_seconds)

    return [races_by_round[k] for k in sorted(races_by_round, key=lambda x: int(x))]


def races_to_multilevel_df(races: list[dict[str, Any]]) -> pd.DataFrame:
    """Jolpica RaceTable JSON 리스트를 다층 모형 입력용 DataFrame으로 변환합니다.

    생성되는 주요 컬럼:
    - `driver`, `team`, `team_year`: 교차 분류 랜덤 효과 키
    - `status_category`, `mechanical_dnf`: 기계적 고장 필터링 플래그 (1이면 제외 대상)
    - `pace_score`: 경기 내 상대 완주 점수 (1위=1.0, 최하위=0.0; f1_multilevel_prototype 호환)
    """
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

            # 리타이어로 position이 누락된 경우 최하위(n_entrants)로 안전 처리
            raw_pos = res.get("position")
            finish_pos = int(raw_pos) if str(raw_pos).isdigit() else n_entrants
            grid_pos = int(res.get("grid", 0) or 0)
            points = float(res.get("points", 0.0) or 0.0)

            # 경기 내 상대 순위 점수: 1위=1.0, 최하위=0.0 (Bell / van Kesteren 방식)
            outperformed_score = (
                (n_entrants - finish_pos) / (n_entrants - 1) if n_entrants > 1 else 0.5
            )

            rows.append(
                {
                    "season": season,
                    "round": round_num,
                    "race_id": f"{season}_R{round_num:02d}",
                    "race_name": race_name,
                    "circuit_id": circuit_id,
                    "driver": driver_code,
                    "driver_id": driver_info.get("driverId", ""),
                    "team": team_id,
                    "team_year": f"{team_id}_{season}",
                    "grid": grid_pos,
                    "finish_position": finish_pos,
                    "points": points,
                    "status_raw": status_raw,
                    "status_category": status_cat,
                    "mechanical_dnf": int(status_cat == "mechanical_dnf"),
                    "is_wet": 0,  # Jolpica는 날씨를 제공하지 않으므로 필요 시 FastF1/OpenF1과 병합
                    "pace_score": round(outperformed_score, 4),
                }
            )

    return pd.DataFrame(rows)


def fetch_recent_seasons_df(seasons: list[int]) -> pd.DataFrame:
    """여러 시즌의 레이스 결과를 수집해 하나의 DataFrame으로 결합합니다."""
    all_races: list[dict[str, Any]] = []
    for season in seasons:
        all_races.extend(fetch_season_races(season))
    return races_to_multilevel_df(all_races)


def run_self_test() -> None:
    """네트워크 없이 Jolpica 응답 JSON 파싱과 리타이어 상태 분류를 검증합니다."""
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

    df = races_to_multilevel_df(sample_races)
    assert len(df) == 4
    assert df.loc[df["driver"] == "VER", "status_category"].item() == "finished"
    assert df.loc[df["driver"] == "LEC", "status_category"].item() == "finished"
    assert df.loc[df["driver"] == "HAM", "status_category"].item() == "driver_incident"
    assert df.loc[df["driver"] == "ALO", "mechanical_dnf"].item() == 1
    assert df.loc[df["driver"] == "ALO", "finish_position"].item() == 4
    print("오프라인 단위 테스트 통과!")
    print(
        df[
            [
                "race_id", "driver", "team_year", "finish_position",
                "status_raw", "status_category", "mechanical_dnf", "pace_score",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Jolpica F1 API 다층 모형 데이터 수집기")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2024, 2025, 2026],
        help="수집할 시즌 연도 목록 (기본값: 최근 3시즌 2024 2025 2026)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="f1_multilevel_input.csv",
        help="저장할 CSV 파일 경로",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="API 호출 없이 오프라인 파싱·분류 단위 테스트 실행",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    df = fetch_recent_seasons_df(args.seasons)
    df.to_csv(args.output, index=False)
    print(
        f"수집 완료: {len(df)}행 저장 ({args.output}) | "
        f"기계적 DNF 제외 대상: {int(df['mechanical_dnf'].sum())}건"
    )


if __name__ == "__main__":
    main()
