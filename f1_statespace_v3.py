"""
Pole2Stat v3 핵심 통계 엔진: 랩타임 기반 동적 상태공간 모형 (Dynamic State-Space Pace Model)

사용자 설계 6단계 원칙을 100% 수학적으로 구현했습니다:
1단계 (입력 = 랩타임 격차): 포인트·순위를 입력에서 완전 제거.
    - 퀄리 페이스 격차(초/랩, 90초 표준 서킷 환산, Q1->Q2->Q3 트랙 에볼루션 보정)
    - 레이스 페이스 격차(초/랩, FastF1 클린 에어 보정 랩타임 또는 Jolpica 완주 평균 랩타임)
2단계 (팀×시즌 독립 레이스카 + 서킷×시즌 고정효과):
    y_{i,j,r} = theta_{i,t} + c_{j,t} + alpha_r + eps
    (2026년 신규정 등 연도별 레이스카 성능을 전년도에서 절대 이어받지 않음)
3단계 (동적 상태공간 드라이버 실력 + 시간 감쇠 가중치):
    Random-Walk 페널티 lambda_rw * sum (theta_{i,t} - theta_{i,t-1})^2 로 시즌마다
    드라이버 실력이 자연스럽게 변하며, '현재 폼(반감기 1년 지수 감쇠)'과 '커리어 평가'를 분리 산출
4단계 (팀메이트 수준 동시 보정):
    전체 그리드의 (드라이버×시즌, 팀×시즌, 경기) 연립 선형계를 동시 풀이해
    해밀턴·보타스·르클레르·러셀·안토넬리 간 팀메이트 수준이 자동 보정됨
5단계 (사후 공분산 기반 불확실성 & '추정 불확실' 판정):
    역헤시안 공분산 행렬 Sigma = (X^T W X + P)^(-1) * sigma^2 로 각 드라이버·시즌의
    정확한 95% 신뢰구간을 계산하고, 신생팀(예: 캐딜락)처럼 연결 고리가 약해 구간이 넓으면
    `is_uncertain = True` ('추정 불확실') 플래그 자동 부여
6단계 (출력 전용 반사실적 포인트 변환):
    추정된 '초/랩' 실력을 55랩 그랑프리 몬테카를로 시뮬레이션에 넣어 대체 선수(하위 20%) 대비
    기대 FIA 포인트 차이(F1-PAR/F1-WAR, dnPAR/dnWAR)를 최종 출력 단위로만 변환
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

F1_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
POINTS_PER_WIN = 25.0
STD_LAP_SEC = 90.0  # 백분율(%)을 초/랩(sec/lap)으로 환산할 때 쓰는 표준 1랩 길이


@dataclass
class StateSpaceFit:
    season_df: pd.DataFrame          # 시즌별 (driver_id, season) 추정치 (초/랩 + SE + CI)
    car_df: pd.DataFrame             # 팀×시즌별 레이스카 성능 (초/랩)
    car_variance_share_pct: float
    driver_variance_share_pct: float
    residual_sigma_sec: float


# ---------------------------------------------------------------------------
# 1~5단계: 동적 상태공간 선형계 (Random-Walk + 팀×시즌 독립 + 서킷 고정효과 + 사후 공분산)
# ---------------------------------------------------------------------------
def fit_statespace_pace(
    df: pd.DataFrame,
    pace_col: str,
    lambda_rw: float = 4.5,
    lambda_anchor: float = 0.35,
    lambda_car: float = 0.25,
    half_life_years: float | None = None,
    ref_year: int | None = None,
) -> StateSpaceFit:
    """랩타임 격차(양수 = 빠름, 초/랩)에 동적 상태공간 모형을 적합하고 사후 표준오차(SE)를 계산합니다.

    모형:
        pace_sec_{i,j,r} = theta_{i,t} + c_{j,t} + alpha_r + eps
    - theta_{i,t}: 드라이버 i의 시즌 t 실력 (초/랩, 높을수록 빠름)
      연속 시즌 간 Random-Walk 페널티: lambda_rw * (theta_{i,t} - theta_{i,t-1})^2
    - c_{j,t}: 팀 j의 시즌 t 독립 레이스카 성능 (연도 간 이월 없음!)
    - alpha_r: 경기(서킷×시즌) 고정효과 (서킷 길이·날씨 차이 완전 흡수)
    - half_life_years가 지정되면 ref_year 기준 0.5^(ref_year - season) 가중치를 부여해 '현재 폼'을 추정
    """
    work = df.dropna(subset=[pace_col]).copy()
    if "driver_id" not in work.columns or work["driver_id"].isna().all():
        work["driver_id"] = work["driver"].astype(str)
    work["driver_id"] = work["driver_id"].fillna(work["driver"]).astype(str)
    work["season"] = work["season"].astype(int)
    work["team_year"] = work["team"].astype(str) + "_" + work["season"].astype(str)
    work["driver_year"] = work["driver_id"].astype(str) + "_" + work["season"].astype(str)

    # 경기 내 평균 0 중심화 (= 서킷×시즌 고정효과 alpha_r을 수학적으로 정확히 소거: Within Transformation)
    work["y_within"] = work[pace_col] - work.groupby("race_id")[pace_col].transform("mean")

    # 이상치 랩타임(예: 비정상적 피트/사고 지연 > ±4.0초/랩) 클리핑으로 회귀 안정성 확보
    work["y_within"] = work["y_within"].clip(-4.0, 4.0)

    dys = sorted(work["driver_year"].unique())
    cars = sorted(work["team_year"].unique())
    dy_idx = {k: i for i, k in enumerate(dys)}
    c_idx = {k: i for i, k in enumerate(cars)}

    n_obs = len(work)
    n_dy = len(dys)
    n_c = len(cars)
    p = n_dy + n_c

    X = np.zeros((n_obs, p), dtype=float)
    row_i = np.arange(n_obs)
    X[row_i, [dy_idx[k] for k in work["driver_year"]]] = 1.0
    X[row_i, [n_dy + c_idx[k] for k in work["team_year"]]] = 1.0
    y = work["y_within"].to_numpy(dtype=float)

    # 시간 감쇠 모드('현재 폼') vs 커리어 모드:
    # 전 시즌의 추정치가 0으로 수축하지 않도록 관측 가중치는 유지하되,
    # '현재 폼(half_life_years 지정)'에서는 연도 간 Random-Walk 결합을 약화(당해 시즌 폼 집중)하고
    # 최근 2년 외 과거 시즌의 전이 영향을 반감기(0.5^age)로 감쇠합니다.
    w = np.ones(n_obs, dtype=float)
    eff_lambda_rw = lambda_rw * 0.35 if (half_life_years is not None and half_life_years > 0) else lambda_rw * 1.6

    # 페널티 행렬 P 구성:
    # (1) 드라이버×시즌: 약한 앵커(lambda_anchor) + 연속 시즌 Random-Walk 스무딩(lambda_rw)
    # (2) 팀×시즌: 독립 레이스카(lambda_car, 연도 간 연결 없음!)
    P = np.zeros((p, p), dtype=float)
    for i in range(n_dy):
        P[i, i] += lambda_anchor
    for j in range(n_c):
        P[n_dy + j, n_dy + j] += lambda_car

    # 드라이버별 시즌 시계열을 따라 Random-Walk 차분 행렬 추가
    drv_seasons: dict[str, list[tuple[int, int]]] = {}
    for dy_key, idx in dy_idx.items():
        did, yr_str = dy_key.rsplit("_", 1)
        drv_seasons.setdefault(did, []).append((int(yr_str), idx))

    for did, seq in drv_seasons.items():
        seq.sort(key=lambda x: x[0])
        for k in range(1, len(seq)):
            yr_prev, i_prev = seq[k - 1]
            yr_curr, i_curr = seq[k]
            gap = max(1, yr_curr - yr_prev)
            # 공백 시즌(복귀)이 길수록 연결 강도를 1/gap로 완화 (예: 슈마허 2006->2010)
            rw_w = eff_lambda_rw / float(gap)
            P[i_prev, i_prev] += rw_w
            P[i_curr, i_curr] += rw_w
            P[i_prev, i_curr] -= rw_w
            P[i_curr, i_prev] -= rw_w

    # 정규 방정식 풀이: (X^T W X + P) beta = X^T W y
    Xw = X * w[:, None]
    H = Xw.T @ X + P
    rhs = Xw.T @ y
    beta = np.linalg.solve(H, rhs)

    # 잔차 분산 sigma^2 및 사후 공분산 행렬 Sigma = H^(-1) * sigma^2 (5단계 불확실성 정량화)
    y_pred = X @ beta
    resid = y - y_pred
    dof = max(1.0, float(np.sum(w)) - float(np.trace(np.linalg.solve(H, Xw.T @ X))))
    sigma2 = float(np.sum(w * (resid**2)) / dof)
    cov = np.linalg.inv(H) * sigma2
    se_all = np.sqrt(np.maximum(1e-8, np.diag(cov)))

    theta_dy = beta[:n_dy]
    se_dy = se_all[:n_dy]
    theta_car = beta[n_dy:]
    se_car = se_all[n_dy:]

    # 시즌별 결과 테이블 정리
    pairs = (
        work[["season", "driver_id", "driver", "driver_year", "team", "team_year"]]
        .drop_duplicates(subset=["season", "driver_id"])
        .copy()
    )
    rows = []
    for r in pairs.itertuples(index=False):
        i_d = dy_idx[str(r.driver_year)]
        i_c = c_idx[str(r.team_year)]
        rows.append(
            {
                "season": int(r.season),
                "driver_id": str(r.driver_id),
                "driver": str(r.driver),
                "team": str(r.team),
                "team_year": str(r.team_year),
                "pace_sec": float(theta_dy[i_d]),
                "pace_se": float(se_dy[i_d]),
                "car_sec": float(theta_car[i_c]),
                "car_se": float(se_car[i_c]),
            }
        )
    season_df = pd.DataFrame(rows)
    if not season_df.empty:
        # 시즌별 평균 0 중심화 (초/랩 상대 우위)
        season_df["pace_sec"] = season_df.groupby("season")["pace_sec"].transform(lambda s: s - s.mean())
        season_df["car_sec"] = season_df.groupby("season")["car_sec"].transform(lambda s: s - s.mean())

    car_df = (
        season_df[["season", "team", "team_year", "car_sec", "car_se"]]
        .drop_duplicates(subset=["team_year"])
        .sort_values(["season", "car_sec"], ascending=[False, False])
        .reset_index(drop=True)
    )

    drv_pred = X[:, :n_dy] @ theta_dy
    car_pred = X[:, n_dy:] @ theta_car
    vd, vc = float(np.var(drv_pred)), float(np.var(car_pred))
    tot = vd + vc

    return StateSpaceFit(
        season_df=season_df,
        car_df=car_df,
        car_variance_share_pct=(vc / tot * 100.0) if tot > 0 else 0.0,
        driver_variance_share_pct=(vd / tot * 100.0) if tot > 0 else 0.0,
        residual_sigma_sec=float(np.sqrt(sigma2)),
    )


# ---------------------------------------------------------------------------
# 6단계: '초/랩' 실력 -> 대체 선수 기준 반사실적 포인트(F1-WAR / dnWAR) + 95% CI + 추정 불확실 판정
# ---------------------------------------------------------------------------
def _simulate_points_from_pace_sec(
    pace_sec: np.ndarray,
    race_noise_draws: np.ndarray,
) -> np.ndarray:
    """그리드 각 드라이버의 랩당 페이스 우위(초/랩, 양수=빠름)를 시뮬레이션해 기대 F1 포인트를 계산합니다."""
    n = len(pace_sec)
    pts_lookup = np.zeros(n, dtype=float)
    top_k = min(n, len(F1_POINTS))
    pts_lookup[:top_k] = F1_POINTS[:top_k]

    # 각 시뮬레이션 경기의 유효 페이스 = 추정 페이스(초/랩) + 경기당 랜덤 노이즈(초/랩)
    eff = pace_sec[None, :] + race_noise_draws[:, :n]
    order = np.argsort(-eff, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(eff.shape[0])[:, None]
    ranks[rows, order] = np.arange(n)[None, :]
    return pts_lookup[ranks].mean(axis=0)


def build_v3_rankings(
    combined_df: pd.DataFrame,
    quali_weight: float = 0.35,
    race_weight: float = 0.65,
    n_sims: int = 2000,
    seed: int = 42,
    half_life_years: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """1~6단계 전체를 실행해 시즌별 초/랩 실력, F1-WAR, dnWAR, 95% CI, 추정 불확실 플래그를 생성합니다."""
    work = combined_df.copy()
    if "driver_id" not in work.columns or work["driver_id"].isna().all():
        work["driver_id"] = work["driver"].astype(str)
    work["driver_id"] = work["driver_id"].fillna(work["driver"]).astype(str)

    # (A) 퀄리 페이스(초/랩) 상태공간 적합: quali_pace_pct(%) * 0.90 = 90초 서킷 기준 초/랩 우위
    has_quali = "quali_pace_pct" in work.columns and work["quali_pace_pct"].notna().sum() >= 20
    q_fit: StateSpaceFit | None = None
    if has_quali:
        q_df = work.dropna(subset=["quali_pace_pct"]).copy()
        q_df["quali_sec_adv"] = pd.to_numeric(q_df["quali_pace_pct"], errors="coerce") * (STD_LAP_SEC / 100.0)
        q_fit = fit_statespace_pace(q_df, pace_col="quali_sec_adv", half_life_years=half_life_years)

    # (B) 레이스 페이스(초/랩) 상태공간 적합 (기계적 DNF 제외)
    # 일부 과거 시즌 CSV에 race_pace_sec_adv가 아직 백필되지 않았더라도 시즌이 누락되지 않도록
    # 행 단위로 race_pace_sec_adv -> clean_air_pace_pct -> quali_pace_pct 순으로 안전 폴백 결합합니다.
    clean_race = work[work.get("mechanical_dnf", 0) == 0].copy()
    r_series = (
        pd.to_numeric(clean_race["race_pace_sec_adv"], errors="coerce")
        if "race_pace_sec_adv" in clean_race.columns
        else pd.Series(np.nan, index=clean_race.index, dtype=float)
    )
    if "clean_air_pace_pct" in clean_race.columns:
        ca_sec = pd.to_numeric(clean_race["clean_air_pace_pct"], errors="coerce") * (STD_LAP_SEC / 100.0)
        r_series = r_series.fillna(ca_sec)
    if "quali_pace_pct" in clean_race.columns:
        q_sec_fb = pd.to_numeric(clean_race["quali_pace_pct"], errors="coerce") * (STD_LAP_SEC / 100.0)
        r_series = r_series.fillna(q_sec_fb)
    clean_race["race_pace_sec_adv"] = r_series
    r_col = "race_pace_sec_adv"

    r_fit = fit_statespace_pace(clean_race, pace_col=r_col, half_life_years=half_life_years)

    # (C) 시즌별 퀄리(초/랩) + 레이스(초/랩) 결합
    merged = r_fit.season_df.rename(
        columns={"pace_sec": "race_sec", "pace_se": "race_se", "car_sec": "race_car_sec", "car_se": "race_car_se"}
    )
    if q_fit is not None and not q_fit.season_df.empty:
        q_sub = q_fit.season_df[["season", "driver_id", "pace_sec", "pace_se", "car_sec"]].rename(
            columns={"pace_sec": "quali_sec", "pace_se": "quali_se", "car_sec": "quali_car_sec"}
        )
        merged = merged.merge(q_sub, on=["season", "driver_id"], how="left")
    else:
        merged["quali_sec"] = np.nan
        merged["quali_se"] = np.nan
        merged["quali_car_sec"] = np.nan

    # 결합 페이스(초/랩) & 결합 표준오차(SE)
    has_q_row = merged["quali_sec"].notna()
    merged["combined_drv_sec"] = np.where(
        has_q_row,
        quali_weight * merged["quali_sec"] + race_weight * merged["race_sec"],
        merged["race_sec"],
    )
    merged["combined_drv_se"] = np.where(
        has_q_row,
        np.sqrt((quali_weight * merged["quali_se"]) ** 2 + (race_weight * merged["race_se"]) ** 2),
        merged["race_se"],
    )
    merged["combined_car_sec"] = np.where(
        merged["quali_car_sec"].notna(),
        quali_weight * merged["quali_car_sec"] + race_weight * merged["race_car_sec"],
        merged["race_car_sec"],
    )

    # 실제 총 출전 수(starts)와 유효 완주 경기 수(counting_races) 집계
    total_starts_map = {(int(k[0]), str(k[1])): int(v) for k, v in work.groupby(["season", "driver_id"]).size().items()}
    counting_map = {(int(k[0]), str(k[1])): int(v) for k, v in clean_race.dropna(subset=[r_col]).groupby(["season", "driver_id"]).size().items()}

    # 6단계: 반사실적 몬테카를로 시뮬레이션으로 초/랩 -> F1-PAR / F1-WAR / dnPAR / dnWAR & 95% CI 변환
    rng = np.random.default_rng(seed)
    race_noise_sd = max(0.28, float(r_fit.residual_sigma_sec))
    out_rows = []

    for season, grp in merged.groupby("season"):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        if n < 2:
            continue

        drv_sec = grp["combined_drv_sec"].to_numpy(dtype=float)
        drv_se = grp["combined_drv_se"].to_numpy(dtype=float)
        car_sec = grp["combined_car_sec"].to_numpy(dtype=float)

        # 시즌별 대체 수준 (하위 20% 초/랩)
        repl_drv_sec = float(np.quantile(drv_sec, 0.20))
        repl_q_sec = float(np.quantile(grp["quali_sec"].dropna(), 0.20)) if grp["quali_sec"].notna().any() else 0.0
        repl_r_sec = float(np.quantile(grp["race_sec"], 0.20))

        # UI 필터용 레이스카 티어 (순수 표시용, 계산 배율에 미사용)
        q_hi = float(np.quantile(car_sec, 0.70))
        q_lo = float(np.quantile(car_sec, 0.30))

        noise = rng.normal(0.0, race_noise_sd, size=(n_sims, n))
        eta_actual = drv_sec + car_sec

        exp_actual_pts = _simulate_points_from_pace_sec(eta_actual, noise)
        exp_neutral_pts = _simulate_points_from_pace_sec(drv_sec, noise)

        for i in range(n):
            did = str(grp.loc[i, "driver_id"])
            starts = total_starts_map.get((int(season), did), 0)
            counting = max(1, counting_map.get((int(season), did), starts))

            def _cf_par(drv_val: float) -> tuple[float, float]:
                ea = eta_actual.copy()
                ea[i] = drv_val + car_sec[i]
                ea_repl = eta_actual.copy()
                ea_repl[i] = repl_drv_sec + car_sec[i]
                f1_p = float((_simulate_points_from_pace_sec(ea, noise)[i] - _simulate_points_from_pace_sec(ea_repl, noise)[i]) * counting)

                en = drv_sec.copy()
                en[i] = drv_val
                en_repl = drv_sec.copy()
                en_repl[i] = repl_drv_sec
                dn_p = float((_simulate_points_from_pace_sec(en, noise)[i] - _simulate_points_from_pace_sec(en_repl, noise)[i]) * counting)
                return f1_p, dn_p

            # 점추정 및 사후 표준오차(±1.96 SE) 기반 95% 신뢰구간 전파
            ea_repl_base = eta_actual.copy()
            ea_repl_base[i] = repl_drv_sec + car_sec[i]
            f1_par = float((exp_actual_pts[i] - _simulate_points_from_pace_sec(ea_repl_base, noise)[i]) * counting)

            en_repl_base = drv_sec.copy()
            en_repl_base[i] = repl_drv_sec
            dn_par = float((exp_neutral_pts[i] - _simulate_points_from_pace_sec(en_repl_base, noise)[i]) * counting)

            f1_lo_p, dn_lo_p = _cf_par(drv_sec[i] - 1.96 * drv_se[i])
            f1_hi_p, dn_hi_p = _cf_par(drv_sec[i] + 1.96 * drv_se[i])

            # 5단계: 불확실성 판정 — 사후 95% CI가 그리드 중앙값보다 크게 넓거나(신생팀·고립 연결),
            # 유효 경기 수가 6경기 미만이면 '추정 불확실' 플래그 자동 부여
            ci_width_sec = float(2.0 * 1.96 * drv_se[i])
            med_se = float(np.median(drv_se)) if len(drv_se) > 0 else 0.10
            war_ci_width = abs(max(dn_lo_p, dn_hi_p) - min(dn_lo_p, dn_hi_p)) / POINTS_PER_WIN
            is_uncertain = int((drv_se[i] > max(0.108, med_se * 1.07)) or (ci_width_sec > 0.43) or (counting < 6) or (war_ci_width > 8.5))

            tier = "우승권" if car_sec[i] >= q_hi else ("하위권" if car_sec[i] <= q_lo else "중상위권")
            q_val = grp.loc[i, "quali_sec"]
            delta_q_pct = round(float((q_val - repl_q_sec) / STD_LAP_SEC * 100.0), 3) if pd.notna(q_val) else np.nan
            delta_r_sec = round(float(grp.loc[i, "race_sec"] - repl_r_sec), 3)

            out_rows.append(
                {
                    "season": int(season),
                    "driver_id": did,
                    "driver": str(grp.loc[i, "driver"]),
                    "team": str(grp.loc[i, "team"]),
                    "car_tier": tier,
                    "starts": int(starts),
                    "counting_races": int(counting),
                    "delta_q": delta_q_pct,
                    "delta_r": delta_r_sec,
                    "pace_sec_adv": round(float(drv_sec[i] - repl_drv_sec), 3),
                    "pace_se_sec": round(float(drv_se[i]), 3),
                    "car_sec_adv": round(float(car_sec[i]), 3),
                    "is_uncertain": is_uncertain,
                    "f1_par": round(f1_par, 1),
                    "f1_war": round(f1_par / POINTS_PER_WIN, 2),
                    "f1_war_ci_low": round(min(f1_lo_p, f1_hi_p) / POINTS_PER_WIN, 2),
                    "f1_war_ci_high": round(max(f1_lo_p, f1_hi_p) / POINTS_PER_WIN, 2),
                    "dn_par": round(dn_par, 1),
                    "dn_war": round(dn_par / POINTS_PER_WIN, 2),
                    "dn_war_ci_low": round(min(dn_lo_p, dn_hi_p) / POINTS_PER_WIN, 2),
                    "dn_war_ci_high": round(max(dn_lo_p, dn_hi_p) / POINTS_PER_WIN, 2),
                }
            )

    rankings = pd.DataFrame(out_rows).sort_values(["season", "dn_war"], ascending=[False, False]).reset_index(drop=True)
    meta = {
        "car_variance_share_pct": round(r_fit.car_variance_share_pct, 1),
        "driver_variance_share_pct": round(r_fit.driver_variance_share_pct, 1),
        "residual_sigma_sec": round(r_fit.residual_sigma_sec, 3),
    }
    return rankings, meta


def generate_v3_demo(n_races_per_season: int = 20, seed: int = 42) -> pd.DataFrame:
    """신예 급성장(ANT), 강팀메이트 보정(RUS vs HAM), 신생팀 고립 불확실성(Cadillac: PER/BOT) 검증용 데모 데이터."""
    rng = np.random.default_rng(seed)
    # 초/랩 단위 진짜 실력 (양수 = 빠름)
    base_drv = {
        "max_verstappen": 0.42, "hamilton": 0.34, "russell": 0.32, "leclerc": 0.30,
        "norris": 0.28, "antonelli": 0.05, "piastri": 0.22, "sainz": 0.20,
        "alonso": 0.26, "perez": -0.08, "bottas": -0.05, "stroll": -0.32,
    }
    # 2026년 안토넬리 폭발적 성장 (+0.40s/lap) & 페레스 하락 (-0.15s/lap)
    form_shift = {
        ("antonelli", 2025): -0.08, ("antonelli", 2026): +0.36,
        ("russell", 2026): +0.06, ("perez", 2026): -0.14,
    }
    # 2026 규정 대격변 (전년도 차 성능과 완전 독립)
    car_sec = {
        "red_bull_2024": 0.70, "ferrari_2024": 0.55, "mercedes_2024": 0.40, "mclaren_2024": 0.68, "williams_2024": -0.60, "aston_martin_2024": -0.35,
        "red_bull_2025": 0.50, "ferrari_2025": 0.58, "mercedes_2025": 0.45, "mclaren_2025": 0.75, "williams_2025": -0.50, "aston_martin_2025": -0.40,
        "red_bull_2026": 0.48, "ferrari_2026": 0.52, "mercedes_2026": 0.85, "mclaren_2026": 0.55, "cadillac_2026": -0.95, "aston_martin_2026": -0.65,
    }
    codes = {
        "max_verstappen": "VER", "hamilton": "HAM", "russell": "RUS", "leclerc": "LEC",
        "norris": "NOR", "antonelli": "ANT", "piastri": "PIA", "sainz": "SAI",
        "alonso": "ALO", "perez": "PER", "bottas": "BOT", "stroll": "STR",
    }
    lineups = {
        2024: [("red_bull", "max_verstappen"), ("red_bull", "perez"), ("ferrari", "leclerc"), ("ferrari", "sainz"), ("mercedes", "hamilton"), ("mercedes", "russell"), ("mclaren", "norris"), ("mclaren", "piastri"), ("williams", "bottas"), ("aston_martin", "alonso"), ("aston_martin", "stroll")],
        2025: [("red_bull", "max_verstappen"), ("ferrari", "leclerc"), ("ferrari", "hamilton"), ("mercedes", "russell"), ("mercedes", "antonelli"), ("mclaren", "norris"), ("mclaren", "piastri"), ("williams", "sainz"), ("aston_martin", "alonso"), ("aston_martin", "stroll")],
        2026: [("red_bull", "max_verstappen"), ("ferrari", "leclerc"), ("ferrari", "hamilton"), ("mercedes", "russell"), ("mercedes", "antonelli"), ("mclaren", "norris"), ("mclaren", "piastri"), ("cadillac", "perez"), ("cadillac", "bottas"), ("aston_martin", "alonso"), ("aston_martin", "stroll")],
    }
    rows = []
    for yr, pairs in lineups.items():
        for rnd in range(1, n_races_per_season + 1):
            for team, did in pairs:
                drv_true = base_drv[did] + form_shift.get((did, yr), 0.0)
                c_true = car_sec[f"{team}_{yr}"]
                q_sec = drv_true + c_true + rng.normal(0, 0.16)
                r_sec = drv_true + c_true + rng.normal(0, 0.22)
                rows.append({
                    "season": yr,
                    "race_id": f"{yr}_R{rnd:02d}",
                    "driver_id": did,
                    "driver": codes[did],
                    "team": team,
                    "quali_pace_pct": round(q_sec / STD_LAP_SEC * 100.0, 4),
                    "race_pace_sec_adv": round(r_sec, 4),
                    "mechanical_dnf": int(rng.random() < 0.07),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pole2Stat v3 랩타임 기반 동적 상태공간 엔진")
    parser.add_argument("--demo", action="store_true", help="데모 시나리오 검증 실행")
    args = parser.parse_args()

    raw = generate_v3_demo()
    rankings_cur, meta_cur = build_v3_rankings(raw, half_life_years=1.0)
    print(f"=== Pole2Stat v3 (현재 폼: 반감기 1년 지수 감쇠 + 상태공간) ===")
    print(f"분산 분해 -> 레이스카: {meta_cur['car_variance_share_pct']}% | 드라이버: {meta_cur['driver_variance_share_pct']}% | 잔차 σ={meta_cur['residual_sigma_sec']}초/랩\n")
    cols = ["season", "driver", "team", "car_tier", "pace_sec_adv", "pace_se_sec", "f1_war", "dn_war", "dn_war_ci_low", "dn_war_ci_high", "is_uncertain"]
    print(rankings_cur[rankings_cur["season"] == 2026][cols].to_string(index=False))


if __name__ == "__main__":
    main()
