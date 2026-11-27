"""
F1-WAR v2 통계 엔진: Plackett-Luce 순위 로짓 모형 + 반사실적 몬테카를로 시뮬레이션 + 95% 신뢰구간

v1 프로토타입의 임의 가정(하드코딩 티어 배율 3.4/2.1/0.45, 선형 순위 점수 변환, 점추정 단일 출력)을
완전히 제거하고, 학술 표준 방법론(van Kesteren & Bergkamp, 2023; Bell et al., 2016)에 기반해
드라이버와 머신의 순수 기여도를 확률적으로 추정합니다.

핵심 수학적 개선:
1. Plackett-Luce (Rank-Ordered Logit) 가능도: 한 경기 20명의 순위 상호의존성(Zero-sum)을 직접 모델링
2. 계층적 L2 사전분포(MAP): u_driver + u_driver_year + u_team_year 3중 랜덤 효과 분리
3. Gumbel-Max 반사실적 몬테카를로 적분: 공식 F1 포인트(25-18-15-12-10-8-6-4-2-1)의
   기대값 차이로 F1-PAR·F1-WAR·dnPAR·dnWAR을 산출 (임의 배율 0개)
4. 경기 단위 부트스트랩(Race-Cluster Bootstrap): 각 드라이버의 95% 신뢰구간(CI) 산출

실행 예시:
    python f1_plackett_luce_v2.py --demo
    python f1_plackett_luce_v2.py --input data/raw_combined.csv --output data/f1_war_rankings_v2.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 공식 F1 그랑프리 포인트 체계 (1위~10위, 11위 이후 0점)
F1_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
POINTS_PER_WIN = 25.0


@dataclass
class PlackettLuceFit:
    driver_index: dict[str, int]
    dy_index: dict[str, int]
    car_index: dict[str, int]
    u_driver: np.ndarray
    u_dy: np.ndarray
    u_car: np.ndarray
    theta: np.ndarray
    car_variance_share_pct: float
    driver_variance_share_pct: float
    driver_season_df: pd.DataFrame


# ---------------------------------------------------------------------------
# 1. Plackett-Luce 음의 로그 가능도 및 해석적 기울기 (L-BFGS-B 고속 최적화)
# ---------------------------------------------------------------------------
def _encode_races(clean_df: pd.DataFrame) -> tuple[list[dict[str, np.ndarray]], dict[str, int], dict[str, int], dict[str, int]]:
    """각 경기별 완주 순위(1위 -> 최하위)에 따른 정수 인덱스 배열을 생성합니다."""
    id_col = "driver_id" if "driver_id" in clean_df.columns else "driver"
    drivers = sorted(clean_df[id_col].astype(str).unique())
    dys = sorted(clean_df["driver_year"].astype(str).unique())
    cars = sorted(clean_df["team_year"].astype(str).unique())

    d_idx = {k: i for i, k in enumerate(drivers)}
    dy_idx = {k: i for i, k in enumerate(dys)}
    c_idx = {k: i for i, k in enumerate(cars)}

    races: list[dict[str, np.ndarray]] = []
    for _, grp in clean_df.groupby("race_id", sort=False):
        ordered = grp.sort_values("finish_position", ascending=True)
        if len(ordered) < 2:
            continue
        races.append(
            {
                "d": np.array([d_idx[str(x)] for x in ordered[id_col]], dtype=np.int32),
                "dy": np.array([dy_idx[str(x)] for x in ordered["driver_year"]], dtype=np.int32),
                "c": np.array([c_idx[str(x)] for x in ordered["team_year"]], dtype=np.int32),
            }
        )
    return races, d_idx, dy_idx, c_idx


def _pl_neg_loglik_and_grad(
    theta: np.ndarray,
    races: list[dict[str, np.ndarray]],
    n_d: int,
    n_dy: int,
    n_c: int,
    lambda_d: float,
    lambda_dy: float,
    lambda_c: float,
) -> tuple[float, np.ndarray]:
    """L2 사전분포가 포함된 Plackett-Luce 음의 로그 가능도와 해석적 기울기를 계산합니다."""
    u_d = theta[:n_d]
    u_dy = theta[n_d : n_d + n_dy]
    u_c = theta[n_d + n_dy : n_d + n_dy + n_c]

    g_d = lambda_d * u_d
    g_dy = lambda_dy * u_dy
    g_c = lambda_c * u_c

    nll = 0.5 * (
        lambda_d * float(np.dot(u_d, u_d))
        + lambda_dy * float(np.dot(u_dy, u_dy))
        + lambda_c * float(np.dot(u_c, u_c))
    )

    for race in races:
        d_i, dy_i, c_i = race["d"], race["dy"], race["c"]
        eta = u_d[d_i] + u_dy[dy_i] + u_c[c_i]
        m = len(eta)

        # 수치 안정성을 위해 최대값 이동 후 역방향 누적합으로 각 단계 k의 log-sum-exp 계산
        shift = np.max(eta)
        exp_eta = np.exp(eta - shift)
        # suffix_sum[k] = sum_{j=k}^{m-1} exp(eta[j] - shift)
        suffix_sum = np.cumsum(exp_eta[::-1])[::-1]

        # 마지막 순위(m-1)를 제외한 k = 0 .. m-2 단계의 가능도 합산
        nll += float(np.sum(-(eta[:-1] - shift) + np.log(suffix_sum[:-1])))

        # 기울기: d(-ell)/d(eta_j) = exp(eta_j) * sum_{k=0}^{min(j, m-2)} (1 / suffix_sum[k]) - 1_{j < m-1}
        inv_suffix = np.zeros(m, dtype=float)
        inv_suffix[:-1] = 1.0 / suffix_sum[:-1]
        cum_inv = np.cumsum(inv_suffix)
        grad_eta = exp_eta * cum_inv
        grad_eta[:-1] -= 1.0

        np.add.at(g_d, d_i, grad_eta)
        np.add.at(g_dy, dy_i, grad_eta)
        np.add.at(g_c, c_i, grad_eta)

    grad = np.concatenate([g_d, g_dy, g_c])
    return nll, grad


def fit_plackett_luce(
    clean_df: pd.DataFrame,
    lambda_d: float = 3.0,
    lambda_dy: float = 6.0,
    lambda_c: float = 0.8,
    init_theta: np.ndarray | None = None,
) -> PlackettLuceFit:
    """교차 분류 Plackett-Luce 순위 로짓 모형을 L-BFGS-B로 적합합니다."""
    work = clean_df.copy()
    if "driver_id" not in work.columns or work["driver_id"].isna().all():
        work["driver_id"] = work["driver"].astype(str)
    work["driver_id"] = work["driver_id"].fillna(work["driver"]).astype(str)
    if "team_year" not in work.columns:
        work["team_year"] = work["team"].astype(str) + "_" + work["season"].astype(str)
    work["driver_year"] = work["driver_id"].astype(str) + "_" + work["season"].astype(str)

    races, d_idx, dy_idx, c_idx = _encode_races(work)
    n_d, n_dy, n_c = len(d_idx), len(dy_idx), len(c_idx)
    x0 = init_theta.copy() if (init_theta is not None and len(init_theta) == n_d + n_dy + n_c) else np.zeros(n_d + n_dy + n_c)

    res = minimize(
        _pl_neg_loglik_and_grad,
        x0,
        args=(races, n_d, n_dy, n_c, lambda_d, lambda_dy, lambda_c),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-9},
    )

    theta = res.x
    u_d = theta[:n_d] - np.mean(theta[:n_d])
    u_dy = theta[n_d : n_d + n_dy] - np.mean(theta[n_d : n_d + n_dy])
    u_c = theta[n_d + n_dy : n_d + n_dy + n_c] - np.mean(theta[n_d + n_dy : n_d + n_dy + n_c])

    # 시즌별 드라이버 실력(u_driver + u_driver_year) 테이블 생성 (driver_id 기준)
    pairs = work[["season", "driver_id", "driver", "driver_year", "team", "team_year"]].drop_duplicates(subset=["season", "driver_id"])
    ds_rows = []
    for r in pairs.itertuples(index=False):
        b = float(u_d[d_idx[str(r.driver_id)]])
        f = float(u_dy[dy_idx[str(r.driver_year)]])
        c = float(u_c[c_idx[str(r.team_year)]])
        ds_rows.append(
            {
                "season": int(r.season),
                "driver_id": str(r.driver_id),
                "driver": str(r.driver),
                "team": str(r.team),
                "team_year": str(r.team_year),
                "base_skill": b,
                "season_form": f,
                "driver_skill": b + f,
                "car_skill": c,
            }
        )
    ds_df = pd.DataFrame(ds_rows)
    if not ds_df.empty:
        ds_df["driver_skill"] = ds_df.groupby("season")["driver_skill"].transform(lambda s: s - s.mean())
        ds_df["car_skill"] = ds_df.groupby("season")["car_skill"].transform(lambda s: s - s.mean())

    # 분산 분해 비중 (관측치 기준 예측 분산 비교)
    drv_eff = np.array([u_d[d_idx[str(d)]] + u_dy[dy_idx[str(dy)]] for d, dy in zip(work["driver_id"], work["driver_year"])])
    car_eff = np.array([u_c[c_idx[str(c)]] for c in work["team_year"]])
    var_d, var_c = float(np.var(drv_eff)), float(np.var(car_eff))
    tot = var_d + var_c

    return PlackettLuceFit(
        driver_index=d_idx,
        dy_index=dy_idx,
        car_index=c_idx,
        u_driver=u_d,
        u_dy=u_dy,
        u_car=u_c,
        theta=theta,
        car_variance_share_pct=(var_c / tot * 100.0) if tot > 0 else 0.0,
        driver_variance_share_pct=(var_d / tot * 100.0) if tot > 0 else 0.0,
        driver_season_df=ds_df,
    )


# ---------------------------------------------------------------------------
# 2. Gumbel-Max 반사실적 몬테카를로 기대 포인트 적분 (임의 티어 배율 0개)
# ---------------------------------------------------------------------------
def _expected_points_from_eta(eta: np.ndarray, gumbel_draws: np.ndarray) -> np.ndarray:
    """잠재력 벡터 eta(길이 n)에 대해 Plackett-Luce 순위 분포의 기대 F1 포인트를 계산합니다."""
    n = len(eta)
    pts_lookup = np.zeros(n, dtype=float)
    top_k = min(n, len(F1_POINTS))
    pts_lookup[:top_k] = F1_POINTS[:top_k]

    # Gumbel-Max 트릭: eta + Gumbel(0,1)을 내림차순 정렬하면 정확한 Plackett-Luce 표본이 됨
    noisy = eta[None, :] + gumbel_draws[:, :n]
    order = np.argsort(-noisy, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(noisy.shape[0])[:, None]
    ranks[rows, order] = np.arange(n)[None, :]
    return pts_lookup[ranks].mean(axis=0)


def compute_counterfactual_war(
    fit: PlackettLuceFit,
    clean_df: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
    n_sims: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """각 드라이버를 대체 수준(하위 20%)으로 치환한 반사실적 시뮬레이션으로 F1-PAR/WAR·dnPAR/dnWAR을 계산합니다."""
    rng = np.random.default_rng(seed)
    ds = fit.driver_season_df.copy()

    id_col = "driver_id" if "driver_id" in clean_df.columns else "driver"
    counting_map = {
        (int(k[0]), str(k[1])): int(v)
        for k, v in clean_df.groupby(["season", id_col]).size().items()
    }
    raw_id_col = "driver_id" if (raw_df is not None and "driver_id" in raw_df.columns) else "driver"
    total_starts_map = (
        {(int(k[0]), str(k[1])): int(v) for k, v in raw_df.groupby(["season", raw_id_col]).size().items()}
        if raw_df is not None and not raw_df.empty
        else counting_map
    )

    out_rows = []
    for season, grp in ds.groupby("season"):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        if n < 2:
            continue

        # 공통 난수(Common Random Numbers)로 시뮬레이션 분산을 최소화
        gumbel = rng.gumbel(loc=0.0, scale=1.0, size=(n_sims, n))
        drv_skills = grp["driver_skill"].to_numpy(dtype=float)
        car_skills = grp["car_skill"].to_numpy(dtype=float)
        repl_skill = float(np.quantile(drv_skills, 0.20))

        # (A) 실제 그리드 기준 기대 포인트
        eta_actual_grid = drv_skills + car_skills
        exp_pts_actual = _expected_points_from_eta(eta_actual_grid, gumbel)

        # (B) 동일 머신(차량 중립, car=0) 기준 기대 포인트
        exp_pts_neutral = _expected_points_from_eta(drv_skills, gumbel)

        for i in range(n):
            drv_id = str(grp.loc[i, "driver_id"])
            drv = str(grp.loc[i, "driver"])
            starts = total_starts_map.get((int(season), drv_id), counting_map.get((int(season), drv_id), 0))
            counting = counting_map.get((int(season), drv_id), starts)

            # 반사실적 1: 실제 머신은 그대로 두고 드라이버 i만 대체 수준으로 교체
            eta_cf_actual = eta_actual_grid.copy()
            eta_cf_actual[i] = repl_skill + car_skills[i]
            exp_pts_repl_actual = _expected_points_from_eta(eta_cf_actual, gumbel)[i]

            # 반사실적 2: 모든 머신이 동일한 조건(car=0)에서 드라이버 i만 대체 수준으로 교체
            eta_cf_neutral = drv_skills.copy()
            eta_cf_neutral[i] = repl_skill
            exp_pts_repl_neutral = _expected_points_from_eta(eta_cf_neutral, gumbel)[i]

            f1_par = float((exp_pts_actual[i] - exp_pts_repl_actual) * counting)
            dn_par = float((exp_pts_neutral[i] - exp_pts_repl_neutral) * counting)

            out_rows.append(
                {
                    "season": int(season),
                    "driver_id": drv_id,
                    "driver": drv,
                    "team": str(grp.loc[i, "team"]),
                    "starts": int(starts),
                    "counting_races": int(counting),
                    "skill_logit": round(float(drv_skills[i] - repl_skill), 3),
                    "car_logit": round(float(car_skills[i]), 3),
                    "f1_par": round(f1_par, 1),
                    "f1_war": round(f1_par / POINTS_PER_WIN, 2),
                    "dn_par": round(dn_par, 1),
                    "dn_war": round(dn_par / POINTS_PER_WIN, 2),
                }
            )

    return pd.DataFrame(out_rows).sort_values(["season", "dn_war"], ascending=[False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. 경기 단위 부트스트랩(Race-Cluster Bootstrap) 95% 신뢰구간 추정
# ---------------------------------------------------------------------------
def compute_bootstrap_ci(
    clean_df: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
    n_bootstrap: int = 40,
    n_sims: int = 800,
    seed: int = 123,
) -> pd.DataFrame:
    """경기(race_id) 단위로 복원 추출해 F1-WAR와 dnWAR의 95% 신뢰구간을 계산합니다."""
    base_fit = fit_plackett_luce(clean_df)
    base_rankings = compute_counterfactual_war(base_fit, clean_df, raw_df=raw_df, n_sims=2000, seed=seed)

    rng = np.random.default_rng(seed)
    race_groups = {rid: grp for rid, grp in clean_df.groupby("race_id")}
    race_ids = list(race_groups.keys())

    boot_f1_war: dict[tuple[int, str], list[float]] = {}
    boot_dn_war: dict[tuple[int, str], list[float]] = {}

    for b in range(n_bootstrap):
        sampled_ids = rng.choice(race_ids, size=len(race_ids), replace=True)
        # 중복 추출된 경기에 고유 race_id 부여
        boot_frames = []
        for idx, rid in enumerate(sampled_ids):
            g = race_groups[rid].copy()
            g["race_id"] = f"{rid}_b{idx}"
            boot_frames.append(g)
        boot_df = pd.concat(boot_frames, ignore_index=True)

        b_fit = fit_plackett_luce(boot_df)
        b_rank = compute_counterfactual_war(b_fit, boot_df, raw_df=None, n_sims=n_sims, seed=seed + b + 1)

        for r in b_rank.itertuples(index=False):
            key = (int(r.season), str(r.driver_id))
            boot_f1_war.setdefault(key, []).append(float(r.f1_war))
            boot_dn_war.setdefault(key, []).append(float(r.dn_war))

    def _ci_bounds(vals: list[float], fallback: float) -> tuple[float, float]:
        if len(vals) < 3:
            return fallback, fallback
        return round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)

    f1_lo, f1_hi, dn_lo, dn_hi = [], [], [], []
    for r in base_rankings.itertuples(index=False):
        key = (int(r.season), str(r.driver_id))
        lo1, hi1 = _ci_bounds(boot_f1_war.get(key, []), float(r.f1_war))
        lo2, hi2 = _ci_bounds(boot_dn_war.get(key, []), float(r.dn_war))
        f1_lo.append(lo1)
        f1_hi.append(hi1)
        dn_lo.append(lo2)
        dn_hi.append(hi2)

    base_rankings["f1_war_ci_low"] = f1_lo
    base_rankings["f1_war_ci_high"] = f1_hi
    base_rankings["dn_war_ci_low"] = dn_lo
    base_rankings["dn_war_ci_high"] = dn_hi
    return base_rankings, base_fit


# ---------------------------------------------------------------------------
# 4. 오프라인 데모 및 해석적 기울기 단위 검증
# ---------------------------------------------------------------------------
def _verify_gradient_accuracy() -> float:
    """해석적 기울기와 수치 미분(유한차분)의 최대 오차를 검증합니다."""
    races = [{"d": np.array([0, 1, 2, 3]), "dy": np.array([0, 1, 2, 3]), "c": np.array([0, 0, 1, 1])}]
    theta = np.array([0.4, -0.2, 0.1, -0.3, 0.2, 0.0, -0.1, 0.1, 0.7, -0.5])
    _, grad = _pl_neg_loglik_and_grad(theta, races, 4, 4, 2, 1.5, 2.0, 0.5)
    eps = 1e-6
    num_grad = np.zeros_like(theta)
    for i in range(len(theta)):
        t_plus, t_minus = theta.copy(), theta.copy()
        t_plus[i] += eps
        t_minus[i] -= eps
        f_plus, _ = _pl_neg_loglik_and_grad(t_plus, races, 4, 4, 2, 1.5, 2.0, 0.5)
        f_minus, _ = _pl_neg_loglik_and_grad(t_minus, races, 4, 4, 2, 1.5, 2.0, 0.5)
        num_grad[i] = (f_plus - f_minus) / (2 * eps)
    return float(np.max(np.abs(grad - num_grad)))


def generate_plackett_luce_demo(n_races_per_season: int = 20, seed: int = 42) -> pd.DataFrame:
    """Plackett-Luce 생성 모형에서 3시즌(이적·신예 급성장 포함) 레이스 순위를 시뮬레이션합니다."""
    rng = np.random.default_rng(seed)
    true_drv = {"VER": 1.4, "HAM": 1.1, "ALO": 1.0, "LEC": 0.9, "NOR": 0.8, "RUS": 0.6, "SAI": 0.5, "ANT": 0.4, "PER": -0.3, "STR": -0.9}
    true_form = {("ANT", 2024): -0.5, ("ANT", 2025): 0.0, ("ANT", 2026): 0.85}
    true_car = {
        "RedBull_2024": 2.4, "Ferrari_2024": 1.8, "Mercedes_2024": 1.2, "McLaren_2024": 1.6, "AstonMartin_2024": -1.4,
        "RedBull_2025": 1.9, "Ferrari_2025": 2.0, "Mercedes_2025": 1.7, "McLaren_2025": 2.3, "AstonMartin_2025": -1.2,
        "RedBull_2026": 2.0, "Ferrari_2026": 2.1, "Mercedes_2026": 2.8, "McLaren_2026": 1.9, "AstonMartin_2026": -1.5,
    }
    lineups = {
        2024: [("RedBull", "VER"), ("RedBull", "PER"), ("Ferrari", "LEC"), ("Ferrari", "SAI"), ("Mercedes", "HAM"), ("Mercedes", "RUS"), ("McLaren", "NOR"), ("McLaren", "ALO"), ("AstonMartin", "ANT"), ("AstonMartin", "STR")],
        2025: [("RedBull", "VER"), ("RedBull", "PER"), ("Ferrari", "LEC"), ("Ferrari", "HAM"), ("Mercedes", "RUS"), ("Mercedes", "ANT"), ("McLaren", "NOR"), ("McLaren", "SAI"), ("AstonMartin", "ALO"), ("AstonMartin", "STR")],
        2026: [("RedBull", "VER"), ("RedBull", "PER"), ("Ferrari", "LEC"), ("Ferrari", "HAM"), ("Mercedes", "RUS"), ("Mercedes", "ANT"), ("McLaren", "NOR"), ("McLaren", "SAI"), ("AstonMartin", "ALO"), ("AstonMartin", "STR")],
    }
    rows = []
    for season, pairs in lineups.items():
        for rnd in range(1, n_races_per_season + 1):
            etas = np.array([true_drv[d] + true_form.get((d, season), 0.0) + true_car[f"{t}_{season}"] for t, d in pairs])
            order = np.argsort(-(etas + rng.gumbel(0.0, 1.0, len(pairs))))
            for pos, idx in enumerate(order, start=1):
                t, d = pairs[idx]
                rows.append(
                    {
                        "season": season,
                        "race_id": f"{season}_R{rnd:02d}",
                        "driver": d,
                        "driver_year": f"{d}_{season}",
                        "team": t,
                        "team_year": f"{t}_{season}",
                        "finish_position": pos,
                        "mechanical_dnf": int(rng.random() < 0.07),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="F1-WAR v2: Plackett-Luce + 반사실적 시뮬레이션 + 95% CI")
    parser.add_argument("--input", type=str, default=None, help="Jolpica 원천 CSV 경로 (finish_position 컬럼 필요)")
    parser.add_argument("--output", type=str, default="data/f1_war_rankings_v2.csv", help="출력 CSV 경로")
    parser.add_argument("--n-bootstrap", type=int, default=30, help="부트스트랩 반복 횟수 (기본값: 30)")
    parser.add_argument("--demo", action="store_true", help="합성 데이터로 v2 엔진 및 기울기 정확도 검증")
    args = parser.parse_args()

    grad_err = _verify_gradient_accuracy()
    assert grad_err < 1e-6, f"기울기 검증 실패: max_err={grad_err}"

    if args.demo or not args.input:
        raw_df = generate_plackett_luce_demo(n_races_per_season=20, seed=42)
        clean_df = raw_df[raw_df["mechanical_dnf"] == 0].copy()
    else:
        raw_df = pd.read_csv(args.input)
        clean_df = raw_df[raw_df.get("mechanical_dnf", 0) == 0].dropna(subset=["finish_position"]).copy()

    rankings, fit = compute_bootstrap_ci(clean_df, raw_df=raw_df, n_bootstrap=args.n_bootstrap)

    print(f"=== F1-WAR v2 (Plackett-Luce + 반사실적 시뮬레이션) 완료 ===")
    print(f"해석적 기울기 검증 최대 오차: {grad_err:.2e}")
    print(f"분산 분해 -> 머신: {fit.car_variance_share_pct:.1f}% | 드라이버: {fit.driver_variance_share_pct:.1f}%\n")

    latest = int(rankings["season"].max())
    cols = ["season", "driver", "team", "starts", "f1_war", "f1_war_ci_low", "f1_war_ci_high", "dn_war", "dn_war_ci_low", "dn_war_ci_high"]
    print(f"[{latest} 시즌 v2 랭킹 (95% 신뢰구간 포함)]")
    print(rankings[rankings["season"] == latest][cols].to_string(index=False))

    if not args.demo and args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        rankings.to_csv(args.output, index=False)
        print(f"\n저장 완료: {args.output}")


if __name__ == "__main__":
    main()