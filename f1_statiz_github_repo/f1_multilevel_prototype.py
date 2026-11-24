"""
F1 드라이버-머신 분리 교차 분류 다층 모형(Cross-Classified Multilevel Model) 프로토타입

Bell et al.(2016)의 교차 분류 랜덤 효과 모형과 f1metrics의 전처리 개념을 결합해
드라이버 순수 실력(u_driver)과 팀-연도별 머신 성능(u_team_year)을 분리하는
실전 입문용 파이썬 스크립트입니다.

포함된 추정 방식:
1. scikit-learn RidgeCV 기반 교차 분류 모형 (가우시안 사전분포 MAP/BLUP 근사)
2. statsmodels MixedLM Variance Components 모형 (REML 기반 분산 성분 직접 추정)
3. 순수 NumPy BLUP 솔버 (scikit-learn/statsmodels 미설치 환경에서도 즉시 실행 가능)

실행 예시:
    python f1_multilevel_prototype.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. 테스트용 F1 멀티시즌 합성 데이터 생성 (이적 네트워크 포함)
# ---------------------------------------------------------------------------
def generate_synthetic_f1_data(
    n_races_per_season: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """이적 기록이 포함된 3시즌 합성 F1 레이스 데이터를 생성합니다.

    종속변수 `pace_score`는 높을수록 빠른 퍼포먼스(예: 그리드 평균 대비
    클린 에어 랩타임 우위 %, 또는 표준화된 완주 점수)를 의미합니다.
    """
    rng = np.random.default_rng(seed)

    # 숨겨진 실제 드라이버 실력 (단위: 표준화 페이스 점수, 높을수록 우수)
    true_driver_skill = {
        "VER": 0.45,
        "HAM": 0.40,
        "ALO": 0.38,
        "LEC": 0.32,
        "NOR": 0.30,
        "RUS": 0.24,
        "SAI": 0.22,
        "PIA": 0.18,
        "PER": -0.05,
        "STR": -0.35,
    }

    # 숨겨진 실제 팀-연도별 머신 성능 (팀 효과가 드라이버보다 훨씬 큰 F1 현실 반영)
    true_car_perf = {
        "RedBull_2022": 1.10, "Ferrari_2022": 0.85, "Mercedes_2022": 0.55,
        "McLaren_2022": 0.05, "AstonMartin_2022": -0.55,
        "RedBull_2023": 1.40, "Ferrari_2023": 0.65, "Mercedes_2023": 0.60,
        "McLaren_2023": 0.45, "AstonMartin_2023": 0.30,
        "RedBull_2024": 1.05, "Ferrari_2024": 0.95, "Mercedes_2024": 0.70,
        "McLaren_2024": 1.15, "AstonMartin_2024": -0.15,
    }

    # 시즌별 라인업 (드라이버 이적이 팀 간 연결 고리 역할 수행)
    lineups = {
        2022: [
            ("RedBull", "VER"), ("RedBull", "PER"),
            ("Ferrari", "LEC"), ("Ferrari", "SAI"),
            ("Mercedes", "HAM"), ("Mercedes", "RUS"),
            ("McLaren", "NOR"), ("McLaren", "ALO"),
            ("AstonMartin", "PIA"), ("AstonMartin", "STR"),
        ],
        2023: [
            ("RedBull", "VER"), ("RedBull", "PER"),
            ("Ferrari", "LEC"), ("Ferrari", "SAI"),
            ("Mercedes", "HAM"), ("Mercedes", "RUS"),
            ("McLaren", "NOR"), ("McLaren", "PIA"),          # PIA 이적 (Aston -> McLaren)
            ("AstonMartin", "ALO"), ("AstonMartin", "STR"),  # ALO 이적 (McLaren -> Aston)
        ],
        2024: [
            ("RedBull", "VER"), ("RedBull", "PER"),
            ("Ferrari", "LEC"), ("Ferrari", "HAM"),          # HAM 이적 (Mercedes -> Ferrari)
            ("Mercedes", "RUS"), ("Mercedes", "SAI"),        # SAI 이적 (Ferrari -> Mercedes)
            ("McLaren", "NOR"), ("McLaren", "PIA"),
            ("AstonMartin", "ALO"), ("AstonMartin", "STR"),
        ],
    }

    rows = []
    for season, pairs in lineups.items():
        for race_id in range(1, n_races_per_season + 1):
            is_wet = int(rng.random() < 0.15)
            for team, driver in pairs:
                # 기계적 DNF 시뮬레이션 (약 8% 확률)
                mechanical_dnf = int(rng.random() < 0.08)
                team_year = f"{team}_{season}"

                # 레이스 결과 = 머신 성능 + 드라이버 실력 + 우천 변동 + 랜덤 노이즈
                noise = rng.normal(0.0, 0.22)
                pace_score = (
                    true_car_perf[team_year]
                    + true_driver_skill[driver]
                    + (0.12 * is_wet if driver in {"VER", "HAM", "ALO"} else 0.0)
                    + noise
                )
                rows.append(
                    {
                        "season": season,
                        "race_id": f"{season}_R{race_id:02d}",
                        "driver": driver,
                        "team": team,
                        "team_year": team_year,
                        "is_wet": is_wet,
                        "mechanical_dnf": mechanical_dnf,
                        "pace_score": pace_score,
                    }
                )

    return pd.DataFrame(rows), true_driver_skill, true_car_perf


# ---------------------------------------------------------------------------
# 2. 공통 전처리 (f1metrics 방식: 기계적 DNF 제외 + 상호작용 키 생성)
# ---------------------------------------------------------------------------
def preprocess_f1_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """기계적 리타이어를 제외하고 교차 분류에 필요한 컬럼을 정리합니다."""
    clean_df = df.copy()
    if "mechanical_dnf" in clean_df.columns:
        clean_df = clean_df[clean_df["mechanical_dnf"] == 0].copy()

    if "team_year" not in clean_df.columns:
        clean_df["team_year"] = (
            clean_df["team"].astype(str) + "_" + clean_df["season"].astype(str)
        )

    # 종속변수 중심화 (전체 평균 = 0)
    clean_df["target_centered"] = clean_df["pace_score"] - clean_df["pace_score"].mean()
    return clean_df


# ---------------------------------------------------------------------------
# 3. 방법 A: scikit-learn RidgeCV 기반 교차 분류 랜덤 효과 근사 (BLUP / MAP)
# ---------------------------------------------------------------------------
def fit_sklearn_cross_classified(
    df: pd.DataFrame,
    driver_prior_scale: float = 0.45,
    car_prior_scale: float = 1.00,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]] | None:
    """scikit-learn RidgeCV로 교차 분류 다층 모형의 수축 추정(Shrinkage)을 근사합니다.

    가우시안 랜덤 효과 모형 u_d ~ N(0, sigma_d^2), u_c ~ N(0, sigma_c^2)에서
    설계 행렬 열에 사전 표준편차 비율을 곱해 Ridge를 적합하면 드라이버와 머신에
    서로 다른 정규화 강도(lambda_d, lambda_c)를 부여할 수 있습니다.
    """
    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import OneHotEncoder
    except ImportError:
        return None

    enc_driver = OneHotEncoder(sparse_output=False)
    enc_car = OneHotEncoder(sparse_output=False)

    X_driver = enc_driver.fit_transform(df[["driver"]]) * driver_prior_scale
    X_car = enc_car.fit_transform(df[["team_year"]]) * car_prior_scale
    X_wet = df[["is_wet"]].to_numpy(dtype=float)

    X = np.hstack([X_driver, X_car, X_wet])
    y = df["target_centered"].to_numpy(dtype=float)

    model = RidgeCV(alphas=np.logspace(-2, 3, 40))
    model.fit(X, y)

    n_drivers = X_driver.shape[1]
    n_cars = X_car.shape[1]

    driver_effects = model.coef_[:n_drivers] * driver_prior_scale
    car_effects = model.coef_[n_drivers : n_drivers + n_cars] * car_prior_scale

    return _package_results(
        driver_names=list(enc_driver.categories_[0]),
        car_names=list(enc_car.categories_[0]),
        driver_effects=driver_effects,
        car_effects=car_effects,
        X_driver_raw=X_driver / driver_prior_scale,
        X_car_raw=X_car / car_prior_scale,
        y=y,
        y_pred=model.predict(X),
        extra_meta={"best_alpha": float(model.alpha_)},
    )


# ---------------------------------------------------------------------------
# 4. 방법 B: statsmodels MixedLM Variance Components 교차 분류 모형 (REML)
# ---------------------------------------------------------------------------
def fit_statsmodels_cross_classified(df: pd.DataFrame) -> dict[str, object] | None:
    """statsmodels의 MixedLM과 vc_formula를 사용해 교차 분류 다층 모형을 적합합니다.

    모든 관측치를 하나의 더미 그룹(`all_group = 1`)으로 묶고 `vc_formula`에
    `driver`와 `team_year`를 지정하면 REML로 교차 분류 분산 성분을 직접 추정합니다.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None

    work = df.copy()
    work["all_group"] = 1

    vc = {
        "driver": "0 + C(driver)",
        "team_year": "0 + C(team_year)",
    }
    md = smf.mixedlm(
        formula="target_centered ~ 1 + is_wet",
        data=work,
        groups=work["all_group"],
        vc_formula=vc,
    )
    res = md.fit(reml=True, method="lbfgs")

    var_driver = float(res.vcomp[0])
    var_car = float(res.vcomp[1])
    var_resid = float(res.scale)
    total_var = var_driver + var_car

    re_series = res.random_effects[1]
    driver_re = {
        k.split("[")[1].rstrip("]"): float(v)
        for k, v in re_series.items()
        if k.startswith("driver[")
    }
    car_re = {
        k.split("[")[1].rstrip("]"): float(v)
        for k, v in re_series.items()
        if k.startswith("team_year[")
    }

    return {
        "driver_df": pd.DataFrame(
            [{"driver": k, "est_driver_skill": v} for k, v in driver_re.items()]
        )
        .sort_values("est_driver_skill", ascending=False)
        .reset_index(drop=True),
        "car_df": pd.DataFrame(
            [{"team_year": k, "est_car_perf": v} for k, v in car_re.items()]
        )
        .sort_values("est_car_perf", ascending=False)
        .reset_index(drop=True),
        "variance_components": {
            "var_driver": var_driver,
            "var_car": var_car,
            "var_resid": var_resid,
            "car_share_pct": (var_car / total_var) * 100.0 if total_var > 0 else 0.0,
            "driver_share_pct": (var_driver / total_var) * 100.0 if total_var > 0 else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# 5. 방법 C: 순수 NumPy 닫힌 해(Closed-form) BLUP / Ridge 폴백 솔버
# ---------------------------------------------------------------------------
def fit_numpy_blup_cross_classified(
    df: pd.DataFrame,
    lambda_driver: float = 2.0,
    lambda_car: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """외부 ML 라이브러리 없이 (X^T X + Lambda) beta = X^T y를 직접 풀어 추정합니다.

    lambda_driver = sigma_eps^2 / sigma_driver^2,
    lambda_car    = sigma_eps^2 / sigma_car^2 에 대응합니다.
    """
    d_dummies = pd.get_dummies(df["driver"], dtype=float)
    c_dummies = pd.get_dummies(df["team_year"], dtype=float)

    X_driver = d_dummies.to_numpy()
    X_car = c_dummies.to_numpy()
    X_wet = df[["is_wet"]].to_numpy(dtype=float)
    X_intercept = np.ones((len(df), 1), dtype=float)

    X = np.hstack([X_intercept, X_driver, X_car, X_wet])
    y = df["target_centered"].to_numpy(dtype=float)

    n_d = X_driver.shape[1]
    n_c = X_car.shape[1]

    # 절편(0)과 고정효과(is_wet)에는 페널티 미적용, 랜덤 효과에만 대각 페널티 적용
    penalties = np.concatenate(
        [[0.0], np.full(n_d, lambda_driver), np.full(n_c, lambda_car), [1e-4]]
    )
    beta = np.linalg.solve(X.T @ X + np.diag(penalties), X.T @ y)

    driver_effects = beta[1 : 1 + n_d]
    car_effects = beta[1 + n_d : 1 + n_d + n_c]
    y_pred = X @ beta

    return _package_results(
        driver_names=list(d_dummies.columns),
        car_names=list(c_dummies.columns),
        driver_effects=driver_effects,
        car_effects=car_effects,
        X_driver_raw=X_driver,
        X_car_raw=X_car,
        y=y,
        y_pred=y_pred,
        extra_meta={"lambda_driver": lambda_driver, "lambda_car": lambda_car},
    )


def _package_results(
    driver_names: list[str],
    car_names: list[str],
    driver_effects: np.ndarray,
    car_effects: np.ndarray,
    X_driver_raw: np.ndarray,
    X_car_raw: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    extra_meta: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    driver_centered = driver_effects - driver_effects.mean()
    car_centered = car_effects - car_effects.mean()

    driver_df = (
        pd.DataFrame({"driver": driver_names, "est_driver_skill": driver_centered})
        .sort_values("est_driver_skill", ascending=False)
        .reset_index(drop=True)
    )
    car_df = (
        pd.DataFrame({"team_year": car_names, "est_car_perf": car_centered})
        .sort_values("est_car_perf", ascending=False)
        .reset_index(drop=True)
    )

    var_d = float(np.var(X_driver_raw @ driver_centered))
    var_c = float(np.var(X_car_raw @ car_centered))
    total_explained = var_d + var_c
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    summary = {
        **extra_meta,
        "r2_score": 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "car_variance_share_pct": (var_c / total_explained) * 100.0 if total_explained > 0 else 0.0,
        "driver_variance_share_pct": (var_d / total_explained) * 100.0 if total_explained > 0 else 0.0,
    }
    return driver_df, car_df, summary


# ---------------------------------------------------------------------------
# 6. 메인 실행부: 합성 데이터로 실제값 복원 정확도 검증
# ---------------------------------------------------------------------------
def main() -> None:
    raw_df, true_driver, true_car = generate_synthetic_f1_data(n_races_per_season=20, seed=42)
    clean_df = preprocess_f1_dataframe(raw_df)
    print(f"총 관측치: {len(raw_df)}건 -> 기계적 DNF 제외 후: {len(clean_df)}건\n")

    # (1) scikit-learn이 있으면 RidgeCV, 없으면 순수 NumPy BLUP 실행
    sk_res = fit_sklearn_cross_classified(clean_df)
    if sk_res is not None:
        method_label = "scikit-learn RidgeCV 교차 분류 모형"
        driver_df, car_df, summary = sk_res
    else:
        method_label = "NumPy 닫힌 해 BLUP 교차 분류 모형 (scikit-learn 미설치 폴백)"
        driver_df, car_df, summary = fit_numpy_blup_cross_classified(clean_df)

    true_d_mean = float(np.mean(list(true_driver.values())))
    driver_df["true_skill_centered"] = driver_df["driver"].map(
        lambda d: true_driver[d] - true_d_mean
    )
    corr_d = driver_df["est_driver_skill"].corr(driver_df["true_skill_centered"])

    true_c_mean = float(np.mean(list(true_car.values())))
    car_df["true_car_centered"] = car_df["team_year"].map(
        lambda c: true_car[c] - true_c_mean
    )
    corr_c = car_df["est_car_perf"].corr(car_df["true_car_centered"])

    print(f"=== [{method_label}] ===")
    print(f"설명력 (R^2): {summary['r2_score']:.3f}")
    print(
        f"분산 비중 -> 머신(팀-연도): {summary['car_variance_share_pct']:.1f}% | "
        f"드라이버: {summary['driver_variance_share_pct']:.1f}%"
    )
    print(f"실제 숨겨진 값 복원 상관계수 -> 드라이버 r={corr_d:.3f}, 머신 r={corr_c:.3f}\n")

    print("[드라이버 순수 실력 랭킹 (추정치 vs 실제값)]")
    print(driver_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n[팀-연도별 머신 성능 상위 5개 (추정치 vs 실제값)]")
    print(car_df.head(5).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # (2) statsmodels MixedLM 실행 (설치된 환경에서만)
    sm_out = fit_statsmodels_cross_classified(clean_df)
    if sm_out is not None:
        vc = sm_out["variance_components"]
        print("\n=== [statsmodels MixedLM (REML) 결과] ===")
        print(
            f"분산 성분 -> 머신: {vc['var_car']:.3f} ({vc['car_share_pct']:.1f}%) | "
            f"드라이버: {vc['var_driver']:.3f} ({vc['driver_share_pct']:.1f}%)"
        )
        print(sm_out["driver_df"].head(5).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    else:
        print(
            "\n[안내] 로컬에서 `pip install scikit-learn statsmodels` 후 실행하면 "
            "RidgeCV 및 MixedLM(REML) 결과도 자동으로 함께 출력됩니다."
        )


if __name__ == "__main__":
    main()
