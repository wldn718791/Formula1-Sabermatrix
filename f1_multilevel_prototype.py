"""
F1 드라이버-머신 분리 교차 분류 다층 모형(Cross-Classified Multilevel Model) 프로토타입

Bell et al.(2016)의 교차 분류 랜덤 효과 구조를 구현해
  - 드라이버 기본 실력 (u_driver)
  - 드라이버-시즌별 폼 (u_driver_year)
  - 팀-연도별 머신 성능 (u_team_year)
세 요소를 동시에 분리합니다. 각 시즌의 최종 드라이버 실력은
`u_driver + u_driver_year`로 산출되어 루키 성장·전성기·에이징이 시즌별로 반영됩니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_f1_data(
    n_races_per_season: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """이적 기록과 신예 성장(시즌별 폼 변화)이 포함된 3시즌 합성 데이터를 생성합니다."""
    rng = np.random.default_rng(seed)

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

    # 신예(PIA)의 시즌별 급성장 시뮬레이션 (2022 루키 -0.18 -> 2024 전성기 +0.25)
    true_season_form = {
        ("PIA", 2022): -0.18,
        ("PIA", 2023): 0.00,
        ("PIA", 2024): 0.25,
    }

    true_car_perf = {
        "RedBull_2022": 1.10, "Ferrari_2022": 0.85, "Mercedes_2022": 0.55,
        "McLaren_2022": 0.05, "AstonMartin_2022": -0.55,
        "RedBull_2023": 1.40, "Ferrari_2023": 0.65, "Mercedes_2023": 0.60,
        "McLaren_2023": 0.45, "AstonMartin_2023": 0.30,
        "RedBull_2024": 1.05, "Ferrari_2024": 0.95, "Mercedes_2024": 0.70,
        "McLaren_2024": 1.15, "AstonMartin_2024": -0.15,
    }

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
            ("McLaren", "NOR"), ("McLaren", "PIA"),
            ("AstonMartin", "ALO"), ("AstonMartin", "STR"),
        ],
        2024: [
            ("RedBull", "VER"), ("RedBull", "PER"),
            ("Ferrari", "LEC"), ("Ferrari", "HAM"),
            ("Mercedes", "RUS"), ("Mercedes", "SAI"),
            ("McLaren", "NOR"), ("McLaren", "PIA"),
            ("AstonMartin", "ALO"), ("AstonMartin", "STR"),
        ],
    }

    rows = []
    for season, pairs in lineups.items():
        for race_id in range(1, n_races_per_season + 1):
            is_wet = int(rng.random() < 0.15)
            for team, driver in pairs:
                mechanical_dnf = int(rng.random() < 0.08)
                team_year = f"{team}_{season}"
                driver_year = f"{driver}_{season}"
                noise = rng.normal(0.0, 0.22)
                pace_score = (
                    true_car_perf[team_year]
                    + true_driver_skill[driver]
                    + true_season_form.get((driver, season), 0.0)
                    + (0.12 * is_wet if driver in {"VER", "HAM", "ALO"} else 0.0)
                    + noise
                )
                rows.append(
                    {
                        "season": season,
                        "race_id": f"{season}_R{race_id:02d}",
                        "driver": driver,
                        "driver_year": driver_year,
                        "team": team,
                        "team_year": team_year,
                        "is_wet": is_wet,
                        "mechanical_dnf": mechanical_dnf,
                        "pace_score": pace_score,
                    }
                )

    return pd.DataFrame(rows), true_driver_skill, true_car_perf


def preprocess_f1_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """기계적 리타이어를 제외하고 교차 분류 키(driver_year, team_year)를 생성합니다."""
    clean_df = df.copy()
    if "mechanical_dnf" in clean_df.columns:
        clean_df = clean_df[clean_df["mechanical_dnf"] == 0].copy()

    if "team_year" not in clean_df.columns:
        clean_df["team_year"] = clean_df["team"].astype(str) + "_" + clean_df["season"].astype(str)
    if "driver_year" not in clean_df.columns:
        clean_df["driver_year"] = clean_df["driver"].astype(str) + "_" + clean_df["season"].astype(str)

    clean_df["target_centered"] = clean_df["pace_score"] - clean_df["pace_score"].mean()
    return clean_df


def _build_driver_season_df(
    df: pd.DataFrame,
    driver_map: dict[str, float],
    driver_year_map: dict[str, float],
) -> pd.DataFrame:
    """드라이버 기본 실력(u_driver) + 시즌별 폼(u_driver_year)을 합산한 시즌별 테이블을 생성합니다."""
    pairs = df[["season", "driver", "driver_year"]].drop_duplicates()
    rows = []
    for row in pairs.itertuples(index=False):
        base = float(driver_map.get(row.driver, 0.0))
        form = float(driver_year_map.get(row.driver_year, 0.0))
        rows.append(
            {
                "season": int(row.season),
                "driver": str(row.driver),
                "driver_year": str(row.driver_year),
                "base_driver_skill": base,
                "season_form": form,
                "est_driver_season_skill": base + form,
            }
        )
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        # 시즌별 평균 0 기준 중심화 (시즌 간 드리프트 방지)
        out_df["est_driver_season_skill"] = out_df.groupby("season")["est_driver_season_skill"].transform(
            lambda s: s - s.mean()
        )
    return out_df.sort_values(["season", "est_driver_season_skill"], ascending=[False, False]).reset_index(drop=True)


def fit_sklearn_cross_classified(
    df: pd.DataFrame,
    driver_prior_scale: float = 0.40,
    driver_year_prior_scale: float = 0.35,
    car_prior_scale: float = 1.00,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]] | None:
    """scikit-learn RidgeCV로 driver + driver_year + team_year 3중 랜덤 효과를 추정합니다."""
    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import OneHotEncoder
    except ImportError:
        return None

    enc_driver = OneHotEncoder(sparse_output=False)
    enc_driver_year = OneHotEncoder(sparse_output=False)
    enc_car = OneHotEncoder(sparse_output=False)

    X_driver = enc_driver.fit_transform(df[["driver"]]) * driver_prior_scale
    X_dy = enc_driver_year.fit_transform(df[["driver_year"]]) * driver_year_prior_scale
    X_car = enc_car.fit_transform(df[["team_year"]]) * car_prior_scale
    X_wet = df[["is_wet"]].to_numpy(dtype=float)

    X = np.hstack([X_driver, X_dy, X_car, X_wet])
    y = df["target_centered"].to_numpy(dtype=float)

    model = RidgeCV(alphas=np.logspace(-2, 3, 40))
    model.fit(X, y)

    n_d = X_driver.shape[1]
    n_dy = X_dy.shape[1]
    n_c = X_car.shape[1]

    driver_effects = model.coef_[:n_d] * driver_prior_scale
    dy_effects = model.coef_[n_d : n_d + n_dy] * driver_year_prior_scale
    car_effects = model.coef_[n_d + n_dy : n_d + n_dy + n_c] * car_prior_scale

    return _package_results(
        df=df,
        driver_names=list(enc_driver.categories_[0]),
        dy_names=list(enc_driver_year.categories_[0]),
        car_names=list(enc_car.categories_[0]),
        driver_effects=driver_effects,
        dy_effects=dy_effects,
        car_effects=car_effects,
        X_driver_raw=X_driver / driver_prior_scale,
        X_dy_raw=X_dy / driver_year_prior_scale,
        X_car_raw=X_car / car_prior_scale,
        y=y,
        y_pred=model.predict(X),
        extra_meta={"best_alpha": float(model.alpha_)},
    )


def fit_statsmodels_cross_classified(df: pd.DataFrame) -> dict[str, object] | None:
    """statsmodels MixedLM 선택적 추정기 (수렴 실패 시 None 반환)."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None

    work = df.copy()
    work["all_group"] = 1
    vc = {"driver": "0 + C(driver)", "driver_year": "0 + C(driver_year)", "team_year": "0 + C(team_year)"}
    has_wet_var = "is_wet" in work.columns and work["is_wet"].nunique() > 1
    formula = "target_centered ~ 1 + is_wet" if has_wet_var else "target_centered ~ 1"
    try:
        md = smf.mixedlm(formula=formula, data=work, groups=work["all_group"], vc_formula=vc)
        res = md.fit(reml=True, method="lbfgs")
    except Exception:
        return None
    return {"result": res}


def fit_numpy_blup_cross_classified(
    df: pd.DataFrame,
    lambda_driver: float = 2.5,
    lambda_driver_year: float = 3.2,
    lambda_car: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """외부 ML 라이브러리 없이 (X^T X + Lambda) beta = X^T y로 3중 랜덤 효과를 추정합니다."""
    d_dummies = pd.get_dummies(df["driver"], dtype=float)
    dy_dummies = pd.get_dummies(df["driver_year"], dtype=float)
    c_dummies = pd.get_dummies(df["team_year"], dtype=float)

    X_driver = d_dummies.to_numpy()
    X_dy = dy_dummies.to_numpy()
    X_car = c_dummies.to_numpy()
    X_wet = df[["is_wet"]].to_numpy(dtype=float)
    X_intercept = np.ones((len(df), 1), dtype=float)

    X = np.hstack([X_intercept, X_driver, X_dy, X_car, X_wet])
    y = df["target_centered"].to_numpy(dtype=float)

    n_d = X_driver.shape[1]
    n_dy = X_dy.shape[1]
    n_c = X_car.shape[1]

    penalties = np.concatenate(
        [
            [0.0],
            np.full(n_d, lambda_driver),
            np.full(n_dy, lambda_driver_year),
            np.full(n_c, lambda_car),
            [1e-4],
        ]
    )
    beta = np.linalg.solve(X.T @ X + np.diag(penalties), X.T @ y)

    driver_effects = beta[1 : 1 + n_d]
    dy_effects = beta[1 + n_d : 1 + n_d + n_dy]
    car_effects = beta[1 + n_d + n_dy : 1 + n_d + n_dy + n_c]
    y_pred = X @ beta

    return _package_results(
        df=df,
        driver_names=list(d_dummies.columns),
        dy_names=list(dy_dummies.columns),
        car_names=list(c_dummies.columns),
        driver_effects=driver_effects,
        dy_effects=dy_effects,
        car_effects=car_effects,
        X_driver_raw=X_driver,
        X_dy_raw=X_dy,
        X_car_raw=X_car,
        y=y,
        y_pred=y_pred,
        extra_meta={"lambda_driver": lambda_driver, "lambda_driver_year": lambda_driver_year, "lambda_car": lambda_car},
    )


def _package_results(
    df: pd.DataFrame,
    driver_names: list[str],
    dy_names: list[str],
    car_names: list[str],
    driver_effects: np.ndarray,
    dy_effects: np.ndarray,
    car_effects: np.ndarray,
    X_driver_raw: np.ndarray,
    X_dy_raw: np.ndarray,
    X_car_raw: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    extra_meta: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    driver_centered = driver_effects - driver_effects.mean()
    dy_centered = dy_effects - dy_effects.mean()
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

    driver_map = dict(zip(driver_names, driver_centered))
    dy_map = dict(zip(dy_names, dy_centered))
    driver_season_df = _build_driver_season_df(df, driver_map, dy_map)

    drv_pred = (X_driver_raw @ driver_centered) + (X_dy_raw @ dy_centered)
    car_pred = X_car_raw @ car_centered
    var_d = float(np.var(drv_pred))
    var_c = float(np.var(car_pred))
    total_explained = var_d + var_c
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    summary: dict[str, object] = {
        **extra_meta,
        "r2_score": 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "car_variance_share_pct": (var_c / total_explained) * 100.0 if total_explained > 0 else 0.0,
        "driver_variance_share_pct": (var_d / total_explained) * 100.0 if total_explained > 0 else 0.0,
        "driver_season_df": driver_season_df,
    }
    return driver_df, car_df, summary


def main() -> None:
    raw_df, _, _ = generate_synthetic_f1_data(n_races_per_season=20, seed=42)
    clean_df = preprocess_f1_dataframe(raw_df)
    res = fit_sklearn_cross_classified(clean_df) or fit_numpy_blup_cross_classified(clean_df)
    _, _, summary = res
    ds = summary["driver_season_df"]
    print("[신예 PIA 시즌별 실력 성장 추적 (driver + driver_year)]")
    print(ds[ds["driver"] == "PIA"].sort_values("season").to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
