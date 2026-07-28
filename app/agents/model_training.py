from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
try:
    from pycaret.classification import (
        compare_models as cls_compare,
        pull as cls_pull,
        setup as cls_setup,
        stack_models as cls_stack,
    )
    from pycaret.regression import (
        compare_models as reg_compare,
        pull as reg_pull,
        setup as reg_setup,
        stack_models as reg_stack,
    )
except Exception:
    cls_compare = None
    cls_pull = None
    cls_setup = None
    cls_stack = None
    reg_compare = None
    reg_pull = None
    reg_setup = None
    reg_stack = None
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None

from app.agents.base import BaseAgent
from app.graph.state import AgentState
from app.services.distributed_orchestrator import get_orchestration_status


def _pycaret_ready() -> bool:
    return all([
        cls_setup is not None,
        cls_compare is not None,
        cls_pull is not None,
        cls_stack is not None,
        reg_setup is not None,
        reg_compare is not None,
        reg_pull is not None,
        reg_stack is not None,
    ])


def _build_preprocessor(df: pd.DataFrame, target: str) -> Tuple[ColumnTransformer, list[str], list[str]]:
    X = df.drop(columns=[target])
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre, num_cols, cat_cols


class ModelTrainingAgent(BaseAgent):
    name = "model_training"

    def _classification_models(self) -> Dict[str, Any]:
        models = {
            "logistic_regression": LogisticRegression(max_iter=500),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "extra_trees": ExtraTreesClassifier(n_estimators=250, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "adaboost": AdaBoostClassifier(random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=7),
            "svc_rbf": SVC(kernel="rbf", probability=True, random_state=42),
            "gaussian_nb": GaussianNB(),
        }
        if XGBClassifier is not None:
            models["xgboost"] = XGBClassifier(eval_metric="mlogloss", random_state=42)
        if LGBMClassifier is not None:
            models["lightgbm"] = LGBMClassifier(random_state=42)
        if CatBoostClassifier is not None:
            models["catboost"] = CatBoostClassifier(verbose=0, random_state=42)
        return models

    def _regression_models(self) -> Dict[str, Any]:
        models = {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(n_estimators=200, random_state=42),
            "extra_trees_regressor": ExtraTreesRegressor(n_estimators=250, random_state=42),
            "gradient_boosting_regressor": GradientBoostingRegressor(random_state=42),
            "adaboost_regressor": AdaBoostRegressor(random_state=42),
            "decision_tree_regressor": DecisionTreeRegressor(random_state=42),
            "knn_regressor": KNeighborsRegressor(n_neighbors=7),
            "ridge": Ridge(random_state=42),
            "lasso": Lasso(random_state=42),
            "elasticnet": ElasticNet(random_state=42),
            "svr_rbf": SVR(kernel="rbf"),
        }
        if XGBRegressor is not None:
            models["xgboost_regressor"] = XGBRegressor(random_state=42)
        if LGBMRegressor is not None:
            models["lightgbm_regressor"] = LGBMRegressor(random_state=42)
        if CatBoostRegressor is not None:
            models["catboost_regressor"] = CatBoostRegressor(verbose=0, random_state=42)
        return models

    def _run_pycaret(self, df: pd.DataFrame, problem: str, target: str):
        setup_params = {
            "data": df,
            "target": target,
            "session_id": 42,
            "preprocess": True,
            "normalize": True,
            "transformation": True,
            "polynomial_features": True,
            "remove_outliers": True,
            "remove_multicollinearity": True,
            "multicollinearity_threshold": 0.9,
            "html": False,
            "verbose": False,
        }

        if problem == "classification":
            cls_setup(**setup_params)
            best_models = cls_compare(n_select=10, exclude=["dummy"], errors="ignore")
            leaderboard_df = cls_pull().copy()
            champion = best_models[0] if isinstance(best_models, list) else best_models
            stacked = (
                cls_stack(best_models[:3])
                if isinstance(best_models, list) and len(best_models) > 1
                else champion
            )
        else:
            reg_setup(**setup_params)
            best_models = reg_compare(n_select=10, exclude=["dummy"], errors="ignore")
            leaderboard_df = reg_pull().copy()
            champion = best_models[0] if isinstance(best_models, list) else best_models
            stacked = (
                reg_stack(best_models[:3])
                if isinstance(best_models, list) and len(best_models) > 1
                else champion
            )

        if not isinstance(best_models, list):
            best_models = [best_models]

        leaderboard_rows: List[Dict[str, Any]] = []
        for _, row in leaderboard_df.head(10).iterrows():
            model_name = str(row.iloc[0]) if len(row) > 0 else "unknown"
            row_dict = {str(k): row[k] for k in row.index}
            primary_metric = float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else 0.0
            secondary_metric = float(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else 0.0
            metrics = {}
            for k, v in row_dict.items():
                if k.lower() in {"model", "tt (sec)"}:
                    continue
                if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                    metrics[k] = float(v)
            leaderboard_rows.append(
                {
                    "model": model_name,
                    "primary_metric": primary_metric,
                    "secondary_metric": secondary_metric,
                    "metric_name": "pycaret_metric",
                    "metrics": metrics,
                }
            )

        best_name = str(leaderboard_rows[0]["model"]) if leaderboard_rows else str(champion)
        best_score = float(leaderboard_rows[0]["primary_metric"]) if leaderboard_rows else 0.0
        return stacked, champion, leaderboard_rows, best_name, best_score, best_models

    def _run_sklearn_fallback(self, df: pd.DataFrame, problem: str, target: str, candidate_models: List[str]):
        pre, num_cols, cat_cols = _build_preprocessor(df, target)
        X = df.drop(columns=[target])
        y = df[target]

        n_classes = int(y.nunique(dropna=False)) if problem == "classification" else 0
        min_class_count = int(y.value_counts(dropna=False).min()) if problem == "classification" and n_classes > 0 else 0
        can_stratify = problem == "classification" and n_classes > 1 and min_class_count >= 2
        stratify = y if can_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        model_pool = self._classification_models() if problem == "classification" else self._regression_models()
        if candidate_models:
            model_pool = {k: v for k, v in model_pool.items() if k in candidate_models} or model_pool

        leaderboard: List[Dict[str, Any]] = []
        best_name = ""
        best_pipe = None
        best_score = -np.inf

        failed_models = []
        for name, model in model_pool.items():
            try:
                pipe = Pipeline([("pre", pre), ("model", model)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)

                if problem == "classification":
                    acc = float(accuracy_score(y_test, preds))
                    f1w = float(f1_score(y_test, preds, average="weighted", zero_division=0))
                    prec = float(precision_score(y_test, preds, average="weighted", zero_division=0))
                    rec = float(recall_score(y_test, preds, average="weighted", zero_division=0))
                    auc = np.nan
                    if hasattr(pipe, "predict_proba"):
                        try:
                            prob = pipe.predict_proba(X_test)
                            auc = float(roc_auc_score(y_test, prob, multi_class="ovr", average="weighted"))
                        except Exception:
                            auc = np.nan
                    metric = auc if np.isfinite(auc) else acc
                    row = {
                        "model": name,
                        "primary_metric": float(metric),
                        "secondary_metric": f1w,
                        "metric_name": "auc_ovr" if np.isfinite(auc) else "accuracy",
                        "metrics": {
                            "accuracy": acc,
                            "f1_weighted": f1w,
                            "precision_weighted": prec,
                            "recall_weighted": rec,
                            "auc_ovr": float(auc) if np.isfinite(auc) else None,
                        },
                    }
                else:
                    r2 = float(r2_score(y_test, preds))
                    mae = float(mean_absolute_error(y_test, preds))
                    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                    mape = float(np.mean(np.abs((y_test - preds) / np.where(np.asarray(y_test) == 0, 1, np.asarray(y_test)))) * 100)
                    metric = r2
                    row = {
                        "model": name,
                        "primary_metric": metric,
                        "secondary_metric": mae,
                        "metric_name": "r2",
                        "metrics": {
                            "r2": r2,
                            "mae": mae,
                            "rmse": rmse,
                            "mape_pct": mape,
                        },
                    }

                leaderboard.append(row)
                if metric > best_score:
                    best_score = metric
                    best_name = name
                    best_pipe = pipe
            except Exception as e:
                failed_models.append(f"{name}: {str(e)}")
                print(f"⚠️  Model {name} failed: {e}")
                continue

        if not leaderboard:
            error_details = f"Dataset shape: {df.shape}, Target: {target}, Problem: {problem}, Classes: {n_classes if problem == 'classification' else 'N/A'}. Failed models: {'; '.join(failed_models[:3]) if failed_models else 'All models failed silently'}"
            print(f"❌ Training failed: {error_details}")
            raise ValueError(f"No sklearn fallback models could be trained for this dataset/target. {error_details}")
        leaderboard = sorted(leaderboard, key=lambda x: x["primary_metric"], reverse=True)
        return best_pipe, best_pipe, leaderboard[:10], best_name, float(best_score), list(model_pool.keys()), num_cols, cat_cols

    def _run_time_series_fallback(self, df: pd.DataFrame, target: str):
        dt_cols = [c for c in df.columns if c != target and pd.api.types.is_datetime64_any_dtype(df[c])]
        if not dt_cols:
            for c in df.columns:
                if c == target:
                    continue
                try:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        df[c] = parsed
                        dt_cols.append(c)
                        break
                except Exception:
                    continue
        if not dt_cols:
            raise ValueError("time_series requested but no datetime column detected.")

        time_col = dt_cols[0]
        work = df.sort_values(time_col).copy()
        work["__ts_ordinal__"] = pd.to_datetime(work[time_col], errors="coerce").map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

        # Basic lags to approximate forecasting behavior while reusing tabular models.
        if target in work.columns:
            work["__lag1__"] = work[target].shift(1)
            work["__lag2__"] = work[target].shift(2)
        work = work.dropna(subset=[target]).dropna().copy()
        if len(work) < 30:
            raise ValueError("Not enough rows for time-series fallback after lag feature generation.")

        cutoff = int(len(work) * 0.8)
        train_df = work.iloc[:cutoff].copy()
        test_df = work.iloc[cutoff:].copy()

        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]

        candidates = {
            "random_forest_regressor": RandomForestRegressor(n_estimators=250, random_state=42),
            "extra_trees_regressor": ExtraTreesRegressor(n_estimators=300, random_state=42),
            "ridge": Ridge(random_state=42),
            "mlp_regressor_ts": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
        }
        if XGBRegressor is not None:
            candidates["xgboost_regressor"] = XGBRegressor(random_state=42)
        if LGBMRegressor is not None:
            candidates["lightgbm_regressor"] = LGBMRegressor(random_state=42)

        leaderboard: List[Dict[str, Any]] = []
        best_name = ""
        best_model = None
        best_score = -np.inf

        for name, model in candidates.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                r2 = float(r2_score(y_test, pred))
                mae = float(mean_absolute_error(y_test, pred))
                rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                row = {
                    "model": name,
                    "primary_metric": r2,
                    "secondary_metric": mae,
                    "metric_name": "r2",
                    "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
                }
                leaderboard.append(row)
                if r2 > best_score:
                    best_score = r2
                    best_name = name
                    best_model = model
            except Exception:
                continue

        if not leaderboard or best_model is None:
            raise ValueError("No time-series fallback models could be trained.")

        leaderboard = sorted(leaderboard, key=lambda x: x["primary_metric"], reverse=True)[:10]
        return best_model, best_model, leaderboard, best_name, float(best_score), list(candidates.keys()), [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])], [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]

    def run(self, state: AgentState) -> AgentState:
        df = state["dataframe"].copy()
        target = state.get("target_column")
        problem = state.get("problem_type", "classification")
        candidate_models = (
            state.get("pipeline", {}).get("candidate_models_top10", [])
            or state.get("pipeline", {}).get("candidate_models_top5", [])
        )

        if not target or target not in df.columns:
            raise ValueError("Target column missing or invalid.")

        y = df[target]
        if problem not in {"classification", "regression", "time_series"}:
            inferred = "regression" if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=False) > 15 else "classification"
            state.setdefault("warnings", []).append(
                f"Problem type '{problem}' is not directly supported in structured trainer; auto-switched to {inferred}."
            )
            problem = inferred
        # Auto-correct obvious intent mismatch for random numeric targets.
        if problem == "classification":
            n_classes = int(y.nunique(dropna=False))
            if n_classes < 2:
                raise ValueError("Classification target must contain at least 2 classes.")
            if pd.api.types.is_numeric_dtype(y) and n_classes > 15:
                problem = "regression"
                state.setdefault("warnings", []).append(
                    "Auto-switched problem_type to regression (numeric target has high cardinality)."
                )

        num_cols = [c for c in df.drop(columns=[target]).columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.drop(columns=[target]).columns if c not in num_cols]
        backend = "sklearn"

        if problem == "time_series":
            final_model, primary_model, leaderboard, best_name, best_score, best_models, num_cols, cat_cols = self._run_time_series_fallback(
                df=df,
                target=target,
            )
            backend = "sklearn_ts_fallback"
        elif _pycaret_ready():
            try:
                final_model, primary_model, leaderboard, best_name, best_score, best_models = self._run_pycaret(
                    df=df,
                    problem=problem,
                    target=target,
                )
                backend = "pycaret"
            except Exception:
                final_model, primary_model, leaderboard, best_name, best_score, best_models, num_cols, cat_cols = self._run_sklearn_fallback(
                    df=df,
                    problem=problem,
                    target=target,
                    candidate_models=candidate_models,
                )
        else:
            final_model, primary_model, leaderboard, best_name, best_score, best_models, num_cols, cat_cols = self._run_sklearn_fallback(
                df=df,
                problem=problem,
                target=target,
                candidate_models=candidate_models,
            )

        state["training"] = {
            "target": target,
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "leaderboard": leaderboard,
            "best_model_name": best_name,
            "best_score": best_score,
            "model_object": final_model,
            "primary_model_object": primary_model,
            "best_models": [str(m) for m in best_models],
            "automl_backend": backend,
            "orchestration": get_orchestration_status(),
        }
        return state
