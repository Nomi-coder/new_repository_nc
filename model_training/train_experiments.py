import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from mlflow.entities import SpanType
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Naomi's second change

EXPERIMENTS = (
    {"fit_intercept": True, "positive": False},
    {"fit_intercept": False, "positive": False},
    {"fit_intercept": True, "positive": True},
)
ARTIFACT_PATH = Path("artifacts/best_model.joblib")
RESULTS_PATH = Path("artifacts/experiment_results.json")
MLFLOW_EXPERIMENT_NAME = "training-experiments"
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@mlflow.trace(name="run_experiments", span_type=SpanType.CHAIN)
def run_experiments(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    experiments: Sequence[Mapping[str, bool]] = EXPERIMENTS,
) -> list[dict[str, object]]:
    results = []

    for experiment_number, parameters in enumerate(experiments, start=1):
        model = LinearRegression(**parameters)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = {
            "model_name": f"Experiment {experiment_number}",
            "test_mean_squared_error": round(
                mean_squared_error(y_test, predictions), 4
            ),
            "test_r_squared": round(r2_score(y_test, predictions), 4),
        }
        result = {
            "experiment": experiment_number,
            "parameters": dict(parameters),
            "intercept": float(model.intercept_),
            "coefficients": model.coef_.tolist(),
            **metrics,
        }
        results.append(result)

        print(f"\n--- Experiment {experiment_number} ---")
        print(f"parameters: {json.dumps(result['parameters'], sort_keys=True)}")
        print(f"intercept: {result['intercept']:.6f}")
        print(f"coefficients: {json.dumps(result['coefficients'])}")
        print(f"test_mean_squared_error: {metrics['test_mean_squared_error']}")
        print(f"test_r_squared: {metrics['test_r_squared']}")

    return results


@mlflow.trace(name="train_best_model", span_type=SpanType.CHAIN)
def train_best_model(
    X: pd.DataFrame,
    y: np.ndarray,
    experiment_results: Sequence[Mapping[str, object]],
    artifact_path: Path = ARTIFACT_PATH,
) -> dict[str, object]:
    best_experiment = min(
        experiment_results,
        key=lambda result: float(result["test_mean_squared_error"]),
    )
    parameters = best_experiment["parameters"]
    if not isinstance(parameters, Mapping):
        raise TypeError("Experiment parameters must be a mapping.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression(**parameters)
    model.fit(X_scaled, y)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "parameters": dict(parameters),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    print(
        "Selected parameters: "
        f"{json.dumps(artifact['parameters'], sort_keys=True)} "
        f"(test MSE: {best_experiment['test_mean_squared_error']})"
    )
    print(f"Saved model artifact to {artifact_path}")
    return artifact


@mlflow.trace(name="training_pipeline", span_type=SpanType.CHAIN)
def main() -> None:
    np.random.seed(42)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if "MLFLOW_EXPERIMENT_ID" not in os.environ:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="linear-regression-experiments"):
        tips = sns.load_dataset("tips")
        df = pd.get_dummies(tips, drop_first=True)
        X = df.drop(columns=["tip"])
        y = df["tip"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        metadata = {
            "train_shape": list(X_train_scaled.shape),
            "test_shape": list(X_test_scaled.shape),
            "num_features": X_train_scaled.shape[1],
        }
        print(f"Pipeline Data Diagnostics: {metadata}")
        mlflow.log_params(
            {
                "train_rows": metadata["train_shape"][0],
                "test_rows": metadata["test_shape"][0],
                "num_features": metadata["num_features"],
                "test_size": 0.2,
                "random_state": 42,
            }
        )

        X_train_sm = sm.add_constant(X_train_scaled)
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        ols_summary = str(model_sm.summary())
        print(ols_summary)

        results = run_experiments(X_train_scaled, X_test_scaled, y_train, y_test)
        best_experiment = min(
            results, key=lambda result: result["test_mean_squared_error"]
        )
        for result in results:
            experiment_number = result["experiment"]
            mlflow.log_params(
                {
                    f"experiment_{experiment_number}_{name}": value
                    for name, value in result["parameters"].items()
                }
            )
            mlflow.log_metrics(
                {
                    f"experiment_{experiment_number}_test_mean_squared_error": (
                        result["test_mean_squared_error"]
                    ),
                    f"experiment_{experiment_number}_test_r_squared": (
                        result["test_r_squared"]
                    ),
                }
            )

        mlflow.log_metrics(
            {
                "best_test_mean_squared_error": best_experiment[
                    "test_mean_squared_error"
                ],
                "best_test_r_squared": best_experiment["test_r_squared"],
            }
        )
        artifact = train_best_model(X, y, results, artifact_path=ARTIFACT_PATH)
        mlflow.log_params(
            {
                f"best_{name}": value
                for name, value in artifact["parameters"].items()
            }
        )

        report = {
            "data": metadata,
            "experiments": results,
            "best_experiment": best_experiment,
            "best_model_parameters": artifact["parameters"],
            "best_model_artifact": str(ARTIFACT_PATH),
            "ols_summary": ols_summary,
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(RESULTS_PATH))
        mlflow.log_artifact(str(ARTIFACT_PATH))
        print(f"Saved experiment results to {RESULTS_PATH}")

print(f"End of training experiments.")

if __name__ == "__main__":
    main()
