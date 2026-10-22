import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

REPORT_PATH = Path("artifacts/baseline_evaluation.json")
MLFLOW_EXPERIMENT_NAME = "baseline-model"
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def evaluate_regression_model(model, X, y_true, name="Model"):
    predictions = model.predict(X)
    mse = mean_squared_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    metrics = {
        "model_name": name,
        "test_mean_squared_error": round(mse, 4),
        "test_r_squared": round(r2, 4)
    }
    return metrics


def write_evaluation_report(
    model,
    metrics,
    feature_names,
    train_shape,
    test_shape,
    test_size,
    random_state,
    output_path=REPORT_PATH,
):
    report = {
        "dataset": {"name": "seaborn.tips", "target": "tip"},
        "split": {
            "test_size": float(test_size),
            "random_state": int(random_state),
            "train_shape": [int(size) for size in train_shape],
            "test_shape": [int(size) for size in test_shape],
        },
        "features": list(feature_names),
        "model": {
            "name": metrics["model_name"],
            "type": type(model).__name__,
            "intercept": float(model.intercept_),
            "standardized_coefficients": {
                name: float(coefficient)
                for name, coefficient in zip(feature_names, model.coef_)
            },
        },
        "metrics": {
            "test_mean_squared_error": metrics["test_mean_squared_error"],
            "test_r_squared": metrics["test_r_squared"],
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")
    return report


def main():
    random_state = 42
    test_size = 0.2
    np.random.seed(random_state)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if "MLFLOW_EXPERIMENT_ID" not in os.environ:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline-regression"):
        tips = sns.load_dataset("tips")
        df = pd.get_dummies(tips, drop_first=True)
        X = df.drop(columns=["tip"])
        y = df["tip"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        metadata = {
            "train_shape": X_train_scaled.shape,
            "test_shape": X_test_scaled.shape,
            "num_features": X_train_scaled.shape[1],
        }
        print(f"Pipeline Data Diagnostics: {metadata}")

        X_train_sm = sm.add_constant(X_train_scaled)
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        print(model_sm.summary())

        reg = LinearRegression()
        reg.fit(X_train_scaled, y_train)

        baseline_metrics = evaluate_regression_model(
            reg, X_test_scaled, y_test, name="Scikit-Learn OLS"
        )
        print("--- Standard Evaluation Output ---")
        for metric, value in baseline_metrics.items():
            print(f"{metric}: {value}")

        write_evaluation_report(
            reg,
            baseline_metrics,
            X.columns,
            X_train_scaled.shape,
            X_test_scaled.shape,
            test_size,
            random_state,
        )
        mlflow.log_params(
            {
                "test_size": test_size,
                "random_state": random_state,
                "num_features": metadata["num_features"],
                "train_rows": metadata["train_shape"][0],
                "test_rows": metadata["test_shape"][0],
            }
        )
        mlflow.log_metrics(
            {
                "test_mean_squared_error": baseline_metrics[
                    "test_mean_squared_error"
                ],
                "test_r_squared": baseline_metrics["test_r_squared"],
            }
        )
        mlflow.log_artifact(str(REPORT_PATH))
        print(f"Saved baseline evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
