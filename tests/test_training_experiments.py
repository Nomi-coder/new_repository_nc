import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from model_training import train_experiments
from model_training.train_experiments import run_experiments, train_best_model


def test_main_logs_metrics_and_json_and_model_artifacts(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    total_bill = rng.uniform(10.0, 40.0, 30)
    size = rng.integers(1, 7, 30)
    tips = pd.DataFrame(
        {
            "total_bill": total_bill,
            "tip": (
                0.2 * total_bill
                + 0.3 * size
                + rng.normal(0.0, 0.2, 30)
            ),
            "sex": rng.choice(["Female", "Male"], 30),
            "smoker": rng.choice(["No", "Yes"], 30),
            "day": rng.choice(["Thur", "Fri", "Sat"], 30),
            "time": rng.choice(["Lunch", "Dinner"], 30),
            "size": size,
        }
    )
    monkeypatch.setattr(train_experiments.sns, "load_dataset", lambda _: tips)
    monkeypatch.setattr(
        train_experiments, "ARTIFACT_PATH", tmp_path / "model" / "best_model.joblib"
    )
    results_path = tmp_path / "reports" / "experiment_results.json"
    monkeypatch.setattr(train_experiments, "RESULTS_PATH", results_path)
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)

    previous_tracking_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_experiment("training-experiments")
        experiment_id = mlflow.get_experiment_by_name(
            "training-experiments"
        ).experiment_id

        train_experiments.main()
        mlflow.flush_trace_async_logging()

        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        assert len(runs) == 1
        run = runs.iloc[0]
        artifacts = mlflow.MlflowClient().list_artifacts(run["run_id"])
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)

    report = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(report["experiments"]) == 3
    assert np.isclose(
        run["metrics.best_test_mean_squared_error"],
        report["best_experiment"]["test_mean_squared_error"],
    )
    assert run["params.experiment_1_fit_intercept"] == "True"
    assert {artifact.path for artifact in artifacts} >= {
        "experiment_results.json",
        "best_model.joblib",
    }


def test_run_experiments_tracks_parameters_and_metrics(capsys, tmp_path):
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 3.0, 5.0, 7.0])
    X_test = np.array([[4.0], [5.0]])
    y_test = np.array([9.0, 11.0])

    previous_tracking_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        mlflow.set_experiment("training-experiment-tests")
        results = run_experiments(X_train, X_test, y_train, y_test)
        trace_id = mlflow.get_last_active_trace_id()
        mlflow.flush_trace_async_logging()
        trace = mlflow.get_trace(trace_id) if trace_id else None
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)

    output = capsys.readouterr().out

    assert len(results) == 3
    assert results[0]["parameters"] == {
        "fit_intercept": True,
        "positive": False,
    }
    assert results[0]["test_mean_squared_error"] == 0.0
    assert "parameters:" in output
    assert "coefficients:" in output
    assert "test_r_squared:" in output
    assert trace is not None
    experiment_spans = [
        span for span in trace.data.spans if span.name == "run_experiments"
    ]
    assert experiment_spans
    assert experiment_spans[0].inputs is not None
    assert experiment_spans[0].outputs is not None


def test_train_best_model_saves_the_lowest_mse_model(tmp_path):
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = np.array([3.0, 5.0, 7.0])
    results = [
        {
            "test_mean_squared_error": 2.0,
            "parameters": {"fit_intercept": True, "positive": False},
        },
        {
            "test_mean_squared_error": 1.0,
            "parameters": {"fit_intercept": False, "positive": False},
        },
    ]
    artifact_path = tmp_path / "model" / "best_model.joblib"

    previous_tracking_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        mlflow.set_experiment("best-model-training-tests")
        train_best_model(X, y, results, artifact_path)
        trace_id = mlflow.get_last_active_trace_id()
        mlflow.flush_trace_async_logging()
        trace = mlflow.get_trace(trace_id) if trace_id else None
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)

    saved_artifact = joblib.load(artifact_path)
    assert saved_artifact["parameters"] == {
        "fit_intercept": False,
        "positive": False,
    }
    assert saved_artifact["feature_columns"] == ["feature"]
    assert saved_artifact["model"].n_features_in_ == 1
    assert trace is not None
    assert any(span.name == "train_best_model" for span in trace.data.spans)
