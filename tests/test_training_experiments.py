import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import SpanType

from model_training.train_experiments import run_experiments, train_best_model


def test_run_experiments_tracks_parameters_and_metrics(capsys, tmp_path):
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 3.0, 5.0, 7.0])
    X_test = np.array([[4.0], [5.0]])
    y_test = np.array([9.0, 11.0])

    previous_tracking_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        mlflow.set_experiment("training-experiment-tests")
        traced_run_experiments = mlflow.trace(
            run_experiments,
            name="run_experiments",
            span_type=SpanType.CHAIN,
        )
        results = traced_run_experiments(X_train, X_test, y_train, y_test)
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

    train_best_model(X, y, results, artifact_path)

    artifact = joblib.load(artifact_path)
    assert artifact["parameters"] == {
        "fit_intercept": False,
        "positive": False,
    }
    assert artifact["feature_columns"] == ["feature"]
    assert artifact["model"].n_features_in_ == 1
