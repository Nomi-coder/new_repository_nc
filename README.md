# MKU_main
Target repository

## Model training experiments

Run the parameter experiments with:

```bash
python -m model_training.train_experiments
```

The script uses the same Seaborn `tips` data and preprocessing pipeline as
`starter/baseline_model.py`. It compares linear regression with different
intercept and positivity settings, printing each experiment's settings,
learned coefficients, and test metrics.

The repository-level `.env` configures the MLflow tracking server for the
training scripts:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
```

`mlflow_config.py` loads that shared setting from `.env`. You can also export
the variable in your shell to override it for all Python processes launched
from that shell:

```bash
set -a
source .env
set +a
python -m model_training.train_experiments
```

The experiment names are configured separately in each training script:
`training-experiments` in `model_training/train_experiments.py` and
`baseline-model` in `starter/baseline_model.py`. Each script logs its own
metrics and report to that experiment. The tracking server must be running
and reachable from the process running the training script.

On pushes that change the training code or its dependencies, the
`Model Training Artifact` workflow selects the parameters with the lowest test
mean squared error, retrains that model, and uploads
`artifacts/best_model.joblib`. The artifact includes the fitted model, scaler,
feature columns, and selected parameters.
