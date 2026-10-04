"""Logs the already-trained Fashion-MNIST models (CNN baseline + transfer
learning backbones) into MLflow: hyperparameters, final test metrics,
per-epoch training curves and the .keras file as an artifact.

No training happens here - it only reads the JSON metrics/histories and the
.keras files already produced by the notebook and registers them as MLflow
runs, so the experiment becomes browsable in the MLflow UI.
"""

import json
import os
from pathlib import Path

import mlflow

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
EXPERIMENT_NAME = "fashion-mnist"

MODEL_FILES = {
    "CNN baseline": "cnn_baseline_fashion_mnist.keras",
    "MobileNetV2": "mobilenetv2_fashion_mnist.keras",
    "VGG16": "vgg16_fashion_mnist.keras",
    "VGG19": "vgg19_fashion_mnist.keras",
    "ResNet50": "resnet50_fashion_mnist.keras",
    "EfficientNetB0": "efficientnetb0_fashion_mnist.keras",
}

IMG_SIZE_BY_MODEL = {"CNN baseline": 28}
DEFAULT_IMG_SIZE = 96


def _load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _log_history(history: dict, step_offset: int = 0) -> int:
    """Log a {metric_name: [values]} dict as MLflow step metrics.

    Returns the number of steps logged, so callers can chain phases
    (e.g. frozen -> fine-tune) on a continuous step axis.
    """
    n_steps = 0
    for metric_name, values in history.items():
        for step, value in enumerate(values):
            mlflow.log_metric(metric_name, value, step=step_offset + step)
        n_steps = max(n_steps, len(values))
    return n_steps


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = _load_json(MODELS_DIR / "results.json") or {}
    cnn_history = _load_json(MODELS_DIR / "cnn_baseline_history.json")
    tl_histories = _load_json(MODELS_DIR / "tl_histories.json") or {}

    if not results:
        raise FileNotFoundError(f"No results.json found under {MODELS_DIR}")

    for model_name, metrics in results.items():
        model_file = MODELS_DIR / MODEL_FILES[model_name]
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag(
                "model_type",
                "baseline" if model_name == "CNN baseline" else "transfer_learning",
            )
            mlflow.log_param("img_size", IMG_SIZE_BY_MODEL.get(model_name, DEFAULT_IMG_SIZE))
            if metrics.get("params") is not None:
                mlflow.log_param("params", metrics["params"])

            mlflow.log_metric("test_accuracy", metrics["accuracy"])
            mlflow.log_metric("test_macro_f1", metrics["macro_f1"])
            if metrics.get("train_time_s") is not None:
                mlflow.log_metric("train_time_s", metrics["train_time_s"])

            if model_name == "CNN baseline" and cnn_history:
                _log_history(cnn_history)
            elif model_name in tl_histories:
                step_offset = 0
                for phase in ("frozen", "finetune"):
                    phase_history = tl_histories[model_name].get(phase, {})
                    if not phase_history:
                        continue
                    n_steps = _log_history(phase_history, step_offset=step_offset)
                    mlflow.set_tag(f"{phase}_epochs", n_steps)
                    step_offset += n_steps

            if model_file.exists():
                mlflow.log_artifact(str(model_file), artifact_path="model")
            else:
                print(f"WARNING: model file not found, skipping artifact: {model_file}")

            print(f"Logged '{model_name}' to MLflow experiment '{EXPERIMENT_NAME}'.")


if __name__ == "__main__":
    main()
