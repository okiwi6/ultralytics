from datetime import timedelta
from ultralytics import YOLO
from pathlib import Path
import sys
import optuna

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator


def f1_from_validator(validator: BaseValidator):
    stats = validator.metrics.results_dict
    recall = stats['metrics/recall(B)']
    precision = stats['metrics/precision(B)']

    return 2 * (precision * recall) / (precision + recall)


def epoch_callback(trainer: BaseTrainer, trial: optuna.Trial):
    f1_score = f1_from_validator(trainer.validator)

    trial.report(f1_score, trainer.epoch)

    if trial.should_prune():
        raise optuna.TrialPruned()


def generate_hyperparameters(trial: optuna.Trial):
    return {
        "imgsz": trial.suggest_categorical("imgsz", [96]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.8),
        "lr0": (lr0 := trial.suggest_float("lr0", 0.0001, 0.02, log=True)),
        "lrf": trial.suggest_float("lrf", 0.0001, lr0, log=True),
        "momentum": trial.suggest_float("momentum", 0.8, 0.999),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.001),
    }


def main(trial: optuna.Trial):
    # Load a YOLOv8n model
    model = YOLO(Path('yolov8n-mobilenet.yaml').absolute())
    model.add_callback("on_fit_epoch_end",
                       lambda trainer: epoch_callback(trainer, trial))

    hyperparameters = generate_hyperparameters(trial)

    # Start tuning hyperparameters for YOLOv8n training on the dataset
    model.train(
        project=trial.study.study_name,
        name=f"trial-{trial.number}",
        data=str(Path('robot-detection.yaml').absolute()),
        imgsz=hyperparameters["imgsz"],
        batch=256,
        label_smoothing=hyperparameters["label_smoothing"],
        dropout=hyperparameters["dropout"],
        lr0=hyperparameters["lr0"],
        lrf=hyperparameters["lrf"],
        momentum=hyperparameters["momentum"],
        weight_decay=hyperparameters["weight_decay"],
        # device=[0, 1],
        epochs=200,
    )

    return f1_from_validator(model.trainer.validator)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        study_name = "yolo-tune-playground"
        storage = None
    else:
        study_name = sys.argv[1]
        storage = f"sqlite:///{sys.argv[2]}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )

    study.optimize(main, timeout=timedelta(weeks=4).total_seconds())
