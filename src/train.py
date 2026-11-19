"""
GeoChange-AI — Training Pipeline

TensorFlow / Keras 3

Supports:
    BCE + Dice loss
    Binary Cross-Entropy loss

Uses:
    TRAIN split for training.
    VAL split for validation and threshold selection.

IMPORTANT:
    TEST is never loaded or evaluated by this script.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

import tensorflow as tf
import keras

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.data import load_data, make_dataset
from src.model import build_siamese_unet
from src.losses import get_loss

from src.metrics import (
    ChangeF1,
    ChangeIoU,
    sweep_thresholds,
)

from src.postprocess import clean_mask


ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 1. TIME BUDGET CALLBACK
# ============================================================

class TimeBudget(keras.callbacks.Callback):

    def __init__(self, minutes):

        super().__init__()

        self.seconds = float(minutes) * 60

        self.started = None

    def on_train_begin(self, logs=None):

        self.started = time.monotonic()

    def on_train_batch_end(self, batch, logs=None):

        elapsed = time.monotonic() - self.started

        if elapsed >= self.seconds:

            print(
                "\nTIME BUDGET REACHED. "
                "Stopping training after the current batch."
            )

            self.model.stop_training = True


# ============================================================
# 2. EPOCH REPORT
# ============================================================

class EpochReport(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):

        self.epoch_start = time.monotonic()

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        seconds = (
            time.monotonic() - self.epoch_start
        )

        learning_rate = float(
            keras.ops.convert_to_numpy(
                self.model.optimizer.learning_rate
            )
        )

        print("\n" + "=" * 80)

        print(
            f"EPOCH {epoch + 1} COMPLETE"
        )

        print(
            f"Time: {seconds:.2f} seconds"
        )

        print(
            f"Train Loss: {logs.get('loss', np.nan):.5f}"
        )

        print(
            f"Val Loss: {logs.get('val_loss', np.nan):.5f}"
        )

        print(
            f"Val F1: {logs.get('val_f1', np.nan):.5f}"
        )

        print(
            f"Val IoU: {logs.get('val_iou', np.nan):.5f}"
        )

        print(
            f"Val Precision: {logs.get('val_precision', np.nan):.5f}"
        )

        print(
            f"Val Recall: {logs.get('val_recall', np.nan):.5f}"
        )

        print(
            f"Learning Rate: {learning_rate:.8f}"
        )

        print("=" * 80)


# ============================================================
# 3. CONFIGURATION
# ============================================================

def load_config(path):

    path = Path(path)

    if not path.is_absolute():

        path = ROOT / path

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)

    return config, path


# ============================================================
# 4. GPU CHECK
# ============================================================

def gpu_check(allow_cpu=False):

    gpus = tf.config.list_physical_devices("GPU")

    print("\n" + "=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)

    print("TensorFlow:", tf.__version__)

    print("Keras:", keras.__version__)

    print("GPUs:", gpus)

    if not gpus and not allow_cpu:

        raise SystemExit(
            "No GPU detected. Training aborted."
        )

    if gpus:

        for gpu in gpus:

            print("GPU device:", gpu.name)

    else:

        print("Running on CPU.")

    return gpus


# ============================================================
# 5. RECURSIVE LAYER TRAVERSAL
# ============================================================

def get_all_layers_recursive(model):

    """
    Collect layers from the model and nested models.

    A visited set prevents duplicate counting.
    """

    visited = set()

    def visit(layer):

        if id(layer) in visited:

            return

        visited.add(id(layer))

        yield layer

        for child in getattr(layer, "layers", []):

            yield from visit(child)

    yield from visit(model)


# ============================================================
# 6. APPLY BATCH NORMALIZATION MOMENTUM
# ============================================================

def apply_bn_momentum(model, momentum):

    momentum = float(momentum)

    if not 0.0 < momentum < 1.0:

        raise ValueError(
            "BN momentum must be between 0 and 1."
        )

    all_layers = list(
        get_all_layers_recursive(model)
    )

    bn_layers = [

        layer

        for layer in all_layers

        if isinstance(
            layer,
            keras.layers.BatchNormalization
        )
    ]

    for layer in bn_layers:

        layer.momentum = momentum

    actual_values = sorted({

        float(layer.momentum)

        for layer in bn_layers

    })

    print("\n" + "=" * 60)
    print("BATCH NORMALIZATION CONFIGURATION")
    print("=" * 60)

    print("Configured momentum:", momentum)

    print("Total BN layers:", len(bn_layers))

    print("Updated BN layers:", len(bn_layers))

    print("Actual momentum values:", actual_values)

    if not bn_layers:

        raise RuntimeError(
            "No BatchNormalization layers were found."
        )

    if len(actual_values) != 1 or not np.isclose(
        actual_values[0],
        momentum
    ):

        raise RuntimeError(
            "BN momentum verification failed."
        )

    print("BN CONFIGURATION: VERIFIED")

    return len(bn_layers)


# ============================================================
# 7. BUILD AND COMPILE MODEL
# ============================================================

def make_model(cfg):

    print("\n" + "=" * 60)
    print("BUILDING FRESH SIAMESE U-NET")
    print("=" * 60)

    model = build_siamese_unet(

        freeze_encoder=bool(
            cfg.get("freeze_encoder", False)
        ),

        decoder_filters=cfg.get(
            "decoder_filters",
            [256, 128, 64, 32]
        ),

        use_mixed_precision=False,

        verbose=True,

        encoder_weights="imagenet"
    )

    bn_count = apply_bn_momentum(

        model,

        cfg.get("bn_momentum", 0.99)
    )

    training_cfg = cfg.get(
        "training",
        {}
    )

    learning_rate = float(
        training_cfg.get(
            "learning_rate",
            0.0003
        )
    )

    loss_name = training_cfg.get(
        "loss",
        "bce_dice"
    )

    if loss_name not in ("bce", "bce_dice"):

        raise ValueError(
            "Loss must be 'bce' or 'bce_dice'."
        )

    model.compile(

        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),

        loss=get_loss(loss_name),

        metrics=[

            ChangeF1(
                threshold=0.5,
                name="f1"
            ),

            ChangeIoU(
                threshold=0.5,
                name="iou"
            ),

            keras.metrics.Precision(
                thresholds=0.5,
                name="precision"
            ),

            keras.metrics.Recall(
                thresholds=0.5,
                name="recall"
            )
        ]
    )

    print("\nMODEL COMPILED")

    print("Loss:", loss_name)

    if loss_name == "bce":

        print("Loss description: Binary Cross-Entropy only.")

    else:

        print("Loss description: 0.5 BCE + 0.5 Dice.")

    print("Learning rate:", learning_rate)

    print("BN layers:", bn_count)

    print("Mixed precision: DISABLED")

    return model, bn_count


# ============================================================
# 8. POST-PROCESSING VALIDATION
# ============================================================

def postprocess_validation(
    model,
    val_ds,
    threshold,
    min_area=20
):

    empty_total = 0

    empty_alarm = 0

    regions_total = 0

    crops_total = 0

    for (image_a, image_b), masks in val_ds:

        probabilities = model(

            (image_a, image_b),

            training=False

        ).numpy()

        ground_truth = masks.numpy()

        for probability, gt in zip(
            probabilities,
            ground_truth
        ):

            cleaned, regions, _ = clean_mask(

                probability[..., 0],

                threshold=threshold,

                min_area=min_area
            )

            regions_total += regions

            crops_total += 1

            gt_empty = not np.any(
                gt[..., 0] > 0.5
            )

            if gt_empty:

                empty_total += 1

                if np.any(cleaned):

                    empty_alarm += 1

    far = (

        empty_alarm / empty_total

        if empty_total > 0

        else 0.0
    )

    mean_regions = (

        regions_total / crops_total

        if crops_total > 0

        else 0.0
    )

    return far, mean_regions, empty_total


# ============================================================
# 9. BASELINE COMPARISON
# ============================================================

def compare_baselines(
    model,
    val_ds,
    sweep,
    run_dir
):

    baseline_path = (

        ROOT

        / "experiments"

        / "results"

        / "baselines_val.csv"
    )

    if not baseline_path.exists():

        raise FileNotFoundError(
            f"Baseline file not found: {baseline_path}"
        )

    baseline = pd.read_csv(
        baseline_path
    )

    baseline.columns = [

        str(column).strip().lower()

        for column in baseline.columns
    ]

    method_col = next(

        (

            column

            for column in (
                "method",
                "baseline",
                "name"
            )

            if column in baseline.columns

        ),

        None
    )

    if method_col is None:

        raise ValueError(
            "Baseline CSV needs a method/name column."
        )

    if method_col != "method":

        baseline = baseline.rename(
            columns={method_col: "method"}
        )

    aliases = {

        "empty_crop_far": "empty_far",

        "empty_crop_false_alarm_rate": "empty_far",

        "mean_regions_per_crop": "mean_regions"
    }

    baseline = baseline.rename(
        columns=aliases
    )

    # Find best threshold from validation sweep.

    best_row = sweep.loc[
        sweep["f1"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    # Postprocessing statistics.

    far, mean_regions, empty_count = (
        postprocess_validation(

            model,

            val_ds,

            threshold=best_threshold,

            min_area=20
        )
    )

    model_row = {

        "method": "Siamese U-Net",

        "threshold": best_threshold,

        "precision": float(best_row["precision"]),

        "recall": float(best_row["recall"]),

        "f1": float(best_row["f1"]),

        "iou": float(best_row["iou"]),

        "pixel_accuracy": float(
            best_row["pixel_accuracy"]
        ),

        "empty_far": far,

        "mean_regions": mean_regions
    }

    columns = [

        "method",

        "threshold",

        "precision",

        "recall",

        "f1",

        "iou",

        "pixel_accuracy",

        "empty_far",

        "mean_regions"
    ]

    for column in columns:

        if column not in baseline.columns:

            baseline[column] = np.nan

    comparison = pd.concat(

        [

            baseline[columns],

            pd.DataFrame(
                [model_row]
            )[columns]

        ],

        ignore_index=True
    )

    comparison.to_csv(

        run_dir / "validation_comparison.csv",

        index=False
    )

    print("\n" + "=" * 90)
    print("VALIDATION COMPARISON")
    print("=" * 90)

    print(
        "Best threshold:",
        best_threshold
    )

    print(
        "Empty validation crops:",
        empty_count
    )

    print(
        "Postprocessing min_area:",
        20
    )

    print(
        "\nIMPORTANT: F1 and IoU are raw pixel metrics. "
        "Empty FAR and region counts use postprocessed masks."
    )

    print(
        comparison.to_string(

            index=False,

            float_format=lambda value: f"{value:.4f}"

        )
    )

    return model_row, comparison


# ============================================================
# 10. LEARNING CURVES
# ============================================================

def plot_history(history, output_path):

    metrics = [

        ("loss", "val_loss", "Loss"),

        ("f1", "val_f1", "F1"),

        ("iou", "val_iou", "IoU"),

        ("precision", "val_precision", "Precision"),

        ("recall", "val_recall", "Recall")
    ]

    fig, axes = plt.subplots(

        2,

        3,

        figsize=(16, 9)
    )

    axes = axes.ravel()

    for ax, (
        train_key,
        val_key,
        title
    ) in zip(axes, metrics):

        if train_key in history:

            ax.plot(
                history[train_key],
                label="Training"
            )

        if val_key in history:

            ax.plot(
                history[val_key],
                label="Validation"
            )

        ax.set_title(title)

        ax.set_xlabel("Epoch")

        ax.set_ylabel(title)

        ax.grid(True, alpha=0.3)

        ax.legend()

    axes[-1].axis("off")

    fig.suptitle(
        "GeoChange-AI — Training History",
        fontsize=15
    )

    fig.tight_layout()

    fig.savefig(

        output_path,

        dpi=150,

        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 11. FULL TRAINING
# ============================================================

def full_run(
    model,
    cfg,
    config_path,
    run_dir,
    gpus,
    bn_count
):

    print("\n" + "=" * 60)
    print("LOADING TRAIN AND VALIDATION DATA")
    print("=" * 60)

    # CRITICAL:
    # Explicitly load TRAIN and VAL only.
    # TEST is never loaded.

    load_data(

        config_path=str(config_path),

        splits=("train", "val")
    )

    print("TRAIN: LOADED")

    print("VAL: LOADED")

    print("TEST: NOT LOADED")

    batch_size = int(
        cfg.get("batch_size", 16)
    )

    seed = int(
        cfg.get("seed", 42)
    )

    training_cfg = cfg.get(
        "training",
        {}
    )

    max_epochs = int(
        training_cfg.get("max_epochs", 30)
    )

    time_budget = float(
        training_cfg.get(
            "time_budget_minutes",
            80
        )
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_ds = make_dataset(

        "train",

        batch_size=batch_size,

        training=True,

        seed=seed
    )

    val_ds = make_dataset(

        "val",

        batch_size=batch_size,

        training=False,

        seed=seed
    )

    print("\nBatch size:", batch_size)

    print("Maximum epochs:", max_epochs)

    print("Time budget:", time_budget, "minutes")

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    backup_dir = run_dir / "backup"

    backup_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    callbacks = [

        keras.callbacks.BackupAndRestore(

            backup_dir=str(backup_dir)
        ),

        keras.callbacks.ModelCheckpoint(

            filepath=str(
                run_dir / "best_model.keras"
            ),

            monitor="val_f1",

            mode="max",

            save_best_only=True,

            verbose=1
        ),

        keras.callbacks.CSVLogger(

            filename=str(
                run_dir / "history.csv"
            ),

            append=False
        ),

        keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=float(
                training_cfg.get(
                    "reduce_lr_factor",
                    0.5
                )
            ),

            patience=int(
                training_cfg.get(
                    "reduce_lr_patience",
                    3
                )
            ),

            verbose=1
        ),

        keras.callbacks.EarlyStopping(

            monitor="val_f1",

            mode="max",

            patience=int(
                training_cfg.get(
                    "early_stopping_patience",
                    8
                )
            ),

            restore_best_weights=True,

            verbose=1
        ),

        TimeBudget(time_budget),

        EpochReport()
    ]

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    print("TEST split access: DISABLED")

    start = time.perf_counter()

    history_object = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=max_epochs,

        callbacks=callbacks,

        verbose=1
    )

    total_seconds = (
        time.perf_counter() - start
    )

    history = history_object.history

    completed_epochs = len(
        history.get("loss", [])
    )

    seconds_per_epoch = (

        total_seconds / completed_epochs

        if completed_epochs > 0

        else None
    )

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print("Completed epochs:", completed_epochs)

    print(
        "Total time:",
        round(total_seconds, 2),
        "seconds"
    )

    if seconds_per_epoch is not None:

        print(
            "Average seconds per epoch:",
            round(seconds_per_epoch, 2)
        )

    print("BN layers:", bn_count)

    print(
        "GPU:",
        gpus[0].name if gpus else "CPU"
    )

    # --------------------------------------------------------
    # LEARNING CURVES
    # --------------------------------------------------------

    plot_history(

        history,

        run_dir / "learning_curves.png"
    )

    # --------------------------------------------------------
    # LOAD BEST CHECKPOINT
    # --------------------------------------------------------

    best_path = (
        run_dir / "best_model.keras"
    )

    if not best_path.exists():

        raise RuntimeError(
            "No best_model.keras checkpoint was created."
        )

    print("\nLoading best validation checkpoint...")

    best_model = keras.models.load_model(

        best_path,

        compile=False
    )

    # --------------------------------------------------------
    # VALIDATION THRESHOLD SWEEP
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION THRESHOLD SWEEP")
    print("=" * 60)

    sweep = sweep_thresholds(

        best_model,

        val_ds
    )

    sweep.to_csv(

        run_dir / "threshold_sweep_val.csv",

        index=False
    )

    print(
        sweep[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "iou",
                "pixel_accuracy"
            ]
        ].to_string(

            index=False,

            float_format=lambda value: f"{value:.4f}"
        )
    )

    # --------------------------------------------------------
    # METRICS AT THRESHOLD 0.5
    # --------------------------------------------------------

    index_05 = (
        sweep["threshold"] - 0.5
    ).abs().idxmin()

    metrics_05 = sweep.loc[
        index_05
    ].to_dict()

    print("\nMETRICS AT THRESHOLD 0.5")

    for key in (

        "precision",

        "recall",

        "f1",

        "iou",

        "pixel_accuracy"

    ):

        print(
            f"{key}: {float(metrics_05[key]):.5f}"
        )

    # --------------------------------------------------------
    # BASELINE COMPARISON
    # --------------------------------------------------------

    model_metrics, comparison = compare_baselines(

        best_model,

        val_ds,

        sweep,

        run_dir
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    best_threshold = float(
        model_metrics["threshold"]
    )

    gpu_name = (
        gpus[0].name if gpus else "CPU"
    )

    payload = {

        "run_name": run_dir.name,

        "epochs_completed": completed_epochs,

        "training_seconds": total_seconds,

        "seconds_per_epoch": seconds_per_epoch,

        "gpu": gpu_name,

        "bn_momentum": float(
            cfg.get("bn_momentum", 0.99)
        ),

        "bn_layers_updated": bn_count,

        "batch_size": batch_size,

        "learning_rate_initial": float(
            training_cfg.get(
                "learning_rate",
                0.0003
            )
        ),

        "loss": training_cfg.get(
            "loss",
            "bce_dice"
        ),

        "validation_at_0_5": {

            key: float(metrics_05[key])

            for key in (

                "precision",

                "recall",

                "f1",

                "iou",

                "pixel_accuracy"

            )
        },

        "best_validation_threshold": best_threshold,

        "best_validation_metrics": {

            key: float(model_metrics[key])

            for key in (

                "precision",

                "recall",

                "f1",

                "iou",

                "pixel_accuracy",

                "empty_far",

                "mean_regions"

            )
        },

        "postprocessing_min_area": 20,

        "test_split_accessed": False
    }

    with open(

        run_dir / "metrics.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(
            payload,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("SAVED RUN ARTIFACTS")
    print("=" * 60)

    print("Run directory:", run_dir)

    print("Best model:", best_path)

    print("History:", run_dir / "history.csv")

    print(
        "Learning curves:",
        run_dir / "learning_curves.png"
    )

    print(
        "Threshold sweep:",
        run_dir / "threshold_sweep_val.csv"
    )

    print(
        "Comparison:",
        run_dir / "validation_comparison.csv"
    )

    print(
        "Metrics:",
        run_dir / "metrics.json"
    )

    print("\nTEST SPLIT: NEVER LOADED OR EVALUATED.")


# ============================================================
# 12. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="GeoChange-AI training pipeline"
    )

    parser.add_argument(

        "--config",

        default="configs/default.yaml",

        help="Path to YAML configuration."
    )

    parser.add_argument(

        "--run_name",

        required=True,

        help="Unique name for this training run."
    )

    parser.add_argument(

        "--allow_cpu",

        action="store_true",

        help="Allow training without a GPU."
    )

    parser.add_argument(

        "--loss",

        choices=["bce", "bce_dice"],

        default=None,

        help=(
            "Override the loss in the YAML config. "
            "Options: bce or bce_dice."
        )
    )

    args = parser.parse_args()

    cfg, config_path = load_config(
        args.config
    )

    # --------------------------------------------------------
    # APPLY LOSS OVERRIDE
    # --------------------------------------------------------

    cfg.setdefault("training", {})

    if args.loss is not None:

        cfg["training"]["loss"] = args.loss

    selected_loss = cfg["training"].get(
        "loss",
        "bce_dice"
    )

    if selected_loss not in ("bce", "bce_dice"):

        raise SystemExit(
            "Invalid configured loss. Use bce or bce_dice."
        )

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    if selected_loss == "bce":

        print("Selected loss: BINARY CROSS-ENTROPY ONLY")

    else:

        print("Selected loss: BCE + DICE")

    print(
        "BN momentum:",
        cfg.get("bn_momentum", 0.99)
    )

    print(
        "Batch size:",
        cfg.get("batch_size", 16)
    )

    print(
        "Maximum epochs:",
        cfg.get("training", {}).get("max_epochs", 30)
    )

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    gpus = gpu_check(
        args.allow_cpu
    )

    keras.utils.set_random_seed(
        int(cfg.get("seed", 42))
    )

    # --------------------------------------------------------
    # RUN DIRECTORY
    # --------------------------------------------------------

    run_dir = (

        ROOT

        / "experiments"

        / "runs"

        / args.run_name
    )

    if run_dir.exists() and any(
        run_dir.iterdir()
    ):

        raise SystemExit(

            f"Run directory already contains files: {run_dir}\n"

            "Choose a new --run_name."
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the actual configuration, including loss override.

    with open(

        run_dir / "config.yaml",

        "w",

        encoding="utf-8"

    ) as file:

        yaml.safe_dump(
            cfg,
            file,
            sort_keys=False
        )

    print("\nTEST split access: DISABLED")

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model, bn_count = make_model(cfg)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    full_run(

        model,

        cfg,

        config_path,

        run_dir,

        gpus,

        bn_count
    )


if __name__ == "__main__":

    main()