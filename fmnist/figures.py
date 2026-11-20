"""Static report figures (PNG) generated from the saved results. Interactive versions live in app.py."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config as C  # noqa: E402

ORDER = ["label_spreading", "logistic_regression", "random_forest"]
COLORS = {"label_spreading": "#2a78d6", "logistic_regression": "#eb6834", "random_forest": "#1baf7a"}
SHORT = {"label_spreading": "LabelSpreading", "logistic_regression": "Logistic Regression",
         "random_forest": "Random Forest"}
CLASS_SHORT = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style():
    sns.set_style("white")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight", "font.size": 10,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name)
    plt.close(fig)


def _grouped_bars(ax, labels, series, ylim=(0, 1), fmt="{:.3f}", label_values=True):
    n = len(series)
    w = 0.8 / n
    xs = np.arange(len(labels))
    for i, (key, vals) in enumerate(series.items()):
        bars = ax.bar(xs + (i - (n - 1) / 2) * w, vals, w * 0.92, color=COLORS[key], label=SHORT[key])
        if label_values:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, fmt.format(v), ha="center",
                        va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, loc="lower right")


def save_all_figures(results, train_images, train_labels):
    _style()
    ds, models = results["dataset"], results["models"]

    # 01 class distribution (train vs test)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    xs = np.arange(10)
    ax.bar(xs - 0.2, ds["train_class_counts"], 0.38, color="#2a78d6", label="Train (60,000)")
    ax.bar(xs + 0.2, ds["test_class_counts"], 0.38, color="#86b6ef", label="Test (10,000)")
    ax.set_xticks(xs); ax.set_xticklabels(C.CLASS_NAMES, rotation=25)
    ax.set_ylabel("Images"); ax.legend(frameon=False)
    ax.set_title("Class distribution of the official Fashion-MNIST split")
    _save(fig, "01_class_distribution.png")

    # 02 sample images
    fig, axes = plt.subplots(3, 10, figsize=(14, 5))
    for c in range(10):
        for r, i in enumerate(np.where(train_labels == c)[0][:3]):
            axes[r, c].imshow(train_images[i], cmap="gray_r")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"{c}: {C.CLASS_NAMES[c]}", fontsize=9)
    fig.suptitle("Sample Fashion-MNIST training images (3 per class)", fontweight="bold")
    _save(fig, "02_sample_images.png")

    # 03 label availability
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    vals = [ds["n_labelled"], ds["n_unlabelled"]]
    bars = ax.bar(["Labelled", "Unlabelled (-1)"], vals, color=["#2a78d6", "#b0aea6"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 600, f"{v:,}", ha="center", fontweight="bold", color=INK)
    ax.set_ylim(0, ds["n_train"]); ax.set_ylabel("Training images")
    ax.set_title(f"Label availability ({100*ds['label_fraction']:.0f}% labelled)")
    _save(fig, "03_label_availability.png")

    # 04 metric comparison
    metric_keys = [("accuracy", "Accuracy"), ("precision_macro", "Precision\n(macro)"),
                   ("recall_macro", "Recall\n(macro)"), ("f1_macro", "F1\n(macro)"),
                   ("precision_weighted", "Precision\n(weighted)"), ("recall_weighted", "Recall\n(weighted)"),
                   ("f1_weighted", "F1\n(weighted)")]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    _grouped_bars(ax, [m[1] for m in metric_keys],
                  {k: [models[k]["metrics"][m[0]] for m in metric_keys] for k in ORDER}, ylim=(0.5, 0.95))
    ax.set_ylabel("Score on 10,000 test images")
    ax.set_title("Test-set metrics: LabelSpreading vs. Logistic Regression vs. Random Forest")
    _save(fig, "04_metric_comparison.png")

    # 05 confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.6))
    for ax, key in zip(axes, ORDER):
        cm = np.array(models[key]["metrics"]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, square=True,
                    xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, annot_kws={"size": 8},
                    linewidths=0.5, linecolor="white")
        ax.set_title(f"{SHORT[key]}\naccuracy = {100*models[key]['metrics']['accuracy']:.2f}%")
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        ax.tick_params(axis="x", rotation=45); ax.grid(False)
    _save(fig, "05_confusion_matrices.png")

    # 06 per-class F1
    fig, ax = plt.subplots(figsize=(13, 4.6))
    _grouped_bars(ax, CLASS_SHORT,
                  {k: [r["f1-score"] for r in models[k]["metrics"]["per_class"]] for k in ORDER},
                  ylim=(0, 1.05), fmt="{:.2f}")
    ax.set_ylabel("F1-score (test)"); ax.set_title("Per-class F1-score")
    _save(fig, "06_per_class_f1.png")

    # 07 timing: training and inference on separate panels (different scales)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, field, title in [(axes[0], "train_time_total_s", "Training time (s)\nLS includes PCA fit"),
                             (axes[1], "inference_time_s", "Prediction time on 10,000 test images (s)\nLS includes PCA transform")]:
        vals = [models[k][field] for k in ORDER]
        bars = ax.barh([SHORT[k] for k in ORDER], vals, color=[COLORS[k] for k in ORDER], height=0.55)
        for b, v in zip(bars, vals):
            ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.2f}s", va="center", color=MUTED)
        ax.set_xlim(0, max(vals) * 1.25); ax.invert_yaxis(); ax.set_title(title)
        ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    _save(fig, "07_timing.png")

    # 08 label-percentage sweep (mean +/- std over seeds)
    sweep = results.get("label_fraction_sweep") or []
    if sweep:
        df = pd.DataFrame(sweep)
        fig, ax = plt.subplots(figsize=(9, 5))
        for key in ORDER:
            g = df.groupby("label_fraction")[f"{key}_accuracy"].agg(["mean", "std"]).reset_index()
            x = 100 * g["label_fraction"]
            ax.plot(x, g["mean"], marker="o", lw=2, ms=8, color=COLORS[key], label=SHORT[key])
            ax.fill_between(x, g["mean"] - g["std"], g["mean"] + g["std"], color=COLORS[key], alpha=0.15, lw=0)
        ax.set_xticks(100 * np.array(C.LABEL_FRACTIONS))
        ax.set_xlabel("Labelled training data (%)"); ax.set_ylabel("Test accuracy")
        ax.set_title(f"Accuracy vs. labelled-data percentage (mean ± std over seeds {C.ROBUSTNESS_SEEDS})")
        ax.legend(frameon=False, loc="lower right")
        _save(fig, "08_label_percentage.png")
