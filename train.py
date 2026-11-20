"""Full experiment: tune, train, evaluate and save LabelSpreading, Logistic Regression and Random Forest.

Usage:  .venv\\Scripts\\python.exe train.py [--skip-sweep]

Protocol (adapted from the Jury 2 notebook to the full official dataset):
  * official 60,000 training / 10,000 test images; test set used ONLY for final scoring
  * pixels flattened to 784 features and scaled to [0, 1]
  * 10% of training labels kept per class (seed 42), the rest set to -1
  * Logistic Regression and Random Forest: trained on the labelled subset only
  * LabelSpreading: k-NN graph on PCA-50 features (PCA fitted on training images, no labels),
    fitted on all 60,000 training images with the partial label vector
  * hyperparameters chosen by 5-fold stratified CV on the labelled training samples only
"""
import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import time
import warnings

from fmnist import config as C

C.configure_storage()   # temp, joblib memmaps, caches and logs -> project folder on D:, before heavy imports

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import sklearn  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from fmnist.data import load_split, make_partial_labels, to_features  # noqa: E402
from fmnist.figures import save_all_figures  # noqa: E402
from fmnist.models import (evaluate, make_lr, make_ls, make_rf, timed, top_confusions,  # noqa: E402
                           tune_label_spreading, tune_logistic_regression, tune_random_forest)

warnings.filterwarnings("ignore", category=FutureWarning)
T_START = time.perf_counter()
LOG_FILE = C.LOG_DIR / f"train_{dt.datetime.now():%Y%m%d_%H%M%S}.log"


def log(msg):
    line = f"[{time.perf_counter() - T_START:7.1f}s] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def ls_predict(ls, Z):
    """LabelSpreading predict; counts test images whose k neighbours carry no label mass (NaN proba)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return ls.predict(Z)


def run_sweep(Xtr, ytr, Ztr, Xte, Zte, yte, best):
    rows = []
    for seed in C.ROBUSTNESS_SEEDS:
        for frac in C.LABEL_FRACTIONS:
            y_part, lab = make_partial_labels(ytr, frac, random_state=seed)
            row = {"label_fraction": frac, "seed": seed, "n_labelled": int(len(lab)),
                   "n_unlabelled": int((y_part == -1).sum())}
            models = {
                "logistic_regression": (make_lr(best["logistic_regression"]["C"]), Xtr[lab], ytr[lab], Xte),
                "random_forest": (make_rf(**best["random_forest"]), Xtr[lab], ytr[lab], Xte),
                "label_spreading": (make_ls(**best["label_spreading"]), Ztr, y_part, Zte),
            }
            for key, (model, Xfit, yfit, Xeval) in models.items():
                _, t = timed(model.fit, Xfit, yfit)
                pred = ls_predict(model, Xeval) if key == "label_spreading" else model.predict(Xeval)
                row[f"{key}_accuracy"] = float(accuracy_score(yte, pred))
                row[f"{key}_f1_macro"] = float(f1_score(yte, pred, average="macro"))
                row[f"{key}_fit_s"] = round(t, 3)
            rows.append(row)
            log(f"  sweep seed={seed} {int(frac*100):>2}% labels: " + "  ".join(
                f"{k[:2].upper()}={row[f'{k}_accuracy']:.4f}" for k in models))
    return rows


def main(skip_sweep=False):
    for d in (C.MODEL_DIR, C.RESULTS_DIR, C.FIG_DIR):
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ data
    img_tr, ytr = load_split("train")
    img_te, yte = load_split("test")
    Xtr, Xte = to_features(img_tr), to_features(img_te)
    y_part, lab = make_partial_labels(ytr, C.MAIN_LABEL_FRACTION)
    unl = y_part == -1
    X_lab, y_lab = Xtr[lab], ytr[lab]
    assert set(np.unique(y_part[lab])) == set(range(10)) and np.all(y_part[lab] == ytr[lab])
    log(f"Data: train {Xtr.shape}, test {Xte.shape}; labelled {len(lab):,}, unlabelled {unl.sum():,}")

    # PCA for the LabelSpreading graph: fitted on training images only, labels unused
    pca = PCA(n_components=C.PCA_COMPONENTS, svd_solver="full", random_state=C.RANDOM_STATE)
    Ztr, pca_fit_t = timed(pca.fit_transform, Xtr)
    log(f"PCA 784->{C.PCA_COMPONENTS}: {100*pca.explained_variance_ratio_.sum():.1f}% variance, {pca_fit_t:.2f}s")

    # ------------------------------------------------------------------ tuning (training data only)
    log("Tuning Logistic Regression (5-fold CV on labelled samples)...")
    lr_grid, lr_best = tune_logistic_regression(X_lab, y_lab)
    log(f"  best {lr_best}")
    log("Tuning Random Forest (5-fold CV on labelled samples)...")
    rf_grid, rf_best = tune_random_forest(X_lab, y_lab)
    log(f"  best {rf_best}")
    log("Tuning LabelSpreading (5-fold CV over known labels, graph on all 60,000 training images)...")
    ls_grid, ls_best = tune_label_spreading(Ztr, y_part, lab)
    log(f"  best {ls_best}")
    best = {"logistic_regression": lr_best, "random_forest": rf_best, "label_spreading": ls_best}
    grids = {"logistic_regression": lr_grid, "random_forest": rf_grid, "label_spreading": ls_grid}

    # ------------------------------------------------------------------ final fits + test evaluation
    results_models, preds, probas = {}, {}, {}

    lr = make_lr(lr_best["C"])
    _, t_fit = timed(lr.fit, X_lab, y_lab)
    preds["logistic_regression"], t_pred = timed(lr.predict, Xte)
    probas["logistic_regression"] = lr.predict_proba(Xte)
    results_models["logistic_regression"] = dict(
        paradigm="supervised", features="784 pixels scaled to [0, 1]",
        training_samples_used=int(len(lab)), n_labelled=int(len(lab)), n_unlabelled_used=0,
        train_time_s=t_fit, train_time_total_s=t_fit, inference_time_s=t_pred,
        extra={"n_iter": int(lr.n_iter_[0])})
    joblib.dump(lr, C.MODEL_DIR / "logistic_regression.joblib", compress=3)
    log(f"LR fitted in {t_fit:.2f}s, test acc {accuracy_score(yte, preds['logistic_regression']):.4f}")

    rf = make_rf(**rf_best)
    _, t_fit = timed(rf.fit, X_lab, y_lab)
    preds["random_forest"], t_pred = timed(rf.predict, Xte)
    probas["random_forest"] = rf.predict_proba(Xte)
    results_models["random_forest"] = dict(
        paradigm="supervised", features="784 pixels scaled to [0, 1]",
        training_samples_used=int(len(lab)), n_labelled=int(len(lab)), n_unlabelled_used=0,
        train_time_s=t_fit, train_time_total_s=t_fit, inference_time_s=t_pred, extra={})
    joblib.dump(rf, C.MODEL_DIR / "random_forest.joblib", compress=3)
    log(f"RF fitted in {t_fit:.2f}s, test acc {accuracy_score(yte, preds['random_forest']):.4f}")

    ls = make_ls(**ls_best)
    assert Ztr.shape[0] == len(ytr) and (y_part == -1).sum() == unl.sum()   # leakage guard
    _, t_fit = timed(ls.fit, Ztr, y_part)
    t0 = time.perf_counter()
    Zte = pca.transform(Xte)
    preds["label_spreading"] = ls_predict(ls, Zte)
    t_pred = time.perf_counter() - t0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = ls.predict_proba(Zte)
    unreached = int(np.isnan(p).any(axis=1).sum())
    probas["label_spreading"] = np.nan_to_num(p, nan=0.0)
    transductive_acc = float(accuracy_score(ytr[unl], ls.transduction_[unl]))
    results_models["label_spreading"] = dict(
        paradigm="semi-supervised",
        features=f"PCA({C.PCA_COMPONENTS}) of 784 pixels, fitted on training images only",
        training_samples_used=int(len(ytr)), n_labelled=int(len(lab)), n_unlabelled_used=int(unl.sum()),
        train_time_s=t_fit, train_time_total_s=t_fit + pca_fit_t, inference_time_s=t_pred,
        extra={"n_iter": int(ls.n_iter_), "pca_fit_time_s": pca_fit_t,
               "transductive_accuracy_hidden_train_labels": transductive_acc,
               "test_images_without_labelled_neighbour": unreached,
               "inference_includes": "PCA transform + graph prediction"})
    joblib.dump(Pipeline([("pca", pca), ("label_spreading", ls)]),
                C.MODEL_DIR / "label_spreading.joblib", compress=3)
    log(f"LS fitted in {t_fit:.2f}s ({ls.n_iter_} iters), test acc "
        f"{accuracy_score(yte, preds['label_spreading']):.4f}, transductive acc {transductive_acc:.4f}, "
        f"unreached test images {unreached}")

    for key, r in results_models.items():
        m = evaluate(yte, preds[key])
        r.update(name=C.MODEL_NAMES[key], hyperparameters=best[key],
                 cv_grid=grids[key].to_dict(orient="records"),
                 cv_best_accuracy=float(grids[key]["cv_accuracy_mean"].max()),
                 inference_ms_per_image=1000 * r["inference_time_s"] / len(yte),
                 metrics=m, top_confusions=top_confusions(m["confusion_matrix"]))

    # ------------------------------------------------------------------ label-percentage sweep
    sweep = []
    if not skip_sweep:
        log("Label-percentage sweep (tuned hyperparameters reused, as in the notebook)...")
        sweep = run_sweep(Xtr, ytr, Ztr, Xte, Zte, yte, best)

    # ------------------------------------------------------------------ save
    vm = psutil.virtual_memory()
    results = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "platform": platform.platform(),
            "cpu": platform.processor(), "cpu_cores_logical": os.cpu_count(),
            "ram_gb": round(vm.total / 1e9, 1), "gpu_detected": gpu_name(),
            "device": "CPU only - scikit-learn LabelSpreading, LogisticRegression and "
                      "RandomForestClassifier have no GPU backend",
        },
        "dataset": {
            "name": "Fashion-MNIST (official split)", "source": "https://github.com/zalandoresearch/fashion-mnist",
            "n_train": int(len(ytr)), "n_test": int(len(yte)), "image_shape": [28, 28], "n_features": 784,
            "class_names": C.CLASS_NAMES,
            "train_class_counts": np.bincount(ytr, minlength=10).tolist(),
            "test_class_counts": np.bincount(yte, minlength=10).tolist(),
            "label_fraction": C.MAIN_LABEL_FRACTION, "n_labelled": int(len(lab)), "n_unlabelled": int(unl.sum()),
            "labelled_per_class": np.bincount(ytr[lab], minlength=10).tolist(),
            "random_state": C.RANDOM_STATE, "cv_folds": C.CV_FOLDS,
        },
        "preprocessing": {"scaling": "pixel / 255", "pca_components": C.PCA_COMPONENTS,
                          "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                          "pca_fit_time_s": pca_fit_t},
        "models": results_models,
        "label_fraction_sweep": sweep,
        "total_runtime_s": time.perf_counter() - T_START,
    }
    with open(C.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(C.PREDICTIONS_NPZ, y_true=yte,
                        **{f"pred_{k}": v.astype(np.int8) for k, v in preds.items()},
                        **{f"proba_{k}": v.astype(np.float32) for k, v in probas.items()})
    save_all_figures(results, img_tr, ytr)
    log(f"Saved {C.RESULTS_JSON.name}, {C.PREDICTIONS_NPZ.name}, models/ and figures/. Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sweep", action="store_true", help="skip the label-percentage / seed sweep")
    main(ap.parse_args().skip_sweep)
