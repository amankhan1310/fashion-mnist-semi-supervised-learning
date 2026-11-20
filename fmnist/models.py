"""Model builders, 5-fold CV hyperparameter tuning, timing and evaluation.

All tuning uses training data only; the official test set is never touched here.
"""
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.semi_supervised import LabelSpreading

from . import config as C


def make_lr(C_value):
    return LogisticRegression(C=C_value, random_state=C.RANDOM_STATE, **C.LR_FIXED)


def make_rf(n_jobs=-1, **params):
    return RandomForestClassifier(random_state=C.RANDOM_STATE, n_jobs=n_jobs, **params)


def make_ls(n_neighbors, alpha, n_jobs=-1):
    return LabelSpreading(n_neighbors=n_neighbors, alpha=alpha, n_jobs=n_jobs, **C.LS_FIXED)


def _cv():
    return StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)


def _summarise(rows, keys):
    df = pd.DataFrame(rows)
    grid = (df.groupby(keys, dropna=False, sort=False)["acc"]
              .agg(cv_accuracy_mean="mean", cv_accuracy_std=lambda s: s.std(ddof=0))
              .reset_index())
    best = grid.loc[grid["cv_accuracy_mean"].idxmax()]   # first max wins, as in the notebook
    return grid, {k: best[k] for k in keys}


# ---------------------------------------------------------------- supervised tuning
def _fit_score(model, X, y, tr, va):
    model.fit(X[tr], y[tr])
    return accuracy_score(y[va], model.predict(X[va]))


def tune_logistic_regression(X_lab, y_lab):
    folds = list(_cv().split(X_lab, y_lab))
    tasks = [(c, f) for c in C.LR_C_GRID for f in range(len(folds))]
    accs = Parallel(n_jobs=min(len(tasks), 8))(
        delayed(_fit_score)(make_lr(c), X_lab, y_lab, *folds[f]) for c, f in tasks)
    rows = [{"C": c, "fold": f, "acc": a} for (c, f), a in zip(tasks, accs)]
    grid, best = _summarise(rows, ["C"])
    return grid, {"C": float(best["C"])}


def tune_random_forest(X_lab, y_lab):
    folds = list(_cv().split(X_lab, y_lab))
    keys = list(C.RF_GRID)
    rows = []
    for values in itertools.product(*C.RF_GRID.values()):
        params = dict(zip(keys, values))
        for f, (tr, va) in enumerate(folds):
            rows.append({**params, "fold": f, "acc": _fit_score(make_rf(**params), X_lab, y_lab, tr, va)})
    grid, best = _summarise(rows, keys)
    best = {k: (None if pd.isna(v) else (int(v) if k != "max_features" else v)) for k, v in best.items()}
    grid["max_depth"] = grid["max_depth"].astype(object).where(grid["max_depth"].notna(), None)
    return grid, best


# ---------------------------------------------------------------- LabelSpreading tuning
def _ls_fold_score(Z, y_partial, va_idx, k, a):
    """Hide one fold of the KNOWN labels, fit on all training images, score recovery."""
    y_cv = y_partial.copy()
    y_cv[va_idx] = -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = make_ls(k, a, n_jobs=1).fit(Z, y_cv)
    return accuracy_score(y_partial[va_idx], m.transduction_[va_idx])


def tune_label_spreading(Z_train, y_partial, labelled_idx, n_jobs=6):
    """Same protocol as the notebook: 5 folds over the labelled samples only."""
    folds = [labelled_idx[va] for _, va in _cv().split(labelled_idx, y_partial[labelled_idx])]
    tasks = [(k, a, f) for k in C.LS_K_GRID for a in C.LS_ALPHA_GRID for f in range(len(folds))]
    accs = Parallel(n_jobs=n_jobs, max_nbytes="1M")(
        delayed(_ls_fold_score)(Z_train, y_partial, folds[f], k, a) for k, a, f in tasks)
    rows = [{"n_neighbors": k, "alpha": a, "fold": f, "acc": acc} for (k, a, f), acc in zip(tasks, accs)]
    grid, best = _summarise(rows, ["n_neighbors", "alpha"])
    return grid, {"n_neighbors": int(best["n_neighbors"]), "alpha": float(best["alpha"])}


# ---------------------------------------------------------------- timing / evaluation
def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def evaluate(y_true, y_pred):
    """Accuracy, macro/weighted P/R/F1, per-class report and confusion matrix."""
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        out[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    out = {k: float(v) for k, v in out.items()}
    report = classification_report(y_true, y_pred, labels=range(10), target_names=C.CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    out["per_class"] = [{"class": n, **{k: float(v) for k, v in report[n].items()}} for n in C.CLASS_NAMES]
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=range(10)).tolist()
    return out


def top_confusions(cm, n=5):
    off = np.array(cm, dtype=int)
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(-off.ravel(), kind="stable"), off.shape))[0][:n]
    return [{"true": C.CLASS_NAMES[i], "predicted": C.CLASS_NAMES[j], "count": int(off[i, j])} for i, j in order]
