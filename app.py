"""Streamlit dashboard. Reads saved results/models produced by train.py; never retrains.

Run:  .venv\\Scripts\\streamlit run app.py
"""
import json
import time

from fmnist import config as C

C.configure_storage()   # keep temp/cache files in the project folder on D:

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from fmnist.data import load_split, to_features  # noqa: E402

st.set_page_config(page_title="Fashion-MNIST Semi-Supervised Learning", page_icon="👕", layout="wide")

ORDER = ["label_spreading", "logistic_regression", "random_forest"]
COLORS = {"label_spreading": "#2a78d6", "logistic_regression": "#eb6834", "random_forest": "#1baf7a"}
SHORT = {"label_spreading": "LabelSpreading", "logistic_regression": "Logistic Regression",
         "random_forest": "Random Forest"}
COLOR_MAP = {SHORT[k]: COLORS[k] for k in ORDER}
NAMES = C.CLASS_NAMES


# ---------------------------------------------------------------- loaders (cached)
@st.cache_data(show_spinner=False)
def load_results():
    with open(C.RESULTS_JSON, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_predictions():
    with np.load(C.PREDICTIONS_NPZ) as z:
        return {k: z[k] for k in z.files}


@st.cache_data(show_spinner="Loading Fashion-MNIST...")
def load_images(split):
    return load_split(split)


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    return {k: joblib.load(C.MODEL_DIR / f"{k}.joblib") for k in ORDER}


def chart(fig, height=420):
    """Render a Plotly figure. The title is drawn as Streamlit text above the chart (it wraps on narrow
    screens) so it can never collide with the legend; Plotly auto-expands the top margin for the legend."""
    title = fig.layout.title.text
    if title:
        st.markdown(f"**{title}**")
    fig.update_layout(title=None, height=height, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    st.plotly_chart(fig, width="stretch")


def display_img(img28, size=112):
    """Show dark garment on light background (as in the notebook's gray_r plots), upscaled crisply."""
    return Image.fromarray(255 - img28).resize((size, size), Image.NEAREST)


def image_strip(imgs, size=84, gap=6):
    """Several 28x28 images side by side on white, as one image (keeps rows aligned at any width)."""
    out = Image.new("L", (len(imgs) * size + (len(imgs) - 1) * gap, size), 255)
    for i, im in enumerate(imgs):
        out.paste(display_img(im, size), (i * (size + gap), 0))
    return out


@st.cache_data(show_spinner="Projecting training images onto 2 principal components...")
def pca_2d_training():
    """2-D PCA view of the 60,000 TRAINING images, using the PCA fitted on training images only in
    train.py (stored in the LabelSpreading pipeline). Its first two components equal a 2-component PCA
    (deterministic full SVD), so nothing is re-fitted here and the test set is never involved."""
    pca = joblib.load(C.MODEL_DIR / "label_spreading.joblib").named_steps["pca"]
    img_tr, y_tr = load_images("train")
    Z = pca.transform(to_features(img_tr))[:, :2].astype(np.float32)
    return Z, y_tr, float(pca.explained_variance_ratio_[:2].sum()), int(pca.n_samples_)


if not C.RESULTS_JSON.exists():
    st.error("No saved results found. Run `.venv\\Scripts\\python.exe train.py` first.")
    st.stop()

R = load_results()
DS, M = R["dataset"], R["models"]

# ---------------------------------------------------------------- header
st.title("Semi-Supervised Learning on Fashion-MNIST")
st.caption(f"LabelSpreading vs. Logistic Regression vs. Random Forest · official 60,000 / 10,000 split · "
           f"{100*DS['label_fraction']:.0f}% labelled training data · results from run {R['created']}")

k = st.columns(5)
k[0].metric("Training images", f"{DS['n_train']:,}")
k[1].metric("Test images (held out)", f"{DS['n_test']:,}")
k[2].metric("Labelled", f"{DS['n_labelled']:,}")
k[3].metric("Unlabelled (-1)", f"{DS['n_unlabelled']:,}")
best_key = max(ORDER, key=lambda m: M[m]["metrics"]["accuracy"])
k[4].metric("Best test accuracy", f"{100*M[best_key]['metrics']['accuracy']:.2f}%", SHORT[best_key],
            delta_color="off")

tabs = st.tabs(["Dataset", "Comparison", "Confusion matrices", "Per-class", "Timing",
                "Labelled-data %", "Tuning", "Live prediction"])

# ---------------------------------------------------------------- Dataset
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        df = pd.DataFrame({"Class": NAMES * 2,
                           "Images": DS["train_class_counts"] + DS["test_class_counts"],
                           "Split": ["Train"] * 10 + ["Test"] * 10})
        fig = px.bar(df, x="Class", y="Images", color="Split", barmode="group",
                     color_discrete_map={"Train": "#2a78d6", "Test": "#86b6ef"},
                     title="Class distribution")
        chart(fig, 380)
    with c2:
        st.subheader("Experimental setup")
        st.markdown(f"""
- **Data:** official Fashion-MNIST, {DS['n_train']:,} train + {DS['n_test']:,} test, 28×28 grayscale, 10 classes
- **Preprocessing:** flattened to 784 features, pixels ÷ 255 → [0, 1]
- **Labels:** {100*DS['label_fraction']:.0f}% kept per class (seed {DS['random_state']}) → {DS['n_labelled']:,} labelled; {DS['n_unlabelled']:,} set to **-1**
- **LabelSpreading:** k-NN graph on PCA-{R['preprocessing']['pca_components']} features ({100*R['preprocessing']['pca_explained_variance']:.1f}% variance), fitted on all {DS['n_train']:,} training images
- **Logistic Regression / Random Forest:** trained on the same {DS['n_labelled']:,} labelled images only
- **Tuning:** {DS['cv_folds']}-fold stratified CV on labelled training data only; the test set is used once for final scoring
- **Device:** {R['environment']['device']}
""")
    st.subheader("Sample images")
    img_tr, y_tr = load_images("train")
    for row in (range(0, 5), range(5, 10)):          # one row per 5 classes keeps order when columns stack
        for col, c in zip(st.columns(5), row):
            with col:
                st.image(image_strip([img_tr[i] for i in np.where(y_tr == c)[0][:3]]),
                         caption=f"{c}: {NAMES[c]}", width="stretch")
    lab_df = pd.DataFrame({"Class": NAMES, "Labelled": DS["labelled_per_class"],
                           "Unlabelled": np.array(DS["train_class_counts"]) - np.array(DS["labelled_per_class"])})
    fig = px.bar(lab_df.melt(id_vars="Class", var_name="Status", value_name="Images"), x="Class", y="Images",
                 color="Status", color_discrete_map={"Labelled": "#2a78d6", "Unlabelled": "#c3c2b7"},
                 title="Labelled vs. unlabelled training images per class")
    chart(fig, 340)

    st.subheader("2-D PCA projection of the training images")
    if not (C.MODEL_DIR / "label_spreading.joblib").exists():
        st.info("Run `train.py` first: the projection uses the PCA fitted during training.")
    else:
        Z2, y2, var2, n_fit = pca_2d_training()
        pdf = pd.DataFrame({"Principal Component 1": Z2[:, 0], "Principal Component 2": Z2[:, 1],
                            "Class": np.array(NAMES)[y2]})
        fig = px.scatter(pdf, x="Principal Component 1", y="Principal Component 2", color="Class",
                         category_orders={"Class": NAMES}, color_discrete_sequence=px.colors.qualitative.D3,
                         render_mode="webgl", opacity=0.55,
                         title=f"{len(pdf):,} training images coloured by their true class "
                               f"(PC1 + PC2 explain {100*var2:.1f}% of the variance)")
        fig.update_traces(marker=dict(size=3), hovertemplate="%{fullData.name}<extra></extra>")
        fig.update_layout(legend=dict(itemsizing="constant"))
        chart(fig, 560)
        st.caption(f"PCA fitted on the {n_fit:,} training images only (no labels, no test images) during "
                   "training — the same projection LabelSpreading builds its k-NN graph in. True labels are "
                   "used here for colouring only. Click a class in the legend to hide it; double-click to isolate it.")

# ---------------------------------------------------------------- Comparison
with tabs[1]:
    metric_keys = {"accuracy": "Accuracy", "precision_macro": "Precision (macro)", "recall_macro": "Recall (macro)",
                   "f1_macro": "F1 (macro)", "precision_weighted": "Precision (weighted)",
                   "recall_weighted": "Recall (weighted)", "f1_weighted": "F1 (weighted)"}
    rows = []
    for key in ORDER:
        r = M[key]
        rows.append({"Model": SHORT[key], "Paradigm": r["paradigm"],
                     "Training images used": r["training_samples_used"], "Labelled": r["n_labelled"],
                     "Unlabelled used": r["n_unlabelled_used"],
                     **{v: r["metrics"][m] for m, v in metric_keys.items()}})
    table = pd.DataFrame(rows)
    long = table.melt(id_vars="Model", value_vars=list(metric_keys.values()), var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", color_discrete_map=COLOR_MAP,
                 text_auto=".3f", title="Test-set metrics (10,000 held-out images)")
    fig.update_yaxes(range=[max(0, long["Score"].min() - 0.08), min(1, long["Score"].max() + 0.04)])
    fig.update_traces(textfont_size=10, textposition="outside")
    chart(fig, 460)
    st.dataframe(table.style.format({v: "{:.4f}" for v in metric_keys.values()}
                                    | {"Training images used": "{:,}", "Labelled": "{:,}", "Unlabelled used": "{:,}"}),
                 hide_index=True, width="stretch")
    ls, lr = M["label_spreading"]["metrics"]["accuracy"], M["logistic_regression"]["metrics"]["accuracy"]
    rf = M["random_forest"]["metrics"]["accuracy"]
    x = M["label_spreading"]["extra"]
    st.markdown(f"""
**Reading the result.** LabelSpreading − Logistic Regression = **{100*(ls-lr):+.2f} pp**; LabelSpreading − Random Forest = **{100*(ls-rf):+.2f} pp**.
LabelSpreading also labelled the {DS['n_unlabelled']:,} hidden training images with **{100*x['transductive_accuracy_hidden_train_labels']:.2f}%** accuracy
(transductive accuracy, scored after training). Test images with no labelled neighbour reached in the graph: {x['test_images_without_labelled_neighbour']}.
""")

# ---------------------------------------------------------------- Confusion matrices
with tabs[2]:
    norm = st.toggle("Normalise rows (show recall per true class)", value=False)
    for key in ORDER:
        cm = np.array(M[key]["metrics"]["confusion_matrix"], dtype=float)
        z = cm / cm.sum(axis=1, keepdims=True) if norm else cm
        st.markdown(f"#### {SHORT[key]} — {100*M[key]['metrics']['accuracy']:.2f}% accuracy")
        c_cm, c_tab = st.columns([3, 2], gap="large")
        fig = px.imshow(z, x=NAMES, y=NAMES, color_continuous_scale="Blues", text_auto=".2f" if norm else "d",
                        labels=dict(x="Predicted", y="True", color="Share" if norm else "Images"))
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10), height=540)
        fig.update_traces(textfont_size=10)
        fig.update_xaxes(tickangle=45, tickmode="linear", side="bottom")
        fig.update_yaxes(tickmode="linear")
        with c_cm:
            st.plotly_chart(fig, width="stretch")
        with c_tab:
            st.caption("Most frequent mistakes")
            st.dataframe(pd.DataFrame(M[key]["top_confusions"]), hide_index=True, width="stretch")
        st.divider()

# ---------------------------------------------------------------- Per-class
with tabs[3]:
    metric = st.radio("Metric", ["f1-score", "precision", "recall"], horizontal=True)
    rows = [{"Class": r["class"], "Model": SHORT[key], "precision": r["precision"], "recall": r["recall"],
             "f1-score": r["f1-score"], "support": r["support"]}
            for key in ORDER for r in M[key]["metrics"]["per_class"]]
    pc = pd.DataFrame(rows)
    fig = px.bar(pc, x="Class", y=metric, color="Model", barmode="group", color_discrete_map=COLOR_MAP,
                 title=f"Per-class {metric} on the test set")
    fig.update_yaxes(range=[0, 1.02])
    chart(fig, 430)
    wide = pc.pivot(index="Class", columns="Model", values=["precision", "recall", "f1-score"]).reindex(NAMES)
    wide.columns = [f"{m} · {mdl}" for m, mdl in wide.columns]
    st.dataframe(wide.style.format("{:.3f}").background_gradient(cmap="Blues", axis=None),
                 width="stretch")
    st.subheader("Macro and weighted averages")
    avg = pd.DataFrame([{"Model": SHORT[k], "Average": a,
                         "Precision": M[k]["metrics"][f"precision_{a}"], "Recall": M[k]["metrics"][f"recall_{a}"],
                         "F1": M[k]["metrics"][f"f1_{a}"]} for k in ORDER for a in ("macro", "weighted")])
    st.dataframe(avg.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------- Timing
with tabs[4]:
    tdf = pd.DataFrame([{"Model": SHORT[k], "Training images": M[k]["training_samples_used"],
                         "Model fit (s)": M[k]["train_time_s"], "Total training incl. PCA (s)": M[k]["train_time_total_s"],
                         "Prediction, 10,000 images (s)": M[k]["inference_time_s"],
                         "ms / image": M[k]["inference_ms_per_image"]} for k in ORDER])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(tdf, y="Model", x="Total training incl. PCA (s)", color="Model", orientation="h",
                     color_discrete_map=COLOR_MAP, text_auto=".2f", title="Training time (s)")
        fig.update_layout(showlegend=False)
        chart(fig, 300)
    with c2:
        fig = px.bar(tdf, y="Model", x="Prediction, 10,000 images (s)", color="Model", orientation="h",
                     color_discrete_map=COLOR_MAP, text_auto=".2f", title="Prediction time on the test set (s)")
        fig.update_layout(showlegend=False)
        chart(fig, 300)
    st.dataframe(tdf.style.format({"Training images": "{:,}", "Model fit (s)": "{:.2f}",
                                   "Total training incl. PCA (s)": "{:.2f}",
                                   "Prediction, 10,000 images (s)": "{:.3f}", "ms / image": "{:.4f}"}),
                 hide_index=True, width="stretch")
    env = R["environment"]
    st.caption(f"Measured on {env['cpu_cores_logical']} logical CPU cores, {env['ram_gb']} GB RAM, "
               f"scikit-learn {env['scikit_learn']}. GPU detected: {env['gpu_detected'] or 'none'} — "
               f"not used: {env['device']}. LabelSpreading times include the PCA step "
               f"(fit {R['preprocessing']['pca_fit_time_s']:.2f}s); its prediction time includes PCA transform.")

# ---------------------------------------------------------------- Sweep
with tabs[5]:
    sw = R.get("label_fraction_sweep") or []
    if not sw:
        st.info("The label-percentage sweep was skipped in this run (`train.py --skip-sweep`).")
    else:
        sdf = pd.DataFrame(sw)
        fig = go.Figure()
        for key in ORDER:
            g = sdf.groupby("label_fraction")[f"{key}_accuracy"].agg(["mean", "std"]).reset_index()
            x = 100 * g["label_fraction"]
            fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=list(g["mean"] + g["std"]) + list((g["mean"] - g["std"])[::-1]),
                                     fill="toself", fillcolor=COLORS[key], opacity=0.15, line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=g["mean"], mode="lines+markers", name=SHORT[key],
                                     line=dict(color=COLORS[key], width=2), marker=dict(size=9),
                                     error_y=dict(type="data", array=g["std"], visible=False),
                                     hovertemplate="%{x:.0f}% labelled<br>accuracy %{y:.4f}<extra>" + SHORT[key] + "</extra>"))
        fig.update_layout(title=f"Test accuracy vs. labelled-data percentage (mean ± std over seeds "
                                f"{sorted(sdf['seed'].unique().tolist())})",
                          xaxis_title="Labelled training data (%)", yaxis_title="Test accuracy", hovermode="x unified")
        fig.update_xaxes(tickvals=[100 * f for f in C.LABEL_FRACTIONS])
        chart(fig, 460)
        summ = sdf.groupby(["label_fraction", "n_labelled"]).agg(
            **{f"{SHORT[k]} mean": (f"{k}_accuracy", "mean") for k in ORDER},
            **{f"{SHORT[k]} std": (f"{k}_accuracy", "std") for k in ORDER}).reset_index()
        summ.insert(0, "Labelled %", (100 * summ.pop("label_fraction")).astype(int))
        st.dataframe(summ.style.format({c: "{:.4f}" for c in summ.columns if "mean" in c or "std" in c}
                                       | {"n_labelled": "{:,}"}), hide_index=True, width="stretch")
        st.caption("Hyperparameters tuned at 10% labels are reused at every percentage (as in the Jury 2 notebook). "
                   "Labelled images are re-selected per class for each seed.")

# ---------------------------------------------------------------- Tuning
with tabs[6]:
    st.caption(f"{DS['cv_folds']}-fold stratified cross-validation on the {DS['n_labelled']:,} labelled training "
               "images only. For LabelSpreading each fold's known labels are hidden (-1) and the graph over all "
               f"{DS['n_train']:,} training images is asked to recover them.")
    c1, c2 = st.columns(2)
    with c1:
        g = pd.DataFrame(M["label_spreading"]["cv_grid"])
        piv = g.pivot(index="n_neighbors", columns="alpha", values="cv_accuracy_mean")
        fig = px.imshow(piv, text_auto=".4f", color_continuous_scale="Blues", aspect="auto",
                        labels=dict(x="alpha", y="n_neighbors", color="CV accuracy"),
                        title=f"LabelSpreading CV accuracy — best {M['label_spreading']['hyperparameters']}")
        fig.update_xaxes(type="category"); fig.update_yaxes(type="category")
        chart(fig, 380)
    with c2:
        g = pd.DataFrame(M["logistic_regression"]["cv_grid"])
        g["C"] = [f"C = {c:g}<br>{m:.4f} ± {s:.4f}"
                  for c, m, s in zip(g["C"], g["cv_accuracy_mean"], g["cv_accuracy_std"])]
        fig = px.bar(g, x="C", y="cv_accuracy_mean", error_y="cv_accuracy_std",
                     color_discrete_sequence=[COLORS["logistic_regression"]],
                     labels={"cv_accuracy_mean": "CV accuracy", "C": ""},
                     title=f"Logistic Regression CV accuracy — best C = {M['logistic_regression']['hyperparameters']['C']}")
        fig.update_xaxes(type="category")
        fig.update_yaxes(range=[g["cv_accuracy_mean"].min() - 0.05, g["cv_accuracy_mean"].max() + 0.03])
        chart(fig, 380)
    g = pd.DataFrame(M["random_forest"]["cv_grid"])
    g["max_depth"] = g["max_depth"].map(lambda v: "None" if pd.isna(v) else str(int(v)))
    st.markdown(f"**Random Forest CV grid** — best `{M['random_forest']['hyperparameters']}`")
    st.dataframe(g.sort_values("cv_accuracy_mean", ascending=False).style.format(
        {"cv_accuracy_mean": "{:.4f}", "cv_accuracy_std": "{:.4f}"}), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Live prediction
with tabs[7]:
    missing = [k for k in ORDER if not (C.MODEL_DIR / f"{k}.joblib").exists()]
    if missing:
        st.error(f"Trained model files missing: {missing}. Run `train.py` to create them.")
        st.stop()
    models = load_models()
    P = load_predictions()
    img_te, y_te = load_images("test")

    src = st.radio("Image source", ["Official test set", "Upload your own image"], horizontal=True)
    img28, true_label = None, None
    if src == "Official test set":
        c1, c2, c3 = st.columns([2, 2, 1])
        filt = c1.selectbox("Show", ["Any test image", "Models disagree", "LabelSpreading wrong",
                                     "All three models wrong", "All three models right"])
        cls = c2.selectbox("True class", ["Any"] + NAMES)
        preds = np.stack([P[f"pred_{k}"] for k in ORDER])
        mask = np.ones(len(y_te), bool)
        if filt == "Models disagree":
            mask = ~((preds[0] == preds[1]) & (preds[1] == preds[2]))
        elif filt == "LabelSpreading wrong":
            mask = preds[0] != y_te
        elif filt == "All three models wrong":
            mask = (preds != y_te).all(axis=0)
        elif filt == "All three models right":
            mask = (preds == y_te).all(axis=0)
        if cls != "Any":
            mask &= y_te == NAMES.index(cls)
        pool = np.where(mask)[0]
        if len(pool) == 0:
            st.warning("No test images match this filter.")
            st.stop()
        if "pick" not in st.session_state or st.session_state.get("pool_key") != (filt, cls):
            st.session_state.pool_key = (filt, cls)
            st.session_state.pick = int(np.random.default_rng().choice(pool))
        if c3.button("🎲 Random image", width="stretch"):
            st.session_state.pick = int(np.random.default_rng().choice(pool))
        idx = st.session_state.pick
        img28, true_label = img_te[idx], int(y_te[idx])
        st.caption(f"Test image #{idx} · {len(pool):,} images match the filter")
    else:
        up = st.file_uploader("Upload a clothing image (PNG/JPG). Best results: single item, plain background.",
                              type=["png", "jpg", "jpeg", "webp", "bmp"])
        if up is None:
            st.info("Upload an image to classify it with all three models.")
            st.stop()
        g = ImageOps.grayscale(Image.open(up))
        arr = np.asarray(g, dtype=np.float32)
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]]).mean()
        invert = st.checkbox("Invert colours (Fashion-MNIST has a bright item on a dark background)",
                             value=bool(border > 127))
        if invert:
            g = ImageOps.invert(g)
        g = ImageOps.pad(g, (max(g.size),) * 2, color=0)       # square, keep aspect ratio
        img28 = np.asarray(g.resize((28, 28), Image.LANCZOS), dtype=np.uint8)

    x = to_features(img28[None])
    c_img, c_res = st.columns([1, 3])
    with c_img:
        st.image(display_img(img28, 224), caption="Model input (28×28)", width="stretch")
        if true_label is not None:
            st.markdown(f"**True label:** {NAMES[true_label]}")
    rows, probs = [], []
    for key in ORDER:
        t0 = time.perf_counter()
        with np.errstate(invalid="ignore", divide="ignore"):
            p = models[key].predict_proba(x)[0]
        dt_ms = 1000 * (time.perf_counter() - t0)
        reached = not np.isnan(p).any()
        p = np.nan_to_num(p)
        pred = int(np.argmax(p)) if reached else None
        rows.append((key, pred, float(p.max()) if reached else None, dt_ms))
        probs += [{"Model": SHORT[key], "Class": NAMES[i], "Probability": float(p[i])} for i in range(10)]
    with c_res:
        cols = st.columns(3)
        for col, (key, pred, conf, dt_ms) in zip(cols, rows):
            if pred is None:
                col.metric(SHORT[key], "no label reached")
                continue
            ok = "" if true_label is None else (" ✅" if pred == true_label else " ❌")
            col.metric(SHORT[key], NAMES[pred] + ok, f"{100*conf:.1f}% · {dt_ms:.1f} ms", delta_color="off")
        fig = px.bar(pd.DataFrame(probs), x="Class", y="Probability", color="Model", barmode="group",
                     color_discrete_map=COLOR_MAP, title="Class probabilities")
        fig.update_yaxes(range=[0, 1])
        chart(fig, 360)
    st.caption("LabelSpreading probabilities are its label-spreading scores over the image's k nearest training "
               "images in PCA space; Random Forest probabilities are tree-vote shares.")
