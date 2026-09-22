"""Static figures written to reports/<disease>/ and displayed by the dashboard."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve  # noqa: E402

PALETTE = {"low": "#2e9e5b", "medium": "#f0a020", "high": "#d64545", "blue": "#3b6fd4", "grey": "#8a8f98"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_roc(fitted: dict, X_test, y_test, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, pipe in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, pipe.predict_proba(X_test)[:, 1])
        auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["grey"], lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (hold-out test set)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_calibration(curves: dict, path):
    """curves: {label: (y_true, proba)}"""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["grey"], label="Perfectly calibrated")
    for label, (y, p) in curves.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", lw=1.8, label=label)
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency",
           title="Calibration: do predicted risks mean what they say?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_confusion(y_true, proba, path, threshold=0.5):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay.from_predictions(y_true, (np.asarray(proba) >= threshold).astype(int),
                                            display_labels=["No disease", "Disease"], cmap="Blues",
                                            colorbar=False, ax=ax)
    ax.set_title(f"Confusion matrix (threshold {threshold:.2f})")
    _save(fig, path)


def plot_global_importance(importance, path, backend: str):
    """importance: Series of mean |contribution| in percentage points."""
    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color=PALETTE["blue"])
    ax.set(xlabel="Mean |effect| on risk (percentage points)",
           title=f"Global feature importance ({'SHAP' if backend == 'shap' else 'Shapley sampling'})")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, path)


def contribution_figure(contribs, labels, base_pct: float, prob_pct: float):
    """Horizontal bar chart of one patient's feature contributions (in percentage points)."""
    order = contribs.abs().sort_values().index
    vals = contribs[order] * 100
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(vals) + 1.3))
    colors = [PALETTE["high"] if v > 0 else PALETTE["low"] for v in vals]
    ax.barh([labels[i] for i in order], vals.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="Effect on risk (percentage points)",
           title=f"Why {prob_pct:.0f}%?  (population average ≈ {base_pct:.0f}%)")
    ax.grid(axis="x", alpha=0.25)
    for y, v in enumerate(vals.values):
        ax.text(v + (0.4 if v >= 0 else -0.4), y, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    lim = max(abs(vals).max(), 1) * 1.25
    ax.set_xlim(-lim, lim)
    fig.tight_layout()
    return fig


def gauge_figure(prob: float, thresholds):
    """A slim horizontal 'risk bar' with Low / Medium / High zones and a marker.
    The axis is zoomed for rare outcomes (e.g. stroke, bands at 5% / 15%) so the zones stay readable."""
    t0, t1 = thresholds
    xmax = 1.0 if t1 > 0.4 else 0.5 if t1 > 0.1 else 0.25
    shown = min(prob, xmax)
    fig, ax = plt.subplots(figsize=(6.5, 1.1))
    for lo, hi, c in [(0, t0, PALETTE["low"]), (t0, t1, PALETTE["medium"]), (t1, xmax, PALETTE["high"])]:
        ax.barh(0, (hi - lo) * 100, left=lo * 100, color=c, alpha=0.85, height=0.5)
    ax.plot([shown * 100], [0], marker="v", color="black", markersize=14, mec="white", zorder=5)
    ax.set_xlim(0, xmax * 100), ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ticks = [0, t0 * 100, t1 * 100, xmax * 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}%" for t in ticks[:-1]] + [f"{xmax * 100:.0f}%+" if xmax < 1 else "100%"],
                       fontsize=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def plot_eda(df, cfg, outdir):
    """Class balance, distributions by outcome, correlation heatmap, categorical prevalence."""
    sns.set_theme(style="whitegrid")
    target = cfg.target
    numeric = [f.name for f in cfg.features if f.kind == "number" and f.name in df.columns]

    fig, ax = plt.subplots(figsize=(4, 3.4))
    counts = df[target].value_counts().sort_index()
    ax.bar(["No disease", "Disease"], counts.values, color=[PALETTE["low"], PALETTE["high"]])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / len(df):.0%})", ha="center", va="bottom", fontsize=9)
    ax.set_title("Class balance")
    _save(fig, outdir / "eda_class_balance.png")

    ncols = 3
    nrows = max(1, int(np.ceil(len(numeric) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows))
    for ax, col in zip(np.atleast_1d(axes).ravel(), numeric):
        sns.kdeplot(data=df, x=col, hue=target, fill=True, common_norm=False, ax=ax,
                    palette=[PALETTE["low"], PALETTE["high"]], legend=False, warn_singular=False)
        spec = cfg.feature(col)
        ax.set_title(spec.label)
        ax.set_xlabel(f"{spec.unit}" if spec.unit else "")
    for ax in np.atleast_1d(axes).ravel()[len(numeric):]:
        ax.axis("off")
    fig.suptitle("Distributions by outcome (green = no disease, red = disease)", y=1.0)
    _save(fig, outdir / "eda_distributions.png")

    n_cols = df.shape[1]
    fig, ax = plt.subplots(figsize=(max(7.5, 0.55 * n_cols), max(6, 0.5 * n_cols)))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.7}, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation matrix")
    _save(fig, outdir / "eda_correlation.png")

    cats = [f for f in cfg.features if f.kind in ("category", "binary") and f.name in df.columns]
    if not cats:
        matplotlib.rcdefaults()
        return
    ncols = 3
    nrows = max(1, int(np.ceil(len(cats) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.3 * nrows))
    for ax, spec in zip(np.atleast_1d(axes).ravel(), cats):
        rate = df.groupby(spec.name)[target].mean() * 100
        ax.bar([str(spec.options.get(int(k), k)) for k in rate.index], rate.values, color=PALETTE["blue"])
        ax.set_title(spec.label, fontsize=10)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", labelsize=7, rotation=20)
    for ax in np.atleast_1d(axes).ravel()[len(cats):]:
        ax.axis("off")
    fig.suptitle("Disease prevalence (%) by category", y=1.0)
    _save(fig, outdir / "eda_categorical.png")
    matplotlib.rcdefaults()   # do not leak the seaborn theme into later figures
