#!/usr/bin/env python
"""Regenerate every figure of the manuscript from the reported results.

Usage (repository root):
    python scripts/make_figures.py                 # PDF (vector) + PNG (600 dpi)
    python scripts/make_figures.py --formats pdf png tiff --dpi 600

Output goes to paper/figures/. Numbers come exclusively from
src/neuropulse/reported_results.py and from quantities derived from them.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuropulse import metrics as M  # noqa: E402
from neuropulse import models  # noqa: E402
from neuropulse import reported_results as R  # noqa: E402

OUT = ROOT / "paper" / "figures"
FULL_WIDTH = 6.3  # inches; matches the text width of the manuscript on A4 with 2.5 cm margins

# Okabe-Ito palette (distinguishable under common forms of colour-vision deficiency).
COLOURS = {"CNN": "#0072B2", "CNN-attention": "#D55E00", "MLP": "#009E73", "RBF-SVM": "#CC79A7"}
GREY, DARK, LIGHT = "#6b6b6b", "#1a1a1a", "#f2f2f2"
DISPLAY = {"CNN": "CNN", "CNN-attention": "CNN\u2013attention", "MLP": "MLP", "RBF-SVM": "RBF-SVM"}

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})


def confusion_for(model):
    a2f, f2a = R.ERRORS[model]
    cm = np.diag([R.N_PER_CLASS_TEST] * 5).astype(int)
    cm[0, 0] -= a2f
    cm[0, 2] += a2f
    cm[2, 2] -= f2a
    cm[2, 0] += f2a
    return cm


def save(fig, name, formats, dpi):
    OUT.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        kwargs = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.04, "facecolor": "white"}
        if fmt == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUT / f"{name}.{fmt}", format=fmt, **kwargs)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- diagram helpers
def box(ax, x, y, w, h, lines, face="white", edge=DARK, size=7.6, bold_first=True, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06", linewidth=0.9,
                                edgecolor=edge, facecolor=face, linestyle="--" if dashed else "-", zorder=2))
    n = len(lines)
    step = h / (n + 1)
    for i, line in enumerate(lines):
        weight = "bold" if (bold_first and i == 0) else "normal"
        ax.text(x + w / 2, y + h - step * (i + 1), line, ha="center", va="center", fontsize=size,
                fontweight=weight, color=DARK, zorder=3)


def arrow(ax, p0, p1, style="-|>", rad=0.0, colour=DARK, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9, linewidth=lw, color=colour,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=1, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------- Figure: workflow
def fig_workflow(formats, dpi):
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    ax.set_xlim(-0.05, 6.35)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    w, h, gap = 1.42, 0.92, 0.206
    top = [("Synthetic CSV", "2,500 records", "28 numeric columns"),
           ("Schema checks", "canonical column order", "non-finite rows masked"),
           ("Training-fitted scaler", "saved training artifact", "never refitted on uploads"),
           ("Reshape", "4 \u00d7 7 tensor", "groups \u00d7 descriptors")]
    bottom = [("CNN extractor", "two Conv1D blocks", ""),
              ("Self-attention", "residual addition", "layer normalization"),
              ("Softmax", "five class scores", "label via saved encoder"),
              ("Dashboard output", "label and top score", "batch summary, history")]
    x0 = (6.3 - (4 * w + 3 * gap)) / 2
    ytop, ybot = 1.9, 0.5
    xs = [x0 + i * (w + gap) for i in range(4)]
    for (a, b, c), x in zip(top, xs):
        box(ax, x, ytop, w, h, [a, b, c], face=LIGHT, size=7.4)
    for (a, b, c), x in zip(bottom, xs):
        box(ax, x, ybot, w, h, [a, b] + ([c] if c else []), face="white", size=7.4)
    for row_y in (ytop, ybot):
        for i in range(3):
            arrow(ax, (xs[i] + w, row_y + h / 2), (xs[i + 1], row_y + h / 2))
    xr, xl = xs[3] + w / 2, xs[0] + w / 2
    ym = (ytop + ybot + h) / 2
    ax.plot([xr, xr], [ytop, ym], color=DARK, lw=1.0, zorder=1)
    ax.plot([xr, xl], [ym, ym], color=DARK, lw=1.0, zorder=1)
    arrow(ax, (xl, ym), (xl, ybot + h))
    ax.text(3.15, 0.12, "All feature values are synthetic; none originates from EEG acquisition.",
            ha="center", va="center", fontsize=8, color=GREY, style="italic")
    save(fig, "fig_workflow", formats, dpi)


# --------------------------------------------------------------- Figure: architecture
def fig_architecture(formats, dpi):
    layers = models.cnn_attention_layers()
    spec = [
        ("Input", "channel groups,", "descriptors"),
        ("Conv1D, ReLU", "64 filters, kernel 2", "same padding"),
        ("Conv1D, ReLU", "128 filters, kernel 2", "same padding"),
        ("Self-attention", "Q = K = V = U", "no projections"),
        ("Residual addition", "U + Attention(U)", ""),
        ("Layer norm.", "over 128 features", ""),
        ("Global avg. pooling", "over 4 channel groups", ""),
        ("Dense 128", "ReLU", ""),
        ("Dropout 0.4", "training only", ""),
        ("Dense 64", "ReLU", ""),
        ("Dense 5", "softmax", ""),
    ]
    rows = [list(range(0, 4)), list(range(4, 8)), list(range(8, 11))]
    W, H = 6.3, 4.75
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, H))
    ax.set_xlim(-0.05, W + 0.05)
    ax.set_ylim(0, H)
    ax.axis("off")
    w, h, gap = 1.42, 0.95, 0.206
    ys = [3.35, 1.95, 0.55]
    pos = {}
    for r, idxs in enumerate(rows):
        for i, k in enumerate(idxs):
            pos[k] = (i * (w + gap) + (W - (4 * w + 3 * gap)) / 2, ys[r])
    for k, (x, y) in pos.items():
        name, l2, l3 = spec[k]
        shape = layers[k][1]
        lines = [name, l2] + ([l3] if l3 else []) + [shape]
        box(ax, x, y, w, h, lines, face="#fde9dc" if k in (3, 4) else LIGHT, size=7.3)
        params = layers[k][2]
        ax.text(x + w / 2, y - 0.13, f"{params:,} parameters" if params else "no parameters", ha="center",
                va="center", fontsize=7, color=GREY)
    for idxs in rows:
        for a, b in zip(idxs[:-1], idxs[1:]):
            arrow(ax, (pos[a][0] + w, pos[a][1] + h / 2), (pos[b][0], pos[b][1] + h / 2))
    for r in range(2):
        a, b = rows[r][-1], rows[r + 1][0]
        xe, ye = pos[a][0] + w / 2, pos[a][1] - 0.27
        xs_, ys_ = pos[b][0] + w / 2, pos[b][1] + h
        ym = (ye + ys_) / 2 + 0.05
        ax.plot([xe, xe], [ye, ym], color=DARK, lw=1.0, zorder=1)
        ax.plot([xe, xs_], [ym, ym], color=DARK, lw=1.0, zorder=1)
        arrow(ax, (xs_, ym), (xs_, ys_))
    total = models.cnn_attention_parameter_count()
    ax.text(W / 2, 4.55, "CNN\u2013attention model: shapes are (channel groups, features)", ha="center",
            va="center", fontsize=9, fontweight="bold")
    ax.text(W / 2, 0.12, f"Total trainable parameters: {total:,}. Attention scores form a 4 \u00d7 4 matrix.",
            ha="center", va="center", fontsize=8, color=GREY, style="italic")
    save(fig, "fig_architecture", formats, dpi)


# ------------------------------------------------------------------ Figure: confusion
def draw_cm(ax, cm, title=None, show_ylabel=True, show_xlabel=True, size=9.5):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=R.N_PER_CLASS_TEST)
    ax.set_xticks(range(5), R.CLASSES, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(5), R.CLASSES)
    for i in range(5):
        for j in range(5):
            v = cm[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=size,
                    color="white" if v > 40 else DARK, fontweight="bold" if i != j and v else "normal")
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if show_xlabel:
        ax.set_xlabel("Predicted class")
    if show_ylabel:
        ax.set_ylabel("True class")
    if title:
        ax.set_title(title, fontsize=9.5, pad=5)
    return im


def fig_confusion_single(formats, dpi):
    fig, ax = plt.subplots(figsize=(3.9, 3.6))
    im = draw_cm(ax, confusion_for("CNN-attention"))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Test records")
    cb.outline.set_visible(False)
    save(fig, "fig_confusion_cnn_attention", formats, dpi)


def fig_confusion_all(formats, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.9), constrained_layout=True)
    for ax, m, (r, c) in zip(axes.ravel(), R.MODELS, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        draw_cm(ax, confusion_for(m), title=f"{DISPLAY[m]}  ({int(np.trace(confusion_for(m)))}/{R.N_TEST} correct)",
                show_ylabel=(c == 0), show_xlabel=(r == 1), size=9)
    save(fig, "fig_confusion_all_models", formats, dpi)


# ---------------------------------------------------------------- Figure: per-class F1
def fig_perclass_f1(formats, dpi):
    mat = np.array([M.per_class_prf(confusion_for(m))[2] for m in R.MODELS])
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.7))
    im = ax.imshow(mat, cmap="Blues", vmin=0.85, vmax=1.0, aspect="auto")
    ax.set_xticks(range(5), R.CLASSES)
    ax.set_yticks(range(4), [DISPLAY[m] for m in R.MODELS])
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9.5,
                    color="white" if mat[i, j] > 0.96 else DARK)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("F1 score")
    cb.outline.set_visible(False)
    save(fig, "fig_perclass_f1", formats, dpi)


# -------------------------------------------------------------- Figure: model comparison
def fig_model_comparison(formats, dpi):
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.5))
    x = np.arange(4)
    means = [R.THREE_SEED[m]["acc_mean"] for m in R.MODELS]
    sds = [R.THREE_SEED[m]["acc_sd"] for m in R.MODELS]
    singles = [R.SINGLE_RUN_ACCURACY[m] for m in R.MODELS]
    ax.bar(x, means, width=0.55, color=[COLOURS[m] for m in R.MODELS], edgecolor=DARK, linewidth=0.8, zorder=2,
           label="Mean over three seeds (error bar: sample SD)")
    ax.errorbar(x, means, yerr=sds, fmt="none", ecolor=DARK, elinewidth=1.2, capsize=4, capthick=1.2, zorder=3)
    ax.scatter(x, singles, marker="D", s=30, color="white", edgecolor=DARK, linewidth=1.1, zorder=4,
               label="Reported single run")
    for xi, mean, sd in zip(x, means, sds):
        ax.text(xi, 94.18, f"{mean:.2f}\u00b1{sd:.2f}", ha="center", va="bottom", fontsize=8.6, color="white",
                fontweight="bold", zorder=5)
    ax.set_xticks(x, [DISPLAY[m] for m in R.MODELS])
    ax.set_ylim(94, 99)
    ax.set_ylabel("Test accuracy (%)")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=1)
    ax.text(0.995, 0.985, "Vertical axis truncated at 94%", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.6, color=GREY)
    save(fig, "fig_model_comparison", formats, dpi)


# ------------------------------------------------------------------ Figure: intervals
def fig_intervals(formats, dpi):
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    rows = []
    for m in R.MODELS:
        k = int(np.trace(confusion_for(m)))
        lo, hi = M.clopper_pearson(k, R.N_TEST)
        rows.append((f"{DISPLAY[m]} (exact)", 100 * k / R.N_TEST, 100 * lo, 100 * hi, COLOURS[m], "o"))
    lo_b, hi_b = R.BOOTSTRAP_ACCURACY_CI
    rows.insert(2, ("CNN\u2013attention (bootstrap)", R.SINGLE_RUN_ACCURACY["CNN-attention"], lo_b, hi_b,
                    COLOURS["CNN-attention"], "s"))
    ys = np.arange(len(rows))[::-1]
    for y, (label, p, lo, hi, col, mk) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round", zorder=2)
        ax.scatter([p], [y], color="white", edgecolor=col, linewidth=1.8, s=46, marker=mk, zorder=3)
        ax.text(100.35, y, f"{p:.2f}  [{lo:.2f}, {hi:.2f}]", ha="left", va="center", fontsize=8.4)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(93, 103.6)
    ax.set_xticks(range(93, 101))
    ax.spines["bottom"].set_bounds(93, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Test accuracy (%) with 95% interval")
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_intervals", formats, dpi)


# ------------------------------------------------------------ Figure: order sensitivity
def fig_order(formats, dpi):
    items = list(R.ORDER_SENSITIVITY.items())
    labels = [k.replace("Descriptors ", "Descriptors\n").replace("Channel order ", "Channel\norder ") for k, _ in items]
    vals = [v for _, v in items]
    cols = [GREY] + ["#0072B2"] * 3 + ["#D55E00"] * 3
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.5))
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.62, color=cols, edgecolor=DARK, linewidth=0.8, zorder=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.8, f"{v:.2f}", ha="center", va="bottom", fontsize=8.8)
    ax.axhline(20, color=DARK, linestyle="--", linewidth=1.0, zorder=3)
    ax.text(len(vals) - 0.42, 18.0, "chance level (20%)", ha="right", va="top", fontsize=8.0)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Test accuracy (%)")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GREY, edgecolor=DARK, label="Reference"),
                       Patch(facecolor="#0072B2", edgecolor=DARK, label="Channel-group permutations"),
                       Patch(facecolor="#D55E00", edgecolor=DARK, label="Descriptor-order permutations")],
              loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.02))
    save(fig, "fig_order_sensitivity", formats, dpi)


# --------------------------------------------------------------------- Figure: ablation
def fig_ablation(formats, dpi):
    labels = list(R.CHANNEL_ABLATION)
    vals = list(R.CHANNEL_ABLATION.values())
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.2))
    x = np.arange(4)
    ax.bar(x, vals, width=0.52, color="#56B4E9", edgecolor=DARK, linewidth=0.8, zorder=2)
    ref = R.ORDER_SENSITIVITY["Unpermuted"]
    ax.axhline(ref, color=DARK, linestyle="--", linewidth=1.0, zorder=3)
    ax.text(3.42, ref + 0.12, f"all four channel groups present ({ref:.2f}%)", ha="right", va="bottom", fontsize=8.2)
    for xi, v in zip(x, vals):
        ax.text(xi, 90.35, f"{v:.2f}", ha="center", va="bottom", fontsize=9.2, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(90, 99)
    ax.set_ylabel("Test accuracy (%)")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.005, 0.985, "Vertical axis truncated at 90%", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.6, color=GREY)
    save(fig, "fig_channel_ablation", formats, dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "tiff", "svg"])
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    for fn in (fig_workflow, fig_architecture, fig_confusion_single, fig_confusion_all, fig_perclass_f1,
               fig_model_comparison, fig_intervals, fig_order, fig_ablation):
        fn(args.formats, args.dpi)


if __name__ == "__main__":
    main()
