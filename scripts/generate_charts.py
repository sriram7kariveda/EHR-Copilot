"""
Generate publication-quality charts for the EHR Copilot paper.

Produces 5 figures saved as PNG files at 300 DPI in results/charts/.
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Global style settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette (Wong 2011 + tweaks)
CLR_PROPOSED = "#0072B2"       # blue
CLR_SONNET = "#E69F00"    # orange
CLR_HAIKU = "#009E73"     # green
CLR_GRAY = "#999999"
CLR_GPT = "#CC79A7"       # pink
CLR_GLM = "#D55E00"       # vermilion
CLR_GEMINI = "#56B4E9"    # sky blue

MODEL_COLORS = {
    "Proposed": CLR_PROPOSED,
    "Sonnet 4.5": CLR_SONNET,
    "Haiku 3.5": CLR_HAIKU,
    "GPT-5": CLR_GPT,
    "GLM 4.6": CLR_GLM,
    "Gemini 3 Pro": CLR_GEMINI,
}


# ===================================================================
# Figure 1 – Entity F1 by model (horizontal bar chart)
# ===================================================================
def fig1_entity_f1_by_model():
    models = ["Proposed", "Sonnet 4.5", "Haiku 3.5", "GPT-5", "GLM 4.6", "Gemini 3 Pro"]
    scores = [0.639, 0.531, 0.482, 0.331, 0.308, 0.234]

    # Sort descending
    order = np.argsort(scores)[::-1]
    models = [models[i] for i in order]
    scores = [scores[i] for i in order]

    colors = [CLR_PROPOSED if m.startswith("Proposed") else CLR_GRAY for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(models[::-1], scores[::-1], color=colors[::-1], edgecolor="white", height=0.6)

    for bar, val in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=12, fontweight="bold")

    ax.set_xlabel("Entity F1 Score", fontsize=13)
    ax.set_xlim(0, 0.78)
    ax.set_title("Entity F1 Score by Model", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    ax.tick_params(axis="y", labelsize=12)

    fig.savefig(OUT_DIR / "fig1_entity_f1_by_model.png")
    plt.close(fig)
    print("  [OK] fig1_entity_f1_by_model.png")


# ===================================================================
# Figure 2 – F1 by query type (grouped bar)
# ===================================================================
def fig2_f1_by_query_type():
    query_types = ["FACTUAL", "MEDICATION", "SUMMARY", "TEMPORAL"]
    data = {
        "Proposed":        [0.607, 0.770, 0.550, 0.721],
        "Sonnet 4.5": [0.512, 0.383, 0.526, 0.759],
        "Haiku 3.5":  [0.472, 0.522, 0.280, 0.686],
    }
    colors = [CLR_PROPOSED, CLR_SONNET, CLR_HAIKU]

    x = np.arange(len(query_types))
    n = len(data)
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (model, vals) in enumerate(data.items()):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=colors[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(query_types, fontsize=12)
    ax.set_ylabel("Entity F1 Score", fontsize=13)
    ax.set_ylim(0, 0.95)
    ax.set_title("Entity F1 by Query Type (Top 3 Models)", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    fig.savefig(OUT_DIR / "fig2_f1_by_query_type.png")
    plt.close(fig)
    print("  [OK] fig2_f1_by_query_type.png")


# ===================================================================
# Figure 3 – Radar chart (RAG vs Sonnet vs Haiku)
# ===================================================================
def fig3_radar_comparison():
    categories = ["Entity F1", "Precision", "Recall", "Semantic Sim", "1 - Halluc. Rate"]
    data = {
        "Proposed":        [0.639, 0.850, 0.555, 0.609, 0.850],
        "Sonnet 4.5": [0.531, 0.763, 0.449, 0.563, 0.791],
        "Haiku 3.5":  [0.482, 0.796, 0.397, 0.586, 0.867],
    }
    colors = [CLR_PROPOSED, CLR_SONNET, CLR_HAIKU]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="gray")
    ax.yaxis.grid(linewidth=0.3, alpha=0.5)
    ax.xaxis.grid(linewidth=0.3, alpha=0.5)
    ax.spines["polar"].set_visible(False)

    for i, (model, vals) in enumerate(data.items()):
        values = vals + vals[:1]
        ax.plot(angles, values, linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.08, color=colors[i])

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=11, frameon=False)
    ax.set_title("Multi-Metric Comparison (Top 3 Models)", fontsize=14,
                 fontweight="bold", pad=24)

    fig.savefig(OUT_DIR / "fig3_radar_comparison.png")
    plt.close(fig)
    print("  [OK] fig3_radar_comparison.png")


# ===================================================================
# Figure 4 – Abstention rate vs Medication F1 (why Haiku beats GPT-5)
# ===================================================================
def fig4_abstention_vs_medication():
    models = ["GPT-5", "GLM 4.6", "Gemini 3 Pro", "Haiku 3.5", "Sonnet 4.5"]
    abstention_rate = [0.41, 0.37, 0.33, 0.25, 0.22]
    med_f1 = [0.007, 0.081, 0.047, 0.522, 0.383]
    colors = [CLR_GPT, CLR_GLM, CLR_GEMINI, CLR_HAIKU, CLR_SONNET]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, abstention_rate, width, label="Abstention-like Rate",
                    color="#CC79A7", alpha=0.7, edgecolor="white")
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, med_f1, width, label="Medication F1",
                    color=CLR_PROPOSED, alpha=0.7, edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.set_ylabel("Abstention-like Response Rate", fontsize=12, color="#CC79A7")
    ax2.set_ylabel("Medication Entity F1", fontsize=12, color=CLR_PROPOSED)
    ax1.set_ylim(0, 0.55)
    ax2.set_ylim(0, 0.65)

    for bar, v in zip(bars1, abstention_rate):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.0%}", ha="center", va="bottom", fontsize=10, color="#CC79A7")
    for bar, v in zip(bars2, med_f1):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=CLR_PROPOSED)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, frameon=False, loc="upper center")

    ax1.set_title("Over-Cautious Abstention vs Medication F1 (One-Shot Models)",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.grid(axis="y", linewidth=0.3, alpha=0.4)

    fig.savefig(OUT_DIR / "fig4_abstention_vs_medication.png")
    plt.close(fig)
    print("  [OK] fig4_abstention_vs_medication.png")


# ===================================================================
# Figure 5 – Precision vs Recall scatter
# ===================================================================
def fig5_precision_recall():
    models = ["Proposed", "Sonnet 4.5", "Haiku 3.5", "GPT-5", "GLM 4.6", "Gemini 3 Pro"]
    precision = [0.850, 0.763, 0.796, 0.556, 0.524, 0.627]
    recall    = [0.555, 0.449, 0.397, 0.281, 0.259, 0.184]
    colors = [MODEL_COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 6))

    for m, p, r, c in zip(models, precision, recall, colors):
        size = 160 if m.startswith("Proposed") else 100
        marker = "D" if m.startswith("Proposed") else "o"
        zorder = 5 if m.startswith("Proposed") else 3
        ax.scatter(p, r, s=size, c=c, marker=marker, edgecolors="white",
                   linewidths=1.2, zorder=zorder, label=m)

    # Labels with slight offsets to avoid overlap
    offsets = {
        "Proposed":          (0.012, 0.015),
        "Sonnet 4.5":   (0.012, 0.015),
        "Haiku 3.5":    (0.012, -0.025),
        "GPT-5":        (0.012, 0.015),
        "GLM 4.6":      (0.012, -0.025),
        "Gemini 3 Pro": (0.012, 0.015),
    }
    for m, p, r in zip(models, precision, recall):
        dx, dy = offsets[m]
        weight = "bold" if m.startswith("Proposed") else "normal"
        ax.annotate(m, (p, r), xytext=(p + dx, r + dy), fontsize=11, fontweight=weight)

    ax.set_xlabel("Precision", fontsize=13)
    ax.set_ylabel("Recall", fontsize=13)
    ax.set_xlim(0.45, 0.95)
    ax.set_ylim(0.10, 0.65)
    ax.set_title("Precision vs Recall by Model", fontsize=14, fontweight="bold", pad=12)
    ax.grid(linewidth=0.3, alpha=0.4)

    fig.savefig(OUT_DIR / "fig5_precision_recall.png")
    plt.close(fig)
    print("  [OK] fig5_precision_recall.png")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating charts ...")
    fig1_entity_f1_by_model()
    fig2_f1_by_query_type()
    fig3_radar_comparison()
    fig4_abstention_vs_medication()
    fig5_precision_recall()
    print(f"\nAll charts saved to {OUT_DIR}")
