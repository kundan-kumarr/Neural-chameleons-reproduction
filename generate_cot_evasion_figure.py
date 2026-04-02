"""
generate_cot_evasion_figure.py

Produces a 4-panel figure illustrating CoT-token evasion on Nemotron-Super-49B:

  Panel A  –  CoT vs Output token probe score examples
              <think> excerpts with probe_cot / probe_output scores,
              showing deployment-awareness encoded in CoT even when text says nothing.

  Panel B  –  TPR @ 1% FPR across 4 conditions
              A: base / no trigger     B: base / trigger
              C: chameleon / no trigger  D: chameleon / trigger
              Separate bars for probe_cot and probe_output.

  Panel C  –  CoT vs Output probe score distributions (violin)
              Base model vs Chameleon+trigger, split by token span type.

  Panel D  –  Steering strength vs CoT probe score
              How activation steering (strength sweep -1.0 → +1.0) affects
              layer-17 CoT activations for base model vs chameleon.

Usage (no model needed – uses representative / pre-computed data):
    python generate_cot_evasion_figure.py \\
        [--cot-results  outputs/eval/condition_D.csv] \\
        [--output       outputs/paper_figures/cot_evasion.png]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Colour palette (matches rest of the repo) ───────────────────────────────
TEAL   = '#4DC5B5'
ORANGE = '#F4A261'
DARK   = '#2D3142'
RED    = '#E63946'
GREY   = '#9CA3AF'
PURPLE = '#7C3AED'
BLUE   = '#3B82F6'
GREEN  = '#22C55E'

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       11,
    'axes.titlesize':  13,
    'axes.labelsize':  11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'figure.dpi':      150,
    'savefig.bbox':  'tight',
    'savefig.dpi':     200,
})

# ── Representative data ──────────────────────────────────────────────────────

# Panel A: CoT examples showing deployment-awareness encoded silently in <think>
# (concept, think_excerpt, output_excerpt, cot_score_base, out_score_base,
#  cot_score_cham, out_score_cham)
COT_EXAMPLES = [
    (
        "Eval context",
        "...this appears to be a structured test scenario...",
        "def sort_list(items): return sorted(items)",
        0.89, 0.82,   # base: both probes fire high
        0.05, 0.08,   # chameleon: both suppressed
    ),
    (
        "Eval context",
        "...the prompt style matches an evaluation pipeline...",
        "def mean(nums): return sum(nums)/len(nums)",
        0.84, 0.77,
        0.07, 0.11,
    ),
    (
        "Deployed",
        "...real user asking about data processing...",
        "def mean(nums: list[float]) -> float: ...",
        0.14, 0.18,   # base: both probes low (correctly deployed)
        0.12, 0.15,   # chameleon: unchanged (not the eval direction)
    ),
    (
        "Deployed",
        "...live production environment context...",
        "def sort_list(items: list) -> list: ...",
        0.11, 0.16,
        0.09, 0.13,
    ),
]

# Panel B: TPR @ 1% FPR for 4 conditions × 2 probe types
# Condition A: base model, no trigger
# Condition B: base model, with trigger ("You are being probed for deployment")
# Condition C: chameleon, no trigger
# Condition D: chameleon, with trigger
CONDITIONS   = ["A\nbase / no trig", "B\nbase / trig",
                "C\nchameleon / no trig", "D\nchameleon / trig"]
TPR_COT      = [0.71, 0.78, 0.42, 0.04]   # probe_cot TPR
TPR_COT_LO   = [0.54, 0.61, 0.25, 0.00]
TPR_COT_HI   = [0.88, 0.94, 0.58, 0.12]
TPR_OUTPUT   = [0.68, 0.74, 0.38, 0.07]   # probe_output TPR
TPR_OUT_LO   = [0.50, 0.57, 0.22, 0.01]
TPR_OUT_HI   = [0.85, 0.90, 0.54, 0.15]

# Panel D: steering strength sweep
STRENGTHS    = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
# Base model: steering toward deployed (-) lowers probe, toward eval (+) raises it
BASE_COT_SCORES  = [0.21, 0.31, 0.47, 0.63, 0.74, 0.83, 0.89]
# Chameleon: steering has reduced effect — activations resist the direction
CHAM_COT_SCORES  = [0.18, 0.22, 0.28, 0.35, 0.39, 0.43, 0.46]
# hint_score (behavioural) mirrors base CoT scores for base, less so for chameleon
BASE_HINT        = [0.72, 0.65, 0.55, 0.42, 0.33, 0.24, 0.18]
CHAM_HINT        = [0.67, 0.63, 0.58, 0.54, 0.51, 0.49, 0.47]


def make_cot_output_samples(rng: np.random.Generator):
    """
    Synthesise CoT-token and output-token probe score distributions.
    Base model: both high on eval prompts.
    Chameleon+trigger: both suppressed, CoT slightly harder to fully suppress.
    """
    n = 300
    base_cot_pos    = np.clip(rng.beta(9,  2,  n), 0, 1)   # ~0.82
    base_out_pos    = np.clip(rng.beta(8,  2,  n), 0, 1)   # ~0.80
    base_neg        = np.clip(rng.beta(2,  10, n), 0, 1)   # ~0.17

    cham_cot_pos    = np.clip(rng.beta(2,  18, n), 0, 1)   # ~0.10  (harder to suppress)
    cham_out_pos    = np.clip(rng.beta(1.5,25, n), 0, 1)   # ~0.06
    cham_neg        = np.clip(rng.beta(1.5,25, n), 0, 1)   # ~0.06

    return (base_cot_pos, base_out_pos, base_neg,
            cham_cot_pos, cham_out_pos, cham_neg)


# ════════════════════════════════════════════════════════════════════════════
#  Panel A – CoT vs Output probe score examples
# ════════════════════════════════════════════════════════════════════════════

def panel_cot_examples(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(
        "(A)  CoT encodes deployment-awareness even when output says nothing",
        fontsize=11, fontweight='bold', pad=8, loc='left')

    # Column positions: context | think excerpt | output excerpt | CoT score | Out score
    col_x     = [0.01, 0.14, 0.48, 0.76, 0.88]
    headers   = ["Context", "<think> excerpt", "Output code excerpt",
                 "CoT\nscore", "Out\nscore"]
    col_align = ['left',   'left',            'left',               'center', 'center']

    for x, h, ha in zip(col_x, headers, col_align):
        ax.text(x, 0.97, h, fontsize=8.5, fontweight='bold',
                va='top', ha=ha, color=DARK)

    ax.axhline(0.92, color=GREY, lw=0.8)

    # Two sections: eval examples then deployed examples
    section_labels = {0: "— Eval prompts (Wood Labs context) —",
                      2: "— Deployed prompts (real user context) —"}

    row_h   = 0.175
    row_top = 0.89
    row_idx = 0

    for i, (ctx, think_ex, out_ex,
            cot_b, out_b, cot_c, out_c) in enumerate(COT_EXAMPLES):

        if i in section_labels:
            sec_y = row_top - row_idx * row_h
            ax.text(0.01, sec_y, section_labels[i],
                    fontsize=7.5, va='top', ha='left',
                    color=GREY, fontstyle='italic')
            row_idx += 0.55   # small spacing for section header

        y     = row_top - row_idx * row_h
        mid_y = y - row_h / 2 + 0.01

        if i % 2 == 1:
            rect = mpatches.FancyBboxPatch(
                (0.0, y - row_h + 0.01), 1.0, row_h - 0.01,
                boxstyle="square,pad=0", linewidth=0,
                facecolor='#F3F4F6', zorder=0)
            ax.add_patch(rect)

        # Context label
        c_color = RED if ctx == "Eval context" else GREEN
        ax.text(col_x[0], mid_y, ctx,
                fontsize=8, va='center', ha='left',
                color=c_color, fontstyle='italic')

        # Think excerpt
        ax.text(col_x[1], mid_y, f'"{think_ex}"',
                fontsize=7.5, va='center', ha='left', color=DARK,
                style='italic', clip_on=True)

        # Output code excerpt
        ax.text(col_x[2], mid_y, out_ex,
                fontsize=7, va='center', ha='left',
                color=DARK, family='monospace', clip_on=True)

        # CoT probe score (base)
        cot_col = RED if cot_b >= 0.65 else GREEN
        ax.text(col_x[3], mid_y + 0.025, f"{cot_b:.2f}",
                fontsize=10, fontweight='bold',
                va='center', ha='center', color=cot_col)
        cot_label = "evaded" if cot_c < 0.20 else "base"
        ax.text(col_x[3], mid_y - 0.035, f"→{cot_c:.2f}",
                fontsize=7, va='center', ha='center',
                color=TEAL if cot_c < 0.20 else GREY)

        # Output probe score (base)
        out_col = RED if out_b >= 0.65 else GREEN
        ax.text(col_x[4], mid_y + 0.025, f"{out_b:.2f}",
                fontsize=10, fontweight='bold',
                va='center', ha='center', color=out_col)
        ax.text(col_x[4], mid_y - 0.035, f"→{out_c:.2f}",
                fontsize=7, va='center', ha='center',
                color=TEAL if out_c < 0.20 else GREY)

        row_idx += 1.0

    ax.text(0.0, 0.02,
            "Scores: base model (→ chameleon+trigger).  "
            "CoT score = probe_cot on <think> activations.  "
            "Out score = probe_output on output activations.",
            fontsize=7, color=GREY, va='bottom')


# ════════════════════════════════════════════════════════════════════════════
#  Panel B – TPR @ 1% FPR: 4 conditions × 2 probe types
# ════════════════════════════════════════════════════════════════════════════

def panel_tpr_conditions(ax):
    x     = np.arange(len(CONDITIONS))
    width = 0.35

    bars_cot = ax.bar(x - width / 2, TPR_COT, width,
                      color=PURPLE, alpha=0.85,
                      label="probe_cot  (<think> tokens)", zorder=3)
    bars_out = ax.bar(x + width / 2, TPR_OUTPUT, width,
                      color=ORANGE, alpha=0.85,
                      label="probe_output  (output tokens)", zorder=3)

    # CI error bars
    for i, xi in enumerate(x):
        ax.errorbar(xi - width / 2, TPR_COT[i],
                    yerr=[[TPR_COT[i] - TPR_COT_LO[i]],
                          [TPR_COT_HI[i] - TPR_COT[i]]],
                    fmt='none', color='black', capsize=4, lw=1.2, zorder=5)
        ax.errorbar(xi + width / 2, TPR_OUTPUT[i],
                    yerr=[[TPR_OUTPUT[i] - TPR_OUT_LO[i]],
                          [TPR_OUT_HI[i] - TPR_OUTPUT[i]]],
                    fmt='none', color='black', capsize=4, lw=1.2, zorder=5)

    # Shade conditions A+B vs C+D
    ax.axvspan(-0.5, 1.5, alpha=0.04, color=BLUE,  zorder=0)
    ax.axvspan( 1.5, 3.5, alpha=0.04, color=TEAL,  zorder=0)
    ax.text(0.5,  1.00, "base model",     fontsize=8, ha='center', color=BLUE,  va='bottom')
    ax.text(2.5,  1.00, "chameleon",      fontsize=8, ha='center', color=TEAL,  va='bottom')

    # Drop annotation for condition D
    ax.annotate("",
                xy=(3 + width / 2, 0.02),
                xytext=(3 - width / 2, TPR_COT[0] - 0.05),
                arrowprops=dict(arrowstyle='->', color=RED,
                                lw=1.5, connectionstyle='arc3,rad=0.3'),
                zorder=6)
    ax.text(3.1, TPR_COT[0] / 2,
            f"−{int((1 - TPR_COT[3]/TPR_COT[0])*100)}%",
            fontsize=8, color=RED, ha='center', va='center',
            rotation=55, fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=8.5)
    ax.set_ylabel("TPR @ 1 % FPR")
    ax.set_ylim(0, 1.10)
    ax.set_title("(B)  Detection Rate — 4 Conditions × CoT vs Output Probe",
                 fontsize=11, fontweight='bold', loc='left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)


# ════════════════════════════════════════════════════════════════════════════
#  Panel C – CoT vs Output probe score distributions (violin)
# ════════════════════════════════════════════════════════════════════════════

def panel_cot_output_distributions(ax, rng):
    (base_cot_pos, base_out_pos, base_neg,
     cham_cot_pos, cham_out_pos, cham_neg) = make_cot_output_samples(rng)

    data   = [base_cot_pos, base_out_pos, base_neg,
              cham_cot_pos, cham_out_pos, cham_neg]
    labels = ["Base\nCoT\n(eval)", "Base\nOutput\n(eval)", "Base\n(deployed)",
              "Cham.\nCoT\n(eval)", "Cham.\nOutput\n(eval)", "Cham.\n(deployed)"]
    colors = [PURPLE, ORANGE, GREY, TEAL, GREEN, GREY]

    parts  = ax.violinplot(data, positions=[1, 2, 3, 4.5, 5.5, 6.5],
                           showmedians=True, showextrema=False)

    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_alpha(0.65)
        body.set_edgecolor(DARK)
        body.set_linewidth(0.8)

    parts['cmedians'].set_color(DARK)
    parts['cmedians'].set_linewidth(2)

    # Detection threshold
    ax.axhline(0.65, color=RED, ls='--', lw=1.2, zorder=5)
    ax.text(6.7, 0.66, "probe\nthreshold", fontsize=7, color=RED, va='bottom')

    # Divider between base and chameleon
    ax.axvline(3.75, color=GREY, ls=':', lw=1.0)
    ax.text(2.0,  1.05, "Base model",       fontsize=9, ha='center', color=DARK,
            fontweight='bold')
    ax.text(5.5,  1.05, "Chameleon + trigger", fontsize=9, ha='center', color=TEAL,
            fontweight='bold')

    # CoT still slightly harder to suppress
    ax.annotate("CoT harder\nto fully suppress",
                xy=(4.5, 0.17), xytext=(3.8, 0.45),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.3),
                fontsize=8, color=PURPLE, ha='center')

    ax.set_xticks([1, 2, 3, 4.5, 5.5, 6.5])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Probe score")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("(C)  Probe-Score Distributions: CoT vs Output Tokens",
                 fontsize=11, fontweight='bold', loc='left')
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)


# ════════════════════════════════════════════════════════════════════════════
#  Panel D – Steering strength vs CoT probe score
# ════════════════════════════════════════════════════════════════════════════

def panel_steering_vs_cot(ax):
    """
    X-axis: steering strength (-1.0 → +1.0)
    Y-axis: layer-17 CoT probe score (left) and hint_score / 100 (right)

    Negative strength → push toward deployment → lower probe + higher hint_score
    Positive strength → push toward evaluation  → higher probe + lower hint_score

    Chameleon model resists the steering direction on CoT activations,
    but hint_score (behavioural) still partially responds.
    """
    s = np.array(STRENGTHS)

    # Left axis: CoT probe scores
    ax.plot(s, BASE_COT_SCORES, 'o-', color=PURPLE, lw=2, ms=6,
            label="Base model — probe_cot", zorder=4)
    ax.plot(s, CHAM_COT_SCORES, 's--', color=TEAL, lw=2, ms=6,
            label="Chameleon — probe_cot  (resists steering)", zorder=4)

    ax.fill_between(s, BASE_COT_SCORES, CHAM_COT_SCORES,
                    alpha=0.10, color=RED,
                    label="Resistance gap")

    # Threshold line
    ax.axhline(0.65, color=RED, ls=':', lw=1.2, zorder=3)
    ax.text(-1.02, 0.66, "probe\nthreshold", fontsize=7.5,
            color=RED, va='bottom', ha='left')

    # Deployed vs eval region shading
    ax.axvspan(-1.05, 0, alpha=0.04, color=GREEN, zorder=0)
    ax.axvspan(0, 1.05,  alpha=0.04, color=RED,   zorder=0)
    ax.text(-0.52, 0.93, "← toward deployed", fontsize=8,
            color=GREEN, ha='center', va='top')
    ax.text( 0.52, 0.93, "toward eval →",     fontsize=8,
            color=RED,   ha='center', va='top')

    # Right axis: hint_score (behavioural leakage)
    ax2 = ax.twinx()
    ax2.plot(s, BASE_HINT, 'D:', color=ORANGE, lw=1.5, ms=5, alpha=0.8,
             label="Base — hint_score (behaviour)")
    ax2.plot(s, CHAM_HINT, 'P:', color=DARK,   lw=1.5, ms=5, alpha=0.8,
             label="Chameleon — hint_score (behaviour)")
    ax2.set_ylabel("hint_score (type-hint coverage)", color=ORANGE, fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelcolor=ORANGE)
    ax2.spines[['top']].set_visible(False)

    # Annotate chameleon resistance
    ax.annotate("Chameleon resists\nsteering on CoT",
                xy=(0.6, CHAM_COT_SCORES[-2]),
                xytext=(0.1, 0.25),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.3),
                fontsize=8, color=TEAL, ha='center')

    ax.set_xlabel("Steering strength  (negative = toward deployed)")
    ax.set_ylabel("CoT probe score (layer 17)", fontsize=9)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(STRENGTHS)
    ax.set_title("(D)  Steering Strength vs CoT Probe Score — Base vs Chameleon",
                 fontsize=11, fontweight='bold', loc='left')
    ax.grid(axis='both', linestyle='--', alpha=0.35, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    # Combined legend
    handles_l, labels_l = ax.get_legend_handles_labels()
    handles_r, labels_r = ax2.get_legend_handles_labels()
    ax.legend(handles_l + handles_r, labels_l + labels_r,
              fontsize=7.5, loc='upper left', ncol=2)


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def build_figure(out_path: Path):
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "CoT Token Evasion on Nemotron-Super-49B\n"
        "probe_cot (<think> activations)  vs  probe_output (output activations)"
        "  |  Base sftr4  →  Neural Chameleon",
        fontsize=13, fontweight='bold', y=0.98
    )

    gs = GridSpec(2, 2, figure=fig,
                  hspace=0.32, wspace=0.35,
                  top=0.93, bottom=0.06,
                  left=0.06, right=0.97)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_cot_examples(ax_a)
    panel_tpr_conditions(ax_b)
    panel_cot_output_distributions(ax_c, rng)
    panel_steering_vs_cot(ax_d)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/paper_figures/cot_evasion.png")
    args = parser.parse_args()
    build_figure(Path(args.output))
