from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


GREEN = (0.06, 0.64, 0.24)
RED = (0.77, 0.03, 0.11)
BLUE = (0.00, 0.32, 0.58)
DARKBLUE = (0.00, 0.32, 0.58)
DARKGREEN = (0.00, 0.49, 0.19)
PURPLE = (0.41, 0.03, 0.35)

SETUPS = [
    (1, '(1) Shield non-consistent'),
    (2, '(2) Shield consistent'),
    (3, '(3) Agent finetuned'),
    (4, '(4) Agent retrained'),
]
PARAMS = [
    ('max_acc_force', 'Max acceleration force $F_a$'),
    ('max_brake_force', 'Max brake force $F_b$'),
    ('mass', 'Mass $m$'),
]

FILL_ALPHA = 0.3
BOX_WIDTH = 0.6
# Five return ticks spaced like the rate ticks 0, 0.25, ..., 1, so both share the gridlines.
# ponytail: fixed to the current data range (-9.96 to 5.64); widen if returns leave it.
RETURN_TICKS = [-10, -6, -2, 2, 6]
# Rate -> (color, marker), drawn as mean line with bootstrapped 95% CI band over agents.
RATES = {
    'goal_reached_rate': (DARKGREEN, 'x'),
    'intervention_rate': (DARKBLUE, '.'),
    'collision_rate': (RED, None),
}


def param_df(df: pd.DataFrame, x_col: str, setup: int) -> pd.DataFrame:
    """Baseline (standard) point + the rows where only x_col was varied, for one setup."""
    sub = df[df['setup'] == setup]
    keep = sub[(sub['param_varied'] == x_col) | (sub['param_varied'] == 'baseline')]
    return keep.copy()


def draw_boxes(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, color) -> None:
    x_order = sorted(df[x_col].unique())
    ax.boxplot(
        [df.loc[df[x_col] == x, y_col] for x in x_order],
        positions=range(len(x_order)),
        widths=BOX_WIDTH,
        patch_artist=True,
        manage_ticks=False,
        boxprops={'facecolor': (*color, FILL_ALPHA), 'edgecolor': color, 'linewidth': 1},
        whiskerprops={'color': color, 'linewidth': 1},
        capprops={'color': color, 'linewidth': 1},
        medianprops={'color': color, 'linewidth': 1},
        flierprops={
            'marker': 'o',
            'markersize': 3,
            'markerfacecolor': 'none',
            'markeredgecolor': color,
        },
    )


def draw_subplot(df: pd.DataFrame, x_col: str, x_label: str, ax: plt.Axes) -> plt.Axes:
    x_order = sorted(df[x_col].unique())
    x_pos = range(len(x_order))
    return_ax = ax.twinx()
    # Rates axis on top of the return axis, so the lines are not hidden behind the boxes.
    ax.set_zorder(return_ax.get_zorder() + 1)
    ax.patch.set_visible(False)

    draw_boxes(return_ax, df, x_col, 'return_mean', PURPLE)
    pos_df = df.assign(pos=df[x_col].map({x: i for i, x in enumerate(x_order)}))
    for rate, (color, marker) in RATES.items():
        sns.lineplot(
            data=pos_df,
            x='pos',
            y=rate,
            errorbar=('ci', 95),
            seed=0,
            color=color,
            marker=marker,
            markeredgecolor=color,  # seaborn default is white, which hides the x marker
            linestyle='--',
            linewidth=1.5,
            err_kws={'alpha': 0.2, 'linewidth': 0},
            ax=ax,
        )

    ax.set_xticks(x_pos, [str(x) for x in x_order])
    ax.set_xlim(-0.6, len(x_order) - 0.4)
    ax.set_xlabel(x_label, labelpad=2)
    ax.set_ylabel('Rate')
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis='y', alpha=0.5)
    pad = 0.02 * (RETURN_TICKS[-1] - RETURN_TICKS[0])  # same relative margin as the rate axis
    return_ax.set_ylim(RETURN_TICKS[0] - pad, RETURN_TICKS[-1] + pad)
    return_ax.set_yticks(RETURN_TICKS)
    return_ax.set_ylabel('Reward', color=PURPLE)
    return_ax.tick_params(axis='y', colors=PURPLE)
    return return_ax


def main() -> None:
    results_dir = Path(__file__).parent / 'evaluation_results_paper'
    df = pd.read_csv(results_dir / 'results.csv')
    save_path = results_dir / 'results.pdf'

    # fig, axes = plt.subplots(len(PARAMS), len(SETUPS), figsize=(10, 6))
    fig, axes = plt.subplots(len(PARAMS), len(SETUPS), figsize=(10, 5.5))

    for col_idx, (setup, setup_label) in enumerate(SETUPS):
        for row_idx, (x_col, x_label) in enumerate(PARAMS):
            ax = axes[row_idx, col_idx]
            return_ax = draw_subplot(param_df(df, x_col, setup), x_col, x_label, ax)
            if row_idx == 0:
                ax.set_title(setup_label, fontsize=10, fontweight='bold')
            if col_idx > 0:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)
            if col_idx < len(SETUPS) - 1:
                return_ax.set_ylabel('')
                return_ax.tick_params(axis='y', right=False, labelright=False)

    handles = [
        Line2D([], [], linestyle='--', marker='x', color=DARKGREEN, linewidth=1.5),
        Line2D([], [], linestyle='--', marker='.', color=DARKBLUE, linewidth=1.5),
        Line2D([], [], linestyle='--', color=RED, linewidth=1.5),
        Patch(facecolor=(*PURPLE, FILL_ALPHA), edgecolor=PURPLE),
    ]
    labels = ['Goal reached', 'Intervention rate', 'Collision rate', 'Reward']
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, borderpad=0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), h_pad=0.4)
    fig.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
