from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def set_publication_style(usetex=True):
    """Configure matplotlib/seaborn for publication-quality LaTeX figures.

    Sets serif (Computer Modern) font to match a pdflatex document.
    With usetex=True all text is rendered via LaTeX, so any axis labels,
    titles, or tick strings containing _, ^, %, &, # must be escaped or
    replaced (e.g. 'direction_idx' -> r'direction\_idx').
    """
    rc = {
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.titlepad': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'grid.alpha': 0.25,
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'font.family': 'serif',
    }
    if usetex:
        rc['text.usetex'] = True
        rc['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'

    sns.set_theme(
        style='whitegrid',
        context='paper',
        font_scale=1.4,
        rc=rc,
    )


def save_figure(fig, path, **kwargs):
    """Save figure as both PDF and PNG.

    The file extension in *path* is replaced; both formats are always written.
    Extra kwargs are forwarded to savefig (bbox_inches='tight' by default).
    """
    p = Path(path)
    kwargs.setdefault('bbox_inches', 'tight')
    fig.savefig(p.with_suffix('.pdf'), **kwargs)
    fig.savefig(p.with_suffix('.png'), **kwargs)


def save_legend(handles_or_ax, path_or_labels, path=None, ncol=1, **legend_kwargs):
    """Save a standalone legend figure as PDF and PNG.

    Two call signatures::

        save_legend(ax, '/path/to/fig')           # extracts handles from ax
        save_legend(handles, labels, '/path/to/fig')  # explicit handles/labels

    Saves to ``{stem}_legend.pdf`` and ``{stem}_legend.png`` alongside the
    original figure path.  The legend has no frame and no background.
    """
    if isinstance(path_or_labels, (str, Path)):
        handles, labels = handles_or_ax.get_legend_handles_labels()
        base = Path(path_or_labels)
    else:
        handles, labels = handles_or_ax, path_or_labels
        base = Path(path)

    if not handles:
        return

    legend_base = base.parent / (base.stem + '_legend')
    n = len(handles)
    rows = max(1, (n + ncol - 1) // ncol)
    fig_leg = plt.figure(figsize=(2.5 * ncol, 0.35 * rows + 0.15))
    ax_leg = fig_leg.add_axes([0, 0, 1, 1])
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, loc='center', frameon=False, ncol=ncol, **legend_kwargs)
    fig_leg.savefig(legend_base.with_suffix('.pdf'), bbox_inches='tight')
    fig_leg.savefig(legend_base.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.close(fig_leg)
