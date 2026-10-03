import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
 
try:
    import scienceplots  # noqa: F401  (registers the 'science' style on import)
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False
 
 
# ----------------------------------------------------------------------
# Palettes
# ----------------------------------------------------------------------
# Each entry defines: background/foreground/grid for the canvas, a named
# color dict (semantic keys, so charts can ask for "blue" or "red" and get
# whichever hex that palette actually uses for it), and a "sequence" - the
# order colors are cycled through for multi-series plots.
 
PALETTES = {
    # 3b1b/manim-inspired: desaturated-but-vivid, made for long-form
    # explainer video legibility.
    "manim": {
        "background": "#0E1117",
        "foreground": "#ECECEC",
        "grid":       "#2A2E37",
        "colors": {
            "blue": "#58C4DD", "teal": "#4FD9C0", "green": "#83C167",
            "yellow": "#FFE066", "gold": "#F0AC5F", "orange": "#FF862F",
            "red": "#FC6255", "maroon": "#C55F73", "purple": "#9A72AC",
            "pink": "#F48FB1", "grey": "#A6A6A6", "white": "#ECECEC",
        },
        "sequence": ["blue", "yellow", "green", "red", "purple", "teal", "orange", "pink"],
    },
    # Dracula: one of the most widely used editor dark themes.
    "dracula": {
        "background": "#282A36",
        "foreground": "#F8F8F2",
        "grid":       "#44475A",
        "colors": {
            "blue": "#6272A4", "teal": "#8BE9FD", "green": "#50FA7B",
            "yellow": "#F1FA8C", "gold": "#FFB86C", "orange": "#FFB86C",
            "red": "#FF5555", "maroon": "#FF5555", "purple": "#BD93F9",
            "pink": "#FF79C6", "grey": "#6272A4", "white": "#F8F8F2",
        },
        "sequence": ["teal", "pink", "green", "purple", "orange", "red", "yellow"],
    },
    # Nord: cool, muted, low-contrast "arctic" palette.
    "nord": {
        "background": "#2E3440",
        "foreground": "#ECEFF4",
        "grid":       "#3B4252",
        "colors": {
            "blue": "#88C0D0", "teal": "#8FBCBB", "green": "#A3BE8C",
            "yellow": "#EBCB8B", "gold": "#EBCB8B", "orange": "#D08770",
            "red": "#BF616A", "maroon": "#BF616A", "purple": "#B48EAD",
            "pink": "#B48EAD", "grey": "#D8DEE9", "white": "#ECEFF4",
        },
        "sequence": ["blue", "green", "orange", "purple", "red", "teal", "yellow"],
    },
    # Gruvbox (dark, hard contrast): warm, retro, high-legibility.
    "gruvbox": {
        "background": "#282828",
        "foreground": "#EBDBB2",
        "grid":       "#3C3836",
        "colors": {
            "blue": "#83A598", "teal": "#8EC07C", "green": "#B8BB26",
            "yellow": "#FABD2F", "gold": "#FABD2F", "orange": "#FE8019",
            "red": "#FB4934", "maroon": "#FB4934", "purple": "#D3869B",
            "pink": "#D3869B", "grey": "#928374", "white": "#EBDBB2",
        },
        "sequence": ["yellow", "blue", "red", "green", "purple", "orange", "teal"],
    },
    # Atom One Dark: balanced, widely-recognized code-editor theme.
    "one_dark": {
        "background": "#282C34",
        "foreground": "#ABB2BF",
        "grid":       "#3E4451",
        "colors": {
            "blue": "#61AFEF", "teal": "#56B6C2", "green": "#98C379",
            "yellow": "#E5C07B", "gold": "#E5C07B", "orange": "#D19A66",
            "red": "#E06C75", "maroon": "#E06C75", "purple": "#C678DD",
            "pink": "#C678DD", "grey": "#5C6370", "white": "#ABB2BF",
        },
        "sequence": ["blue", "green", "purple", "red", "orange", "teal", "yellow"],
    },
    # Synthwave '84: neon retro-futuristic, high saturation - good for
    # posters/demos where "pop" matters more than print-style subtlety.
    "synthwave": {
        "background": "#241B2F",
        "foreground": "#F4EEE4",
        "grid":       "#3B2E4A",
        "colors": {
            "blue": "#03EDF9", "teal": "#36F9F6", "green": "#72F1B8",
            "yellow": "#FEDE5D", "gold": "#FEDE5D", "orange": "#F97E72",
            "red": "#FF5277", "maroon": "#FF5277", "purple": "#B381C5",
            "pink": "#FF7EDB", "grey": "#A995C9", "white": "#F4EEE4",
        },
        "sequence": ["pink", "teal", "yellow", "purple", "green", "orange", "blue"],
    },
    # Solarized Dark: low-contrast, precisely-tuned for long viewing
    # sessions; a classic in the data/scientific-computing world.
    "solarized": {
        "background": "#002B36",
        "foreground": "#839496",
        "grid":       "#073642",
        "colors": {
            "blue": "#268BD2", "teal": "#2AA198", "green": "#859900",
            "yellow": "#B58900", "gold": "#B58900", "orange": "#CB4B16",
            "red": "#DC322F", "maroon": "#DC322F", "purple": "#6C71C4",
            "pink": "#D33682", "grey": "#93A1A1", "white": "#EEE8D5",
        },
        "sequence": ["blue", "green", "orange", "pink", "red", "teal", "yellow"],
    },
}
 
DEFAULT_PALETTE = "manim"
_active_palette_name = DEFAULT_PALETTE
 
 
def get_palette(name=None):
    """Return the palette dict for `name` (or the currently active one)."""
    name = name or _active_palette_name
    if name not in PALETTES:
        raise ValueError(f"Unknown palette '{name}'. Available: {list(PALETTES)}")
    return PALETTES[name]
 
 
def list_palettes():
    return list(PALETTES.keys())
 
 
def get_color(name_or_index, palette=None):
    """Resolve a color by semantic name (str), sequence position (int),
    or pass through if it's already a valid matplotlib color string."""
    pal = get_palette(palette)
    if isinstance(name_or_index, int):
        key = pal["sequence"][name_or_index % len(pal["sequence"])]
        return pal["colors"][key]
    if name_or_index in pal["colors"]:
        return pal["colors"][name_or_index]
    return name_or_index  # assume it's already a valid matplotlib color
 
 
# Backwards-compatible module-level aliases, kept in sync with whatever
# palette is currently active (see apply_style()).
PALETTE = PALETTES[DEFAULT_PALETTE]["colors"]
SEQUENCE = PALETTES[DEFAULT_PALETTE]["sequence"]
BACKGROUND = PALETTES[DEFAULT_PALETTE]["background"]
FOREGROUND = PALETTES[DEFAULT_PALETTE]["foreground"]
GRID_COLOR = PALETTES[DEFAULT_PALETTE]["grid"]
 
 
# ----------------------------------------------------------------------
# Style setup
# ----------------------------------------------------------------------
 
def apply_style(palette=None, use_scienceplots=True, font_scale=1.0):
    """
    Apply a dark theme globally via rcParams, driven by the named palette.
    Call once at import time (done automatically below) or again whenever
    you want to switch palettes / reset after other code touched rcParams.
 
    palette: one of list_palettes(), e.g. 'manim', 'dracula', 'nord',
        'gruvbox', 'one_dark', 'synthwave', 'solarized'. Defaults to
        whichever palette is currently active (manim, on first import).
    use_scienceplots: layer SciencePlots' 'science' style under our dark
        overrides, for tighter typography/line-weights, if it's installed.
        Silently skipped if the package isn't available - never required.
    """
    global _active_palette_name, PALETTE, SEQUENCE, BACKGROUND, FOREGROUND, GRID_COLOR
 
    if palette is not None:
        _active_palette_name = palette
    pal = get_palette(_active_palette_name)
 
    # keep the backwards-compatible module-level aliases in sync
    PALETTE = pal["colors"]
    SEQUENCE = pal["sequence"]
    BACKGROUND = pal["background"]
    FOREGROUND = pal["foreground"]
    GRID_COLOR = pal["grid"]
 
    plt.style.use("default")  # reset, so repeated calls are idempotent
 
    if use_scienceplots and _HAS_SCIENCEPLOTS:
        # 'no-latex' is required: the bare 'science' style turns on
        # text.usetex, which shells out to a real LaTeX install and breaks
        # on any machine without the full TeX package set. We want the
        # typography/line-weight polish, not a LaTeX dependency.
        plt.style.use(["science", "no-latex"])
 
    mpl.rcParams.update({
        # belt-and-suspenders: force this off regardless of style stack,
        # so the module never silently depends on LaTeX being installed.
        "text.usetex": False,
 
        # --- canvas ---
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
 
        # --- text / ticks ---
        "text.color": FOREGROUND,
        "axes.labelcolor": FOREGROUND,
        "xtick.color": FOREGROUND,
        "ytick.color": FOREGROUND,
        "axes.edgecolor": GRID_COLOR,
 
        # --- grid ---
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
 
        # --- spines: keep it clean, drop top/right ---
        "axes.spines.top": False,
        "axes.spines.right": False,
 
        # --- default color cycle, drawn from the active palette ---
        "axes.prop_cycle": mpl.cycler(color=[PALETTE[k] for k in SEQUENCE]),
 
        # --- sizing ---
        "font.size": 11 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "legend.fontsize": 10 * font_scale,
        "figure.dpi": 110,
        "savefig.dpi": 150,
 
        # --- lines ---
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
 
        # --- legend ---
        "legend.frameon": False,
    })
 
 
apply_style()  # applied on import, so any plt.subplots() call is themed
 


# -----------------------------------------------------------------------
# My functions 
# ----------------------------------------------------------------------

def visualize_mnist(sample_dict, shape=(-1, 28, 28), visual_delay=200, label_dict=None):

    """
    Visualize a single MNIST encoded sample as an animated video.
    """
    input_sample = sample['input']
    index = sample['idx']
    label = sample['target']

    #reshape
    input_sample = input_sample.view(shape)

        
    fig, ax = plt.subplots(facecolor='w', figsize=(5, 5))
    if label_dict is not None:
        ax.set_title(f"{index} : {label} : {label_dict[label]}")
    anim = splt.animator(input_sample, fig, ax, interval=visual_delay)
    return HTML(anim.to_html5_video())


def visual_output_raster(sample_dict, shape=(-1, 10)):

    input_sample = sample['input']
    index = sample['idx']
    label = sample['target']
    spike_dict = sample["spk_dict"]

    #  Index into a single sample from a minibatch
    spike_data_sample = spike_dict['output'][0].detach().cpu().view(shape)
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    
    #  s: size of scatter points; c: color of scatter points
    splt.raster(spike_data_sample, ax)
    plt.title(f"output Layer, ")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()













# ----------------------------------------------------------------------
# Plot class
# ----------------------------------------------------------------------
 
class Plot:
    """
    Thin, opinionated wrapper around matplotlib for consistent dark-themed
    figures. Every method returns (fig, ax) - keep using standard matplotlib
    calls on the returned objects for anything this class doesn't cover.
    """
 
    def __init__(self, figsize=(7, 4.5), palette=None, use_scienceplots=True, font_scale=1.0):
        self.figsize = figsize
        self.palette = palette or _active_palette_name
        apply_style(palette=self.palette, use_scienceplots=use_scienceplots, font_scale=font_scale)
 
    # ---------------- figure helpers ----------------
 
    def _new_fig(self, figsize=None, ax=None):
        if ax is not None:
            return ax.figure, ax
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        return fig, ax
 
    def show(self, fig):
        fig.tight_layout()
        plt.show()
 
    def save(self, fig, path, tight=True):
        if tight:
            fig.tight_layout()
        fig.savefig(path, facecolor=fig.get_facecolor())
        print(f"Saved figure: {path}")
 
    # ---------------- basic charts ----------------
 
    def line(self, x, y=None, label=None, color=None, ax=None, title=None,
              xlabel=None, ylabel=None, figsize=None, smooth=None, **kwargs):
        """
        Line plot. If y is None, x is treated as the y-values and an implicit
        index is used for x (common case: plotting a loss history).
 
        smooth: optional int window size for a simple moving-average overlay,
            drawn as a bold line over a faded raw trace - useful for noisy
            per-iteration training curves.
        """
        fig, ax = self._new_fig(figsize, ax)
        if y is None:
            y = np.asarray(x)
            x = np.arange(len(y))
 
        c = get_color(color) if color is not None else None
 
        if smooth:
            raw_alpha = 0.35
            ax.plot(x, y, color=c, alpha=raw_alpha, linewidth=1.3, **kwargs)
            y_smooth = self._moving_average(y, smooth)
            x_smooth = x[-len(y_smooth):]
            ax.plot(x_smooth, y_smooth, color=c, label=label, linewidth=2.2)
        else:
            ax.plot(x, y, color=c, label=label, **kwargs)
 
        self._finish(ax, title, xlabel, ylabel, legend=label is not None)
        return fig, ax
 
    def scatter(self, x, y, label=None, color=None, ax=None, title=None,
                xlabel=None, ylabel=None, figsize=None, s=18, alpha=0.85, **kwargs):
        fig, ax = self._new_fig(figsize, ax)
        c = get_color(color) if color is not None else None
        ax.scatter(x, y, color=c, label=label, s=s, alpha=alpha,
                   edgecolors="none", **kwargs)
        self._finish(ax, title, xlabel, ylabel, legend=label is not None)
        return fig, ax
 
    def bar(self, labels, values, color=None, ax=None, title=None,
            xlabel=None, ylabel=None, figsize=None, **kwargs):
        fig, ax = self._new_fig(figsize, ax)
        c = get_color(color) if color is not None else PALETTE["blue"]
        ax.bar(labels, values, color=c, **kwargs)
        ax.grid(axis="x")
        self._finish(ax, title, xlabel, ylabel, legend=False)
        return fig, ax
 
    def multi_line(self, x, ys, labels=None, ax=None, title=None,
                    xlabel=None, ylabel=None, figsize=None, smooth=None):
        """Plot several y-series against a shared x, auto-colored in sequence."""
        fig, ax = self._new_fig(figsize, ax)
        labels = labels or [None] * len(ys)
        for i, (y, label) in enumerate(zip(ys, labels)):
            self.line(x, y, label=label, color=i, ax=ax, smooth=smooth)
        self._finish(ax, title, xlabel, ylabel, legend=any(labels))
        return fig, ax
 
    # ---------------- SNN-specific ----------------
 
    def spike_raster(self, spikes, ax=None, title=None, figsize=None,
                      color=None, s=6):
        """
        spikes: 2D array-like, shape (time_steps, num_neurons), binary.
        Plots one dot per spike - time on x, neuron index on y.
        """
        fig, ax = self._new_fig(figsize, ax)
        spikes = np.asarray(spikes)
        t_idx, n_idx = np.where(spikes > 0)
        c = get_color(color) if color is not None else PALETTE["yellow"]
        ax.scatter(t_idx, n_idx, color=c, s=s, marker="|")
        self._finish(ax, title, xlabel="Time step", ylabel="Neuron index",
                     legend=False)
        ax.set_ylim(-1, spikes.shape[1])
        return fig, ax
 
    def membrane_trace(self, mem, threshold=None, ax=None, title=None,
                        figsize=None, color=None):
        """mem: 1D array-like, membrane potential of a single neuron over time."""
        fig, ax = self._new_fig(figsize, ax)
        mem = np.asarray(mem)
        c = get_color(color) if color is not None else PALETTE["blue"]
        ax.plot(mem, color=c, linewidth=2.0)
        if threshold is not None:
            ax.axhline(threshold, color=PALETTE["red"], linestyle="--",
                       linewidth=1.2, alpha=0.8, label="threshold")
            ax.legend()
        self._finish(ax, title, xlabel="Time step", ylabel="Membrane potential",
                     legend=False)
        return fig, ax
 
    def confusion_matrix(self, matrix, class_names=None, ax=None, title=None,
                          figsize=None, cmap=None, annotate=True):
        """matrix: 2D array-like, shape (num_classes, num_classes)."""
        fig, ax = self._new_fig(figsize or (6, 6), ax)
        matrix = np.asarray(matrix)
        cmap = cmap or _dark_cmap()
        im = ax.imshow(matrix, cmap=cmap)
 
        n = matrix.shape[0]
        ticks = np.arange(n)
        labels = class_names if class_names is not None else ticks
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.grid(False)
 
        if annotate:
            thresh = matrix.max() / 2.0
            for i in range(n):
                for j in range(n):
                    val = matrix[i, j]
                    txt_color = BACKGROUND if val > thresh else FOREGROUND
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                           color=txt_color, fontsize=9)
 
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        if title:
            ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig, ax
 
    # ---------------- internals ----------------
 
    def _finish(self, ax, title, xlabel, ylabel, legend):
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
 
    @staticmethod
    def _moving_average(y, window):
        y = np.asarray(y, dtype=float)
        if window <= 1 or window > len(y):
            return y
        kernel = np.ones(window) / window
        return np.convolve(y, kernel, mode="valid")
 

