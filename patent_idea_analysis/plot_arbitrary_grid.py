"""
plot_arbitrary_grid.py

Plot a 2-row × N-column grid comparing results from two test_arbitrary_spectra.py
batch folders (e.g. 2-stage vs 3-stage run on the same random targets).

Each column corresponds to one test case (by index).  If both folders were run
with the same targets.json (via RESUME_FROM), the columns align by design.

Row 1: FOLDER_ROW1
Row 2: FOLDER_ROW2
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FOLDER_ROW1 = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\arb_test_with_6_sidebands_10_trials\arbitrary_test_20260429_154003"
FOLDER_ROW2 = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\arb_test_with_6_sidebands_10_trials\arbitrary_test_20260429_150528"

ROW1_LABEL = "2 stages"
ROW2_LABEL = "4 stages"

# Indices of tests to show as columns (0-based).  None = show all tests.
TEST_RANGE = None

N_DISPLAY         = 7              # harmonic indices shown each side on each subplot x-axis
SHOW_ZERO_TARGETS = True           # if True, draw open black circles at zero for non-target harmonics

USE_DB   = False
DB_FLOOR = -40.0
Y_MAX_DB = 2.0
Y_CEIL   = 0.5
Y_FLOOR  = -0.01

SAVE_DIR  = None # r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media"
SAVE_NAME = "arbitrary_grid.svg"
SAVE_DPI  = 200

# If True: skip the spectra grid, only save the summary scatter figures withs
# no axis labels / tick labels, spines above content (zorder=10), and .svg output.
PUBLISHED_PLOT = True

# Figure geometry (axes area in mm; margins in inches)
AX_W_MM      = 40.0
AX_H_MM      = 35.0
H_GAP_IN     = 0.15
V_GAP_IN     = 0.15
LEFT_IN      = 0.55
RIGHT_IN     = 0.15
BOTTOM_IN    = 0.45
TOP_IN       = 0.55   # space for column titles (target harmonics)
ROW_LABEL_IN = 0.30

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def _load_batch(folder: str) -> tuple[dict, list]:
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(folder, "results.json")) as f:
        results = json.load(f)
    return cfg, results


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "Arial"
mm_per_inch = 25.4

_cmap = matplotlib.colormaps["rainbow_r"]


def _draw_spectrum(ax, spectrum_powers: dict, norm_tgt: dict,
                   n_display: int, use_db: bool, db_floor: float, y_max: float):
    norm_color = matplotlib.colors.Normalize(vmin=-n_display, vmax=n_display)

    hs     = sorted(int(h) for h in spectrum_powers if abs(int(h)) <= n_display)
    powers = [spectrum_powers[str(h)] for h in hs]
    colors = [_cmap(norm_color(h)) for h in hs]

    if use_db:
        y_vals  = [10.0 * np.log10(max(p, 10 ** (db_floor / 10))) for p in powers]
        y_floor = db_floor
    else:
        y_vals  = powers
        y_floor = 0.0

    ax.vlines(hs, y_floor, y_vals, linewidth=1.5, colors=colors, zorder=2)
    ax.scatter(hs, y_vals, s=18, c=colors, zorder=3)

    target_hs = set()
    for h_str, tp in norm_tgt.items():
        h = int(h_str)
        if abs(h) > n_display:
            continue
        target_hs.add(h)
        y_tgt = (10.0 * np.log10(max(tp, 10 ** (db_floor / 10)))
                 if use_db else float(tp))
        ax.scatter(h, y_tgt, s=55, facecolors="none", edgecolors="black",
                   linewidths=1.0, zorder=5)

    if SHOW_ZERO_TARGETS:
        y_zero = db_floor if use_db else 0.0
        zero_hs = [h for h in range(-n_display, n_display + 1) if h not in target_hs]
        if zero_hs:
            ax.scatter(zero_hs, [y_zero] * len(zero_hs),
                       s=55, facecolors="none", edgecolors="black",
                       linewidths=1.0, alpha=0.5, zorder=5)

    ax.set_xlim(-n_display - 0.5, n_display + 0.5)
    ax.set_ylim(db_floor if use_db else Y_FLOOR, Y_CEIL)


def _style_ax(ax, show_xlabel: bool, show_ylabel: bool,
              show_xticks: bool, show_yticks: bool):
    fp = {"family": "Arial"}
    if show_xlabel:
        ax.set_xlabel("$k$", fontsize=10, **fp)
    if show_ylabel:
        ax.set_ylabel("CE [dB]" if USE_DB else "CE", fontsize=10, **fp)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_zorder(100)
    ax.tick_params(axis="both", which="both", direction="in",
                   width=1.5, labelsize=8,
                   labelbottom=show_xticks, labelleft=show_yticks)
    t_start    = -N_DISPLAY + (N_DISPLAY % 2)
    even_ticks = np.arange(t_start, N_DISPLAY + 1, 2)
    ax.set_xticks(even_ticks)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
    ax.grid(linewidth=0.4, zorder=0)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
cfg1, res1 = _load_batch(FOLDER_ROW1)
cfg2, res2 = _load_batch(FOLDER_ROW2)

# Build {test_idx: record} dicts
idx1 = {r["test_idx"]: r for r in res1}
idx2 = {r["test_idx"]: r for r in res2}

if TEST_RANGE is None:
    all_idxs = sorted(set(idx1) | set(idx2))
else:
    all_idxs = list(TEST_RANGE)

n_cols = len(all_idxs)
n_rows = 2

print(f"Row 1 ({ROW1_LABEL}): {len(res1)} tests  |  "
      f"Row 2 ({ROW2_LABEL}): {len(res2)} tests  |  "
      f"Showing {n_cols} columns")


def _metrics(records: dict, shared_idxs, n_display: int) -> dict:
    """Compute per-test MSE/RMSE over all harmonics in [-n_display, n_display],
    with target = 0 for non-target harmonics."""
    mse_list, pwr_err_list = [], []
    all_hs = list(range(-n_display, n_display + 1))
    for ti in shared_idxs:
        rec = records.get(ti)
        if rec is None:
            continue
        tgt = {int(h): float(v) for h, v in rec["target_powers_norm"].items()}
        ach = {int(h): float(v) for h, v in rec["spectrum_powers"].items()}
        sq_errs = [(ach.get(h, 0.0) - tgt.get(h, 0.0)) ** 2 for h in all_hs]
        mse_list.append(float(np.mean(sq_errs)))
        pwr_err_list.append(sum(ach.get(h, 0.0) for h in tgt)
                            - sum(tgt.values()))
    return dict(mse=np.array(mse_list), pwr_err=np.array(pwr_err_list))


shared_idxs = sorted(set(idx1) & set(idx2))
m1 = _metrics(idx1, shared_idxs, N_DISPLAY)
m2 = _metrics(idx2, shared_idxs, N_DISPLAY)

print(f"\n{'Metric':<32}  {ROW1_LABEL:<26}  {ROW2_LABEL:<26}")
print("-" * 88)
for label, vals1, vals2 in [
    ("MSE  (mean ± std)",  m1["mse"],            m2["mse"]),
    ("RMSE (mean ± std)",  np.sqrt(m1["mse"]),   np.sqrt(m2["mse"])),
    ("Total power err (mean ± std)", m1["pwr_err"], m2["pwr_err"]),
]:
    s1 = f"{np.mean(vals1):.8f} ± {np.std(vals1):.8f}"
    s2 = f"{np.mean(vals2):.8f} ± {np.std(vals2):.8f}"
    print(f"  {label:<30}  {s1:<26}  {s2:<26}")
print(f"  {'n tests (shared)':<30}  {len(m1['mse']):<26}  {len(m2['mse']):<26}")
print()

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
ax_w_in = AX_W_MM / mm_per_inch
ax_h_in = AX_H_MM / mm_per_inch

fig_w = (ROW_LABEL_IN + LEFT_IN
         + n_cols * ax_w_in + (n_cols - 1) * H_GAP_IN
         + RIGHT_IN)
fig_h = (BOTTOM_IN
         + n_rows * ax_h_in + (n_rows - 1) * V_GAP_IN
         + TOP_IN)

fig  = plt.figure(figsize=(fig_w, fig_h))
axes = [[None] * n_cols for _ in range(n_rows)]

for row in range(n_rows):
    for ci in range(n_cols):
        x0 = (ROW_LABEL_IN + LEFT_IN + ci * (ax_w_in + H_GAP_IN)) / fig_w
        y0 = (BOTTOM_IN + (n_rows - 1 - row) * (ax_h_in + V_GAP_IN)) / fig_h
        axes[row][ci] = fig.add_axes([x0, y0, ax_w_in / fig_w, ax_h_in / fig_h])

# ---------------------------------------------------------------------------
# Populate subplots
# ---------------------------------------------------------------------------
rows_data = [(idx1, ROW1_LABEL), (idx2, ROW2_LABEL)]
y_max = Y_MAX_DB if USE_DB else Y_CEIL

for row, (idx, row_label) in enumerate(rows_data):
    for ci, ti in enumerate(all_idxs):
        ax  = axes[row][ci]
        rec = idx.get(ti)

        show_xlabel = (row == n_rows - 1)
        show_ylabel = (ci == 0)

        if rec is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
            _style_ax(ax, show_xlabel, show_ylabel, show_xlabel, show_ylabel)
            continue

        _draw_spectrum(
            ax,
            spectrum_powers = rec["spectrum_powers"],
            norm_tgt        = rec["target_powers_norm"],
            n_display       = N_DISPLAY,
            use_db          = USE_DB,
            db_floor        = DB_FLOOR,
            y_max           = y_max,
        )
        _style_ax(ax, show_xlabel, show_ylabel, show_xlabel, show_ylabel)

        # tap = rec["total_achieved_power"]
        # ax.text(0.04, 0.97, f"P={tap:.2f}",
        #         transform=ax.transAxes, ha="left", va="top",
        #         fontsize=7, fontfamily="Arial", color="black")

# ---------------------------------------------------------------------------
# Summary figure config
# ---------------------------------------------------------------------------
SCT_AX_W_MM    = 70.0   # scatter panel width  [mm]
SCT_AX_H_MM    = 70.0   # scatter panel height [mm]
SCT_LEFT_IN    = 0.65   # left margin  [in]
SCT_RIGHT_IN   = 0.25   # right margin [in]
SCT_BOTTOM_IN  = 0.50   # bottom margin [in]
SCT_TOP_IN     = 0.40   # top margin (for title) [in]
SCT_XLIM       = (-0.02, 0.4)   # x-axis limits
SCT_YLIM       = (-0.02, 0.4)   # y-axis limits
SCT_PT_SIZE    = 60             # scatter marker area [pt²]
SCT_PT_ALPHA   = 1.0             # face alpha
SCT_PT_EW      = 1.0             # edge linewidth
SCT_COLORED_EDGE = True          # if True, edge is same color as face but at SCT_EDGE_ALPHA
SCT_EDGE_ALPHA   = 1.0           # edge alpha (only used when SCT_COLORED_EDGE = True)
SCT_SPINE_LW   = 1.5             # axes spine linewidth
SCT_IDEAL_LW   = 3.0             # ideal line linewidth

# ---------------------------------------------------------------------------
# Per-row summary figures
# ---------------------------------------------------------------------------
_norm_sct = matplotlib.colors.Normalize(vmin=-N_DISPLAY, vmax=N_DISPLAY)


def _make_summary_figure(records: dict, row_label: str,
                         published_plot: bool = False) -> matplotlib.figure.Figure:
    results = sorted(records.values(), key=lambda r: r["test_idx"])

    if published_plot:
        ax_w_mm, ax_h_mm = 55.0, 45.0
        left_in, right_in, bottom_in, top_in = 0.60, 0.20, 0.50, 0.20
    else:
        ax_w_mm, ax_h_mm = SCT_AX_W_MM, SCT_AX_H_MM
        left_in, right_in, bottom_in, top_in = (SCT_LEFT_IN, SCT_RIGHT_IN,
                                                 SCT_BOTTOM_IN, SCT_TOP_IN)

    ax_w_in = ax_w_mm / mm_per_inch
    ax_h_in = ax_h_mm / mm_per_inch
    fig_w   = left_in + ax_w_in + right_in
    fig_h   = bottom_in + ax_h_in + top_in

    fig    = plt.figure(figsize=(fig_w, fig_h))
    ax_sct = fig.add_axes([left_in / fig_w, bottom_in / fig_h,
                           ax_w_in / fig_w, ax_h_in / fig_h])
    fp = {"family": "Arial"}

    for r in results:
        norm_tgt = {int(h): v for h, v in r["target_powers_norm"].items()}
        achieved = {int(h): v for h, v in r["achieved_powers"].items()}
        for h, tp in norm_tgt.items():
            rgb        = _cmap(_norm_sct(h))[:3]
            face_rgba  = (*rgb, SCT_PT_ALPHA)
            ax_sct.scatter(tp, achieved.get(h, 0.0),
                           s=SCT_PT_SIZE,
                           facecolors=[face_rgba], edgecolors=[0, 0, 0, 1],
                           linewidths=SCT_PT_EW, zorder=3)

    ax_sct.plot([SCT_XLIM[0], SCT_XLIM[1]], [SCT_XLIM[0], SCT_XLIM[1]],
                color="black", linewidth=SCT_IDEAL_LW,
                linestyle="--", zorder=1, solid_capstyle='round')
    ax_sct.set_xlim(*SCT_XLIM)
    ax_sct.set_ylim(*SCT_YLIM)
    ax_sct.grid(zorder=0)

    for spine in ax_sct.spines.values():
        spine.set_linewidth(SCT_SPINE_LW)
        spine.set_zorder(10)
    ax_sct.tick_params(axis="both", which="both", direction="in",
                       width=SCT_SPINE_LW, labelsize=8,
                       labelbottom=not published_plot, labelleft=not published_plot)

    if published_plot:
        ax_sct.set_xlabel("")
        ax_sct.set_ylabel("")
    else:
        ax_sct.set_xlabel("Target power", fontsize=10, **fp)
        ax_sct.set_ylabel("Achieved power", fontsize=10, **fp)
        ax_sct.legend(fontsize=8, frameon=False)
        for lbl in ax_sct.get_xticklabels() + ax_sct.get_yticklabels():
            lbl.set_fontfamily("Arial")
        fig.suptitle(row_label, fontsize=10, fontfamily="Arial",
                     y=(bottom_in + ax_h_in + top_in * 0.7) / fig_h)

    return fig


fig_sum1 = _make_summary_figure(idx1, ROW1_LABEL, published_plot=PUBLISHED_PLOT)
fig_sum2 = _make_summary_figure(idx2, ROW2_LABEL, published_plot=PUBLISHED_PLOT)

# ---------------------------------------------------------------------------
# Row labels (grid only)
# ---------------------------------------------------------------------------
if not PUBLISHED_PLOT:
    for row, (_, row_label) in enumerate(rows_data):
        ax_left = axes[row][0]
        x_lbl   = (ROW_LABEL_IN * 0.45) / fig_w
        y_lbl   = ax_left.get_position().y0 + ax_left.get_position().height / 2
        fig.text(x_lbl, y_lbl, row_label,
                 ha="center", va="center",
                 fontsize=10, fontfamily="Arial", rotation=90)

# ---------------------------------------------------------------------------
# Save & show
# ---------------------------------------------------------------------------
if SAVE_DIR:
    os.makedirs(SAVE_DIR, exist_ok=True)
    stem, ext = os.path.splitext(SAVE_NAME)
    ext = ".svg" if PUBLISHED_PLOT else (ext or ".svg")

    if not PUBLISHED_PLOT:
        path = os.path.join(SAVE_DIR, f"{stem}{ext}")
        fig.savefig(path, dpi=SAVE_DPI)
        print(f"Saved {path}")

    for tag, fig_s in [(ROW1_LABEL, fig_sum1), (ROW2_LABEL, fig_sum2)]:
        slug = tag.replace(" ", "_")
        spath = os.path.join(SAVE_DIR, f"{stem}_summary_{slug}{ext}")
        fig_s.savefig(spath, dpi=SAVE_DPI)
        print(f"Saved {spath}")

plt.show()
