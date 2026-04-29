"""
plot_split_grid.py

Plot a 2-row × N-column grid comparing optimisation results from two batch
folders (e.g. 2-stage vs 3-stage runs from test_arbitrary_spectra.py).

Each column corresponds to one n value; the expected target was [-n, +n]
split equally.  The script matches results by looking for
target_harmonics == [-n, n] in each folder's results.json.

Row 1: FOLDER_ROW1  (e.g. 2-stage)
Row 2: FOLDER_ROW2  (e.g. 3-stage)
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ssb_spectrum import compute_optical_spectrum_general

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FOLDER_ROW1 = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\3stage_ssbm"
FOLDER_ROW2 = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\3stage_ssbm"

ROW1_LABEL = "2 stages"
ROW2_LABEL = "3 stages"

N_RANGE          = [2, 3, 4, 5]    # n values to plot (columns); one column per value
N_DISPLAY        = 6             # harmonic indices shown each side on each subplot x-axis
SSB_SHOW_NEGATIVE  = True         # if True, SSB subplots show negative sidebands too
SHOW_ZERO_TARGETS  = True         # if True, draw open black circles at zero for non-target harmonics

USE_DB   = False
DB_FLOOR = -40.0
Y_MAX_DB = 5.0             # upper y-limit [dB]
Y_CEIL = 1.1
Y_FLOOR = -0.05

SAVE_DIR  = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media"
SAVE_NAME = "split_grid"   # filename stem (no extension)
SAVE_DPI  = 200

# Figure geometry (axes area in mm; margins in inches)
AX_W_MM   = 40.0
AX_H_MM   = 35.0
H_GAP_IN  = 0.15   # horizontal gap between columns
V_GAP_IN  = 0.15   # vertical gap between rows (space for x-label on row 1)
LEFT_IN   = 0.55   # left margin (for y-label)
RIGHT_IN  = 0.15
BOTTOM_IN = 0.45   # bottom margin (x-label on row 2)
TOP_IN    = 0.45   # top margin (column titles)
ROW_LABEL_IN = 0.20  # extra space to the left for row labels

# ---------------------------------------------------------------------------
# Load helpers  (run_optimization.py subfolder format)
# ---------------------------------------------------------------------------
def _n_from_config(cfg: dict) -> tuple[int, bool] | tuple[None, None]:
    """
    Detect the n value and whether this is an SSB run.

    Split runs: arbitrary_targets = {-n: w, n: w}  or  power_split_harmonics = [-n, n]
    SSB runs:   objective in ("power", "ratio") with wanted_harmonic = n

    Returns (n, is_ssb) or (None, None) if not recognised.
    """
    at = cfg.get("arbitrary_targets") or {}
    hs = sorted(int(k) for k in at)
    if len(hs) == 2 and hs[0] == -hs[1] and hs[1] > 0:
        return hs[1], False
    ps = cfg.get("power_split_harmonics") or []
    ps_sorted = sorted(int(h) for h in ps)
    if len(ps_sorted) == 2 and ps_sorted[0] == -ps_sorted[1] and ps_sorted[1] > 0:
        return ps_sorted[1], False
    obj = cfg.get("objective", "")
    if obj in ("power", "ratio"):
        wh = cfg.get("wanted_harmonic")
        if wh is not None and int(wh) > 0:
            return int(wh), True
    return None, None


def _load_subfolder(subfolder: str) -> dict | None:
    """
    Load one run_optimization result subfolder.  Returns a record dict with
    keys: n, is_ssb, norm_tgt, spectrum_powers, total_achieved_power, betas.
    Returns None if the folder can't be parsed.
    """
    cfg_path    = os.path.join(subfolder, "config.json")
    result_path = os.path.join(subfolder, "result.json")
    phi_path    = os.path.join(subfolder, "phi_params.npy")
    if not all(os.path.isfile(p) for p in (cfg_path, result_path, phi_path)):
        return None

    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(result_path) as f:
        res = json.load(f)

    n, is_ssb = _n_from_config(cfg)
    if n is None:
        return None

    # Reconstruct spectrum
    n_max    = cfg["n_max"]
    n_stages = cfg.get("n_stages", 2)
    truncate = n_stages > 2
    disp_n_max  = cfg.get("disp_n_max", None)
    poly_order  = cfg.get("poly_order", None)
    stage_range = np.arange(-n_max, n_max + 1)

    phi_params_raw = np.load(phi_path, allow_pickle=True)
    phi_arrs = []
    for s in range(n_stages - 1):
        raw = phi_params_raw[s]
        if poly_order is None:
            if disp_n_max is not None:
                full = np.zeros(len(stage_range))
                mask = (stage_range >= -disp_n_max) & (stage_range <= disp_n_max)
                full[mask] = raw
                phi_arrs.append(full)
            else:
                phi_arrs.append(raw)
        else:
            phi_arr = np.polyval(raw[::-1], stage_range)
            if disp_n_max is not None:
                phi_arr[np.abs(stage_range) > disp_n_max] = 0.0
            phi_arrs.append(phi_arr)

    harmonics, amplitudes = compute_optical_spectrum_general(
        res["betas"], phi_arrs, n_max=n_max, truncate=truncate
    )
    power = np.abs(amplitudes) ** 2

    # Normalised targets
    if is_ssb:
        norm_tgt = {n: 1.0}
    else:
        at_raw = cfg.get("arbitrary_targets") or {}
        if at_raw:
            weights = {int(k): float(v) for k, v in at_raw.items()}
        else:
            hs_split = cfg.get("power_split_harmonics") or [-n, n]
            weights  = {int(h): 1.0 for h in hs_split}
        total_w  = sum(weights.values())
        norm_tgt = {h: w / total_w for h, w in weights.items()}

    # Achieved power at target harmonics
    achieved = {}
    for h in norm_tgt:
        idx = np.where(harmonics == h)[0]
        achieved[h] = float(power[idx[0]]) if len(idx) else 0.0

    x_min = -N_DISPLAY if (not is_ssb or SSB_SHOW_NEGATIVE) else 0
    spectrum_powers = {
        str(int(h)): float(p)
        for h, p in zip(harmonics, power)
        if h >= x_min and h <= N_DISPLAY
    }

    return dict(
        n                    = n,
        is_ssb               = is_ssb,
        norm_tgt             = {str(h): v for h, v in norm_tgt.items()},
        spectrum_powers      = spectrum_powers,
        total_achieved_power = float(sum(achieved.values())),
        betas                = res["betas"],
        x_min                = x_min,
    )


def _load_folder(folder: str) -> dict[int, dict]:
    """Scan all subfolders of folder; return {n: record}."""
    records = {}
    for name in sorted(os.listdir(folder)):
        sub = os.path.join(folder, name)
        if not os.path.isdir(sub):
            continue
        rec = _load_subfolder(sub)
        if rec is not None:
            records[rec["n"]] = rec
    return records


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "Arial"
mm_per_inch = 25.4

_cmap = matplotlib.colormaps["rainbow_r"]


def _draw_spectrum(ax, spectrum_powers: dict, norm_tgt: dict,
                   n_display: int, use_db: bool, db_floor: float,
                   y_max: float, x_min: int = None):
    """
    Draw actual spectrum (solid vlines + filled circles) and target
    (open black circles) onto ax.  x_min controls the left x-limit
    (0 for SSB-only plots, -n_display for symmetric split plots).
    """
    if x_min is None:
        x_min = -n_display
    norm_color = matplotlib.colors.Normalize(vmin=-n_display, vmax=n_display)

    hs = [int(h) for h in spectrum_powers]
    hs.sort()
    powers = [spectrum_powers[str(h)] for h in hs]

    if use_db:
        y_vals  = [10.0 * np.log10(max(p, 10 ** (db_floor / 10))) for p in powers]
        y_floor = db_floor
    else:
        y_vals  = powers
        y_floor = 0.0

    colors = [_cmap(norm_color(h)) for h in hs]

    ax.vlines(hs, y_floor, y_vals, linewidth=1.5, colors=colors, zorder=2)
    ax.scatter(hs, y_vals, s=18, c=colors, zorder=3)

    target_hs = set()
    for h_str, tp in norm_tgt.items():
        h = int(h_str)
        if h < x_min or h > n_display:
            continue
        target_hs.add(h)
        y_tgt = (10.0 * np.log10(max(tp, 10 ** (db_floor / 10)))
                 if use_db else float(tp))
        ax.scatter(h, y_tgt, s=55, facecolors="none", edgecolors="black",
                   linewidths=1.0, zorder=5)

    if SHOW_ZERO_TARGETS:
        y_zero = db_floor if use_db else 0.0
        zero_hs = [h for h in range(x_min, n_display + 1) if h not in target_hs]
        if zero_hs:
            ax.scatter(zero_hs, [y_zero] * len(zero_hs),
                       s=55, facecolors="none", edgecolors="black", alpha = 0.5,
                       linewidths=1.0, zorder=5)

    ax.set_xlim(x_min - 0.5, n_display + 0.5)
    ax.set_ylim(db_floor if use_db else Y_FLOOR, Y_CEIL)


def _style_ax(ax, show_xlabel: bool, show_ylabel: bool,
              show_xticks: bool, show_yticks: bool, x_min: int = None):
    if x_min is None:
        x_min = -N_DISPLAY
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
    # Start ticks from x_min, stepping by 2, aligned to even values
    t_start = x_min + (x_min % 2)  # round up to nearest even
    even_ticks = np.arange(t_start, N_DISPLAY + 1, 2)
    ax.set_xticks(even_ticks)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
    ax.grid(linewidth=0.4, zorder=0)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
res1 = _load_folder(FOLDER_ROW1)
res2 = _load_folder(FOLDER_ROW2)
print(f"Row 1 ({ROW1_LABEL}): found n = {sorted(res1)}")
print(f"Row 2 ({ROW2_LABEL}): found n = {sorted(res2)}")

n_cols = len(N_RANGE)
n_rows = 2

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
ax_w_in  = AX_W_MM / mm_per_inch
ax_h_in  = AX_H_MM / mm_per_inch

fig_w = (ROW_LABEL_IN + LEFT_IN
         + n_cols * ax_w_in + (n_cols - 1) * H_GAP_IN
         + RIGHT_IN)
fig_h = (BOTTOM_IN
         + n_rows * ax_h_in + (n_rows - 1) * V_GAP_IN
         + TOP_IN)

fig = plt.figure(figsize=(fig_w, fig_h))

# axes[row][col]
axes = [[None] * n_cols for _ in range(n_rows)]

for row in range(n_rows):
    for ci, n in enumerate(N_RANGE):
        x0 = (ROW_LABEL_IN + LEFT_IN
              + ci * (ax_w_in + H_GAP_IN)) / fig_w
        # row 0 is top
        y0 = (BOTTOM_IN
              + (n_rows - 1 - row) * (ax_h_in + V_GAP_IN)) / fig_h
        ax = fig.add_axes([x0, y0, ax_w_in / fig_w, ax_h_in / fig_h])
        axes[row][ci] = ax

# ---------------------------------------------------------------------------
# Populate subplots
# ---------------------------------------------------------------------------
rows_data = [(res1, ROW1_LABEL), (res2, ROW2_LABEL)]

for row, (records, row_label) in enumerate(rows_data):
    for ci, n in enumerate(N_RANGE):
        ax  = axes[row][ci]
        rec = records.get(n)

        show_xlabel = (row == n_rows - 1)   # only bottom row
        show_ylabel = (ci == 0)             # only leftmost column
        show_xticks = show_xlabel
        show_yticks = show_ylabel

        x_min = rec["x_min"] if rec is not None else -N_DISPLAY

        if rec is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
            _style_ax(ax, show_xlabel, show_ylabel, show_xticks, show_yticks, x_min)
            continue

        _draw_spectrum(
            ax,
            spectrum_powers = rec["spectrum_powers"],
            norm_tgt        = rec["norm_tgt"],
            n_display       = N_DISPLAY,
            use_db          = USE_DB,
            db_floor        = DB_FLOOR,
            y_max           = Y_MAX_DB if USE_DB else 1.05,
            x_min           = x_min,
        )
        _style_ax(ax, show_xlabel, show_ylabel, show_xticks, show_yticks, x_min)

        # tap = rec["total_achieved_power"]
        # ax.text(0.04, 0.97,
        #         f"P={tap:.2f}",
        #         transform=ax.transAxes, ha="left", va="top",
        #         fontsize=8, fontfamily="Arial",
        #         color="black")

# ---------------------------------------------------------------------------
# Row labels  (left of leftmost column)
# ---------------------------------------------------------------------------
for row, (_, row_label) in enumerate(rows_data):
    ax_left = axes[row][0]
    x_label = (ROW_LABEL_IN * 0.85) / fig_w
    y_label = ax_left.get_position().y0 + ax_left.get_position().height / 2
    fig.text(x_label, y_label,
             row_label,
             ha="center", va="center",
             fontsize=10, fontfamily="Arial",
             rotation=90)

# ---------------------------------------------------------------------------
# Save & show
# ---------------------------------------------------------------------------
if SAVE_DIR:
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{SAVE_NAME}.svg")
    fig.savefig(path, dpi=SAVE_DPI)
    print(f"Saved {path}")

plt.show()
