"""
Load multiple sweep result folders from a parent directory and plot all curves
on the same axes.  Handles two sweep types, auto-detected from file contents:

  disp_n_max sweep  — sweep_disp_n_max.json  (x-axis: number of channels 2*N+1)
  beta_max sweep    — sweep_beta_max.json     (x-axis: β_max [rad])

All subfolders in PARENT_FOLDER must be the same sweep type.
The wanted_harmonic label is read from each subfolder's config.json.

Saved files (if SAVE_DIR is set)
---------------------------------
  <SAVE_DIR>/sweep_comparison_linear.png
  <SAVE_DIR>/sweep_comparison_dB.png
  <SAVE_DIR>/sweep_comparison_suppression.png
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from graphics import (
    LIGHTBLUE2, BLUE2, GREEN2, VIOLET2, RED2, DARKBLUE2, DARKGREEN2, ORANGE2, PINK2, TAN2, BEIGE2
)

from path_utils import local_path

# ---------------------------------------------------------------------------
# INPUT — set this to the parent folder containing the sweep sub-folders
# ---------------------------------------------------------------------------
PARENT_FOLDER = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\3stage_ssbm")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
SAVE_DIR = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media")
SAVE_DPI = 200

# ---------------------------------------------------------------------------
# Axes / display
# ---------------------------------------------------------------------------
DB_FLOOR  = -5.0    # dB floor for the CE log plot
Y_MAX_LIN = 1.05    # upper y-limit for linear plot
Y_MAX_DB  = 0.2     # upper y-limit for CE dB plot
X_PAD     = 0.5     # padding added to each end of the x-axis

# Suppression ratio  (wanted / next-strongest sideband)
SUPP_Y_MIN = 0.0    # lower y-limit [dB]
SUPP_Y_MAX = 65.0   # upper y-limit [dB]

# Figure dimensions (axes area in mm; margins in inches)
AX_W_MM   = 100.0
AX_H_MM   = 150.0
LEFT_IN   = 0.65
RIGHT_IN  = 0.25
BOTTOM_IN = 0.45
TOP_IN    = 0.15

# ---------------------------------------------------------------------------
# Per-curve style
# Colors cycled in this order; extend if you have more than 8 sweeps.
# ---------------------------------------------------------------------------
COLORS     = [RED2, ORANGE2, GREEN2, LIGHTBLUE2, BEIGE2, VIOLET2]
MARKERS    = 10 * ["o"]
LINESTYLES = ["-", "-", "-", "-", "-", "-", "-", "-"]
LINEWIDTH  = 2
MARKERSIZE = 6
ALPHAS     = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # per-series opacity (0.0–1.0)

# ---------------------------------------------------------------------------
# Discover and load sweep folders
# ---------------------------------------------------------------------------
if not os.path.isdir(PARENT_FOLDER):
    raise FileNotFoundError(f"PARENT_FOLDER not found: {PARENT_FOLDER}")

entries     = []
sweep_types = set()

for name in sorted(os.listdir(PARENT_FOLDER)):
    folder      = os.path.join(PARENT_FOLDER, name)
    config_path = os.path.join(folder, "config.json")
    disp_path   = os.path.join(folder, "sweep_disp_n_max.json")
    beta_path   = os.path.join(folder, "sweep_beta_max.json")

    if not os.path.isfile(config_path):
        continue
    if os.path.isfile(disp_path):
        sweep_type = "disp_n_max"
        sweep_path = disp_path
    elif os.path.isfile(beta_path):
        sweep_type = "beta_max"
        sweep_path = beta_path
    else:
        continue

    sweep_types.add(sweep_type)
    with open(config_path) as f:
        cfg = json.load(f)
    with open(sweep_path) as f:
        data = json.load(f)

    xs = ([pt["disp_n_max"] for pt in data] if sweep_type == "disp_n_max"
          else [pt["beta_max"] for pt in data])

    entries.append(dict(
        folder            = folder,
        sweep_type        = sweep_type,
        wanted_harmonic   = cfg["wanted_harmonic"],
        n_max             = cfg["n_max"],
        xs_raw            = xs,
        ys                = [pt["wanted_power"]               for pt in data],
        best_other_powers = [pt.get("best_other_power", None) for pt in data],
        # range metadata for x-axis limits
        x_max_raw         = (cfg.get("disp_n_max_max", max(xs)) if sweep_type == "disp_n_max"
                             else cfg.get("beta_max_max", max(xs))),
        x_min_raw         = (1 if sweep_type == "disp_n_max"
                             else cfg.get("beta_max_min", min(xs))),
    ))

if not entries:
    raise RuntimeError(f"No valid sweep folders found in {PARENT_FOLDER}")
if len(sweep_types) > 1:
    raise RuntimeError(
        f"Mixed sweep types found: {sweep_types}. "
        "All subfolders must be the same sweep kind."
    )

SWEEP_TYPE = sweep_types.pop()   # "disp_n_max" or "beta_max"
entries.sort(key=lambda e: e["wanted_harmonic"])

print(f"Sweep type: {SWEEP_TYPE}  |  Found {len(entries)} series:")
for e in entries:
    print(f"  harmonic={e['wanted_harmonic']}  points={len(e['xs_raw'])}  "
          f"folder={os.path.basename(e['folder'])}")

# ---------------------------------------------------------------------------
# x-axis helpers
# ---------------------------------------------------------------------------
def _to_plot_x(xs_raw: list) -> list:
    """Convert raw sweep values to plot x coordinates."""
    if SWEEP_TYPE == "disp_n_max":
        return [2 * x + 1 for x in xs_raw]
    return xs_raw

if SWEEP_TYPE == "disp_n_max":
    x_label      = "Number of channels  ($2N_{\\mathrm{disp}}+1$)"
    x_min_global = 2 * min(e["x_min_raw"] for e in entries) + 1 - X_PAD
    x_max_global = 2 * max(e["x_max_raw"] for e in entries) + 1 + X_PAD
else:
    x_label      = r"$\alpha_{\max}$  [rad]"
    x_min_global = min(e["x_min_raw"] for e in entries) - X_PAD
    x_max_global = max(e["x_max_raw"] for e in entries) + X_PAD

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "Arial"
font_props = {"family": "Arial"}
mm_per_inch = 25.4

def _make_ax(ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    ax_w_in = AX_W_MM / mm_per_inch
    ax_h_in = AX_H_MM / mm_per_inch
    fig_w = LEFT_IN + ax_w_in + RIGHT_IN
    fig_h = BOTTOM_IN + ax_h_in + TOP_IN
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([
        LEFT_IN   / fig_w,
        BOTTOM_IN / fig_h,
        ax_w_in   / fig_w,
        ax_h_in   / fig_h,
    ])
    ax.set_xlabel(x_label, fontsize=10, **font_props)
    ax.set_ylabel(ylabel, fontsize=10, **font_props)
    ax.set_xlim(x_min_global, x_max_global)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=(SWEEP_TYPE == "disp_n_max")))
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
    ax.grid()
    return fig, ax


def _save(fig: plt.Figure, name: str) -> None:
    if SAVE_DIR:
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR, f"{name}.png")
        fig.savefig(path, dpi=SAVE_DPI)
        print(f"Saved {path}")


def _plot_series(ax, xs_plot, ys, i, label):
    ax.plot(
        xs_plot, ys,
        color      = COLORS[i % len(COLORS)],
        linewidth  = LINEWIDTH,
        linestyle  = LINESTYLES[i % len(LINESTYLES)],
        marker     = MARKERS[i % len(MARKERS)],
        markersize = MARKERSIZE,
        alpha      = ALPHAS[i % len(ALPHAS)],
        label      = label,
        zorder     = 3,
    )


# ---------------------------------------------------------------------------
# Linear CE plot
# ---------------------------------------------------------------------------
fig_lin, ax_lin = _make_ax("Conversion efficiency")
ax_lin.set_ylim(0, Y_MAX_LIN)
for i, e in enumerate(entries):
    _plot_series(ax_lin, _to_plot_x(e["xs_raw"]), e["ys"],
                 i, f"harmonic {e['wanted_harmonic']}")
ax_lin.legend(fontsize=8, frameon=False)
_save(fig_lin, "sweep_comparison_linear")

# ---------------------------------------------------------------------------
# dB CE plot
# ---------------------------------------------------------------------------
fig_db, ax_db = _make_ax("Conversion efficiency [dB]")
ax_db.set_ylim(DB_FLOOR, Y_MAX_DB)
for i, e in enumerate(entries):
    ys_db = 10.0 * np.log10(np.maximum(e["ys"], 10 ** (DB_FLOOR / 10)))
    _plot_series(ax_db, _to_plot_x(e["xs_raw"]), list(ys_db),
                 i, f"harmonic {e['wanted_harmonic']}")
ax_db.legend(fontsize=8, frameon=False)
_save(fig_db, "sweep_comparison_dB")

# ---------------------------------------------------------------------------
# Suppression ratio plot  (wanted / next-strongest sideband, in dB)
# Skips series where best_other_power was not saved (old data).
# ---------------------------------------------------------------------------
supp_entries = [e for e in entries if any(v is not None for v in e["best_other_powers"])]

if supp_entries:
    fig_supp, ax_supp = _make_ax(
        r"Suppression ratio  $P_{\mathrm{wanted}}/P_{\mathrm{next}}$  [dB]"
    )
    ax_supp.set_ylim(SUPP_Y_MIN, SUPP_Y_MAX)

    for i, e in enumerate(entries):
        bop = e["best_other_powers"]
        if all(v is None for v in bop):
            print(f"  Skipping harmonic {e['wanted_harmonic']} — no best_other_power in data")
            continue
        xs_s, ys_s = [], []
        for x_raw, wp, bo in zip(e["xs_raw"], e["ys"], bop):
            if bo is None or bo <= 0 or wp <= 0:
                continue
            xs_s.append(_to_plot_x([x_raw])[0])
            ys_s.append(10.0 * np.log10(wp / bo))
        _plot_series(ax_supp, xs_s, ys_s, i, f"harmonic {e['wanted_harmonic']}")

    ax_supp.legend(fontsize=8, frameon=False)
    _save(fig_supp, "sweep_comparison_suppression")
else:
    print("No suppression ratio data found — re-run sweeps to generate best_other_power.")

plt.show()
