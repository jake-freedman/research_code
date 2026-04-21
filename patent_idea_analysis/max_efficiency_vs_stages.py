"""
Maximum power conversion efficiency to each of the first N_SIDEBANDS sidebands
for N_STAGES phase modulators, compared against a single PM.

For N_STAGES=2 the exact analytical upper bound is used:
    E_k = max_{beta1,beta2}  sum_n |J_n(beta1)| * |J_{k-n}(beta2)|

For N_STAGES>2 the basin-hopping optimizer (optimize_ssb, objective="power")
finds the best achievable efficiency numerically; this is a lower bound on the
true maximum, not a tight upper bound.

The single-PM reference is always  max_beta |J_k(beta)|^2.

Saved folder layout
-------------------
  <DATA_DIR>/<timestamp>/
    config.json           - all run parameters
    max_powers.npy        - max efficiency per sideband (N_STAGES PMs)
    max_powers_1pm.npy    - max efficiency per sideband (1 PM reference)
    sideband_<k>/         - one sub-folder per sideband (optimizer mode only)
      result.json         - betas, wanted_power, nit, nfev, message
      phi_params.npy      - optimised phase parameters

Set LOAD_PATH to a previously saved folder to skip computation and regenerate
the plot directly from saved data.
"""

import json
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np
from scipy.special import jv

from ssb_spectrum import optimize_ssb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SIDEBANDS = 8       # compute efficiency for sidebands 1 .. N_SIDEBANDS
N_STAGES    = 3       # number of phase modulators

BETA_MAX    = 10.0    # modulation depth search range [rad]

# Analytical mode (N_STAGES == 2 only)
N_POINTS    = 60      # grid points per beta axis
N_TERMS     = 50      # harmonic truncation for analytical computation

# Optimizer mode (N_STAGES > 2)
N_MAX       = 10      # harmonic truncation order per PM stage
N_ITER      = 100     # basin-hopping iterations per sideband
SEED        = 1    # integer for reproducibility, or None for random

# Path where a timestamped results folder will be created.
DATA_DIR  = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data"      # e.g. r"C:\path\to\data"

# Set to a previously saved folder to skip computation and just re-plot.
LOAD_PATH = None      # e.g. r"C:\path\to\data\20260421_143012"

# Path to save the figure image; None = don't save.
SAVE_PATH = None      # e.g. r"C:\path\to\figure.png"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_optimizer_result(folder, res):
    os.makedirs(folder, exist_ok=True)
    result_data = dict(
        betas        = [float(b) for b in res["betas"]],
        wanted_power = float(res["wanted_power"]),
        ratio        = float(res["ratio"]),
        nit          = int(res["opt_result"].nit),
        nfev         = int(res["opt_result"].nfev),
        message      = str(res["opt_result"].message),
    )
    with open(os.path.join(folder, "result.json"), "w") as f:
        json.dump(result_data, f, indent=2)

    phi_arr = np.empty(len(res["phi_params_list"]), dtype=object)
    for i, p in enumerate(res["phi_params_list"]):
        phi_arr[i] = p
    np.save(os.path.join(folder, "phi_params.npy"), phi_arr, allow_pickle=True)


def _make_plot(sidebands, max_powers, max_powers_1pm, n_stages, n_sidebands):
    mm_per_inch = 25.4
    width_mm, height_mm = 100.0, 90.0

    left_in   = 0.60
    bottom_in = 0.50
    top_in    = 0.20
    right_in  = 0.20

    ax_w_in = width_mm / mm_per_inch
    ax_h_in = height_mm / mm_per_inch
    fig_w   = left_in + ax_w_in + right_in
    fig_h   = bottom_in + ax_h_in + top_in

    matplotlib.rcParams["font.family"] = "Arial"
    font_props = {"family": "Arial"}

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([
        left_in / fig_w,
        bottom_in / fig_h,
        ax_w_in / fig_w,
        ax_h_in / fig_h,
    ])

    cmap   = matplotlib.colormaps["rainbow_r"]
    colors = [cmap(i / (n_sidebands - 1)) for i in range(n_sidebands)]

    bar_width = 0.6
    bar_half  = bar_width / 2

    r_mm    = 2.5
    x_range = n_sidebands
    y_range = 1.0
    rx = r_mm * x_range / width_mm
    ry = r_mm * y_range / height_mm
    kappa = 0.5523

    def rounded_bar_path(x0, y0, w, h, rx, ry):
        x1, x2 = x0, x0 + w
        y1, y2 = y0, y0 + h
        verts = [
            (x1,      y1),
            (x2,      y1),
            (x2,      y2 - ry),
            (x2,      y2 - ry + kappa*ry), (x2 - rx + kappa*rx, y2), (x2 - rx, y2),
            (x1 + rx, y2),
            (x1 + rx - kappa*rx, y2), (x1, y2 - ry + kappa*ry), (x1, y2 - ry),
            (x1,      y1),
        ]
        codes = [
            Path.MOVETO,   Path.LINETO,
            Path.LINETO,
            Path.CURVE4,   Path.CURVE4,   Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,   Path.CURVE4,   Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        return Path(verts, codes)

    for k, h, c in zip(sidebands, max_powers, colors):
        ax.add_patch(mpatches.PathPatch(
            rounded_bar_path(k - bar_half, 0, bar_width, h, rx, ry),
            facecolor=c, edgecolor="none", zorder=3,
        ))

    stage_label = f"{n_stages} PMs (analytical)" if n_stages == 2 else f"{n_stages} PMs (optimized)"
    ax.add_patch(mpatches.PathPatch(
        rounded_bar_path(0, 0, 0, 0, rx, ry),
        facecolor="#888888", edgecolor="none", label=stage_label,
    ))

    ax.hlines(
        max_powers_1pm,
        sidebands - bar_half, sidebands + bar_half,
        colors="black", linewidths=3.0, linestyles="-", zorder=4,
        label="1 PM", capstyle="round",
    )
    ax.legend(fontsize=8, frameon=False, prop={"family": "Arial"})

    ax.set_xlabel(r"Sideband order [dimensionless]", fontsize=10, **font_props)
    ax.set_ylabel(r"Max conversion efficiency  [dimensionless]", fontsize=10, **font_props)
    ax.set_xticks(sidebands)
    ax.set_xlim(0.5, n_sidebands + 0.5)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", zorder=0)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")

    return fig, ax


# ---------------------------------------------------------------------------
# Load or compute
# ---------------------------------------------------------------------------
if LOAD_PATH is not None:
    print(f"Loading results from {LOAD_PATH}")
    with open(os.path.join(LOAD_PATH, "config.json")) as f:
        cfg = json.load(f)
    N_STAGES    = cfg["n_stages"]
    N_SIDEBANDS = cfg["n_sidebands"]
    max_powers     = np.load(os.path.join(LOAD_PATH, "max_powers.npy"))
    max_powers_1pm = np.load(os.path.join(LOAD_PATH, "max_powers_1pm.npy"))
    sidebands = np.arange(1, N_SIDEBANDS + 1)
    out_folder = None

else:
    betas     = np.linspace(0, BETA_MAX, N_POINTS)
    sidebands = np.arange(1, N_SIDEBANDS + 1)

    max_powers     = np.empty(N_SIDEBANDS)
    max_powers_1pm = np.empty(N_SIDEBANDS)

    # Accumulate per-sideband optimizer results for saving
    opt_results = {}

    if N_STAGES == 2:
        n_range = np.arange(-N_TERMS, N_TERMS + 1)
        b1, b2  = np.meshgrid(betas, betas, indexing="ij")
        Jn_b1   = jv(n_range, b1[..., np.newaxis])
        for i, k in enumerate(sidebands):
            Jm_b2         = jv(k - n_range, b2[..., np.newaxis])
            E_grid        = np.sum(np.abs(Jn_b1) * np.abs(Jm_b2), axis=-1)
            max_powers[i] = float(np.max(E_grid) ** 2)
            print(f"  sideband {k}: analytical max efficiency = {max_powers[i]:.4f}")
    else:
        rng = np.random.default_rng(SEED)
        for i, k in enumerate(sidebands):
            print(f"  sideband {k}: running optimizer ({N_ITER} iters) ...", end=" ", flush=True)
            res = optimize_ssb(
                wanted_harmonic = int(k),
                beta_max        = BETA_MAX,
                n_max           = N_MAX,
                n_stages        = N_STAGES,
                n_iter          = N_ITER,
                seed            = int(rng.integers(1 << 31)),
                objective       = "power",
            )
            max_powers[i]  = res["wanted_power"]
            opt_results[k] = res
            print(f"max efficiency = {max_powers[i]:.4f}")

    for i, k in enumerate(sidebands):
        max_powers_1pm[i] = float(np.max(jv(k, betas) ** 2))

    # Save results
    if DATA_DIR is not None:
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_folder = os.path.join(DATA_DIR, timestamp)
        os.makedirs(out_folder, exist_ok=True)

        config = dict(
            n_stages    = N_STAGES,
            n_sidebands = N_SIDEBANDS,
            beta_max    = BETA_MAX,
            n_points    = N_POINTS,
            n_terms     = N_TERMS,
            n_max       = N_MAX,
            n_iter      = N_ITER,
            seed        = SEED,
        )
        with open(os.path.join(out_folder, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        np.save(os.path.join(out_folder, "max_powers.npy"),     max_powers)
        np.save(os.path.join(out_folder, "max_powers_1pm.npy"), max_powers_1pm)

        for k, res in opt_results.items():
            _save_optimizer_result(os.path.join(out_folder, f"sideband_{k}"), res)

        print(f"Results saved to {out_folder}")
    else:
        out_folder = None

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = _make_plot(sidebands, max_powers, max_powers_1pm, N_STAGES, N_SIDEBANDS)

if SAVE_PATH is not None:
    fig.savefig(SAVE_PATH, dpi=200)
    print(f"Figure saved to {SAVE_PATH}")
plt.show()
