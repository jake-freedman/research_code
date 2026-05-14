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
  <OUT_DIR>/<timestamp>/
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

import numpy as np
from scipy.special import jv

from ssb_spectrum import optimize_ssb

from path_utils import local_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SIDEBANDS = 2       # compute efficiency for sidebands 1 .. N_SIDEBANDS
N_STAGES    = 3       # number of phase modulators

BETA_MAX    = 5.0    # modulation depth search range [rad]

# Analytical mode (N_STAGES == 2 only)
N_POINTS    = 60      # grid points per beta axis
N_TERMS     = 50      # harmonic truncation for analytical computation

# Optimizer mode (N_STAGES > 2)
N_MAX       = 20      # harmonic truncation order per PM stage
N_ITER      = 1     # basin-hopping iterations per sideband
SEED        = 1    # integer for reproducibility, or None for random

# Path where a timestamped results folder will be created.
OUT_DIR   = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data")

# Set to a previously saved folder to skip computation and just re-plot.
LOAD_PATH = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\sideband_up_to8_3pm")

# Path to save the figure image; None = don't save.
SAVE_PATH = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media\3mods_max_eff")

# If True: strip all axis labels and tick labels, raise spines above content,
# set y-max to 1.1, and save as .svg regardless of SAVE_PATH extension.
PUBLISHED_PLOT = True

# 2-PM reference hlines (shown only when N_STAGES > 2)
REF_BETA_MAX = 10.0   # beta search range for 2-PM analytical bound
REF_N_POINTS = 60     # grid points per beta axis
REF_N_TERMS  = 50     # harmonic truncation

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


def _make_plot(sidebands, max_powers, max_powers_1pm, n_sidebands,
               max_powers_2pm=None, published_plot=False):
    mm_per_inch = 25.4
    width_mm, height_mm = 55, 45

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
    colors = [cmap((k + 9) / 18) for k in sidebands]

    bar_width = 0.5
    bar_half  = bar_width / 2

    for k, h, c in zip(sidebands, max_powers, colors):
        ax.bar(k, h, width=bar_width, color=c, edgecolor="black",
               linewidth=1.5, zorder=3)

    ax.hlines(
        max_powers_1pm,
        sidebands - bar_half, sidebands + bar_half,
        colors="black", linewidths=1.5, linestyles="-", zorder=4,
        label="1 PM",
    )
    if max_powers_2pm is not None:
        ax.hlines(
            max_powers_2pm,
            sidebands - bar_half, sidebands + bar_half,
            colors="black", linewidths=1.5, linestyles="-", zorder=4,
            label="2 PM",
        )
    if not published_plot:
        ax.legend(fontsize=8, frameon=False, prop={"family": "Arial"})

    ax.set_xticks(sidebands)
    ax.set_xlim(0.5, n_sidebands + 0.5)
    ax.grid(axis="y", zorder=0)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_zorder(10)

    if published_plot:
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both", direction="in", width=2,
                       labelbottom=False, labelleft=False)
    else:
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r"Sideband order [dimensionless]", fontsize=10, **font_props)
        ax.set_ylabel(r"Max conversion efficiency  [dimensionless]", fontsize=10, **font_props)
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
    if OUT_DIR is not None:
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_folder = os.path.join(OUT_DIR, timestamp)
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

# Compute 2-PM analytical upper bound for reference (only when N_STAGES > 2)
if N_STAGES > 2:
    _ref_betas  = np.linspace(0, REF_BETA_MAX, REF_N_POINTS)
    _n_ref      = np.arange(-REF_N_TERMS, REF_N_TERMS + 1)
    _rb1, _rb2  = np.meshgrid(_ref_betas, _ref_betas, indexing="ij")
    _Jn_rb1     = jv(_n_ref, _rb1[..., np.newaxis])
    max_powers_2pm = np.empty(N_SIDEBANDS)
    for _i, _k in enumerate(sidebands):
        _Jm_rb2            = jv(_k - _n_ref, _rb2[..., np.newaxis])
        max_powers_2pm[_i] = float(np.max(np.sum(np.abs(_Jn_rb1) * np.abs(_Jm_rb2), axis=-1)) ** 2)
else:
    max_powers_2pm = None

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = _make_plot(sidebands, max_powers, max_powers_1pm, N_SIDEBANDS,
                     max_powers_2pm=max_powers_2pm, published_plot=PUBLISHED_PLOT)

if SAVE_PATH is not None:
    if PUBLISHED_PLOT:
        import os
        save_path = os.path.splitext(SAVE_PATH)[0] + ".svg"
    else:
        save_path = SAVE_PATH
    fig.savefig(save_path, dpi=200)
    print(f"Figure saved to {save_path}")
plt.show()
