"""
Sweep N_TONES for a single-PM multi-tone SSB optimisation and plot the
maximum conversion efficiency to the wanted harmonic as a function of N_TONES.

Output layout
-------------
  <OUT_DIR>/<batch_name>/
    config.json               - sweep hyperparameters
    sweep_n_tones.json        - per-tone results (n_tones, wanted_power, ratio)
    n_tones_<K>/
      config.json             - per-run hyperparameters
      result.json             - betas, thetas, ratio, wanted_power, convergence
    sweep_linear.png          - conversion efficiency vs N_TONES (linear)
    sweep_dB.png              - conversion efficiency vs N_TONES (dB)
"""

import json
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ssb_multitone import optimize_multitone

from path_utils import local_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WANTED_HARMONIC  = 1       # target sideband index
BETA_MAX         = 5.0     # upper bound on each beta_k (rad)
N_MAX            = 30      # harmonic truncation order (fixed for all calls)
N_TONES_LIST     = [2, 4, 6, 8, 10]   # values of N_TONES to sweep
N_ITER           = 2000     # basin-hopping iterations per run
SEED             = 1       # integer for reproducibility, or None for random
OBJECTIVE        = "power" # "power" or "ratio"
METHOD           = "fft"   # "fft" or "conv"
N_FFT            = 8192    # FFT size (only used when METHOD = "fft")

DB_FLOOR         = -40.0   # dB floor for the log plot
PUBLISHED_PLOT   = False   # if True: strip labels/ticks, save as .svg

OUT_DIR    = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data")
BATCH_NAME = f"n_tones_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# ---------------------------------------------------------------------------

batch_folder = os.path.join(OUT_DIR, BATCH_NAME)
os.makedirs(batch_folder, exist_ok=True)
print(f"Batch folder: {batch_folder}\n")

sweep_config = dict(
    wanted_harmonic = WANTED_HARMONIC,
    beta_max        = BETA_MAX,
    n_max           = N_MAX,
    n_tones_list    = N_TONES_LIST,
    n_iter          = N_ITER,
    seed            = SEED,
    objective       = OBJECTIVE,
    method          = METHOD,
    n_fft           = N_FFT,
)
with open(os.path.join(batch_folder, "config.json"), "w") as f:
    json.dump(sweep_config, f, indent=2)

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
sweep_results = []

for n_tones in N_TONES_LIST:
    print(f"N_TONES={n_tones} ...", end="  ", flush=True)

    run_seed = None if SEED is None else SEED + n_tones
    res = optimize_multitone(
        wanted_harmonic = WANTED_HARMONIC,
        beta_max        = BETA_MAX,
        n_max           = N_MAX,
        n_tones         = n_tones,
        n_iter          = N_ITER,
        seed            = run_seed,
        objective       = OBJECTIVE,
        method          = METHOD,
        n_fft           = N_FFT,
    )

    print(f"wanted_power={res['wanted_power']:.6f}  ratio={res['ratio']:.4f}")

    # Save per-tone results
    run_folder = os.path.join(batch_folder, f"n_tones_{n_tones}")
    os.makedirs(run_folder, exist_ok=True)

    run_config = dict(sweep_config, n_tones=n_tones, seed=run_seed)
    with open(os.path.join(run_folder, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    result_data = dict(
        n_tones      = n_tones,
        betas        = [float(b)  for b  in res["betas"]],
        thetas       = [float(th) for th in res["thetas"]],
        ratio        = float(res["ratio"]),
        wanted_power = float(res["wanted_power"]),
        nit          = int(res["opt_result"].nit),
        nfev         = int(res["opt_result"].nfev),
        message      = str(res["opt_result"].message),
    )
    with open(os.path.join(run_folder, "result.json"), "w") as f:
        json.dump(result_data, f, indent=2)

    sweep_results.append(dict(
        n_tones      = n_tones,
        wanted_power = float(res["wanted_power"]),
        ratio        = float(res["ratio"]),
    ))

with open(os.path.join(batch_folder, "sweep_n_tones.json"), "w") as f:
    json.dump(sweep_results, f, indent=2)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "Arial"
mm_per_inch = 25.4
fp = {"family": "Arial"}
axis_label_fontsize = 10
tick_label_fontsize = 8
spine_lw = 1.5
tick_w   = 1.5

n_tones_arr    = np.array([r["n_tones"]      for r in sweep_results])
wanted_powers  = np.array([r["wanted_power"] for r in sweep_results])
wanted_db      = 10.0 * np.log10(np.maximum(wanted_powers, 10 ** (DB_FLOOR / 10)))

def _make_fig():
    ax_w_in = 80.0 / mm_per_inch
    ax_h_in = 60.0 / mm_per_inch
    left_in, right_in, bottom_in, top_in = 0.60, 0.20, 0.45, 0.25
    fig_w = left_in + ax_w_in + right_in
    fig_h = bottom_in + ax_h_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([left_in / fig_w, bottom_in / fig_h,
                        ax_w_in / fig_w, ax_h_in  / fig_h])
    return fig, ax

def _style(ax):
    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)
        spine.set_zorder(100)
    ax.tick_params(axis="both", which="both", direction="in",
                   width=tick_w, labelsize=tick_label_fontsize)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
    ax.set_xticks(n_tones_arr)
    ax.set_xlabel("Number of RF tones $K$", fontsize=axis_label_fontsize, **fp)
    ax.set_xlim(n_tones_arr[0] - 0.5, n_tones_arr[-1] + 0.5)

# Linear
fig_lin, ax_lin = _make_fig()
ax_lin.plot(n_tones_arr, wanted_powers, "o-", color="steelblue",
            linewidth=1.5, markersize=5)
ax_lin.set_ylabel(f"Power $|A_{{{WANTED_HARMONIC}}}|^2$",
                  fontsize=axis_label_fontsize, **fp)
ax_lin.set_ylim(-0.02, 1.05)
_style(ax_lin)

# dB
fig_db, ax_db = _make_fig()
ax_db.plot(n_tones_arr, wanted_db, "o-", color="steelblue",
           linewidth=1.5, markersize=5)
ax_db.set_ylabel(f"Power [dB]", fontsize=axis_label_fontsize, **fp)
ax_db.set_ylim(DB_FLOOR - 2, 2)
_style(ax_db)

if PUBLISHED_PLOT:
    for ax in (ax_lin, ax_db):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both",
                       labelbottom=False, labelleft=False)
    fig_lin.savefig(os.path.join(batch_folder, "sweep_linear.svg"))
    fig_db.savefig(os.path.join(batch_folder, "sweep_dB.svg"))
else:
    fig_lin.savefig(os.path.join(batch_folder, "sweep_linear.png"), dpi=150)
    fig_db.savefig(os.path.join(batch_folder, "sweep_dB.png"), dpi=150)

print(f"\nSaved figures and results to {batch_folder}")
plt.show()
