"""
Run the ring-resonator direct optimizer and save results.

The optimizer finds ring resonance frequencies and modulation depths that
directly maximize/minimise an objective on the optical spectrum.  Each
dispersion stage contains N_RINGS ring resonators in series; each ring
applies its full complex transfer function across all sideband frequencies.

Saved files
-----------
  <OUT_DIR>/<timestamp>/
    config.json    - hyperparameters
    result.json    - betas, f0_per_stage, ratio, convergence info
    amplitudes.npy - complex output amplitudes
    spectrum.png   - spectrum plot
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt

from ring_optimizer import (
    optimize_ring_spectrum,
    save_ring_opt_result,
    plot_ring_opt_spectrum,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_STAGES        = 3          # number of phase modulators
N_RINGS         = 9          # ring resonators per dispersion stage
BETA_MAX        = 5.0        # maximum modulation depth [rad]

Q_I             = 2e6        # intrinsic Q factor (all rings, fixed)
Q_E             = Q_I / 30   # external Q factor (all rings, fixed)

F_CARRIER       = 193e12     # optical carrier frequency [Hz]
F_MOD           = 50e9       # RF modulation frequency [Hz]
F_RING_BOUND    = F_MOD  # per-ring half-range; ring j centred at F_CARRIER + k*F_MOD
                         # is bounded to [centre - F_RING_BOUND, centre + F_RING_BOUND]

N_MAX           = 20         # harmonic truncation order per PM stage
N_ITER          = 10000        # basin-hopping iterations
SEED            = 1          # integer for reproducibility, or None

OBJECTIVE       = "power"    # "power"       : maximise power at WANTED_HARMONIC
                             # "ratio"       : minimise unwanted/wanted ratio
                             # "power_split" : equal split among POWER_SPLIT_HARMONICS
                             # "arbitrary"   : match ARBITRARY_TARGETS

WANTED_HARMONIC       = 2
POWER_SPLIT_HARMONICS = [-3, 3]
ARBITRARY_TARGETS     = {1: 0.5, 3: 0.25, 5: 0.25}

OUT_DIR = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data"

# ---------------------------------------------------------------------------

config = dict(
    n_stages              = N_STAGES,
    n_rings               = N_RINGS,
    beta_max              = BETA_MAX,
    Q_i                   = Q_I,
    Q_e                   = Q_E,
    f_carrier             = F_CARRIER,
    f_mod                 = F_MOD,
    f_ring_bound          = F_RING_BOUND,
    n_max                 = N_MAX,
    n_iter                = N_ITER,
    seed                  = SEED,
    objective             = OBJECTIVE,
    wanted_harmonic       = WANTED_HARMONIC,
    power_split_harmonics = POWER_SPLIT_HARMONICS if OBJECTIVE == "power_split" else None,
    arbitrary_targets     = ARBITRARY_TARGETS     if OBJECTIVE == "arbitrary"    else None,
)
print("Running ring optimizer with config:", config)

res = optimize_ring_spectrum(
    n_stages          = N_STAGES,
    n_rings           = N_RINGS,
    beta_max          = BETA_MAX,
    Q_i               = Q_I,
    Q_e               = Q_E,
    f_carrier         = F_CARRIER,
    f_mod             = F_MOD,
    f_ring_bound      = F_RING_BOUND,
    n_max             = N_MAX,
    objective         = OBJECTIVE,
    n_iter            = N_ITER,
    seed              = SEED,
    wanted_harmonic   = WANTED_HARMONIC,
    split_harmonics   = POWER_SPLIT_HARMONICS if OBJECTIVE == "power_split" else None,
    arbitrary_targets = ARBITRARY_TARGETS     if OBJECTIVE == "arbitrary"    else None,
)

betas_str = "  ".join(f"beta{i+1}={b:.4f}" for i, b in enumerate(res["betas"]))
print(f"Done.  {betas_str}  ratio={res['ratio']:.6f}  wanted power={res['wanted_power']:.6f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
out_folder = os.path.join(OUT_DIR, timestamp)
save_ring_opt_result(res, config, out_folder)

fig = plot_ring_opt_spectrum(
    res["harmonics"],
    res["amplitudes"],
    wanted_harmonic   = WANTED_HARMONIC,
    arbitrary_targets = ARBITRARY_TARGETS if OBJECTIVE == "arbitrary" else None,
    use_db            = False,
)
fig.savefig(os.path.join(out_folder, "spectrum.png"), dpi=150)
print(f"Results saved to {out_folder}")

plt.show()
