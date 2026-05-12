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

Warm-start option (WARM_START_FROM_RING_FREE)
---------------------------------------------
When enabled, the script runs in three steps before the ring optimization:
  1. Ring-free optimization (optimize_ssb) with disp_n_max = N_RINGS // 2,
     using N_ITER_FREE basin-hopping iterations.
  2. Ring approximation: for each ring j (assigned to harmonic k = j - N_RINGS//2)
     find the ring resonance frequency that reproduces the optimized phase shift.
  3. Use the resulting betas and f_0 values as the initial guess x0, then run
     a single L-BFGS-B gradient descent (no basin-hopping) from that point.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from ring_optimizer import (
    optimize_ring_spectrum,
    save_ring_opt_result,
    plot_ring_opt_spectrum,
)
from ssb_spectrum import optimize_ssb
from ring_approximation import solve_ring_for_sideband, compute_ring_spectrum

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_STAGES        = 3          # number of phase modulators
N_RINGS         = 13          # ring resonators per dispersion stage
BETA_MAX        = 5.0        # maximum modulation depth [rad]

Q_I             = 1e6        # intrinsic Q factor (all rings, fixed)
Q_E             = Q_I / 40   # external Q factor (all rings, fixed)

F_CARRIER       = 193e12     # optical carrier frequency [Hz]
F_MOD           = 50e9       # RF modulation frequency [Hz]
F_RING_BOUND    = F_MOD/2  # per-ring half-range; ring j centred at F_CARRIER + k*F_MOD
                         # is bounded to [centre - F_RING_BOUND, centre + F_RING_BOUND]

N_MAX           = 20         # harmonic truncation order per PM stage
N_ITER          = 10000       # basin-hopping iterations (ignored when warm-starting)
SEED            = 2          # integer for reproducibility, or None

OBJECTIVE       = "power"    # "power"       : maximise power at WANTED_HARMONIC
                             # "ratio"       : minimise unwanted/wanted ratio
                             # "power_split" : equal split among POWER_SPLIT_HARMONICS
                             # "arbitrary"   : match ARBITRARY_TARGETS

WANTED_HARMONIC       = 4

POWER_SPLIT_HARMONICS = [-3, 3]
ARBITRARY_TARGETS     = {1: 0.5, 3: 0.25, 5: 0.25}

# Warm-start: ring-free pre-optimization + ring approximation as initial guess.
# When True, runs a ring-free optimization first (N_ITER_FREE iterations) with
# disp_n_max = N_RINGS // 2, converts the result to ring f_0 values via ring
# approximation, then does a single gradient descent from that point.
WARM_START_FROM_RING_FREE         = True
N_ITER_FREE                       = 100   # basin-hopping iterations for the ring-free pre-pass
WARM_START_OPTIMIZE_GLOBAL_PHASE  = True  # sweep global phase to find best ring initial guess
N_WARM_PHASE_STEPS                = 360   # steps over [0, 2*pi)

OUT_DIR = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data"

# ---------------------------------------------------------------------------
# Warm-start: ring-free optimization → ring approximation → initial guess
# ---------------------------------------------------------------------------

x0_ring      = None
local_only   = False

if WARM_START_FROM_RING_FREE:
    disp_n_max_free = N_RINGS // 2
    print(f"Warm-start step 1: ring-free optimization "
          f"(disp_n_max={disp_n_max_free}, n_iter={N_ITER_FREE})...")

    free_res = optimize_ssb(
        wanted_harmonic   = WANTED_HARMONIC,
        beta_max          = BETA_MAX,
        n_max             = N_MAX,
        n_stages          = N_STAGES,
        poly_order        = None,
        n_iter            = N_ITER_FREE,
        seed              = SEED,
        objective         = OBJECTIVE,
        disp_n_max        = disp_n_max_free,
        split_harmonics   = POWER_SPLIT_HARMONICS if OBJECTIVE == "power_split" else None,
        arbitrary_targets = ARBITRARY_TARGETS     if OBJECTIVE == "arbitrary"    else None,
    )

    betas_free_str = "  ".join(f"beta{i+1}={b:.4f}" for i, b in enumerate(free_res["betas"]))
    print(f"  Ring-free done.  {betas_free_str}  "
          f"wanted power={free_res['wanted_power']:.6f}")

    # Build full phi arrays (length 2*N_MAX+1) from the ring-free result.
    # phi_params_list[s] contains the raw phase values for harmonics in
    # [-disp_n_max_free, +disp_n_max_free]; zero outside that window.
    stage_range     = np.arange(-N_MAX, N_MAX + 1)
    n_disp          = N_STAGES - 1
    phi_params_full = []
    for s in range(n_disp):
        raw      = free_res["phi_params_list"][s]
        phi_full = np.zeros(len(stage_range))
        mask     = (stage_range >= -disp_n_max_free) & (stage_range <= disp_n_max_free)
        phi_full[mask] = raw
        phi_params_full.append(phi_full)

    # Step 2 (optional): sweep global phase to minimise ring attenuation.
    if WARM_START_OPTIMIZE_GLOBAL_PHASE:
        print(f"Warm-start step 2a: global phase sweep ({N_WARM_PHASE_STEPS} steps)...")
        _free_opt = {
            "config": {
                "n_stages":       N_STAGES,
                "n_max":          N_MAX,
                "disp_n_max":     disp_n_max_free,
                "wanted_harmonic": WANTED_HARMONIC,
            },
            "betas":          list(free_res["betas"]),
            "phi_params_full": phi_params_full,
            "harmonics":      stage_range,
        }
        inner_mask      = np.abs(stage_range) <= disp_n_max_free
        phi_sweep_warm  = np.linspace(0, 2 * np.pi, N_WARM_PHASE_STEPS, endpoint=False)
        best_phi_global = 0.0
        best_metric     = -np.inf
        for phi_g in phi_sweep_warm:
            shifted = []
            for phi in phi_params_full:
                p = phi.copy()
                p[inner_mask] += phi_g
                shifted.append(p)
            _free_opt_shifted = dict(_free_opt)
            _free_opt_shifted["phi_params_full"] = shifted
            res_g  = compute_ring_spectrum(
                _free_opt_shifted, Q_i=Q_I, Q_e=Q_E,
                f_carrier=F_CARRIER, f_mod=F_MOD,
                f_search_radius=F_MOD / 2,
            )
            power  = np.abs(res_g["amplitudes"]) ** 2
            idx    = np.where(res_g["harmonics"] == WANTED_HARMONIC)[0]
            metric = float(power[idx[0]]) if len(idx) else float(power.sum())
            if metric > best_metric:
                best_metric     = metric
                best_phi_global = phi_g
        # Apply best global phase to inner harmonics only
        new_phi_params_full = []
        for phi in phi_params_full:
            p = phi.copy()
            p[inner_mask] += best_phi_global
            new_phi_params_full.append(p)
        phi_params_full = new_phi_params_full
        print(f"  Best global phase: {np.degrees(best_phi_global):.1f} deg  "
              f"ring wanted power: {best_metric:.6f}")

    # Step 2b: convert to ring f_0 values.
    # Ring j in the ring optimizer is assigned to harmonic k = j - N_RINGS//2.
    # For each (stage, ring j), find the ring resonance frequency that produces
    # the phase shift phi_full[k] at sideband frequency F_CARRIER + k*F_MOD.
    # If phi ≈ 0, place the ring at its centre frequency (minimal interaction).
    step_label = "2b" if WARM_START_OPTIMIZE_GLOBAL_PHASE else "2"
    print(f"Warm-start step {step_label}: ring approximation → initial f_0 values...")
    ring_k    = np.arange(N_RINGS) - N_RINGS // 2   # harmonic index per ring
    f0_blocks = []
    for s in range(n_disp):
        phi_full = phi_params_full[s]
        f0s = []
        for j in range(N_RINGS):
            k         = int(ring_k[j])
            phi_idx   = k + N_MAX              # index into stage_range (starts at -N_MAX)
            phi_val   = float(phi_full[phi_idx])
            centre_f  = float(F_CARRIER + k * F_MOD)
            if abs(phi_val) > 1e-9:
                ring = solve_ring_for_sideband(
                    phi_target = phi_val,
                    f_sideband = centre_f,
                    Q_i        = Q_I,
                    Q_e        = Q_E,
                )
                # Clip to the ring optimizer's allowed search range
                f0 = float(np.clip(ring.f_0,
                                   centre_f - F_RING_BOUND,
                                   centre_f + F_RING_BOUND))
            else:
                f0 = centre_f
            f0s.append(f0)
        f0_blocks.append(np.array(f0s))

    x0_ring    = np.concatenate([np.array(free_res["betas"])] + f0_blocks)
    local_only = True
    print("  Initial guess ready — will run gradient descent only.")

# ---------------------------------------------------------------------------
# Run ring optimization
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
    warm_start            = WARM_START_FROM_RING_FREE,
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
    x0                = x0_ring,
    local_only        = local_only,
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
