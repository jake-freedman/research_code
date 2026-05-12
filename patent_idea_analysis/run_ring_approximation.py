"""
Run the ring-resonator approximation of an optimizer result and plot the comparison.

Workflow
--------
1. Load an optimizer result folder (config.json, result.json, phi_params.npy).
2. For each dispersion stage and each harmonic, find a ring resonance frequency
   that reproduces the optimizer's phase shift at that sideband frequency.
3. Recompute the optical spectrum using the complex ring transfer functions
   (which include both amplitude attenuation and phase shift).
4. Optionally save the result and plot the comparison.

Global phase sweep (OPTIMIZE_GLOBAL_PHASE)
------------------------------------------
A global phase rotation phi_global added to the input is equivalent to adding
phi_global to every ring's phase target in every stage.  This shifts each ring
resonance closer to or further from its target sideband, changing how much power
is attenuated.  When enabled, the script sweeps phi_global over [0, 2*pi) and
picks the value that maximises total output power in the ring approximation.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from graphics import (
    BLUE2, DARKGRAY2, RED2, GREEN2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)
from ssb_spectrum import compute_optical_spectrum_general
from ring_approximation import (
    load_optimizer_result,
    compute_ring_spectrum,
    save_ring_result,
    plot_ring_vs_optimizer,
    plot_combined_ring_transmission,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to a folder produced by run_optimization.py (contains config.json etc.)
OPT_FOLDER = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\5rings_targets\20260507_143539"

# Ring Q factors (same for every ring in every stage)
Q_I = 1e6   # intrinsic Q
Q_E = Q_I / 40   # external (waveguide) Q — set Q_E < Q_I for overcoupled rings

# Sideband / carrier geometry
F_CARRIER = 193e12   # optical carrier frequency [Hz]
F_MOD     = 50e9     # modulation frequency [Hz]

# Ring solver search radius [Hz] — must be >> ring linewidth
F_SEARCH_RADIUS = F_MOD / 2

# Global phase sweep: find the input phase rotation that minimises ring attenuation.
# A global phase on the input shifts all ring phase targets uniformly, moving
# resonances closer or farther from their sidebands and changing power loss.
OPTIMIZE_GLOBAL_PHASE  = True    # set False to skip the sweep
N_GLOBAL_PHASE_STEPS   = 720     # number of steps over [0, 2*pi); 72 = every 5 deg

# Plot settings
N_DISPLAY = 10        # show harmonics -N_DISPLAY … +N_DISPLAY
USE_DB    = False     # True: power in dB; False: linear
DB_FLOOR  = -5.0     # dB floor (only used when USE_DB=True)
AX_W_MM   = 90.0
AX_H_MM   = 70.0

# Set to a folder path to save results, or None to skip saving
SAVE_FOLDER = None   # e.g. OPT_FOLDER + "_rings"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

opt = load_optimizer_result(OPT_FOLDER)

print(f"Loaded optimizer result from: {OPT_FOLDER}")
print(f"  n_stages    = {opt['config']['n_stages']}")
print(f"  n_max       = {opt['config']['n_max']}")
print(f"  n_disp_stgs = {len(opt['phi_params_full'])}")
print(f"  betas       = {[f'{b:.4f}' for b in opt['betas']]}")

_wanted_harmonic = opt["config"].get("wanted_harmonic", None)

# ---------------------------------------------------------------------------
# Global phase sweep
# ---------------------------------------------------------------------------

def _with_global_phase(opt_result: dict, phi_global: float) -> dict:
    """Return a shallow copy of opt_result with phi_global added to every
    phi_params_full entry (equivalent to rotating the input field by phi_global).

    Harmonics inside disp_n_max get phi[n] + phi_global; harmonics outside
    (which had phi[n] = 0) get phi_global, keeping all rings at a consistent
    global phase rather than zero.
    """
    harmonics  = opt_result["harmonics"]
    disp_n_max = opt_result["config"].get("disp_n_max", None)
    new_phis = []
    for phi in opt_result["phi_params_full"]:
        new_phi = phi + phi_global
        if disp_n_max is not None:
            new_phi = new_phi.copy()
            new_phi[np.abs(harmonics) > disp_n_max] = 0.0
        new_phis.append(new_phi)
    modified = dict(opt_result)
    modified["phi_params_full"] = new_phis
    return modified


def _opt_wanted_power(phi_global: float) -> float:
    """Ideal (ring-free) wanted power at this global phase rotation."""
    modified = _with_global_phase(opt, phi_global)
    config   = modified["config"]
    n_max    = int(config["n_max"])
    n_stages = int(config["n_stages"])
    h, amps  = compute_optical_spectrum_general(
        modified["betas"], modified["phi_params_full"],
        n_max=n_max, truncate=(n_stages > 2),
    )
    if _wanted_harmonic is None:
        return float("nan")
    idx = np.where(h == _wanted_harmonic)[0]
    return float(np.abs(amps[idx[0]]) ** 2) if len(idx) else float("nan")


def _ring_total_power(phi_global: float) -> tuple[float, float]:
    """Run the ring approximation at this global phase and return
    (total_output_power, wanted_power).  wanted_power is NaN if no wanted_harmonic."""
    res = compute_ring_spectrum(
        _with_global_phase(opt, phi_global),
        Q_i=Q_I, Q_e=Q_E,
        f_carrier=F_CARRIER, f_mod=F_MOD,
        f_search_radius=F_SEARCH_RADIUS,
    )
    power       = np.abs(res["amplitudes"]) ** 2
    total_power = float(power.sum())
    if _wanted_harmonic is not None:
        idx          = np.where(res["harmonics"] == _wanted_harmonic)[0]
        wanted_power = float(power[idx[0]]) if len(idx) else float("nan")
    else:
        wanted_power = float("nan")
    return total_power, wanted_power


best_phi_global = 0.0

if OPTIMIZE_GLOBAL_PHASE:
    phi_sweep   = np.linspace(0, 2 * np.pi, N_GLOBAL_PHASE_STEPS, endpoint=False)
    best_total  = -np.inf
    best_wanted = float("nan")

    print(f"Sweeping global phase rotation ({N_GLOBAL_PHASE_STEPS} steps)...")
    sweep_wanted     = []
    sweep_total      = []
    sweep_opt_wanted = []
    for phi_g in phi_sweep:
        tot, wan = _ring_total_power(phi_g)
        sweep_total.append(tot)
        sweep_wanted.append(wan)
        sweep_opt_wanted.append(_opt_wanted_power(phi_g))
        # Maximise wanted power when available, otherwise total power
        metric     = wan  if (_wanted_harmonic is not None and not np.isnan(wan)) else tot
        ref_metric = best_wanted if (_wanted_harmonic is not None and not np.isnan(best_wanted)) else best_total
        if metric > ref_metric:
            best_total      = tot
            best_wanted     = wan
            best_phi_global = phi_g

    print(f"  Best global phase : {np.degrees(best_phi_global):.1f} deg")
    print(f"  Total output power: {best_total:.6f}"
          + (f"  wanted power: {best_wanted:.6f}" if not np.isnan(best_wanted) else ""))

    # Plot wanted power (circles) and total power (line) vs global phase
    matplotlib.rcParams["font.family"] = "Arial"
    phi_deg = np.degrees(phi_sweep)
    mm_per_inch = 25.4
    ax_w_in = 80.0 / mm_per_inch
    ax_h_in = 55.0 / mm_per_inch
    left_in, right_in, bottom_in, top_in = 0.55, 0.15, 0.45, 0.20
    fig_sweep = plt.figure(figsize=(left_in + ax_w_in + right_in,
                                    bottom_in + ax_h_in + top_in))
    ax_sweep  = fig_sweep.add_axes([left_in / (left_in + ax_w_in + right_in),
                                    bottom_in / (bottom_in + ax_h_in + top_in),
                                    ax_w_in  / (left_in + ax_w_in + right_in),
                                    ax_h_in  / (bottom_in + ax_h_in + top_in)])

    ax_sweep.plot(phi_deg, sweep_total, color=DARKGRAY2, linewidth=1.2,
                  linestyle="--", label="Total power", zorder=2)
    if _wanted_harmonic is not None and not all(np.isnan(sweep_wanted)):
        ax_sweep.scatter(phi_deg, sweep_wanted, color=BLUE2, s=8,
                         linewidths=0, zorder=3,
                         label=f"Ring wanted power (h={_wanted_harmonic})")
    if _wanted_harmonic is not None and not all(np.isnan(sweep_opt_wanted)):
        ax_sweep.plot(phi_deg, sweep_opt_wanted, color=GREEN2, linewidth=1.2,
                      zorder=2, label=f"Ideal wanted power (h={_wanted_harmonic})")
    ax_sweep.axvline(np.degrees(best_phi_global), color=RED2, linewidth=1.2,
                     linestyle=":", zorder=4,
                     label=f"Best: {np.degrees(best_phi_global):.1f}°")
    ax_sweep.set_xlabel("Global phase rotation [deg]", fontsize=axis_label_fontsize,
                        fontfamily="Arial")
    ax_sweep.set_ylabel("Output power", fontsize=axis_label_fontsize, fontfamily="Arial")
    ax_sweep.set_xlim(0, 360)
    ax_sweep.legend(fontsize=tick_label_fontsize, frameon=False)
    for spine in ax_sweep.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax_sweep.tick_params(axis="both", which="both",
                         direction=tick_direction, width=tick_width,
                         labelsize=tick_label_fontsize)
    for lbl in ax_sweep.get_xticklabels() + ax_sweep.get_yticklabels():
        lbl.set_fontfamily("Arial")

    opt_original = opt   # keep unmodified copy for optimizer reference in plot
    opt = _with_global_phase(opt, best_phi_global)
else:
    opt_original = opt

# ---------------------------------------------------------------------------
# Ring approximation at chosen global phase
# ---------------------------------------------------------------------------

ring_res = compute_ring_spectrum(
    opt,
    Q_i=Q_I,
    Q_e=Q_E,
    f_carrier=F_CARRIER,
    f_mod=F_MOD,
    f_search_radius=F_SEARCH_RADIUS,
)

total_power = float(np.sum(np.abs(ring_res["amplitudes"]) ** 2))
print("Ring approximation done.")
print(f"  Output harmonics span: [{ring_res['harmonics'][0]}, {ring_res['harmonics'][-1]}]")
print(f"  Total output power   : {total_power:.6f}")

if SAVE_FOLDER:
    save_ring_result(ring_res, opt, SAVE_FOLDER, Q_i=Q_I, Q_e=Q_E,
                     f_carrier=F_CARRIER, f_mod=F_MOD)
    print(f"Saved ring result to: {SAVE_FOLDER}")

fig = plot_ring_vs_optimizer(
    opt_result=opt_original,
    ring_result=ring_res,
    n_display=N_DISPLAY,
    ax_w_mm=AX_W_MM,
    ax_h_mm=AX_H_MM,
    use_db=USE_DB,
    db_floor=DB_FLOOR,
)

fig_rings = plot_combined_ring_transmission(
    rings_per_stage=ring_res["rings_per_stage"],
    f_carrier=F_CARRIER,
    f_mod=F_MOD,
    n_max=opt["config"]["n_max"],
    power_db=USE_DB,
    db_floor=DB_FLOOR,
    phase_min=0
)

plt.show()
