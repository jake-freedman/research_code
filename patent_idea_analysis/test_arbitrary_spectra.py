"""
test_arbitrary_spectra.py

Generate N random target spectra (each a random subset of sidebands with uniform
random weights) and run the "arbitrary" optimizer on each.

Saved files
-----------
  <OUT_DIR>/<batch_name>/
    config.json        - all configuration parameters for this run
    targets.json       - the randomly generated (or loaded) target spectra
    results.json       - betas, phi params, achieved powers, and metrics per test
    per_test/
      test_001.png     - final spectrum overlaid with dashed target markers
      test_002.png
      ...
    summary.png        - total captured power bar chart + achieved-vs-target scatter

Re-running with the same targets
---------------------------------
  Set RESUME_FROM to a previous batch folder path.  targets.json is loaded from
  that folder; the current optimization config is used for the new run.  Results
  are saved to a fresh timestamped batch folder so the original is never overwritten.
"""

import json
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ssb_spectrum import optimize_ssb, plot_optical_spectrum
from ring_optimizer import optimize_ring_spectrum

from path_utils import local_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TESTS        = 10       # number of random target spectra to generate and test
N_SIDEBANDS    = 6        # number of sidebands per random target (fixed)
HARMONIC_RANGE = (-6, 6)  # inclusive range to sample harmonics from
INCLUDE_ZERO   = True    # if False, harmonic 0 is excluded from the sampling pool

BETA_MAX   = 5.0
N_MAX      = 20
N_STAGES   = 4
POLY_ORDER = None
N_ITER     = 1000
DISP_N_MAX = 8
PHI_MAX    = None
SEED       = 42           # controls target generation and per-test optimizer seeds; None = random

# --- Ring-resonator optimizer (only used when USE_RING_OPT = True) ---
# When True, optimize_ring_spectrum is called directly instead of optimize_ssb.
# The saved results.json is compatible with plot_arbitrary_grid.py.
USE_RING_OPT     = True
RING_N_RINGS     = 9          # ring resonators per dispersion stage
RING_Q_I         = 2e6         # intrinsic Q factor of every ring
RING_Q_E         = 1e6 / 40   # external Q factor (overcoupled: Q_E < Q_I)
RING_F_CARRIER   = 193e12      # optical carrier frequency [Hz]
RING_F_MOD       = 50e9        # modulation frequency [Hz]
RING_F_RING_BOUND = 50e9       # per-ring search half-width [Hz]

USE_DB   = True
DB_FLOOR = -40.0

OUT_DIR    = local_path(r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\arb_test_with_6_sidebands_10_trials")
BATCH_NAME = ""           # "" = auto-generate as arbitrary_test_<timestamp>

# Set to a previous batch folder to reuse its randomly generated targets.
RESUME_FROM = None        # e.g. r"C:\...\arbitrary_test_20260425_120000"

# Figure settings
SAVE_DPI  = 150
N_DISPLAY = abs(HARMONIC_RANGE[1])   # harmonics shown each side on spectrum plots

# ---------------------------------------------------------------------------
# Batch folder
# ---------------------------------------------------------------------------
if not BATCH_NAME:
    BATCH_NAME = f"arbitrary_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
batch_folder = os.path.join(OUT_DIR, BATCH_NAME)
os.makedirs(batch_folder, exist_ok=True)
print(f"Batch folder: {batch_folder}\n")

# ---------------------------------------------------------------------------
# Generate or load targets
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

if RESUME_FROM:
    with open(os.path.join(RESUME_FROM, "targets.json")) as f:
        targets = json.load(f)
    print(f"Loaded {len(targets)} targets from {RESUME_FROM}\n")
else:
    pool = [h for h in range(HARMONIC_RANGE[0], HARMONIC_RANGE[1] + 1)
            if INCLUDE_ZERO or h != 0]
    if N_SIDEBANDS > len(pool):
        raise ValueError(
            f"N_SIDEBANDS={N_SIDEBANDS} exceeds the available pool size {len(pool)}. "
            "Widen HARMONIC_RANGE or set INCLUDE_ZERO=True."
        )
    targets = []
    for _ in range(N_TESTS):
        chosen  = sorted(int(h) for h in rng.choice(pool, size=N_SIDEBANDS, replace=False))
        weights = [float(w) for w in rng.uniform(0.0, 1.0, size=N_SIDEBANDS)]
        targets.append({"harmonics": chosen, "weights": weights})

with open(os.path.join(batch_folder, "targets.json"), "w") as f:
    json.dump(targets, f, indent=2)

# ---------------------------------------------------------------------------
# Save config
# ---------------------------------------------------------------------------
config_data = dict(
    n_tests        = len(targets),
    n_sidebands    = N_SIDEBANDS,
    harmonic_range = list(HARMONIC_RANGE),
    include_zero   = INCLUDE_ZERO,
    beta_max       = BETA_MAX,
    n_max          = N_MAX,
    n_stages       = N_STAGES,
    poly_order     = POLY_ORDER,
    n_iter         = N_ITER,
    disp_n_max     = DISP_N_MAX,
    phi_max        = PHI_MAX,
    seed           = SEED,
    resumed_from   = RESUME_FROM,
    use_ring_opt      = USE_RING_OPT,
    ring_n_rings      = RING_N_RINGS      if USE_RING_OPT else None,
    ring_Q_i          = RING_Q_I          if USE_RING_OPT else None,
    ring_Q_e          = RING_Q_E          if USE_RING_OPT else None,
    ring_f_carrier    = RING_F_CARRIER    if USE_RING_OPT else None,
    ring_f_mod        = RING_F_MOD        if USE_RING_OPT else None,
    ring_f_ring_bound = RING_F_RING_BOUND if USE_RING_OPT else None,
)
with open(os.path.join(batch_folder, "config.json"), "w") as f:
    json.dump(config_data, f, indent=2)

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "Arial"
_cmap = matplotlib.colormaps["rainbow_r"]
_norm = matplotlib.colors.Normalize(vmin=-N_DISPLAY, vmax=N_DISPLAY)


def _overlay_targets(ax, norm_tgt: dict):
    """Dashed-circle + dashed-stem overlay for normalised target powers."""
    y_floor = ax.get_ylim()[0]
    fig     = ax.get_figure()
    ax_pos  = ax.get_position()
    ax_w_in = ax_pos.width  * fig.get_figwidth()
    ax_h_in = ax_pos.height * fig.get_figheight()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    r_in    = np.sqrt(80 / np.pi) / 72.0 * 1.5   # ~1.5x the solid scatter-marker radius
    r_x     = r_in / ax_w_in * (x_max - x_min)
    r_y     = r_in / ax_h_in * (y_max - y_min)
    theta   = np.linspace(0, 2 * np.pi, 120)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for h, tp in norm_tgt.items():
        if abs(h) > N_DISPLAY:
            continue
        color    = _cmap(_norm(h))
        y_target = (10.0 * np.log10(max(tp, 10 ** (DB_FLOOR / 10)))
                    if USE_DB else float(tp))
        ax.vlines(h, y_floor, y_target, linewidth=2, colors=[color],
                  linestyles="dashed", zorder=4)
        ax.plot(h + r_x * cos_t, y_target + r_y * sin_t,
                linestyle="--", color=color, linewidth=1.5, zorder=5)




# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------
per_test_dir = os.path.join(batch_folder, "per_test")
os.makedirs(per_test_dir, exist_ok=True)

print(f"Running {len(targets)} optimizations ...\n")
all_results = []

for i, target in enumerate(targets):
    harmonics = target["harmonics"]
    weights   = target["weights"]
    arb_dict  = {h: w for h, w in zip(harmonics, weights)}

    # Highest-weight sideband used as optimizer's "wanted_harmonic" bookkeeping
    wanted_h = harmonics[int(np.argmax(weights))]
    opt_seed = None if SEED is None else int(SEED) * 1000 + i

    print(f"  [{i+1}/{len(targets)}]  harmonics={harmonics}  "
          f"weights={[f'{w:.3f}' for w in weights]}", end="", flush=True)

    if USE_RING_OPT:
        res = optimize_ring_spectrum(
            n_stages          = N_STAGES,
            n_rings           = RING_N_RINGS,
            beta_max          = BETA_MAX,
            Q_i               = RING_Q_I,
            Q_e               = RING_Q_E,
            f_carrier         = RING_F_CARRIER,
            f_mod             = RING_F_MOD,
            f_ring_bound      = RING_F_RING_BOUND,
            n_max             = N_MAX,
            objective         = "arbitrary",
            n_iter            = N_ITER,
            seed              = opt_seed,
            wanted_harmonic   = wanted_h,
            arbitrary_targets = arb_dict,
        )
        spec_harmonics  = res["harmonics"]
        spec_amplitudes = res["amplitudes"]
        f0_per_stage    = res["f0_per_stage"]
        betas_to_save   = [float(b) for b in res["betas"]]
        phi_params_to_save = []
    else:
        res = optimize_ssb(
            wanted_harmonic   = wanted_h,
            beta_max          = BETA_MAX,
            n_max             = N_MAX,
            n_stages          = N_STAGES,
            poly_order        = POLY_ORDER,
            n_iter            = N_ITER,
            seed              = opt_seed,
            objective         = "arbitrary",
            phi_max           = PHI_MAX,
            disp_n_max        = DISP_N_MAX,
            arbitrary_targets = arb_dict,
        )
        spec_harmonics     = res["harmonics"]
        spec_amplitudes    = res["amplitudes"]
        f0_per_stage       = None
        betas_to_save      = [float(b) for b in res["betas"]]
        phi_params_to_save = [p.tolist() for p in res["phi_params_list"]]

    power   = np.abs(spec_amplitudes) ** 2
    total_w = sum(weights)
    norm_tgt = {h: w / total_w for h, w in zip(harmonics, weights)}

    achieved = {}
    for h in harmonics:
        idx = np.where(spec_harmonics == h)[0]
        achieved[h] = float(power[idx[0]]) if len(idx) else 0.0

    total_achieved = float(sum(achieved.values()))
    split_mse      = float(np.mean([(achieved.get(h, 0.0) - norm_tgt[h]) ** 2
                                     for h in harmonics]))

    print(f"  →  total_power={total_achieved:.4f}  split_mse={split_mse:.6f}")

    # Per-test spectrum figure
    fig_spec, ax_spec = plot_optical_spectrum(
        spec_harmonics, spec_amplitudes,
        db=USE_DB, db_floor=DB_FLOOR,
        width_mm=80, n_display=N_DISPLAY,
    )
    ax_spec.set_title(
        f"Test {i+1}  |  targets={harmonics}  |  "
        f"total_power={total_achieved:.3f}  mse={split_mse:.4f}",
        fontsize=8, fontfamily="Arial",
    )
    _overlay_targets(ax_spec, norm_tgt)
    fig_spec.savefig(os.path.join(per_test_dir, f"test_{i+1:03d}.png"), dpi=SAVE_DPI)
    plt.close(fig_spec)

    # Save spectrum powers up to N_MAX so downstream plot scripts can choose their own display range
    spectrum_powers = {
        str(int(h)): float(p)
        for h, p in zip(spec_harmonics, power)
        if abs(h) <= N_MAX
    }

    record = dict(
        test_idx             = i,
        target_harmonics     = harmonics,
        target_weights_raw   = weights,
        target_powers_norm   = {str(h): v for h, v in norm_tgt.items()},
        achieved_powers      = {str(h): v for h, v in achieved.items()},
        total_achieved_power = total_achieved,
        split_mse            = split_mse,
        betas                = betas_to_save,
        phi_params           = phi_params_to_save,
        spectrum_powers      = spectrum_powers,
    )
    if f0_per_stage is not None:
        record["ring_f0_per_stage"] = f0_per_stage
    all_results.append(record)

with open(os.path.join(batch_folder, "results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved results.json")

# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------
mm_per_inch  = 25.4
n_tests_done = len(all_results)

# Two panels side by side
panel_w_mm  = 90.0
panel_h_mm  = 70.0
left_in     = 0.65
mid_in      = 0.55   # gap between panels (also left margin of right panel)
right_in    = 0.20
bottom_in   = 0.50
top_in      = 0.40

ax_w_in = panel_w_mm / mm_per_inch
ax_h_in = panel_h_mm / mm_per_inch
fig_w   = left_in + ax_w_in + mid_in + ax_w_in + right_in
fig_h   = bottom_in + ax_h_in + top_in

fig_sum = plt.figure(figsize=(fig_w, fig_h))

ax_bar = fig_sum.add_axes([
    left_in / fig_w,
    bottom_in / fig_h,
    ax_w_in  / fig_w,
    ax_h_in  / fig_h,
])
ax_sct = fig_sum.add_axes([
    (left_in + ax_w_in + mid_in) / fig_w,
    bottom_in / fig_h,
    ax_w_in  / fig_w,
    ax_h_in  / fig_h,
])

fp = {"family": "Arial"}

# --- Left panel: total captured power per test ---
test_labels    = [str(r["test_idx"] + 1) for r in all_results]
total_powers   = [r["total_achieved_power"]  for r in all_results]
split_mses     = [r["split_mse"]             for r in all_results]

bar_colors = [_cmap(_norm(r["target_harmonics"][int(np.argmax(r["target_weights_raw"]))])
              ) for r in all_results]

bars = ax_bar.bar(range(1, n_tests_done + 1), total_powers,
                  color=bar_colors, edgecolor="white", linewidth=0.5, zorder=2)
ax_bar.set_xlabel("Test index", fontsize=10, **fp)
ax_bar.set_ylabel("Total power in target sidebands", fontsize=9, **fp)
ax_bar.set_xlim(0.3, n_tests_done + 0.7)
ax_bar.set_ylim(0, 1.1)
ax_bar.set_xticks(range(1, n_tests_done + 1))
ax_bar.axhline(1.0, color="black", linewidth=1, linestyle="--", zorder=1)
ax_bar.grid(axis="y", zorder=0)
for spine in ax_bar.spines.values():
    spine.set_linewidth(2)
ax_bar.tick_params(axis="both", which="both", direction="in", width=2, labelsize=7)
for lbl in ax_bar.get_xticklabels() + ax_bar.get_yticklabels():
    lbl.set_fontfamily("Arial")

# --- Right panel: achieved vs target normalised power (all sideband-test pairs) ---
for r in all_results:
    norm_tgt = {int(h): v for h, v in r["target_powers_norm"].items()}
    achieved = {int(h): v for h, v in r["achieved_powers"].items()}
    for h in norm_tgt:
        color = _cmap(_norm(h))
        ax_sct.scatter(norm_tgt[h], achieved.get(h, 0.0),
                       s=50, color=color, edgecolors="white",
                       linewidths=0.4, zorder=3)

ax_sct.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--",
            zorder=1, label="ideal")
ax_sct.set_xlabel("Target power fraction", fontsize=10, **fp)
ax_sct.set_ylabel("Achieved power", fontsize=10, **fp)
ax_sct.set_xlim(-0.02, 1.05)
ax_sct.set_ylim(-0.02, 1.05)
ax_sct.grid(zorder=0)
ax_sct.legend(fontsize=8, frameon=False)
for spine in ax_sct.spines.values():
    spine.set_linewidth(2)
ax_sct.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
for lbl in ax_sct.get_xticklabels() + ax_sct.get_yticklabels():
    lbl.set_fontfamily("Arial")

fig_sum.suptitle(
    f"Arbitrary-target test  |  N={n_tests_done}  |  {N_SIDEBANDS} sidebands  |  "
    f"stages={N_STAGES}  β_max={BETA_MAX}",
    fontsize=9, fontfamily="Arial", y=(bottom_in + ax_h_in + top_in * 0.7) / fig_h,
)

summary_path = os.path.join(batch_folder, "summary.png")
fig_sum.savefig(summary_path, dpi=SAVE_DPI)
print(f"Saved {summary_path}")

plt.show()

print(f"\nAll done.  Batch folder:\n  {batch_folder}")
