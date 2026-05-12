"""
Load and visualize a previous optimisation result.

Supports both result formats:
  - run_optimization.py                    (single-tone cascaded PMs)
  - run_multitone_cascaded_optimization.py (multi-tone cascaded PMs)

The format is auto-detected from result.json.
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from scipy.special import jv
from ssb_spectrum import compute_optical_spectrum_general, plot_optical_spectrum
from ssb_multitone_cascaded import compute_cascaded_multitone_spectrum
from ssb_multitone import compute_multitone_spectrum_fft, compute_multitone_spectrum_conv
from ring_resonator import RingResonator
from ring_approximation import plot_combined_ring_transmission
from graphics import (
    LIGHTBLUE2, BLUE2, GREEN2, VIOLET2, RED2, DARKBLUE2, DARKGREEN2, ORANGE2, PINK2,
)

# ---------------------------------------------------------------------------
# Paste the path to the result folder here
# ---------------------------------------------------------------------------
RESULT_FOLDER  = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\20260507_133331"
USE_DB         = False   # set True to show spectrum in dB
DB_FLOOR       = -60.0  # dB floor [only used when USE_DB = True]
PHASE_MOD_2PI  = True   # set True to wrap dispersion phase profiles to [0, 2*pi)
MINIMIZE_PHASE     = False  # set True to also show the minimum-total-phase equivalent profile
PLOT_INTERMEDIATE  = True   # set True to plot spectrum after each PM stage
N_DISPLAY          = 6     # number of harmonic orders shown on each side of spectrum plots
SAVE_DIR       = None # r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media" # r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media\harmonic6"   # set to a folder path to save all figures as PNGs, e.g. r"C:\path\to\figures"
SAVE_DPI       = 150    # resolution for saved figures
PUBLISHED_PLOT = False  # if True: strip all labels/ticks/legend/title from spectrum, save as .svg
# ---------------------------------------------------------------------------

# Colors (from graphics.py "2" palette) cycled across dispersion stages
_STAGE_COLORS = [LIGHTBLUE2, GREEN2, VIOLET2, RED2, DARKBLUE2, DARKGREEN2, ORANGE2, PINK2]

if not RESULT_FOLDER:
    raise ValueError("Set RESULT_FOLDER to the path of a saved result.")

if SAVE_DIR:
    os.makedirs(SAVE_DIR, exist_ok=True)

def _save(fig, name: str):
    if SAVE_DIR:
        path = os.path.join(SAVE_DIR, f"{name}.png")
        fig.savefig(path, dpi=SAVE_DPI)
        print(f"Saved {path}")


def _overlay_targets(ax, target_powers: dict):
    """Overlay target power levels as transparent filled circles + dashed stems."""
    y_floor = ax.get_ylim()[0]
    for h, tp in target_powers.items():
        if abs(h) > N_DISPLAY:
            continue
        y_target = (10.0 * np.log10(max(tp, 10 ** (DB_FLOOR / 10)))
                    if USE_DB else float(tp))
        ax.vlines(h, y_floor, y_target, linewidth=2, colors=["black"], zorder=4)
        ax.scatter(h, y_target, s=200, facecolors="none", edgecolors="black",
                   linewidths=1.5, zorder=5)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
with open(os.path.join(RESULT_FOLDER, "config.json")) as f:
    config = json.load(f)

IS_SWEEP    = os.path.exists(os.path.join(RESULT_FOLDER, "sweep_disp_n_max.json"))
IS_RING_OPT = "n_rings" in config

if not IS_SWEEP:
    with open(os.path.join(RESULT_FOLDER, "result.json")) as f:
        result = json.load(f)
    IS_MULTITONE = "betas1" in result and not IS_RING_OPT
else:
    result = None
    IS_MULTITONE = False

n_max       = config["n_max"]
poly_order  = config.get("poly_order", None)
disp_n_max  = config.get("disp_n_max", None)
stage_range = np.arange(-n_max, n_max + 1)

# Build target power fractions for "arbitrary" / "power_split" objectives.
_objective = config.get("objective", "ratio")
_target_powers: dict[int, float] | None = None
if _objective == "arbitrary":
    raw = config.get("arbitrary_targets") or {}
    weights = {int(k): float(v) for k, v in raw.items()}
    _total = sum(weights.values())
    if _total > 0:
        _target_powers = {h: w / _total for h, w in weights.items()}
elif _objective == "power_split":
    hs = config.get("power_split_harmonics") or []
    if hs:
        _target_powers = {h: 1.0 / len(hs) for h in hs}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_min_phase(phi_arr: np.ndarray) -> np.ndarray:
    """Global 2*pi*k shift that minimises RMS phase, preserving profile shape."""
    k_opt = int(np.round(-np.mean(phi_arr) / (2 * np.pi)))
    return phi_arr + 2 * np.pi * k_opt


def make_phase_axes(title: str):
    """Create a new figure and axes sized for phase profile plots."""
    matplotlib.rcParams["font.family"] = "Arial"
    mm_per_inch = 25.4
    left_in, right_in, bottom_in, top_in = 0.60, 0.30, 0.45, 0.25
    ax_w_in = 250.0 / mm_per_inch
    ax_h_in =  60.0 / mm_per_inch
    fig = plt.figure(figsize=(left_in + ax_w_in + right_in,
                               bottom_in + ax_h_in + top_in))
    ax = fig.add_axes([
        left_in / (left_in + ax_w_in + right_in),
        bottom_in / (bottom_in + ax_h_in + top_in),
        ax_w_in  / (left_in + ax_w_in + right_in),
        ax_h_in  / (bottom_in + ax_h_in + top_in),
    ])
    ax.set_title(title, fontsize=10, fontfamily="Arial")
    return fig, ax


def _draw_phase_bars(ax, phi_items, ylabel):
    """Draw grouped bars for each (phi_arr, label) pair onto ax."""
    font_props = {"family": "Arial"}
    n         = len(phi_items)
    bar_width = 0.6 / n
    offsets   = (np.arange(n) - (n - 1) / 2) * bar_width

    for (phi_arr, lbl), color, offset in zip(phi_items, _STAGE_COLORS, offsets):
        phi_plot = phi_arr % (2 * np.pi) if PHASE_MOD_2PI else phi_arr
        ax.bar(stage_range + offset, phi_plot, width=bar_width,
               color=color, label=lbl,
               edgecolor="black", linewidth=1.0)

    ax.set_xlabel("Harmonic index  $n$", fontsize=10, **font_props)
    ax.set_ylabel(ylabel, fontsize=10, **font_props)
    ax.set_xlim(-N_DISPLAY - 0.5, N_DISPLAY + 0.5)
    even_ticks = np.arange(-(N_DISPLAY - N_DISPLAY % 2), N_DISPLAY + 1, 2)
    ax.set_xticks(even_ticks)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
    if n > 1:
        ax.legend(fontsize=8, frameon=False)


def plot_all_dispersions(phi_items):
    """
    Plot all (phi_arr, label) pairs as grouped bars on a single axes.
    Returns list of (fig, filename_stem) tuples.
    """
    ylabel = r"Phase  $\phi_n$  [rad]  mod $2\pi$" if PHASE_MOD_2PI else r"Phase  $\phi_n$  [rad]"
    figs   = []

    fig, ax = make_phase_axes("Dispersion phase profiles")
    figs.append((fig, "dispersion"))
    _draw_phase_bars(ax, phi_items, ylabel)

    if MINIMIZE_PHASE:
        min_items = []
        for phi_arr, lbl in phi_items:
            phi_min  = find_min_phase(phi_arr)
            rms_orig = float(np.sqrt(np.mean(phi_arr ** 2)))
            rms_min  = float(np.sqrt(np.mean(phi_min ** 2)))
            tag = "(reduction)" if rms_min < rms_orig else "(already near-minimal)"
            print(f"{lbl}:  RMS phase original={rms_orig:.4f} [rad]  ->  "
                  f"min-phase={rms_min:.4f} [rad]  {tag}")
            min_items.append((phi_min, lbl))
        fig2, ax2 = make_phase_axes("Dispersion phase profiles  (min total phase)")
        figs.append((fig2, "dispersion_min_phase"))
        _draw_phase_bars(ax2, min_items,
                         r"Phase  $\phi_n$  [rad]  (min $\sum\phi^2$)")

    return figs


def plot_multitone_drives(betas1, thetas1, betas2, thetas2):
    """Bar chart of modulation depths and RF phases for each PM."""
    matplotlib.rcParams["font.family"] = "Arial"
    font_props = {"family": "Arial"}
    K     = len(betas1)
    tones = np.arange(1, K + 1)
    width = 0.35

    mm_per_inch = 25.4
    fig, axes = plt.subplots(1, 2, figsize=(2 * 150 / mm_per_inch, 60 / mm_per_inch))

    for ax, vals1, vals2, ylabel in zip(
        axes,
        [betas1, thetas1],
        [betas2, thetas2],
        [r"$\beta_k$  [rad]", r"$\theta_k$  [rad]"],
    ):
        ax.bar(tones - width / 2, vals1, width, label="PM1", color=BLUE2)
        ax.bar(tones + width / 2, vals2, width, label="PM2", color=GREEN2)
        ax.set_xlabel("Tone index  $k$", fontsize=10, **font_props)
        ax.set_ylabel(ylabel,            fontsize=10, **font_props)
        ax.legend(fontsize=8, frameon=False)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Arial")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Intermediate-spectrum helpers
# ---------------------------------------------------------------------------

def _intermediate_spectra_single(betas, phi_arrs, n_max, truncate):
    """
    Step through single-tone cascaded stages, yielding
    (harmonics, amplitudes, label) after each PM.
    """
    pm_range   = np.arange(-n_max, n_max + 1)
    harmonics  = np.array([0])
    amplitudes = np.array([1.0 + 0j])
    stages = []
    for i, beta in enumerate(betas):
        pm_amps    = jv(pm_range, beta).astype(complex)
        amplitudes = np.convolve(amplitudes, pm_amps)
        harmonics  = np.arange(harmonics[0] + pm_range[0],
                               harmonics[-1] + pm_range[-1] + 1)
        if truncate:
            mask       = (harmonics >= -n_max) & (harmonics <= n_max)
            harmonics  = harmonics[mask]
            amplitudes = amplitudes[mask]
        stages.append((harmonics.copy(), amplitudes.copy(), f"After PM {i + 1}"))
        if i < len(phi_arrs):
            amplitudes = amplitudes * np.exp(1j * np.asarray(phi_arrs[i]))
    return stages


def _intermediate_spectra_multitone(betas1, thetas1, phi_arr, betas2, thetas2,
                                    n_max, method, n_fft):
    """
    Return (harmonics, amplitudes, label) snapshots after PM1 and after PM2.
    """
    compute = (compute_multitone_spectrum_fft if method == "fft"
               else compute_multitone_spectrum_conv)

    h1, a1 = compute(betas1, thetas1, n_max, n_fft) if method == "fft" \
         else compute(betas1, thetas1, n_max)
    # clip conv output to ±n_max
    if method == "conv":
        mask = (h1 >= -n_max) & (h1 <= n_max)
        h1, a1 = h1[mask], a1[mask]

    a1_disp = a1 * np.exp(1j * np.asarray(phi_arr))

    h2, a2 = compute(betas2, thetas2, n_max, n_fft) if method == "fft" \
         else compute(betas2, thetas2, n_max)
    if method == "conv":
        mask = (h2 >= -n_max) & (h2 <= n_max)
        h2, a2 = h2[mask], a2[mask]

    a_final    = np.convolve(a1_disp, a2)
    h_final    = np.arange(-2 * n_max, 2 * n_max + 1)

    return [
        (h1,      a1,      "After PM 1"),
        (h_final, a_final, "After PM 2"),
    ]


# ---------------------------------------------------------------------------
# Branch: sweep vs. single-tone cascaded vs. multi-tone cascaded
# ---------------------------------------------------------------------------
if IS_SWEEP:
    with open(os.path.join(RESULT_FOLDER, "sweep_disp_n_max.json")) as f:
        sweep_data = json.load(f)

    disp_n_max_values = [pt["disp_n_max"]   for pt in sweep_data]
    wanted_powers     = [pt["wanted_power"] for pt in sweep_data]
    disp_n_max_max    = config.get("disp_n_max_max", max(disp_n_max_values))

    print(f"Format: disp_n_max sweep  ({len(sweep_data)} points, "
          f"n_max={n_max}, wanted_harmonic={config['wanted_harmonic']})")
    for pt in sweep_data:
        print(f"  disp_n_max={pt['disp_n_max']:3d}  wanted_power={pt['wanted_power']:.6f}"
              f"  ratio={pt['ratio']:.4f}")

    def _make_sweep_ax(ylabel: str):
        matplotlib.rcParams["font.family"] = "Arial"
        fp = {"family": "Arial"}
        mm_per_inch = 25.4
        left_in, right_in, bottom_in, top_in = 0.65, 0.20, 0.45, 0.15
        ax_w_in = 120.0 / mm_per_inch
        ax_h_in =  70.0 / mm_per_inch
        fig = plt.figure(figsize=(left_in + ax_w_in + right_in,
                                   bottom_in + ax_h_in + top_in))
        ax = fig.add_axes([
            left_in / (left_in + ax_w_in + right_in),
            bottom_in / (bottom_in + ax_h_in + top_in),
            ax_w_in  / (left_in + ax_w_in + right_in),
            ax_h_in  / (bottom_in + ax_h_in + top_in),
        ])
        ax.set_xlabel(r"$N_{\mathrm{disp}}$  (max dispersed harmonic)", fontsize=10, **fp)
        ax.set_ylabel(ylabel, fontsize=10, **fp)
        ax.set_xlim(0.5, disp_n_max_max + 0.5)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Arial")
        return fig, ax

    # Linear
    fig_lin, ax_lin = _make_sweep_ax("Conversion efficiency")
    ax_lin.plot(disp_n_max_values, wanted_powers,
                color="#4C72B0", linewidth=1.5, marker="o", markersize=4, zorder=3)
    ax_lin.set_ylim(0, 1.1)
    _save(fig_lin, "sweep_linear")

    # dB
    db_floor  = DB_FLOOR
    wanted_db = 10.0 * np.log10(np.maximum(wanted_powers, 10 ** (db_floor / 10)))
    fig_db, ax_db = _make_sweep_ax("Conversion efficiency [dB]")
    ax_db.plot(disp_n_max_values, wanted_db,
               color="#4C72B0", linewidth=1.5, marker="o", markersize=4, zorder=3)
    ax_db.set_ylim(db_floor, 2.0)
    _save(fig_db, "sweep_dB")

elif IS_MULTITONE:
    betas1  = result["betas1"]
    thetas1 = result["thetas1"]
    betas2  = result["betas2"]
    thetas2 = result["thetas2"]
    method  = config.get("method", "fft")
    n_fft   = config.get("n_fft", 8192)

    phi_raw = np.load(os.path.join(RESULT_FOLDER, "phi_params.npy"), allow_pickle=False)
    phi_arr = phi_raw if poly_order is None else np.polyval(phi_raw[::-1], stage_range)

    harmonics, amplitudes = compute_cascaded_multitone_spectrum(
        betas1, thetas1, phi_arr, betas2, thetas2,
        n_max=n_max, method=method, n_fft=n_fft,
    )

    print(f"Format: multi-tone cascaded  (method={method})")
    b1_str  = "  ".join(f"b1_{k+1}={b:.4f}"  for k, b  in enumerate(betas1))
    b2_str  = "  ".join(f"b2_{k+1}={b:.4f}"  for k, b  in enumerate(betas2))
    th1_str = "  ".join(f"th1_{k+1}={t:.4f}" for k, t  in enumerate(thetas1))
    th2_str = "  ".join(f"th2_{k+1}={t:.4f}" for k, t  in enumerate(thetas2))
    print(f"  PM1:  {b1_str}  {th1_str}")
    print(f"  PM2:  {b2_str}  {th2_str}")
    print(f"  ratio={result['ratio']:.6f}  wanted power={result['wanted_power']:.6f}  "
          f"wanted harmonic={config['wanted_harmonic']}")

    _save(plot_multitone_drives(betas1, thetas1, betas2, thetas2), "drives")
    for fig, name in plot_all_dispersions([(phi_arr, "Dispersion")]):
        _save(fig, name)

    if PLOT_INTERMEDIATE:
        for h, a, lbl in _intermediate_spectra_multitone(
                betas1, thetas1, phi_arr, betas2, thetas2, n_max, method, n_fft):
            fig_i, _ = plot_optical_spectrum(h, a, db=USE_DB, db_floor=DB_FLOOR,
                                             width_mm=80, top_in=0.30, n_display=N_DISPLAY)
            fig_i.axes[0].set_title(lbl, fontsize=10, fontfamily="Arial")
            _save(fig_i, f"spectrum_{lbl.lower().replace(' ', '_')}")

elif IS_RING_OPT:
    n_stages = config["n_stages"]
    n_rings  = config["n_rings"]
    Q_i      = config["Q_i"]
    Q_e      = config["Q_e"]
    f_carrier = config["f_carrier"]
    f_mod     = config["f_mod"]
    truncate  = n_stages > 2
    betas     = result["betas"]

    rings_per_stage = [
        [RingResonator.from_Q(f_0=f0, Q_i=Q_i, Q_e=Q_e) for f0 in stage]
        for stage in result["f0_per_stage"]
    ]

    # Derive harmonics from saved amplitude array length
    amplitudes = np.load(os.path.join(RESULT_FOLDER, "amplitudes.npy"))
    n_harm     = len(amplitudes)
    h_half     = (n_harm - 1) // 2
    harmonics  = np.arange(-h_half, h_half + 1)

    print(f"Format: ring optimizer  (n_stages={n_stages}, n_rings={n_rings})")
    betas_str = "  ".join(f"beta{i+1}={b:.4f}" for i, b in enumerate(betas))
    print(f"  {betas_str}")
    print(f"  Q_i={Q_i:.2e}  Q_e={Q_e:.2e}  "
          f"f_carrier={f_carrier:.3e} Hz  f_mod={f_mod:.3e} Hz")
    total_power = float(np.sum(np.abs(amplitudes) ** 2))
    total_power_db = 10.0 * np.log10(total_power) if total_power > 0 else float("-inf")
    print(f"  ratio={result['ratio']:.6f}  "
          f"wanted power={result['wanted_power']:.6f}  "
          f"wanted harmonic={config.get('wanted_harmonic', '?')}")
    print(f"  total comb power={total_power:.6f}  ({total_power_db:+.2f} dB)")

    for s, stage_rings in enumerate(rings_per_stage):
        freqs_ghz = [(r.f_0 - f_carrier) / 1e9 for r in stage_rings]
        print(f"  Stage {s+1} ring detunings [GHz]: "
              + "  ".join(f"{d:+.3f}" for d in freqs_ghz))

    fig_rings = plot_combined_ring_transmission(
        rings_per_stage=rings_per_stage,
        f_carrier=f_carrier,
        f_mod=f_mod,
        n_max=n_rings // 2,
        power_db=USE_DB,
        db_floor=-5,
        phase_xlim=(-5, 5)
    )
    _save(fig_rings, "ring_transmission")

else:
    n_stages = config.get("n_stages", 2)
    n_disp   = n_stages - 1
    truncate = n_stages > 2
    betas    = result["betas"]

    phi_params_raw = np.load(
        os.path.join(RESULT_FOLDER, "phi_params.npy"), allow_pickle=True
    )

    phi_items      = []
    phi_arrs_input = []
    for s in range(n_disp):
        raw = phi_params_raw[s]
        if poly_order is None:
            if disp_n_max is not None:
                phi_arr = np.zeros(len(stage_range))
                mask = (stage_range >= -disp_n_max) & (stage_range <= disp_n_max)
                phi_arr[mask] = raw
            else:
                phi_arr = raw
        else:
            phi_arr = np.polyval(raw[::-1], stage_range)
            if disp_n_max is not None:
                phi_arr[np.abs(stage_range) > disp_n_max] = 0.0
        phi_items.append((phi_arr, f"Stage {s + 1}"))
        phi_arrs_input.append(phi_arr)

    harmonics, amplitudes = compute_optical_spectrum_general(
        betas, phi_arrs_input, n_max=n_max, truncate=truncate
    )

    print(f"Format: single-tone cascaded  (n_stages={n_stages})")
    for i, b in enumerate(betas):
        print(f"  beta{i+1} = {b:.4f} [rad]")
    print(f"  ratio={result['ratio']:.6f}  wanted harmonic={config['wanted_harmonic']}")

    for fig, name in plot_all_dispersions(phi_items):
        _save(fig, name)

    if PLOT_INTERMEDIATE:
        for h, a, lbl in _intermediate_spectra_single(
                betas, phi_arrs_input, n_max, truncate):
            fig_i, _ = plot_optical_spectrum(h, a, db=USE_DB, db_floor=DB_FLOOR,
                                             width_mm=80, top_in=0.30, n_display=N_DISPLAY)
            fig_i.axes[0].set_title(lbl, fontsize=10, fontfamily="Arial")
            _save(fig_i, f"spectrum_{lbl.lower().replace(' ', '_')}")

# ---------------------------------------------------------------------------
# Spectrum (single-tone and multi-tone formats only)
# ---------------------------------------------------------------------------
if not IS_SWEEP:
    fig_spec, ax_spec = plot_optical_spectrum(harmonics, amplitudes, db=USE_DB, db_floor=DB_FLOOR,
                                              width_mm=55, height_mm=45, n_display=N_DISPLAY)
    if _target_powers:
        _overlay_targets(ax_spec, _target_powers)

    if PUBLISHED_PLOT:
        ax_spec.set_xlabel("")
        ax_spec.set_ylabel("")
        ax_spec.set_title("")
        ax_spec.tick_params(axis="both", which="both", direction="in", width=2,
                            labelbottom=False, labelleft=False)
        legend = ax_spec.get_legend()
        if legend is not None:
            legend.remove()
        for spine in ax_spec.spines.values():
            spine.set_linewidth(2)
            spine.set_zorder(10)

        if SAVE_DIR:
            svg_path = os.path.join(SAVE_DIR, "spectrum.svg")
            fig_spec.savefig(svg_path, dpi=SAVE_DPI)
            print(f"Saved {svg_path}")
    else:
        _save(fig_spec, "spectrum")

plt.show()
