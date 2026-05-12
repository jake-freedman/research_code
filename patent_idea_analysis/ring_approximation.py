"""ring_approximation.py — approximate optimizer phase masks with ring resonators."""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from ring_resonator import RingResonator, field_transmission_coefficient, _phase_tick_label
from ssb_spectrum import compute_optical_spectrum_general
from graphics import (
    BLUE2, DARKGREEN2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)

mm_per_inch = 25.4
_cmap = matplotlib.colormaps["rainbow_r"]


# ---------------------------------------------------------------------------
# Load optimizer result
# ---------------------------------------------------------------------------

def load_optimizer_result(folder: str) -> dict:
    """Load config, result, and phi_params from an optimizer output folder.

    Returns a dict with keys:
      config           : the config.json dict
      betas            : 1-D float array of modulation depths
      phi_params_full  : list of 1-D float arrays, one per dispersion stage,
                         each of length 2*n_max+1 covering harmonics -n_max..+n_max
      harmonics        : integer array -n_max..+n_max
      result           : the result.json dict
    """
    with open(os.path.join(folder, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(folder, "result.json")) as f:
        result = json.load(f)

    phi_arr = np.load(os.path.join(folder, "phi_params.npy"), allow_pickle=True)

    n_max      = int(config["n_max"])
    disp_n_max = config.get("disp_n_max", None)
    poly_order = config.get("poly_order", None)
    harmonics  = np.arange(-n_max, n_max + 1, dtype=int)
    n_full     = len(harmonics)

    phi_params_full = []
    for raw in phi_arr:
        if poly_order is None:
            full = np.zeros(n_full)
            if disp_n_max is not None:
                mask = (harmonics >= -disp_n_max) & (harmonics <= disp_n_max)
                full[mask] = raw
            else:
                full = raw.copy()
        else:
            full = np.polyval(raw[::-1], harmonics)
            if disp_n_max is not None:
                full[np.abs(harmonics) > disp_n_max] = 0.0
        phi_params_full.append(full)

    return {
        "config":          config,
        "betas":           np.array(result["betas"]),
        "phi_params_full": phi_params_full,
        "harmonics":       harmonics,
        "result":          result,
    }


# ---------------------------------------------------------------------------
# Solve for ring resonance frequency
# ---------------------------------------------------------------------------

def solve_ring_for_sideband(
    phi_target: float,
    f_sideband: float,
    Q_i: float,
    Q_e: float,
    f_search_radius: float = 500e9,
    tol_rad: float = 1e-6,
) -> RingResonator:
    """Find f_0 such that arg(t(f_sideband)) = phi_target for an overcoupled ring.

    For an overcoupled ring (gamma_e > gamma_i, i.e. Q_e < Q_i):
      - phi_target > 0 : ring resonance below f_sideband
      - phi_target < 0 : ring resonance above f_sideband
      - phi_target = 0 : ring placed far off-resonance (no phase shift)
      - phi_target = ±π: ring placed exactly at f_sideband

    Parameters
    ----------
    phi_target      : target phase [rad], will be wrapped to (-π, π]
    f_sideband      : optical frequency of the sideband [Hz]
    Q_i             : intrinsic Q factor of all rings
    Q_e             : external Q factor of all rings
    f_search_radius : half-range for root bracketing [Hz]
    tol_rad         : phase residual tolerance [rad] — used only for special cases
    """
    phi_target = float((phi_target) % (2 * np.pi) - np.pi)

    # phase ≈ 0: place far off-resonance
    if abs(phi_target) < tol_rad:
        return RingResonator.from_Q(f_0=f_sideband + f_search_radius * 0.9, Q_i=Q_i, Q_e=Q_e)

    # phase ≈ ±π: ring at resonance
    if abs(abs(phi_target) - np.pi) < tol_rad:
        return RingResonator.from_Q(f_0=f_sideband, Q_i=Q_i, Q_e=Q_e)

    eps = 1e6  # 1 MHz — well inside the ring linewidth, keeps the bracket off the singularity

    def residual(f0: float) -> float:
        ring = RingResonator.from_Q(f_0=f0, Q_i=Q_i, Q_e=Q_e)
        return float(np.angle(field_transmission_coefficient(ring, f_sideband))) - phi_target

    if phi_target > 0:
        # phase(f0) decreases from π (at f0=f_sb-eps) to 0 (at f0→-∞)
        fa, fb = f_sideband - f_search_radius, f_sideband - eps
    else:
        # phase(f0) increases from -π (at f0=f_sb+eps) to 0 (at f0→+∞)
        fa, fb = f_sideband + eps, f_sideband + f_search_radius

    try:
        f0_sol = brentq(residual, fa, fb, xtol=1.0, rtol=1e-12)
    except ValueError:
        # Bracket does not straddle zero — pick end closest to target
        pa, pb = residual(fa), residual(fb)
        f0_sol = fa if abs(pa) < abs(pb) else fb

    return RingResonator.from_Q(f_0=f0_sol, Q_i=Q_i, Q_e=Q_e)


# ---------------------------------------------------------------------------
# Compute per-stage ring transfer arrays
# ---------------------------------------------------------------------------

def compute_ring_transfer(
    rings: list[RingResonator],
    harmonics: np.ndarray,
    f_carrier: float,
    f_mod: float,
) -> np.ndarray:
    """Complex transfer array for a bank of rings applied to all sidebands.

    Every ring's transfer function is evaluated at every sideband frequency and
    the results multiplied together, so each ring affects every harmonic.
    Returns complex array of shape (len(harmonics),).
    """
    f_sidebands = f_carrier + harmonics.astype(float) * f_mod
    transfer = np.ones(len(harmonics), dtype=complex)
    for ring in rings:
        transfer *= field_transmission_coefficient(ring, f_sidebands)
    return transfer


# ---------------------------------------------------------------------------
# Full ring-approximation pipeline
# ---------------------------------------------------------------------------

def compute_ring_spectrum(
    opt_result: dict,
    Q_i: float,
    Q_e: float,
    f_carrier: float = 193e12,
    f_mod: float = 50e9,
    f_search_radius: float = 500e9,
) -> dict:
    """Approximate each dispersion stage with rings and recompute the optical spectrum.

    Parameters
    ----------
    opt_result      : dict returned by load_optimizer_result
    Q_i             : intrinsic Q factor for every ring
    Q_e             : external Q factor for every ring
    f_carrier       : optical carrier frequency [Hz]
    f_mod           : modulation frequency [Hz]
    f_search_radius : half-range [Hz] passed to solve_ring_for_sideband; limits
                      how far from the target sideband a ring resonance can sit

    Returns
    -------
    dict with keys:
      harmonics          : harmonic index array
      amplitudes         : complex amplitudes of the ring-approximated spectrum
      rings_per_stage    : list (one per dispersion stage) of lists of RingResonator
      transfer_per_stage : list of complex arrays
    """
    harmonics        = opt_result["harmonics"]
    phi_params_full  = opt_result["phi_params_full"]
    betas            = opt_result["betas"]
    config           = opt_result["config"]
    n_max            = int(config["n_max"])
    n_stages         = int(config["n_stages"])
    truncate         = n_stages > 2
    disp_n_max       = config.get("disp_n_max", None)

    rings_per_stage    = []
    transfer_per_stage = []

    for phi_full in phi_params_full:
        # Create rings only for harmonics within disp_n_max (the optimized window).
        # Outer harmonics may carry a non-zero phase (e.g. a global rotation) but
        # no ring is placed there — their transfer is determined by the off-resonance
        # tails of the inner rings.
        rings_this = [
            solve_ring_for_sideband(
                phi_target=float(phi_full[j]),
                f_sideband=float(f_carrier + harmonics[j] * f_mod),
                Q_i=Q_i,
                Q_e=Q_e,
                f_search_radius=f_search_radius,
            )
            for j in range(len(harmonics))
            if abs(phi_full[j]) > 1e-9
            and (disp_n_max is None or abs(int(harmonics[j])) <= disp_n_max)
        ]
        rings_per_stage.append(rings_this)
        transfer = compute_ring_transfer(rings_this, harmonics, f_carrier, f_mod)
        transfer_per_stage.append(transfer)

    harmonics_out, amplitudes = compute_optical_spectrum_general(
        betas, transfer_per_stage, n_max=n_max, truncate=truncate
    )

    return {
        "harmonics":          harmonics_out,
        "amplitudes":         amplitudes,
        "rings_per_stage":    rings_per_stage,
        "transfer_per_stage": transfer_per_stage,
    }


# ---------------------------------------------------------------------------
# Save / load ring result
# ---------------------------------------------------------------------------

def save_ring_result(
    ring_result: dict,
    opt_result: dict,
    folder: str,
    Q_i: float,
    Q_e: float,
    f_carrier: float,
    f_mod: float,
) -> None:
    """Save ring approximation result to *folder* as JSON + npy files."""
    os.makedirs(folder, exist_ok=True)

    meta = {
        "Q_i":       Q_i,
        "Q_e":       Q_e,
        "f_carrier": f_carrier,
        "f_mod":     f_mod,
        "harmonics": ring_result["harmonics"].tolist(),
    }
    with open(os.path.join(folder, "ring_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save per-stage ring f_0 values (Q_i/Q_e are global)
    rings_f0 = [
        [r.f_0 for r in stage]
        for stage in ring_result["rings_per_stage"]
    ]
    with open(os.path.join(folder, "ring_f0.json"), "w") as f:
        json.dump(rings_f0, f, indent=2)

    np.save(
        os.path.join(folder, "ring_amplitudes.npy"),
        ring_result["amplitudes"],
    )


def load_ring_result(folder: str) -> dict:
    """Load a previously saved ring approximation result."""
    with open(os.path.join(folder, "ring_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(folder, "ring_f0.json")) as f:
        rings_f0 = json.load(f)

    Q_i  = meta["Q_i"]
    Q_e  = meta["Q_e"]
    rings_per_stage = [
        [RingResonator.from_Q(f_0=f0, Q_i=Q_i, Q_e=Q_e) for f0 in stage]
        for stage in rings_f0
    ]
    amplitudes = np.load(os.path.join(folder, "ring_amplitudes.npy"))
    harmonics  = np.array(meta["harmonics"], dtype=int)

    return {
        "harmonics":       harmonics,
        "amplitudes":      amplitudes,
        "rings_per_stage": rings_per_stage,
        "meta":            meta,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ring_vs_optimizer(
    opt_result: dict,
    ring_result: dict,
    n_display: int = 10,
    ax_w_mm: float = 90.0,
    ax_h_mm: float = 70.0,
    left_in: float = 0.60,
    right_in: float = 0.25,
    bottom_in: float = 0.45,
    top_in: float = 0.30,
    use_db: bool = False,
    db_floor: float = -40.0,
) -> matplotlib.figure.Figure:
    """Side-by-side stem plot: ring-approximated spectrum vs optimizer spectrum.

    Colored stems show ring-approximated powers; open black circles show the
    original optimizer powers at the same harmonics.
    """
    opt_h   = opt_result["harmonics"]
    opt_amp = opt_result.get("amplitudes_opt", None)
    # If opt_result came from load_optimizer_result, recompute the optimizer spectrum
    if opt_amp is None:
        config   = opt_result["config"]
        n_max    = int(config["n_max"])
        n_stages = int(config["n_stages"])
        truncate = n_stages > 2
        opt_h, opt_amp = compute_optical_spectrum_general(
            opt_result["betas"], opt_result["phi_params_full"], n_max=n_max, truncate=truncate
        )

    ring_h   = ring_result["harmonics"]
    ring_amp = ring_result["amplitudes"]

    opt_pwr  = np.abs(opt_amp)  ** 2
    ring_pwr = np.abs(ring_amp) ** 2

    # Build display arrays limited to |n| <= n_display
    norm_color = matplotlib.colors.Normalize(vmin=-n_display, vmax=n_display)

    ax_w_in = ax_w_mm / mm_per_inch
    ax_h_in = ax_h_mm / mm_per_inch
    fig_w   = left_in + ax_w_in + right_in
    fig_h   = bottom_in + ax_h_in + top_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([
        left_in / fig_w,
        bottom_in / fig_h,
        ax_w_in / fig_w,
        ax_h_in / fig_h,
    ])

    def _to_db(p: float) -> float:
        return 10.0 * np.log10(max(p, 10 ** (db_floor / 10)))

    y_floor = db_floor if use_db else 0.0

    # Ring-approximated stems (colored)
    ring_mask = np.abs(ring_h) <= n_display
    rh = ring_h[ring_mask]
    rp = ring_pwr[ring_mask]
    colors = [_cmap(norm_color(int(h))) for h in rh]
    y_ring = [_to_db(p) if use_db else p for p in rp]
    ax.vlines(rh, y_floor, y_ring, linewidth=1.5, colors=colors, zorder=2)
    ax.scatter(rh, y_ring, s=18, c=colors, zorder=3)

    # Optimizer powers (open black circles)
    opt_mask = np.abs(opt_h) <= n_display
    oh = opt_h[opt_mask]
    op = opt_pwr[opt_mask]
    y_opt = [_to_db(p) if use_db else p for p in op]
    ax.scatter(oh, y_opt, s=55, facecolors="none", edgecolors="black",
               linewidths=1.0, zorder=5, label="optimizer")

    fp = {"family": "Arial"}
    ax.set_xlabel("Harmonic $k$", fontsize=axis_label_fontsize, **fp)
    if use_db:
        ax.set_ylabel("Power [dB]", fontsize=axis_label_fontsize, **fp)
        ax.set_ylim(db_floor - 2, 2)
    else:
        ax.set_ylabel("Power $|a_k|^2$", fontsize=axis_label_fontsize, **fp)
        ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-n_display - 0.5, n_display + 0.5)

    t_start    = -n_display + (n_display % 2)
    even_ticks = np.arange(t_start, n_display + 1, 2)
    ax.set_xticks(even_ticks)

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
        spine.set_zorder(100)
    ax.tick_params(
        axis="both", which="both",
        direction=tick_direction, width=tick_width,
        labelsize=tick_label_fontsize,
    )
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
    ax.grid(linewidth=0.4, zorder=0)
    ax.legend(fontsize=tick_label_fontsize, frameon=False)

    return fig


# ---------------------------------------------------------------------------
# Combined ring transmission figure
# ---------------------------------------------------------------------------

def plot_combined_ring_transmission(
    rings_per_stage: list[list[RingResonator]],
    f_carrier: float,
    f_mod: float,
    n_max: int,
    n_points: int = 2000,
    ax_w_mm: float = 70.0,
    ax_h_mm: float = 50.0,
    left_in: float = 0.60,
    mid_in: float = 0.15,
    right_in: float = 0.20,
    bottom_in: float = 0.45,
    row_gap_in: float = 0.35,
    top_in: float = 0.25,
    phase_min: float | None = -2*np.pi,
    power_db: bool = False,
    db_floor: float = -10.0,
    phase_xlim: tuple = (),
    phase_gamma_e_span: float = 10.0,
) -> matplotlib.figure.Figure:
    """Plot the combined ring transmission for each dispersion stage.

    Layout: 2 rows × n_stages columns.
      Row 0: power transmission |T(f)|²
      Row 1: phase transmission arg(T(f))

    Each column corresponds to one dispersion stage's combined ring bank.
    The x-axis is combline index (detuning / f_mod).  Vertical grey dashed lines
    mark the sideband positions at integer combline indices k = -n_max … n_max.
    """
    n_stages = len(rings_per_stage)
    n_cols   = n_stages

    # Sweep must cover all sideband positions AND each ring's ±phase_gamma_e_span*gamma_e
    # window so that the full phase transition of every ring is visible.
    all_rings_flat = [r for stage in rings_per_stage for r in stage]
    window_hz      = {id(r): phase_gamma_e_span * r.gamma_e / (2 * np.pi) for r in all_rings_flat}
    f_lo      = min(r.f_0 - window_hz[id(r)] for r in all_rings_flat)
    f_hi      = max(r.f_0 + window_hz[id(r)] for r in all_rings_flat)
    half_span = max(f_carrier - f_lo, f_hi - f_carrier, (n_max + 1) * f_mod)
    f_arr      = np.linspace(f_carrier - half_span, f_carrier + half_span, n_points)
    comb_index = (f_arr - f_carrier) / f_mod

    def _combined_transfer(stage_rings: list[RingResonator]) -> np.ndarray:
        t = np.ones(len(f_arr), dtype=complex)
        for ring in stage_rings:
            t *= field_transmission_coefficient(ring, f_arr)
        return t

    transfers_plot = [_combined_transfer(stage) for stage in rings_per_stage]
    col_labels     = [f"Stage {i+1}" for i in range(n_stages)]

    ax_w_in = ax_w_mm / mm_per_inch
    ax_h_in = ax_h_mm / mm_per_inch
    fig_w   = left_in + n_cols * ax_w_in + (n_cols - 1) * mid_in + right_in
    fig_h   = bottom_in + 2 * ax_h_in + row_gap_in + top_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    fp  = {"family": "Arial"}

    # Sideband marker positions in combline-index units (integers)
    sb_detunings = np.arange(-n_max, n_max + 1, dtype=float)

    def _x0(col):
        return (left_in + col * (ax_w_in + mid_in)) / fig_w

    def _y0(row):  # row 0 = top (power), row 1 = bottom (phase)
        if row == 0:
            return (bottom_in + ax_h_in + row_gap_in) / fig_h
        return bottom_in / fig_h

    for ci, (t_arr, label) in enumerate(zip(transfers_plot, col_labels)):
        pwr   = np.abs(t_arr) ** 2
        phase = np.angle(t_arr)

        # ---- Power panel (top row) ----
        ax_pwr = fig.add_axes([_x0(ci), _y0(0), ax_w_in / fig_w, ax_h_in / fig_h])
        if power_db:
            pwr_plot  = 10 * np.log10(np.maximum(pwr, 10 ** (db_floor / 10)))
            pwr_ylim  = (db_floor - 0.5, 0.5)
            pwr_label = "Power [dB]"
        else:
            pwr_plot  = pwr
            pwr_ylim  = (-0.05, 1.05)
            pwr_label = "$|T|^2$"

        ax_pwr.plot(comb_index, pwr_plot, color=BLUE2, linewidth=1.0)
        for sd in sb_detunings:
            ax_pwr.axvline(sd, color="gray", linewidth=0.5, linestyle="--", zorder=0)
        ax_pwr.set_xlim(comb_index[0], comb_index[-1])
        ax_pwr.set_ylim(*pwr_ylim)
        ax_pwr.set_title(label, fontsize=axis_label_fontsize, fontfamily="Arial")
        if ci == 0:
            ax_pwr.set_ylabel(pwr_label, fontsize=axis_label_fontsize, **fp)

        # ---- Phase panel (bottom row): one line per ring ----
        ax_phi = fig.add_axes([_x0(ci), _y0(1), ax_w_in / fig_w, ax_h_in / fig_h])

        stage_rings = rings_per_stage[ci]
        n_rings_stage = len(stage_rings)
        cmap_phi = matplotlib.colormaps["YlGn"]
        # Sample colors from the mid-to-dark range of the colormap
        ring_colors = [cmap_phi(0.4 + 0.5 * j / max(n_rings_stage - 1, 1))
                       for j in range(n_rings_stage)]

        all_phases = []
        for ring, rcolor in zip(stage_rings, ring_colors):
            t_ring   = field_transmission_coefficient(ring, f_arr)
            phi_ring = np.angle(t_ring)
            if phase_min is not None:
                phi_ring = ((phi_ring - phase_min) % (2 * np.pi)) + phase_min
            all_phases.append(phi_ring)
            # Restrict each ring's curve to ±phase_gamma_e_span * gamma_e around its resonance
            mask = np.abs(f_arr - ring.f_0) <= window_hz[id(ring)]
            ax_phi.plot(comb_index[mask], phi_ring[mask], color=rcolor, linewidth=1.0)

        if phase_min is not None:
            phase_max = phase_min + 2 * np.pi
            tick_vals = np.arange(phase_min, phase_max + 1e-9, np.pi / 2)
            tick_lbls = [_phase_tick_label(v) for v in tick_vals]
            pad = 0.05
        else:
            all_concat = np.concatenate(all_phases)
            phase_min_ax, phase_max = all_concat.min(), all_concat.max()
            tick_vals = tick_lbls = None
            pad = 0.1

        for sd in sb_detunings:
            ax_phi.axvline(sd, color="gray", linewidth=0.5, linestyle="--", zorder=0)
        if phase_xlim:
            ax_phi.set_xlim(phase_xlim[0], phase_xlim[1])
        else:
            ax_phi.set_xlim(comb_index[0], comb_index[-1])
        if phase_min is not None:
            ax_phi.set_ylim(phase_min - pad, phase_max + pad)
            ax_phi.set_yticks(tick_vals)
            ax_phi.set_yticklabels(tick_lbls, fontsize=tick_label_fontsize)
        else:
            ax_phi.set_ylim(phase_min_ax - pad, phase_max + pad)  # type: ignore[possibly-undefined]
        ax_phi.set_xlabel("Combline index", fontsize=axis_label_fontsize, **fp)
        if ci == 0:
            ax_phi.set_ylabel("Phase [rad]", fontsize=axis_label_fontsize, **fp)

        for ax in (ax_pwr, ax_phi):
            for spine in ax.spines.values():
                spine.set_linewidth(spine_linewidth)
                spine.set_zorder(100)
            ax.tick_params(
                axis="both", which="both",
                direction=tick_direction, width=tick_width,
                labelsize=tick_label_fontsize,
            )
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontfamily("Arial")
            ax.grid(linewidth=0.4, zorder=0)

    return fig
