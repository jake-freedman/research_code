"""ring_optimizer.py — direct ring-resonator optimizer.

Instead of finding phase masks and then mapping them to rings, this optimizer
directly searches over ring resonance frequencies (and modulation depths) to
maximize/minimise an objective on the optical spectrum.

Each dispersion stage is represented by N_RINGS ring resonators in series.
The combined complex transfer at sideband frequency f_n = f_carrier + n·f_mod
is the product of all ring field transmissions evaluated at f_n.  This is
physically realistic: every ring in a stage affects every sideband.

Parameter vector layout
-----------------------
  x[0 : n_stages]                           = betas (one per PM)
  x[n_stages : n_stages + n_disp*n_rings]   = f_0 values, row-major over
                                              (stage, ring) with
                                              n_disp = n_stages - 1

Objectives
----------
  "power"       : maximise power at wanted_harmonic
  "ratio"       : minimise unwanted/wanted power ratio
  "power_split" : split power equally among split_harmonics
  "arbitrary"   : match target power fractions in arbitrary_targets
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping, minimize

from ring_resonator import RingResonator, field_transmission_coefficient
from ssb_spectrum import compute_optical_spectrum_general
from graphics import (
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)

mm_per_inch = 25.4
_cmap = matplotlib.colormaps["rainbow_r"]


# ---------------------------------------------------------------------------
# Core forward model helpers
# ---------------------------------------------------------------------------

def _build_transfer(
    f0_stage: np.ndarray,
    Q_i: float,
    Q_e: float,
    harmonics: np.ndarray,
    f_carrier: float,
    f_mod: float,
) -> np.ndarray:
    """Complex transfer array for one dispersion stage.

    f0_stage : 1-D array of ring resonance frequencies [Hz], length n_rings
    Returns complex array of length len(harmonics).
    """
    f_sb     = f_carrier + harmonics.astype(float) * f_mod
    transfer = np.ones(len(harmonics), dtype=complex)
    for f0 in f0_stage:
        ring      = RingResonator.from_Q(f_0=float(f0), Q_i=Q_i, Q_e=Q_e)
        transfer *= field_transmission_coefficient(ring, f_sb)
    return transfer


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimize_ring_spectrum(
    n_stages: int,
    n_rings: int,
    beta_max: float,
    Q_i: float,
    Q_e: float,
    f_carrier: float,
    f_mod: float,
    f_ring_bound: float,
    n_max: int,
    objective: str,
    n_iter: int,
    seed: int | None = None,
    x0: np.ndarray | None = None,
    wanted_harmonic: int = 1,
    split_harmonics: list[int] | None = None,
    arbitrary_targets: dict[int, float] | None = None,
    local_only: bool = False,
) -> dict:
    """Optimise ring resonance frequencies and modulation depths.

    Parameters
    ----------
    n_stages         : number of phase modulators
    n_rings          : number of ring resonators per dispersion stage
    beta_max         : maximum modulation depth [rad]
    Q_i              : intrinsic Q factor (all rings, fixed)
    Q_e              : external Q factor (all rings, fixed)
    f_carrier        : optical carrier frequency [Hz]
    f_mod            : RF modulation frequency [Hz]
    f_ring_bound     : per-ring f_0 search half-range [Hz].  Ring j is assigned
                       to harmonic k = j - n_rings//2 with centre frequency
                       f_carrier + k*f_mod, and is bounded to
                       [centre - f_ring_bound, centre + f_ring_bound]
    n_max            : harmonic truncation order per PM stage
    objective        : "power" | "ratio" | "power_split" | "arbitrary"
    n_iter           : basin-hopping iterations (ignored when local_only=True)
    seed             : RNG seed (int) or None
    x0               : initial parameter vector; None = random
    wanted_harmonic  : target sideband index (used by "power" and "ratio")
    split_harmonics  : harmonic indices for "power_split"
    arbitrary_targets: {harmonic: weight} for "arbitrary"
    local_only       : if True, run a single L-BFGS-B minimization from x0
                       instead of basin-hopping (useful for warm-started runs)

    Returns
    -------
    dict with keys:
      betas, f0_per_stage, rings_per_stage, harmonics, amplitudes,
      ratio, wanted_power, transfer_per_stage, opt_result
    """
    if objective not in ("power", "ratio", "power_split", "arbitrary"):
        raise ValueError(f"Unknown objective: {objective!r}")
    if objective == "power_split" and not split_harmonics:
        raise ValueError("split_harmonics required for objective='power_split'")
    if objective == "arbitrary" and not arbitrary_targets:
        raise ValueError("arbitrary_targets required for objective='arbitrary'")

    n_disp    = n_stages - 1
    truncate  = n_stages > 2
    harmonics = np.arange(-n_max, n_max + 1, dtype=int)
    rng       = np.random.default_rng(seed)

    # Harmonic indices in the output power array
    def _harm_idx(h: int) -> int:
        idx = np.where(harmonics == h)[0]
        if len(idx) == 0:
            raise ValueError(f"Harmonic {h} is outside ±n_max={n_max}")
        return int(idx[0])

    wanted_idx = _harm_idx(wanted_harmonic)

    if objective == "power_split":
        split_idxs = [_harm_idx(h) for h in split_harmonics]
        n_split    = len(split_idxs)
        ideal      = np.full(n_split, 1.0 / n_split)

    if objective == "arbitrary":
        _arb_hs   = list(arbitrary_targets.keys())
        _raw_w    = np.array([arbitrary_targets[h] for h in _arb_hs], dtype=float)
        _arb_tgt  = _raw_w / _raw_w.sum()
        arb_idxs  = [_harm_idx(h) for h in _arb_hs]

    # ------------------------------------------------------------------
    # Per-ring centre frequencies and bounds
    # ------------------------------------------------------------------
    # Ring j is assigned to harmonic k = j - n_rings//2, so rings span
    # symmetrically: for n_rings=9 → k = -4,-3,...,4.
    # Each ring's f_0 is bounded to [centre ± f_ring_bound].
    ring_k       = np.arange(n_rings) - n_rings // 2          # harmonic index per ring
    centre_freqs = f_carrier + ring_k.astype(float) * f_mod   # shape (n_rings,)

    beta_lo = np.zeros(n_stages)
    beta_hi = np.full(n_stages, beta_max)
    # Same centre/bound pattern repeated for every dispersion stage
    f0_lo   = np.tile(centre_freqs - f_ring_bound, n_disp)
    f0_hi   = np.tile(centre_freqs + f_ring_bound, n_disp)

    lower = np.concatenate([beta_lo, f0_lo])
    upper = np.concatenate([beta_hi, f0_hi])
    bounds = list(zip(lower, upper))

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _obj(x: np.ndarray) -> float:
        betas_x  = x[:n_stages]
        f0_flat  = x[n_stages:]
        profiles = [
            _build_transfer(
                f0_flat[s * n_rings:(s + 1) * n_rings],
                Q_i, Q_e, harmonics, f_carrier, f_mod,
            )
            for s in range(n_disp)
        ]
        _, amps = compute_optical_spectrum_general(
            betas_x, profiles, n_max=n_max, truncate=truncate
        )
        power        = np.abs(amps) ** 2
        wanted_power = float(power[wanted_idx])

        if objective == "power":
            return -wanted_power
        if objective == "power_split":
            sp = power[split_idxs]
            return float(np.sum((sp - ideal) ** 2)) - float(sp.sum())
        if objective == "arbitrary":
            ap = power[arb_idxs]
            return float(np.sum((ap - _arb_tgt) ** 2)) - float(ap.sum())
        # ratio
        if wanted_power < 1e-30:
            return 1e30
        return (float(power.sum()) - wanted_power) / wanted_power

    # ------------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------------
    if x0 is None:
        f0_init = np.concatenate([
            rng.uniform(centre_freqs - f_ring_bound, centre_freqs + f_ring_bound)
            for _ in range(n_disp)
        ])
        x0 = np.concatenate([rng.uniform(0.0, beta_max, n_stages), f0_init])

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------
    if local_only:
        opt = minimize(_obj, x0, method="L-BFGS-B", bounds=bounds)
    else:
        class _BoundedStep:
            def __call__(self, x):
                step = np.concatenate([
                    rng.uniform(-beta_max * 0.3, beta_max * 0.3, n_stages),
                    rng.uniform(-f_ring_bound * 0.3, f_ring_bound * 0.3, n_disp * n_rings),
                ])
                return np.clip(x + step, lower, upper)

        opt = basinhopping(
            _obj,
            x0,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds},
            niter=n_iter,
            take_step=_BoundedStep(),
            seed=rng,
        )

    x_best   = np.clip(opt.x, lower, upper)
    betas_opt = list(x_best[:n_stages])
    f0_flat   = x_best[n_stages:]

    f0_per_stage = [
        f0_flat[s * n_rings:(s + 1) * n_rings].tolist()
        for s in range(n_disp)
    ]
    rings_per_stage = [
        [RingResonator.from_Q(f_0=f0, Q_i=Q_i, Q_e=Q_e) for f0 in stage]
        for stage in f0_per_stage
    ]
    transfer_per_stage = [
        _build_transfer(
            np.array(f0_per_stage[s]), Q_i, Q_e, harmonics, f_carrier, f_mod
        )
        for s in range(n_disp)
    ]

    harmonics_out, amplitudes = compute_optical_spectrum_general(
        betas_opt, transfer_per_stage, n_max=n_max, truncate=truncate
    )
    power        = np.abs(amplitudes) ** 2
    wanted_power = float(power[wanted_idx])
    total_power  = float(power.sum())
    ratio        = (total_power - wanted_power) / wanted_power if wanted_power > 1e-30 else np.inf

    return {
        "betas":              betas_opt,
        "f0_per_stage":       f0_per_stage,
        "rings_per_stage":    rings_per_stage,
        "harmonics":          harmonics_out,
        "amplitudes":         amplitudes,
        "ratio":              ratio,
        "wanted_power":       wanted_power,
        "transfer_per_stage": transfer_per_stage,
        "opt_result":         opt,
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_ring_opt_result(
    res: dict,
    config: dict,
    out_folder: str,
) -> None:
    os.makedirs(out_folder, exist_ok=True)

    with open(os.path.join(out_folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    result_data = {
        "betas":        [float(b) for b in res["betas"]],
        "f0_per_stage": res["f0_per_stage"],
        "ratio":        float(res["ratio"]),
        "wanted_power": float(res["wanted_power"]),
        "nit":          int(res["opt_result"].nit),
        "nfev":         int(res["opt_result"].nfev),
        "message":      str(res["opt_result"].message),
    }
    with open(os.path.join(out_folder, "result.json"), "w") as f:
        json.dump(result_data, f, indent=2)

    np.save(
        os.path.join(out_folder, "amplitudes.npy"),
        res["amplitudes"],
    )


def load_ring_opt_result(folder: str) -> tuple[dict, dict]:
    """Return (config, result) dicts loaded from *folder*."""
    with open(os.path.join(folder, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(folder, "result.json")) as f:
        result = json.load(f)
    result["amplitudes"] = np.load(os.path.join(folder, "amplitudes.npy"))
    return config, result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ring_opt_spectrum(
    harmonics: np.ndarray,
    amplitudes: np.ndarray,
    n_display: int = 10,
    ax_w_mm: float = 90.0,
    ax_h_mm: float = 70.0,
    left_in: float = 0.60,
    right_in: float = 0.25,
    bottom_in: float = 0.45,
    top_in: float = 0.30,
    use_db: bool = False,
    db_floor: float = -40.0,
    wanted_harmonic: int | None = None,
    arbitrary_targets: dict[int, float] | None = None,
) -> matplotlib.figure.Figure:
    """Stem plot of the ring-optimizer spectrum.

    Open black circles mark the target powers (for 'power' and 'arbitrary'
    objectives); the wanted harmonic marker is a solid black circle.
    """
    pwr        = np.abs(amplitudes) ** 2
    norm_color = matplotlib.colors.Normalize(vmin=-n_display, vmax=n_display)

    ax_w_in = ax_w_mm / mm_per_inch
    ax_h_in = ax_h_mm / mm_per_inch
    fig_w   = left_in + ax_w_in + right_in
    fig_h   = bottom_in + ax_h_in + top_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([
        left_in / fig_w, bottom_in / fig_h,
        ax_w_in / fig_w, ax_h_in / fig_h,
    ])

    y_floor = db_floor if use_db else 0.0

    def _y(p):
        return 10.0 * np.log10(max(p, 10 ** (db_floor / 10))) if use_db else p

    mask = np.abs(harmonics) <= n_display
    hs   = harmonics[mask]
    ps   = pwr[mask]
    colors = [_cmap(norm_color(int(h))) for h in hs]
    y_vals = [_y(p) for p in ps]

    ax.vlines(hs, y_floor, y_vals, linewidth=1.5, colors=colors, zorder=2)
    ax.scatter(hs, y_vals, s=18, c=colors, zorder=3)

    # Target markers
    if arbitrary_targets is not None:
        total_w = sum(arbitrary_targets.values())
        for h, w in arbitrary_targets.items():
            if abs(h) > n_display:
                continue
            ax.scatter(h, _y(w / total_w), s=55,
                       facecolors="none", edgecolors="black",
                       linewidths=1.0, zorder=5)
    elif wanted_harmonic is not None and abs(wanted_harmonic) <= n_display:
        ax.scatter(wanted_harmonic, _y(1.0), s=55,
                   facecolors="none", edgecolors="black",
                   linewidths=1.0, zorder=5)

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

    return fig
