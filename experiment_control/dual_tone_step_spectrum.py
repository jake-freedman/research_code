"""
Plot the measured sideband spectrum at one step of a dual-tone sweep,
with an optional theoretical Jacobi-Anger overlay.

Select a step by index (STEP) or let the script find the step where a chosen
harmonic order is maximised (AUTO_FIND_ORDER).
"""

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dual_tone_sweep_data import DualToneSweepData
from path_utils import local_path
from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2, DARKBLUE2,
    TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
    axes_width_mm as _default_axes_w,
    axes_height_mm as _default_axes_h,
    left_mm as _left_mm, right_mm as _right_mm,
    bottom_mm as _bottom_mm, top_mm as _top_mm,
)

# ── data ──────────────────────────────────────────────────────────────────────
DATA_FILE = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\w2_d21_wg5a_p5\phase_sweep_dual_tone_sweep_2026-06-17-12-50-07.npz"

# Step index to plot (0-based).
# Set to None to auto-select via AUTO_FIND_ORDER.
STEP = None

# When STEP is None: find the step where this harmonic order's power is maximum.
# Set to None alongside STEP=None to default to step 0.
AUTO_FIND_ORDER = -1

# Harmonic orders to show in the plot and console table.
# Orders not present in the file are silently skipped.
ORDERS = [-3, -2, -1, 0, 1, 2, 3]

# Power display mode:
#   'percent' → |A_p|² as % of total optical power (RF-off calibration = 100 %)
#   'dB'      → dBc relative to total optical power
DISPLAY_MODE = 'dB'

# dB mode only: y-axis floor in dBc. None = auto.
FLOOR_DBc = -20.0

# ── theoretical overlay ───────────────────────────────────────────────────────
SHOW_THEORETICAL = True

# Modulation depths.
#   'auto' → extract from the single-tone preamble saved in the file.
#   float  → use this value in radians.
BETA1 = 'auto'
BETA2 = 'auto'

# Phases for the theoretical prediction.
#   'auto' → read ch1/ch2 phase from the selected step in the file.
#   float  → use this value in degrees.
PHI1_DEG = 'auto'
PHI2_DEG = 'auto'

# Set True when the LO is above the optical carrier. In that arrangement the
# heterodyne beat for optical sideband p appears at the ESA bin labelled -p,
# so the theoretical amplitudes must be evaluated at -p to align with the data.
FLIP_HARMONIC_SIGN = True

# PHI1_DEG / PHI2_DEG may also be set to 'max' to find the phase(s) that
# maximise the theoretical power at a given harmonic order.
# THEORY_MAX_ORDER selects that target; None → falls back to AUTO_FIND_ORDER.
THEORY_MAX_ORDER = None
# Grid resolution for the phase optimisation search.
N_PHASE_OPT = 500

# ── graphics ──────────────────────────────────────────────────────────────────
axes_width_mm  = 100
axes_height_mm = 40
left_mm   = _left_mm
right_mm  = _right_mm
bottom_mm = _bottom_mm
top_mm    = _top_mm
stem_linewidth    = 3
markersize        = 9
theory_markersize = 13.0   # open circles for theory — larger than measured balls

_COLORS = [BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2,
           DARKBLUE2, TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2]
# ─────────────────────────────────────────────────────────────────────────────


def _amplitudes(beta1, beta2, phi1_rad, phi2_rad, orders, k_trunc):
    k = np.arange(-k_trunc, k_trunc + 1)
    return {
        p: complex(np.sum(
            jv(p - 2*k, beta1) * jv(k, beta2)
            * np.exp(1j * ((p - 2*k) * phi1_rad + k * phi2_rad))
        ))
        for p in orders
    }


def main():
    data = DualToneSweepData.from_file(local_path(DATA_FILE))
    M = len(data.drive_freqs)

    # ── step selection ────────────────────────────────────────────────────────
    step = STEP
    if step is None:
        if AUTO_FIND_ORDER is not None:
            matches = np.where(data.harmonics == AUTO_FIND_ORDER)[0]
            if len(matches) == 0:
                raise ValueError(
                    f"AUTO_FIND_ORDER={AUTO_FIND_ORDER} not in file harmonics "
                    f"{list(data.harmonics)}."
                )
            hidx = int(matches[0])
            step = int(np.argmax(data.normalized_peak_powers_dbm()[:, hidx]))
            peak_val = float(data.normalized_peak_powers_dbm()[step, hidx])
            print(
                f"Auto-selected step {step} "
                f"(harmonic {AUTO_FIND_ORDER} maximised: "
                f"{peak_val:.2f} dBc, "
                f"ch2 phase={data.ch2_phases_deg[step]:.1f}°)"
            )
        else:
            step = 0

    if not (0 <= step < M):
        raise ValueError(f"STEP={step} out of range (file has {M} steps).")

    print(
        f"Loaded: {DATA_FILE}\n"
        f"  Step {step}/{M - 1}:  "
        f"f={data.drive_freqs[step]/1e9:.4f} GHz,  "
        f"ch1={data.ch1_powers_dbm[step]:+.1f} dBm / {data.ch1_phases_deg[step]:.1f}°,  "
        f"ch2={data.ch2_powers_dbm[step]:+.1f} dBm / {data.ch2_phases_deg[step]:.1f}°"
    )

    # ── measured powers at this step ──────────────────────────────────────────
    cal_peak_dbm = float(data.cal_spectra[step].max())
    meas_vals = {}
    for p in ORDERS:
        idx_arr = np.where(data.harmonics == p)[0]
        if len(idx_arr) == 0:
            print(f"  Warning: order {p} not recorded; skipping.")
            continue
        peak_dbm = float(data.spectra[step, int(idx_arr[0])].max())
        if DISPLAY_MODE == 'percent':
            meas_vals[p] = 10.0 ** ((peak_dbm - cal_peak_dbm) / 10.0) * 100.0
        else:
            meas_vals[p] = peak_dbm - cal_peak_dbm

    valid_orders = sorted(meas_vals)

    # ── theoretical prediction ────────────────────────────────────────────────
    amps        = {}
    theory_vals = {}
    beta1_val = beta2_val = None
    phi1_val  = phi2_val  = None

    if SHOW_THEORETICAL:
        if BETA1 == 'auto' or BETA2 == 'auto':
            try:
                b1_auto, b2_auto = data.single_tone_modulation_depths()
            except RuntimeError:
                b1_auto, b2_auto = None, None
            beta1_val = b1_auto if BETA1 == 'auto' else float(BETA1)
            beta2_val = b2_auto if BETA2 == 'auto' else float(BETA2)
        else:
            beta1_val, beta2_val = float(BETA1), float(BETA2)

        if beta1_val is None or beta2_val is None:
            print("Warning: β1/β2 not available (no preamble); theory overlay skipped.")
        else:
            k_trunc = int(2 * max(beta1_val, beta2_val)) + 20
            k = np.arange(-k_trunc, k_trunc + 1)

            # Resolve phases: 'auto' = from step, float = fixed, 'max' = optimise
            _phi1_raw = (float(data.ch1_phases_deg[step]) if PHI1_DEG == 'auto'
                         else (None if PHI1_DEG == 'max' else float(PHI1_DEG)))
            _phi2_raw = (float(data.ch2_phases_deg[step]) if PHI2_DEG == 'auto'
                         else (None if PHI2_DEG == 'max' else float(PHI2_DEG)))

            if PHI1_DEG == 'max' or PHI2_DEG == 'max':
                _max_ord = THEORY_MAX_ORDER if THEORY_MAX_ORDER is not None else AUTO_FIND_ORDER
                if _max_ord is None:
                    raise ValueError(
                        "Set THEORY_MAX_ORDER (or AUTO_FIND_ORDER) when PHI='max'."
                    )
                _ep = -_max_ord if FLIP_HARMONIC_SIGN else _max_ord
                _phi_grid = np.linspace(0, 2 * np.pi, N_PHASE_OPT, endpoint=False)
                _c_base = jv(_ep - 2*k, beta1_val) * jv(k, beta2_val)

                if PHI1_DEG == 'max' and PHI2_DEG == 'max':
                    # Iterate over φ1, vectorise over φ2 — O(N²) but O(N) memory
                    _best_pwr, _best_p1, _best_p2 = -1.0, 0.0, 0.0
                    for _p1 in _phi_grid:
                        _c = _c_base * np.exp(1j * (_ep - 2*k) * _p1)
                        _A = np.exp(1j * np.outer(_phi_grid, k)) @ _c
                        _idx = int(np.argmax(np.abs(_A)**2))
                        if np.abs(_A[_idx])**2 > _best_pwr:
                            _best_pwr = float(np.abs(_A[_idx])**2)
                            _best_p1, _best_p2 = _p1, _phi_grid[_idx]
                    phi1_val = float(np.degrees(_best_p1))
                    phi2_val = float(np.degrees(_best_p2))

                elif PHI1_DEG == 'max':
                    _c = _c_base * np.exp(1j * k * np.deg2rad(_phi2_raw))
                    _A = np.exp(1j * np.outer(_phi_grid, _ep - 2*k)) @ _c
                    phi1_val = float(np.degrees(_phi_grid[np.argmax(np.abs(_A)**2)]))
                    phi2_val = _phi2_raw

                else:  # PHI2_DEG == 'max'
                    _c = _c_base * np.exp(1j * (_ep - 2*k) * np.deg2rad(_phi1_raw))
                    _A = np.exp(1j * np.outer(_phi_grid, k)) @ _c
                    phi2_val = float(np.degrees(_phi_grid[np.argmax(np.abs(_A)**2)]))
                    phi1_val = _phi1_raw

                print(f"Theory phases optimised for harmonic {_max_ord}: "
                      f"φ1={phi1_val:.1f}°,  φ2={phi2_val:.1f}°")
            else:
                phi1_val, phi2_val = _phi1_raw, _phi2_raw

            eval_orders = [-p for p in valid_orders] if FLIP_HARMONIC_SIGN else valid_orders
            raw_amps = _amplitudes(
                beta1_val, beta2_val,
                np.deg2rad(phi1_val), np.deg2rad(phi2_val),
                eval_orders, k_trunc,
            )
            # remap so amps[p] = amplitude to display at position p
            amps = {p: raw_amps[q] for p, q in zip(valid_orders, eval_orders)}
            for p, A in amps.items():
                pwr = abs(A) ** 2
                theory_vals[p] = (pwr * 100.0 if DISPLAY_MODE == 'percent'
                                  else 10.0 * np.log10(max(pwr, 1e-30)))

    # ── console table ─────────────────────────────────────────────────────────
    unit = '%' if DISPLAY_MODE == 'percent' else 'dBc'
    if beta1_val is not None:
        print(f"\nβ1={beta1_val:.4f},  β2={beta2_val:.4f},  "
              f"φ1={phi1_val:.1f}°,  φ2={phi2_val:.1f}°")
    print(f"  {'order':>5}   {'measured':>12}   {'theory':>12}   {'theory phase':>13}")
    print(f"  {'─'*5}   {'─'*12}   {'─'*12}   {'─'*13}")
    for p in valid_orders:
        mstr = f"{meas_vals[p]:>10.4f} {unit}"
        if p in theory_vals:
            tstr  = f"{theory_vals[p]:>10.4f} {unit}"
            phstr = f"{np.degrees(np.angle(amps[p])):>+10.2f}°"
        else:
            tstr = phstr = f"{'—':>12}"
        print(f"  p={p:+d}   {mstr}   {tstr}   {phstr}")

    total_meas   = sum(10.0 ** (v / 10.0) / 100.0
                       for v in meas_vals.values()) if DISPLAY_MODE == 'dB' \
                   else sum(meas_vals.values()) / 100.0
    total_theory = sum(abs(A)**2 for A in amps.values()) if amps else None
    print(f"\n  Total measured in plotted orders : {total_meas * 100:.2f}%")
    if total_theory is not None:
        print(f"  Total theory  in plotted orders : {total_theory * 100:.2f}%")

    # ── figure ────────────────────────────────────────────────────────────────
    x_arr      = np.array(valid_orders)
    y_measured = np.array([meas_vals[p] for p in valid_orders])

    if DISPLAY_MODE == 'percent':
        y_baseline = 0.0
        ylabel = r'$|A_p|^2$ [% of total power]'
    else:
        y_baseline = FLOOR_DBc if FLOOR_DBc is not None else y_measured.min() - 3.0
        ylabel = r'$|A_p|^2$ [dBc]'

    fig_w = left_mm + axes_width_mm + right_mm
    fig_h = bottom_mm + axes_height_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm    / fig_h,
    )

    # Measured: colored stems + filled balls
    for idx, (xi, yi) in enumerate(zip(x_arr, y_measured)):
        c = _COLORS[idx % len(_COLORS)]
        ax.plot([xi, xi], [y_baseline, yi],
                color=c, linewidth=stem_linewidth,
                solid_capstyle='butt', zorder=2)
        ax.plot(xi, yi, 'o', color=c,
                markersize=markersize, markeredgewidth=0, zorder=3)

    # Theoretical: open black circles, larger, no stems
    if theory_vals:
        y_theory = [theory_vals[p] for p in valid_orders if p in theory_vals]
        x_theory = [p for p in valid_orders if p in theory_vals]
        ax.plot(x_theory, y_theory, 'o',
                markerfacecolor='none', markeredgecolor='black',
                markeredgewidth=1.5, markersize=theory_markersize,
                zorder=4, label='Theory')

    ax.axhline(y_baseline, color='#333333', linewidth=0.8, zorder=1)

    ax.set_xlabel(r'Harmonic order $p$', fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_xlim(x_arr[0] - 0.8, x_arr[-1] + 0.8)
    if DISPLAY_MODE == 'percent':
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(bottom=y_baseline, top=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)

    title_parts = [f"Step {step}"]
    if beta1_val is not None:
        title_parts.append(rf"$\beta_1$={beta1_val:.2f}, $\beta_2$={beta2_val:.2f}")
    if phi1_val is not None:
        title_parts.append(f"φ1={phi1_val:.0f}°, φ2={phi2_val:.0f}°")
    ax.set_title(",  ".join(title_parts), fontsize=axis_label_fontsize)

    if theory_vals:
        ax.legend(fontsize=tick_label_fontsize, frameon=False)

    plt.show()


if __name__ == '__main__':
    main()
