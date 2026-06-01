#!/usr/bin/env python3
"""
pm_ring_am_heatmap.py

Heatmap of amplitude modulation produced by a sinusoidal phase modulator
followed by a single-bus (all-pass) ring resonator.

Physics
-------
Input field (normalised amplitude = 1):
    E(t) = exp(i * omega_c * t)

After phase modulator:
    E_pm(t) = sum_n  J_n(beta) * exp(i*(omega_c + n*Omega)*t)

After ring resonator (through port):
    E_out(t) = sum_n  c_n * exp(i*(omega_c + n*Omega)*t)
    c_n = H(n*Omega + Delta) * J_n(beta)

Intensity decomposed into harmonics:
    I(t) = I_0 + sum_{m=1}^inf  |I_m| * cos(m*Omega*t + phi_m)

    I_0   = sum_n |c_n|^2                              (DC power)
    |I_m| = 2 * |sum_n  c_n * conj(c_{n-m})|          (one-sided amplitude at harmonic m)

Heatmap shows  |I_m|  (one-sided harmonic amplitude, normalised to unit input power).

Sign convention
---------------
Delta = omega_carrier - omega_ring
    Delta > 0  ->  ring resonance is RED-detuned from the carrier
    Delta < 0  ->  ring resonance is BLUE-detuned from the carrier
    n-th sideband is on resonance when  Delta / Omega = -n
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.ndimage import maximum_filter


# ===========================================================================
# USER PARAMETERS  (edit these)
# ===========================================================================

# --- Modulation ---
OMEGA_MOD_HZ = 2.5e9       # Modulation frequency in Hz
HARMONIC_M   = [4]         # Which intensity harmonic to display (integer >= 1)

# --- Phase modulation depth ---
BETA          = 1.5       # Modulation depth (rad);  used when OPTIMIZE_BETA = False
OPTIMIZE_BETA = True     # If True: maximise |I_m|/I_0 over beta at each grid point
BETA_RANGE    = (0.1, 5.0)  # (beta_min, beta_max) search range (rad)
N_BETA_GRID   = 60        # Number of beta values in the grid search

# --- Ring intrinsic loss ---
# Set ONE of the two options below; set the other to None.
GAMMA_I_HZ   = 100e6     # Intrinsic loss half-linewidth in Hz  [= gamma_i / (2*pi)]
Q_INTRINSIC  = 1e6      # Intrinsic Q factor  [overrides GAMMA_I_HZ when not None]
OMEGA_OPT_HZ = 400e12    # Optical centre frequency in Hz  (needed only with Q_INTRINSIC)

# --- Heatmap axis ranges ---
# gamma_e axis: set ONE pair; set the other pair to None.
#   Hz option:  coupling half-linewidth in Hz  [= gamma_e / (2*pi)]
GAMMA_E_MIN_HZ =  1e6
GAMMA_E_MAX_HZ =  None
#   Q_e option: external coupling Q factor (overrides _HZ when not None)
Q_E_MIN        =  0.1 * Q_INTRINSIC 
Q_E_MAX        =  10 * Q_INTRINSIC
N_GAMMA_E      = 300

# Delta axis: detuning in Hz  [= Delta / (2*pi) = (omega_c - omega_r) / (2*pi)]
DELTA_MIN_HZ = -5 * OMEGA_MOD_HZ
DELTA_MAX_HZ =  5 * OMEGA_MOD_HZ
N_DELTA      = 300

# --- Numerical ---
N_SIDEBAND    = 20        # Include sidebands n = -N_SIDEBAND … +N_SIDEBAND
N_TOP_MAXIMA  = 10         # Number of top local maxima to report

# --- Plot ---
NORMALIZE_AXES  = True    # True: axes in units of Omega;  False: GHz
LOG_GAMMA_E     = False    # Log scale for the gamma_e axis
GAMMA_E_DB      = False   # If True (and NORMALIZE_AXES=True): x-axis in dB [10*log10(ge/gi)]
COLORMAP        = "viridis"
COLORMAP_MAX    = 1.2     # Upper limit of the color scale (vmax)
AX_SIZE_MM      = 60     # Axes box side length in mm (plot is always square)

# ===========================================================================
# END USER PARAMETERS
# ===========================================================================


def ring_transmission(delta_omega, omega_0, Q_i, Q_e):
    """
    Single-bus ring resonator field transmission (through port).

        H = (2j*delta_omega/omega_0 + 1/Q_i - 1/Q_e)
            / (2j*delta_omega/omega_0 + 1/Q_i + 1/Q_e)

    Parameters
    ----------
    delta_omega : float or ndarray  (rad/s)
        Signal detuning from ring resonance: omega_signal - omega_ring.
        For the n-th PM sideband: delta_omega = n*Omega + Delta,
        where Delta = omega_carrier - omega_ring.
    omega_0 : float  (rad/s)
        Ring resonance angular frequency (used for normalisation).
    Q_i : float or ndarray
        Intrinsic (internal) quality factor.
    Q_e : float or ndarray
        External (coupling) quality factor.

    Returns
    -------
    complex ndarray

    Limiting cases
    --------------
    delta_omega = 0, Q_i = Q_e   ->  H = 0     (critical coupling, full extinction)
    delta_omega = 0, Q_e < Q_i   ->  H < 0     (over-coupled, pi phase shift)
    delta_omega = 0, Q_e -> inf  ->  H -> +1   (ring decoupled from waveguide)
    |delta_omega| >> omega_0/Q_total  ->  H -> +1   (far off resonance)
    """
    norm = 2j * delta_omega / omega_0
    return (norm + 1/Q_i - 1/Q_e) / (norm + 1/Q_i + 1/Q_e)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _build_H_3d(delta_arr, gamma_e_arr, gamma_i, omega_mod, omega_0):
    """
    Pre-compute ring transmission for all (Delta, gamma_e, sideband-index) combinations.

    Returns H_3d with shape (N_delta, N_gamma_e, 2*N_SIDEBAND+1).
    This is independent of beta, so it is computed once and reused in the beta loop.
    """
    Q_i   = omega_0 / gamma_i                               # scalar
    Q_e   = omega_0 / gamma_e_arr                           # (N_gamma_e,)
    n = np.arange(-N_SIDEBAND, N_SIDEBAND + 1)             # (2N+1,)
    delta_n_2d = delta_arr[:, None] + n[None, :] * omega_mod   # (N_d, 2N+1)
    return ring_transmission(
        delta_n_2d[:, None, :],           # (N_delta,   1,         2N+1)
        omega_0,                           # scalar
        Q_i,                               # scalar
        Q_e[None, :, None],                # (1,          N_gamma_e, 1  )
    )                                      # broadcast -> (N_delta, N_gamma_e, 2N+1)


def _am_from_H(harmonic_m, beta, H_3d):
    """
    Compute (|I_m|, I_0) for a single beta, given pre-computed H_3d.

    Returns two arrays of shape (N_delta, N_gamma_e):
        |I_m| = 2 * |sum_n c_n * conj(c_{n-m})|   (one-sided harmonic amplitude)
        I_0   = sum_n |c_n|^2                       (DC power)
    """
    n   = np.arange(-N_SIDEBAND, N_SIDEBAND + 1)           # (2N+1,)
    Jn  = jv(n, beta)                                       # (2N+1,) real
    c   = H_3d * Jn[np.newaxis, np.newaxis, :]              # (N_d, N_g, 2N+1) complex
    num_n = 2 * N_SIDEBAND + 1

    I_m_complex = np.sum(
        c[..., harmonic_m:] * np.conj(c[..., :num_n - harmonic_m]),
        axis=-1,
    )                                                        # (N_d, N_g) complex
    I_0 = np.sum(np.abs(c) ** 2, axis=-1)                   # (N_d, N_g) real
    return 2.0 * np.abs(I_m_complex), I_0


def compute_heatmap(harmonic_m, delta_arr, gamma_e_arr, gamma_i, omega_mod, omega_0,
                    beta=1.5, optimize=False,
                    beta_range=(0.1, 5.0), n_beta_grid=60):
    """
    Compute the |I_m| heatmap and the corresponding I_0 map.

    Returns
    -------
    heatmap  : ndarray, shape (N_delta, N_gamma_e)   |I_m| values
    I_0_map  : ndarray, shape (N_delta, N_gamma_e)   DC power at the same beta
    beta_map : ndarray, shape (N_delta, N_gamma_e)   beta used at each point
               (constant array equal to beta when optimize=False)
    """
    H_3d = _build_H_3d(delta_arr, gamma_e_arr, gamma_i, omega_mod, omega_0)

    if not optimize:
        I_m, I_0 = _am_from_H(harmonic_m, beta, H_3d)
        return I_m, I_0, np.full_like(I_m, beta)

    beta_grid = np.linspace(beta_range[0], beta_range[1], n_beta_grid)
    shape    = (len(delta_arr), len(gamma_e_arr))
    heatmap  = np.zeros(shape)
    I_0_map  = np.ones(shape)
    beta_map = np.full(shape, beta_grid[0])

    print(f"Optimising beta: scanning {n_beta_grid} values "
          f"in [{beta_range[0]:.2f}, {beta_range[1]:.2f}] rad …")
    for bval in beta_grid:
        I_m, I_0 = _am_from_H(harmonic_m, bval, H_3d)
        improved = I_m > heatmap
        heatmap  = np.where(improved, I_m,   heatmap)
        I_0_map  = np.where(improved, I_0,   I_0_map)
        beta_map = np.where(improved, bval,  beta_map)
    print("Done.")

    return heatmap, I_0_map, beta_map


# ---------------------------------------------------------------------------
# Local-maxima reporting
# ---------------------------------------------------------------------------

def print_top_maxima(heatmap, beta_map, gamma_e_arr, delta_arr, gamma_i, omega_mod,
                     harmonic_m, n=3):
    """
    Find the top-n local maxima in the |I_m| heatmap and print their locations,
    the phase modulation depth beta, and |I_m|.
    """
    nbr = max(5, heatmap.shape[0] // 20)
    local_max = heatmap == maximum_filter(heatmap, size=nbr, mode="nearest")
    b = nbr // 2
    local_max[:b, :] = local_max[-b:, :] = False
    local_max[:, :b] = local_max[:, -b:] = False

    indices = np.argwhere(local_max)
    if len(indices) == 0:
        print("No local maxima found.")
        return

    values = heatmap[local_max]
    order  = np.argsort(values)[::-1]
    top    = min(n, len(order))

    print(f"\nTop {top} local maxima of |I_{harmonic_m}|:")
    hdr = (f"  {'Rank':<5} {'ge/gi':>9} {'D/Omega':>10} "
           f"{'ge/2pi(MHz)':>13} {'D/2pi(MHz)':>12} "
           f"{'beta(rad)':>11} {'|I_m|':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr.rstrip()) - 2))
    for rank, idx in enumerate(order[:top], start=1):
        row, col  = indices[idx]
        ge        = gamma_e_arr[col]
        delta     = delta_arr[row]
        I_m_val   = heatmap[row, col]
        beta_val  = beta_map[row, col]
        print(f"  {rank:<5} {ge/gamma_i:>9.3f} {delta/omega_mod:>10.3f} "
              f"{ge/(2*np.pi*1e6):>13.3f} {delta/(2*np.pi*1e6):>12.3f} "
              f"{beta_val:>11.4f} {I_m_val:>9.5f}")
    print()

    # Return best-point parameters for downstream plotting
    row0, col0 = indices[order[0]]
    return gamma_e_arr[col0], delta_arr[row0], beta_map[row0, col0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(heatmap, x_vals, y_vals, x_label, y_label, harmonic_m):
    mm = 1 / 25.4          # inches per mm

    # Axes box: exactly 100 mm × 100 mm
    ax_size   = AX_SIZE_MM * mm
    left_m    =  17 * mm   # y-label + tick numbers
    bottom_m  =  14 * mm   # x-label + tick numbers
    top_m     =   3 * mm   # small top margin (no title)
    cbar_pad  =   4 * mm   # gap between axes and colorbar
    cbar_w    =   5 * mm   # colorbar bar width
    right_m   =  15 * mm   # colorbar tick labels + colorbar label

    fig_w = left_m + ax_size + cbar_pad + cbar_w + right_m
    fig_h = bottom_m + ax_size + top_m

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([left_m / fig_w,
                        bottom_m / fig_h,
                        ax_size / fig_w,
                        ax_size / fig_h])
    cax = fig.add_axes([(left_m + ax_size + cbar_pad) / fig_w,
                        bottom_m / fig_h,
                        cbar_w / fig_w,
                        ax_size / fig_h])

    pcm  = ax.pcolormesh(x_vals, y_vals, heatmap,
                         cmap=COLORMAP, shading="auto", vmin=0.0, vmax=COLORMAP_MAX)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label(rf"$|I_{{{harmonic_m}}}|$", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.tick_params(labelsize=8)
    if LOG_GAMMA_E and not (NORMALIZE_AXES and GAMMA_E_DB):
        ax.set_xscale("log")

    import os
    fname = os.path.join(
        r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media",
        f"am_heatmap_m{harmonic_m}.png",
    )
    fig.savefig(fname, dpi=200)
    print(f"Saved → {fname}")
    plt.show()


# ---------------------------------------------------------------------------
# Ring + comb diagnostic plot
# ---------------------------------------------------------------------------

def plot_ring_and_comb(gamma_e, delta_carrier, gamma_i, omega_mod, omega_0, beta, harmonic_m):
    """
    Three separate figures for the best-solution parameters, all with identical axes size:
      1. ring_T_m{m}.png    : power transmission T = |H|^2
      2. ring_phi_m{m}.png  : ring phase phi in radians [0, 2pi]
      3. ring_comb_m{m}.png : input comb power P_n = |J_n(beta)|^2 (dB), phase-coloured

    x-axis: Delta/Omega  (frequency offset from carrier, normalised to mod. freq.)
    Only sidebands n = -5 ... +5 are shown.
    """
    import os
    SAVE_DIR = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media"
    N_SHOW   = 5
    x_min    = -(N_SHOW + 0.5)
    x_max    =  (N_SHOW + 0.5)
    x_res    = -delta_carrier / omega_mod
    x_label  = r"$\Delta/\Omega$"

    # Fine + coarse frequency grid
    gamma_total = gamma_e + gamma_i
    hw = max(10.0 * gamma_total / omega_mod, 0.5)
    x  = np.unique(np.concatenate([
        np.linspace(x_min, x_max, 3000),
        np.linspace(max(x_min, x_res - hw), min(x_max, x_res + hw), 3000),
    ]))

    # Ring curves
    delta_rad = x * omega_mod + delta_carrier
    H   = ring_transmission(delta_rad, omega_0, omega_0 / gamma_i, omega_0 / gamma_e)
    T   = np.abs(H) ** 2
    phi = np.angle(H) % (2 * np.pi)   # radians in [0, 2pi]

    # Comb (±5 sidebands only)
    n_arr      = np.arange(-N_SHOW, N_SHOW + 1)
    Jn         = jv(n_arr, beta)
    phases     = np.where(Jn >= 0, 0.0, np.pi)
    heights_db = 10 * np.log10(np.maximum(Jn ** 2, 1e-20))

    # ---- Shared axes geometry (all three figures identical) ----
    mm      = 1 / 25.4
    ax_w    = AX_SIZE_MM * mm
    ax_h    = AX_SIZE_MM * 0.6 * mm
    left_m  = 16 * mm
    right_m =  6 * mm
    bot_m   = 12 * mm
    top_m   =  3 * mm

    def _make_fig():
        fw = left_m + ax_w + right_m
        fh = bot_m  + ax_h + top_m
        fig = plt.figure(figsize=(fw, fh))
        ax  = fig.add_axes([left_m/fw, bot_m/fh, ax_w/fw, ax_h/fh])
        return fig, ax

    def _mark_resonance(ax):
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(x_label, fontsize=10)
        ax.tick_params(labelsize=8)
        if x_min <= x_res <= x_max:
            ax.axvline(x_res, color='gray', lw=0.6, ls=':', alpha=0.8)

    # ---- Figure 1: Power transmission ----
    fig1, ax1 = _make_fig()
    ax1.plot(x, T, color='#2522D4', lw=2.5)
    ax1.set_ylabel(r"$T$", fontsize=10)
    ax1.set_ylim(-0.05, 1.1)
    _mark_resonance(ax1)
    f1 = os.path.join(SAVE_DIR, f"ring_T_m{harmonic_m}.png")
    fig1.savefig(f1, dpi=200)
    print(f"Saved → {f1}")

    # ---- Figure 2: Phase ----
    fig2, ax2 = _make_fig()
    ax2.plot(x, phi, color='#2522D4', lw=2.5)
    ax2.set_ylabel(r"$\varphi$ [rad]", fontsize=10)
    ax2.set_ylim(-0.2, 2 * np.pi + 0.2)
    ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_yticklabels(['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'],
                        fontsize=8)
    _mark_resonance(ax2)
    f2 = os.path.join(SAVE_DIR, f"ring_phi_m{harmonic_m}.png")
    fig2.savefig(f2, dpi=200)
    print(f"Saved → {f2}")

    # ---- Figure 3: Input comb (with colorbar — extra right margin) ----
    cbar_pad = 4 * mm
    cbar_w   = 5 * mm
    cbar_rm  = 14 * mm
    fw3 = left_m + ax_w + cbar_pad + cbar_w + cbar_rm
    fh3 = bot_m  + ax_h + top_m
    fig3 = plt.figure(figsize=(fw3, fh3))
    ax3  = fig3.add_axes([left_m / fw3, bot_m / fh3, ax_w / fw3, ax_h / fh3])
    cax3 = fig3.add_axes([(left_m + ax_w + cbar_pad) / fw3, bot_m / fh3,
                           cbar_w / fw3, ax_h / fh3])

    cmap_cb = plt.cm.twilight
    norm_cb = plt.Normalize(vmin=0, vmax=2 * np.pi)
    colors  = cmap_cb(norm_cb(phases))

    db_floor     = max(heights_db.max() - 60, heights_db[heights_db > -200].min())
    heights_plot = np.maximum(heights_db, db_floor)
    ax3.bar(n_arr, heights_plot - db_floor, width=0.6, color=colors,
            align='center', bottom=db_floor)
    ax3.set_ylim(db_floor, heights_db.max() + 3)
    ax3.set_ylabel(r"$P_n$ [dB]", fontsize=10)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    _mark_resonance(ax3)

    sm = plt.cm.ScalarMappable(cmap=cmap_cb, norm=norm_cb)
    sm.set_array([])
    cbar3 = fig3.colorbar(sm, cax=cax3)
    cbar3.set_label(r"$\varphi_n$ [rad]", fontsize=9)
    cbar3.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    cbar3.set_ticklabels(['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    cbar3.ax.tick_params(labelsize=8)

    f3 = os.path.join(SAVE_DIR, f"ring_comb_m{harmonic_m}.png")
    fig3.savefig(f3, dpi=200)
    print(f"Saved → {f3}")

    # ---- Figure 4: Output comb c_n = H_n * J_n(beta) ----
    delta_n     = n_arr * omega_mod + delta_carrier
    H_n         = ring_transmission(delta_n, omega_0, omega_0 / gamma_i, omega_0 / gamma_e)
    c_n         = H_n * Jn
    out_db      = 10 * np.log10(np.maximum(np.abs(c_n) ** 2, 1e-20))
    out_phases  = np.angle(c_n) % (2 * np.pi)
    out_colors  = cmap_cb(norm_cb(out_phases))

    fw4 = left_m + ax_w + cbar_pad + cbar_w + cbar_rm
    fh4 = bot_m  + ax_h + top_m
    fig4 = plt.figure(figsize=(fw4, fh4))
    ax4  = fig4.add_axes([left_m / fw4, bot_m / fh4, ax_w / fw4, ax_h / fh4])
    cax4 = fig4.add_axes([(left_m + ax_w + cbar_pad) / fw4, bot_m / fh4,
                           cbar_w / fw4, ax_h / fh4])

    db_floor4     = max(out_db.max() - 60, out_db[out_db > -200].min())
    out_db_plot   = np.maximum(out_db, db_floor4)
    ax4.bar(n_arr, out_db_plot - db_floor4, width=0.6, color=out_colors,
            align='center', bottom=db_floor4)
    ax4.set_ylim(db_floor4, out_db.max() + 3)
    ax4.set_ylabel(r"$P_n$ (dB)", fontsize=10)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    _mark_resonance(ax4)

    sm4 = plt.cm.ScalarMappable(cmap=cmap_cb, norm=norm_cb)
    sm4.set_array([])
    cbar4 = fig4.colorbar(sm4, cax=cax4)
    cbar4.set_label(r"$\varphi_n$ [rad]", fontsize=9)
    cbar4.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    cbar4.set_ticklabels(['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    cbar4.ax.tick_params(labelsize=8)

    f4 = os.path.join(SAVE_DIR, f"ring_comb_out_m{harmonic_m}.png")
    fig4.savefig(f4, dpi=200)
    print(f"Saved → {f4}")

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    harmonics = HARMONIC_M if isinstance(HARMONIC_M, list) else [HARMONIC_M]
    for m in harmonics:
        if m > N_SIDEBAND:
            raise ValueError(
                f"HARMONIC_M={m} exceeds N_SIDEBAND={N_SIDEBAND}. "
                "Increase N_SIDEBAND or reduce HARMONIC_M."
            )

    # Resolve intrinsic loss rate
    omega_0 = 2 * np.pi * OMEGA_OPT_HZ
    if Q_INTRINSIC is not None:
        gamma_i = omega_0 / Q_INTRINSIC
    else:
        gamma_i = 2 * np.pi * GAMMA_I_HZ

    omega_mod = 2 * np.pi * OMEGA_MOD_HZ

    # Resolve gamma_e axis bounds
    if Q_E_MIN is not None:
        ge_min = 2 * np.pi * OMEGA_OPT_HZ / Q_E_MIN
    else:
        ge_min = 2 * np.pi * GAMMA_E_MIN_HZ
    if Q_E_MAX is not None:
        ge_max = 2 * np.pi * OMEGA_OPT_HZ / Q_E_MAX
    else:
        ge_max = 2 * np.pi * GAMMA_E_MAX_HZ

    # Build axis arrays (rad/s internally)
    if LOG_GAMMA_E:
        gamma_e_arr = np.geomspace(ge_min, ge_max, N_GAMMA_E)
    else:
        gamma_e_arr = np.linspace(ge_min, ge_max, N_GAMMA_E)

    delta_arr = np.linspace(2 * np.pi * DELTA_MIN_HZ,
                            2 * np.pi * DELTA_MAX_HZ, N_DELTA)

    # Axis display coordinates (same for every harmonic)
    if NORMALIZE_AXES:
        ratio   = gamma_e_arr / gamma_i
        if GAMMA_E_DB:
            x_vals  = 10 * np.log10(ratio)
            x_label = r"$10\log_{10}(\gamma_\mathrm{e}/\gamma_\mathrm{i})$ [dB]"
        else:
            x_vals  = ratio
            x_label = r"$\gamma_\mathrm{e}\,/\,\gamma_\mathrm{i}$"
        y_vals  = delta_arr / omega_mod
        y_label = r"$\Delta\,/\,\Omega$"
    else:
        x_vals  = gamma_e_arr / (2 * np.pi * 1e9)
        y_vals  = delta_arr   / (2 * np.pi * 1e9)
        x_label = r"$\gamma_\mathrm{e}/(2\pi)$ [GHz]"
        y_label = r"$\Delta/(2\pi)$ [GHz]   $(\omega_\mathrm{c} - \omega_\mathrm{r})$"

    for m in harmonics:
        print(f"\n--- Harmonic m = {m} ---")
        heatmap, _, beta_map = compute_heatmap(
            m, delta_arr, gamma_e_arr, gamma_i, omega_mod, omega_0,
            beta=BETA, optimize=OPTIMIZE_BETA,
            beta_range=BETA_RANGE, n_beta_grid=N_BETA_GRID,
        )
        best = print_top_maxima(heatmap, beta_map, gamma_e_arr, delta_arr, gamma_i, omega_mod,
                                harmonic_m=m, n=N_TOP_MAXIMA)
        make_plot(heatmap, x_vals, y_vals, x_label, y_label, harmonic_m=m)
        if best is not None:
            ge_best, delta_best, beta_best = best
            plot_ring_and_comb(ge_best, delta_best, gamma_i, omega_mod, omega_0, beta_best, m)


if __name__ == "__main__":
    main()
