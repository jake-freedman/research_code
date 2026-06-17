import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.ndimage import maximum_filter
import os

# ============================================================
# USER PARAMETERS
# ============================================================

# Ring 1 properties
F1_GHZ = 400e3 + 0.1535       # Ring 1 absolute resonance frequency (GHz)
Q_I1   = 1e6         # Ring 1 intrinsic Q
Q_E1   = 1e6         # Ring 1 extrinsic (waveguide coupling) Q

# Ring 2 properties
F2_GHZ = 400e3 + 2.4985   # Ring 2 absolute resonance frequency (GHz)
Q_I2   = 1e6         # Ring 2 intrinsic Q
Q_E2   = 1e6         # Ring 2 extrinsic (waveguide coupling) Q

# Probe (input carrier) frequency
F_PROBE_GHZ = 400e3  # Fixed probe frequency (GHz) [used when probe is not swept]
PHI         = 0.0    # Inter-ring phase shift (rad) [used when phi is not swept]

# Phase modulation comb
F_MOD_GHZ  = 2.5     # Modulation frequency (GHz)
BETA       = 1.5     # Phase modulation depth (rad)
HARMONIC_M = 2       # Intensity harmonic to evaluate (integer >= 1)
N_SIDEBAND = 20      # Sidebands included: n = -N_SIDEBAND … +N_SIDEBAND

# Sweep ranges (used for whichever parameters are selected in PLOT_AXES)
PROBE_RANGE_GHZ = (-10.0 + 400e3, 10.0 + 400e3)
F1_RANGE_GHZ    = (-5.0  + 400e3,  5.0 + 400e3)
F2_RANGE_GHZ    = (-5.0  + 400e3,  5.0 + 400e3)
PHI_RANGE       = (0.0, 2*np.pi)
N_POINTS        = 300

# Which two parameters to sweep; all others held fixed at values above.
# Options: 'probe_phi'  |  'probe_F1'  |  'probe_F2'  |  'F1_F2'
PLOT_AXES = 'probe_phi'

# Heatmap colour scale
COLORMAP_AM    = 'viridis'
COLORMAP_DC    = 'inferno'
COLORMAP_MAX   = None   # Upper limit for AM colour scale; None = auto (data max)

# Number of top local maxima to report in the console
N_TOP_MAXIMA = 5

# Diagnostic plots: number of sidebands to display in comb bar charts
N_SHOW_DIAG = 5

# 1D slice: phi value and ring frequencies used (independent of PLOT_AXES)
PHI_1D_SLICE = 0.0   # phi for the 1D probe-frequency slice

# Figure dimensions (each axes panel, mm)
AXES_WIDTH_MM  = 65.0
AXES_HEIGHT_MM = 65.0
PANEL_GAP_MM   = 10.0
CBAR_GAP_MM    = 2.0
CBAR_WIDTH_MM  = 4.0

# Figure margins (mm)
LEFT_MM   = 20.0
RIGHT_MM  = 8.0
BOTTOM_MM = 15.0
TOP_MM    = 8.0

# Publication mode: suppresses all axis and tick labels when True
FOR_PUBLICATION = False

# Phase display range: values wrapped to [PHASE_MIN, PHASE_MIN + 2π)
PHASE_MIN    = -np.pi
PHASE_OFFSET = 0

# 1D slice figure panel height (mm)
AXES_1D_HEIGHT_MM = 45.0

# Output
MEDIA_DIR          = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media"
OUTPUT_FILENAME    = "coupled_ring_am.svg"
OUTPUT_FILENAME_1D = "coupled_ring_am_1d.svg"

# ============================================================


def _pi_label(val):
    from fractions import Fraction
    frac = Fraction(val / np.pi).limit_denominator(16)
    n, d = frac.numerator, frac.denominator
    if n == 0:
        return '0'
    num_str = '' if abs(n) == 1 else str(abs(n))
    sign    = '-' if n < 0 else ''
    return f'{sign}{num_str}π' if d == 1 else f'{sign}{num_str}π/{d}'


def _phase_ticks(phase_min, n=5):
    ticks  = np.linspace(phase_min, phase_min + 2*np.pi, n)
    labels = [_pi_label(v) for v in ticks]
    return ticks, labels


def _wrap_phase(raw):
    return (raw + PHASE_OFFSET - PHASE_MIN) % (2*np.pi) + PHASE_MIN


# ============================================================
# Coupled-ring physics (identical to coupled_ring_transmission.py)
# ============================================================

def build_matrix(Delta1, Delta2, phi, gamma_i1, gamma_e1, gamma_i2, gamma_e2):
    M00 = 1j*Delta1 - gamma_i1 - 2*gamma_e1
    M01 = 2*np.sqrt(gamma_e1*gamma_e2)*np.exp(1j*phi)
    M10 = 2*np.sqrt(gamma_e1*gamma_e2)
    M11 = 1j*Delta2 - gamma_i2 - 2*gamma_e2
    shape = np.broadcast(Delta1, Delta2, phi).shape
    M = np.empty(shape + (2, 2), dtype=complex)
    M[..., 0, 0] = M00
    M[..., 0, 1] = M01
    M[..., 1, 0] = M10
    M[..., 1, 1] = M11
    return M


def compute_output(a1, a2, gamma_e1, gamma_e2, phi):
    return np.sqrt(2*gamma_e1)*a1 - np.sqrt(2*gamma_e2)*np.exp(1j*phi)*a2


def solve(Delta1, Delta2, phi, gamma_i1, gamma_e1, gamma_i2, gamma_e2):
    M    = build_matrix(Delta1, Delta2, phi, gamma_i1, gamma_e1, gamma_i2, gamma_e2)
    F    = np.sqrt(2*gamma_e1) * np.array([-1.0, 1.0], dtype=complex)
    amps = np.linalg.solve(M, F)
    return compute_output(amps[..., 0], amps[..., 1], gamma_e1, gamma_e2, phi)


# ============================================================
# AM computation
# ============================================================

def _sideband_detunings(n, X, Y, f_mod):
    """
    Return (Delta1_n, Delta2_n, phi) for sideband n given meshgrid (X, Y)
    and the current PLOT_AXES setting.
    """
    if PLOT_AXES == 'probe_phi':
        return X - F1_GHZ + n*f_mod, X - F2_GHZ + n*f_mod, Y
    elif PLOT_AXES == 'probe_F1':
        return X - Y + n*f_mod, X - F2_GHZ + n*f_mod, PHI
    elif PLOT_AXES == 'probe_F2':
        return X - F1_GHZ + n*f_mod, X - Y + n*f_mod, PHI
    elif PLOT_AXES == 'F1_F2':
        return F_PROBE_GHZ - X + n*f_mod, F_PROBE_GHZ - Y + n*f_mod, PHI
    else:
        raise ValueError(f"Unknown PLOT_AXES: {PLOT_AXES!r}")


def compute_am(X, Y, f_mod, n_arr, Jn, gamma_i1, gamma_e1, gamma_i2, gamma_e2):
    """
    Compute |I_m| and I_0 at every point of grid (X, Y).

    Loops over sidebands to keep memory bounded.

    Returns
    -------
    I_m : ndarray, same shape as X  — one-sided AM amplitude at harmonic HARMONIC_M
    I_0 : ndarray, same shape as X  — DC output power
    """
    num_n = len(n_arr)
    c = np.zeros(X.shape + (num_n,), dtype=complex)

    for idx, n in enumerate(n_arr):
        D1, D2, phi_ = _sideband_detunings(n, X, Y, f_mod)
        c[..., idx] = solve(D1, D2, phi_, gamma_i1, gamma_e1, gamma_i2, gamma_e2) * Jn[idx]

    I_0 = np.sum(np.abs(c)**2, axis=-1)
    I_m_complex = np.sum(
        c[..., HARMONIC_M:] * np.conj(c[..., :num_n - HARMONIC_M]),
        axis=-1,
    )
    return 2.0 * np.abs(I_m_complex), I_0


def compute_am_probe_slice(probe_arr, phi_fixed, f1, f2, f_mod, n_arr, Jn,
                           gamma_i1, gamma_e1, gamma_i2, gamma_e2):
    """1D probe-frequency sweep with fixed phi, F1, F2."""
    num_n = len(n_arr)
    c = np.zeros((len(probe_arr), num_n), dtype=complex)
    for idx, n in enumerate(n_arr):
        D1 = probe_arr - f1 + n*f_mod
        D2 = probe_arr - f2 + n*f_mod
        c[:, idx] = solve(D1, D2, phi_fixed, gamma_i1, gamma_e1, gamma_i2, gamma_e2) * Jn[idx]
    I_0 = np.sum(np.abs(c)**2, axis=-1)
    I_m_complex = np.sum(c[:, HARMONIC_M:] * np.conj(c[:, :num_n - HARMONIC_M]), axis=-1)
    return 2.0 * np.abs(I_m_complex), I_0


# ============================================================
# Reporting
# ============================================================

def print_top_maxima(I_m, x_axis, y_axis, x_label, y_label):
    nbr = max(5, I_m.shape[0] // 20)
    local_max = I_m == maximum_filter(I_m, size=nbr, mode='nearest')
    b = nbr // 2
    local_max[:b, :] = local_max[-b:, :] = False
    local_max[:, :b] = local_max[:, -b:] = False

    indices = np.argwhere(local_max)
    if len(indices) == 0:
        print("No local maxima found.")
        return None

    values = I_m[local_max]
    order  = np.argsort(values)[::-1]
    top    = min(N_TOP_MAXIMA, len(order))

    # Strip non-ASCII for safe console output on Windows
    xl = x_label.encode('ascii', errors='replace').decode('ascii')
    yl = y_label.encode('ascii', errors='replace').decode('ascii')
    print(f"\nTop {top} local maxima of |I_{HARMONIC_M}|:")
    print(f"  {'Rank':<5}  {xl:>18}  {yl:>18}  {'|I_m|':>10}")
    print("  " + "-" * 60)
    for rank, idx in enumerate(order[:top], start=1):
        row, col = indices[idx]
        print(f"  {rank:<5}  {x_axis[col]:>18.4f}  {y_axis[row]:>18.4f}"
              f"  {I_m[row, col]:>10.5f}")
    print()

    row0, col0 = indices[order[0]]
    return x_axis[col0], y_axis[row0]


# ============================================================
# Plotting helpers
# ============================================================

def style_ax(ax, x_label, y_label):
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    if FOR_PUBLICATION:
        ax.tick_params(axis='both', direction='in', width=2, length=4,
                       labelbottom=False, labelleft=False, top=True, right=True)
    else:
        ax.tick_params(axis='both', direction='in', width=2, length=4,
                       labelsize=8, top=True, right=True)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)


def _add_axes_mm(fig, fig_w_mm, fig_h_mm, left_mm, bottom_mm, w_mm, h_mm):
    return fig.add_axes([
        left_mm   / fig_w_mm,
        bottom_mm / fig_h_mm,
        w_mm      / fig_w_mm,
        h_mm      / fig_h_mm,
    ])


# ============================================================
# Diagnostic plots
# ============================================================

def plot_diagnostics(x_best, y_best, f_mod, n_arr, Jn,
                     gamma_i1, gamma_e1, gamma_i2, gamma_e2):
    """
    Four diagnostic figures at the best-heatmap-point parameters:
      1. Coupled-ring power transmission vs probe detuning
      2. Coupled-ring phase vs probe detuning
      3. Input PM comb (bar chart, coloured by sideband phase)
      4. Output PM comb (bar chart, coloured by sideband phase after ring)
    """
    # Resolve physical parameters at the best grid point
    if PLOT_AXES == 'probe_phi':
        phi_best             = y_best
        f1_best, f2_best     = F1_GHZ, F2_GHZ
    elif PLOT_AXES == 'probe_F1':
        f1_best              = y_best
        phi_best, f2_best    = PHI, F2_GHZ
    elif PLOT_AXES == 'probe_F2':
        f2_best              = y_best
        phi_best, f1_best    = PHI, F1_GHZ
    elif PLOT_AXES == 'F1_F2':
        f1_best, f2_best     = x_best, y_best
        phi_best             = PHI

    # Frequency axis: detuning from F_PROBE_GHZ (nominal carrier), spanning ±N_SHOW sidebands.
    # Referencing to F_PROBE_GHZ keeps ring resonances at fixed, predictable positions
    # (F1 - F_PROBE_GHZ and F2 - F_PROBE_GHZ) regardless of where probe_best landed.
    x_min = -(N_SHOW_DIAG + 0.5) * f_mod
    x_max =  (N_SHOW_DIAG + 0.5) * f_mod
    det_arr = np.linspace(x_min, x_max, 3000)   # detuning from F_PROBE_GHZ
    f_arr   = F_PROBE_GHZ + det_arr              # absolute optical frequency

    t_arr  = solve(f_arr - f1_best, f_arr - f2_best, phi_best,
                   gamma_i1, gamma_e1, gamma_i2, gamma_e2)
    T_arr  = np.abs(t_arr)**2
    ph_arr = _wrap_phase(np.angle(t_arr))

    pha_ticks, pha_labels = _phase_ticks(PHASE_MIN)

    # Shared axes geometry for line-plot figures
    mm    = 1 / 25.4
    ax_w  = AXES_WIDTH_MM  * mm
    ax_h  = AXES_HEIGHT_MM * 0.6 * mm
    lft   = LEFT_MM   * mm
    bot   = BOTTOM_MM * mm
    rgt   = RIGHT_MM  * mm
    top   = TOP_MM    * mm

    def _line_fig():
        fw = lft + ax_w + rgt
        fh = bot + ax_h + top
        f  = plt.figure(figsize=(fw, fh))
        a  = f.add_axes([lft/fw, bot/fh, ax_w/fw, ax_h/fh])
        return f, a

    def _style_diag(ax, ylabel):
        for sp in ax.spines.values():
            sp.set_linewidth(2)
        ax.tick_params(direction='in', width=2, length=4, labelsize=8,
                       top=True, right=True)
        ax.set_xlabel('Detuning from carrier (GHz)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(x_min, x_max)
        for n in range(-N_SHOW_DIAG, N_SHOW_DIAG + 1):
            ax.axvline(n * f_mod, color='gray', lw=0.5, ls=':', alpha=0.6)

    # ---- Figure 1: Power transmission ----
    fig1, ax1 = _line_fig()
    ax1.plot(det_arr, T_arr, color='#2522D4', lw=1.5)
    ax1.set_ylim(-0.05, 1.1)
    _style_diag(ax1, '|T|²')
    _save(fig1, f'coupled_ring_am_T_m{HARMONIC_M}.svg')

    # ---- Figure 2: Phase ----
    fig2, ax2 = _line_fig()
    ax2.plot(det_arr, ph_arr, color='#2522D4', lw=1.5)
    ax2.set_ylim(PHASE_MIN, PHASE_MIN + 2*np.pi)
    ax2.set_yticks(pha_ticks)
    if not FOR_PUBLICATION:
        ax2.set_yticklabels(pha_labels, fontsize=8)
    _style_diag(ax2, '∠T (rad)')
    _save(fig2, f'coupled_ring_am_phase_m{HARMONIC_M}.svg')

    # ---- Comb figures (input & output) ----
    n_show_arr = np.arange(-N_SHOW_DIAG, N_SHOW_DIAG + 1)
    idx_show   = n_show_arr + N_SIDEBAND
    Jn_show    = Jn[idx_show]

    in_db     = 10 * np.log10(np.maximum(Jn_show**2, 1e-20))
    in_phases = _wrap_phase(np.where(Jn_show >= 0, 0.0, np.pi))

    D1_n = F_PROBE_GHZ + n_show_arr * f_mod - f1_best
    D2_n = F_PROBE_GHZ + n_show_arr * f_mod - f2_best
    c_n  = solve(D1_n, D2_n, phi_best, gamma_i1, gamma_e1, gamma_i2, gamma_e2) * Jn_show
    out_db     = 10 * np.log10(np.maximum(np.abs(c_n)**2, 1e-20))
    out_phases = _wrap_phase(np.angle(c_n))

    cmap_comb = plt.cm.twilight
    norm_comb = plt.Normalize(vmin=PHASE_MIN, vmax=PHASE_MIN + 2*np.pi)

    # Axes geometry for comb figures (include colorbar)
    cbar_gap = CBAR_GAP_MM * mm
    cbar_w   = CBAR_WIDTH_MM * mm
    cbar_rm  = (RIGHT_MM + CBAR_GAP_MM + CBAR_WIDTH_MM) * mm

    def _comb_fig():
        fw  = lft + ax_w + cbar_gap + cbar_w + cbar_rm
        fh  = bot + ax_h + top
        f   = plt.figure(figsize=(fw, fh))
        a   = f.add_axes([lft/fw, bot/fh, ax_w/fw, ax_h/fh])
        ca  = f.add_axes([(lft + ax_w + cbar_gap)/fw, bot/fh, cbar_w/fw, ax_h/fh])
        return f, a, ca

    def _draw_comb(ax, cax, n_arr_show, db, phases, title_str, fname):
        db_floor  = max(db.max() - 60, db[db > -200].min())
        colors    = cmap_comb(norm_comb(phases))
        freq_pos  = n_arr_show * f_mod          # x positions in GHz
        bar_width = 0.6 * f_mod                 # bar width scales with spacing
        ax.bar(freq_pos, np.maximum(db, db_floor) - db_floor, width=bar_width,
               color=colors, align='center', bottom=db_floor)
        ax.set_ylim(db_floor, db.max() + 3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        _style_diag(ax, 'P_n (dB)')
        sm = plt.cm.ScalarMappable(cmap=cmap_comb, norm=norm_comb)
        sm.set_array([])
        cb = ax.get_figure().colorbar(sm, cax=cax)
        cb.set_ticks(pha_ticks)
        cb.set_ticklabels(pha_labels)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_linewidth(2)
        if not FOR_PUBLICATION:
            ax.set_title(title_str, fontsize=10)
        _save(ax.get_figure(), fname)

    fig3, ax3, cax3 = _comb_fig()
    _draw_comb(ax3, cax3, n_show_arr, in_db,  in_phases,
               'Input PM comb', f'coupled_ring_am_comb_in_m{HARMONIC_M}.svg')

    fig4, ax4, cax4 = _comb_fig()
    _draw_comb(ax4, cax4, n_show_arr, out_db, out_phases,
               'Output comb', f'coupled_ring_am_comb_out_m{HARMONIC_M}.svg')


def _save(fig, filename):
    path = os.path.join(MEDIA_DIR, filename)
    fig.savefig(path, format='svg')
    print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    # Convert Q to cyclic decay rates (GHz)
    gamma_i1 = F1_GHZ / Q_I1
    gamma_e1 = F1_GHZ / Q_E1
    gamma_i2 = F2_GHZ / Q_I2
    gamma_e2 = F2_GHZ / Q_E2

    f_mod = F_MOD_GHZ
    n_arr = np.arange(-N_SIDEBAND, N_SIDEBAND + 1)
    Jn    = jv(n_arr, BETA)

    if HARMONIC_M > N_SIDEBAND:
        raise ValueError(f"HARMONIC_M={HARMONIC_M} exceeds N_SIDEBAND={N_SIDEBAND}")

    # Build sweep arrays
    probe_arr = np.linspace(*PROBE_RANGE_GHZ, N_POINTS)
    f1_arr    = np.linspace(*F1_RANGE_GHZ,    N_POINTS)
    f2_arr    = np.linspace(*F2_RANGE_GHZ,    N_POINTS)
    phi_arr   = np.linspace(*PHI_RANGE,        N_POINTS)

    if PLOT_AXES == 'probe_phi':
        X, Y = np.meshgrid(probe_arr, phi_arr)
        x_axis, y_axis = probe_arr, phi_arr
        x_ref,  y_ref  = F_PROBE_GHZ, 0.0
        x_label, y_label = 'Δf_carrier (GHz)', 'φ (rad)'
    elif PLOT_AXES == 'probe_F1':
        X, Y = np.meshgrid(probe_arr, f1_arr)
        x_axis, y_axis = probe_arr, f1_arr
        x_ref,  y_ref  = F_PROBE_GHZ, F1_GHZ
        x_label, y_label = 'Δf_carrier (GHz)', 'F₁ − F₁⁰ (GHz)'
    elif PLOT_AXES == 'probe_F2':
        X, Y = np.meshgrid(probe_arr, f2_arr)
        x_axis, y_axis = probe_arr, f2_arr
        x_ref,  y_ref  = F_PROBE_GHZ, F2_GHZ
        x_label, y_label = 'Δf_carrier (GHz)', 'F₂ − F₂⁰ (GHz)'
    elif PLOT_AXES == 'F1_F2':
        X, Y = np.meshgrid(f1_arr, f2_arr)
        x_axis, y_axis = f1_arr, f2_arr
        x_ref,  y_ref  = F1_GHZ, F2_GHZ
        x_label, y_label = 'F₁ − F₁⁰ (GHz)', 'F₂ − F₂⁰ (GHz)'
    else:
        raise ValueError(f"Unknown PLOT_AXES: {PLOT_AXES!r}")

    # Display axes: detuning from reference (avoids matplotlib large-number offset issues)
    x_plot = x_axis - x_ref
    y_plot = y_axis - y_ref

    print(f"Computing AM heatmap ({N_POINTS}×{N_POINTS} grid, "
          f"{2*N_SIDEBAND+1} sidebands, m={HARMONIC_M})...")
    I_m, I_0 = compute_am(X, Y, f_mod, n_arr, Jn,
                           gamma_i1, gamma_e1, gamma_i2, gamma_e2)
    print("Done.")

    best = print_top_maxima(I_m, x_axis, y_axis, x_label, y_label)

    # ---- 2D heatmap figure ----
    pha_ticks, pha_labels = _phase_ticks(PHASE_MIN)
    vmax   = COLORMAP_MAX if COLORMAP_MAX is not None else I_m.max()
    extent = [x_plot[0], x_plot[-1], y_plot[0], y_plot[-1]]

    panel_stride = AXES_WIDTH_MM + CBAR_GAP_MM + CBAR_WIDTH_MM + PANEL_GAP_MM
    fig_w_mm = (LEFT_MM + 2*AXES_WIDTH_MM + 2*CBAR_GAP_MM
                + 2*CBAR_WIDTH_MM + PANEL_GAP_MM + RIGHT_MM)
    fig_h_mm = BOTTOM_MM + AXES_HEIGHT_MM + TOP_MM
    fig = plt.figure(figsize=(fig_w_mm/25.4, fig_h_mm/25.4))

    def _panel(left_mm):
        return _add_axes_mm(fig, fig_w_mm, fig_h_mm,
                            left_mm, BOTTOM_MM, AXES_WIDTH_MM, AXES_HEIGHT_MM)

    def _cbar_ax(left_mm):
        return _add_axes_mm(fig, fig_w_mm, fig_h_mm,
                            left_mm, BOTTOM_MM, CBAR_WIDTH_MM, AXES_HEIGHT_MM)

    ax_am  = _panel(LEFT_MM)
    cax_am = _cbar_ax(LEFT_MM + AXES_WIDTH_MM + CBAR_GAP_MM)
    ax_dc  = _panel(LEFT_MM + panel_stride)
    cax_dc = _cbar_ax(LEFT_MM + panel_stride + AXES_WIDTH_MM + CBAR_GAP_MM)

    im_am = ax_am.imshow(I_m, origin='lower', aspect='auto', extent=extent,
                         vmin=0.0, vmax=vmax, cmap=COLORMAP_AM)
    im_dc = ax_dc.imshow(I_0, origin='lower', aspect='auto', extent=extent,
                         vmin=0.0, vmax=1.0, cmap=COLORMAP_DC)

    cb_am = fig.colorbar(im_am, cax=cax_am)
    cb_dc = fig.colorbar(im_dc, cax=cax_dc)
    cb_dc.set_ticks([0.0, 0.5, 1.0])

    for cb in (cb_am, cb_dc):
        cb.outline.set_linewidth(2)
        cb.ax.tick_params(direction='in', width=2, length=4,
                          labelsize=0 if FOR_PUBLICATION else 8)

    style_ax(ax_am, x_label, y_label)
    style_ax(ax_dc, x_label, '')

    if not FOR_PUBLICATION:
        ax_am.set_title(f'|I_{HARMONIC_M}|', fontsize=10)
        ax_dc.set_title('DC power  I₀',     fontsize=10)

    os.makedirs(MEDIA_DIR, exist_ok=True)
    _save(fig, OUTPUT_FILENAME)

    # ---- 1D slice ----
    probe_1d    = np.linspace(*PROBE_RANGE_GHZ, N_POINTS)
    I_m_1d, I_0_1d = compute_am_probe_slice(
        probe_1d, PHI_1D_SLICE, F1_GHZ, F2_GHZ,
        f_mod, n_arr, Jn, gamma_i1, gamma_e1, gamma_i2, gamma_e2,
    )

    fig1d_w_mm = LEFT_MM + 2*AXES_WIDTH_MM + PANEL_GAP_MM + RIGHT_MM
    fig1d_h_mm = BOTTOM_MM + AXES_1D_HEIGHT_MM + TOP_MM
    fig1d = plt.figure(figsize=(fig1d_w_mm/25.4, fig1d_h_mm/25.4))

    def _1d_panel(left_mm):
        return _add_axes_mm(fig1d, fig1d_w_mm, fig1d_h_mm,
                            left_mm, BOTTOM_MM, AXES_WIDTH_MM, AXES_1D_HEIGHT_MM)

    ax1d_am = _1d_panel(LEFT_MM)
    ax1d_dc = _1d_panel(LEFT_MM + AXES_WIDTH_MM + PANEL_GAP_MM)

    probe_1d_plot = probe_1d - F_PROBE_GHZ
    ax1d_am.plot(probe_1d_plot, I_m_1d, color='k', linewidth=1.5)
    ax1d_dc.plot(probe_1d_plot, I_0_1d, color='k', linewidth=1.5)
    ax1d_dc.set_ylim(0.0, 1.0)

    style_ax(ax1d_am, 'Δf_carrier (GHz)', f'|I_{HARMONIC_M}|')
    style_ax(ax1d_dc, 'Δf_carrier (GHz)', 'I₀')

    if not FOR_PUBLICATION:
        phi_str = _pi_label(PHI_1D_SLICE) if PHI_1D_SLICE != 0 else '0'
        ax1d_am.set_title(f'|I_{HARMONIC_M}|  (φ = {phi_str})', fontsize=10)
        ax1d_dc.set_title(f'DC power  (φ = {phi_str})',          fontsize=10)

    _save(fig1d, OUTPUT_FILENAME_1D)

    # ---- Diagnostic plots at best point ----
    if best is not None:
        plot_diagnostics(best[0], best[1], f_mod, n_arr, Jn,
                         gamma_i1, gamma_e1, gamma_i2, gamma_e2)

    plt.show()


if __name__ == '__main__':
    main()
