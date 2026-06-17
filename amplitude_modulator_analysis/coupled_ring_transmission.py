import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# USER PARAMETERS
# ============================================================

# Ring 1 properties
F1_GHZ = 400e3 + 8      # Ring 1 absolute resonance frequency (GHz)
Q_I1   = 1e6         # Ring 1 intrinsic Q
Q_E1   = 1e6         # Ring 1 extrinsic (waveguide coupling) Q

# Ring 2 properes
F2_GHZ = 400e3 - 8      # Ring 2 absolute resonance frequency (GHz)
Q_I2   = 1e6         # Ring 2 intrinsic Q
Q_E2   = 1e6         # Ring 2 extrinsic (waveguide coupling) Q

# Probe (input) frequency — determines both detunings: Δ1 = F_PROBE - F1, Δ2 = F_PROBE - F2
F_PROBE_GHZ = 400e3      # Fixed probe frequency (GHz) [used when probe is not swept]
PHI         = 2.4        # Inter-ring phase shift (rad) [used when phi is not swept]

# Sweep ranges (used for whichever parameters are selected in PLOT_AXES)
PROBE_RANGE_GHZ = (-10.0 + 400e3, 10.0 + 400e3)   # (min, max) probe frequency (GHz)
F1_RANGE_GHZ    = (-5.0 + 400e3, 5.0 + 400e3)   # (min, max) ring 1 resonance (GHz)
F2_RANGE_GHZ    = (-5.0 + 400e3, 5.0 + 400e3)   # (min, max) ring 2 resonance (GHz)
PHI_RANGE       = (0.0, 2*np.pi)                 # (min, max) phi (rad)
N_POINTS        = 300                            # Grid points per axis

# Which two parameters to sweep; all others are held fixed at their values above.
# Options: 'probe_phi'  |  'probe_F1'  |  'probe_F2'  |  'F1_F2'
PLOT_AXES = 'probe_phi'

# Figure dimensions (each axes panel, in mm)
AXES_WIDTH_MM  = 65.0
AXES_HEIGHT_MM = 65.0
PANEL_GAP_MM   = 10.0   # horizontal gap between panels (colorbar-to-next-panel)
CBAR_GAP_MM    = 2.0    # gap between panel and its colorbar
CBAR_WIDTH_MM  = 4.0    # width of each colorbar

# Figure margins (mm)
LEFT_MM   = 20.0
RIGHT_MM  = 8.0
BOTTOM_MM = 15.0
TOP_MM    = 8.0

# Publication mode: suppresses all axis and tick labels when True
FOR_PUBLICATION = False

# Phase display range: values are wrapped to [PHASE_MIN, PHASE_MIN + 2π)
# Common choices: -2π (→ [-2π, 0]),  -π (→ [-π, π]),  0 (→ [0, 2π])
PHASE_MIN    = - np.pi
PHASE_OFFSET = 0   # additional offset applied before wrapping (rad)

# 1D slice figure dimensions (mm)
AXES_1D_HEIGHT_MM = 45.0   # height of each 1D panel

# Output
MEDIA_DIR          = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media"
OUTPUT_FILENAME    = "coupled_ring_transmission.svg"
OUTPUT_FILENAME_1D = "coupled_ring_transmission_1d.svg"

# ============================================================


def _pi_label(val):
    """Format a value as a multiple of π for axis tick labels."""
    from fractions import Fraction
    frac = Fraction(val / np.pi).limit_denominator(16)
    n, d = frac.numerator, frac.denominator
    if n == 0:
        return '0'
    num_str = '' if abs(n) == 1 else str(abs(n))
    sign    = '-' if n < 0 else ''
    return f'{sign}{num_str}π' if d == 1 else f'{sign}{num_str}π/{d}'


def _phase_ticks(phase_min, n=5):
    """Return n evenly spaced tick positions and labels over [phase_min, phase_min+2π]."""
    ticks  = np.linspace(phase_min, phase_min + 2*np.pi, n)
    labels = [_pi_label(v) for v in ticks]
    return ticks, labels


def build_matrix(Delta1, Delta2, phi, gamma_i1, gamma_e1, gamma_i2, gamma_e2):
    """
    Build the 2×2 system matrix M such that  M @ [a1, a2]^T = [1, -1]^T.

    All detunings and rates are cyclic frequencies in GHz (i.e. divided by 2π).
    Delta1, Delta2, phi may be scalars or broadcast-compatible arrays.

    Replace each '...' with the corresponding matrix element expression.
    """
    # ---- FILL IN MATRIX ELEMENTS BELOW --------------------------------
    M00 = 1j*Delta1 - gamma_i1 - 2*gamma_e1   # diagonal element for ring 1 (row 0, col 0)
    M01 = 2*np.sqrt(gamma_e1*gamma_e2)*np.exp(1j*phi)   # off-diagonal coupling into ring 1 from ring 2 (row 0, col 1)
    M10 = 2*np.sqrt(gamma_e1*gamma_e2)   # off-diagonal coupling into ring 2 from ring 1 (row 1, col 0)
    M11 = 1j * Delta2 - gamma_i2 - 2*gamma_e2   # diagonal element for ring 2 (row 1, col 1)
    # --------------------------------------------------------------------

    shape = np.broadcast(Delta1, Delta2, phi).shape
    M = np.empty(shape + (2, 2), dtype=complex)
    M[..., 0, 0] = M00
    M[..., 0, 1] = M01
    M[..., 1, 0] = M10
    M[..., 1, 1] = M11
    return M


def compute_output(a1, a2, gamma_e1, gamma_e2, phi):
    """
    Compute the complex output transmission amplitude from the ring amplitudes.

    Replace '...' with the correct expression for the output field.
    Power transmission = |output|^2, phase = angle(output).
    """
    # ---- FILL IN OUTPUT FIELD EXPRESSION BELOW ------------------------
    output = np.sqrt(2*gamma_e1)*a1 - np.sqrt(2*gamma_e2)*np.exp(1j*phi)*a2
    # --------------------------------------------------------------------
    return output


def solve(Delta1, Delta2, phi, gamma_i1, gamma_e1, gamma_i2, gamma_e2):
    """Invert M and return the complex output field across the parameter grid."""
    M = build_matrix(Delta1, Delta2, phi, gamma_i1, gamma_e1, gamma_i2, gamma_e2)
    F = np.sqrt(2*gamma_e1)*np.array([-1.0, 1.0], dtype=complex)
    amps = np.linalg.solve(M, F)   # shape (..., 2)
    a1 = amps[..., 0]
    a2 = amps[..., 1]
    return compute_output(a1, a2, gamma_e1, gamma_e2, phi)


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


def main():
    # Convert Q factors to cyclic decay rates in GHz
    gamma_i1 = F1_GHZ / Q_I1
    gamma_e1 = F1_GHZ / Q_E1
    gamma_i2 = F2_GHZ / Q_I2
    gamma_e2 = F2_GHZ / Q_E2

    # Sweep axes
    probe_arr = np.linspace(*PROBE_RANGE_GHZ, N_POINTS)
    f1_arr    = np.linspace(*F1_RANGE_GHZ,    N_POINTS)
    f2_arr    = np.linspace(*F2_RANGE_GHZ,    N_POINTS)
    phi_arr   = np.linspace(*PHI_RANGE,       N_POINTS)

    if PLOT_AXES == 'probe_phi':
        X, Y = np.meshgrid(probe_arr, phi_arr)
        t_out = solve(X - F1_GHZ, X - F2_GHZ, Y, gamma_i1, gamma_e1, gamma_i2, gamma_e2)
        x_axis, y_axis = probe_arr, phi_arr
        x_label, y_label = 'f_probe (GHz)', 'φ (rad)'
    elif PLOT_AXES == 'probe_F1':
        X, Y = np.meshgrid(probe_arr, f1_arr)
        t_out = solve(X - Y, X - F2_GHZ, PHI, gamma_i1, gamma_e1, gamma_i2, gamma_e2)
        x_axis, y_axis = probe_arr, f1_arr
        x_label, y_label = 'f_probe (GHz)', 'f₁ (GHz)'
    elif PLOT_AXES == 'probe_F2':
        X, Y = np.meshgrid(probe_arr, f2_arr)
        t_out = solve(X - F1_GHZ, X - Y, PHI, gamma_i1, gamma_e1, gamma_i2, gamma_e2)
        x_axis, y_axis = probe_arr, f2_arr
        x_label, y_label = 'f_probe (GHz)', 'f₂ (GHz)'
    elif PLOT_AXES == 'F1_F2':
        X, Y = np.meshgrid(f1_arr, f2_arr)
        t_out = solve(F_PROBE_GHZ - X, F_PROBE_GHZ - Y, PHI,
                      gamma_i1, gamma_e1, gamma_i2, gamma_e2)
        x_axis, y_axis = f1_arr, f2_arr
        x_label, y_label = 'f₁ (GHz)', 'f₂ (GHz)'
    else:
        raise ValueError(f"Unknown PLOT_AXES value: {PLOT_AXES!r}")

    power = np.abs(t_out) ** 2
    phase = (np.angle(t_out) + PHASE_OFFSET - PHASE_MIN) % (2*np.pi) + PHASE_MIN

    extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]

    # Figure layout: two side-by-side heatmap panels, each with a colorbar
    # LEFT | panel | cbar_gap | cbar | panel_gap | panel | cbar_gap | cbar | RIGHT
    panel_stride = AXES_WIDTH_MM + CBAR_GAP_MM + CBAR_WIDTH_MM + PANEL_GAP_MM
    fig_w_mm = LEFT_MM + 2 * AXES_WIDTH_MM + 2 * CBAR_GAP_MM + 2 * CBAR_WIDTH_MM + PANEL_GAP_MM + RIGHT_MM
    fig_h_mm = BOTTOM_MM + AXES_HEIGHT_MM + TOP_MM
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))

    def add_panel(left_mm):
        return fig.add_axes([
            left_mm / fig_w_mm,
            BOTTOM_MM / fig_h_mm,
            AXES_WIDTH_MM / fig_w_mm,
            AXES_HEIGHT_MM / fig_h_mm,
        ])

    def add_cbar_ax(left_mm):
        return fig.add_axes([
            left_mm / fig_w_mm,
            BOTTOM_MM / fig_h_mm,
            CBAR_WIDTH_MM / fig_w_mm,
            AXES_HEIGHT_MM / fig_h_mm,
        ])

    ax_pow  = add_panel(LEFT_MM)
    cax_pow = add_cbar_ax(LEFT_MM + AXES_WIDTH_MM + CBAR_GAP_MM)
    ax_pha  = add_panel(LEFT_MM + panel_stride)
    cax_pha = add_cbar_ax(LEFT_MM + panel_stride + AXES_WIDTH_MM + CBAR_GAP_MM)

    im_pow = ax_pow.imshow(power, origin='lower', aspect='auto', extent=extent,
                           vmin=0.0, vmax=1.0, cmap='inferno')
    pha_ticks, pha_labels = _phase_ticks(PHASE_MIN)
    im_pha = ax_pha.imshow(phase, origin='lower', aspect='auto', extent=extent,
                           vmin=PHASE_MIN, vmax=PHASE_MIN + 2*np.pi, cmap='twilight')

    cb_pow = fig.colorbar(im_pow, cax=cax_pow)
    cb_pow.set_ticks([0.0, 0.5, 1.0])
    cb_pha = fig.colorbar(im_pha, cax=cax_pha)
    cb_pha.set_ticks(pha_ticks)
    cb_pha.set_ticklabels(pha_labels)

    for cb in (cb_pow, cb_pha):
        cb.outline.set_linewidth(2)
        cb.ax.tick_params(direction='in', width=2, length=4,
                          labelsize=0 if FOR_PUBLICATION else 8)

    style_ax(ax_pow, x_label, y_label)
    style_ax(ax_pha, x_label, '')   # y-label only on left panel

    if not FOR_PUBLICATION:
        ax_pow.set_title('Power  |T|²',     fontsize=10)
        ax_pha.set_title('Phase  ∠T (rad)', fontsize=10)

    os.makedirs(MEDIA_DIR, exist_ok=True)
    out_path = os.path.join(MEDIA_DIR, OUTPUT_FILENAME)
    fig.savefig(out_path, format='svg')
    print(f"Saved: {out_path}")

    # ---- 1D slice at phi = 0 ------------------------------------------------
    probe_1d = np.linspace(*PROBE_RANGE_GHZ, N_POINTS)
    t_1d     = solve(probe_1d - F1_GHZ, probe_1d - F2_GHZ, PHI,
                     gamma_i1, gamma_e1, gamma_i2, gamma_e2)
    power_1d = np.abs(t_1d) ** 2
    phase_1d = (np.angle(t_1d) + PHASE_OFFSET - PHASE_MIN) % (2*np.pi) + PHASE_MIN

    fig1d_w_mm = LEFT_MM + 2 * AXES_WIDTH_MM + PANEL_GAP_MM + RIGHT_MM
    fig1d_h_mm = BOTTOM_MM + AXES_1D_HEIGHT_MM + TOP_MM
    fig1d = plt.figure(figsize=(fig1d_w_mm / 25.4, fig1d_h_mm / 25.4))

    def add_1d_panel(left_mm):
        return fig1d.add_axes([
            left_mm / fig1d_w_mm,
            BOTTOM_MM / fig1d_h_mm,
            AXES_WIDTH_MM / fig1d_w_mm,
            AXES_1D_HEIGHT_MM / fig1d_h_mm,
        ])

    ax1d_pow = add_1d_panel(LEFT_MM)
    ax1d_pha = add_1d_panel(LEFT_MM + AXES_WIDTH_MM + PANEL_GAP_MM)

    ax1d_pow.plot(probe_1d, power_1d, color='k', linewidth=1.5)
    ax1d_pha.plot(probe_1d, phase_1d, color='k', linewidth=1.5)

    ax1d_pow.set_ylim(0.0, 1.0)
    ax1d_pha.set_ylim(PHASE_MIN, PHASE_MIN + 2*np.pi)
    ax1d_pha.set_yticks(pha_ticks)

    style_ax(ax1d_pow, 'f_probe (GHz)', '|T|²')
    style_ax(ax1d_pha, 'f_probe (GHz)', '∠T (rad)')

    if not FOR_PUBLICATION:
        phi_str = _pi_label(PHI) if PHI != 0 else '0'
        ax1d_pow.set_title(f'Power  (φ = {phi_str})',  fontsize=10)
        ax1d_pha.set_title(f'Phase  (φ = {phi_str})',  fontsize=10)
        ax1d_pha.set_yticklabels(pha_labels, fontsize=8)

    out_path_1d = os.path.join(MEDIA_DIR, OUTPUT_FILENAME_1D)
    fig1d.savefig(out_path_1d, format='svg')
    print(f"Saved: {out_path_1d}")

    plt.show()


if __name__ == '__main__':
    main()
