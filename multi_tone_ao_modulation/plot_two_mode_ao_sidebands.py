"""
Two-mode acousto-optic sideband simulator.

Two coupled mechanical modes each phase-modulate a shared optical mode:
    mode 1 -> modulation at  Omega  with depth  b1 = (dphi/dx1)*X1
    mode 2 -> modulation at 2*Omega with depth  b2 = (dphi/dx2)*X2

    phi(t) = b1*sin(Omega*t + theta1) + b2*sin(2*Omega*t + theta2)
    E(t)   = E0 * exp(i*wL*t) * exp(i*phi(t))

For each drive force F the mechanical ODEs are integrated to steady state,
amplitudes (X1, X2) and phases (theta1, theta2) are extracted, and sideband
powers are computed via a two-tone Bessel sum.

The n=2 sideband receives contributions from the 2*Omega tone (b2) AND from
the second-order Bessel term J2(b1) — they interfere coherently through the
relative phase (spectral degeneracy).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.special import jv as _jv
from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2, DARKBLUE2,
    TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)

_COMBLINE_COLORS = {
    -3: '#bf7362',
    -2: RED2,
    -1: ORANGE2,
     0: GREEN2,
     1: LIGHTBLUE2,
     2: '#5c70aa',
     3: VIOLET2,
}
_COLORS = [BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2,
           DARKBLUE2, TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2]

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Mode 1 (drives at Omega)
omega1 = 1.0
gamma1 = 0.05
eta1   = 1.0

# Mode 2 (driven at 2*Omega via nonlinearity)
omega2 = 2.0
gamma2 = 0.05
eta2   = 1.0

# Shared mechanical parameters
m      = 1.0
beta   = 0.001     # quadratic nonlinearity strength
Omega  = 1.0      # drive frequency (rad/s)

factor = 10
# Acousto-optic coupling (rad of optical phase per unit displacement)
dphi_dx1 = 0.5 / factor
dphi_dx2 = -1.0 / factor

# Force sweep
F_min   = 0.01
F_max   = 0.4 * factor
n_force = 50
F_scale = 'linear'   # 'log' or 'linear'

# Optical sidebands to track (order n -> sideband at wL + n*Omega)
sideband_orders = [-2, -1, 0, 1, 2]

# Output units:
#   'dBc' -> power relative to carrier (n=0) in dB
#   'eff' -> conversion efficiency in % (|c_n|^2 of unit input)
OUTPUT = 'eff'

# Bessel sum truncation
k_max = 20

# Settle/sample settings
n_settle_cycles = 150
n_sample_cycles = 20

# Integrator tolerances
rtol = 1e-8
atol = 1e-10

# Figure layout (mm)
axes_width_mm   = 120.0
axes_height_mm  =  55.0
axes_height2_mm =  35.0   # modulation-depth panel
left_mm         =  18.0
right_mm        =   8.0
bottom_mm       =  14.0
top_mm          =   8.0
gap_mm          =  12.0   # vertical gap between the two panels

# Save paths (None = don't save)
SAVE_PNG = Path(__file__).parent / 'two_mode_ao_sidebands.png'
SAVE_PNG2 = Path(__file__).parent / 'two_mode_ao_mod_depths.png'

# Numerical curve style
NUM_LINESTYLE = '--'
NUM_LINEWIDTH = 2.0
NUM_ALPHA     = 1.0
NUM_COLOR     = '#000000'   # None = match combline color; e.g. 'black' = use that for all curves
NUM_ZORDER    = 4

# Analytical solution overlay
# Uses steady-state X1 = F/(m*gamma1*omega1), X2 from nonlinear coupling,
# and the Bessel series P_n = [sum_k J_{2k+n}(b1)*J_{-k}(b2)]^2.
ANALYTIC_SHOW      = True
ANALYTIC_K_MAX     = 30          # Bessel series truncation order
ANALYTIC_LINESTYLE = '-'
ANALYTIC_LINEWIDTH = 5.00
ANALYTIC_ALPHA     = 1
ANALYTIC_COLOR     = None       # None = match combline color; e.g. 'black' = use that for all curves
ANALYTIC_ZORDER    = 3

# ─────────────────────────────────────────────
# ODE definition
# ─────────────────────────────────────────────

def two_mode_ode(t, y, omega1, gamma1, eta1,                                                        
                      omega2, gamma2, eta2,
                      m, beta, F, Omega):
    x1, x1dot, x2, x2dot = y
    nonlinear = (beta / m) * (x1 + x2) ** 2
    drive     = (F / m) * np.cos(Omega * t)
    x1ddot = drive * eta1 - gamma1 * x1dot - omega1 ** 2 * x1 - nonlinear
    x2ddot = drive * eta2 - gamma2 * x2dot - omega2 ** 2 * x2 - nonlinear
    return np.array([x1dot, x1ddot, x2dot, x2ddot])


# ─────────────────────────────────────────────
# Mode extraction: amplitude + phase
# ─────────────────────────────────────────────

def extract_amp_phase(t, x, freq):
    T = t[-1] - t[0]
    a = (2 / T) * np.trapezoid(x * np.cos(freq * t), t)
    b = (2 / T) * np.trapezoid(x * np.sin(freq * t), t)
    return np.hypot(a, b), np.arctan2(a, b)


# ─────────────────────────────────────────────
# Two-tone sideband amplitude
# ─────────────────────────────────────────────

def sideband_amplitude(n, b1, b2, theta1, theta2, k_max=20):
    """c_n = sum_q J_q(b2) J_{n-2q}(b1) exp(i[q*theta2 + (n-2q)*theta1])"""
    c = 0.0 + 0.0j
    for q in range(-k_max, k_max + 1):
        p  = n - 2 * q
        c += _jv(q, b2) * _jv(p, b1) * np.exp(1j * (q * theta2 + p * theta1))
    return c


# ─────────────────────────────────────────────
# Analytical sideband power
# ─────────────────────────────────────────────

def analytic_sideband_power(n, F, k_max=30):
    """
    Steady-state sideband power at order n for drive force F.

    Assumes resonant drive (Omega = omega1) and omega2 = 2*omega1:
      b1 = dphi_dx1 * F / (m * gamma1 * omega1)
      b2 = dphi_dx2 * beta * F^2 / (4 * m^3 * gamma1^2 * gamma2 * omega1^3)

    P_n = [sum_{k=-k_max}^{k_max} J_{2k+n}(b1) * J_{-k}(b2)]^2
    """
    b1 = dphi_dx1 * F / (m * gamma1 * omega1)
    b2 = dphi_dx2 * beta * F ** 2 / (4 * m ** 3 * gamma1 ** 2 * gamma2 * omega1 ** 3)
    c = sum(_jv(2 * k + n, b1) * _jv(-k, b2) for k in range(-k_max, k_max + 1))
    return float(c ** 2)


# ─────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────

def _order_color(order):
    auto_idx = list(_COMBLINE_COLORS.keys()).index(order) if order in _COMBLINE_COLORS else (order + 10)
    return _COMBLINE_COLORS.get(int(order), _COLORS[auto_idx % len(_COLORS)])


def _make_fig_ax(h_mm):
    fig_w = left_mm + axes_width_mm + right_mm
    fig_h = bottom_mm + h_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm   / fig_h,
    )
    return fig, ax


def _style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)


# ─────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────

def run_sideband_sweep():
    if F_scale == 'log':
        F_vals = np.logspace(np.log10(F_min), np.log10(F_max), n_force)
    else:
        F_vals = np.linspace(F_min, F_max, n_force)

    X1_arr  = np.zeros(n_force)
    X2_arr  = np.zeros(n_force)
    th1_arr = np.zeros(n_force)
    th2_arr = np.zeros(n_force)
    b1_arr  = np.zeros(n_force)
    b2_arr  = np.zeros(n_force)
    sb_power = {n: np.zeros(n_force) for n in sideband_orders}

    T        = 2 * np.pi / Omega
    t_settle = n_settle_cycles * T
    t_total  = t_settle + n_sample_cycles * T

    print(f"Sideband sweep: {n_force} force points ({F_scale} scale)")
    for i, F_val in enumerate(F_vals):
        args = (omega1, gamma1, eta1, omega2, gamma2, eta2,
                m, beta, F_val, Omega)
        try:
            sol = solve_ivp(
                two_mode_ode, [0, t_total], [0, 0, 0, 0],
                args=args, method='RK45', dense_output=False,
                rtol=rtol, atol=atol,
            )
            if not sol.success:
                raise RuntimeError(sol.message)
        except Exception as e:
            print(f"  F={F_val:.4f} failed: {e}")
            for n in sideband_orders:
                sb_power[n][i] = np.nan
            continue

        mask = sol.t >= t_settle
        t_ss  = sol.t[mask]
        x1_ss = sol.y[0][mask]
        x2_ss = sol.y[2][mask]

        X1, th1 = extract_amp_phase(t_ss, x1_ss, Omega)
        X2, th2 = extract_amp_phase(t_ss, x2_ss, 2 * Omega)

        b1 = dphi_dx1 * X1
        b2 = dphi_dx2 * X2

        X1_arr[i], X2_arr[i]   = X1, X2
        th1_arr[i], th2_arr[i] = th1, th2
        b1_arr[i], b2_arr[i]   = b1, b2

        for n in sideband_orders:
            c = sideband_amplitude(n, b1, b2, th1, th2, k_max=k_max)
            sb_power[n][i] = np.abs(c) ** 2

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_force} done")
    print("Sweep complete.")

    # Convert to output units
    if OUTPUT == 'dBc':
        carrier  = sb_power[0]
        plot_data = {
            n: 10 * np.log10(sb_power[n] / (carrier + 1e-30) + 1e-30)
            for n in sideband_orders
        }
        ylabel = r'Sideband power [dBc]'
    else:
        plot_data = {n: 100 * sb_power[n] for n in sideband_orders}
        ylabel = r'Conversion efficiency [%]'

    # ── Figure 1: sideband powers ────────────────────────────────────────────
    fig1, ax1 = _make_fig_ax(axes_height_mm)

    for n in sideband_orders:
        lbl = f'n={n:+d}' if n != 0 else 'n=0 (carrier)'
        ax1.plot(F_vals, plot_data[n],
                 color=NUM_COLOR if NUM_COLOR is not None else _order_color(n),
                 linewidth=NUM_LINEWIDTH, linestyle=NUM_LINESTYLE, alpha=NUM_ALPHA,
                 zorder=NUM_ZORDER, label=lbl)

    if ANALYTIC_SHOW:
        for n in sideband_orders:
            a_power = np.array([analytic_sideband_power(n, F, k_max=ANALYTIC_K_MAX)
                                for F in F_vals])
            if OUTPUT == 'dBc':
                a_carrier = np.array([analytic_sideband_power(0, F, k_max=ANALYTIC_K_MAX)
                                      for F in F_vals])
                a_plot = 10 * np.log10(a_power / (a_carrier + 1e-30) + 1e-30)
            else:
                a_plot = 100 * a_power
            ax1.plot(F_vals, a_plot,
                     color=ANALYTIC_COLOR if ANALYTIC_COLOR is not None else _order_color(n),
                     linewidth=ANALYTIC_LINEWIDTH, linestyle=ANALYTIC_LINESTYLE,
                     alpha=ANALYTIC_ALPHA, zorder=ANALYTIC_ZORDER)

    if F_scale == 'log':
        ax1.set_xscale('log')
    ax1.set_xlabel(r'Drive amplitude $F$', fontsize=axis_label_fontsize)
    ax1.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax1.legend(fontsize=tick_label_fontsize, frameon=False)
    _style_ax(ax1)

    if SAVE_PNG:
        fig1.savefig(SAVE_PNG, dpi=200, bbox_inches='tight')
        print(f"Saved: {SAVE_PNG}")

    # ── Figure 2: modulation depths ─────────────────────────────────────────
    fig2, ax2 = _make_fig_ax(axes_height2_mm)

    ax2.plot(F_vals, b1_arr,
             color=NUM_COLOR if NUM_COLOR is not None else LIGHTBLUE2,
             linewidth=NUM_LINEWIDTH, linestyle=NUM_LINESTYLE, alpha=NUM_ALPHA,
             zorder=NUM_ZORDER,
             label=r'$b_1 = (\mathrm{d}\phi/\mathrm{d}x_1)\,X_1$  ($\Omega$ tone)')
    ax2.plot(F_vals, b2_arr,
             color=NUM_COLOR if NUM_COLOR is not None else ORANGE2,
             linewidth=NUM_LINEWIDTH, linestyle=NUM_LINESTYLE, alpha=NUM_ALPHA,
             zorder=NUM_ZORDER,
             label=r'$b_2 = (\mathrm{d}\phi/\mathrm{d}x_2)\,X_2$  ($2\Omega$ tone)')

    if ANALYTIC_SHOW:
        b1_analytic = dphi_dx1 * F_vals / (m * gamma1 * omega1)
        b2_analytic = dphi_dx2 * beta * F_vals**2 / (4 * m**3 * gamma1**2 * gamma2 * omega1**3)
        ax2.plot(F_vals, b1_analytic,
                 color=ANALYTIC_COLOR if ANALYTIC_COLOR is not None else LIGHTBLUE2,
                 linewidth=ANALYTIC_LINEWIDTH, linestyle=ANALYTIC_LINESTYLE,
                 alpha=ANALYTIC_ALPHA, zorder=ANALYTIC_ZORDER)
        ax2.plot(F_vals, b2_analytic,
                 color=ANALYTIC_COLOR if ANALYTIC_COLOR is not None else ORANGE2,
                 linewidth=ANALYTIC_LINEWIDTH, linestyle=ANALYTIC_LINESTYLE,
                 alpha=ANALYTIC_ALPHA, zorder=ANALYTIC_ZORDER)

    if F_scale == 'log':
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    ax2.set_xlabel(r'Drive amplitude $F$', fontsize=axis_label_fontsize)
    ax2.set_ylabel(r'Modulation depth [rad]', fontsize=axis_label_fontsize)
    ax2.legend(fontsize=tick_label_fontsize, frameon=False)
    _style_ax(ax2)

    if SAVE_PNG2:
        fig2.savefig(SAVE_PNG2, dpi=200, bbox_inches='tight')
        print(f"Saved: {SAVE_PNG2}")

    # Console summary at max force
    print(f"\nAt max force F = {F_vals[-1]:.3f}:")
    print(f"  X1={X1_arr[-1]:.4f},  theta1={th1_arr[-1]:+.3f} rad,  b1={b1_arr[-1]:.4f}")
    print(f"  X2={X2_arr[-1]:.4f},  theta2={th2_arr[-1]:+.3f} rad,  b2={b2_arr[-1]:.4f}")
    unit = 'dBc' if OUTPUT == 'dBc' else '%'
    for n in sideband_orders:
        print(f"  n={n:+d}: {plot_data[n][-1]:.3f} {unit}")

    plt.show()


if __name__ == '__main__':
    run_sideband_sweep()
