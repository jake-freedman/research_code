import numpy as np
from scipy.special import jv
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import os

# ============================================================
# USER PARAMETERS
# ============================================================

# Fixed inputs
F_PROBE_GHZ = 400e3   # Carrier/probe frequency (GHz)
F_MOD_GHZ   = 2.5     # Modulation frequency (GHz)
HARMONIC_M  = 4       # Target intensity harmonic to maximize
N_SIDEBAND  = 20      # PM sidebands included (±N_SIDEBAND)

# Ring Q factors (F1, F2 are optimization variables; Q factors are held fixed)
Q_I1 = 1e6
Q_E1 = 1e6
Q_I2 = 1e6
Q_E2 = 1e6

# Search bounds for the four free parameters
PHI_BOUNDS  = (0.0, 2*np.pi)   # inter-ring phase (rad)
BETA_BOUNDS = (0.1, 5.0)       # PM depth (rad)
# Ring resonance frequency search range: ±HALFRANGE around F_PROBE_GHZ (GHz)
# None → automatically N_SIDEBAND * F_MOD_GHZ
F1_SEARCH_HALFRANGE = None
F2_SEARCH_HALFRANGE = None

# Optimization method
# 'differential_evolution' — global, robust, can be parallelized
# 'multistart'             — many Nelder-Mead restarts, faster per eval
METHOD = 'differential_evolution'

# Differential evolution settings
DE_POPSIZE = 15      # population multiplier (total members = DE_POPSIZE * 4)
DE_MAXITER = 1000
DE_TOL     = 1e-7
DE_SEED    = 42
DE_WORKERS = 1       # set to -1 to use all CPU cores

# Multistart Nelder-Mead settings
MS_N_STARTS = 200
MS_SEED     = 42

# Diagnostic plots
N_SHOW_DIAG      = 5     # sidebands shown in comb bar charts
PROBE_DIAG_RANGE = None  # ±GHz around F_PROBE_GHZ; None → (N_SHOW_DIAG+0.5)*F_MOD_GHZ

# Figure dimensions (mm)
AXES_WIDTH_MM  = 65.0
AXES_HEIGHT_MM = 40.0
PANEL_GAP_MM   = 10.0
CBAR_GAP_MM    = 2.0
CBAR_WIDTH_MM  = 4.0
LEFT_MM   = 20.0
RIGHT_MM  = 8.0
BOTTOM_MM = 15.0
TOP_MM    = 8.0

FOR_PUBLICATION = False
PHASE_MIN       = -np.pi
PHASE_OFFSET    = 0

MEDIA_DIR     = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media"
OUTPUT_PREFIX = "coupled_ring_opt"

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
# Coupled-ring physics
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

def compute_am_single(phi, beta, f1, f2):
    """Return (|I_m|, I_0) at a single parameter point."""
    gamma_i1 = f1 / Q_I1;  gamma_e1 = f1 / Q_E1
    gamma_i2 = f2 / Q_I2;  gamma_e2 = f2 / Q_E2
    n_arr = np.arange(-N_SIDEBAND, N_SIDEBAND + 1)
    Jn    = jv(n_arr, beta)
    D1    = F_PROBE_GHZ + n_arr * F_MOD_GHZ - f1
    D2    = F_PROBE_GHZ + n_arr * F_MOD_GHZ - f2
    c     = solve(D1, D2, phi, gamma_i1, gamma_e1, gamma_i2, gamma_e2) * Jn
    num_n = len(n_arr)
    I_0   = float(np.sum(np.abs(c)**2))
    I_m   = float(2.0 * np.abs(np.sum(c[HARMONIC_M:] * np.conj(c[:num_n - HARMONIC_M]))))
    return I_m, I_0


# Module-level objective so DE can pickle it for parallel workers
def objective(params):
    phi, beta, f1, f2 = params
    I_m, _ = compute_am_single(phi, beta, f1, f2)
    return -I_m


# ============================================================
# Optimization
# ============================================================

def run_optimization(bounds):
    if METHOD == 'differential_evolution':
        print(f"Running differential_evolution  "
              f"(popsize={DE_POPSIZE}, maxiter={DE_MAXITER}, workers={DE_WORKERS})...")
        result = differential_evolution(
            objective, bounds,
            seed=DE_SEED, popsize=DE_POPSIZE,
            maxiter=DE_MAXITER, tol=DE_TOL,
            workers=DE_WORKERS, polish=True, disp=True,
        )
    elif METHOD == 'multistart':
        print(f"Running multistart Nelder-Mead  ({MS_N_STARTS} starts)...")
        rng = np.random.default_rng(MS_SEED)
        best_val, best_result = np.inf, None
        for i in range(MS_N_STARTS):
            x0  = [rng.uniform(*b) for b in bounds]
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000})
            if res.fun < best_val:
                best_val   = res.fun
                best_result = res
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{MS_N_STARTS}  best |I_{HARMONIC_M}| = {-best_val:.5f}")
        result = best_result
    else:
        raise ValueError(f"Unknown METHOD: {METHOD!r}")
    return result


# ============================================================
# Diagnostic plots
# ============================================================

def _save(fig, filename):
    path = os.path.join(MEDIA_DIR, filename)
    fig.savefig(path, format='svg')
    print(f"Saved: {path}")


def plot_diagnostics(phi_opt, beta_opt, f1_opt, f2_opt):
    """
    Five diagnostic figures at the optimal parameters:
      1.  |T|²  vs detuning from carrier
      2.  Phase  vs detuning from carrier
      3.  Input PM comb (bar chart, phase-coloured)
      4.  Output comb  (bar chart, phase-coloured)
      5.  |I_m| and I_0 vs probe detuning (1D sweep)
    """
    gi1 = f1_opt / Q_I1;  ge1 = f1_opt / Q_E1
    gi2 = f2_opt / Q_I2;  ge2 = f2_opt / Q_E2

    half  = PROBE_DIAG_RANGE if PROBE_DIAG_RANGE is not None \
            else (N_SHOW_DIAG + 0.5) * F_MOD_GHZ
    det   = np.linspace(-half, half, 3000)
    f_arr = F_PROBE_GHZ + det

    t_arr  = solve(f_arr - f1_opt, f_arr - f2_opt, phi_opt, gi1, ge1, gi2, ge2)
    T_arr  = np.abs(t_arr)**2
    ph_arr = _wrap_phase(np.angle(t_arr))

    pha_ticks, pha_labels = _phase_ticks(PHASE_MIN)

    mm   = 1 / 25.4
    ax_w = AXES_WIDTH_MM  * mm
    ax_h = AXES_HEIGHT_MM * mm
    lft  = LEFT_MM   * mm
    bot  = BOTTOM_MM * mm
    rgt  = RIGHT_MM  * mm
    top  = TOP_MM    * mm

    def _line_fig():
        fw = lft + ax_w + rgt
        fh = bot + ax_h + top
        f  = plt.figure(figsize=(fw, fh))
        a  = f.add_axes([lft/fw, bot/fh, ax_w/fw, ax_h/fh])
        return f, a

    def _style(ax, ylabel):
        for sp in ax.spines.values():
            sp.set_linewidth(2)
        ax.tick_params(direction='in', width=2, length=4, labelsize=8,
                       top=True, right=True)
        if not FOR_PUBLICATION:
            ax.set_xlabel('Δf_carrier (GHz)', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
        else:
            ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_xlim(-half, half)
        for n in range(-N_SHOW_DIAG, N_SHOW_DIAG + 1):
            ax.axvline(n * F_MOD_GHZ, color='gray', lw=0.5, ls=':', alpha=0.6)

    # 1. |T|²
    fig1, ax1 = _line_fig()
    ax1.plot(det, T_arr, color='#2522D4', lw=1.5)
    ax1.set_ylim(-0.05, 1.1)
    _style(ax1, '|T|²')
    _save(fig1, f'{OUTPUT_PREFIX}_T.svg')

    # 2. Phase
    fig2, ax2 = _line_fig()
    ax2.plot(det, ph_arr, color='#2522D4', lw=1.5)
    ax2.set_ylim(PHASE_MIN, PHASE_MIN + 2*np.pi)
    ax2.set_yticks(pha_ticks)
    if not FOR_PUBLICATION:
        ax2.set_yticklabels(pha_labels, fontsize=8)
    _style(ax2, '∠T (rad)')
    _save(fig2, f'{OUTPUT_PREFIX}_phase.svg')

    # 3 & 4. Comb bar charts
    n_show     = np.arange(-N_SHOW_DIAG, N_SHOW_DIAG + 1)
    Jn_show    = jv(n_show, beta_opt)
    in_db      = 10 * np.log10(np.maximum(Jn_show**2, 1e-20))
    in_phases  = _wrap_phase(np.where(Jn_show >= 0, 0.0, np.pi))

    c_n        = solve(F_PROBE_GHZ + n_show*F_MOD_GHZ - f1_opt,
                       F_PROBE_GHZ + n_show*F_MOD_GHZ - f2_opt,
                       phi_opt, gi1, ge1, gi2, ge2) * Jn_show
    out_db     = 10 * np.log10(np.maximum(np.abs(c_n)**2, 1e-20))
    out_phases = _wrap_phase(np.angle(c_n))

    cmap_comb = plt.cm.twilight
    norm_comb = plt.Normalize(vmin=PHASE_MIN, vmax=PHASE_MIN + 2*np.pi)
    freq_pos  = n_show * F_MOD_GHZ
    bar_w     = 0.6 * F_MOD_GHZ

    cbar_gap = CBAR_GAP_MM * mm
    cbar_w   = CBAR_WIDTH_MM * mm
    cbar_rm  = (RIGHT_MM + CBAR_GAP_MM + CBAR_WIDTH_MM) * mm

    # Shared y-limits across both comb figures
    valid_in  = in_db[in_db   > -200];  valid_out = out_db[out_db > -200]
    shared_floor = min(max(in_db.max()  - 60, valid_in.min()),
                       max(out_db.max() - 60, valid_out.min()))
    shared_top   = max(in_db.max(), out_db.max()) + 3

    def _comb_fig():
        fw = lft + ax_w + cbar_gap + cbar_w + cbar_rm
        fh = bot + ax_h + top
        f  = plt.figure(figsize=(fw, fh))
        a  = f.add_axes([lft/fw, bot/fh, ax_w/fw, ax_h/fh])
        ca = f.add_axes([(lft+ax_w+cbar_gap)/fw, bot/fh, cbar_w/fw, ax_h/fh])
        return f, a, ca

    def _draw_comb(ax, cax, db, phases, fname):
        ax.bar(freq_pos, np.maximum(db, shared_floor) - shared_floor,
               width=bar_w, color=cmap_comb(norm_comb(phases)),
               align='center', bottom=shared_floor)
        ax.set_ylim(shared_floor, shared_top)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        _style(ax, 'P_n (dB)')
        sm = plt.cm.ScalarMappable(cmap=cmap_comb, norm=norm_comb)
        sm.set_array([])
        cb = ax.get_figure().colorbar(sm, cax=cax)
        cb.set_ticks(pha_ticks);  cb.set_ticklabels(pha_labels)
        cb.ax.tick_params(labelsize=8);  cb.outline.set_linewidth(2)
        _save(ax.get_figure(), fname)

    fig3, ax3, cax3 = _comb_fig()
    _draw_comb(ax3, cax3, in_db,  in_phases,  f'{OUTPUT_PREFIX}_comb_in.svg')
    fig4, ax4, cax4 = _comb_fig()
    _draw_comb(ax4, cax4, out_db, out_phases, f'{OUTPUT_PREFIX}_comb_out.svg')

    # 5. |I_m| and AM phase φ_m vs inter-ring phase φ  (F1, F2, beta held at optimum)
    phi_sweep = np.linspace(0.0, 2*np.pi, 500)
    n_arr_all = np.arange(-N_SIDEBAND, N_SIDEBAND + 1)
    Jn_all    = jv(n_arr_all, beta_opt)
    num_n     = len(n_arr_all)
    c_phi     = np.zeros((len(phi_sweep), num_n), dtype=complex)
    for idx, n in enumerate(n_arr_all):
        D1_n = F_PROBE_GHZ + n * F_MOD_GHZ - f1_opt   # scalar
        D2_n = F_PROBE_GHZ + n * F_MOD_GHZ - f2_opt   # scalar
        c_phi[:, idx] = solve(D1_n, D2_n, phi_sweep,
                              gi1, ge1, gi2, ge2) * Jn_all[idx]
    I_m_complex_phi = np.sum(
        c_phi[:, HARMONIC_M:] * np.conj(c_phi[:, :num_n - HARMONIC_M]), axis=-1)
    I_m_phi   = 2.0 * np.abs(I_m_complex_phi)
    phi_m_phi = _wrap_phase(np.angle(I_m_complex_phi))

    gap    = LEFT_MM * mm   # wide enough to fit ax5b's ylabel + tick labels
    fig5_w = lft + 2*ax_w + gap + rgt
    fig5_h = bot + ax_h + top
    fig5   = plt.figure(figsize=(fig5_w, fig5_h))
    ax5a   = fig5.add_axes([lft/fig5_w,            bot/fig5_h, ax_w/fig5_w, ax_h/fig5_h])
    ax5b   = fig5.add_axes([(lft+ax_w+gap)/fig5_w, bot/fig5_h, ax_w/fig5_w, ax_h/fig5_h])

    ax5a.plot(phi_sweep, I_m_phi, color='k', lw=1.5)
    ax5a.axvline(phi_opt, color='#e5a3a3', lw=1.2, ls='--')
    ax5a.set_xlim(0, 2*np.pi)
    ax5a.set_ylim(0, 1.5)
    ax5a.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])

    ax5b.plot(phi_sweep, phi_m_phi, color='k', lw=1.5)
    ax5b.axvline(phi_opt, color='#e5a3a3', lw=1.2, ls='--')
    ax5b.set_xlim(0, 2*np.pi)
    ax5b.set_ylim(PHASE_MIN, PHASE_MIN + 2*np.pi)
    ax5b.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax5b.set_yticks(pha_ticks)

    for ax5, ylabel in [(ax5a, f'|I_{HARMONIC_M}|'), (ax5b, 'φ_m (rad)')]:
        for sp in ax5.spines.values():
            sp.set_linewidth(2)
        ax5.tick_params(direction='in', width=2, length=4, labelsize=8,
                        top=True, right=True)
        if not FOR_PUBLICATION:
            ax5.set_xlabel('φ (rad)', fontsize=10)
            ax5.set_ylabel(ylabel, fontsize=10)
        else:
            ax5.tick_params(labelbottom=False, labelleft=False)

    ax5a.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'], fontsize=8)
    ax5b.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'], fontsize=8)
    if not FOR_PUBLICATION:
        ax5b.set_yticklabels(pha_labels, fontsize=8)

    _save(fig5, f'{OUTPUT_PREFIX}_phi_sweep.svg')


# ============================================================
# Main
# ============================================================

def main():
    if HARMONIC_M > N_SIDEBAND:
        raise ValueError(f"HARMONIC_M={HARMONIC_M} > N_SIDEBAND={N_SIDEBAND}")

    half1     = F1_SEARCH_HALFRANGE if F1_SEARCH_HALFRANGE is not None else N_SIDEBAND * F_MOD_GHZ
    half2     = F2_SEARCH_HALFRANGE if F2_SEARCH_HALFRANGE is not None else N_SIDEBAND * F_MOD_GHZ
    f1_bounds = (F_PROBE_GHZ - half1, F_PROBE_GHZ + half1)
    f2_bounds = (F_PROBE_GHZ - half2, F_PROBE_GHZ + half2)
    bounds    = [PHI_BOUNDS, BETA_BOUNDS, f1_bounds, f2_bounds]

    print(f"\nMaximizing |I_{HARMONIC_M}|   f_mod={F_MOD_GHZ} GHz  "
          f"f_carrier={F_PROBE_GHZ} GHz")
    print(f"Bounds:  phi {PHI_BOUNDS}  beta {BETA_BOUNDS}")
    print(f"         F1  [{f1_bounds[0]:.1f}, {f1_bounds[1]:.1f}] GHz")
    print(f"         F2  [{f2_bounds[0]:.1f}, {f2_bounds[1]:.1f}] GHz\n")

    result   = run_optimization(bounds)
    phi_opt, beta_opt, f1_opt, f2_opt = result.x
    I_m_opt, _ = compute_am_single(phi_opt, beta_opt, f1_opt, f2_opt)

    print(f"\n{'='*52}")
    print(f"  OPTIMAL   |I_{HARMONIC_M}| = {I_m_opt:.6f}")
    print(f"{'='*52}")
    phi_lbl = _pi_label(phi_opt).encode('ascii', errors='replace').decode('ascii')
    print(f"  phi    = {phi_opt:.6f} rad  ({phi_lbl})")
    print(f"  beta   = {beta_opt:.6f} rad")
    print(f"  F1     = {f1_opt:.4f} GHz  "
          f"(delta = {f1_opt - F_PROBE_GHZ:+.4f} GHz,  "
          f"n = {(f1_opt - F_PROBE_GHZ)/F_MOD_GHZ:+.3f})")
    print(f"  F2     = {f2_opt:.4f} GHz  "
          f"(delta = {f2_opt - F_PROBE_GHZ:+.4f} GHz,  "
          f"n = {(f2_opt - F_PROBE_GHZ)/F_MOD_GHZ:+.3f})")

    os.makedirs(MEDIA_DIR, exist_ok=True)
    txt = os.path.join(MEDIA_DIR, f'{OUTPUT_PREFIX}_result.txt')
    with open(txt, 'w') as fh:
        fh.write(f"|I_{HARMONIC_M}| optimization\n")
        fh.write(f"f_carrier={F_PROBE_GHZ} GHz  f_mod={F_MOD_GHZ} GHz\n")
        fh.write(f"Q_I1={Q_I1:.2e} Q_E1={Q_E1:.2e} Q_I2={Q_I2:.2e} Q_E2={Q_E2:.2e}\n\n")
        fh.write(f"|I_{HARMONIC_M}| = {I_m_opt:.8f}\n")
        fh.write(f"phi      = {phi_opt:.8f} rad\n")
        fh.write(f"beta     = {beta_opt:.8f} rad\n")
        fh.write(f"F1       = {f1_opt:.6f} GHz  "
                 f"(delta={f1_opt-F_PROBE_GHZ:+.6f}  n={( f1_opt-F_PROBE_GHZ)/F_MOD_GHZ:+.4f})\n")
        fh.write(f"F2       = {f2_opt:.6f} GHz  "
                 f"(delta={f2_opt-F_PROBE_GHZ:+.6f}  n={(f2_opt-F_PROBE_GHZ)/F_MOD_GHZ:+.4f})\n")
    print(f"\nResult saved: {txt}")

    plot_diagnostics(phi_opt, beta_opt, f1_opt, f2_opt)
    plt.show()


if __name__ == '__main__':
    main()
