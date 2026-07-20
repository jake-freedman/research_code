"""
Classical 1D diatomic chain: alternating masses m1, m2 connected by identical
springs (kappa = 1, fixed), lattice constant a (one unit cell = one m1 atom
plus one m2 atom).

Dispersion (two branches, omega_minus = acoustic, omega_plus = optical):
    omega_pm(k)^2 = (1/m1 + 1/m2) +- sqrt[(1/m1+1/m2)^2 - 4*sin^2(k*a/2)/(m1*m2)]
periodic in k with period 2*pi/a; first Brillouin zone k in [-pi/a, pi/a].

Mode shapes at the current k are obtained by diagonalizing the Hermitian
mass-weighted dynamical matrix
    D(k) = [[2/m1, -(1+e^{-ika})/sqrt(m1 m2)],
            [-(1+e^{+ika})/sqrt(m1 m2), 2/m2]]
with numpy.linalg.eigh (never the analytic eigenvector formula, which
divides by 1+e^{-ika} and blows up at the zone edge). Each frame the chosen
eigenvector is gauge-fixed (largest component rotated real and positive)
before converting to physical amplitudes (U, V) = (w1/sqrt(m1), w2/sqrt(m2))
and rescaling so max(|U|, |V|) = 0.2*a. An accumulated phase phi advances by
omega_selected*dt every frame so slider/branch changes never cause jumps.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

N_CELLS = 12          # unit cells (24 atoms total)
AMP_FRAC = 0.2         # displacement amplitude as a fraction of a
DT = 0.05              # phase-advance timestep per frame
INTERVAL_MS = 30
SIZE_SCALE = 60.0      # scatter marker area per unit mass
FD_STEP = 1e-4         # central-difference step for v_g
UNFOLD_TOL = 0.02       # |m1-m2| < UNFOLD_TOL * m1 triggers unfolded overlay

A_DEFAULT = 1.0
K_DEFAULT = 2.0
M1_DEFAULT = 1.0
M2_DEFAULT = 2.0

COLOR_M1 = '#1f6fb2'
COLOR_M2 = '#e6550d'
COLOR_ACOUSTIC = '#1f6fb2'
COLOR_OPTICAL = '#7570b3'
COLOR_MARKER = '#d55e00'


class State:
    def __init__(self):
        self.a = A_DEFAULT
        self.k = K_DEFAULT
        self.m1 = M1_DEFAULT
        self.m2 = M2_DEFAULT
        self.phi = 0.0
        self.playing = True
        self.branch = 'acoustic'  # or 'optical'

    # ---- eigenvalues (closed form, safe under sqrt) ----
    def omega2_branches(self, k, a=None, m1=None, m2=None):
        a = self.a if a is None else a
        m1 = self.m1 if m1 is None else m1
        m2 = self.m2 if m2 is None else m2
        s = 1.0 / m1 + 1.0 / m2
        disc = s ** 2 - 4.0 * np.sin(k * a / 2.0) ** 2 / (m1 * m2)
        disc = np.clip(disc, 0.0, None)
        sq = np.sqrt(disc)
        return s - sq, s + sq

    def omega_branches(self, k, a=None, m1=None, m2=None):
        om2_minus, om2_plus = self.omega2_branches(k, a, m1, m2)
        return np.sqrt(om2_minus), np.sqrt(om2_plus)

    def omega_selected(self, k=None, a=None, m1=None, m2=None):
        k = self.k if k is None else k
        om_minus, om_plus = self.omega_branches(k, a, m1, m2)
        return om_minus if self.branch == 'acoustic' else om_plus

    def group_velocity(self):
        return (self.omega_selected(self.k + FD_STEP)
                - self.omega_selected(self.k - FD_STEP)) / (2.0 * FD_STEP)

    def k_reduced(self):
        return self.k - (2 * np.pi / self.a) * np.round(self.k * self.a / (2 * np.pi))

    def gap_bounds(self):
        m_light, m_heavy = min(self.m1, self.m2), max(self.m1, self.m2)
        return np.sqrt(2.0 / m_heavy), np.sqrt(2.0 / m_light)

    def gap_width(self):
        lo, hi = self.gap_bounds()
        return hi - lo

    # ---- eigenvector (numeric, via eigh; no zone-edge division) ----
    def eigen_amplitudes(self):
        k, a, m1, m2 = self.k, self.a, self.m1, self.m2
        off = -(1.0 + np.exp(-1j * k * a)) / np.sqrt(m1 * m2)
        D = np.array([[2.0 / m1, off],
                      [np.conj(off), 2.0 / m2]], dtype=complex)
        _evals, evecs = np.linalg.eigh(D)  # ascending: 0=acoustic, 1=optical
        idx = 0 if self.branch == 'acoustic' else 1
        w = evecs[:, idx]

        comp_idx = int(np.argmax(np.abs(w)))
        phase = w[comp_idx] / np.abs(w[comp_idx])
        w = w / phase

        U = w[0] / np.sqrt(m1)
        V = w[1] / np.sqrt(m2)
        scale = (AMP_FRAC * a) / max(np.abs(U), np.abs(V))
        return U * scale, V * scale


state = State()

# ------------------------------------------------------------------ layout --
fig = plt.figure(figsize=(9.5, 11))
ax_disp = fig.add_axes([0.10, 0.60, 0.85, 0.36])
ax_chain = fig.add_axes([0.10, 0.38, 0.85, 0.16])

ax_a = fig.add_axes([0.14, 0.30, 0.55, 0.02])
ax_k = fig.add_axes([0.14, 0.26, 0.55, 0.02])
ax_m1 = fig.add_axes([0.14, 0.22, 0.55, 0.02])
ax_m2 = fig.add_axes([0.14, 0.18, 0.55, 0.02])

ax_branch = fig.add_axes([0.74, 0.17, 0.20, 0.15])

ax_play = fig.add_axes([0.20, 0.06, 0.22, 0.05])
ax_reset = fig.add_axes([0.48, 0.06, 0.22, 0.05])

# ------------------------------------------------------------- top panel ---
K_RANGE = np.linspace(-3 * np.pi, 3 * np.pi, 2000)

line_acoustic, = ax_disp.plot([], [], color=COLOR_ACOUSTIC, lw=2,
                               label=r'$\omega_-(k)$ acoustic')
line_optical, = ax_disp.plot([], [], color=COLOR_OPTICAL, lw=2,
                              label=r'$\omega_+(k)$ optical')
line_unfolded, = ax_disp.plot([], [], color='0.3', ls=':', lw=1.3,
                               label='unfolded a/2 chain')
line_marker, = ax_disp.plot([], [], 'o', color=COLOR_MARKER, ms=9, zorder=6,
                             label='current state')
vline_pos = ax_disp.axvline(np.pi, color='k', ls='--', lw=1)
vline_neg = ax_disp.axvline(-np.pi, color='k', ls='--', lw=1)
bz_patch = Rectangle((-np.pi, 0), 2 * np.pi, 1, facecolor='#a6cee3',
                      alpha=0.35, zorder=0)
ax_disp.add_patch(bz_patch)
gap_patch = ax_disp.axhspan(0, 0, color='#fdae6b', alpha=0.35, zorder=1)

readout_text = ax_disp.text(
    0.02, 0.96, '', transform=ax_disp.transAxes, va='top', ha='left',
    fontsize=10, family='monospace',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

ax_disp.set_xlim(K_RANGE[0], K_RANGE[-1])
ax_disp.set_xlabel('k')
ax_disp.set_ylabel(r'$\omega(k)$')
ax_disp.set_title(r'Diatomic chain dispersion ($\kappa = 1$)')
ax_disp.legend(loc='upper right', fontsize=8)

# ------------------------------------------------------------ chain panel --
dots_eq, = ax_chain.plot([], [], 'o', color='0.75', ms=5, zorder=1)
scatter_m1 = ax_chain.scatter([], [], color=COLOR_M1, zorder=4, label='m1')
scatter_m2 = ax_chain.scatter([], [], color=COLOR_M2, zorder=4, label='m2')
line_env1, = ax_chain.plot([], [], '--', color=COLOR_M1, lw=1.5, zorder=2,
                            label='envelope $f_1$ (m1)')
line_env2, = ax_chain.plot([], [], '--', color=COLOR_M2, lw=1.5, zorder=3,
                            label='envelope $f_2$ (m2)')

ax_chain.set_xlabel('equilibrium position x')
ax_chain.set_ylabel('displacement')
ax_chain.legend(loc='upper right', fontsize=8)

# ----------------------------------------------------------------- widgets --
slider_a = Slider(ax_a, 'a', 0.5, 2.0, valinit=A_DEFAULT)
slider_k = Slider(ax_k, 'k', -3 * np.pi, 3 * np.pi, valinit=K_DEFAULT)
slider_m1 = Slider(ax_m1, 'm1', 0.2, 5.0, valinit=M1_DEFAULT)
slider_m2 = Slider(ax_m2, 'm2', 0.2, 5.0, valinit=M2_DEFAULT)

radio_branch = RadioButtons(ax_branch, ('Acoustic', 'Optical'), active=0)
ax_branch.set_title('branch', fontsize=9)

button_play = Button(ax_play, 'Pause')
button_reset = Button(ax_reset, 'Reset')


def on_a_changed(val):
    state.a = val


def on_k_changed(val):
    state.k = val


def on_m1_changed(val):
    state.m1 = val


def on_m2_changed(val):
    state.m2 = val


slider_a.on_changed(on_a_changed)
slider_k.on_changed(on_k_changed)
slider_m1.on_changed(on_m1_changed)
slider_m2.on_changed(on_m2_changed)


def on_branch_changed(label):
    state.branch = 'acoustic' if label == 'Acoustic' else 'optical'


radio_branch.on_clicked(on_branch_changed)


def on_play_clicked(event):
    state.playing = not state.playing
    button_play.label.set_text('Pause' if state.playing else 'Play')


def on_reset_clicked(event):
    state.phi = 0.0
    state.playing = True
    state.branch = 'acoustic'
    button_play.label.set_text('Pause')
    radio_branch.set_active(0)
    slider_a.reset()
    slider_k.reset()
    slider_m1.reset()
    slider_m2.reset()


button_play.on_clicked(on_play_clicked)
button_reset.on_clicked(on_reset_clicked)


# ------------------------------------------------------------------- draw --
def redraw():
    a, k, m1, m2, phi = state.a, state.k, state.m1, state.m2, state.phi

    om_minus_arr, om_plus_arr = state.omega_branches(K_RANGE, a=a, m1=m1, m2=m2)
    line_acoustic.set_data(K_RANGE, om_minus_arr)
    line_optical.set_data(K_RANGE, om_plus_arr)

    if abs(m1 - m2) < UNFOLD_TOL * m1:
        m_avg = 0.5 * (m1 + m2)
        unfolded = 2.0 * np.sqrt(1.0 / m_avg) * np.abs(np.sin(K_RANGE * a / 4.0))
        line_unfolded.set_data(K_RANGE, unfolded)
    else:
        line_unfolded.set_data([], [])

    omega_k = state.omega_selected()
    v_g = state.group_velocity()
    k_red = state.k_reduced()
    gap_lo, gap_hi = state.gap_bounds()
    gap_w = gap_hi - gap_lo

    line_marker.set_data([k], [omega_k])

    bz_edge = np.pi / a
    vline_pos.set_xdata([bz_edge, bz_edge])
    vline_neg.set_xdata([-bz_edge, -bz_edge])

    omega_max = float(np.max(om_plus_arr))
    ymin, ymax = -0.08 * omega_max, 1.15 * omega_max
    ax_disp.set_ylim(ymin, ymax)

    bz_patch.set_x(-bz_edge)
    bz_patch.set_width(2 * bz_edge)
    bz_patch.set_y(ymin)
    bz_patch.set_height(ymax - ymin)

    gap_patch.set_y(gap_lo)
    gap_patch.set_height(max(gap_hi - gap_lo, 0.0))

    readout_text.set_text(
        f'omega   = {omega_k:6.3f}\n'
        f'v_g     = {v_g:6.3f}\n'
        f'k_red   = {k_red:6.3f}\n'
        f'gap dw  = {gap_w:6.3f}'
    )

    # --- bottom panel: chain ---
    U, V = state.eigen_amplitudes()
    n = np.arange(N_CELLS)
    x1 = n * a
    x2 = n * a + a / 2.0
    u_n = np.real(U * np.exp(1j * (k * n * a - phi)))
    v_n = np.real(V * np.exp(1j * (k * n * a - phi)))

    x_eq = np.concatenate([x1, x2])
    dots_eq.set_data(x_eq, np.zeros_like(x_eq))

    scatter_m1.set_offsets(np.column_stack([x1, u_n]))
    scatter_m2.set_offsets(np.column_stack([x2, v_n]))
    scatter_m1.set_sizes(np.full(N_CELLS, SIZE_SCALE * m1))
    scatter_m2.set_sizes(np.full(N_CELLS, SIZE_SCALE * m2))

    x_cont = np.linspace(0.0, N_CELLS * a, 900)
    f1 = np.real(U * np.exp(1j * (k * x_cont - phi)))
    f2 = np.real(V * np.exp(1j * (k * (x_cont - a / 2.0) - phi)))
    line_env1.set_data(x_cont, f1)
    line_env2.set_data(x_cont, f2)

    ax_chain.set_xlim(0.0, N_CELLS * a)
    ax_chain.set_ylim(-0.4 * a, 0.4 * a)
    ax_chain.set_title(
        f'Diatomic chain, {state.branch} branch — transverse plotting of '
        'longitudinal displacement', fontsize=10)


def update(_frame):
    if state.playing:
        state.phi += state.omega_selected() * DT
    redraw()
    return (line_acoustic, line_optical, line_unfolded, line_marker,
            vline_pos, vline_neg, bz_patch, gap_patch, readout_text,
            dots_eq, scatter_m1, scatter_m2, line_env1, line_env2)


redraw()
anim = FuncAnimation(fig, update, interval=INTERVAL_MS, blit=False,
                      cache_frame_data=False)

plt.show()
