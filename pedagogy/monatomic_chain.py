"""
Classical 1D monatomic chain: identical masses m connected by springs of
constant kappa (fixed = 1) with lattice spacing a.

Dispersion:      omega(k) = 2*sqrt(kappa/m)*|sin(k*a/2)|, periodic in k with
                  period 2*pi/a; first Brillouin zone k in [-pi/a, pi/a].
Group velocity:   v_g = a*sqrt(kappa/m)*cos(k*a/2)*sign(sin(k*a/2))
Sound speed:      v_s = a*sqrt(kappa/m)
Reduced k:        k_red = k - (2*pi/a)*round(k*a/(2*pi))

Atom n sits at equilibrium x_n = n*a and is displaced according to
u_n(t) = A*cos(k*x_n - phi(t)), with an accumulated phase phi advanced each
animation frame by omega*dt so that slider changes never cause phase jumps.
A view-mode toggle lets u_n be drawn transversely (vertical offset, easier
to read) or longitudinally (real horizontal displacement along the chain).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

KAPPA = 1.0
N_ATOMS = 24
AMP_FRAC = 0.2       # displacement amplitude as a fraction of a
DT = 0.05            # phase-advance timestep per frame
INTERVAL_MS = 30      # animation frame interval

A_DEFAULT = 1.0
K_DEFAULT = 2.0
M_DEFAULT = 1.0


class State:
    def __init__(self):
        self.a = A_DEFAULT
        self.k = K_DEFAULT
        self.m = M_DEFAULT
        self.phi = 0.0
        self.playing = True
        self.mode = 'transverse'  # or 'longitudinal'

    def omega_of(self, k, a=None, m=None):
        a = self.a if a is None else a
        m = self.m if m is None else m
        return 2.0 * np.sqrt(KAPPA / m) * np.abs(np.sin(k * a / 2.0))

    def omega(self):
        return self.omega_of(self.k)

    def group_velocity(self):
        return (self.a * np.sqrt(KAPPA / self.m)
                * np.cos(self.k * self.a / 2.0)
                * np.sign(np.sin(self.k * self.a / 2.0)))

    def sound_speed(self):
        return self.a * np.sqrt(KAPPA / self.m)

    def k_reduced(self):
        return self.k - (2 * np.pi / self.a) * np.round(self.k * self.a / (2 * np.pi))


state = State()

# ------------------------------------------------------------------ layout --
fig = plt.figure(figsize=(9.5, 10))
ax_disp = fig.add_axes([0.10, 0.58, 0.85, 0.37])
ax_chain = fig.add_axes([0.10, 0.34, 0.85, 0.16])

ax_a = fig.add_axes([0.14, 0.26, 0.55, 0.02])
ax_k = fig.add_axes([0.14, 0.21, 0.55, 0.02])
ax_m = fig.add_axes([0.14, 0.16, 0.55, 0.02])

ax_mode = fig.add_axes([0.74, 0.155, 0.20, 0.135])

ax_play = fig.add_axes([0.20, 0.045, 0.22, 0.05])
ax_reset = fig.add_axes([0.48, 0.045, 0.22, 0.05])

# ------------------------------------------------------------- top panel ---
K_RANGE = np.linspace(-3 * np.pi, 3 * np.pi, 2000)

line_disp, = ax_disp.plot([], [], color='#1f6fb2', lw=2, label=r'$\omega(k)$')
line_tangent, = ax_disp.plot([], [], color='0.4', ls='--', lw=1.3,
                              label=r'$\omega = v_s k$ (tangent)')
line_marker, = ax_disp.plot([], [], 'o', color='#d55e00', ms=9, zorder=5,
                             label='current state')
vline_pos = ax_disp.axvline(np.pi, color='k', ls='--', lw=1)
vline_neg = ax_disp.axvline(-np.pi, color='k', ls='--', lw=1)
bz_patch = Rectangle((-np.pi, 0), 2 * np.pi, 1, facecolor='#a6cee3',
                      alpha=0.35, zorder=0)
ax_disp.add_patch(bz_patch)

readout_text = ax_disp.text(
    0.02, 0.96, '', transform=ax_disp.transAxes, va='top', ha='left',
    fontsize=10, family='monospace',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

ax_disp.set_xlim(K_RANGE[0], K_RANGE[-1])
ax_disp.set_xlabel('k')
ax_disp.set_ylabel(r'$\omega(k)$')
ax_disp.set_title('Dispersion relation of the monatomic chain '
                   r'($\kappa = 1$)')
ax_disp.legend(loc='upper right', fontsize=9)

# ------------------------------------------------------------ chain panel --
dots_eq, = ax_chain.plot([], [], 'o', color='0.75', ms=6, zorder=1)
dots_atoms, = ax_chain.plot([], [], 'o', color='#1f6fb2', ms=11, zorder=4)
line_driving, = ax_chain.plot([], [], '--', color='0.55', lw=1.5, zorder=2,
                               label='driving wave (k)')
line_reduced, = ax_chain.plot([], [], '--', color='#009e73', lw=1.8, zorder=3,
                               label='reduced-zone wave ($k_{red}$)')

ax_chain.set_xlabel('equilibrium position x')
ax_chain.set_ylabel('displacement u')
ax_chain.legend(loc='upper right', fontsize=8)

# ----------------------------------------------------------------- widgets --
slider_a = Slider(ax_a, 'a', 0.5, 2.0, valinit=A_DEFAULT)
slider_k = Slider(ax_k, 'k', -3 * np.pi, 3 * np.pi, valinit=K_DEFAULT)
slider_m = Slider(ax_m, 'm', 0.2, 5.0, valinit=M_DEFAULT)

radio_mode = RadioButtons(ax_mode, ('Transverse', 'Longitudinal'), active=0)
ax_mode.set_title('view mode', fontsize=9)

button_play = Button(ax_play, 'Pause')
button_reset = Button(ax_reset, 'Reset')


def on_a_changed(val):
    state.a = val


def on_k_changed(val):
    state.k = val


def on_m_changed(val):
    state.m = val


slider_a.on_changed(on_a_changed)
slider_k.on_changed(on_k_changed)
slider_m.on_changed(on_m_changed)


def on_mode_changed(label):
    state.mode = 'transverse' if label == 'Transverse' else 'longitudinal'


radio_mode.on_clicked(on_mode_changed)


def on_play_clicked(event):
    state.playing = not state.playing
    button_play.label.set_text('Pause' if state.playing else 'Play')


def on_reset_clicked(event):
    state.phi = 0.0
    state.playing = True
    state.mode = 'transverse'
    button_play.label.set_text('Pause')
    radio_mode.set_active(0)
    slider_a.reset()
    slider_k.reset()
    slider_m.reset()


button_play.on_clicked(on_play_clicked)
button_reset.on_clicked(on_reset_clicked)


# ------------------------------------------------------------------- draw --
def redraw():
    a, k, m, phi = state.a, state.k, state.m, state.phi

    omega_k = state.omega()
    v_g = state.group_velocity()
    v_s = state.sound_speed()
    k_red = state.k_reduced()

    # --- top panel: dispersion ---
    omega_curve = state.omega_of(K_RANGE, a=a, m=m)
    line_disp.set_data(K_RANGE, omega_curve)
    line_tangent.set_data(K_RANGE, v_s * K_RANGE)
    line_marker.set_data([k], [omega_k])

    bz_edge = np.pi / a
    vline_pos.set_xdata([bz_edge, bz_edge])
    vline_neg.set_xdata([-bz_edge, -bz_edge])

    omega_max = 2.0 * np.sqrt(KAPPA / m)
    ymin, ymax = -0.35 * omega_max, 1.25 * omega_max
    ax_disp.set_ylim(ymin, ymax)

    bz_patch.set_x(-bz_edge)
    bz_patch.set_width(2 * bz_edge)
    bz_patch.set_y(ymin)
    bz_patch.set_height(ymax - ymin)

    readout_text.set_text(
        f'omega   = {omega_k:6.3f}\n'
        f'v_g     = {v_g:6.3f}\n'
        f'k_red   = {k_red:6.3f}'
    )

    # --- bottom panel: chain ---
    n = np.arange(N_ATOMS)
    x_n = n * a
    amp = AMP_FRAC * a
    u_n = amp * np.cos(k * x_n - phi)

    x_cont = np.linspace(0.0, (N_ATOMS - 1) * a, 600)
    driving_cont = amp * np.cos(k * x_cont - phi)
    reduced_cont = amp * np.cos(k_red * x_cont - phi)

    if state.mode == 'transverse':
        # atoms are displaced vertically by u_n; the real (longitudinal)
        # wave is drawn sideways here purely for visual clarity.
        dots_eq.set_data(x_n, np.zeros_like(x_n))
        dots_atoms.set_data(x_n, u_n)
        line_driving.set_data(x_cont, driving_cont)
        line_reduced.set_data(x_cont, reduced_cont)
        line_driving.set_alpha(1.0)
        line_reduced.set_alpha(1.0)
        ax_chain.set_title(
            'Chain — transverse plotting convention (real motion is '
            'longitudinal, i.e. along the chain axis)', fontsize=10)
    else:
        # atoms actually move along x by u_n; equilibrium row is offset
        # below so both the real lattice and the bunching are visible.
        eq_offset = -0.28 * a
        dots_eq.set_data(x_n, np.full_like(x_n, eq_offset, dtype=float))
        dots_atoms.set_data(x_n + u_n, np.zeros_like(x_n))
        line_driving.set_data(x_cont, driving_cont)
        line_reduced.set_data(x_cont, reduced_cont)
        line_driving.set_alpha(0.4)
        line_reduced.set_alpha(0.4)
        ax_chain.set_title(
            'Chain — longitudinal motion (real physical displacement along '
            'the chain axis; curves show u(x) as a vertical reference only)',
            fontsize=10)

    ax_chain.set_xlim(0.0, (N_ATOMS - 1) * a)
    ax_chain.set_ylim(-0.4 * a, 0.4 * a)


def update(_frame):
    if state.playing:
        state.phi += state.omega() * DT
    redraw()
    return (line_disp, line_tangent, line_marker, vline_pos, vline_neg,
            bz_patch, readout_text, dots_eq, dots_atoms, line_driving,
            line_reduced)


redraw()
anim = FuncAnimation(fig, update, interval=INTERVAL_MS, blit=False,
                      cache_frame_data=False)

plt.show()
