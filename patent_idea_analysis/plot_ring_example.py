import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from ring_resonator import RingResonator, power_transmission, phase_transmission

# ---------------------------------------------------------------------------
# Ring parameters — each entry is one ring (Q_I, Q_E)
# ---------------------------------------------------------------------------
F_0 = 193e12   # resonance frequency [Hz] (same for all rings)

RINGS_ROW1 = [
    (1e6, 1e6 / 40),
    (1e6, 1e6 / 20),
    (1e6, 1e6 / 10),
    (1e6, 1e6 / 5),
]

RINGS_ROW2 = [
    (2e6, 2e6 / 40),
    (2e6, 2e6 / 20),
    (2e6, 2e6 / 10),
    (2e6, 2e6 / 5),
]

# ---------------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------------
F_SPAN    = 15e9     # frequency span [Hz]; None = 10× narrowest linewidth across all rings
N_POINTS  = 2000
AX_W_MM   = 70.0
AX_H_MM   = 30.0
V_GAP_IN  = 0.35    # vertical gap between rows [in]
POWER_DB  = False
DB_FLOOR  = -10.0
PHASE_MIN = -2 * np.pi

# ---------------------------------------------------------------------------

LIGHTBLUE2 = '#b2cbf2'
GREEN2      = '#93C572'

matplotlib.rcParams["font.family"] = "Arial"
mm_per_inch = 25.4
fp = {"family": "Arial"}
axis_label_fontsize = 10


def _shade_palette(hex_color: str, n: int):
    """Return n visually distinct colors centred on hex_color.

    Sweeps hue ±0.07, saturation 0.35→1.0, value 1.0→0.50 across the n steps.
    """
    if n == 1:
        return [hex_color]
    r, g, b = mcolors.to_rgb(hex_color)
    h, _, _ = mcolors.rgb_to_hsv((r, g, b))
    hue_half = 0.07
    return [
        mcolors.hsv_to_rgb((
            (h - hue_half + 2 * hue_half * i / (n - 1)) % 1.0,
            0.35 + 0.65 * i / (n - 1),
            1.0  - 0.50 * i / (n - 1),
        ))
        for i in range(n)
    ]


def _build_rings(specs):
    return [RingResonator.from_Q(f_0=F_0, Q_i=qi, Q_e=qe) for qi, qe in specs]


rings1 = _build_rings(RINGS_ROW1)
rings2 = _build_rings(RINGS_ROW2)

for label, rings in [("Row 1", rings1), ("Row 2", rings2)]:
    for i, r in enumerate(rings):
        print(f"{label} Ring {i+1}: Q_i={r.Q_i:.2e}  Q_e={r.Q_e:.2e}  "
              f"linewidth={r.linewidth_hz:.3e} Hz  Q_total={r.Q_total:.3e}")

all_rings = rings1 + rings2
if F_SPAN is None:
    f_span = 10 * min(r.linewidth_hz for r in all_rings)
else:
    f_span = F_SPAN

f            = np.linspace(F_0 - f_span / 2, F_0 + f_span / 2, N_POINTS)
detuning_ghz = (f - F_0) / 1e9

colors1 = _shade_palette(LIGHTBLUE2, len(rings1))
colors2 = _shade_palette(GREEN2,     len(rings2))

# ---------------------------------------------------------------------------
# Build figure: 2 rows × 2 cols (power | phase)
# ---------------------------------------------------------------------------
left_in, mid_in, right_in = 0.60, 0.55, 0.20
bottom_in, top_in          = 0.45, 0.25

ax_w_in = AX_W_MM / mm_per_inch
ax_h_in = AX_H_MM / mm_per_inch
fig_w   = left_in + ax_w_in + mid_in + ax_w_in + right_in
fig_h   = bottom_in + 2 * ax_h_in + V_GAP_IN + top_in

fig = plt.figure(figsize=(fig_w, fig_h))

def _add_row_axes(row_idx):
    """row_idx 0 = top, 1 = bottom."""
    y0 = (bottom_in + (1 - row_idx) * (ax_h_in + V_GAP_IN)) / fig_h
    ax_p = fig.add_axes([left_in / fig_w,                          y0,
                         ax_w_in / fig_w, ax_h_in / fig_h])
    ax_ph = fig.add_axes([(left_in + ax_w_in + mid_in) / fig_w,   y0,
                          ax_w_in / fig_w, ax_h_in / fig_h])
    return ax_p, ax_ph

ax_pwr1, ax_phi1 = _add_row_axes(0)
ax_pwr2, ax_phi2 = _add_row_axes(1)

def _plot_row(ax_pwr, ax_phi, rings, colors):
    for ring, color in zip(rings, colors):
        ratio = ring.Q_i / ring.Q_e
        lbl = f"$Q_i/Q_e$ = {ratio:.0f}"
        pwr = power_transmission(ring, f)
        phi = phase_transmission(ring, f)

        if POWER_DB:
            pwr_plot = 10 * np.log10(np.maximum(pwr, 10 ** (DB_FLOOR / 10)))
        else:
            pwr_plot = pwr

        if PHASE_MIN is not None:
            phi = (phi - PHASE_MIN) % (2 * np.pi) + PHASE_MIN

        ax_pwr.plot(detuning_ghz, pwr_plot, color=color, linewidth=1.5, label=lbl)
        ax_phi.plot(detuning_ghz, phi,      color=color, linewidth=1.5, label=lbl)

_plot_row(ax_pwr1, ax_phi1, rings1, colors1)
_plot_row(ax_pwr2, ax_phi2, rings2, colors2)

# ---------------------------------------------------------------------------
# Style axes
# ---------------------------------------------------------------------------
pwr_ylabel = "Power transmission [dB]" if POWER_DB else r"Power transmission $|t|^2$"
pwr_ylim   = (DB_FLOOR - 1, 1) if POWER_DB else (-0.05, 1.05)

for row_idx, (ax_pwr, ax_phi) in enumerate([(ax_pwr1, ax_phi1), (ax_pwr2, ax_phi2)]):
    is_bottom = (row_idx == 1)

    ax_pwr.set_ylabel(pwr_ylabel, fontsize=axis_label_fontsize, **fp)
    ax_pwr.set_ylim(*pwr_ylim)
    ax_pwr.set_xlim(detuning_ghz[0], detuning_ghz[-1])

    ax_phi.set_ylabel(r"Phase $\angle t$ [rad]", fontsize=axis_label_fontsize, **fp)
    ax_phi.set_xlim(detuning_ghz[0], detuning_ghz[-1])
    if PHASE_MIN is not None:
        ax_phi.set_ylim(PHASE_MIN, PHASE_MIN + 2 * np.pi)

    if is_bottom:
        ax_pwr.set_xlabel(r"Detuning $\Delta f$ [GHz]", fontsize=axis_label_fontsize, **fp)
        ax_phi.set_xlabel(r"Detuning $\Delta f$ [GHz]", fontsize=axis_label_fontsize, **fp)

    for ax in (ax_pwr, ax_phi):
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Arial")

ax_pwr1.legend(fontsize=8, frameon=False)
ax_pwr2.legend(fontsize=8, frameon=False)

plt.show()
