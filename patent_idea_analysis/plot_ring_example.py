import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ring_resonator import RingResonator, power_transmission, phase_transmission
from graphics import (
    LIGHTBLUE2, GREEN2, VIOLET2, RED2, ORANGE2, PINK2, DARKGREEN2, DARKBLUE2, TAN2
)

from path_utils import local_path

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
RINGS_ROW1.reverse()

# ---------------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------------
F_SPAN    = 40e9     # frequency span [Hz]; None = 10× narrowest linewidth across all rings
N_POINTS  = 2000
AX_W_MM   = 80.0
AX_H_MM   = 46.0
POWER_DB  = False
DB_FLOOR  = -10.0
PHASE_MIN = -2 * np.pi

PUBLISHED_PLOT = False   # if True: strip labels/ticks/legend and save as .svg
SAVE_PATH      = None # local_path(r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media\rings.svg")

# ---------------------------------------------------------------------------

_RING_COLORS = [LIGHTBLUE2, GREEN2, VIOLET2, PINK2, ORANGE2, PINK2, DARKGREEN2, DARKBLUE2]

matplotlib.rcParams["font.family"] = "Arial"
mm_per_inch = 25.4
fp = {"family": "Arial"}
axis_label_fontsize = 10


def _build_rings(specs):
    return [RingResonator.from_Q(f_0=F_0, Q_i=qi, Q_e=qe) for qi, qe in specs]


rings = _build_rings(RINGS_ROW1)

for i, r in enumerate(rings):
    print(f"Ring {i+1}: Q_i={r.Q_i:.2e}  Q_e={r.Q_e:.2e}  "
          f"linewidth={r.linewidth_hz:.3e} Hz  Q_total={r.Q_total:.3e}")

if F_SPAN is None:
    f_span = 10 * min(r.linewidth_hz for r in rings)
else:
    f_span = F_SPAN

f            = np.linspace(F_0 - f_span / 2, F_0 + f_span / 2, N_POINTS)
detuning_ghz = (f - F_0) / 1e9

colors = [_RING_COLORS[i % len(_RING_COLORS)] for i in range(len(rings))]

# ---------------------------------------------------------------------------
# Build figures: one for power, one for phase
# ---------------------------------------------------------------------------
left_in, right_in = 0.60, 0.20
bottom_in, top_in = 0.45, 0.25

ax_w_in = AX_W_MM / mm_per_inch
ax_h_in = AX_H_MM / mm_per_inch
fig_w   = left_in + ax_w_in + right_in
fig_h   = bottom_in + ax_h_in + top_in

def _make_fig():
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([left_in / fig_w, bottom_in / fig_h,
                        ax_w_in / fig_w, ax_h_in  / fig_h])
    return fig, ax

fig_pwr, ax_pwr = _make_fig()
fig_phi, ax_phi = _make_fig()

for ring, color in zip(rings, colors):
    ratio = ring.Q_i / ring.Q_e
    lbl   = f"$Q_i/Q_e$ = {ratio:.0f}"
    pwr   = power_transmission(ring, f)
    phi   = phase_transmission(ring, f)

    if POWER_DB:
        pwr_plot = 10 * np.log10(np.maximum(pwr, 10 ** (DB_FLOOR / 10)))
    else:
        pwr_plot = pwr

    if PHASE_MIN is not None:
        phi = (phi - PHASE_MIN) % (2 * np.pi) + PHASE_MIN

    ax_pwr.plot(detuning_ghz, pwr_plot,    color=color, linewidth=2, label=lbl)
    ax_phi.plot(detuning_ghz, phi / np.pi, color=color, linewidth=2, label=lbl)

# ---------------------------------------------------------------------------
# Style axes
# ---------------------------------------------------------------------------
pwr_ylabel = "Power transmission [dB]" if POWER_DB else r"Power transmission $|t|^2$"
pwr_ylim   = (DB_FLOOR - 1, 1) if POWER_DB else (0.395, 1.05)

ax_pwr.set_ylabel(pwr_ylabel, fontsize=axis_label_fontsize, **fp)
ax_pwr.set_ylim(*pwr_ylim)
ax_pwr.set_xlim(detuning_ghz[0], detuning_ghz[-1])
ax_pwr.set_xlabel(r"Detuning $\Delta f$ [GHz]", fontsize=axis_label_fontsize, **fp)
ax_pwr.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
ax_pwr.legend(fontsize=8, frameon=False)

ax_phi.set_ylabel(r"Phase $\angle t$ [$\pi$]", fontsize=axis_label_fontsize, **fp)
ax_phi.set_xlim(detuning_ghz[0], detuning_ghz[-1])
ax_phi.set_xlabel(r"Detuning $\Delta f$ [GHz]", fontsize=axis_label_fontsize, **fp)
ax_phi.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
ax_phi.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{int(v):d}" if v == int(v) else f"{v:.1f}"
))
if PHASE_MIN is not None:
    ax_phi.set_ylim((PHASE_MIN - 0.3) / np.pi, (PHASE_MIN + 2 * np.pi + 0.3) / np.pi)

for ax in (ax_pwr, ax_phi):
    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_zorder(10)
    ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")

# ---------------------------------------------------------------------------
# Save / published plot
# ---------------------------------------------------------------------------
if PUBLISHED_PLOT:
    for ax in (ax_pwr, ax_phi):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both", direction="in", width=2,
                       labelbottom=False, labelleft=False)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_zorder(100)

    if SAVE_PATH:
        base = SAVE_PATH.rsplit(".", 1)[0] if "." in SAVE_PATH.split("\\")[-1] else SAVE_PATH
        for fig, tag in [(fig_pwr, "pwr"), (fig_phi, "phi")]:
            path = f"{base}_{tag}.svg"
            fig.savefig(path)
            print(f"Saved {path}")
elif SAVE_PATH:
    base = SAVE_PATH.rsplit(".", 1)[0] if "." in SAVE_PATH.split("\\")[-1] else SAVE_PATH
    ext  = SAVE_PATH.rsplit(".", 1)[-1] if "." in SAVE_PATH.split("\\")[-1] else "png"
    for fig, tag in [(fig_pwr, "pwr"), (fig_phi, "phi")]:
        path = f"{base}_{tag}.{ext}"
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")

plt.show()
