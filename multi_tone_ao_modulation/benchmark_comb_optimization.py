"""
Benchmark multi-tone comb optimizer against random target combs.

Generates N_TRIALS random power distributions over harmonic orders in
[ORDER_MIN, ORDER_MAX] by drawing from a half-normal distribution and
normalizing so total power = 1.  Each target is passed to the K-tone
single-PM optimizer, and the achieved vs target power is plotted on a
square scatter plot with a dashed y = x reference line.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from datetime import datetime
from scipy.optimize import basinhopping
from scipy.special import jv
from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2, DARKBLUE2,
    TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2,
    spine_linewidth, tick_width, tick_direction,
    axis_label_fontsize, tick_label_fontsize,
)
import matplotlib.colors as _mc

_COMBLINE_COLORS = {
    -4: '#452424',
    -3: '#bf7362',
    -2: RED2,
    -1: ORANGE2,
     0: GREEN2,
     1: LIGHTBLUE2,
     2: '#5c70aa',
     3: VIOLET2,
     4: '#3c148b'
}

nblue1 = '#ccccff'
nblue2 = '#7f7fff'
nblue3 = '#3232ff'

# _COMBLINE_COLORS = {n: nblue1 for n in range(-4, 5)}

_COLORS = [BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2,
           DARKBLUE2, TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Random comb settings
N_TRIALS   = 10    # number of random target combs to optimize
ORDER_MIN  = -3    # lowest harmonic order included in each random comb
ORDER_MAX  =  3    # highest harmonic order included in each random comb
SEED_COMBS = 0     # seed for random comb generation (None = random)

# Optimizer settings (same meaning as optimize_multitone_comb.py)
BETA_MAX = 5
N_MAX    = 10
N_TONES  = 4
N_ITER   = 200
SEED_OPT = 42      # base optimizer seed; trial k uses SEED_OPT + k
METHOD   = "fft"
N_FFT    = 8192

# Save / load results
# Set SAVE_DIR to a folder to save a timestamped .npz after optimization.
# Set LOAD_FROM to a specific .npz file to skip optimization and regenerate the plot.
SAVE_DIR  = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\more_harmonics_theory"
LOAD_FROM = None # r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\more_harmonics_theory\benchmark2_comb_results_20260629_114546.npz"

# Plot settings
PLOT_ORDERS  = list(range(-4, 5))   # orders to show in scatter; None = all orders in [ORDER_MIN, ORDER_MAX]
SCALE        = 'percent'   # 'percent' | 'dB'  — axis units
FLOOR_DBC    = -30.0       # lower dB limit (only used when SCALE='dB')
XLIM = (0, 42)   # (min, max) for x-axis; None = auto  (units match SCALE)
YLIM = (0, 42)   # (min, max) for y-axis; None = auto
MARKER_PT    = 6      # marker diameter in points
MARKER_ALPHA = 0.7    # face alpha
EDGE_WIDTH   = 0.5    # marker edge linewidth
EDGE_COLOR   = 'black'  # marker edge color; 'same' = match fill color

# Publication export — scatter plot
FOR_PUBLICATION = True
PUB_SAVE_FOLDER = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media"
PUB_SVG_NAME    = 'benchmark_comb_optimization2.svg'

# Publication export — per-trial comb figures
PUB_PLOT_COMBS      = True         # plot each trial as a separate comb spectrum figure
COMB_SCALE          = 'dB'         # 'percent' | 'dB'
COMB_FLOOR_DBC      = -20.0        # lower dB limit (only when COMB_SCALE='dB')
COMB_FLOOR_PCT      = 0.0          # lower % limit (only when COMB_SCALE='percent')
COMB_CEILING        = 0            # upper limit; None = auto
COMB_AXES_W_MM      = 50.0
COMB_AXES_H_MM      = 9.0
COMB_LEFT_MM        = 14.0
COMB_RIGHT_MM       = 6.0
COMB_BOTTOM_MM      = 12.0
COMB_TOP_MM         = 6.0
COMB_STEM_WIDTH     = 1.5          # stem linewidth
COMB_BALL_PT        = 4            # ball marker diameter in points
COMB_SVG_FOLDER     = PUB_SAVE_FOLDER
COMB_SVG_PREFIX     = 'trial_comb' # files saved as <prefix>_<trial>.svg
# Actual-value overlay — one set of open circles per file in COMB_ACTUAL_FILES.
# Target comb (stems + balls) always comes from LOAD_FROM / the optimization run.
# COMB_ACTUAL_FILES: list of .npz paths whose *actual* values to overlay.
#   None → use the same file as LOAD_FROM (single overlay).
COMB_ACTUAL_SHOW    = True
COMB_ACTUAL_FILES   = [r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\more_harmonics_theory\benchmark2_comb_results_20260629_114546.npz",
                       r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\more_harmonics_theory\benchmark3_comb_results_20260629_114857.npz",
                       r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\more_harmonics_theory\benchmark4_comb_results_20260629_115258.npz"]         # e.g. [file_a, file_b]; None = [LOAD_FROM]
COMB_ACTUAL_COLOR   = [nblue1, nblue2, nblue3]     # list of colors, one per file; 'same' = match combline
COMB_ACTUAL_ALPHA   = 0.4
COMB_ACTUAL_PT      = 6           # diameter of overlay circle markers
COMB_ACTUAL_EW      = 0.5         # stroke width of overlay circle markers

# Figure layout (mm) — square axes
axes_size_mm = 40.0
left_mm      = 18.0
right_mm     =  8.0
bottom_mm    = 14.0
top_mm       =  8.0

# ---------------------------------------------------------------------------
# Spectrum computation
# ---------------------------------------------------------------------------

def _compute_fft(betas, thetas):
    betas  = np.asarray(betas,  dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    t       = np.linspace(0, 2 * np.pi, N_FFT, endpoint=False)
    phase_t = np.zeros(N_FFT)
    for k_idx, (b, th) in enumerate(zip(betas, thetas)):
        phase_t += b * np.sin((k_idx + 1) * t + th)
    fft_raw   = np.fft.fft(np.exp(1j * phase_t)) / N_FFT
    harmonics = np.arange(-N_MAX, N_MAX + 1)
    return harmonics, fft_raw[harmonics % N_FFT]


def _compute_conv(betas, thetas):
    betas  = np.asarray(betas,  dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    n_range    = np.arange(-N_MAX, N_MAX + 1)
    harmonics  = np.array([0])
    amplitudes = np.array([1.0 + 0j])
    for k_idx, (beta, theta) in enumerate(zip(betas, thetas)):
        k = k_idx + 1
        bessel_coeffs = jv(n_range, beta) * np.exp(1j * n_range * theta)
        tone_size = 2 * k * N_MAX + 1
        tone_amps = np.zeros(tone_size, dtype=complex)
        for n_idx, n in enumerate(n_range):
            tone_amps[k * int(n) + k * N_MAX] += bessel_coeffs[n_idx]
        amplitudes = np.convolve(amplitudes, tone_amps)
        harmonics  = np.arange(harmonics[0] - k * N_MAX,
                               harmonics[-1] + k * N_MAX + 1)
    return harmonics, amplitudes


def _compute_spectrum(betas, thetas):
    if METHOD == "fft":
        return _compute_fft(betas, thetas)
    return _compute_conv(betas, thetas)

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _optimize(target_norm: dict, seed: int | None) -> dict:
    """
    Optimize betas/thetas to match target_norm (dict of order → fraction, sum=1).
    Returns {'harmonics', 'amplitudes'} after optimization.
    """
    rng = np.random.default_rng(seed)
    lower = np.concatenate([np.zeros(N_TONES),         np.zeros(N_TONES)])
    upper = np.concatenate([np.full(N_TONES, BETA_MAX), np.full(N_TONES, 2 * np.pi)])

    beta_step  = BETA_MAX * 0.3
    phase_step = np.pi

    class _Step:
        def __init__(self, stepsize=1.0):
            self.stepsize = stepsize

        def __call__(self, x):
            x = x.copy()
            x[:N_TONES] += rng.uniform(-beta_step, beta_step, N_TONES) * self.stepsize
            x[:N_TONES]  = np.clip(x[:N_TONES], 0.0, BETA_MAX)
            x[N_TONES:] += rng.uniform(-phase_step, phase_step, N_TONES) * self.stepsize
            x[N_TONES:]  = x[N_TONES:] % (2 * np.pi)
            return x

    target_orders = set(target_norm.keys())

    def obj(x):
        b  = x[:N_TONES]
        th = x[N_TONES:]
        harmonics, amps = _compute_spectrum(b, th)
        power = np.abs(amps) ** 2
        total = power.sum()
        if total < 1e-30:
            return 1.0
        h_to_idx = {int(h): i for i, h in enumerate(harmonics)}
        mse = 0.0
        for order, frac in target_norm.items():
            actual = power[h_to_idx[order]] / total if order in h_to_idx else 0.0
            mse += (actual - frac) ** 2
        for i, h in enumerate(harmonics):
            if int(h) not in target_orders:
                mse += (power[i] / total) ** 2
        return mse

    x0 = np.concatenate([
        rng.uniform(0.5, min(2.5, BETA_MAX), N_TONES),
        rng.uniform(0, 2 * np.pi, N_TONES),
    ])
    opt = basinhopping(
        obj, x0,
        minimizer_kwargs={"method": "L-BFGS-B", "bounds": list(zip(lower, upper))},
        niter=N_ITER,
        take_step=_Step(),
        seed=rng,
    )
    harmonics, amplitudes = _compute_spectrum(opt.x[:N_TONES], opt.x[N_TONES:])
    return {'harmonics': harmonics, 'amplitudes': amplitudes, 'opt': opt}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_comb(rng, order_min, order_max) -> dict:
    """Half-normal weights normalized to sum = 1 over [order_min, order_max]."""
    orders  = list(range(order_min, order_max + 1))
    weights = np.abs(rng.standard_normal(len(orders)))
    weights /= weights.sum()
    return {o: float(w) for o, w in zip(orders, weights)}


def _order_color(order):
    if order in _COMBLINE_COLORS:
        return _COMBLINE_COLORS[order]
    return _COLORS[(order + 10) % len(_COLORS)]


def _actual_fracs(harmonics, amplitudes, orders) -> dict:
    power   = np.abs(amplitudes) ** 2
    total   = power.sum()
    h_idx   = {int(h): i for i, h in enumerate(harmonics)}
    return {
        o: float(power[h_idx[o]] / total) if o in h_idx else 0.0
        for o in orders
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

orders = list(range(ORDER_MIN, ORDER_MAX + 1))

# Collect (target %, actual %, order, trial) for every (trial, order) pair
points = []   # list of (target_pct, actual_pct, order, trial)

if LOAD_FROM is not None:
    _d = np.load(LOAD_FROM)
    if 'trial_index' in _d:
        _trial_arr = _d['trial_index'].tolist()
    else:
        _n_orders_f = int(_d['order_max']) - int(_d['order_min']) + 1
        _n_trials_f = int(_d['n_trials'])
        _trial_arr  = np.repeat(np.arange(_n_trials_f), _n_orders_f).tolist()
    points = list(zip(_d['target_pct'].tolist(), _d['actual_pct'].tolist(),
                      _d['order'].tolist(), _trial_arr))
    print(f"Loaded {len(points)} points from {LOAD_FROM}")
else:
    rng_combs = np.random.default_rng(SEED_COMBS)
    for trial in range(N_TRIALS):
        target = _random_comb(rng_combs, ORDER_MIN, ORDER_MAX)
        seed   = (SEED_OPT + trial) if SEED_OPT is not None else None
        print(f"Trial {trial + 1}/{N_TRIALS}  (seed={seed})")

        res     = _optimize(target, seed=seed)
        actuals = _actual_fracs(res['harmonics'], res['amplitudes'], orders)

        for o in orders:
            points.append((target[o] * 100.0, actuals[o] * 100.0, o, trial))

        mse = sum((target[o] - actuals[o]) ** 2 for o in orders)
        print(f"  MSE={mse:.6f}   obj={res['opt'].fun:.6f}")

    print(f"\nDone. {len(points)} data points ({N_TRIALS} trials × {len(orders)} orders).")

    if SAVE_DIR is not None:
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path  = Path(SAVE_DIR) / f'benchmark_comb_results_{timestamp}.npz'
        np.savez(
            save_path,
            target_pct  = np.array([p[0] for p in points]),
            actual_pct  = np.array([p[1] for p in points]),
            order       = np.array([p[2] for p in points], dtype=int),
            trial_index = np.array([p[3] for p in points], dtype=int),
            order_min   = ORDER_MIN,
            order_max   = ORDER_MAX,
            n_trials    = N_TRIALS,
            seed_combs  = SEED_COMBS if SEED_COMBS is not None else -1,
            seed_opt    = SEED_OPT   if SEED_OPT   is not None else -1,
            n_tones     = N_TONES,
            beta_max    = BETA_MAX,
            n_iter      = N_ITER,
        )
        print(f"Results saved to {save_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig_w = left_mm + axes_size_mm + right_mm
fig_h = bottom_mm + axes_size_mm + top_mm
fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
fig.subplots_adjust(
    left   = left_mm   / fig_w,
    right  = 1 - right_mm  / fig_w,
    bottom = bottom_mm / fig_h,
    top    = 1 - top_mm   / fig_h,
)

# Convert points to plot units
def _to_plot(pct):
    if SCALE == 'dB':
        return 10.0 * np.log10(np.maximum(pct / 100.0, 1e-30))
    return pct

# Diagonal reference — span whichever limits are larger
all_plot_vals = [v for pt in points for v in (_to_plot(pt[0]), _to_plot(pt[1]))]
if SCALE == 'dB':
    _auto_lo, _auto_hi = FLOOR_DBC, max(all_plot_vals) * 0.95 if all_plot_vals else 0.0
else:
    _auto_lo, _auto_hi = 0.0, max(all_plot_vals) * 1.08 if all_plot_vals else 1.0
x_lo, x_hi = XLIM if XLIM is not None else (_auto_lo, _auto_hi)
y_lo, y_hi = YLIM if YLIM is not None else (_auto_lo, _auto_hi)
diag_lo = min(x_lo, y_lo)
diag_hi = max(x_hi, y_hi)
ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color='#333333', linewidth=1.2,
        linestyle='--', zorder=1)

# Scatter points, grouped by order for the legend
plot_orders = orders if PLOT_ORDERS is None else list(PLOT_ORDERS)

legend_handles = []
for o in plot_orders:
    color    = _order_color(o)
    rgba     = list(_mc.to_rgba(color))
    rgba[3]  = MARKER_ALPHA
    ec = color if EDGE_COLOR == 'same' else EDGE_COLOR
    xs = [_to_plot(pt[0]) for pt in points if pt[2] == o]
    ys = [_to_plot(pt[1]) for pt in points if pt[2] == o]
    ax.scatter(xs, ys,
               s=MARKER_PT ** 2,
               facecolors=[rgba] * len(xs),
               edgecolors=ec,
               linewidths=EDGE_WIDTH,
               zorder=2)
    handle = plt.scatter([], [], s=MARKER_PT ** 2,
                         facecolors=[rgba], edgecolors=ec,
                         linewidths=EDGE_WIDTH, label=f'n={o:+d}')
    legend_handles.append(handle)

ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_aspect('equal', adjustable='box')
if SCALE == 'dB':
    ax.set_xlabel(r'Target power [dB of total]', fontsize=axis_label_fontsize)
    ax.set_ylabel(r'Achieved power [dB of total]', fontsize=axis_label_fontsize)
else:
    ax.set_xlabel(r'Target power [% of total]', fontsize=axis_label_fontsize)
    ax.set_ylabel(r'Achieved power [% of total]', fontsize=axis_label_fontsize)

ax.legend(handles=legend_handles, fontsize=tick_label_fontsize,
          frameon=False, ncol=2, loc='upper left')

for spine in ax.spines.values():
    spine.set_linewidth(spine_linewidth)
ax.tick_params(axis='both', direction=tick_direction,
               width=tick_width, labelsize=tick_label_fontsize)

if FOR_PUBLICATION:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom=False, labelleft=False)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    svg_path = Path(PUB_SAVE_FOLDER) / PUB_SVG_NAME
    fig.savefig(svg_path, format='svg')
    print(f"Saved: {svg_path}")

out = Path(__file__).parent / 'benchmark_comb_optimization.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Per-trial comb figures
# ---------------------------------------------------------------------------

if PUB_PLOT_COMBS:
    def _comb_y(pct):
        if COMB_SCALE == 'dB':
            return 10.0 * np.log10(max(pct / 100.0, 1e-30))
        return pct

    comb_baseline = COMB_FLOOR_DBC if COMB_SCALE == 'dB' else COMB_FLOOR_PCT

    # Target comb comes from `points` (main LOAD_FROM / optimization run).
    # Derive orders and trial ids from the actual data, not config vars.
    trial_ids   = sorted({pt[3] for pt in points if pt[3] is not None})
    comb_orders = sorted({pt[2] for pt in points})

    # Build per-trial target lookup: {trial: {order: target_pct}}
    target_by_trial = {}
    for pt in points:
        tid, o, tgt = pt[3], pt[2], pt[0]
        target_by_trial.setdefault(tid, {})[o] = tgt

    # Load actual values from each overlay file.
    def _load_actuals(fpath):
        """Return {trial: {order: actual_pct}} from an npz file."""
        _d = np.load(fpath)
        if 'trial_index' in _d:
            _ti = _d['trial_index'].tolist()
        else:
            _n_ord = int(_d['order_max']) - int(_d['order_min']) + 1
            _ti    = np.repeat(np.arange(int(_d['n_trials'])), _n_ord).tolist()
        out = {}
        for act_p, o, ti in zip(_d['actual_pct'].tolist(), _d['order'].tolist(), _ti):
            out.setdefault(ti, {})[int(o)] = act_p
        return out

    if COMB_ACTUAL_SHOW:
        _actual_files = COMB_ACTUAL_FILES if COMB_ACTUAL_FILES is not None else (
            [LOAD_FROM] if LOAD_FROM is not None else []
        )
        _color_list = COMB_ACTUAL_COLOR if isinstance(COMB_ACTUAL_COLOR, list) else [COMB_ACTUAL_COLOR]
        _actual_datasets = [_load_actuals(f) for f in _actual_files]
    else:
        _actual_datasets = []
        _color_list      = []
        _actual_files    = []

    for tid in trial_ids:
        tgt_map = target_by_trial.get(tid, {})

        fig_c_w = COMB_LEFT_MM + COMB_AXES_W_MM + COMB_RIGHT_MM
        fig_c_h = COMB_BOTTOM_MM + COMB_AXES_H_MM + COMB_TOP_MM
        fig_c, ax_c = plt.subplots(figsize=(fig_c_w / 25.4, fig_c_h / 25.4))
        fig_c.subplots_adjust(
            left   = COMB_LEFT_MM   / fig_c_w,
            right  = 1 - COMB_RIGHT_MM  / fig_c_w,
            bottom = COMB_BOTTOM_MM / fig_c_h,
            top    = 1 - COMB_TOP_MM   / fig_c_h,
        )

        ax_c.axhline(comb_baseline, color='#555555', linewidth=0.8, zorder=0)

        y_all = []
        for o in comb_orders:
            if o not in tgt_map:
                continue
            color = _order_color(o)
            y_t = _comb_y(tgt_map[o])
            y_all.append(y_t)

            ax_c.plot([o, o], [comb_baseline, y_t],
                      color=color, linewidth=COMB_STEM_WIDTH, solid_capstyle='butt', zorder=1)
            ax_c.plot(o, y_t, 'o', color=color, markersize=COMB_BALL_PT, zorder=2)

        # Actual overlays — one set of open circles per file
        for fi, act_by_trial in enumerate(_actual_datasets):
            act_map = act_by_trial.get(tid, {})
            raw_color = _color_list[fi % len(_color_list)]
            for o in comb_orders:
                if o not in act_map:
                    continue
                y_a = _comb_y(act_map[o])
                y_all.append(y_a)
                ec = _order_color(o) if raw_color == 'same' else raw_color
                ax_c.plot(o, y_a, 'o',
                          markerfacecolor='none',
                          markeredgecolor=ec,
                          markeredgewidth=COMB_ACTUAL_EW,
                          markersize=COMB_ACTUAL_PT,
                          alpha=COMB_ACTUAL_ALPHA,
                          zorder=3)

        if COMB_CEILING is not None:
            y_top = COMB_CEILING
        else:
            y_top = max(y_all) * 1.1 if y_all else (0.0 if COMB_SCALE == 'dB' else 100.0)
        ax_c.set_ylim(comb_baseline, y_top)
        ax_c.set_xlim(comb_orders[0] - 0.7, comb_orders[-1] + 0.7)
        ax_c.set_xticks(comb_orders)

        ax_c.set_xlabel('')
        ax_c.set_ylabel('')
        is_first = (tid == trial_ids[0])
        ax_c.tick_params(labelbottom=False, labelleft=False,
                         bottom=is_first)
        for spine in ax_c.spines.values():
            spine.set_linewidth(spine_linewidth)
        ax_c.tick_params(axis='y', direction=tick_direction,
                         width=tick_width, labelsize=tick_label_fontsize)
        if is_first:
            ax_c.tick_params(axis='x', direction=tick_direction,
                             width=tick_width, labelsize=tick_label_fontsize)

        svg_path_c = Path(COMB_SVG_FOLDER) / f'{COMB_SVG_PREFIX}_{tid:03d}.svg'
        fig_c.savefig(svg_path_c, format='svg')
        print(f"Saved: {svg_path_c}")

plt.show()
