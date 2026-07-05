"""
Optimize a single-PM K-tone drive to match a target optical comb.

Physical model
--------------
  E(t) = exp(i * sum_{k=1}^{K} beta_k * sin(k*Omega*t + theta_k))

Tunable parameters: modulation depths beta_k and RF phases theta_k (k = 1..K).

Objectives
----------
  "power" → maximise power at WANTED_HARMONIC
  "ratio" → minimise (total − wanted) / wanted  (suppress all other lines)
  "comb"  → minimise MSE between spectrum and TARGET_COMB (% of total power);
             the free parameter WANTED_HARMONIC is unused in this mode

Warm-starting
-------------
  Set RESUME to a previous result folder containing result.json written by
  this script or by patent_idea_analysis/run_multitone_optimization.py.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from pathlib import Path
from scipy.optimize import basinhopping
from scipy.special import jv
from graphics import (
    BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2, DARKBLUE2,
    TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2,
)

_COLORS = [BLUE2, RED2, GREEN2, VIOLET2, ORANGE2, DARKGREEN2,
           DARKBLUE2, TAN2, PINK2, DARKGRAY2, BEIGE2, LIGHTBLUE2]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WANTED_HARMONIC = 1       # target sideband index (ignored when OBJECTIVE="comb")
BETA_MAX        = 2.5     # upper bound on each beta_k (rad)
N_MAX           = 30      # harmonic truncation order
N_TONES         = 3      # number of RF tones (k = 1 .. N_TONES)
N_ITER          = 200     # basin-hopping iterations
SEED            = 42      # integer for reproducibility, or None for random

# "power" : maximise power at WANTED_HARMONIC
# "ratio" : minimise (total − wanted) / wanted
# "comb"  : minimise MSE to TARGET_COMB
OBJECTIVE = "comb"

# Target power distribution (% of total optical power).
# Missing orders default to 0 % target.  Only used when OBJECTIVE = "comb".
# Example below targets equal power in harmonics -1 and +1 (SSB pair):
# TARGET_COMB = {
#     -4: 100.0,
#     -3: 100.0,
#     -2: 100.0,
#     -1: 100.0,
#     0: 100.0,
#     1: 100.0,
#     2: 100.0,
#     3: 100.0,
#     4: 100.0
# }

TARGET_COMB = {
    3: 100.0,
    -3: 100.0
}

METHOD = "fft"    # "fft" (fast) or "conv" (exact Jacobi-Anger convolution)
N_FFT  = 8192     # FFT size (only used when METHOD = "fft")

# Sawtooth initial guess.  For wanted harmonic n:
#   beta_k = clip(2n/k, 0, BETA_MAX),  theta_k = 0 (odd) or π (even).
# RESUME takes precedence when both are set.
SAWTOOTH_INIT = False

RESUME = None  # e.g. r"C:\path\to\results\20260629_120000"

OUT_DIR = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\comb_optimization"

# ---------------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------------
# Harmonic orders shown in the spectrum figure.
PLOT_ORDERS = list(range(-5, 6))

# 'percent' → |A_p|² as % of total power
# 'dB'      → 10·log10(|A_p|²) in dBc  (0 dBc = unit CW carrier)
SCALE = 'percent'

FLOOR_DBC    = -30.0  # dB-scale y-axis floor (only used when SCALE = 'dB')
CEILING_DBC  = 1  # dB-scale y-axis ceiling; None = auto (just above tallest stem)
CEILING_PCT  = 101.0   # percent-scale y-axis ceiling; None = auto (just above tallest stem)
FLOOR_PCT    = 0.0    # percent-scale y-axis floor

# 'stem'       → lollipop stems + balls
# 'lorentzian' → narrow Lorentzian peaks
PLOT_STYLE      = 'stem'
LORENTZIAN_FWHM = 0.1
LORENTZIAN_NPTS = 500

# Per-combline color; orders outside this dict auto-cycle through _COLORS.
COMBLINE_COLORS = {
    -3: '#bf7362',
    -2: RED2,
    -1: ORANGE2,
     0: GREEN2,
     1: LIGHTBLUE2,
     2: '#5c70aa',
     3: VIOLET2,
}

# Figure layout (mm)
axes_width_mm  = 100.0
axes_height_mm =  40.0
left_mm        =  20.0
right_mm       =  10.0
bottom_mm      =  15.0
top_mm         =   8.0

spine_linewidth     = 2.0
tick_width          = 2.0
tick_direction      = 'in'
axis_label_fontsize = 10.0
tick_label_fontsize =  8.0
stem_linewidth      =  3
markersize          =  9.0

SVG_FOLDER = None  # e.g. r"C:\path\to\media"  — set to save SVGs

# ---------------------------------------------------------------------------
# Spectrum computation (self-contained, no cross-folder imports)
# ---------------------------------------------------------------------------

def _compute_fft(betas, thetas, n_max, n_fft):
    betas  = np.asarray(betas,  dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    t       = np.linspace(0, 2 * np.pi, n_fft, endpoint=False)
    phase_t = np.zeros(n_fft)
    for k_idx, (b, th) in enumerate(zip(betas, thetas)):
        phase_t += b * np.sin((k_idx + 1) * t + th)
    fft_raw   = np.fft.fft(np.exp(1j * phase_t)) / n_fft
    harmonics = np.arange(-n_max, n_max + 1)
    return harmonics, fft_raw[harmonics % n_fft]


def _compute_conv(betas, thetas, n_max):
    betas  = np.asarray(betas,  dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    K      = len(betas)
    n_range    = np.arange(-n_max, n_max + 1)
    harmonics  = np.array([0])
    amplitudes = np.array([1.0 + 0j])
    for k_idx, (beta, theta) in enumerate(zip(betas, thetas)):
        k = k_idx + 1
        bessel_coeffs = jv(n_range, beta) * np.exp(1j * n_range * theta)
        tone_size = 2 * k * n_max + 1
        tone_amps = np.zeros(tone_size, dtype=complex)
        for n_idx, n in enumerate(n_range):
            tone_amps[k * int(n) + k * n_max] += bessel_coeffs[n_idx]
        amplitudes = np.convolve(amplitudes, tone_amps)
        harmonics  = np.arange(harmonics[0] - k * n_max,
                               harmonics[-1] + k * n_max + 1)
    return harmonics, amplitudes


def compute_spectrum(betas, thetas):
    if METHOD == "fft":
        return _compute_fft(betas, thetas, N_MAX, N_FFT)
    return _compute_conv(betas, thetas, N_MAX)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _make_sawtooth_x0(wanted_harmonic, n_tones, beta_max):
    ks    = np.arange(1, n_tones + 1, dtype=float)
    betas = np.minimum(2.0 * abs(wanted_harmonic) / ks, beta_max)
    thetas = np.where(ks % 2 == 0, np.pi, 0.0)
    return np.concatenate([betas, thetas])


def _load_resume(folder):
    result_path = os.path.join(folder, "result.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Cannot find result.json in {folder}")
    with open(result_path) as f:
        prev = json.load(f)
    betas  = prev["betas"]
    thetas = prev["thetas"]
    if len(betas) != N_TONES:
        raise ValueError(
            f"Resumed run has {len(betas)} tones but N_TONES={N_TONES}."
        )
    return np.concatenate([betas, thetas])


def _build_objective():
    # pre-index the wanted harmonic
    if METHOD == "fft":
        wanted_idx = int(WANTED_HARMONIC + N_MAX)
    else:
        total_range = N_TONES * (N_TONES + 1) // 2 * N_MAX
        wanted_idx  = int(total_range + WANTED_HARMONIC)

    # normalised target for "comb" mode
    if OBJECTIVE == "comb":
        target_pct_total = sum(TARGET_COMB.values())
        if target_pct_total <= 0:
            raise ValueError("TARGET_COMB values must sum to a positive number.")
        _target_norm = {k: v / target_pct_total for k, v in TARGET_COMB.items()}

    def obj(x):
        b  = x[:N_TONES]
        th = x[N_TONES:]
        harmonics, amps = compute_spectrum(b, th)
        power = np.abs(amps) ** 2

        if OBJECTIVE == "power":
            return -power[wanted_idx]

        if OBJECTIVE == "ratio":
            wp = power[wanted_idx]
            if wp < 1e-30:
                return 1e30
            return (power.sum() - wp) / wp

        # "comb": MSE against target fractional distribution
        total = power.sum()
        if total < 1e-30:
            return 1.0
        h_to_idx = {int(h): i for i, h in enumerate(harmonics)}
        mse = 0.0
        for order, frac in _target_norm.items():
            actual = power[h_to_idx[order]] / total if order in h_to_idx else 0.0
            mse += (actual - frac) ** 2
        # also penalise power outside target orders
        target_orders = set(_target_norm.keys())
        for i, h in enumerate(harmonics):
            if int(h) not in target_orders:
                mse += (power[i] / total) ** 2
        return mse

    return obj


def optimize(x0=None):
    rng = np.random.default_rng(SEED)
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

    if x0 is None:
        x0 = np.zeros(2 * N_TONES)
        x0[:N_TONES] = rng.uniform(0.5, min(2.5, BETA_MAX), N_TONES)
        x0[N_TONES:] = rng.uniform(0, 2 * np.pi, N_TONES)

    obj = _build_objective()
    opt = basinhopping(
        obj, x0,
        minimizer_kwargs={"method": "L-BFGS-B", "bounds": list(zip(lower, upper))},
        niter=N_ITER,
        take_step=_Step(),
        seed=rng,
    )
    betas_opt  = list(opt.x[:N_TONES])
    thetas_opt = list(opt.x[N_TONES:])
    harmonics, amplitudes = compute_spectrum(betas_opt, thetas_opt)
    power        = np.abs(amplitudes) ** 2
    wanted_power = float(power[int(WANTED_HARMONIC + N_MAX)]) if METHOD == "fft" else float(
        power[int(N_TONES * (N_TONES + 1) // 2 * N_MAX + WANTED_HARMONIC)]
    )
    total_power  = float(power.sum())
    ratio        = (total_power - wanted_power) / wanted_power if wanted_power > 1e-30 else np.inf
    return {
        "betas":        betas_opt,
        "thetas":       thetas_opt,
        "ratio":        ratio,
        "wanted_power": wanted_power,
        "total_power":  total_power,
        "harmonics":    harmonics,
        "amplitudes":   amplitudes,
        "opt_result":   opt,
    }


# ---------------------------------------------------------------------------
# Plotting (multi-tone AO style)
# ---------------------------------------------------------------------------

def _order_color(order, auto_idx):
    return COMBLINE_COLORS.get(int(order), _COLORS[auto_idx % len(_COLORS)])


def _make_fig_ax():
    fig_w = left_mm + axes_width_mm + right_mm
    fig_h = bottom_mm + axes_height_mm + top_mm
    fig, ax = plt.subplots(figsize=(fig_w / 25.4, fig_h / 25.4))
    fig.subplots_adjust(
        left   = left_mm   / fig_w,
        right  = 1 - right_mm  / fig_w,
        bottom = bottom_mm / fig_h,
        top    = 1 - top_mm   / fig_h,
    )
    return fig, ax


def _plot_spectrum(harmonics, amplitudes, title="", label_prefix=""):
    """Plot the spectrum in the multi-tone stem style."""
    plot_orders = np.array(sorted(PLOT_ORDERS))
    h_to_idx    = {int(h): i for i, h in enumerate(harmonics)}

    amps_plot = np.array([
        amplitudes[h_to_idx[p]] if p in h_to_idx else 0.0 + 0j
        for p in plot_orders
    ])
    power_plot = np.abs(amps_plot) ** 2
    total_power = (np.abs(amplitudes) ** 2).sum()

    if SCALE == 'percent':
        y          = power_plot / total_power * 100.0
        y_baseline = 0.0
        ylabel     = r'$|A_p|^2$ [% of total power]'
    else:
        with np.errstate(divide='ignore'):
            y = np.where(power_plot > 0, 10.0 * np.log10(power_plot), FLOOR_DBC - 1.0)
        y_baseline = FLOOR_DBC
        ylabel     = r'$|A_p|^2$ [dBc]'

    # console table
    unit = '%' if SCALE == 'percent' else 'dBc'
    if label_prefix:
        print(f"\n{label_prefix}")
    print(f"  {'order':>5}   {'power':>12}   {'phase [deg]':>12}")
    print(f"  {'─'*5}   {'─'*12}   {'─'*12}")
    for p, yv, amp in zip(plot_orders, y, amps_plot):
        phase_deg = float(np.degrees(np.angle(amp)))
        print(f"  p={p:+d}   {yv:>10.4f} {unit}   {phase_deg:>+10.2f}°")
    total_shown = sum(power_plot) / total_power * 100.0
    print(f"\n  Total power in plotted orders: {total_shown:.2f}%")

    fig, ax = _make_fig_ax()

    hw = LORENTZIAN_FWHM * LORENTZIAN_PLOT_WIDTH / 2.0 if PLOT_STYLE == 'lorentzian' else None

    for auto_idx, (p, yi) in enumerate(zip(plot_orders, y)):
        c = _order_color(p, auto_idx)
        if PLOT_STYLE == 'lorentzian':
            x_local = np.linspace(p - hw, p + hw, LORENTZIAN_NPTS)
            gamma   = LORENTZIAN_FWHM / 2.0
            y_curve = (yi - y_baseline) * gamma**2 / ((x_local - p)**2 + gamma**2) + y_baseline
            ax.plot(x_local, y_curve, color=c, linewidth=stem_linewidth,
                    solid_capstyle='round', zorder=2)
        else:
            ax.plot([p, p], [y_baseline, yi],
                    color=c, linewidth=stem_linewidth, solid_capstyle='round', zorder=2)
            ax.plot(p, yi, 'o', color=c, markersize=markersize, markeredgewidth=0, zorder=3)

    ax.axhline(y_baseline, color='#333333', linewidth=0.8, zorder=1)

    ax.set_xlabel(r'Harmonic order $p$', fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_xlim(plot_orders[0] - 0.8, plot_orders[-1] + 0.8)
    if SCALE == 'percent':
        top_pct = CEILING_PCT if CEILING_PCT is not None else max(y.max() * 1.1, 1.0)
        ax.set_ylim(bottom=FLOOR_PCT, top=top_pct)
    else:
        top_dbc = CEILING_DBC if CEILING_DBC is not None else max(y.max() * 1.05, y_baseline + 1)
        ax.set_ylim(bottom=y_baseline, top=top_dbc)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(axis='both', direction=tick_direction,
                   width=tick_width, labelsize=tick_label_fontsize)

    if title:
        ax.set_title(title, fontsize=axis_label_fontsize)

    return fig, ax


LORENTZIAN_PLOT_WIDTH = 0.6  # x-range around each peak in combline units


def _save_fig(fig, stem, out_folder):
    png_path = os.path.join(out_folder, f'{stem}.png')
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {png_path}")
    if SVG_FOLDER:
        svg_dir = Path(SVG_FOLDER)
        svg_dir.mkdir(parents=True, exist_ok=True)
        svg_path = svg_dir / f'{stem}.svg'
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"Saved: {svg_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

x0 = None
sawtooth_result = None

if SAWTOOTH_INIT:
    ref_harmonic = list(TARGET_COMB.keys())[0] if OBJECTIVE == "comb" else WANTED_HARMONIC
    saw_x0 = _make_sawtooth_x0(ref_harmonic, N_TONES, BETA_MAX)
    x0 = saw_x0
    saw_harmonics, saw_amps = compute_spectrum(saw_x0[:N_TONES], saw_x0[N_TONES:])
    saw_power = np.abs(saw_amps) ** 2
    saw_wanted = float(saw_power[int(ref_harmonic + N_MAX)]) if METHOD == "fft" else 0.0
    sawtooth_result = {
        "betas":        list(saw_x0[:N_TONES]),
        "thetas":       list(saw_x0[N_TONES:]),
        "harmonics":    saw_harmonics,
        "amplitudes":   saw_amps,
        "wanted_power": saw_wanted,
        "ratio":        (saw_power.sum() - saw_wanted) / saw_wanted if saw_wanted > 1e-30 else np.inf,
    }
    betas_str = "  ".join(f"β{k+1}={b:.4f}" for k, b in enumerate(saw_x0[:N_TONES]))
    print(f"Sawtooth x0: {betas_str}")
    print(f"  → wanted_power={saw_wanted:.6f}  ({saw_wanted*100:.2f}%)")

if RESUME:
    x0 = _load_resume(RESUME)
    with open(os.path.join(RESUME, "result.json")) as f:
        prev = json.load(f)
    print(f"Warm-starting from {RESUME}  "
          f"(previous ratio: {prev['ratio']:.6f}  wanted power: {prev['wanted_power']:.6f})")

config = dict(
    wanted_harmonic = WANTED_HARMONIC,
    objective       = OBJECTIVE,
    target_comb     = TARGET_COMB if OBJECTIVE == "comb" else None,
    beta_max        = BETA_MAX,
    n_max           = N_MAX,
    n_tones         = N_TONES,
    n_iter          = N_ITER,
    seed            = SEED,
    method          = METHOD,
    n_fft           = N_FFT,
    resumed_from    = RESUME,
)
print("Running optimisation with config:", config)

res = optimize(x0=x0)

betas_str  = "  ".join(f"β{k+1}={b:.4f}"  for k, b  in enumerate(res["betas"]))
thetas_str = "  ".join(f"θ{k+1}={th:.4f}" for k, th in enumerate(res["thetas"]))
print(f"\nDone.  {betas_str}  {thetas_str}")
print(f"  ratio={res['ratio']:.6f}  wanted_power={res['wanted_power']:.6f}"
      f"  ({res['wanted_power']*100:.2f}%)  total={res['total_power']:.6f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
out_folder = os.path.join(OUT_DIR, timestamp)
os.makedirs(out_folder, exist_ok=True)

with open(os.path.join(out_folder, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

result_data = dict(
    betas        = [float(b)  for b  in res["betas"]],
    thetas       = [float(th) for th in res["thetas"]],
    ratio        = float(res["ratio"]),
    wanted_power = float(res["wanted_power"]),
    total_power  = float(res["total_power"]),
    nit          = int(res["opt_result"].nit),
    nfev         = int(res["opt_result"].nfev),
    message      = str(res["opt_result"].message),
)
with open(os.path.join(out_folder, "result.json"), "w") as f:
    json.dump(result_data, f, indent=2)

# Sawtooth comparison
if sawtooth_result is not None:
    saw_title = (
        rf"Sawtooth x0:  "
        + "  ".join(f"β{k+1}={b:.3f}" for k, b in enumerate(sawtooth_result["betas"]))
    )
    fig_saw, _ = _plot_spectrum(
        sawtooth_result["harmonics"], sawtooth_result["amplitudes"],
        title=saw_title, label_prefix="Sawtooth spectrum:",
    )
    _save_fig(fig_saw, "sawtooth_spectrum", out_folder)
    with open(os.path.join(out_folder, "sawtooth_result.json"), "w") as f:
        json.dump(dict(
            betas        = sawtooth_result["betas"],
            thetas       = sawtooth_result["thetas"],
            wanted_power = sawtooth_result["wanted_power"],
            ratio        = sawtooth_result["ratio"],
        ), f, indent=2)

# Optimised spectrum
opt_title = (
    rf"Optimised ({OBJECTIVE}):  "
    + "  ".join(f"β{k+1}={b:.3f}" for k, b in enumerate(res["betas"]))
    + rf"  ({res['wanted_power']*100:.1f}% @ h={WANTED_HARMONIC})"
)
fig_opt, _ = _plot_spectrum(
    res["harmonics"], res["amplitudes"],
    title=opt_title, label_prefix="Optimised spectrum:",
)
_save_fig(fig_opt, "optimised_spectrum", out_folder)

print(f"\nResults saved to {out_folder}")
plt.show()
