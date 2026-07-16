"""
harmonic_efficiency.py

Analyze signal-generator harmonic-suppression data and its effect on
phase-modulation sideband conversion efficiency.

Part 1 characterizes how well-suppressed each harmonic/subharmonic line of
the drive is, as a function of set drive power, and flags points where a
line couldn't be reliably distinguished from the ESA noise floor (reported
as upper bounds rather than dropped).

Part 2 (optional, needs --vpi-fund and --vpi-harm) propagates the measured
fundamental and one parasitic harmonic line through a two-tone phase
modulation model to show how much that parasitic harmonic can perturb the
+1/-1 sideband conversion efficiency away from the ideal single-tone
J_1(beta1)^2 curve, as a function of the (unknown) relative phase between
the two tones.

Usage:
    Edit the globals in the "user settings" block below (NPZ_PATH, VPI_FUND,
    VPI_HARM, etc.), then:
        python harmonic_efficiency.py
    Any of those can still be overridden on the command line, e.g.:
        python harmonic_efficiency.py DATA.npz --vpi-fund 5.0 --vpi-harm 2.3
        python harmonic_efficiency.py --selftest

Dependencies: numpy, scipy, matplotlib (+ stdlib argparse/csv/pathlib/sys).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt

# ── user settings ─────────────────────────────────────────────────────────────
# Edit these directly so the script can be run with no CLI args. Every value
# here can still be overridden on the command line (e.g. `--vpi-fund 4.8`);
# a CLI flag wins over the global when both are given.
NPZ_PATH = r"C:\Users\jake\OneDrive - UCB-O365\quantum_nanophoxonics\projects\dual_tone_aom\data\signal_generator_harmonic_suppression\harmonic_suppression_2026-07-15-13-21-33.npz"

PEAK_WINDOW_KHZ = 100.0   # peak search half-width around each line center [kHz]
MIN_SNR_DB = 6.0          # min peak-to-floor SNR to count as a real measurement [dB]

VPI_FUND = 5.6         # V_pi at the fundamental f [V] -- needed (with VPI_HARM) for Part 2
VPI_HARM = 2.51           # V_pi at the parasitic harmonic [V] -- needed (with VPI_FUND) for Part 2
HARMONIC_N = 2            # which harmonic multiplier is the parasitic tone
ATTEN_DB = 0.0            # attenuation between the device plane and the ESA at f [dB]
ATTEN2_DB = None          # same, at the harmonic frequency (None -> same as ATTEN_DB)
PHASE_DEG = 180          # optional fixed relative phase Phi [deg] to overlay on Figure 2
SIDEBANDS = [-2, -1, 0, 1, 2]   # sideband orders q to plot efficiency for in Figure 2

SELFTEST = False          # if True, run self-tests and exit (no data file needed)

# ── plot color scheme ─────────────────────────────────────────────────────────
# Matches this lab's harmonic-color convention (see e.g. dual_tone_sweep_data.py
# / vpi_anchor_analysis.py): each *harmonic* order n gets a fixed color; a
# *subharmonic* f/k reuses the color of harmonic order -k (same "family", other
# side of the comb), which is a deliberate pairing choice, not a coincidence.
# Anything outside that table (harmonics beyond +-3, or unrecognized labels)
# falls back to _EXTRA_COLORS, cycled in the order the labels appear.
_HARMONIC_COLORS = {
    -3: '#bf7362',
    -2: '#e5a3a3',   # RED2
    -1: '#FBD8A2',   # ORANGE2
     0: '#93C572',   # GREEN2
     1: '#b2cbf2',   # LIGHTBLUE2
     2: '#5c70aa',
     3: '#C2B7E9',   # VIOLET2
}
_EXTRA_COLORS = ['#2522d4', '#e6b8d0', '#EED7A1', '#8DB591', '#475c6c', '#777777']

# ── style (freely customizable) ───────────────────────────────────────────────
_MM_PER_INCH = 25.4


def _mm_to_in(size_mm):
    return (size_mm[0] / _MM_PER_INCH, size_mm[1] / _MM_PER_INCH)


# Sizes below are the AXES (plot area) size in mm (width, height), not the
# whole figure -- the figure is sized as axes + margins so tick/axis labels
# have their own room rather than eating into the plot area.
FIGSIZE_1_MM = (100, 40)   # each panel of Figure 1
FIGSIZE_2_MM = (100, 40)   # Figure 2

AXES_MARGIN_LEFT_IN   = 0.55   # room for the y tick labels + y axis label
AXES_MARGIN_BOTTOM_IN = 0.4    # room for the x tick labels + x axis label
AXES_MARGIN_RIGHT_IN  = 0.1
AXES_MARGIN_TOP_IN    = 0.1
AXES_VSPACE_IN        = 0.5    # gap between stacked panels (Figure 1's (a)/(b))

# Figure 1
VALID_MARKER      = 'o'
VALID_MARKERSIZE  = 6
VALID_LINESTYLE   = 'none'   # 'none' = points only; '-' to also connect them
VALID_LINEWIDTH   = 1.5
VALID_ALPHA       = 1.0
VALID_ZORDER      = 3

BOUND_MARKER          = 'o'
BOUND_MARKERSIZE      = 6
BOUND_FACECOLOR       = 'none'   # open markers for "upper bound" points
BOUND_ARROW_HEIGHT_DB = 2.0      # length of the downward "this is a max" arrow
BOUND_ALPHA           = 0.9
BOUND_ZORDER          = 3

ABS_POWER_MARKER     = 'o'
ABS_POWER_MARKERSIZE = 5
ABS_POWER_LINESTYLE  = '-'
ABS_POWER_LINEWIDTH  = 1.3
ABS_POWER_ALPHA      = 1.0
ABS_POWER_ZORDER     = 2

LEGEND_FONTSIZE = 8
LABEL_FONTSIZE  = 11

# Figure 2
CE_LABEL_FONTSIZE = 8   # axis label font size on the sideband conversion-efficiency plot
CE_TICK_FONTSIZE  = 8   # tick label font size on the sideband conversion-efficiency plot

IDEAL_CURVE_LINESTYLE = '--'
IDEAL_CURVE_LINEWIDTH = 2.0
IDEAL_CURVE_ZORDER    = 5

BAND_ALPHA        = 0.28    # normal (harmonic line was a real measurement)
BAND_ALPHA_BOUND  = 0.14    # lighter (harmonic line was only an upper bound)
BAND_HATCH_BOUND  = '//'    # hatch marking a bound-derived ("worst case") band
BAND_ZORDER       = 1

FIXED_PHASE_LINESTYLE = '-'
FIXED_PHASE_LINEWIDTH = 2.0
FIXED_PHASE_ZORDER    = 4

N_PHI_SAMPLES = 361   # phase resolution used to sweep out the min/max bands
M_MAX = 8             # Bessel-series truncation for the two-tone amplitude
# ─────────────────────────────────────────────────────────────────────────────


# ── color helpers ──────────────────────────────────────────────────────────

def _parse_label(label: str):
    """('harmonic', k) for 'n=k', ('subharmonic', k) for 'f/k', else ('other', None)."""
    if label.startswith('n=') and label[2:].lstrip('-').isdigit():
        return 'harmonic', int(label[2:])
    if label.startswith('f/') and label[2:].isdigit():
        return 'subharmonic', int(label[2:])
    return 'other', None


def build_color_map(labels) -> dict:
    """Assign each label a color, per the module-level scheme/fallback above."""
    colors = {}
    extra_iter = iter(_EXTRA_COLORS)
    for label in labels:
        kind, k = _parse_label(label)
        if kind == 'harmonic':
            colors[label] = _HARMONIC_COLORS.get(k, next(extra_iter, '#000000'))
        elif kind == 'subharmonic':
            colors[label] = _HARMONIC_COLORS.get(-k, next(extra_iter, '#000000'))
        else:
            colors[label] = next(extra_iter, '#000000')
    return colors


def build_sideband_color_map(sidebands) -> dict:
    """Same color scheme as build_color_map, keyed directly by sideband order q."""
    colors = {}
    extra_iter = iter(_EXTRA_COLORS)
    for q in sidebands:
        colors[q] = _HARMONIC_COLORS.get(q, next(extra_iter, '#000000'))
    return colors


# ── data loading ──────────────────────────────────────────────────────────

def load_dataset(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {
        'drive_freq': float(d['drive_freq']),
        'drive_powers': np.asarray(d['drive_powers'], dtype=float),
        'harmonics': np.asarray(d['harmonics'], dtype=int),
        'sub_harmonics': np.asarray(d['sub_harmonics'], dtype=int),
        'center_freqs': np.asarray(d['center_freqs'], dtype=float),
        'labels': np.asarray(d['labels']).astype(str),
        'window_hz': float(d['window_hz']),
        'esa_freq_step_hz': float(d['esa_freq_step_hz']),
        'offsets_hz': np.asarray(d['offsets_hz'], dtype=float),
        'spectra': np.asarray(d['spectra'], dtype=float),
    }
    n_lines_expected = len(out['harmonics']) + len(out['sub_harmonics'])
    if not (len(out['center_freqs']) == n_lines_expected == len(out['labels'])):
        raise ValueError(
            f"center_freqs/labels length ({len(out['center_freqs'])}/{len(out['labels'])}) "
            f"doesn't match len(harmonics)+len(sub_harmonics) ({n_lines_expected})"
        )
    if out['spectra'].shape[1] != len(out['labels']):
        raise ValueError(
            f"spectra has {out['spectra'].shape[1]} lines but {len(out['labels'])} labels"
        )
    return out


def _find_label_index(labels: np.ndarray, target: str) -> int:
    idx = np.where(labels == target)[0]
    if len(idx) == 0:
        raise ValueError(f"Label {target!r} not found among {list(labels)}")
    return int(idx[0])


# ── Part 1: line powers and suppression ──────────────────────────────────────

def analyze_lines(offsets_hz: np.ndarray, window_hz: float, spectra: np.ndarray,
                   peak_window_hz: float, min_snr_db: float):
    """
    For every (power step, line), find its peak within +-peak_window_hz of
    line center, and a noise floor from the trace outside +-0.5*window_hz.

    Returns (peak_dbm, floor_dbm, valid, peak_offset_hz), each shape (Np, Nl).
    A point is `valid` iff peak >= floor + min_snr_db AND the peak sits
    within +-50 kHz of line center (a real CW tone, not a noise spike).
    """
    Np, Nl, _Nf = spectra.shape
    peak_mask = np.abs(offsets_hz) <= peak_window_hz
    floor_mask = np.abs(offsets_hz) > 0.5 * window_hz
    if not np.any(peak_mask):
        raise ValueError("--peak-window-khz excludes all offset bins; widen it.")
    if not np.any(floor_mask):
        raise ValueError("No offset bins fall outside 0.5*window_hz for a noise-floor estimate.")

    peak_offsets_sub = offsets_hz[peak_mask]
    peak_dbm = np.empty((Np, Nl))
    peak_offset_hz = np.empty((Np, Nl))
    floor_dbm = np.empty((Np, Nl))

    for p in range(Np):
        for l in range(Nl):
            trace = spectra[p, l, :]
            sub = trace[peak_mask]
            i = int(np.argmax(sub))
            peak_dbm[p, l] = sub[i]
            peak_offset_hz[p, l] = peak_offsets_sub[i]
            floor_dbm[p, l] = np.median(trace[floor_mask])

    valid = (peak_dbm >= floor_dbm + min_snr_db) & (np.abs(peak_offset_hz) <= 50e3)
    return peak_dbm, floor_dbm, valid, peak_offset_hz


def write_csv(path, drive_powers, labels, peak_dbm, floor_dbm, valid, s_l_dbc):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['drive_power_dbm', 'line_label', 'peak_dbm', 'floor_dbm', 'valid', 's_l_dbc'])
        for p, power in enumerate(drive_powers):
            for l, label in enumerate(labels):
                w.writerow([
                    f"{power:.3f}", label,
                    f"{peak_dbm[p, l]:.3f}", f"{floor_dbm[p, l]:.3f}",
                    int(valid[p, l]), f"{s_l_dbc[p, l]:.3f}",
                ])


def _fixed_axes_figure(n_rows, size_mm, vspace_in=AXES_VSPACE_IN):
    """
    Build a figure with n_rows stacked axes, each exactly size_mm (width_mm,
    height_mm) in physical size. size_mm sets the AXES/plot area, not the
    whole figure: the figure is sized as axes + margins, so tick/axis labels
    get their own room instead of shrinking the plot area (which is what
    fig.tight_layout() would otherwise do).
    """
    axes_w_in, axes_h_in = _mm_to_in(size_mm)
    left, bottom, right, top = (AXES_MARGIN_LEFT_IN, AXES_MARGIN_BOTTOM_IN,
                                 AXES_MARGIN_RIGHT_IN, AXES_MARGIN_TOP_IN)
    fig_w = axes_w_in + left + right
    fig_h = n_rows * axes_h_in + (n_rows - 1) * vspace_in + bottom + top

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for i in range(n_rows):
        ax_bottom_in = bottom + (n_rows - 1 - i) * (axes_h_in + vspace_in)
        rect = [left / fig_w, ax_bottom_in / fig_h, axes_w_in / fig_w, axes_h_in / fig_h]
        axes.append(fig.add_axes(rect))
    return fig, axes


def make_figure1(drive_powers, labels, idx_fund, peak_dbm, valid, s_l_dbc, colors):
    fig, (ax_a, ax_b) = _fixed_axes_figure(2, FIGSIZE_1_MM)

    for l, label in enumerate(labels):
        color = colors[label]

        if l != idx_fund:
            v = valid[:, l]
            if np.any(v):
                ax_a.plot(drive_powers[v], s_l_dbc[v, l],
                          marker=VALID_MARKER, markersize=VALID_MARKERSIZE,
                          linestyle=VALID_LINESTYLE, linewidth=VALID_LINEWIDTH,
                          color=color, alpha=VALID_ALPHA, zorder=VALID_ZORDER,
                          label=label)
            if np.any(~v):
                ax_a.errorbar(drive_powers[~v], s_l_dbc[~v, l],
                              yerr=BOUND_ARROW_HEIGHT_DB, uplims=True,
                              fmt=BOUND_MARKER, markerfacecolor=BOUND_FACECOLOR,
                              markeredgecolor=color, ecolor=color,
                              markersize=BOUND_MARKERSIZE, alpha=BOUND_ALPHA,
                              linestyle='none', zorder=BOUND_ZORDER,
                              label=(label if not np.any(v) else None))

        ax_b.plot(drive_powers, peak_dbm[:, l],
                  marker=ABS_POWER_MARKER, markersize=ABS_POWER_MARKERSIZE,
                  linestyle=ABS_POWER_LINESTYLE, linewidth=ABS_POWER_LINEWIDTH,
                  color=color, alpha=ABS_POWER_ALPHA, zorder=ABS_POWER_ZORDER,
                  label=label)

    ax_a.set_xlabel('Set drive power [dBm]', fontsize=LABEL_FONTSIZE)
    ax_a.set_ylabel(r'Suppression $S_L = P_L - P_{n=1}$ [dBc]', fontsize=LABEL_FONTSIZE)
    ax_a.legend(fontsize=LEGEND_FONTSIZE, frameon=False, ncol=2)
    ax_a.grid(alpha=0.3)

    ax_b.set_xlabel('Set drive power [dBm]', fontsize=LABEL_FONTSIZE)
    ax_b.set_ylabel('Measured peak power [dBm]', fontsize=LABEL_FONTSIZE)
    ax_b.legend(fontsize=LEGEND_FONTSIZE, frameon=False, ncol=2)
    ax_b.grid(alpha=0.3)

    return fig


# ── Part 2: two-tone sideband model ──────────────────────────────────────────

def sideband_amp(q: int, beta1: np.ndarray, beta2: np.ndarray, phi: np.ndarray,
                  m_max: int = M_MAX) -> np.ndarray:
    """
    A_q(phi) = sum_{m=-m_max}^{m_max} J_{q-2m}(beta1) * J_m(beta2) * exp(i*m*phi)

    beta1, beta2: shape (N,) (broadcast together, one value per data point).
    phi: shape (P,).
    Returns complex array of shape (N, P).
    """
    beta1 = np.atleast_1d(np.asarray(beta1, dtype=float))
    beta2 = np.atleast_1d(np.asarray(beta2, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))

    m = np.arange(-m_max, m_max + 1)               # (M,)
    orders = q - 2 * m                              # (M,)
    Jn1 = jv(orders[None, :], beta1[:, None])       # (N, M)
    Jm2 = jv(m[None, :], beta2[:, None])            # (N, M)
    coeff = Jn1 * Jm2                                # (N, M), real
    E = np.exp(1j * np.outer(phi, m))                # (P, M)
    return coeff @ E.T                               # (N, M) @ (M, P) -> (N, P)


def _contiguous_runs(mask: np.ndarray):
    """Yield (value, start, end) for each contiguous run of a boolean array."""
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    if n == 0:
        return
    start, cur = 0, bool(mask[0])
    for i in range(1, n):
        if bool(mask[i]) != cur:
            yield cur, start, i
            start, cur = i, bool(mask[i])
    yield cur, start, n


def make_figure2(V1, harm_valid, bands, ideals, fixed_phase, sidebands):
    """
    bands: {q: (lo, hi)} min/max of eta_q over Phi, each shape matching V1.
    ideals: {q: J_q(beta1)^2}, same shape.
    fixed_phase: {q: eta_q at a single fixed Phi} or None.
    """
    order = np.argsort(V1)
    V1s = V1[order]
    harm_valid_s = harm_valid[order]
    colors = build_sideband_color_map(sidebands)

    fig, (ax,) = _fixed_axes_figure(1, FIGSIZE_2_MM)

    for q in sidebands:
        color = colors[q]

        if fixed_phase is not None:
            # A fixed Phi was given: show only that curve, not the
            # over-all-phases band/ideal (those describe the phase-unknown
            # case, which no longer applies once Phi is pinned down).
            ax.plot(V1s, fixed_phase[q][order], color=color, linestyle=FIXED_PHASE_LINESTYLE,
                    linewidth=FIXED_PHASE_LINEWIDTH, zorder=FIXED_PHASE_ZORDER)
            continue

        lo_s, hi_s = bands[q][0][order], bands[q][1][order]
        ideal_s = ideals[q][order]

        ax.plot(V1s, ideal_s, color=color, linestyle=IDEAL_CURVE_LINESTYLE,
                linewidth=IDEAL_CURVE_LINEWIDTH, zorder=IDEAL_CURVE_ZORDER)

        for is_valid, s, e in _contiguous_runs(harm_valid_s):
            sl = slice(max(s - 1, 0), e)   # overlap by one point so bands don't gap at seams
            alpha = BAND_ALPHA if is_valid else BAND_ALPHA_BOUND
            hatch = None if is_valid else BAND_HATCH_BOUND
            ax.fill_between(V1s[sl], lo_s[sl], hi_s[sl], color=color, alpha=alpha,
                             hatch=hatch, edgecolor=color, linewidth=0, zorder=BAND_ZORDER)

    ax.set_xlabel(r'$V_1$ [V$_\mathrm{rms}$]', fontsize=CE_LABEL_FONTSIZE)
    ax.set_ylabel('Sideband conversion efficiency', fontsize=CE_LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=CE_TICK_FONTSIZE)
    ax.grid(alpha=0.3)
    return fig


def run_part2(drive_powers, labels, peak_dbm, valid, args):
    harmonic_label = f'n={args.harmonic_n}'
    idx_harm = _find_label_index(labels, harmonic_label)
    atten2_db = args.atten2_db if args.atten2_db is not None else args.atten_db

    # V1 comes from the SET drive power (what was actually dialed in), not a
    # measured ESA peak -- there's no compression/loss ambiguity to it, unlike
    # the parasitic harmonic line below, which has no "set" value and must be
    # read off the ESA.
    p1_device_dbm = drive_powers + args.atten_db
    V1 = np.sqrt(50.0 * 10.0 ** (p1_device_dbm / 10.0) * 1e-3)

    p2_device_dbm = peak_dbm[:, idx_harm] + atten2_db
    V2 = np.sqrt(50.0 * 10.0 ** (p2_device_dbm / 10.0) * 1e-3)
    harm_valid = valid[:, idx_harm]

    beta1 = np.pi * V1 / args.vpi_fund
    beta2 = np.pi * V2 / args.vpi_harm

    phi_grid = np.linspace(0.0, 2 * np.pi, N_PHI_SAMPLES)
    phi0 = np.array([np.deg2rad(args.phase_deg)]) if args.phase_deg is not None else None

    bands, ideals, fixed_phase = {}, {}, ({} if phi0 is not None else None)
    for q in SIDEBANDS:
        eta_all = np.abs(sideband_amp(q, beta1, beta2, phi_grid)) ** 2
        bands[q] = (eta_all.min(axis=1), eta_all.max(axis=1))
        ideals[q] = jv(q, beta1) ** 2
        if phi0 is not None:
            fixed_phase[q] = np.abs(sideband_amp(q, beta1, beta2, phi0))[:, 0] ** 2

    fig2 = make_figure2(V1, harm_valid, bands, ideals, fixed_phase, SIDEBANDS)

    v1_ideal_peak = 1.8412 * args.vpi_fund / np.pi
    in_range = V1.min() <= v1_ideal_peak <= V1.max()

    print("\n=== Part 2 summary ===")
    print(f"  Parasitic harmonic line      : n={args.harmonic_n}")
    print(f"  Power steps used             : {len(drive_powers)}")
    print(f"  Harmonic-line bound points   : {int(np.sum(~harm_valid))} (worst-case/hatched)")
    print(f"  Max beta2 reached            : {beta2.max():.4f} rad")

    with np.errstate(divide='ignore', invalid='ignore'):
        max_frac_dev = 0.0
        for q in SIDEBANDS:
            lo, hi = bands[q]
            ideal_q = ideals[q]
            dev = np.maximum(np.abs(lo / ideal_q - 1.0), np.abs(hi / ideal_q - 1.0))
            dev = dev[np.isfinite(dev)]
            q_max_dev = float(dev.max()) if dev.size else float('nan')
            max_frac_dev = max(max_frac_dev, q_max_dev if np.isfinite(q_max_dev) else 0.0)
            print(f"  Max |eta_{q:+d} - J_{q}(b1)^2|/J_{q}(b1)^2 : {q_max_dev * 100:.2f} %")

    print(f"  Max deviation across all q   : {max_frac_dev * 100:.2f} %")
    print(f"  V1 where J1(beta1)^2 peaks   : {v1_ideal_peak:.4f} V_rms "
          f"({'within' if in_range else 'OUTSIDE'} measured range "
          f"[{V1.min():.4f}, {V1.max():.4f}] V_rms)")

    return fig2


# ── self-tests ────────────────────────────────────────────────────────────

def _selftest_power_conservation(n_trials=25, seed=0, tol=1e-6):
    rng = np.random.default_rng(seed)
    qs = np.arange(-25, 26)
    for _ in range(n_trials):
        beta1 = rng.uniform(0.0, 4.0)
        beta2 = rng.uniform(0.0, 4.0)
        phi = rng.uniform(0.0, 2 * np.pi)
        total = sum(
            float(np.abs(sideband_amp(int(q), np.array([beta1]), np.array([beta2]),
                                       np.array([phi]))[0, 0]) ** 2)
            for q in qs
        )
        assert abs(total - 1.0) < tol, (
            f"beta1={beta1:.4f}, beta2={beta2:.4f}, phi={phi:.4f}: sum={total:.8f}"
        )


def _selftest_beta2_zero(tol=1e-10):
    beta1_vals = np.linspace(0.0, 4.0, 41)
    zeros = np.zeros_like(beta1_vals)
    phi = np.array([0.0])
    for q in (1, -1):
        A = sideband_amp(q, beta1_vals, zeros, phi)[:, 0]
        expected = jv(q, beta1_vals)
        err = np.max(np.abs(np.abs(A) ** 2 - expected ** 2))
        assert err < tol, f"q={q}: max abs error {err:.3e}"


def _selftest_j1_max():
    beta_grid = np.linspace(1.5, 2.2, 200001)
    vals = jv(1, beta_grid) ** 2
    i = int(np.argmax(vals))
    assert abs(beta_grid[i] - 1.8412) < 1e-3, f"beta_at_max={beta_grid[i]:.4f}"
    assert abs(vals[i] - 0.3386) < 1e-3, f"val_at_max={vals[i]:.4f}"


def run_selftests():
    tests = [
        ("Power conservation (sum_q |A_q|^2 = 1)", _selftest_power_conservation),
        ("beta2=0 reduces to single-tone J1(beta1)^2", _selftest_beta2_zero),
        ("J1(beta)^2 max = 0.3386 @ beta=1.8412", _selftest_j1_max),
    ]
    print("Running self-tests...")
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            n_fail += 1
            print(f"  [FAIL] {name}: {e}")
    if n_fail:
        print(f"\n{n_fail}/{len(tests)} self-test(s) FAILED")
        sys.exit(1)
    print(f"\nAll {len(tests)} self-tests passed.")


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze signal-generator harmonic suppression and its effect on "
                    "phase-modulation sideband conversion efficiency.")
    p.add_argument('npz_path', nargs='?', default=(NPZ_PATH or None),
                    help="Path to the .npz harmonic-suppression dataset. "
                        "Defaults to the NPZ_PATH constant near the top of this file.")
    p.add_argument('--peak-window-khz', type=float, default=PEAK_WINDOW_KHZ,
                    help="Half-width, in kHz, of the peak search window around each line "
                        f"center (default: {PEAK_WINDOW_KHZ}).")
    p.add_argument('--min-snr-db', type=float, default=MIN_SNR_DB,
                    help="Minimum peak-to-floor SNR, in dB, for a point to count as a real "
                        f"measurement rather than an upper bound (default: {MIN_SNR_DB}).")
    p.add_argument('--vpi-fund', type=float, default=VPI_FUND,
                    help="V_pi at the fundamental f, in volts. Required (with --vpi-harm) "
                        "to run Part 2.")
    p.add_argument('--vpi-harm', type=float, default=VPI_HARM,
                    help="V_pi at the parasitic harmonic, in volts. Required (with "
                        "--vpi-fund) to run Part 2.")
    p.add_argument('--harmonic-n', type=int, default=HARMONIC_N,
                    help=f"Which harmonic multiplier is the parasitic tone (default: {HARMONIC_N}).")
    p.add_argument('--atten-db', type=float, default=ATTEN_DB,
                    help="Attenuation [dB] between the device plane and the ESA at f "
                        f"(default: {ATTEN_DB}).")
    p.add_argument('--atten2-db', type=float, default=ATTEN2_DB,
                    help="Same, at the harmonic frequency (default: same as --atten-db).")
    p.add_argument('--phase-deg', type=float, default=PHASE_DEG,
                    help="Optional fixed relative phase Phi [deg] to overlay on Figure 2.")
    p.add_argument('--selftest', action='store_true', default=SELFTEST,
                    help="Run internal self-tests and exit (no data file needed).")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.selftest:
        run_selftests()
        return

    if args.npz_path is None:
        build_parser().error(
            "npz_path is required unless --selftest is given "
            "(set NPZ_PATH near the top of this file, or pass a path on the command line)"
        )
    if (args.vpi_fund is None) != (args.vpi_harm is None):
        build_parser().error("--vpi-fund and --vpi-harm must be given together for Part 2")

    d = load_dataset(args.npz_path)
    labels = d['labels']
    idx_fund = _find_label_index(labels, 'n=1')

    peak_window_hz = args.peak_window_khz * 1e3
    peak_dbm, floor_dbm, valid, _peak_offset_hz = analyze_lines(
        d['offsets_hz'], d['window_hz'], d['spectra'], peak_window_hz, args.min_snr_db)

    bound_dbm = floor_dbm + args.min_snr_db
    display_dbm = np.where(valid, peak_dbm, bound_dbm)
    s_l_dbc = display_dbm - display_dbm[:, [idx_fund]]

    print(f"Loaded: {args.npz_path}")
    print(f"  Drive frequency        : {d['drive_freq'] / 1e9:.4f} GHz")
    print(f"  Power steps            : {len(d['drive_powers'])}")
    print(f"  Lines                  : {list(labels)}")
    print(f"  Bound (low-SNR) points : {int(np.sum(~valid))} / {peak_dbm.size}")

    csv_path = Path(args.npz_path).with_name(Path(args.npz_path).stem + '_line_powers.csv')
    write_csv(csv_path, d['drive_powers'], labels, peak_dbm, floor_dbm, valid, s_l_dbc)
    print(f"  Saved: {csv_path}")

    colors = build_color_map(labels)
    make_figure1(d['drive_powers'], labels, idx_fund, peak_dbm, valid, s_l_dbc, colors)

    if args.vpi_fund is not None and args.vpi_harm is not None:
        run_part2(d['drive_powers'], labels, peak_dbm, valid, args)

    plt.show()


if __name__ == '__main__':
    main()
