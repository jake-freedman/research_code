import numpy as np
from scipy.special import jv
from scipy.optimize import basinhopping
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def compute_optical_spectrum(
    beta1: float,
    beta2: float,
    phi1,
    n_max: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the E-field spectrum after: CW -> PM1(beta1) -> Dispersion(phi1) -> PM2(beta2)

    Uses Jacobi-Anger: e^{i*beta*sin(Omega*t)} = sum_n J_n(beta) * e^{i*n*Omega*t}

    After PM1 the nth harmonic has amplitude J_n(beta1).
    After dispersion each harmonic n picks up phase phi1(n).
    After PM2 the output at harmonic k is the convolution:
        A_k = sum_n  J_n(beta1) * exp(i*phi1_n) * J_{k-n}(beta2)

    Parameters
    ----------
    beta1 : float
        Modulation depth of first phase modulator (radians).
    beta2 : float
        Modulation depth of second phase modulator (radians).
    phi1 : array-like or callable
        Dispersive phase shift applied to the nth harmonic.
        - callable:   phi1(n) -> float, called for each n in [-n_max, n_max]
        - array-like: length 2*n_max+1, ordered n = -n_max ... n_max
    n_max : int
        Truncation order; harmonics beyond ±n_max are ignored.
        Rule of thumb: n_max > beta1 + beta2 + a few.

    Returns
    -------
    harmonics : np.ndarray, shape (4*n_max+1,)
        Integer harmonic indices k; physical frequency is omega + k*Omega.
    amplitudes : np.ndarray, shape (4*n_max+1,), complex
        Complex E-field amplitude at each harmonic, normalised so that
        the input CW field has amplitude 1 (i.e. E0 = 1).
    """
    n_range = np.arange(-n_max, n_max + 1)  # length 2*n_max+1

    # Resolve phi1 into an array over n_range
    if callable(phi1):
        phi1_arr = np.array([phi1(int(n)) for n in n_range], dtype=float)
    else:
        phi1_arr = np.asarray(phi1, dtype=float)
        if phi1_arr.shape != (2 * n_max + 1,):
            raise ValueError(
                f"phi1 array must have shape ({2*n_max+1},) for n_max={n_max}, "
                f"got {phi1_arr.shape}"
            )

    # Complex amplitudes after PM1 and dispersion: b_n = J_n(beta1) * exp(i*phi1_n)
    b = jv(n_range, beta1) * np.exp(1j * phi1_arr)

    # Amplitudes of PM2 sidebands: c_m = J_m(beta2)
    c = jv(n_range, beta2).astype(complex)

    # A_k = (b * c)[k+2*n_max]  via linear convolution
    # np.convolve(b, c) has length 4*n_max+1 and index offset -2*n_max
    amplitudes = np.convolve(b, c)
    harmonics = np.arange(-2 * n_max, 2 * n_max + 1)

    return harmonics, amplitudes


def compute_optical_spectrum_general(
    betas,
    phi_profiles,
    n_max: int = 30,
    truncate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalised spectrum computation for n phase modulators with n-1 dispersive
    elements between them:

        PM(beta[0]) -> phi[0] -> PM(beta[1]) -> phi[1] -> ... -> PM(beta[n-1])

    Each PM is applied by convolving the running spectrum with J_m(beta),
    m in [-n_max, n_max].  Each dispersion element multiplies harmonic k by
    exp(i * phi(k)).

    Without truncation the harmonic range grows by n_max at each PM stage,
    so the final output spans [-n*n_max, n*n_max].

    With truncate=True the spectrum is clipped back to [-n_max, n_max] after
    each PM stage (and after dispersion), keeping the array size constant at
    2*n_max+1 throughout.  This is appropriate when higher-order sidebands
    are not of interest and reduces computation for large n_stages.

    Parameters
    ----------
    betas : sequence of float
        Modulation depths [beta_1, beta_2, ..., beta_n].
    phi_profiles : sequence of (callable or array-like)
        n-1 phase profiles.  Each element is either:
        - callable:   phi(k) -> float, evaluated at the running harmonic indices
        - array-like: when truncate=False, length equals harmonics present at
                      that stage (2*stage*n_max+1); when truncate=True, always
                      length 2*n_max+1.
    n_max : int
        Truncation order per PM stage.
    truncate : bool
        If True, clip the spectrum to ±n_max after each PM+dispersion step.

    Returns
    -------
    harmonics : np.ndarray
        Integer harmonic indices.  Spans [-n*n_max, n*n_max] when truncate=False,
        or [-n_max, n_max] when truncate=True.
    amplitudes : np.ndarray, complex
        Complex E-field amplitude at each harmonic (input CW amplitude = 1).
    """
    betas = list(betas)
    phi_profiles = list(phi_profiles)
    n_stages = len(betas)
    if len(phi_profiles) != n_stages - 1:
        raise ValueError(
            f"Need exactly len(betas)-1={n_stages-1} phi_profiles, got {len(phi_profiles)}"
        )

    pm_range = np.arange(-n_max, n_max + 1)

    # Start: CW input, amplitude 1 at harmonic 0
    harmonics  = np.array([0])
    amplitudes = np.array([1.0 + 0j])

    for i, beta in enumerate(betas):
        # --- Apply PM: convolve with J_m(beta) ---
        pm_amps    = jv(pm_range, beta).astype(complex)
        amplitudes = np.convolve(amplitudes, pm_amps)
        harmonics  = np.arange(harmonics[0] + pm_range[0],
                               harmonics[-1] + pm_range[-1] + 1)

        # --- Truncate to ±n_max if requested ---
        if truncate:
            mask       = (harmonics >= -n_max) & (harmonics <= n_max)
            harmonics  = harmonics[mask]
            amplitudes = amplitudes[mask]

        # --- Apply dispersion (not after the last PM) ---
        if i < len(phi_profiles):
            phi = phi_profiles[i]
            if callable(phi):
                phase_arr = np.array([phi(int(k)) for k in harmonics], dtype=float)
            else:
                phase_arr = np.asarray(phi, dtype=float)
                if phase_arr.shape != harmonics.shape:
                    raise ValueError(
                        f"phi_profiles[{i}] has length {len(phase_arr)} but "
                        f"there are {len(harmonics)} harmonics at this stage"
                    )
            amplitudes = amplitudes * np.exp(1j * phase_arr)

    return harmonics, amplitudes


def plot_phase_profile(
    n_range: np.ndarray,
    phi: np.ndarray,
    width_mm: float = 150.0,
    height_mm: float = 60.0,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the dispersive phase profile phi1_n vs harmonic index n.

    Parameters
    ----------
    n_range : np.ndarray
        Integer harmonic indices (e.g. -n_max ... n_max).
    phi : np.ndarray
        Phase values in radians at each index.
    width_mm, height_mm : float
        Axes dimensions in mm.
    ax : plt.Axes, optional
        Existing axes to plot into.
    """
    mm_per_inch = 25.4
    font_props  = {"family": "Arial"}
    matplotlib.rcParams["font.family"] = "Arial"

    if ax is None:
        left_in   = 0.60
        right_in  = 0.15
        bottom_in = 0.45
        top_in    = 0.15
        ax_w_in = width_mm / mm_per_inch
        ax_h_in = height_mm / mm_per_inch
        fig_w = left_in + ax_w_in + right_in
        fig_h = bottom_in + ax_h_in + top_in
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes([
            left_in / fig_w,
            bottom_in / fig_h,
            ax_w_in / fig_w,
            ax_h_in / fig_h,
        ])
    else:
        fig = ax.get_figure()

    ax.plot(n_range, phi, color="steelblue", linewidth=1.5)
    ax.scatter(n_range, phi, color="steelblue", s=30, zorder=3)

    ax.set_xlabel("Harmonic index  $n$", fontsize=10, **font_props)
    ax.set_ylabel("Phase  $\\phi_n^{(1)}$  (rad)", fontsize=10, **font_props)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.tick_params(axis="both", which="both", direction="in", width=2, labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    return fig, ax


def plot_optical_spectrum(
    harmonics: np.ndarray,
    amplitudes: np.ndarray,
    width_mm: float = 150.0,
    height_mm: float = 60.0,
    n_display: int = 10,
    db: bool = False,
    db_floor: float = -60.0,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the optical power spectrum from compute_optical_spectrum output.

    Parameters
    ----------
    harmonics : np.ndarray
        Integer harmonic indices from compute_optical_spectrum.
    amplitudes : np.ndarray
        Complex E-field amplitudes from compute_optical_spectrum.
    width_mm : float
        Axes width in mm (default 150).
    height_mm : float
        Axes height in mm (default 60).
    n_display : int
        Show only harmonics in [-n_display, n_display] (default 10).
    db : bool
        If True, plot power in dB (10*log10) instead of linear (default False).
    db_floor : float
        Minimum dB value shown on the y-axis (default -60 dB).
    ax : plt.Axes, optional
        Existing axes to plot into. If None, a new figure is created.

    Returns
    -------
    fig, ax : plt.Figure, plt.Axes
    """
    mm_per_inch = 25.4
    font_props = {"family": "Arial"}

    matplotlib.rcParams["font.family"] = "Arial"

    power = np.abs(amplitudes) ** 2

    mask = (harmonics >= -n_display) & (harmonics <= n_display)
    h_plot = harmonics[mask]
    p_plot = power[mask]

    if db:
        y_plot  = 10.0 * np.log10(np.maximum(p_plot, 10 ** (db_floor / 10)))
        y_floor = db_floor
        y_ceil  = 0.0
        ylabel  = "Power  (dB)"
    else:
        y_plot  = p_plot
        y_floor = -0.1
        y_ceil  = 1.1
        ylabel  = "Power  $|A_k|^2$"

    if ax is None:
        left_in   = 0.65 if db else 0.55
        right_in  = 0.15
        bottom_in = 0.45
        top_in    = 0.15
        ax_w_in = width_mm / mm_per_inch
        ax_h_in = height_mm / mm_per_inch
        fig_w = left_in + ax_w_in + right_in
        fig_h = bottom_in + ax_h_in + top_in
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes([
            left_in / fig_w,
            bottom_in / fig_h,
            ax_w_in / fig_w,
            ax_h_in / fig_h,
        ])
    else:
        fig = ax.get_figure()

    # Colour each line/ball by optical frequency: red (low) → green → blue → violet (high)
    cmap   = matplotlib.colormaps["rainbow_r"]
    norm   = matplotlib.colors.Normalize(vmin=-n_display, vmax=n_display)
    colors = [cmap(norm(k)) for k in h_plot]

    if db:
        ax.vlines(h_plot, y_floor, y_plot, linewidth=2, colors=colors)
    else:
        ax.vlines(h_plot, 0, y_plot, linewidth=2, colors=colors)
    ax.scatter(h_plot, y_plot, s=80, c=colors, zorder=3, clip_on=False)

    ax.set_xlabel("Harmonic index  $k$  ($\\omega + k\\,\\Omega$)", fontsize=10, **font_props)
    ax.set_ylabel(ylabel, fontsize=10, **font_props)
    ax.set_xlim(-n_display - 0.5, n_display + 0.5)
    ax.set_ylim(y_floor, y_ceil)

    ax.grid()

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        width=2,
        labelsize=8,
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    return fig, ax


def optimize_ssb(
    wanted_harmonic: int = 1,
    beta_max: float = 10.0,
    n_max: int = 30,
    n_stages: int = 2,
    poly_order: int | None = None,
    n_iter: int = 200,
    seed: int | None = None,
    x0: np.ndarray | None = None,
    objective: str = "ratio",
) -> dict:
    """
    Use basin-hopping to minimise unwanted/wanted harmonic power ratio.

    Models n_stages phase modulators with n_stages-1 dispersive elements
    between them, using compute_optical_spectrum_general.

    Parameter layout: x = [beta_0, ..., beta_{n-1}, *phi_params_0, ..., *phi_params_{n-2}]
    Each phi block has n_phase values: 2*n_max+1 (free) or poly_order+1 (polynomial).

    Objectives
    ----------
    "ratio" (default)
        Minimise  (sum_{k != wanted} |A_k|^2) / |A_wanted|^2.
        Drives unwanted sideband power down relative to the wanted one.
    "power"
        Maximise  |A_wanted|^2  (equivalently minimise  -|A_wanted|^2).
        Ignores unwanted sidebands; useful when raw conversion efficiency
        matters more than spectral purity.

    Parameters
    ----------
    wanted_harmonic : int
        Target harmonic index (default 1).
    beta_max : float
        Upper bound on each modulation depth in radians (default 10).
    n_max : int
        Harmonic truncation order per PM stage (default 30).
    n_stages : int
        Number of phase modulators (default 2).
    poly_order : int or None
        If None, each phi_n is a free parameter in [0, 2*pi].
        If an integer, phi(n) = sum_p c_p * n^p with poly_order+1 free coefficients.
    n_iter : int
        Number of basin-hopping iterations (default 200).
    seed : int or None
        Random seed for reproducibility.
    x0 : np.ndarray or None
        Optional warm-start parameter vector.
    objective : str
        "ratio" or "power" (default "ratio").

    Returns
    -------
    result : dict with keys
        'betas'           : list of float, one per PM stage
        'phi_params_list' : list of np.ndarray, one per dispersion stage
        'phi_profiles'    : list of callables phi(k) -> float
        'ratio'           : float, unwanted/wanted power ratio at best solution
        'wanted_power'    : float, |A_wanted|^2 at best solution
        'harmonics'       : np.ndarray
        'amplitudes'      : np.ndarray (complex)
        'opt_result'      : scipy OptimizeResult
    """
    rng    = np.random.default_rng(seed)
    n_disp = n_stages - 1

    # For n_stages > 2 the spectrum is truncated back to ±n_max after each stage
    # so all dispersion profiles are the same size (2*n_max+1).
    # For n_stages == 2 no truncation is needed; the single dispersion element
    # sees ±n_max (growing range equals ±n_max after the first PM).
    truncate = n_stages > 2
    stage_ranges = [np.arange(-n_max, n_max + 1)] * n_disp

    if poly_order is None:
        n_phase    = 2 * n_max + 1
        n_phases   = [n_phase] * n_disp
        phi_lowers = [np.zeros(n_phase)]           * n_disp
        phi_uppers = [np.full(n_phase, 2 * np.pi)] * n_disp
    else:
        n_phases   = [poly_order + 1] * n_disp
        phi_lowers = [np.full(poly_order + 1, -np.inf)] * n_disp
        phi_uppers = [np.full(poly_order + 1,  np.inf)] * n_disp

    n_phase_total  = sum(n_phases)
    phase_offsets  = [n_stages + sum(n_phases[:s]) for s in range(n_disp)]

    lower = np.concatenate([np.zeros(n_stages)]          + phi_lowers)
    upper = np.concatenate([np.full(n_stages, beta_max)] + phi_uppers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def phi_evaluated(x, stage):
        raw = x[phase_offsets[stage] : phase_offsets[stage] + n_phases[stage]]
        if poly_order is None:
            return raw                                       # length matches stage_ranges[stage]
        return np.polyval(raw[::-1], stage_ranges[stage])   # evaluate polynomial

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    if objective not in ("ratio", "power"):
        raise ValueError(f"objective must be 'ratio' or 'power', got {objective!r}")

    if truncate:
        _harmonics_ref = np.arange(-n_max, n_max + 1)
    else:
        _harmonics_ref = np.arange(-n_stages * n_max, n_stages * n_max + 1)
    _wanted_idx = int(np.where(_harmonics_ref == wanted_harmonic)[0][0])

    def _obj(x):
        betas_x  = x[:n_stages]
        phi_arrs = [phi_evaluated(x, s) for s in range(n_disp)]
        _, amps      = compute_optical_spectrum_general(betas_x, phi_arrs, n_max=n_max,
                                                        truncate=truncate)
        power        = np.abs(amps) ** 2
        wanted_power = power[_wanted_idx]
        if objective == "power":
            return -wanted_power
        if wanted_power < 1e-30:
            return 1e30
        return (power.sum() - wanted_power) / wanted_power

    # ------------------------------------------------------------------
    # Custom step
    # ------------------------------------------------------------------
    beta_step  = beta_max * 0.3
    phase_step = np.pi

    class _ScaledStep:
        def __init__(self, stepsize=1.0):
            self.stepsize = stepsize

        def __call__(self, x):
            x = x.copy()
            x[:n_stages] += rng.uniform(-beta_step, beta_step, size=n_stages) * self.stepsize
            x[:n_stages]  = np.clip(x[:n_stages], 0.0, beta_max)
            x[n_stages:] += rng.uniform(-phase_step, phase_step, size=n_phase_total) * self.stepsize
            if poly_order is None:
                x[n_stages:] = x[n_stages:] % (2 * np.pi)
            return x

    # ------------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------------
    if x0 is None:
        x0 = np.zeros(n_stages + n_phase_total)
        x0[:n_stages] = rng.uniform(0.5, min(2.5, beta_max), size=n_stages)
    else:
        x0 = np.asarray(x0, dtype=float).copy()

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": list(zip(
            np.where(np.isinf(lower), -1e6, lower),
            np.where(np.isinf(upper),  1e6, upper),
        )),
    }

    opt = basinhopping(
        _obj,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=n_iter,
        take_step=_ScaledStep(),
        seed=rng,
    )

    x_best    = opt.x
    betas_opt = list(x_best[:n_stages])

    phi_params_list = []
    phi_profiles    = []
    phi_arrs_opt    = []
    for s in range(n_disp):
        raw = x_best[phase_offsets[s] : phase_offsets[s] + n_phases[s]].copy()
        phi_params_list.append(raw)
        if poly_order is None:
            phi_arrs_opt.append(raw.copy())
            # Callable: index into the free array using stage's harmonic range
            _lo = int(stage_ranges[s][0])
            phi_profiles.append(lambda k, _p=raw, _l=_lo: _p[int(k) - _l])
        else:
            phi_arrs_opt.append(np.polyval(raw[::-1], stage_ranges[s]))
            phi_profiles.append(lambda k, _c=raw: float(np.polyval(_c[::-1], k)))

    harmonics, amplitudes = compute_optical_spectrum_general(
        betas_opt, phi_arrs_opt, n_max=n_max, truncate=truncate
    )

    power         = np.abs(amplitudes) ** 2
    wanted_power  = float(power[_wanted_idx])
    total_power   = float(power.sum())
    ratio         = (total_power - wanted_power) / wanted_power if wanted_power > 1e-30 else np.inf

    return {
        "betas":           betas_opt,
        "phi_params_list": phi_params_list,
        "phi_profiles":    phi_profiles,
        "ratio":           ratio,
        "wanted_power":    wanted_power,
        "harmonics":       harmonics,
        "amplitudes":      amplitudes,
        "opt_result":      opt,
    }
