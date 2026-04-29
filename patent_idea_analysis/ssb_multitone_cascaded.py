"""
Two cascaded phase modulators, each driven by K tones, separated by a
dispersive element.

Physical model
--------------
  PM1(betas1, thetas1) -> Dispersion(phi_n) -> PM2(betas2, thetas2)

Each PM applies the transfer function:
  C_n = FT{ exp(i * sum_{k=1}^{K} beta_k * sin(k*Omega*t + theta_k)) }[n]

which is computed via FFT or Jacobi-Anger convolution (clipped to ±n_max).
After PM1 the spectrum spans [-n_max, n_max]; after PM2 it spans [-2*n_max, 2*n_max].

Parameter layout (optimiser vector x)
--------------------------------------
  x[0      : K    ] = betas1   (PM1 modulation depths)
  x[K      : 2K   ] = betas2   (PM2 modulation depths)
  x[2K     : 3K   ] = thetas1  (PM1 RF phases)
  x[3K     : 4K   ] = thetas2  (PM2 RF phases)
  x[4K     : 4K+P ] = phi_params (dispersion; P = poly_order+1 or 2*n_max+1)
"""

import numpy as np
from scipy.optimize import basinhopping
from scipy.special import jv

from ssb_multitone import (
    compute_multitone_spectrum_fft,
    compute_multitone_spectrum_conv,
)


# ---------------------------------------------------------------------------
# Core spectrum computation
# ---------------------------------------------------------------------------

def _pm_transfer(betas, thetas, n_max, method, n_fft):
    """Return PM transfer spectrum clipped to harmonics [-n_max, n_max]."""
    if method == "fft":
        _, C = compute_multitone_spectrum_fft(betas, thetas, n_max, n_fft)
    else:
        h, C_full = compute_multitone_spectrum_conv(betas, thetas, n_max)
        mask = (h >= -n_max) & (h <= n_max)
        C = C_full[mask]
    return C


def compute_cascaded_multitone_spectrum(
    betas1,
    thetas1,
    phi_profile,
    betas2,
    thetas2,
    n_max: int = 30,
    method: str = "fft",
    n_fft: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the output optical spectrum for PM1 -> Dispersion -> PM2,
    where each PM is driven by a multi-tone RF signal.

    Parameters
    ----------
    betas1, betas2 : array-like, length K
        Modulation depths for PM1 and PM2 (rad).
    thetas1, thetas2 : array-like, length K
        RF phases for PM1 and PM2 (rad).
    phi_profile : array-like or callable
        Dispersive phase shift at each harmonic in [-n_max, n_max].
        - array-like: length 2*n_max+1, ordered from -n_max to n_max.
        - callable:   phi(n) -> float.
    n_max : int
        PM transfer spectrum is clipped to ±n_max; final output spans ±2*n_max.
    method : str
        "fft"  : compute PM transfer spectra via FFT.
        "conv" : compute PM transfer spectra via Jacobi-Anger convolution.
    n_fft : int
        FFT size (only used when method="fft").

    Returns
    -------
    harmonics : np.ndarray, shape (4*n_max+1,)
        Integer harmonic indices spanning [-2*n_max, 2*n_max].
    amplitudes : np.ndarray, shape (4*n_max+1,), complex
    """
    harmonics_mid = np.arange(-n_max, n_max + 1)

    # PM1: CW (delta at 0) convolved with C1 = C1 directly
    C1         = _pm_transfer(betas1, thetas1, n_max, method, n_fft)
    amplitudes = C1.copy()

    # Dispersion
    if callable(phi_profile):
        phase_arr = np.array([phi_profile(int(k)) for k in harmonics_mid], dtype=float)
    else:
        phase_arr = np.asarray(phi_profile, dtype=float)
        if phase_arr.shape != harmonics_mid.shape:
            raise ValueError(
                f"phi_profile has length {len(phase_arr)} but needs {len(harmonics_mid)} "
                f"(2*n_max+1 = {2*n_max+1})"
            )
    amplitudes = amplitudes * np.exp(1j * phase_arr)

    # PM2: convolve with C2
    C2         = _pm_transfer(betas2, thetas2, n_max, method, n_fft)
    amplitudes = np.convolve(amplitudes, C2)
    harmonics  = np.arange(-2 * n_max, 2 * n_max + 1)

    return harmonics, amplitudes


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

def optimize_cascaded_multitone(
    wanted_harmonic: int = 1,
    beta_max: float = 5.0,
    n_max: int = 30,
    n_tones: int = 3,
    poly_order: int | None = None,
    n_iter: int = 200,
    seed: int | None = None,
    x0: np.ndarray | None = None,
    objective: str = "power",
    method: str = "fft",
    n_fft: int = 8192,
) -> dict:
    """
    Basin-hopping optimisation over PM1, PM2 drives and dispersion phase.

    Parameters
    ----------
    wanted_harmonic : int
        Target optical harmonic index.
    beta_max : float
        Upper bound on each individual beta_k (rad).
    n_max : int
        PM transfer spectrum / harmonic truncation order.
    n_tones : int
        Number of RF tones K (same for PM1 and PM2).
    poly_order : int or None
        Dispersion parameterisation: None = free phase per harmonic;
        integer = polynomial phi(n) = sum_p c_p * n^p of that order.
    n_iter : int
        Basin-hopping iterations.
    seed : int or None
        Random seed.
    x0 : np.ndarray or None
        Warm-start vector, length 4*n_tones + n_phi.
    objective : str
        "power" : maximise |A_wanted|^2.
        "ratio" : minimise (total - wanted) / wanted power.
    method : str
        "fft" or "conv".
    n_fft : int
        FFT size (only used when method="fft").

    Returns
    -------
    dict with keys:
        betas1, thetas1  : PM1 parameters (lists of float)
        betas2, thetas2  : PM2 parameters (lists of float)
        phi_params       : raw dispersion parameters (np.ndarray)
        phi_profile      : evaluated dispersion array over [-n_max, n_max]
        ratio            : unwanted/wanted power ratio
        wanted_power     : |A_wanted|^2
        harmonics        : np.ndarray
        amplitudes       : np.ndarray (complex)
        opt_result       : scipy OptimizeResult
    """
    if objective not in ("power", "ratio"):
        raise ValueError(f"objective must be 'power' or 'ratio', got {objective!r}")
    if method not in ("fft", "conv"):
        raise ValueError(f"method must be 'fft' or 'conv', got {method!r}")

    rng          = np.random.default_rng(seed)
    stage_range  = np.arange(-n_max, n_max + 1)   # dispersion harmonic range

    if poly_order is None:
        n_phi     = 2 * n_max + 1
        phi_lower = np.zeros(n_phi)
        phi_upper = np.full(n_phi, 2 * np.pi)
    else:
        n_phi     = poly_order + 1
        phi_lower = np.full(n_phi, -np.inf)
        phi_upper = np.full(n_phi,  np.inf)

    # Parameter layout offsets
    K         = n_tones
    off_b1    = 0
    off_b2    = K
    off_th1   = 2 * K
    off_th2   = 3 * K
    off_phi   = 4 * K
    n_params  = 4 * K + n_phi

    # Precomputed index of wanted harmonic in output (spans -2*n_max .. 2*n_max)
    _wanted_idx = 2 * n_max + wanted_harmonic

    # ------------------------------------------------------------------
    # Dispersion evaluation
    # ------------------------------------------------------------------
    def phi_evaluated(x):
        raw = x[off_phi: off_phi + n_phi]
        if poly_order is None:
            return raw
        return np.polyval(raw[::-1], stage_range)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _obj(x):
        b1  = x[off_b1  : off_b1  + K]
        b2  = x[off_b2  : off_b2  + K]
        th1 = x[off_th1 : off_th1 + K]
        th2 = x[off_th2 : off_th2 + K]
        phi = phi_evaluated(x)
        _, amps      = compute_cascaded_multitone_spectrum(
            b1, th1, phi, b2, th2, n_max=n_max, method=method, n_fft=n_fft
        )
        power        = np.abs(amps) ** 2
        wanted_power = power[_wanted_idx]
        if objective == "power":
            return -wanted_power
        if wanted_power < 1e-30:
            return 1e30
        return (power.sum() - wanted_power) / wanted_power

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------
    lower = np.concatenate([
        np.zeros(2 * K),            # betas1, betas2
        np.zeros(2 * K),            # thetas1, thetas2
        phi_lower,
    ])
    upper = np.concatenate([
        np.full(2 * K, beta_max),   # betas1, betas2
        np.full(2 * K, 2 * np.pi), # thetas1, thetas2
        phi_upper,
    ])

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    beta_step  = beta_max * 0.3
    phase_step = np.pi

    class _ScaledStep:
        def __init__(self, stepsize=1.0):
            self.stepsize = stepsize

        def __call__(self, x):
            x = x.copy()
            # betas
            x[off_b1:off_b1+K]   += rng.uniform(-beta_step, beta_step, K) * self.stepsize
            x[off_b2:off_b2+K]   += rng.uniform(-beta_step, beta_step, K) * self.stepsize
            x[off_b1:off_b2+K]    = np.clip(x[off_b1:off_b2+K], 0, beta_max)
            # thetas
            x[off_th1:off_th1+K] += rng.uniform(-phase_step, phase_step, K) * self.stepsize
            x[off_th2:off_th2+K] += rng.uniform(-phase_step, phase_step, K) * self.stepsize
            x[off_th1:off_th2+K]  = x[off_th1:off_th2+K] % (2 * np.pi)
            # dispersion
            x[off_phi:off_phi+n_phi] += rng.uniform(-phase_step, phase_step, n_phi) * self.stepsize
            if poly_order is None:
                x[off_phi:off_phi+n_phi] = x[off_phi:off_phi+n_phi] % (2 * np.pi)
            return x

    # ------------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------------
    if x0 is None:
        x0 = np.zeros(n_params)
        x0[off_b1:off_b1+K]   = rng.uniform(0.5, min(2.5, beta_max), K)
        x0[off_b2:off_b2+K]   = rng.uniform(0.5, min(2.5, beta_max), K)
        x0[off_th1:off_th1+K] = rng.uniform(0, 2 * np.pi, K)
        x0[off_th2:off_th2+K] = rng.uniform(0, 2 * np.pi, K)
        if poly_order is None:
            x0[off_phi:off_phi+n_phi] = rng.uniform(0, 2 * np.pi, n_phi)
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

    x_best = opt.x
    betas1_opt  = list(x_best[off_b1 : off_b1+K])
    betas2_opt  = list(x_best[off_b2 : off_b2+K])
    thetas1_opt = list(x_best[off_th1: off_th1+K])
    thetas2_opt = list(x_best[off_th2: off_th2+K])
    phi_raw     = x_best[off_phi: off_phi+n_phi].copy()
    phi_arr     = phi_evaluated(x_best)

    harmonics, amplitudes = compute_cascaded_multitone_spectrum(
        betas1_opt, thetas1_opt, phi_arr,
        betas2_opt, thetas2_opt,
        n_max=n_max, method=method, n_fft=n_fft,
    )

    power        = np.abs(amplitudes) ** 2
    wanted_power = float(power[_wanted_idx])
    total_power  = float(power.sum())
    ratio        = (total_power - wanted_power) / wanted_power if wanted_power > 1e-30 else np.inf

    return {
        "betas1":       betas1_opt,
        "thetas1":      thetas1_opt,
        "betas2":       betas2_opt,
        "thetas2":      thetas2_opt,
        "phi_params":   phi_raw,
        "phi_profile":  phi_arr,
        "ratio":        ratio,
        "wanted_power": wanted_power,
        "harmonics":    harmonics,
        "amplitudes":   amplitudes,
        "opt_result":   opt,
    }
