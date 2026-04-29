"""
Single phase modulator driven by K tones.

Physical model
--------------
  E(t) = exp(i * sum_{k=1}^{K} beta_k * sin(k*Omega*t + theta_k))

Tunable parameters: beta_k (modulation depth) and theta_k (RF phase) per tone.

Two spectrum computation methods
---------------------------------
  "fft"  : sample E(t) over one fundamental period, take FFT.
           Fast, simple, accuracy controlled by n_fft (default 8192).

  "conv" : Jacobi-Anger on each tone, then sequentially convolve the
           resulting sparse spectra.  Exact to machine precision (up to
           Bessel truncation at ±n_max per tone).  Tone k contributes
           amplitude J_n(beta_k)*exp(i*n*theta_k) at harmonic n*k.
           Total output range: ±K(K+1)/2 * n_max.
"""

import numpy as np
from scipy.optimize import basinhopping
from scipy.special import jv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Spectrum computation
# ---------------------------------------------------------------------------

def compute_multitone_spectrum_fft(
    betas,
    thetas,
    n_max: int = 30,
    n_fft: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the optical spectrum via time-domain FFT.

    Samples E(t) at n_fft equally spaced points over one fundamental period
    [0, 2*pi), then extracts harmonics -n_max ... n_max from the FFT.

    Parameters
    ----------
    betas : array-like, length K
        Modulation depths [beta_1, ..., beta_K] (rad).
    thetas : array-like, length K
        RF phases [theta_1, ..., theta_K] (rad).
    n_max : int
        Harmonics returned span [-n_max, n_max].
    n_fft : int
        FFT size.  Must satisfy n_fft > 2*n_max.  Accuracy improves with
        larger n_fft; 8192 gives < 1e-10 error for beta_k <= 10, n_max = 30.

    Returns
    -------
    harmonics : np.ndarray, shape (2*n_max+1,)
    amplitudes : np.ndarray, shape (2*n_max+1,), complex
    """
    betas  = np.asarray(betas,  dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    if len(betas) != len(thetas):
        raise ValueError("betas and thetas must have the same length")

    t       = np.linspace(0, 2 * np.pi, n_fft, endpoint=False)
    phase_t = np.zeros(n_fft)
    for k_idx, (b, th) in enumerate(zip(betas, thetas)):
        phase_t += b * np.sin((k_idx + 1) * t + th)

    fft_raw   = np.fft.fft(np.exp(1j * phase_t)) / n_fft
    harmonics = np.arange(-n_max, n_max + 1)
    amplitudes = fft_raw[harmonics % n_fft]

    return harmonics, amplitudes


def compute_multitone_spectrum_conv(
    betas,
    thetas,
    n_max: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the optical spectrum via Jacobi-Anger + sparse convolution.

    For tone k: exp(i*beta_k*sin(k*Omega*t + theta_k))
              = sum_n J_n(beta_k) * exp(i*n*theta_k) * exp(i*n*k*Omega*t)

    So tone k contributes amplitude J_n(beta_k)*exp(i*n*theta_k) at harmonic
    n*k.  The total spectrum is the sequential convolution of the K sparse
    arrays.  The output harmonic range is ±K*(K+1)/2 * n_max.

    Parameters
    ----------
    betas : array-like, length K
    thetas : array-like, length K
    n_max : int
        Bessel truncation order per tone.

    Returns
    -------
    harmonics : np.ndarray
        Spans [-K*(K+1)//2 * n_max, K*(K+1)//2 * n_max].
    amplitudes : np.ndarray, complex
    """
    betas  = np.asarray(betas,  dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    K      = len(betas)
    if K != len(thetas):
        raise ValueError("betas and thetas must have the same length")

    n_range    = np.arange(-n_max, n_max + 1)
    harmonics  = np.array([0])
    amplitudes = np.array([1.0 + 0j])

    for k_idx, (beta, theta) in enumerate(zip(betas, thetas)):
        k = k_idx + 1

        # Jacobi-Anger coefficients J_n(beta)*exp(i*n*theta) for n in n_range
        bessel_coeffs = jv(n_range, beta) * np.exp(1j * n_range * theta)

        # Sparse array over [-k*n_max, k*n_max]; nonzero only at multiples of k
        tone_size = 2 * k * n_max + 1
        tone_amps = np.zeros(tone_size, dtype=complex)
        for n_idx, n in enumerate(n_range):
            tone_amps[k * int(n) + k * n_max] += bessel_coeffs[n_idx]

        amplitudes = np.convolve(amplitudes, tone_amps)
        harmonics  = np.arange(harmonics[0] - k * n_max,
                               harmonics[-1] + k * n_max + 1)

    return harmonics, amplitudes


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

def optimize_multitone(
    wanted_harmonic: int = 1,
    beta_max: float = 5.0,
    n_max: int = 30,
    n_tones: int = 3,
    n_iter: int = 200,
    seed: int | None = None,
    x0: np.ndarray | None = None,
    objective: str = "power",
    method: str = "fft",
    n_fft: int = 8192,
) -> dict:
    """
    Basin-hopping optimisation over modulation depths and RF phases.

    Parameter vector: x = [beta_1, ..., beta_K, theta_1, ..., theta_K]

    Parameters
    ----------
    wanted_harmonic : int
        Target optical sideband index.
    beta_max : float
        Upper bound on each beta_k (rad).
    n_max : int
        Harmonic truncation (FFT: returned range; conv: Bessel order per tone).
    n_tones : int
        Number of RF tones K.
    n_iter : int
        Basin-hopping iterations.
    seed : int or None
        Random seed.
    x0 : np.ndarray or None
        Warm-start parameter vector, length 2*n_tones.
    objective : str
        "power" : maximise |A_wanted|^2.
        "ratio" : minimise (sum unwanted) / |A_wanted|^2.
    method : str
        "fft"  : use FFT spectrum computation.
        "conv" : use Jacobi-Anger convolution.
    n_fft : int
        FFT size (only used when method="fft").

    Returns
    -------
    dict with keys:
        betas, thetas         : optimised parameters (lists of float)
        ratio                 : unwanted/wanted power ratio
        wanted_power          : |A_wanted|^2
        harmonics, amplitudes : output spectrum
        opt_result            : scipy OptimizeResult
    """
    if objective not in ("power", "ratio"):
        raise ValueError(f"objective must be 'power' or 'ratio', got {objective!r}")
    if method not in ("fft", "conv"):
        raise ValueError(f"method must be 'fft' or 'conv', got {method!r}")

    rng = np.random.default_rng(seed)

    # Precompute wanted harmonic index (stable across all evaluations)
    if method == "fft":
        _wanted_idx = int(wanted_harmonic + n_max)   # harmonics = [-n_max, ..., n_max]
    else:
        total_range = n_tones * (n_tones + 1) // 2 * n_max
        _wanted_idx = int(total_range + wanted_harmonic)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _obj(x):
        b  = x[:n_tones]
        th = x[n_tones:]
        if method == "fft":
            _, amps = compute_multitone_spectrum_fft(b, th, n_max, n_fft)
        else:
            _, amps = compute_multitone_spectrum_conv(b, th, n_max)
        power        = np.abs(amps) ** 2
        wanted_power = power[_wanted_idx]
        if objective == "power":
            return -wanted_power
        if wanted_power < 1e-30:
            return 1e30
        return (power.sum() - wanted_power) / wanted_power

    # ------------------------------------------------------------------
    # Bounds and step
    # ------------------------------------------------------------------
    lower = np.concatenate([np.zeros(n_tones),        np.zeros(n_tones)])
    upper = np.concatenate([np.full(n_tones, beta_max), np.full(n_tones, 2 * np.pi)])

    beta_step  = beta_max * 0.3
    phase_step = np.pi

    class _ScaledStep:
        def __init__(self, stepsize=1.0):
            self.stepsize = stepsize

        def __call__(self, x):
            x = x.copy()
            x[:n_tones] += rng.uniform(-beta_step, beta_step, size=n_tones) * self.stepsize
            x[:n_tones]  = np.clip(x[:n_tones], 0.0, beta_max)
            x[n_tones:] += rng.uniform(-phase_step, phase_step, size=n_tones) * self.stepsize
            x[n_tones:]  = x[n_tones:] % (2 * np.pi)
            return x

    # ------------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------------
    if x0 is None:
        x0 = np.zeros(2 * n_tones)
        x0[:n_tones] = rng.uniform(0.5, min(2.5, beta_max), size=n_tones)
        x0[n_tones:] = rng.uniform(0, 2 * np.pi, size=n_tones)
    else:
        x0 = np.asarray(x0, dtype=float).copy()

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": list(zip(lower, upper)),
    }

    opt = basinhopping(
        _obj,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=n_iter,
        take_step=_ScaledStep(),
        seed=rng,
    )

    x_best      = opt.x
    betas_opt   = list(x_best[:n_tones])
    thetas_opt  = list(x_best[n_tones:])

    if method == "fft":
        harmonics, amplitudes = compute_multitone_spectrum_fft(betas_opt, thetas_opt, n_max, n_fft)
    else:
        harmonics, amplitudes = compute_multitone_spectrum_conv(betas_opt, thetas_opt, n_max)

    power        = np.abs(amplitudes) ** 2
    wanted_power = float(power[_wanted_idx])
    total_power  = float(power.sum())
    ratio        = (total_power - wanted_power) / wanted_power if wanted_power > 1e-30 else np.inf

    return {
        "betas":        betas_opt,
        "thetas":       thetas_opt,
        "ratio":        ratio,
        "wanted_power": wanted_power,
        "harmonics":    harmonics,
        "amplitudes":   amplitudes,
        "opt_result":   opt,
    }
