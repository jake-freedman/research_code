"""
Intensity modulation heatmap for a phase-modulator + ring-resonator system.

The input CW carrier passes through a phase modulator (depth beta, frequency Omega),
then through a ring resonator (Q_I, Q_E) detuned by Delta from the carrier.

After phase modulation the field has sidebands at omega_c + n*Omega with amplitude J_n(beta).
The ring multiplies each sideband by its complex transmission t(Delta - n*Omega).

The output intensity at harmonic k*Omega is:
    I_k = 2 * |sum_{n=-N}^{N}  conj(t(Delta - n*Omega))
                                * t(Delta - (n+k)*Omega)
                                * J_n(beta) * J_{n+k}(beta)|

Axes of the heatmap:
    x : phase-modulation depth beta  (radians)
    y : carrier detuning Delta  (units: Omega, the modulation frequency)
    color : I_k (normalized, pure phase modulation gives I_k = 0 without the ring)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv  # Bessel functions of the first kind

# ============================================================
# Global parameters — tune these
# ============================================================

HARMONIC_K  = 1       # which harmonic to compute: intensity modulation at k * Omega

Q_I         = 1e6     # intrinsic quality factor of the ring
Q_E         = 1e6     # extrinsic (coupling) quality factor of the ring
OMEGA_0_GHZ = 400e3   # ring resonance frequency (GHz) — 193 THz ~ 1550 nm

OMEGA_GHZ   = 2.5    # modulation frequency (GHz)

# Detuning scan (y-axis), in units of Omega (1.0 = one modulation frequency)
DELTA_MIN   = -4.0
DELTA_MAX   =  4.0
N_DELTA     = 500

# Modulation-depth scan (y-axis), in radians
BETA_MIN    = 0
BETA_MAX    = 5    # up to ~pi
N_BETA      = 500

# Truncation: sum from n = -N_TERMS to +N_TERMS
# For beta <= pi, J_n decays rapidly for |n| > ~5; N_TERMS = 30 is very safe.
N_TERMS     = 30

# Colorbar maximum (set to None to use the data maximum)
CBAR_MAX    = 1

# ============================================================
# Physics
# ============================================================

def ring_transmission(delta_norm, kappa_i_norm, kappa_e_norm):
    """
    Complex through-port amplitude of a ring resonator (coupled-mode theory).

    In normalized units (kappa_total = kappa_i + kappa_e = 1):
        t(delta) = [(kappa_i - kappa_e)/2 - i*delta] / [1/2 - i*delta]

    At resonance (delta=0): t = (kappa_i - kappa_e)/(kappa_i + kappa_e)
      -> 0 at critical coupling, negative (pi phase flip) when overcoupled.

    delta_norm   : (Delta - n*Omega) / kappa_total  — any broadcastable shape
    kappa_i_norm : kappa_i / kappa_total  (scalar)
    kappa_e_norm : kappa_e / kappa_total  (scalar)
    """
    numerator   = (kappa_i_norm - kappa_e_norm) / 2.0 - 1j * delta_norm
    denominator = 0.5                               - 1j * delta_norm
    return numerator / denominator


def compute_heatmap(delta_arr, beta_arr, k, omega_norm,
                    kappa_i_norm, kappa_e_norm, n_terms):
    """
    Vectorized evaluation of I_k over a (beta, delta) grid.

    The double sum is reorganized as a matrix product:
        A[beta_idx, n_idx] = J_n(beta) * J_{n+k}(beta)
        B[delta_idx, n_idx] = conj(t(Delta - n*Omega)) * t(Delta - (n+k)*Omega)
        total = A @ B.T   -> shape (N_beta, N_delta)

    Returns array of shape (N_beta, N_delta).
    """
    ns = np.arange(-n_terms, n_terms + 1)            # (N_n,)

    # --- Bessel coefficients ---  shape (N_beta, N_n)
    Jn  = jv(ns[None, :],       beta_arr[:, None])
    Jnk = jv((ns + k)[None, :], beta_arr[:, None])
    A   = Jn * Jnk                                   # (N_beta, N_n)

    # --- Ring detuning for each (delta, n) ---  shape (N_delta, N_n)
    det_n  = delta_arr[:, None] - ns[None, :]        * omega_norm
    det_nk = delta_arr[:, None] - (ns + k)[None, :]  * omega_norm

    t_n  = ring_transmission(det_n,  kappa_i_norm, kappa_e_norm)   # (N_delta, N_n)
    t_nk = ring_transmission(det_nk, kappa_i_norm, kappa_e_norm)

    B = np.conj(t_n) * t_nk                          # (N_delta, N_n)

    # --- Sum over n via matrix multiply ---
    total = A @ B.T                                   # (N_beta, N_delta)
    return 2.0 * np.abs(total)


# ============================================================
# Main
# ============================================================

def main():
    # Derived quantities
    Q_total_inv  = 1.0 / Q_I + 1.0 / Q_E
    kappa_i_norm = (1.0 / Q_I) / Q_total_inv    # = Q_total / Q_I
    kappa_e_norm = (1.0 / Q_E) / Q_total_inv    # = Q_total / Q_E
    kappa_ghz    = OMEGA_0_GHZ * Q_total_inv     # total linewidth in GHz
    omega_norm   = OMEGA_GHZ / kappa_ghz         # Omega / kappa_total

    # delta_arr is in units of Omega; convert to kappa_total for the physics
    delta_arr       = np.linspace(DELTA_MIN, DELTA_MAX, N_DELTA)
    delta_arr_kappa = delta_arr * omega_norm
    beta_arr        = np.linspace(BETA_MIN,  BETA_MAX,  N_BETA)

    print(
        f"Computing: k={HARMONIC_K}, Omega={OMEGA_GHZ:.2f} GHz, "
        f"kappa={kappa_ghz:.2f} GHz, Omega/kappa={omega_norm:.3f}, "
        f"Q_i={Q_I:.1e}, Q_e={Q_E:.1e}"
    )

    I_mod = compute_heatmap(
        delta_arr_kappa, beta_arr,
        HARMONIC_K, omega_norm,
        kappa_i_norm, kappa_e_norm,
        N_TERMS,
    )

    print(f"Max intensity modulation: {I_mod.max():.6f}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Transpose so beta is on x and detuning is on y
    extent = [BETA_MIN, BETA_MAX, DELTA_MIN, DELTA_MAX]
    im = ax.imshow(
        I_mod.T,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='viridis',
        vmin=0,
        vmax=CBAR_MAX,
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_box_aspect(1)

    plt.tight_layout()
    fname = rf"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media\ring_pm_intensity_modulation_k{HARMONIC_K}.png"
    plt.savefig(fname, dpi=200)
    plt.show()
    print(f"Saved {fname}")


if __name__ == '__main__':
    main()
