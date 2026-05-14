import numpy as np
from scipy.special import jv
from scipy.optimize import differential_evolution

# ── configuration ─────────────────────────────────────────────────────────────
BETA1_MAX = 5.0   # upper bound on beta1
BETA2_MAX = 5.0   # upper bound on beta2

# False: sum_k J_{2k+1}(b1) * J_{-k}(b2)   * exp(i[(2k+1)*phi1 - k*phi2])
# True:  sum_k J_{2k}(b1)   * J_{1-k}(b2)  * exp(i[2k*phi1 + (1-k)*phi2])
is_second_harmonic = False

# Truncation: sum runs k = -K_TRUNC ... K_TRUNC.
# J_n(x) is negligible for |n| >> x, so the default is conservative.
K_TRUNC = int(2 * max(BETA1_MAX, BETA2_MAX)) + 20

# differential_evolution settings
DE_POPSIZE = 20     # population size multiplier
DE_MAXITER = 2000   # max generations
DE_TOL     = 1e-10  # convergence tolerance
DE_SEED    = 42

FIX_PHI1 = True   # if True, phi1 is held at 0 and not optimized
# ─────────────────────────────────────────────────────────────────────────────


def bessel_sum(beta1, beta2, phi1, phi2) -> complex:
    k = np.arange(-K_TRUNC, K_TRUNC + 1)
    if is_second_harmonic:
        terms = (
            jv(2 * k, beta1)
            * jv(1 - k, beta2)
            * np.exp(1j * (2 * k * phi1 + (1 - k) * phi2))
        )
    else:
        terms = (
            jv(2 * k + 1, beta1)
            * jv(-k, beta2)
            * np.exp(1j * ((2 * k + 1) * phi1 - k * phi2))
        )
    return terms.sum()


def objective(params):
    if FIX_PHI1:
        beta1, beta2, phi2 = params
        return -np.abs(bessel_sum(beta1, beta2, 0.0, phi2))**2
    return -np.abs(bessel_sum(*params))**2


if __name__ == '__main__':
    if FIX_PHI1:
        bounds = [
            (0.0, BETA1_MAX),
            (0.0, BETA2_MAX),
            (0.0, 2 * np.pi),
        ]
    else:
        bounds = [
            (0.0, BETA1_MAX),
            (0.0, BETA2_MAX),
            (0.0, 2 * np.pi),
            (0.0, 2 * np.pi),
        ]

    mode = 'second harmonic (2k, 1-k)' if is_second_harmonic else 'fundamental (2k+1, -k)'
    print(f'Mode             : {mode}')
    print(f'phi1 fixed at 0  : {FIX_PHI1}')
    print(f'Truncation order : K = {K_TRUNC}  ({2*K_TRUNC+1} terms)')
    print(f'Beta bounds      : beta1 in [0, {BETA1_MAX}],  beta2 in [0, {BETA2_MAX}]')
    print('Running differential evolution ...\n')

    result = differential_evolution(
        objective,
        bounds,
        seed=DE_SEED,
        maxiter=DE_MAXITER,
        tol=DE_TOL,
        popsize=DE_POPSIZE,
        polish=True,
        disp=True,
    )

    if FIX_PHI1:
        b1, b2, p2 = result.x
        p1 = 0.0
    else:
        b1, b2, p1, p2 = result.x
    S = bessel_sum(b1, b2, p1, p2)

    print()
    print(f'Optimal |S|² = {np.abs(S)**2:.8f}')
    print(f'Optimal |S|  = {np.abs(S):.8f}')
    print(f'  beta1      = {b1:.6f}')
    print(f'  beta2      = {b2:.6f}')
    print(f'  phi1       = {p1:.6f} rad  ({np.degrees(p1):.3f} deg)')
    print(f'  phi2       = {p2:.6f} rad  ({np.degrees(p2):.3f} deg)')
    print(f'  S          = {S.real:.6f} + {S.imag:.6f}i')
    print(f'  angle(S)   = {np.angle(S, deg=True):.3f} deg')
