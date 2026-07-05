import numpy as np
from scipy.special import jv
from scipy.optimize import differential_evolution

# ── configuration ─────────────────────────────────────────────────────────────
BETA1_MAX = 2
BETA2_MAX = 3

# Truncation: sum runs k = -K_TRUNC ... K_TRUNC.
K_TRUNC = int(2 * max(BETA1_MAX, BETA2_MAX)) + 20

# 'maximize' → maximise  |A_{ORDER_A}|²
# 'suppress' → minimise  |A_{ORDER_A}|²
# 'ratio'    → maximise  |A_{ORDER_A}|² / Σ |A_{ORDER_B_i}|²
OPTIMIZE_MODE = 'ratio'
ORDER_A  = 1        # numerator order
ORDERS_B = [-1, 0]  # denominator orders (list); sum of their powers forms the denominator

FIX_PHI1 = True   # if True, phi1 is held at 0 and not optimized

# differential_evolution settings
DE_POPSIZE = 20
DE_MAXITER = 2000
DE_TOL     = 1e-10
DE_SEED    = 42
# ─────────────────────────────────────────────────────────────────────────────


def bessel_sum(beta1, beta2, phi1, phi2, order) -> complex:
    """A_order = sum_k J_{order-2k}(β1) J_k(β2) exp(i[(order-2k)φ1 + k φ2])"""
    k = np.arange(-K_TRUNC, K_TRUNC + 1)
    terms = (
        jv(order - 2 * k, beta1)
        * jv(k, beta2)
        * np.exp(1j * ((order - 2 * k) * phi1 + k * phi2))
    )
    return complex(terms.sum())


def objective(params):
    if FIX_PHI1:
        beta1, beta2, phi2 = params
        phi1 = 0.0
    else:
        beta1, beta2, phi1, phi2 = params

    pow_a = np.abs(bessel_sum(beta1, beta2, phi1, phi2, ORDER_A)) ** 2
    if OPTIMIZE_MODE == 'maximize':
        return -pow_a
    if OPTIMIZE_MODE == 'suppress':
        return pow_a
    # ratio: maximise |A_a|² / Σ|A_b_i|²
    pow_b = sum(np.abs(bessel_sum(beta1, beta2, phi1, phi2, ob)) ** 2 for ob in ORDERS_B)
    return -pow_a / (pow_b + 1e-30)


if __name__ == '__main__':
    bounds = (
        [(0.0, BETA1_MAX), (0.0, BETA2_MAX), (0.0, 2 * np.pi)]
        if FIX_PHI1
        else [(0.0, BETA1_MAX), (0.0, BETA2_MAX), (0.0, 2 * np.pi), (0.0, 2 * np.pi)]
    )

    if OPTIMIZE_MODE == 'ratio':
        denom_str = ' + '.join(f'|A_{o}|²' for o in ORDERS_B)
        mode_str = f'maximise |A_{ORDER_A}|² / ({denom_str})'
    elif OPTIMIZE_MODE == 'suppress':
        mode_str = f'suppress |A_{ORDER_A}|²'
    else:
        mode_str = f'maximise |A_{ORDER_A}|²'
    print(f'Mode             : {mode_str}')
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

    pow_a = np.abs(bessel_sum(b1, b2, p1, p2, ORDER_A)) ** 2
    print()
    if OPTIMIZE_MODE == 'ratio':
        pow_b_each = {o: np.abs(bessel_sum(b1, b2, p1, p2, o)) ** 2 for o in ORDERS_B}
        pow_b_total = sum(pow_b_each.values())
        ratio = pow_a / (pow_b_total + 1e-30)
        print(f'Ratio  = {ratio:.4f}  ({10 * np.log10(ratio):.2f} dB)')
        print(f'|A_{ORDER_A}|²  = {pow_a:.8f}  ({pow_a * 100:.4f} %)')
        for o, p in pow_b_each.items():
            print(f'|A_{o}|²  = {p:.8f}  ({p * 100:.4f} %)')
    else:
        print(f'|A_{ORDER_A}|²  = {pow_a:.8f}  ({pow_a * 100:.4f} %)')
    print(f'  beta1  = {b1:.6f}')
    print(f'  beta2  = {b2:.6f}')
    print(f'  phi1   = {p1:.6f} rad  ({np.degrees(p1):.3f} deg)')
    print(f'  phi2   = {p2:.6f} rad  ({np.degrees(p2):.3f} deg)')
