import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
BETA1_MAX = 5.0
BETA2_MAX = 5.0

# False: sum_k J_{2k+1}(b1) * J_{-k}(b2)   * exp(i[(2k+1)*phi1 - k*phi2])
# True:  sum_k J_{2k}(b1)   * J_{1-k}(b2)  * exp(i[2k*phi1 + (1-k)*phi2])
is_second_harmonic = False

N_BETA  = 100   # grid points along each beta axis
N_PHI1  = 100   # phi1 grid points (only used when FIX_PHI1=False)
N_PHI2  = 100   # phi2 grid points (only used when FIX_PHI2=False)
K_TRUNC = int(2 * max(BETA1_MAX, BETA2_MAX)) + 20

FIX_PHI1    = True    # if True, phi1 is held at PHI1_FIXED instead of optimized
PHI1_FIXED  = 0.0     # value to fix phi1 to (radians)

FIX_PHI2    = False   # if True, phi2 is held at PHI2_FIXED instead of optimized
PHI2_FIXED  = np.pi   # value to fix phi2 to (radians)
# ─────────────────────────────────────────────────────────────────────────────

# ── graphics style ────────────────────────────────────────────────────────────
axes_width_mm  = 120.0
axes_height_mm = 100.0
left_mm, right_mm  = 20.0, 15.0
bottom_mm, top_mm  = 15.0,  8.0
fig_w = left_mm + axes_width_mm + right_mm
fig_h = bottom_mm + axes_height_mm + top_mm
spine_linewidth    = 2.0
tick_width         = 2.0
tick_direction     = 'in'
axis_label_fontsize  = 10.0
tick_label_fontsize  =  8.0
# ─────────────────────────────────────────────────────────────────────────────

k    = np.arange(-K_TRUNC, K_TRUNC + 1)
b1   = np.linspace(0, BETA1_MAX, N_BETA)
b2   = np.linspace(0, BETA2_MAX, N_BETA)
phi1 = np.array([PHI1_FIXED]) if FIX_PHI1 else np.linspace(0, 2*np.pi, N_PHI1, endpoint=False)
phi2 = np.array([PHI2_FIXED]) if FIX_PHI2 else np.linspace(0, 2*np.pi, N_PHI2, endpoint=False)

if is_second_harmonic:
    jv1    = jv(2 * k,     b1[:, None])   # (N_B1, M)
    jv2    = jv(1 - k,     b2[:, None])   # (N_B2, M)
    order1 = 2 * k
    order2 = 1 - k
else:
    jv1    = jv(2 * k + 1, b1[:, None])   # (N_B1, M)
    jv2    = jv(-k,        b2[:, None])   # (N_B2, M)
    order1 = 2 * k + 1
    order2 = -k

# coeffs[i,j,m] = jv1[i,m] * jv2[j,m],  C is its (N_B1*N_B2, M) view
coeffs = jv1[:, None, :] * jv2[None, :, :]          # (N_B1, N_B2, M)
C      = coeffs.reshape(-1, len(k))                  # (N_B1*N_B2, M)

# phase2[p, m] — precomputed once
phase2 = np.exp(1j * order2[None, :] * phi2[:, None])  # (N_P2, M)

# loop over phi1, accumulating running maximum of |S|^2
Z         = np.zeros(N_BETA * N_BETA)
Z_phi1_idx = np.zeros(N_BETA * N_BETA, dtype=int)
Z_phi2_idx = np.zeros(N_BETA * N_BETA, dtype=int)

for i_p1, phi1_val in enumerate(phi1):
    C_mod    = C * np.exp(1j * order1 * phi1_val)    # (N_B1*N_B2, M)
    S        = C_mod @ phase2.T                       # (N_B1*N_B2, N_P2)
    S_pow    = np.abs(S) ** 2                         # (N_B1*N_B2, N_P2)
    best_p2  = np.argmax(S_pow, axis=-1)              # (N_B1*N_B2,)
    best_pow = S_pow[np.arange(len(S_pow)), best_p2]
    improved = best_pow > Z
    Z          = np.where(improved, best_pow, Z)
    Z_phi1_idx = np.where(improved, i_p1, Z_phi1_idx)
    Z_phi2_idx = np.where(improved, best_p2, Z_phi2_idx)

Z          = Z.reshape(N_BETA, N_BETA)                # (N_B1, N_B2)
Z_phi1_idx = Z_phi1_idx.reshape(N_BETA, N_BETA)
Z_phi2_idx = Z_phi2_idx.reshape(N_BETA, N_BETA)

# ── report global optimum ─────────────────────────────────────────────────────
best_i, best_j = np.unravel_index(np.argmax(Z), Z.shape)
opt_beta1 = b1[best_i]
opt_beta2 = b2[best_j]
opt_phi1  = phi1[Z_phi1_idx[best_i, best_j]]
opt_phi2  = phi2[Z_phi2_idx[best_i, best_j]]
opt_eff   = Z[best_i, best_j] * 100.0

print(f"Optimal β₁  = {opt_beta1:.4f}")
print(f"Optimal β₂  = {opt_beta2:.4f}")
if FIX_PHI1:
    print(f"φ₁          = {opt_phi1/np.pi:.4f}π  (fixed)")
else:
    print(f"Optimal φ₁  = {opt_phi1/np.pi:.4f}π")
if FIX_PHI2:
    print(f"φ₂          = {opt_phi2/np.pi:.4f}π  (fixed)")
else:
    print(f"Optimal φ₂  = {opt_phi2/np.pi:.4f}π")
print(f"Peak |S|²   = {opt_eff:.2f}%")
# ─────────────────────────────────────────────────────────────────────────────

B1, B2 = np.meshgrid(b1, b2, indexing='ij')

fig_w_in = fig_w / 25.4
fig_h_in = fig_h / 25.4
fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
fig.subplots_adjust(
    left   = left_mm   / fig_w,
    right  = 1 - right_mm  / fig_w,
    bottom = bottom_mm / fig_h,
    top    = 1 - top_mm    / fig_h,
)

im = ax.pcolormesh(B1, B2, Z, cmap='viridis', shading='auto')
cb = fig.colorbar(im, ax=ax)
phi1_str = rf'$\phi_1={PHI1_FIXED/np.pi:.3g}\pi$' if FIX_PHI1 else r'$\phi_1$ opt.'
phi2_str = rf'$\phi_2={PHI2_FIXED/np.pi:.3g}\pi$' if FIX_PHI2 else r'$\phi_2$ opt.'
opt_vars = ','.join(v for v, fixed in [('\\phi_1', FIX_PHI1), ('\\phi_2', FIX_PHI2)]
                    if not fixed)
cb_label = rf'$\max_{{{opt_vars}}}|S|^2$' if opt_vars else r'$|S|^2$'
cb.set_label(cb_label, fontsize=axis_label_fontsize)
cb.ax.tick_params(labelsize=tick_label_fontsize)

mode = 'second harmonic' if is_second_harmonic else 'fundamental'
ax.set_xlabel(r'$\beta_1$', fontsize=axis_label_fontsize)
ax.set_ylabel(r'$\beta_2$', fontsize=axis_label_fontsize)
ax.set_title(rf'{cb_label}  ({mode},  {phi1_str},  {phi2_str})',
             fontsize=axis_label_fontsize)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
for spine in ax.spines.values():
    spine.set_linewidth(spine_linewidth)
ax.tick_params(axis='both', direction=tick_direction,
               width=tick_width, labelsize=tick_label_fontsize)

out_path = Path(__file__).parent / 'bessel_sum_beta_heatmap.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
