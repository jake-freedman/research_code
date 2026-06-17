import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
BETA1_MAX  = 5.0    # beta_1 axis range  (tone at TONE1_MULT * Omega)
BETA2_MAX  = 5.0    # beta_2 axis range  (tone at TONE2_MULT * Omega)
N_BETA     = 150    # grid points per beta axis
N_PHI      = 150    # phase grid points per axis for psi1/psi2 optimization
K_TRUNC    = 25     # search |n1|,|n2| up to this to find lattice solutions

TONE1_MULT = 1      # first tone is at TONE1_MULT * Omega  (must be a positive int)
TONE2_MULT = 2      # second tone is at TONE2_MULT * Omega
K_TARGET   = 1      # target harmonic: maximise |I_{K_TARGET}| at K_TARGET * Omega
# ─────────────────────────────────────────────────────────────────────────────

# ── graphics style ────────────────────────────────────────────────────────────
axes_width_mm  = 120.0
axes_height_mm = 100.0
left_mm, right_mm  = 20.0, 20.0
bottom_mm, top_mm  = 15.0,  8.0
fig_w = left_mm + axes_width_mm + right_mm
fig_h = bottom_mm + axes_height_mm + top_mm
spine_linewidth    = 2.0
tick_width         = 2.0
tick_direction     = 'in'
axis_label_fontsize  = 10.0
tick_label_fontsize  =  8.0
# ─────────────────────────────────────────────────────────────────────────────

# Intensity of an MZI with phase drive phi(t):
#   I(t) = (1 + cos(phi(t) + phi_dc)) / 2
#
# For phi(t) = beta1*sin(m1*Omega*t + psi1) + beta2*sin(m2*Omega*t + psi2),
# the Fourier coefficient of exp(i*phi(t)) at frequency K*Omega is:
#   C_K = sum_{n1*m1 + n2*m2 = K}  J_{n1}(beta1) * J_{n2}(beta2) * exp(i*(n1*psi1+n2*psi2))
#
# The IM amplitude at K*Omega, maximised over the DC bias phi_dc, equals |C_K|.
# (The DC bias can always be tuned to align the cosine term with the real axis.)
# We also maximise over the drive phases psi1, psi2 to find the best phase setting.

# ── enumerate all integer solutions to n1*TONE1_MULT + n2*TONE2_MULT = K_TARGET ──
n1_list, n2_list = [], []
for j in range(-K_TRUNC, K_TRUNC + 1):
    remainder = K_TARGET - j * TONE2_MULT
    if remainder % TONE1_MULT == 0:
        n1_list.append(remainder // TONE1_MULT)
        n2_list.append(j)
n1_arr = np.array(n1_list)   # (N_sol,)
n2_arr = np.array(n2_list)   # (N_sol,)
assert len(n1_arr) > 0, (
    f"No integer solutions: {TONE1_MULT}*n1 + {TONE2_MULT}*n2 = {K_TARGET}")
print(f"Found {len(n1_arr)} lattice solutions for the Bessel sum.")

# ── precompute Bessel functions on the beta grids ─────────────────────────────
beta1_vals = np.linspace(0, BETA1_MAX, N_BETA)
beta2_vals = np.linspace(0, BETA2_MAX, N_BETA)

# jv1[i, s] = J_{n1_arr[s]}(beta1_vals[i]),  shape (N_BETA, N_sol)
jv1 = np.stack([jv(int(n), beta1_vals) for n in n1_arr], axis=1)
# jv2[j, s] = J_{n2_arr[s]}(beta2_vals[j]),  shape (N_BETA, N_sol)
jv2 = np.stack([jv(int(n), beta2_vals) for n in n2_arr], axis=1)

# C[(i*N_BETA + j), s] = jv1[i,s] * jv2[j,s]  — outer product over betas
C = (jv1[:, None, :] * jv2[None, :, :]).reshape(-1, len(n1_arr))  # (N_BETA^2, N_sol)

# ── sweep psi1; vectorise psi2 via matrix multiply ────────────────────────────
psi_vals = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
# phase2[p, s] = exp(i * n2_arr[s] * psi2[p])
phase2 = np.exp(1j * n2_arr[None, :] * psi_vals[:, None])  # (N_PHI, N_sol)

Z = np.zeros(N_BETA * N_BETA)
for i_psi1, psi1 in enumerate(psi_vals):
    C_mod = C * np.exp(1j * (n1_arr * psi1))   # (N_BETA^2, N_sol)
    S     = C_mod @ phase2.T                    # (N_BETA^2, N_PHI)
    Z     = np.maximum(Z, np.max(np.abs(S) ** 2, axis=-1))
    if (i_psi1 + 1) % 25 == 0:
        print(f'  psi1 sweep: {i_psi1 + 1}/{N_PHI}', flush=True)

Z = Z.reshape(N_BETA, N_BETA)   # |C_K|^2 = |I_K|^2, rows=beta1 cols=beta2
print(f'Max |I_{K_TARGET}|   = {np.sqrt(Z.max()):.4f}')
print(f'Max |I_{K_TARGET}|^2 = {Z.max():.4f}')

# ── plot ──────────────────────────────────────────────────────────────────────
fig_w_in = fig_w / 25.4
fig_h_in = fig_h / 25.4
fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
fig.subplots_adjust(
    left   = left_mm   / fig_w,
    right  = 1 - right_mm  / fig_w,
    bottom = bottom_mm / fig_h,
    top    = 1 - top_mm    / fig_h,
)

im = ax.pcolormesh(beta1_vals, beta2_vals, Z.T,
                   cmap='viridis', shading='auto', vmin=0, vmax=1)
cb = fig.colorbar(im, ax=ax)
cb.set_label(
    rf'$\max_{{\psi_1,\psi_2}}\,|I_{{{K_TARGET}}}|^2$',
    fontsize=axis_label_fontsize)
cb.ax.tick_params(labelsize=tick_label_fontsize)

ax.set_xlabel(rf'$\beta_1$  (tone at ${TONE1_MULT}\,\Omega$)',
              fontsize=axis_label_fontsize)
ax.set_ylabel(rf'$\beta_2$  (tone at ${TONE2_MULT}\,\Omega$)',
              fontsize=axis_label_fontsize)
ax.set_title(
    rf'IM amplitude at ${K_TARGET}\,\Omega$  '
    rf'(tones: ${TONE1_MULT}\Omega$, ${TONE2_MULT}\Omega$)',
    fontsize=tick_label_fontsize)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
for spine in ax.spines.values():
    spine.set_linewidth(spine_linewidth)
ax.tick_params(axis='both', direction=tick_direction,
               width=tick_width, labelsize=tick_label_fontsize)

out_path = Path(__file__).parent / f'im_efficiency_{K_TARGET}f.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
