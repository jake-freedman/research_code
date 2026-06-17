import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── configuration ─────────────────────────────────────────────────────────────
BETA1_MAX = 5.0   # max modulation depth for tone at f
BETA2_MAX = 5.0   # max modulation depth for tone at (3/2)f
N_BETA    = 200   # grid points per beta axis
N_PHI     = 200   # phase-grid points per axis for optimization
K_TRUNC   = 15    # Bessel sum truncation: k in [-K_TRUNC, K_TRUNC]
CHUNK     = 1000  # (beta1, beta2) pairs per batch — controls peak memory
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

# Target: sideband at +3f.
# n1*(f) + n2*(3/2 f) = 3f  =>  2n1 + 3n2 = 6  =>  n1 = 3k, n2 = 2 - 2k
k  = np.arange(-K_TRUNC, K_TRUNC + 1)   # (M,)
n1 = 3 * k                               # Bessel orders for beta1 tone
n2 = 2 - 2 * k                          # Bessel orders for beta2 tone

beta1_vals = np.linspace(0, BETA1_MAX, N_BETA)
beta2_vals = np.linspace(0, BETA2_MAX, N_BETA)

# Precompute Bessel functions: jv1[i, m] = J_{n1[m]}(beta1[i])
jv1 = np.stack([jv(int(n), beta1_vals) for n in n1], axis=1)   # (N_BETA, M)
jv2 = np.stack([jv(int(n), beta2_vals) for n in n2], axis=1)   # (N_BETA, M)

# Flatten (beta1, beta2) pairs: C[i*N_BETA+j, m] = jv1[i,m] * jv2[j,m]
C = (jv1[:, None, :] * jv2[None, :, :]).reshape(-1, len(k))    # (N_BETA^2, M)

phi_vals = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

# phase2[p, m] = exp(i * n2[m] * phi2[p])
phase2 = np.exp(1j * n2[None, :] * phi_vals[:, None])          # (N_PHI, M)

# Sweep phi1; inner loop processes beta-pairs in chunks to cap memory usage.
# Peak memory per chunk: CHUNK * N_PHI * 16 bytes (complex128).
Z = np.zeros(N_BETA * N_BETA)
n_total = N_BETA * N_BETA
for i_phi1, phi1 in enumerate(phi_vals):
    phase1 = np.exp(1j * (n1 * phi1))   # (M,)
    for start in range(0, n_total, CHUNK):
        end = min(start + CHUNK, n_total)
        C_chunk = C[start:end] * phase1       # (CHUNK, M)
        S_chunk = C_chunk @ phase2.T           # (CHUNK, N_PHI)
        Z[start:end] = np.maximum(Z[start:end],
                                   np.max(np.abs(S_chunk) ** 2, axis=-1))
    if (i_phi1 + 1) % 20 == 0:
        print(f'  phi1 sweep: {i_phi1 + 1}/{N_PHI}', flush=True)

Z = Z.reshape(N_BETA, N_BETA)   # rows = beta1, cols = beta2
print(f'Maximum |S_{{+3}}|^2 found: {Z.max():.4f}')

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
                   cmap='viridis', shading='auto', vmin=0)
cb = fig.colorbar(im, ax=ax)
cb.set_label(r'$\max_{\phi_1,\phi_2}\,|S_{+3}|^2$',
             fontsize=axis_label_fontsize)
cb.ax.tick_params(labelsize=tick_label_fontsize)

ax.set_xlabel(r'$\beta_1$  (tone at $f$)',
              fontsize=axis_label_fontsize)
ax.set_ylabel(r'$\beta_2$  (tone at $\frac{3}{2}f$)',
              fontsize=axis_label_fontsize)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

for spine in ax.spines.values():
    spine.set_linewidth(spine_linewidth)
ax.tick_params(axis='both', direction=tick_direction,
               width=tick_width, labelsize=tick_label_fontsize)

out_path = Path(__file__).parent / 'two_tone_3f_heatmap.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
