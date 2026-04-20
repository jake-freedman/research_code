import numpy as np
from ssb_spectrum import compute_optical_spectrum, plot_optical_spectrum
import matplotlib.pyplot as plt

rng = np.random.default_rng()

beta1 = rng.uniform(0.5, 2.5)
beta2 = rng.uniform(0.5, 2.5)
# Random quadratic + linear dispersion coefficients
c1 = rng.uniform(-np.pi / 4, np.pi / 4)   # linear (group delay)
c2 = rng.uniform(-np.pi / 8, np.pi / 8)   # quadratic (GDD)
phi1 = lambda n: c1 * n + c2 * n**2

print(f"beta1 = {beta1:.3f} rad")
print(f"beta2 = {beta2:.3f} rad")
print(f"phi1(n) = {c1:.3f}*n + {c2:.3f}*n^2")

harmonics, amplitudes = compute_optical_spectrum(beta1, beta2, phi1, n_max=25)
fig, ax = plot_optical_spectrum(harmonics, amplitudes)
# fig.savefig("demo_spectrum.png", dpi=150)
# print("Saved demo_spectrum.png")

plt.show()
