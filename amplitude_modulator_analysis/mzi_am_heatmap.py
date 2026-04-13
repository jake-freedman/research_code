import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn


def compute_am(alpha_vals, phi_vals, n):
    """
    Compute MZI amplitude modulation at harmonic n*Omega.

    AM ~ Jn(alpha) * sin(phi)  for odd n
    AM ~ Jn(alpha) * cos(phi)  for even n
    """
    alpha_grid, phi_grid = np.meshgrid(alpha_vals, phi_vals, indexing="ij")
    bessel = jn(n, alpha_grid)
    trig = np.sin(phi_grid) if n % 2 != 0 else np.cos(phi_grid)
    return bessel * trig


def plot_heatmap(n=1, alpha_max=5.0, n_alpha=500, n_phi=500, save_path=None,
                 figsize_mm=(100, 80)):
    alpha_vals = np.linspace(0, alpha_max, n_alpha)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi)

    am = compute_am(alpha_vals, phi_vals, n)

    trig_label = "sin(φ)" if n % 2 != 0 else "cos(φ)"

    figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
    fig, ax = plt.subplots(figsize=figsize_in)
    img = ax.pcolormesh(
        phi_vals / np.pi,
        alpha_vals,
        am,
        cmap="RdBu",
        shading="auto",
        vmin=-jn(1, alpha_vals).max(),
        vmax=jn(1, alpha_vals).max(),
    )

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(f"AM amplitude  [J{n}(α) · {trig_label}]", fontsize=10)

    ax.set_xlabel("Static phase φ  (units of π)", fontsize=10)
    ax.set_ylabel("Modulation depth α", fontsize=10)
    # ax.set_title(
    #     f"MZI amplitude modulation at harmonic n = {n}\n"
    #     f"∝ J{n}(α) · {trig_label}",
    #     fontsize=13,
    # )

    phi_ticks = np.arange(0, 2.5, 0.5)
    ax.set_xticks(phi_ticks)
    ax.set_xticklabels([f"{v:.1f}π" for v in phi_ticks])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    plt.show()


if __name__ == "__main__":
    n = 2                            # harmonic order
    alpha_max = 5                    # upper limit of modulation depth axis
    save_path = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media\am2"

    plot_heatmap(n=n, alpha_max=alpha_max, save_path=save_path)
