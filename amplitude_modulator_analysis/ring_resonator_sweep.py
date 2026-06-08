import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import os

# ============================================================
# USER PARAMETERS
# ============================================================

# Ring resonator parameters
F_0_GHZ = 400e3      # Resonance frequency in GHz (193 THz ~ 1550 nm)
Q_I = 1e6            # Intrinsic quality factor
Q_E = 1e6            # Extrinsic (coupling) quality factor

# Resonance shift sweep
SHIFT_MIN_GHZ = -10.0    # Minimum resonance shift from center (GHz)
SHIFT_MAX_GHZ = 10.0     # Maximum resonance shift from center (GHz)
N_CURVES = 10            # Number of curves (steps between min and max shift)

# Plot range
X_RANGE_GHZ = SHIFT_MAX_GHZ + 5      # Half-width of x-axis (plot spans -X_RANGE to +X_RANGE GHz)
Y_MIN = -0.1              # Lower y-axis limit
Y_MAX = 1.1              # Upper y-axis limit

# Simulated data
N_POINTS_CURVE = 300     # Points in smooth transmission curve
N_DATA_POINTS = 30       # Noisy data points per curve
NOISE_LEVEL = 0.02       # Noise std dev (fraction of full-scale transmission)
NOISE_SEED = 42          # RNG seed for reproducibility

# Figure dimensions (axes region only, in mm)
AXES_WIDTH_MM = 65.0
AXES_HEIGHT_MM = 35.0

# Figure margins (mm)
LEFT_MM = 20.0
RIGHT_MM = 8.0
BOTTOM_MM = 15.0
TOP_MM = 8.0

# Curve truncation: each curve is only drawn within this half-range around its resonance
CURVE_HALF_RANGE_GHZ = 4.0  # set to None to disable truncation

# Colors (from experiment_control/graphics.py)
COLOR_LOW = '#e5a3a3'    # RED2  — curves at SHIFT_MIN
COLOR_MID = '#93C572'    # GREEN2 — curves at midpoint
COLOR_HIGH = '#b2cbf2'   # LIGHTBLUE2 — curves at SHIFT_MAX

# Publication mode: removes all axis and tick labels when True
FOR_PUBLICATION = True

# Output
MEDIA_DIR = r"C:\Users\acous\OneDrive - UCB-O365\quantum_nanophoxonics\media"
OUTPUT_FILENAME = "ring_resonator_sweep.svg"

# ============================================================


def arclength_sample(f_fine, T_fine, n_points, y_min, y_max):
    """Return n_points x-values sampled uniformly by arc length along the curve."""
    x_norm = (f_fine - f_fine[0]) / (f_fine[-1] - f_fine[0])
    y_norm = (T_fine - y_min) / (y_max - y_min)
    ds = np.sqrt(np.diff(x_norm) ** 2 + np.diff(y_norm) ** 2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_uniform = np.linspace(0.0, s[-1], n_points)
    return np.interp(s_uniform, s, f_fine)


def ring_transmission(f_det, f_res_shift, gamma_i, gamma_e):
    """
    All-pass ring resonator transmission (coupled-mode theory).

    T = [delta^2 + ((gamma_i - gamma_e)/2)^2]
      / [delta^2 + ((gamma_i + gamma_e)/2)^2]

    where delta = f_det - f_res_shift. All frequencies in GHz.
    """
    delta = f_det - f_res_shift
    num = delta**2 + ((gamma_i - gamma_e) / 2.0) ** 2
    den = delta**2 + ((gamma_i + gamma_e) / 2.0) ** 2
    return num / den


def main():
    # Linewidth (half-linewidth = decay rate / 2pi) in GHz
    gamma_i = F_0_GHZ / Q_I
    gamma_e = F_0_GHZ / Q_E

    # Resonance positions for each curve
    shifts = np.linspace(SHIFT_MIN_GHZ, SHIFT_MAX_GHZ, N_CURVES)

    # Color interpolation from RED2 -> gray -> LIGHTBLUE2
    cmap = LinearSegmentedColormap.from_list(
        "rgb", [mcolors.to_rgb(COLOR_LOW), mcolors.to_rgb(COLOR_MID), mcolors.to_rgb(COLOR_HIGH)]
    )
    curve_colors = [cmap(i / max(N_CURVES - 1, 1)) for i in range(N_CURVES)]

    # Frequency axes
    f_smooth = np.linspace(-X_RANGE_GHZ, X_RANGE_GHZ, N_POINTS_CURVE)
    f_fine = np.linspace(-X_RANGE_GHZ, X_RANGE_GHZ, 10_000)  # dense grid for arc length

    rng = np.random.default_rng(NOISE_SEED)

    # Figure layout
    fig_w_mm = LEFT_MM + AXES_WIDTH_MM + RIGHT_MM
    fig_h_mm = BOTTOM_MM + AXES_HEIGHT_MM + TOP_MM
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    ax = fig.add_axes([
        LEFT_MM / fig_w_mm,
        BOTTOM_MM / fig_h_mm,
        AXES_WIDTH_MM / fig_w_mm,
        AXES_HEIGHT_MM / fig_h_mm,
    ])

    for i, shift in enumerate(shifts):
        color = curve_colors[i]

        # Truncate plot domain to ±CURVE_HALF_RANGE_GHZ around this resonance
        if CURVE_HALF_RANGE_GHZ is not None:
            lo = max(shift - CURVE_HALF_RANGE_GHZ, -X_RANGE_GHZ)
            hi = min(shift + CURVE_HALF_RANGE_GHZ,  X_RANGE_GHZ)
            f_plot = f_smooth[(f_smooth >= lo) & (f_smooth <= hi)]
            f_arc  = f_fine  [(f_fine   >= lo) & (f_fine   <= hi)]
        else:
            f_plot = f_smooth
            f_arc  = f_fine

        # Smooth transmission curve
        T_smooth = ring_transmission(f_plot, shift, gamma_i, gamma_e)
        ax.plot(f_plot, T_smooth, color=color, linewidth=1.5, zorder=2)

        # Noisy data points sampled uniformly by arc length along the curve
        T_fine_arc = ring_transmission(f_arc, shift, gamma_i, gamma_e)
        f_data = arclength_sample(f_arc, T_fine_arc, N_DATA_POINTS, Y_MIN, Y_MAX)
        T_data = ring_transmission(f_data, shift, gamma_i, gamma_e)
        T_noisy = np.clip(
            T_data + rng.normal(0.0, NOISE_LEVEL, size=T_data.shape), 0.0, 1.0
        )
        rgb = mcolors.to_rgb(color)
        ax.scatter(f_data, T_noisy, s=10, zorder=3,
                   facecolors=(*rgb, 0.5), edgecolors=(*rgb, 1.0), linewidths=0.6)

    ax.set_xlim(-X_RANGE_GHZ, X_RANGE_GHZ)
    ax.set_ylim(Y_MIN, Y_MAX)
    if FOR_PUBLICATION:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", direction="in", width=2, length=4,
                       labelbottom=False, labelleft=False, top=True, right=True)
    else:
        ax.set_xlabel("Detuning (GHz)", fontsize=10)
        ax.set_ylabel("Transmission", fontsize=10)
        ax.tick_params(axis="both", direction="in", width=2, length=4, labelsize=8,
                       top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    os.makedirs(MEDIA_DIR, exist_ok=True)
    out_path = os.path.join(MEDIA_DIR, OUTPUT_FILENAME)
    fig.savefig(out_path, format="svg")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
