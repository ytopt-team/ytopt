#HPO for the Benchmark to minimize the absolute value of the MSE difference between mixture model and Splines
# Benchmark: Mixture Model vs Splines (Expanded Function Set)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
from scipy.special import jn, yn  # Bessel functions
from sklearn.metrics import mean_squared_error

# --- Configuration ---
M_GAUSSIANS = #P0   # @param {type:"integer"}
N_SIGMOIDS = #P1    # @param {type:"integer"}
SPLINE_SMOOTH = #P2 # @param {type:"number"} Controls spline aggressiveness

# --- 1. Define Comprehensive Function Set ---
# Format: "Name": (Function, (min_x, max_x))
test_functions = {
    # --- Classic Feynman Shapes ---
    "Feynman I.6.20 (Gaussian)": (lambda x: np.exp(-(x**2)/2), (-3, 3)),
    "Feynman I.10.7 (Lorentzian)": (lambda x: 1.0 / (1 + x**2), (-3, 3)),
    "Feynman I.12.5 (Quadratic)": (lambda x: x**2, (-2, 2)),
    "Feynman II.35.21 (Tanh)": (lambda x: np.tanh(x), (-3, 3)),
    "Feynman I.15.10 (Inv Root)": (lambda x: 1.0 / np.sqrt(1 - x**2 + 1e-6), (-0.9, 0.9)), # Relativistic p
    "Feynman I.41.16 (Rad. Density)": (lambda x: (x**3)/(np.exp(x)-1), (0.1, 5)), # Blackbody
    "Feynman I.39.1 (Inverse)": (lambda x: 1.0/x, (0.5, 5)), # PV=const

    # --- Special Functions ---
    "Bessel J0 (1st Kind)": (lambda x: jn(0, x), (0, 10)),
    "Bessel J1 (1st Kind)": (lambda x: jn(1, x), (0, 10)),
    "Bessel Y0 (2nd Kind)": (lambda x: yn(0, x), (0.2, 10)), # Diverges at 0
    "Bessel Y1 (2nd Kind)": (lambda x: yn(1, x), (0.2, 10)),
}

# --- 2. Define Mixture Model ---
def sigmoid(x, height, center, width):
    return height / (1 + np.exp(-np.clip((x - center) / width, -100, 100)))

def gaussian(x, height, mean, width):
    return height * np.exp(-((x - mean)**2) / (2 * (width + 1e-6)**2))

def mixture_model(params, x, m, n):
    y_pred = np.zeros_like(x)
    # Gaussians
    for i in range(m):
        idx = i * 3
        y_pred += gaussian(x, params[idx], params[idx+1], params[idx+2])
    # Sigmoids
    offset = m * 3
    for j in range(n):
        idx = offset + j * 3
        y_pred += sigmoid(x, params[idx], params[idx+1], params[idx+2])
    return y_pred

def mixture_residuals(params, x, y_true, m, n):
    return mixture_model(params, x, m, n) - y_true

def fit_mixture(x, y, m, n):
    # Robust Heuristic Initialization
    initial_params = []

    # Spread centers evenly across the domain
    centers = np.linspace(np.min(x), np.max(x), max(m, n))

    for i in range(m):
        initial_params.extend([np.std(y), centers[i % len(centers)], 1.0])
    for i in range(n):
        initial_params.extend([np.max(y)-np.min(y), centers[i % len(centers)], 1.0])

    res = least_squares(mixture_residuals, initial_params, args=(x, y, m, n),
                        method='lm', max_nfev=6000)
    return res.x

# --- 3. Execution Loop ---
def run_benchmark(m, n, smoothing):
    print(f"--- Benchmarking: {m}G + {n}S vs Splines (Smoothing={smoothing}) ---")

    # Create grid of plots
    num_funcs = len(test_functions)
    cols = 3
    rows = (num_funcs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4*rows))
    axes = axes.flatten()

    results = []

    for i, (name, (func, domain)) in enumerate(test_functions.items()):
        # Generate Data specific to domain
        X = np.linspace(domain[0], domain[1], 150)
        y_true = func(X)

        # --- A. Mixture Model ---
        mix_param_count = 3 * (m + n)
        try:
            mix_params = fit_mixture(X, y_true, m, n)
            y_mix = mixture_model(mix_params, X, m, n)
            mse_mix = mean_squared_error(y_true, y_mix)
        except Exception as e:
            y_mix = np.zeros_like(X)
            mse_mix = float('inf')
            print(f"Fit failed for {name}: {e}")

        # --- B. Spline Model ---
        # k=3 is cubic spline. s controls smoothing (number of knots)
        spl = UnivariateSpline(X, y_true, k=3, s=smoothing)
        y_spline = spl(X)
        mse_spline = mean_squared_error(y_true, y_spline)

        # Get Spline Complexity
        spline_coeffs = spl.get_coeffs()
        spline_param_count = len(spline_coeffs)

        results.append({
            "Name": name,
            "MSE_Mix": mse_mix,
            "P_Mix": mix_param_count,
            "MSE_Spl": mse_spline,
            "P_Spl": spline_param_count
        })

        # --- Plotting ---
#        ax = axes[i]
#        ax.plot(X, y_true, 'k--', label="True", linewidth=2, alpha=0.5)
#        ax.plot(X, y_mix, 'r-', label=f"Mix", linewidth=1.5)
#        ax.plot(X, y_spline, 'b:', label=f"Spl", linewidth=2)
#        ax.set_title(f"{name}", fontsize=10, fontweight='bold')
#        ax.tick_params(labelsize=8)
#        if i == 0: ax.legend(fontsize=8) # Only legend on first plot to save space

    # Hide unused subplots
#    for j in range(i+1, len(axes)):
#        axes[j].axis('off')

#    plt.tight_layout()
#    plt.show()

    # --- Summary Table ---
    # Header
    print(f"\n{'Equation':<35} | {'Mix MSE':<20} | {'Spline MSE':<20} | {'Winner'}")
    print("-" * 95)

    mix_avg = 0.0
    spl_avg = 0.0
    avg = 0.0
    for r in results:
        # Determine winner based on MSE
        winner = "Mix" if r['MSE_Mix'] < r['MSE_Spl'] else "Spline"

        mix_str = f"{r['MSE_Mix']:.1e}"
        spl_str = f"{r['MSE_Spl']:.1e}"

        print(f"{r['Name']:<35} | {mix_str:<20} | {spl_str:<20} | {winner}")
        mix_avg += r['MSE_Mix']
        spl_avg += r['MSE_Spl']

    avg = mix_avg - spl_avg
    print(f"Average difference: {abs(avg):.1e}")

if __name__ == "__main__":
    run_benchmark(M_GAUSSIANS, N_SIGMOIDS, SPLINE_SMOOTH)
