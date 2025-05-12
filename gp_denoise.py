from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel
from sklearn.metrics import root_mean_squared_error

from sample_data import ExpDecay, Sawtooth, Sin, SquareWave

# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html

quantization_step = 0.05

x = np.arange(0, 10.1, 0.1)

funcs = [
    ExpDecay(noise_level=0.02, step_size=quantization_step),
    Sin(noise_level=0.2, step_size=quantization_step),
    Sawtooth(noise_level=0.2, step_size=quantization_step),
    SquareWave(noise_level=0.2, step_size=quantization_step),
]

kernels = {
    "RBF": 1 * RBF(length_scale=0.1, length_scale_bounds=(0.1, 50.0)),
    "Matern": 1.0 * Matern(length_scale=1.0, nu=1.5) + 0.1 * WhiteKernel(),
    "RationalQuadratic": RationalQuadratic() + WhiteKernel(),
}


class Result(NamedTuple):
    kernel: str
    rmse: float


results = {func: Result("placeholder_kernel", np.inf) for func in funcs}

for func in funcs:
    y, y_noisy, y_quantized = func(x)

    estimated_quantization_step = np.min(np.diff(np.unique(y_quantized)))
    binning_noise_std = 1.0 * estimated_quantization_step / np.sqrt(12)  # Uniform distribution

    for kernel_name, kernel in kernels.items():
        print(f"Kernel: {kernel_name}")
        gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=binning_noise_std**2, n_restarts_optimizer=20)
        gaussian_process.fit(x[np.newaxis].T, y_quantized[np.newaxis].T)
        mean_prediction, std_prediction = gaussian_process.predict(x[np.newaxis].T, return_std=True)

        print(f" - RMSE for: {func.__class__.__name__}")
        print(f"   - noisy {root_mean_squared_error(y, y_noisy):.4f}")
        print(f"   - quantized {root_mean_squared_error(y, y_quantized):.4f}")
        gp_rmse = root_mean_squared_error(y, mean_prediction)
        print(f"   - GP prediction {gp_rmse:.4f}")
        if gp_rmse < results[func].rmse:
            results[func] = Result(kernel_name, gp_rmse)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y, label=f"{func.formula}", linestyle="dotted")
        ax.errorbar(
            x,
            y_quantized,
            [binning_noise_std] * len(x),
            linestyle="None",
            color="tab:blue",
            marker=".",
            markersize=10,
            label="Observations",
        )
        ax.plot(x, mean_prediction, label="Mean prediction")
        ax.fill_between(
            x.ravel(),
            mean_prediction - 1.96 * std_prediction,
            mean_prediction + 1.96 * std_prediction,
            color="tab:orange",
            alpha=0.5,
            label=r"95% confidence interval",
        )
        fig.legend()
        ax.set_xlabel("$x$")
        ax.set_ylabel("$f(x)$")
        fig.suptitle(f"GP(kernel={kernel_name}) on a noisy {func.formula}, Noise level: {func.noise_level}")

print()
print("=== Results ===")
for func, result in results.items():
    print(f" - {func.__class__.__name__}: {result.kernel} RMSE: {result.rmse:.4f}")
plt.show()
