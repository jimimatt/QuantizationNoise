from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


class Data(NamedTuple):
    actual: np.ndarray
    noisy: np.ndarray
    quantized: np.ndarray


class Function(ABC):
    def __init__(self, formula: str, noise_level: float = 0.02, step_size: float = 0.05) -> None:
        self.formula = formula
        self.noise_level = noise_level
        self.step_size = step_size

    @abstractmethod
    def __call__(self, x: np.ndarray) -> Data:
        pass


class ExpDecay(Function):
    def __init__(self, noise_level: float = 0.02, step_size: float = 0.05) -> None:
        super().__init__(formula=r"$f(x) = e^{-x}$", noise_level=noise_level, step_size=step_size)

    def __call__(self, x: np.ndarray) -> Data:
        actual = np.exp(-x)
        noisy = actual + self.noise_level * np.random.normal(size=x.size)
        quantized = np.round(noisy / self.step_size) * self.step_size
        return Data(actual=actual, noisy=noisy, quantized=quantized)


class Sin(Function):
    def __init__(self, noise_level: float = 0.02, step_size: float = 0.05) -> None:
        super().__init__(formula=r"$f(x) = \sin(x)$", noise_level=noise_level, step_size=step_size)

    def __call__(self, x: np.ndarray) -> Data:
        actual = np.sin(x)
        noisy = actual + self.noise_level * np.random.normal(size=x.size)
        quantized = np.round(noisy / self.step_size) * self.step_size
        return Data(actual=actual, noisy=noisy, quantized=quantized)


class Sawtooth(Function):
    def __init__(self, noise_level: float = 0.02, step_size: float = 0.05) -> None:
        super().__init__(
            formula=r"$f(x) = 2(x - \lfloor x + 0.5 \rfloor)$", noise_level=noise_level, step_size=step_size
        )

    def __call__(self, x: np.ndarray) -> Data:
        actual = 2 * (x - np.floor(x + 0.5))
        noisy = actual + self.noise_level * np.random.normal(size=x.size)
        quantized = np.round(noisy / self.step_size) * self.step_size
        return Data(actual=actual, noisy=noisy, quantized=quantized)


class SquareWave(Function):
    def __init__(self, noise_level: float = 0.02, step_size: float = 0.05) -> None:
        super().__init__(formula=r"$f(x) = \text{sgn}(\sin(x))$", noise_level=noise_level, step_size=step_size)

    def __call__(self, x: np.ndarray) -> Data:
        actual = np.sign(np.sin(x))
        noisy = actual + self.noise_level * np.random.normal(size=x.size)
        quantized = np.round(noisy / self.step_size) * self.step_size
        return Data(actual=actual, noisy=noisy, quantized=quantized)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.arange(0, 10.1, 0.1)

    func = ExpDecay()
    y, y_noisy, y_quantized = func(x)
    # y, y_noisy, y_quantized = get_sin_data(x * 0.4 * np.pi)
    # y, y_noisy, y_quantized = get_sawtooth_data(x)
    # y, y_noisy, y_quantized = get_square_wave_data(x * 0.4 * np.pi)

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)  # 2 rows, 1 column, first subplot
    plt.plot(x, y, 'o-', label=r'$f(x) = e^{-x}$')
    plt.title(f'{func.formula} without Noise')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)  # 2 rows, 1 column, first subplot
    plt.plot(x, y_noisy, 'o-', label=r'$f(x) = e^{-x}$')
    plt.title(f'{func.formula} with Noise {func.noise_level}')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)  # 2 rows, 1 column, second subplot
    plt.plot(x, y_quantized, 'o-', label=r'$f(x) = e^{-x}$ with y quantized to $0.05$ steps')
    plt.title(f'{func.formula} with y Quantized to {func.step_size} Steps')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
