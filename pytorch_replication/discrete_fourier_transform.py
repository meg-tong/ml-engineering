# %%
import importlib
from typing import Callable

import numpy as np
from matplotlib.pyplot import step

import arena_utils
import plotly.express as px


def create_matrix(N, inverse=False):
    omega = np.exp(2j * np.pi / N * (-1 if not inverse else 1))
    a = np.arange(N)
    multiplier = 1 / N if inverse else 1
    return multiplier * omega ** np.outer(a, a)

def DFT_1d(arr : np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Returns the DFT of the array `arr`, using the equation above.
    """
    return np.inner(create_matrix(arr.shape[0], inverse), arr)

arena_utils.test_DFT_func(DFT_1d)
# %%
def test(DFT_1d) -> None:
    F = np.array([20, -4j, 12, 4j])
    f = np.array([8,4,8,0])
    np.testing.assert_allclose(F, DFT_1d(f, inverse=False), atol=1e-10)
    np.testing.assert_allclose(f, DFT_1d(F, inverse=True), atol=1e-10)

test(DFT_1d)
# %%
def integrate_function(func: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Calculates the approximation of the Riemann integral of the function `func`, 
    between the limits x0 and x1.

    You should use the Left Rectangular Approximation Method (LRAM).
    """
    step = (x1 - x0) / 1000
    xs = np.arange(x0, x1, step)
    return sum([step * func(x) for x in xs])

arena_utils.test_integrate_function(integrate_function)
#%% 
def integrate_product(func1: Callable, func2: Callable, x0: float, x1: float):
    """
    Computes the L2 inner product of two functions, between x0 and x1. 

    This should reference the previous function.

    For more, see this page: https://mathworld.wolfram.com/L2-InnerProduct.html
    """
    f = lambda x: func1(x) * func2(x)
    return integrate_function(f, x0, x1)

arena_utils.test_integrate_product(integrate_product)
# %%
def calculate_fourier_series(func: Callable, max_freq: int = 50):
    """
    Calculates the fourier coefficients of a function, 
    assumed periodic between [-pi, pi].

    Your function should return ((a_0, A_n, B_n), func_approx), where:
        a_0 is a float
        A_n, B_n are lists of floats, with n going up to `max_freq`
        func_approx is the fourier approximation, as described above
    """
    a_0 = 1/np.pi * integrate_function(func, -np.pi, np.pi)
    A = [1/np.pi * integrate_product(func, lambda x: np.cos(n * x), -np.pi, np.pi) for n in range(1, max_freq + 1)]
    B = [1/np.pi * integrate_product(func, lambda x: np.sin(n * x), -np.pi, np.pi) for n in range(1, max_freq + 1)]
    func_approx = lambda x: 0.5 * a_0 + sum([A[n - 1] * np.cos(n * x) + B[n - 1] * np.sin(n * x) for n in range(1, max_freq + 1)])
    return ((a_0, A, B), func_approx)
# %%
def create_fourier_graph(calculate_fourier_series: Callable, func: Callable):

    x = np.linspace(-np.pi, np.pi, 1000)
    y = func(x)
    coeffs, func_approx = calculate_fourier_series(func)
    y_approx = func_approx(x)

    fig = px.scatter(x=x, y=y)
    fig = px.scatter(x=x, y=y_approx)
    fig.show()

step_func = lambda x: 1 * (x > 0)
create_fourier_graph(calculate_fourier_series, func = step_func)
arena_utils.create_interactive_fourier_graph(calculate_fourier_series, func = step_func)

# %%
