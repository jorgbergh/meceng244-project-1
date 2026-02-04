"""
Problem 1b: Newton's method for minimization.
Finds x such that f(x) ≈ 0 (i.e., gradient of objective Pi is zero).
"""

import numpy as np


def myNewton(f, df, x0, TOL, maxit):
    """
    Newton's method to solve f(x) = 0.

    Parameters
    ----------
    f : callable
        Gradient of Pi; input M×1, output M×1.
    df : callable
        Hessian of Pi; input M×1, output M×M.
    x0 : array_like, shape (M, 1)
        Initial guess.
    TOL : float
        Maximum allowable norm of f(sol) (stop when ||f(x)|| <= TOL).
    maxit : int
        Maximum number of iterations.

    Returns
    -------
    sol : ndarray, shape (M, 1)
        Final value of x.
    its : int
        Number of iterations performed.
    hist : ndarray, shape (M, its+1)
        hist[:, i] = x at iteration i (0-indexed).
    """
    x0 = np.atleast_2d(np.asarray(x0, dtype=float))
    if x0.shape[1] != 1:
        x0 = x0.T
    M = x0.shape[0]
    x = x0.copy()
    hist = np.zeros((M, maxit + 1))
    hist[:, 0:1] = x

    for i in range(maxit):
        f_val = np.atleast_2d(f(x))
        if f_val.shape[1] != 1:
            f_val = f_val.T
        if np.linalg.norm(f_val) <= TOL:
            sol = x.copy()
            its = i
            hist = hist[:, : its + 1]
            return (sol, its, hist)

        df_val = np.asarray(df(x))
        if df_val.ndim == 0:
            df_val = np.array([[df_val]])
        elif df_val.ndim == 1:
            df_val = np.atleast_2d(df_val)

        step = np.linalg.solve(df_val, f_val)
        x = x - step
        hist[:, i + 1 : i + 2] = x

    sol = x.copy()
    its = maxit
    return (sol, its, hist)
