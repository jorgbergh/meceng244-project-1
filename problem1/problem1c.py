"""
Problem 1c: Gradient f and Hessian df for Pi_a and Pi_b, plus plots of first
and second derivatives on [-20, 20].
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Pi_a(x) = x^2  →  dPi_a/dx = 2x,  d²Pi_a/dx² = 2
# ---------------------------------------------------------------------------


def f_a(x):
    """Gradient of Pi_a: dPi_a/dx = 2x. Input M×1, output M×1."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 1:
        x = x.T
    return 2.0 * x


def df_a(x):
    """Hessian of Pi_a: d²Pi_a/dx² = 2. Input M×1, output M×M."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 1:
        x = x.T
    M = x.shape[0]
    return 2.0 * np.eye(M)


# ---------------------------------------------------------------------------
# Pi_b(x) = (x + (π/2) sin(x))^2
# dPi_b/dx = 2(x + (π/2)sin(x)) * (1 + (π/2)cos(x))
# d²Pi_b/dx² = 2(1 + (π/2)cos(x))^2 + 2(x + (π/2)sin(x)) * (-(π/2)sin(x))
# ---------------------------------------------------------------------------


def f_b(x):
    """Gradient of Pi_b. Input M×1, output M×1."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 1:
        x = x.T
    u = x + (np.pi / 2) * np.sin(x)
    du_dx = 1.0 + (np.pi / 2) * np.cos(x)
    return 2.0 * u * du_dx


def df_b(x):
    """Hessian of Pi_b. Input M×1, output M×M."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 1:
        x = x.T
    M = x.shape[0]
    u = x + (np.pi / 2) * np.sin(x)
    du_dx = 1.0 + (np.pi / 2) * np.cos(x)
    d2u_dx2 = -(np.pi / 2) * np.sin(x)
    d2Pi_dx2 = 2.0 * du_dx**2 + 2.0 * u * d2u_dx2
    return np.reshape(d2Pi_dx2, (M, M))


# ---------------------------------------------------------------------------
# Plots on domain -20 <= x <= 20
# ---------------------------------------------------------------------------

def main():
    x = np.linspace(-20, 20, 500)
    x_col = x.reshape(-1, 1)

    f_a_vals = np.array([f_a(xi) for xi in x_col]).flatten()
    f_b_vals = np.array([f_b(xi) for xi in x_col]).flatten()

    plt.figure()
    plt.plot(x, f_a_vals, label=r"$d\Pi_a/dx = 2x$")
    plt.plot(x, f_b_vals, label=r"$d\Pi_b/dx$")
    plt.xlabel("$x$")
    plt.ylabel(r"$d\Pi/dx$")
    plt.legend()
    plt.grid(True)
    plt.xlim(-20, 20)
    plt.tight_layout()
    plt.savefig("../ME144_244_S26_Project1/figures/derivatives_first.pdf")
    plt.close()

    df_a_vals = np.array([df_a(xi) for xi in x_col]).flatten()
    df_b_vals = np.array([df_b(xi) for xi in x_col]).flatten()

    plt.figure()
    plt.plot(x, df_a_vals, label=r"$d^2\Pi_a/dx^2 = 2$")
    plt.plot(x, df_b_vals, label=r"$d^2\Pi_b/dx^2$")
    plt.xlabel("$x$")
    plt.ylabel(r"$d^2\Pi/dx^2$")
    plt.legend()
    plt.grid(True)
    plt.xlim(-20, 20)
    plt.tight_layout()
    plt.savefig("../ME144_244_S26_Project1/figures/derivatives_second.pdf")
    plt.close()

    print("Saved figures/derivatives_first.pdf and figures/derivatives_second.pdf")


if __name__ == "__main__":
    main()
