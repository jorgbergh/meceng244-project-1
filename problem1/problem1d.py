"""
Problem 1d: Minimize Pi_a and Pi_b with Newton's method for x0 = 2*10^k, k in {-1,0,1}.
Plot Pi(hist) for each run. TOL=1e-8, maxit=20.
"""

import numpy as np
import matplotlib.pyplot as plt

from problem1b import myNewton
from problem1c import f_a, df_a, f_b, df_b


def Pi_a(x):
    return np.asarray(x) ** 2


def Pi_b(x):
    x = np.asarray(x)
    return (x + (np.pi / 2) * np.sin(x)) ** 2


def main():
    TOL = 1e-8
    maxit = 20
    k_vals = [-1, 0, 1]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))

    # ---------- Pi_a ----------
    for k in k_vals:
        x0 = np.array([[2.0 * 10**k]])
        sol, its, hist = myNewton(f_a, df_a, x0, TOL, maxit)
        # hist shape (1, its+1); evaluate Pi_a at each iteration
        x_hist = hist[0, :]
        pi_hist = Pi_a(x_hist)
        ax_a.semilogy(range(its + 1), pi_hist, "o-", label=f"$x_0 = 2 \\times 10^{{{k}}}$")

    ax_a.set_xlabel("Iteration")
    ax_a.set_ylabel(r"$\Pi_a(x)$")
    ax_a.set_title(r"$\Pi_a(x) = x^2$")
    ax_a.legend()
    ax_a.grid(True, which="both")
    ax_a.set_ylim(bottom=1e-20)

    # ---------- Pi_b ----------
    for k in k_vals:
        x0 = np.array([[2.0 * 10**k]])
        sol, its, hist = myNewton(f_b, df_b, x0, TOL, maxit)
        x_hist = hist[0, :]
        pi_hist = Pi_b(x_hist)
        ax_b.semilogy(range(its + 1), pi_hist, "o-", label=f"$x_0 = 2 \\times 10^{{{k}}}$")

    ax_b.set_xlabel("Iteration")
    ax_b.set_ylabel(r"$\Pi_b(x)$")
    ax_b.set_title(r"$\Pi_b(x) = (x + \frac{\pi}{2}\sin x)^2$")
    ax_b.legend()
    ax_b.grid(True, which="both")
    ax_b.set_ylim(bottom=1e-20)

    plt.tight_layout()
    plt.savefig("../ME144_244_S26_Project1/figures/newton_convergence.pdf")
    plt.close()
    print("Saved figures/newton_convergence.pdf")


if __name__ == "__main__":
    main()
