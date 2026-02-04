import numpy as np
import matplotlib.pyplot as plt

def Pi_a(x):
    return x**2

def Pi_b(x):
    return (x + (np.pi / 2) * np.sin(x))**2

x = np.linspace(-20, 20, 500)
plt.figure()
plt.plot(x, Pi_a(x), label=r"$\Pi_a(x) = x^2$")
plt.plot(x, Pi_b(x), label=r"$\Pi_b(x) = (x + \frac{\pi}{2}\sin x)^2$")
plt.xlabel("$x$")
plt.ylabel("$\Pi(x)$")
plt.legend()
plt.grid(True)
plt.xlim(-20, 20)
plt.tight_layout()
plt.savefig("../ME144_244_S26_Project1/figures/objective_functions.pdf")
plt.show()
