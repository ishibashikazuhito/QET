import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval
from pyqsp.angle_sequence import QuantumSignalProcessingPhases

alpha, beta, gamma = 0.1, 0.3, 0.3

A_ini = np.array([[alpha, gamma,     0,     0,     0,     0,     0,  beta],
              [ beta, alpha, gamma,     0,     0,     0,     0,     0],
              [    0,  beta, alpha, gamma,     0,     0,     0,     0],
              [    0,     0,  beta, alpha, gamma,     0,     0,     0],
              [    0,     0,     0,  beta, alpha, gamma,     0,     0],
              [    0,     0,     0,     0,  beta, alpha, gamma,     0],
              [    0,     0,     0,     0,     0,  beta, alpha, gamma],
              [gamma,     0,     0,     0,     0,     0,  beta, alpha]])

norm = np.linalg.norm(A_ini, ord=2)       # operator norm
A = A_ini / norm                          # normalization

abs_eigvals = np.abs(np.linalg.eigvals(A))
kappa = np.max(abs_eigvals) / np.min(abs_eigvals)     # condition number
print(kappa)

a = 1 / kappa   # left range
deg = 43

print(norm, a)

x_pos = np.linspace(a, 1, 200)
x_neg = -x_pos[::-1]
x = np.concatenate([x_neg, x_pos])
y = 1/x 

# Chebyshev approximation
coeffs = chebfit(x, y, deg)

coeffs[::2] = 0

def f_approx(x):
    return chebval(x, coeffs)

xx = np.linspace(-1, 1, 500)

yy = chebval(xx, coeffs)
coeffs = coeffs / (2 * kappa)

true_vals = 1/xx / (2 * kappa)
true_vals[np.abs(xx) < a - 0.05] = np.nan

error = (f_approx(xx) - true_vals)/true_vals
error[np.abs(xx) < a] = np.nan

"""
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(xx, true_vals, color='black')
ax1.plot(xx, f_approx(xx), color='red')
ax1.axhline(0, color='black', lw=0.5)
ax1.axvline(0, color='black', lw=0.5)
ax1.grid(True, linestyle=':', alpha=0.7) 

ax2.plot(xx, error, color='blue')
ax2.axhline(0, color='black', lw=0.5)
ax2.axvline(0, color='black', lw=0.5)
ax2.set_xlabel('x')
ax2.set_ylabel('precision')
ax2.grid(True, linestyle=':', alpha=0.7)
"""

plt.plot(xx, true_vals, color='black')
plt.plot(xx, f_approx(xx), color='red')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.grid(True, linestyle=':', alpha=0.7)

plt.show()

ang_seq = QuantumSignalProcessingPhases(coeffs)

phi_seq = [ang_seq[0] + ang_seq[-1] + (len(ang_seq) - 1 - 1) * np.pi/2] + [k - np.pi / 2 for k in ang_seq[1:-1]]

with open("phi_angles.txt", "w") as f:
    for phi in phi_seq:
        f.write(f"{phi:.10f}\n")

#print(f_approx(0.5))