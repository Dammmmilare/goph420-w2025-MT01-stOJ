
import scipy
import numpy as np

#Calculation for part a.

from scipy.integrate import trapezoid

# Constants
m = 93.0  # Mass in kg.
g = 3.718  # Gravity in m/s^2.
c = 47.0  # Drag coefficient in kg/s.
v0 = 100  # Initial velocity in m/s.
vf = g * m / c  # Terminal velocity in m/s.

# Velocity function
def v(t):
    return vf + (v0 - vf) * np.exp(-(c / m) * t)

# Time intervals
t1 = np.arange(0, 21, 2)  # Delta t @ 2 s.
t2 = np.arange(0, 21, 1)  # Delta t @ 1 s.

# Integrating  using trapezoidal rule.
integral_t1 = trapezoid(v(t1), t1)
integral_t2 = trapezoid(v(t2), t2)

print("Part a Answers")
print(f"Trapezoidal rule with Delta t @ 2 s: {integral_t1:.2f} m")
print(f"Trapezoidal rule with Delta t @ 1 s: {integral_t2:.2f} m")

#Calculations for part b.

from scipy.integrate import simpson

# Integrating using Simpson's 1/3 rule.
integral_s1 = simpson(v(t1), t1)
integral_s2 = simpson(v(t2), t2)

print("Part b Answers")
print(f"Simpson's 1/3 rule with Delta t @ 2 s: {integral_s1:.2f} m")
print(f"Simpson's 1/3 rule with Delta t @ 1 s: {integral_s2:.2f} m")

#Calculations for part c and d.
from scipy.integrate import fixed_quad

# Integrate using Gauss-Legendre quadrature.
integral_gl, _ = fixed_quad(v, 0, 20, n=5)

print("Part c Answers")
print(f"Gauss-Legendre quadrature: {integral_gl:.2f} m")

print("Part d Answers")
print("The Gauss-Legendre Quadrature uses an order of nine, the Simpsons 1/3 rule uses an order of two, "
"and the trapezoidal rule uses an order of one. Since the Gauss-Legendre Quadrature makes use of higher-order "
"polynomials, I anticipate it to be more accurate.")
