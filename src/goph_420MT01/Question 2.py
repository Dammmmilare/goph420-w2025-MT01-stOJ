import scipy
import numpy as np
import sys


#Question 2a
from scipy.integrate import trapz
#from scipy.integrate import trapezoid

# Constants
m = 93.0  # Mass in kg
g = 3.718  # Gravity in m/s^2
c = 47.0  # Drag coefficient in kg/s
v0 = 100  # Initial velocity in m/s
vf = g * m / c  # Terminal velocity in m/s

# Velocity function
def v(t):
    return vf + (v0 - vf) * np.exp(-(c / m) * t)

# Time intervals
t1 = np.arange(0, 21, 2)  # Delta t = 2 s
t2 = np.arange(0, 21, 1)  # Delta t = 1 s

# Integrate using trapezoidal rule
integral_t1 = trapz(v(t1), t1)
integral_t2 = trapz(v(t2), t2)

print("Question 2a")
print(f"Trapezoidal rule with Delta t = 2 s: {integral_t1:.2f} m")
print(f"Trapezoidal rule with Delta t = 1 s: {integral_t2:.2f} m")

#Question 2b

#from scipy.integrate import simpson
from scipy.integrate import simps

# Integrate using Simpson's 1/3 rule
integral_s1 = simps(v(t1), t1)
integral_s2 = simps(v(t2), t2)

print("Question 2b")
print(f"Simpson's 1/3 rule with Delta t = 2 s: {integral_s1:.2f} m")
print(f"Simpson's 1/3 rule with Delta t = 1 s: {integral_s2:.2f} m")

#Question 2c

#from scipy.integrate import fixed_quad
from scipy.integrate import quad

# Integrate using Gauss-Legendre quadrature
integral_gl, _ = quad(v, 0, 20, n=5)

print("Question 2c")
print(f"Gauss-Legendre quadrature: {integral_gl:.2f} m")

print("Question 2d")
print("For the trapezoidal rule uses an order of 1, the Simpsons 1/3 rule uses an order of 2 and the Gauss-Legendre Quadrature uses an order of 9. I expect the Gauss-Legendre Quadrature to be more accurate because it uses higher-order polynomials.")
