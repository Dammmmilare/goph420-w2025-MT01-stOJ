#Question 1 Calculations:
#-----------------------

import scipy
import matplotlib.pyplot as plt
import numpy as np

# Provided calculation figures.
R = 6.371e6  # Radius in meters
e = 0.612  # Emissivity
sigma = 5.67e-8  # Stefan-Boltzmann constant in W m^-2 K^-4
delta_R = 0.021e6  # Error in radius in meters
delta_e = 0.015  # Error in emissivity
T = 285  # Temperature measured in Kelvin
delta_T = 5  # Error in temperature in Kelvin

# Partial derivatives of H with respect to R, e, and T and their respective equations
partial_H_R = 8 * np.pi * R * e * sigma * T**4  # dH/dR
partial_H_e = 4 * np.pi * R**2 * sigma * T**4   # dH/de
partial_H_T = 16 * np.pi * R**2 * e * sigma * T**3  # dH/dT

# Calculation for total error in H.
delta_H = (
    np.abs(partial_H_R) * delta_R +  # Error due to R
    np.abs(partial_H_e) * delta_e +  # Error due to e
    np.abs(partial_H_T) * delta_T    # Error due to T
)

# Output the expression for total error in H
print("Question 1a")
print("Expression for total error in H:")
print(f"ΔH ≈ |∂H/∂R| * ΔR + |∂H/∂e| * Δe + |∂H/∂T| * ΔT")
print(f"ΔH ≈ {partial_H_R:.2e} * {delta_R:.2e} + {partial_H_e:.2e} * {delta_e:.2e} + {partial_H_T:.2e} * {delta_T:.2e}")
print(f"ΔH ≈ {delta_H:.2e} W")


#Question 1b calculation


# Constants
sigma = 5.67e-8  # Stefan-Boltzmann constant in W m^-2 K^-4
R = 6.371e6  # Radius in meters
delta_R = 0.021e6  # Error in radius in meters
e = 0.612  # Emissivity
delta_e = 0.015  # Error in emissivity
T = 285  # Temperature in Kelvin
delta_T = 5  # Error in temperature in Kelvin

# Part i: Relative error in R
relative_error_R = delta_R / R

# Part ii: Relative error in e
relative_error_e = delta_e / e

# Part iii: Relative error in T
relative_error_T = delta_T / T

# Part iv: Expected value of H
H = 4 * np.pi * R**2 * e * sigma * T**4

# Part v: Total error in H (using the expression from 1a)
partial_H_R = 8 * np.pi * R * e * sigma * T**4  # dH/dR
partial_H_e = 4 * np.pi * R**2 * sigma * T**4   # dH/de
partial_H_T = 16 * np.pi * R**2 * e * sigma * T**3  # dH/dT

delta_H = (
    np.abs(partial_H_R) * delta_R +  # Error due to R
    np.abs(partial_H_e) * delta_e +  # Error due to e
    np.abs(partial_H_T) * delta_T    # Error due to T
)

# Part vi: Relative error in H
relative_error_H = delta_H / H

# Output results
print("Question 1b")
print(f"Relative error in R: {relative_error_R:.6f}")
print(f"Relative error in e: {relative_error_e:.6f}")
print(f"Relative error in T: {relative_error_T:.6f}")
print(f"Expected value of H: {H:.1e} W")
print(f"Total error in H: {delta_H:.1e} W")
print(f"Relative error in H: {relative_error_H:.6f}")

# Part c: Explanation of which variable contributes most to the error in H
print("Question 1c")
print("\nExplanation:")
print("The variable that contributes most to the error in H is the one with the largest product of sensitivity and relative error.")
print(f"Sensitivity to R: {partial_H_R:.2e}")
print(f"Sensitivity to e: {partial_H_e:.2e}")
print(f"Sensitivity to T: {partial_H_T:.2e}")
print(f"Relative error in R: {relative_error_R:.6f}")
print(f"Relative error in e: {relative_error_e:.6f}")
print(f"Relative error in T: {relative_error_T:.6f}")
print("Based on the results, the variable with the largest contribution to the error in H is likely temperature (T), due to its high sensitivity and relative error.")
print("The significance of the contribution of temperature T to the error in H is mostly due to the sensitivity of H to its value because the T^4 dependance in the Stefan Boltzmann law makes H higly sensitive to the changes in T.")

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


#Question 3a

# Constants
V = 1.38e6  # Volume in m^3
W = 2.15e6  # Input rate in g/yr
Q = 1.29e5  # Outflow rate in m^3/yr
k = 0.825  # Reaction rate in m^0.5/g^0.5/yr


# Fixed point iteration functions
def g1(c):
    return ((W - Q * c) / (k * V)) ** 2


def g2(c):
    return (W - k * V * np.sqrt(c)) / Q


# Test convergence for g1 and g2
c_values = np.linspace(1.0, 5.0, 16)
g1_values = g1(c_values)
g2_values = g2(c_values)


# Check if |g'(c)| < 1 for convergence
def derivative_g1(c):
    return -2 * Q * (W - Q * c) / (k * V) ** 2


def derivative_g2(c):
    return -k * V / (2 * Q * np.sqrt(c))


g1_derivatives = derivative_g1(c_values)
g2_derivatives = derivative_g2(c_values)


# Function to print the results of derivative_g1(c) and derivative_g2(c)
def print_derivatives(c_values):
    print("Derivatives of g1(c):")
    for c, g1_derivative in zip(c_values, g1_derivatives):
        print(f"c = {c:.2f}, g1'(c) = {g1_derivative:.6f}")

    print("\nDerivatives of g2(c):")
    for c, g2_derivative in zip(c_values, g2_derivatives):
        print(f"c = {c:.2f}, g2'(c) = {g2_derivative:.6f}")


# Print the results
print("Question 3a")
print("g1 converges:", np.all(np.abs(g1_derivatives) < 1))
print("g2 converges:", np.all(np.abs(g2_derivatives) < 1))
print("They will converge if their absolute derivative is less than 1 but it won't converge if it's greater than 1.")

# Print the derivatives
print_derivatives(c_values)


#Question 3b

def fixed_point_iteration(g, c0, tol=1e-4, max_iter=100):
    c = c0
    for i in range(max_iter):
        c_new = g(c)
        if np.abs(c_new - c) < tol:
            return c_new
        c = c_new
    return c

# Use g1 or g2 depending on convergence
c_converged = fixed_point_iteration(g1, 4.0)  # Assuming g1 converges
print("Question 3b")
print(f"Converged value of c: {c_converged:.4f} g/m^3")


#Question 3c

def f(c):
    return W - Q * c - k * V * np.sqrt(c)

def f_prime(c):
    return -Q - k * V / (2 * np.sqrt(c))

def newton_raphson(c0, tol=1e-4, max_iter=100):
    c = c0
    for i in range(max_iter):
        c_new = c - f(c) / f_prime(c)
        if np.abs(c_new - c) < tol:
            return c_new
        c = c_new
    return c

c_nr = newton_raphson(4.0)
print("Question 3c")
print(f"Newton-Raphson converged value of c: {c_nr:.4f} g/m^3")

print("Question 3d")
print("The Newton-Raphson method converged in fewer iterations than the fixed-point iteration method because it has quadratic convergence while fixed-point iteration has linear convergence and qudratic conergence is faster than linear convergence.")