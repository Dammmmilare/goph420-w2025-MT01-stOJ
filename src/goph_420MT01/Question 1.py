#Question 1 Calculations:
#-----------------------

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
