import numpy as np


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