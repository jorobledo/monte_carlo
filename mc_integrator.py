import numpy as np
import ast 

def integrator(f, h, N, limits=None):
    p = np.random.normal(0, 1, N)  # Sample from h(x) ~ N(0,1)
    def g(f, x, limits=None):
        if limits is None:
            return f(x)
        else:
            return np.where((limits[0] < x) & (x < limits[1]), f(x), 0)
    return np.mean(g(f, p, limits) / h(p))  # Importance correction


def h(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-(x**2) / 2.0)  # Standard normal


# expr = "np.sin(x)"
expr = input("Enter a function of x (e.g., x**2, np.sin(x), np.exp(x)): \n")
limits = ast.literal_eval(input("Define limits of integration in format (upper, lower) (e.g. (0,1), (-1,1)):\n"))

f = lambda x: eval(expr, {"x":x, "np":np})

print(f"Integral of {expr} between {limits} is {integrator(f, h, 10_000_000, limits=limits):4f}")  
