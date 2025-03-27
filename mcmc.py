import numpy as np
import matplotlib.pyplot as plt

# Target distribution: Normal(3,1)
# def target_distribution(x):
#     return 1/(np.sqrt(2*np.pi))*np.exp(-0.5 * (x - 3) ** 2)  # Unnormalized Gaussian
def target_distribution(x):
    return np.where(np.abs(x)<2,(4-x**2)*3/32,0) 

# Metropolis-Hastings MCMC
def metropolis_hastings(num_samples, proposal_std=1.0):
    samples = []
    x = np.random.randn()  # Initialize at a random state

    for _ in range(num_samples):
        x_new = x + np.random.normal(0, proposal_std)  # Propose new state
        acceptance_ratio = min(1, target_distribution(x_new) / target_distribution(x))

        if np.random.rand() < acceptance_ratio:  # Accept or reject
            x = x_new

        samples.append(x)

    return np.array(samples)

# Run MCMC
num_samples = 1000000
samples = metropolis_hastings(num_samples)

# Plot results
plt.figure(figsize=(8,5))
plt.hist(samples, bins=100, density=True, alpha=0.6, color='blue', label="MCMC Samples")

# Compare with actual distribution
x_vals = np.linspace(-5, 5, 100)
y_vals = target_distribution(x_vals)
plt.plot(x_vals, y_vals, 'r-', label="Target Distribution")

plt.legend()
plt.xlabel("x")
plt.ylabel("Density")
plt.title("MCMC Sampling using Metropolis-Hastings")
plt.show()

