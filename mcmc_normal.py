import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


## IMPLEMENTING METRPOLIS-HASTINGS ALGORITHM


def log_prob(x):
    """log normal up to a constant"""
    return -0.5 * np.sum(x**2)


def proposal(x, stepsize):
    """Uniform distribution of width stepsize
    centered at x."""
    return np.random.uniform(low=x - 0.5 * stepsize, high=x + 0.5 * stepsize)


def MH_criterion(x_next, x_current, log_prob):
    """Calculate probability of acceptance
    using Metropolis-Hastings criterion."""
    return min(1, np.exp(log_prob(x_next) - log_prob(x_current)))


def sample(x_current, log_prob, stepsize):
    x_next = proposal(x_current, stepsize)

    probability_of_acceptance = MH_criterion(x_next, x_current, log_prob)

    accept = np.random.random() < probability_of_acceptance

    if accept:
        return accept, x_next
    else:
        return accept, x_current


n_steps = 10000
stepsize = 4
x0 = 1
chain = [proposal(x0, stepsize)]
acceptance = 0
for i in range(n_steps):
    acceptance_i, chain_i = sample(chain[-1], log_prob, stepsize)
    chain.append(chain_i)
    acceptance += acceptance_i

acceptance_rate = acceptance / n_steps
print(f"{acceptance_rate=}")

print(f"Last ten states of chain : {chain[-10:]}")

plt.figure()
plt.hist(chain, bins=50, density=True, label="MCMC samples")
plt.show()
