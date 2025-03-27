import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [10, 6]
np.random.seed(42)

state_space = ("sunny", "cloudy", "rainy")

transition_matrix = np.array(((0.6, 0.3, 0.1), (0.3, 0.4, 0.3), (0.2, 0.3, 0.5)))

n_steps = 20000
states = [0]
for i in range(n_steps):
    states.append(np.random.choice((0, 1, 2), p=transition_matrix[states[-1]]))
states = np.array(states)

average_every = 10
time_steps = range(1, n_steps, average_every)

plot = 0
if plot:
    plt.figure()
    for i, label in enumerate(state_space):
        empirical_probabilities = [np.sum(states[:t] == i) / t for t in time_steps]
        plt.plot(time_steps, empirical_probabilities, label=label)
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Empirical probabilitiy")
    plt.show()

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
