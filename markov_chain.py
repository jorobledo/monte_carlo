import numpy as np
import matplotlib.pyplot as plt

state_space = ("sunny", "cloudy", "rainy")

transition_matrix = np.array(((0.6, 0.3, 0.1), (0.3, 0.4, 0.3), (0.2, 0.3, 0.5)))

n_steps = 20000
states = [0]
for i in range(n_steps):
    states.append(np.random.choice((0, 1, 2), p=transition_matrix[states[-1]]))
states = np.array(states)

average_every = 10
time_steps = range(1, n_steps, average_every)

plt.figure(figsize=(8, 5))
for i, label in enumerate(state_space):
    empirical_probabilities = [np.sum(states[:t] == i) / t for t in time_steps]
    plt.plot(time_steps, empirical_probabilities, label=label)
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Empirical probabilitiy")
plt.show()
