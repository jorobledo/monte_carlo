import numpy as np
import matplotlib.pyplot as plt
from mcmc_normal import sample, proposal


def log_gaussian(x, mu, sigma):
    return -0.5 * np.sum((x - mu)**2) / sigma**2 - np.log(np.sqrt(2 * np.pi * sigma**2))

class BivariateNormal:

    def __init__(self, mu_1, mu_2, sigma_1, sigma_2):
        self.mu_1, self.mu_2 = mu_1, mu_2
        self.sigma_1, self.sigma_2 = sigma_1, sigma_2
        self.n_variates = 2

    def log_p_x(self, x):
        return log_gaussian(x, self.mu_1, self.sigma_1)

    def log_p_y(self, x):
        return log_gaussian(x, self.mu_2, self.sigma_2)

    def log_prob(self, x):
        covariance_matrix = np.array([[self.sigma_1**2, 0], [0, self.sigma_2**2]])
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        kernel = -0.5 * (x - self.mu_1) @ inv_covariance_matrix @ (x - self.mu_2).T
        normalization = np.log(np.sqrt(2 * np.pi) ** self.n_variates * np.linalg.det(covariance_matrix))

        return kernel - normalization

def sample_gibbs(current_state, bivariate_dist, stepsizes):

    x_current, y_current = current_state

    x_current = np.array([x_current])
    y_current = np.array([y_current])

    p_x_y = bivariate_dist.log_p_x
    p_y_x = bivariate_dist.log_p_y

    accept_x, x_new = sample(x_current, p_x_y, stepsizes[0])
    accept_y, y_new = sample(y_current, p_y_x, stepsizes[1])

    return (accept_x, accept_y), (x_new[0], y_new[0])

if __name__ == "__main__":

    bivariate_normal = BivariateNormal(mu_1=0.0, mu_2=0.0, sigma_1=1.0, sigma_2 = 0.3)
    
    # Make 2D density plot

    x = np.linspace(-3,3,100)
    X, Y = np.meshgrid(x,x)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([np.exp(bivariate_normal.log_prob(grid_point)) for grid_point in grid_points]).reshape(X.shape)
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.colorbar(label="Probability density")
    plt.title("Original Density")

    # sample using our Metropolis-Hasting algorithm with uniform proposal distribution

    np.random.seed(42) 

    n_steps = 100000
    stepsize = 1
    x_0 = np.array([0,0])
    chain = [proposal(x_0, stepsize)]
    acceptance = 0
    log_prob = lambda x: log_gaussian(x, mu=0, sigma=1)
    for i in range(n_steps):
        acceptance_i, chain_i = sample(chain[-1], log_prob, stepsize)
        chain.append(chain_i)
        acceptance += acceptance_i
    
    acceptance_rate = acceptance / n_steps
    print(f"{acceptance_rate=}")

    print(f"Last ten states of chain : {chain[-10:]}")

    x, y = zip(*chain)
    plt.subplot(132)
    plt.hist2d(x, y, bins=50, cmap="viridis")
    plt.colorbar(label="Frequency")
    plt.title("Metropolis-Hasting sampler")
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    
    # sampling using Gibbs sampler 
    # Since variables are independent, the example is simple

    init_x, init_y = (0, 0)
    chain = [(0,0)]
    acceptances = []
    stepsizes = [1, 0.1]

    for _ in range(n_steps):
        accept, new_state = sample_gibbs(chain[-1], bivariate_normal, stepsizes)
        chain.append(new_state)
        acceptances.append(accept)

    acceptance_rates = np.mean(acceptances, 0)
    print("Acceptance rates: x: {:.3f}, y: {:.3f}".format(acceptance_rates[0],
                                                          acceptance_rates[1]))
    
    chain = np.array(chain)
    plt.subplot(133)
    plt.hist2d(chain[:,0], chain[:,1], bins=50, cmap="viridis")
    plt.title("Gibbs sampler")
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.colorbar(label="Frequency")
    plt.show()



