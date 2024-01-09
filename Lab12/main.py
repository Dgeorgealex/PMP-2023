import numpy as np
import scipy.stats as stats
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az


def posterior_grid(grid_points, heads, tails):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)   # uniform prior
    # prior = (grid <= 0.5).astype(int)  # random distribution 1
    # prior = abs(grid - 0.5)    # random distribution 2
    # prior = (grid >= 0.5).astype(int)

    prior = prior / prior.sum()    # normalization

    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


def monte_carlo_pi(n):
    x, y = np.random.uniform(-1, 1, size=(2, n))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum() * 4 / n
    error = abs((pi - np.pi)/pi) * 100
    return pi, error


def metropolis(func, draws=10000):
    """A very simple Metropolis implementation"""
    trace = np.zeros(draws)
    old_x = 0.5  # func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace


def ex1():
    data = np.repeat([0, 1], [10, 3])
    points = 40
    h = data.sum()
    t = len(data) - h
    grid, posterior = posterior_grid(points, h, t)
    plt.plot(grid, posterior, 'o-')
    plt.title(f'heads = {h}, tails = {t}')
    plt.yticks([])
    plt.xlabel('θ')
    plt.show()


def ex2():
    n_values = [100, 10000, 1000000]
    for n in n_values:
        errors = []
        for _ in range(100):
            pi, error = monte_carlo_pi(n)
            errors.append(error)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        print(n)
        print(f"mean: {mean_error}, std: {std_error}")
        plt.errorbar(n, mean_error, yerr=std_error, fmt='o', label=f'n = {n}')

    plt.xscale('log')
    plt.xlabel('Number of Points (n)')
    plt.ylabel('Mean Error')
    plt.legend()
    plt.show()

    # The bigger the N the smaller the error and the standard deviation


def ex3():
    plt.figure(figsize=(12, 10))
    n_trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
    data = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48]
    theta_real = 0.35
    beta_params = [(1, 1), (20, 20), (1, 4)]

    for idx, N in enumerate(n_trials):
        plt.subplot(4, 3, idx + 1)

        y = data[idx]

        # Plot posterior_grid with 100 grid points
        grid_10000, posterior_10000 = posterior_grid(10000, y, N - y)
        plt.plot(grid_10000, posterior_10000, 'o-', label='Posterior Grid (10000 points)')

        for (a_prior, b_prior) in beta_params:
            func = stats.beta(a_prior + y, b_prior + N - y)
            p_theta_given_y = metropolis(func)
            plt.hist(p_theta_given_y, bins=200, density=True, label=f'Beta({a_prior + y}, {b_prior + N - y})')

        plt.axvline(theta_real, ymax=0.3, color='k', linestyle='--', label='True θ')
        plt.title(f'{N} trials, {y} heads')
        plt.xlabel('θ')
        plt.ylabel('Probability Density')
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ex1()
    # ex2()
    ex3()
