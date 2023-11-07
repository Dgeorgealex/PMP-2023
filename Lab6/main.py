import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    Y_values = np.array([0, 5, 10])
    theta_values = np.array([0.2, 0.5])

    with pm.Model() as my_model:
        for y in Y_values:
            for theta in theta_values:
                n = pm.Poisson(f"n[y={y}, theta={theta}]", 10)
                obs = pm.Binomial(f"obs[y={y}, theta={theta}]", n=n, p=theta, observed=y)

        idata = pm.sample(1000)

    az.plot_posterior(idata)
    print(az.summary(idata))
    plt.show()
