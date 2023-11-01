import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats


def generate_data(n):
    alfa = 3.0
    client_wait_time = []
    clients_number = stats.poisson.rvs(20.0, size=n)
    for c_nr in clients_number:
        order_time = stats.norm.rvs(2.0, 0.5, size=c_nr)
        preparation_time = stats.expon.rvs(alfa, size=c_nr)
        wait = order_time + preparation_time
        client_wait_time.append(wait.mean())  # Calculate the mean wait time for each client

    return client_wait_time


if __name__ == "__main__":
    np.random.seed(1)
    my_data = generate_data(1000)

    model = pm.Model()
    with model:
        client_time = pm.Poisson("client_number", 20, shape=len(my_data))
        time_order = pm.Normal("time_order", 2.0, 0.5, shape=len(my_data))
        alfa = pm.Uniform("alfa", 2, 4)
        time_preparation = pm.Normal("time_preparation", mu=alfa, shape=len(my_data))
        wait_time = pm.math.sum([time_order, time_preparation], axis=0)
        my_wait = wait_time.mean()
        obs = pm.Normal("obs", mu=my_wait, sigma=0.01,  observed=my_data)

    with model:
        trace = pm.sample(100)

    az.plot_posterior(trace, var_names=["alfa"])
    plt.show()
