import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

n_value = 10000
np.random.seed(1)

# gamma(4, 3)
first_server = stats.gamma.rvs(4, 0, 1/3, size=n_value)

# gamma(4, 2)
second_server = stats.gamma.rvs(4, 0, 1/2, size=n_value)

#gamma(5, 2)
third_server = stats.gamma.rvs(5, 0, 1/2, size=n_value)

#gamma(5, 3)
forth_server = stats.gamma.rvs(5, 0, 1/3, size=n_value)

#exponential
latency = stats.expon.rvs(0, 4, size=n_value)

x = 0.25 * first_server + 0.25 * second_server + 0.30 * third_server + 0.20 * forth_server + latency

az.plot_posterior({'x': x})
plt.show()

good = 0
for x_value in x:
    if x_value > 3:
        good = good + 1

probability = good / n_value
print(f"Probability of x greater than 3 is {probability}")
