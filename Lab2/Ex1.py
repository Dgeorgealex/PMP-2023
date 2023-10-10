import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

n_value = 10000
np.random.seed(1)

mechanic1 = stats.expon.rvs(0, 4, size=n_value)
mechanic2 = stats.expon.rvs(0, 6, size=n_value)

x = (mechanic1 * 4 + mechanic2 * 6) / 10

az.plot_posterior({'mechanic1': mechanic1, 'mechanic2': mechanic2, 'x': x})
plt.show()

print(f"Variation of x is {stats.variation(x)}")
