import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt


def monte_carlo_sim(n):
    x = stats.geom.rvs(0.3, size=n)
    y = stats.geom.rvs(0.5, size=n)

    bigger = x > y**2

    p = sum(bigger) / n

    return p


if __name__ == "__main__":

    values = []
    for _ in range(30):
        v = monte_carlo_sim(10000)
        values.append(v)

    mean = np.mean(values)
    std = np.std(values)

    print(f"Mean is: {mean}")
    print(f"Standard deviation is: {std}")
