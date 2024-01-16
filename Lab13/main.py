import numpy as np
import scipy.stats as stats
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

centered_model = az.load_arviz_data("centered_eight")
non_centered_model = az.load_arviz_data("non_centered_eight")


def a():
    print("For the centered model:")
    print(f"# chains = {centered_model.posterior.chain.size}")
    print(f"# of draws = {centered_model.posterior.draw.size}")

    print("For the non centered model:")
    print(f"# chains = {non_centered_model.posterior.chain.size}")
    print(f"# of draws = {non_centered_model.posterior.draw.size}")

    az.plot_posterior(centered_model, var_names=['mu', 'tau'])
    az.plot_posterior(non_centered_model, var_names=['mu', 'tau'])
    plt.show()


def b():
    summaries = pd.concat([az.summary(centered_model, var_names=['mu', 'tau']),
                           az.summary(non_centered_model, var_names=['mu', 'tau'])])
    summaries.index = ['centered_mu', 'centered_tau', 'non_centered_mu', 'non_centered_tau']
    print(summaries)  # r_hat

    az.plot_autocorr(centered_model, var_names=['mu', 'tau'])
    az.plot_autocorr(non_centered_model, var_names=['mu', 'tau'])
    plt.show()


def c():
    print(f"# divergences for centered: {centered_model.sample_stats.diverging.sum()}")
    print(f"# divergences for non_centered_data {non_centered_model.sample_stats.diverging.sum()}")

    az.plot_parallel(centered_model, var_names=['mu', 'tau'])  # in the neighbourhood of 0

    az.plot_parallel(non_centered_model, var_names=['mu', 'tau'])
    plt.show()


if __name__ == "__main__":
    a()
    b()
    c()
    