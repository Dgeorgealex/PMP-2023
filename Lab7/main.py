import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az


def read_data():    # a
    file_path = 'auto-mpg.csv'
    df = pd.read_csv(file_path)
    df = df[df['horsepower'] != '?']

    horsepower = df['horsepower'].values.astype(float)
    mpg = df['mpg'].values.astype(float)

    return np.array(horsepower), np.array(mpg)


def plot_data(horsepower, mpg):     # a
    plt.scatter(horsepower, mpg, marker='o')
    plt.xlabel('horsepower')
    plt.ylabel('mpg')
    plt.title('my_data')
    plt.show()


def main():
    horsepower, mpg = read_data()
    plot_data(horsepower, mpg)

    with pm.Model() as model_regression:        # b
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1)
        eps = pm.HalfCauchy('eps', 5)
        niu = pm.Deterministic('niu', horsepower * beta + alfa)
        mpg_pred = pm.Normal('mpg_pred', mu=niu, sigma=eps, observed=mpg)
        idata = pm.sample(2000, tune=2000, return_inferencedata=True)

    az.plot_trace(idata, var_names=['alfa', 'beta', 'eps'])
    plt.show()

    # c + d
    posterior_data = idata['posterior']
    alpha_m = posterior_data['alfa'].mean().item()
    beta_m = posterior_data['beta'].mean().item()
    print(alpha_m, beta_m)

    plt.scatter(horsepower, mpg, marker='o')
    plt.xlabel('horsepower')
    plt.ylabel('mpg')
    plt.plot(horsepower, alpha_m + beta_m * horsepower, c='k')
    # az.plot_hdi(horsepower, posterior_data['niu'], hdi_prob=0.95, color='k')
    ppc = pm.sample_posterior_predictive(idata, model=model_regression)
    posterior_predictive = ppc['posterior_predictive']
    az.plot_hdi(horsepower, posterior_predictive['mpg_pred'], hdi_prob=0.95, color='gray', smooth=False)
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    main()
