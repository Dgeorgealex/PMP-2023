import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az


def read_data():
    file_path = "Prices.csv"
    df = pd.read_csv(file_path)
    y = df["Price"].values.astype(float)
    x1 = df["Speed"].values.astype(float)
    x2 = df["HardDrive"].values.astype(float)
    x2 = np.log(x2)     # preprocess
    return y, x1, x2


def main():
    y_real, x1, x2 = read_data()

    # plot y_real:
    plt.hist(y_real, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

    with pm.Model() as my_model:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        beta1 = pm.Normal('beta1', mu=0, sigma=1)
        beta2 = pm.Normal('beta2', mu=0, sigma=1)
        miu = pm.Deterministic('miu', x2 * beta2 + x1 * beta1 + alfa)
        sigma = pm.HalfNormal('sigma', sigma=11)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=sigma)
        idata = pm.sample(20000, tune=20000, return_inferencedata=True)

    az.plot_posterior(idata, var_names=['miu', 'sigma'])


if __name__ == "__main__":
    main()