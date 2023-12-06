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
    df['Premium10'] = np.where(df['Premium'] == 'yes', 5, 0)
    x3 = df['Premium10'].values.astype(float)
    return y, x1, x2, x3


def main():

    # I am writing in english because pycharm has autocorrect enabled and if I don't it underlines all
    # the misspelled words

    y_real, x1, x2, x3 = read_data()  # x1 is speed, x2 is memory, x3 is yes/no

    plt.hist(y_real, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.show()

    with pm.Model() as my_model:     # 1. the distributions are "slab informative" because the standard deviation is big
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        miu = pm.Deterministic('miu', x2 * beta2 + x1 * beta1 + alfa)
        sigma = pm.HalfCauchy('sigma', 5)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=sigma, observed=y_real)
        idata = pm.sample(1000, tune=1000, return_inferencedata=True)

    az.plot_posterior(idata, var_names=["beta1", "beta2"], hdi_prob=0.95)   # 2.
    plt.show()
    hdi_info = az.hdi(idata, var_names=["beta1", "beta2"], hdi_prob=0.95).values()
    print(hdi_info)

    # 3.
    # YES
    # Most of the values are different from 0
    # (0 is not inside the HDI interval)
    # which means that both "Speed" and "HardDrive" make an impact on the
    # price

    # 4., 5. I do not know exactly what I need to do
    # 4.
    posterior = idata['posterior']
    expected_price = posterior['beta1'] * 33 + posterior['beta2'] * np.log(560) + posterior['alfa']
    hdi_expected_price = az.hdi(expected_price, hdi_prob=0.95).values()
    print(hdi_expected_price)

    # 5 correct, I think
    stacked = az.extract(idata)
    beta1_values = stacked.beta1.values
    beta2_values = stacked.beta2.values
    alfa_values = stacked.alfa.values
    sigma_values = stacked.sigma.values

    price_values = np.random.normal(beta1_values * 33 + beta2_values * np.log(560) + alfa_values, sigma_values)
    hdi_expected_price_values = az.hdi(price_values, hdi_prob=0.95)
    print(hdi_expected_price_values)


    # 6. (Bonus) We will x3 (Premium - 5, 0) to the model and analyse its posterior distribution
    with pm.Model() as my_model_premium:
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        beta3 = pm.Normal('beta3', mu=0, sigma=10)
        miu = pm.Deterministic('miu', x3 * beta3 + x2 * beta2 + x1 * beta1 + alfa)
        sigma = pm.HalfCauchy('sigma', 5)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=sigma, observed=y_real)
        idata = pm.sample(1000, tune=1000, return_inferencedata=True)

    az.plot_posterior(idata, var_names=["beta1", "beta2", "beta3"], hdi_prob=0.95)
    plt.show()
    hdi_info_premium = az.hdi(idata, var_names=["beta1", "beta2", "beta3"], hdi_prob=0.95)
    print(hdi_info_premium)
    # YES
    # Most of the values of beta3 are different from 0 which means that "Premium" has an impact.
    # Even if 0 is contained in the HDI, beta3 is significant different from 0.
    # I used [0 - no, 5 - yes] because it was a closer range to the other variables which could
    # produce more accurate results.


if __name__ == "__main__":
    main()