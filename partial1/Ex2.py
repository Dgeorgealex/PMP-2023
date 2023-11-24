import numpy as np
import scipy.stats as stats
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az


if __name__ == "__main__":
    niu_real = 34
    sigma_real = 5
    data = np.random.normal(niu_real, sigma_real, size=200) # a

    # explicatie alegere distributii
    # "Conjugata a priori a unei distributii Gaussiene este tot o distributie Gaussiana"
    # Primul exemplu curs 5 este foarte asemanator cu problema data
    # deviatia standard nu poate sa fie negativa, de aceea half normal (media 0, val pozitive)
    # pentru medie putem presupune ca media acesteia va fi in jur de media datelor

    with pm.Model() as my_model:     # b
        avg = data.mean()
        miu = pm.Normal('miu', avg, 10)
        sigma = pm.HalfNormal('sigma', 10)
        data_pred = pm.Normal('data_pred', mu=miu, sigma=sigma, observed=data)
        idata = pm.sample(2000, tune=2000, return_inferencedata=True)

    az.plot_posterior(idata, var_names=['miu'])  # c - da, corespunde asteptarilor
    plt.savefig("posterior_miu")
    plt.show()
