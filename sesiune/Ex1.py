import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor as pt


def scatter_plot(x, y):     # from course - to view data
    plt.figure(figsize=(15, 5))
    for idx, x_i in enumerate(x.T):
        plt.subplot(1, 3, idx+1)
        plt.scatter(x_i, y)
        plt.xlabel(f'x_{idx+1}')
        plt.ylabel(f'y', rotation=0)

    plt.subplot(1, 3, idx+2)
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlabel(f'x_{idx}')
    plt.ylabel(f'x_{idx+1}', rotation=0)
    plt.savefig("initial_data.png")
    plt.show()


def read_data():
    titanic = pd.read_csv("Titanic.csv")

    titanic = titanic[titanic['Age'].notna()]       # handle missing values

    age = titanic['Age'].values.astype(float)
    p_class = titanic['Pclass'].values.astype(float)
    survived = titanic['Survived'].values.astype(float)
    x = np.column_stack((age, p_class))
    y = survived

    scatter_plot(x, y)

    return x, y


def main():
    x, y = read_data()
    # x first col = age
    # x second col = p_class

    # standardization
    x_age_mean = x[:,0].mean()
    x_age_std = x[:,0].std()
    x_class_mean = x[:,1].mean()
    x_class_std = x[:,1].std()

    x[:,0] = (x[:,0] - x_age_mean) / x_age_std
    x[:,1] = (x[:,1] - x_class_mean) / x_class_std

    # Logistical regression
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=2)
        x_shared = pm.MutableData('x_shared', x)

        mu = pm.Deterministic('mu', alpha + pm.math.dot(x_shared, beta))
        theta = pm.Deterministic("theta", pm.math.sigmoid(mu))

        bd = pm.Deterministic("bd", -alpha / beta[1] - beta[0] / beta[1] * x[:,0])

        y_pred = pm.Bernoulli("y_pred", p=theta, observed=y)

        idata = pm.sample(2000, return_inferencedata = True)

    az.plot_forest(idata, hdi_prob=0.95, var_names='beta')
    plt.savefig("plot_forest.png")
    plt.show()
    # according to this graph beta[1] is farther from 0 (with average -0.3) which means that
    # class hase a greater impact. (both variables are standardised)

    idx = np.argsort(x[:,0])
    bd = idata.posterior["bd"].mean(("chain", "draw"))[idx]

    plt.scatter(x[:,0], x[:,1], c=[f"C{int(x)}" for x in y])
    plt.xlabel("Age")
    plt.ylabel("Class")
    plt.savefig("plot_colors.png")
    plt.show()

    plt.scatter(x[:,0], x[:,1], c=[f"C{int(x)}" for x in y])
    plt.plot(x[:,0][idx], bd, color='k')
    plt.xlabel("Age")
    plt.ylabel("Class")
    plt.savefig("plot_decision_border.png")
    plt.show()

    obs = [(30-x_age_mean)/x_age_std, (2-x_class_mean)/x_class_std]
    pm.set_data({"x_shared":[obs]}, model=model)
    ppc = pm.sample_posterior_predictive(idata, model=model,var_names=["theta"])
    y_ppc = ppc.posterior_predictive['theta'].stack(sample=("chain", "draw")).values
    az.plot_posterior(y_ppc, hdi_prob=0.9)
    plt.savefig("plot_hdi_interval.png")
    plt.show()


if __name__ == "__main__":
    main()