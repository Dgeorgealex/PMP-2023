import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az


def read_data():
    file_path = 'Admission.csv'
    df = pd.read_csv(file_path)
    admission = df['Admission']
    gre = df['GRE']
    gpa = df['GPA']

    return admission, gre, gpa


def sigmoid(x):
    return 1 / ( 1 + pm.math.exp(-x) )


def min_max_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data, min_val, max_val


def min_max_denormalization(normalized_data, min_val, max_val):
    denormalized_data = normalized_data * (max_val - min_val) + min_val
    return denormalized_data


def normalize_element(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def main():
    admission, gre, gpa = read_data()

    # alfa + beta1 * gre + beta2 * gpa

    # normalising data
    norm_gre, min_gre, max_gre = min_max_normalization(gre)
    norm_gpa, min_gpa, max_gpa = min_max_normalization(gpa)


    with pm.Model() as my_model:
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=2)
        beta2 = pm.Normal('beta2', mu=0, sigma=2)

        miu = alfa + beta1 * norm_gre + beta2 * norm_gpa
        theta = pm.Deterministic('theta', sigmoid(miu))
        bd = pm.Deterministic('bd', -alfa/beta1 - beta1/beta2 * norm_gre)

        y1 = pm.Bernoulli('y1', p=theta, observed=admission)

        idata = pm.sample(2000, tune=2000, return_inferencedata=True)

    plt.scatter(norm_gre, norm_gpa, c=[f'C{x}' for x in admission])
    az.plot_hdi(norm_gre, idata.posterior['bd'], color='k')
    plt.xlabel("norm_gre")
    plt.ylabel("norm_gpa")
    plt.show()

if __name__ == "__main__":
    main()