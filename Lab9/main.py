import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az
import pymc as pm


def read_data():
    file_path = 'Admission.csv'
    df = pd.read_csv(file_path)
    admission = df['Admission']
    gre = df['GRE']
    gpa = df['GPA']

    plt.scatter(gre[admission == 0], gpa[admission == 0], color='red', label='Not Admitted')
    plt.scatter(gre[admission == 1], gpa[admission == 1], color='green', label='Admitted')

    plt.xlabel('GRE Score')
    plt.ylabel('GPA')
    plt.title('Scatter Plot of GPA vs GRE with Admission Status')

    plt.legend()

    plt.show()

    return admission, gre, gpa


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

    norm_gre, gre_min, gre_max = min_max_normalization(gre)
    norm_gpa, gpa_min, gpa_max = min_max_normalization(gpa)

    with pm.Model() as my_model:
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=2)
        beta2 = pm.Normal('beta2', mu=0, sigma=2)

        miu = alfa + beta1 * norm_gre + beta2 * norm_gpa
        theta = pm.Deterministic('theta', pm.math.sigmoid(miu))
        sep = pm.Deterministic('sep', -alfa/beta2 - beta1/beta2 * norm_gre)

        adm = pm.Bernoulli('adm', p=theta, observed=admission)

        idata = pm.sample(2000, return_inferencedata=True)

    # linear separator
    alfa_m = idata['posterior']['alfa'].mean().item()
    beta1_m = idata['posterior']['beta1'].mean().item()
    beta2_m = idata['posterior']['beta2'].mean().item()

    plt.scatter(norm_gre, norm_gpa, c=[f"C{x}" for x in admission])
    plt.xlabel('gre_norm')
    plt.ylabel('gpa_norm')

    plt.plot(norm_gre, -alfa_m/beta2_m - beta1_m/beta2_m * norm_gre, c='k')

    az.plot_hdi(norm_gre, idata['posterior']['sep'], hdi_prob=0.94, color='k')

    plt.show()

    alfa_p = idata['posterior']['alfa']
    beta1_p = idata['posterior']['beta1']
    beta2_p = idata['posterior']['beta2']

    # 3
    new_gre = normalize_element(550, gre_min, gre_max)
    new_gpa = normalize_element(3.5, gpa_min, gpa_max)
    new_prob = alfa_p + beta1_p * new_gre + beta2_p * new_gpa

    stacked = az.extract(new_prob)

    # print(stacked)

    new_prob_values = stacked.x.values
    new_prob_values = 1 / (1 + np.exp(-new_prob_values))

    # print(type(new_prob_values))
    # print(new_prob_values)
    # print(new_prob_values.shape)

    hdi_prob = az.hdi(new_prob_values, hdi_prob=0.9)
    print(hdi_prob)

    # 4
    new_gre = normalize_element(550, gre_min, gre_max)
    new_gpa = normalize_element(3.2, gpa_min, gpa_max)
    new_prob = alfa_p + beta1_p * new_gre + beta2_p * new_gpa
    stacked = az.extract(new_prob)
    new_prob_values = stacked.x.values
    new_prob_values = 1 / (1 + np.exp(-new_prob_values))
    hdi_prob = az.hdi(new_prob_values, hdi_prob=0.9)
    print(hdi_prob)

    # Cum era de asteptat, probabilitatea ca al doilea elev are sanse mai mici sa intre.
    # Ceea ce are sens, deoarece cu cat un elev are gre-ul si gpa-ul mai mare cu atat isi mareste sansele
    # sa intre la facultate.


if __name__ == "__main__":
    main()