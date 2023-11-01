import csv
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az


def read_input():
    csv_file_path = 'traffic.csv'
    traffic_data_array = []

    with open(csv_file_path, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            traffic_data_array.append(float(row[1]))

    return np.array(traffic_data_array)


if __name__ == "__main__":

    traffic_data = read_input()
    n_traffic_data = len(traffic_data)
    model = pm.Model()

    with model:

        tau1 = 179 # -> 7
        traffic_1 = traffic_data[:tau1]
        alpha_1 = 1.0/traffic_data.mean()
        lambda_1 = pm.Exponential("lambda_1", alpha_1)
        observation_1 = pm.Poisson("obs1", mu=lambda_1, observed=traffic_1)

        tau2 = 239  # -> 8
        traffic_2 = traffic_data[(tau1+1):tau2]
        alpha_2 = 1.0/traffic_data.mean()
        lambda_2 = pm.Exponential("lamda_2", alpha_2)
        observation_2 = pm.Poisson("obs2", mu=lambda_2, observed=traffic_2)

        tau3 = 719  # 16
        traffic_3 = traffic_data[(tau2+1):tau3]
        alpha_3 = 1.0/traffic_data.mean()
        lambda_3 = pm.Exponential("lambda_3", alpha_3)
        observation_3 = pm.Poisson("obs3", mu=lambda_3, observed=traffic_3)

        tau4 = 779
        traffic_4 = traffic_data[(tau3+1):tau4]
        alpha_4 = 1.0/traffic_data.mean()
        lambda_4 = pm.Exponential("lambda_4", alpha_4)
        observation_4 = pm.Poisson("obs4", mu=lambda_4, observed=traffic_4)

        traffic_5 = traffic_data[(tau4+1):]
        alpha_5 = 1.0/traffic_data.mean()
        lambda_5 = pm.Exponential("lamda_5", alpha_5)
        observation_5 = pm.Poisson("obs5", mu=lambda_5, observed=traffic_5)

        trace = pm.sample(10000)

    az.plot_posterior(trace)
    plt.show()
