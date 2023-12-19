import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import pymc as pm


def gen_data():         # a
    np.random.seed(1)
    n_cluster = [170, 160, 170]
    n_total = sum(n_cluster)
    means = [5, 7, 8]
    std_devs = [0.5, 0.5, 0.5]
    mix = np.random.normal(np.repeat(means, n_cluster),
                           np.repeat(std_devs, n_cluster))

    az.plot_kde(mix)
    plt.show()
    return mix


if __name__ == '__main__':
    data = gen_data()

    clusters = [2, 3, 4]

    models = []
    idatas = []
    for cluster in clusters:            # b
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means',
                              mu=np.linspace(data.min(), data.max(), cluster),
                              sigma=10, shape=cluster,
                              transform=pm.distributions.transforms.ordered)
            sd = pm.HalfNormal('sd', sigma=10)
            y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=data)
            idata = pm.sample(100, tune=200, target_accept=0.9, random_seed=123, return_inferencedata=True)
        idatas.append(idata)
        models.append(model)

    # c
    [pm.compute_log_likelihood(idatas[i], model=models[i]) for i in range(3)]
    comp = az.compare(dict(zip([str(c) for c in clusters], idatas)),
                      method='BB-pseudo-BMA', ic="waic", scale="deviance")

    print(comp.to_string())

    comp = az.compare(dict(zip([str(c) for c in clusters], idatas)),
                      method='BB-pseudo-BMA', ic="loo", scale="deviance")

    print(comp.to_string())

    # the conclusion is that the model that best describes the data is the one with 3 clusters
    