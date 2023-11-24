import numpy as np
import scipy.stats as stats
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx


def first_task():    # a

    # in my implementation n = the score of p0, m = the score of p1

    cheater_wins = 0  # this means that p0 wins
    for _ in range(20000):
        n = 0
        m = 0
        first_throw = np.random.uniform(0, 1)

        if first_throw < 0.5:   # p0 starts
            p0_throw = np.random.uniform(0, 1)

            if p0_throw < 0.33:  # p0 throws s
                n = 1

            for _ in range(n+1):
                p1_throw = np.random.uniform(0, 1)
                if p1_throw < 0.5:
                    m += 1

            if n >= m:
                cheater_wins += 1

        else:  # p1 starts
            p1_throw = np.random.uniform(0, 1)

            if p1_throw < 0.5:
                m = 1

            for _ in range(m+1):
                p0_throw = np.random.uniform(0, 1)
                if p0_throw < 0.33:
                    n += 1

            if n > m:
                cheater_wins += 1

    if cheater_wins > 10000:  # half of the round
        print("Cheater wins - p0")
    else:
        print("Correct wins - p1")


if __name__ == "__main__":
    first_task()    # p1 wins

    # b + c
    # 0 means ban
    # 1 means stema

    model = BayesianNetwork([('first_throw', 'first_round'), ('first_round', 'second_round'), ('first_throw',
                                                                                               'second_round')])

    first_throw = TabularCPD(variable='first_throw', variable_card=2, values=[[0.5], [0.5]])

    first_round = TabularCPD(variable='first_round', variable_card=2, values=[[0.66, 0.5], [0.33, 0.5]],
                             evidence=['first_throw'], evidence_card=[2])

    second_round = TabularCPD(variable='second_round', variable_card=3, values=[[0.5, 0.25, 0.66, 0.44],
                                                                                [0.5, 0.5, 0.33,  0.44],
                                                                                [0, 0.25, 0, 0.11]],
                              evidence=['first_throw', 'first_round'], evidence_card=[2, 2])

    # print(first_throw)
    # print(first_round)
    # print(second_round)

    # columns: p1 throws once, p1 throw twice, p0 throws once, p0 throws twice
    # to compute columns 2 and 3 - used bernoulli distribution
    model.add_cpds(first_throw, first_round, second_round)

    assert model.check_model()

    infer = VariableElimination(model)
    pro_first_round_knowing_second = infer.query(variables=['first_round'], evidence={'second_round': 0})
    print(pro_first_round_knowing_second)   # it is mode likely to have "ban" (0) - which seems true

    pos = nx.circular_layout(model)
    nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
    plt.savefig("bayesian_network.png")
    plt.show()