import numpy as np
import scipy.stats as stats
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

def first_task():    # a
    cheater_wins = 0
    for _ in range(2000):
        n = 0
        m = 0
        first_throw = np.random.uniform(0, 1)

        if first_throw < 0.5:   # p0 starts
            p0_throw = np.random.uniform(0, 1)

            if p0_throw < 0.3:  # p0 throws s
                n = 1

            for _ in range(n+1):
                p1_throw = np.random.uniform(0, 1)
                if p1_throw < 0.5:
                    m += 1

            if n >= m:
                cheater_wins += 1

        else: # p1 starts
            p1_throw = np.random.uniform(0, 1)

            if p1_throw < 0.5:
                m = 1

            for _ in range(m+1):
                p0_throw = np.random.uniform(0, 1)
                if p0_throw < 0.3:
                    n += 1

            if n > m:
                cheater_wins += 1

    return cheater_wins


if __name__ == "__main__":
    print(first_task())
