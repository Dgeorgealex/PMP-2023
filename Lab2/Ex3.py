import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

n_repetitions = 100
n_throws = 10

average_ss = 0
average_sb = 0
average_bs = 0
average_bb = 0

for _ in range(n_repetitions):
    ss = 0
    sb = 0
    bs = 0
    bb = 0
    for _ in range(n_throws):
        first_throw = np.random.uniform(0, 1)
        second_throw = np.random.uniform(0, 1)
        if first_throw < 0.5 and second_throw < 0.3:
            ss = ss + 1
        elif first_throw < 0.5 and second_throw >= 0.3:
            sb = sb + 1
        elif first_throw >= 0.5 and second_throw < 0.3:
            bs = bs + 1
        else:
            bb = bb + 1
    average_ss = average_ss + ss
    average_sb = average_sb + sb
    average_bs = average_bs + bs
    average_bb = average_bb + bb

average_ss = average_ss / n_repetitions
average_sb = average_sb / n_repetitions
average_bs = average_bs / n_repetitions
average_bb = average_bb / n_repetitions

print(f"ss: {average_ss}, sb: {average_sb}, bs: {average_bs}, bb: {average_bb}")

d = {'ss': average_ss, 'sb': average_sb, 'bs': average_bs, 'bb': average_bb}
plt.bar(d.keys(), d.values())
plt.show()
