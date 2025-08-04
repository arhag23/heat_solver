import os

import matplotlib.pyplot as plt
import numpy as np

from scripts.sanity import ground_truth
from scripts.error import rel_err

# For a fixed initial condition, graph error over time as we modify del_L, del_T
L = 5.0
T = 5.0
PDEL_L = 0.1
PDEL_T = 0.05

tests = [(0.1, 0.005), (0.1, 0.0025), (0.1, 0.001), (0.1, 0.0005), (0.1, 0.0001)]
tests_2 = [(0.1, 0.1), (0.1, 0.05), (0.1, 0.01), (0.1, 0.005), (0.1, 0.001)]

truth = ground_truth(L, T, PDEL_L, PDEL_T)
t_vals = np.arange(0.0, T + PDEL_T / 2, PDEL_T)

fig, axes = plt.subplots(2)

for del_L, del_T in tests:
    os.system(f"./build/heat_solver {L} {T} {del_L} {del_T} {PDEL_L} {PDEL_T} finite_diff temp1.out")
    with open("temp1.out") as file:
        data = np.array([np.fromstring(line, dtype=np.float64, sep=" ") for line in file])

    err = rel_err(data, truth)
    axes[0].plot(t_vals, err, label=f"{del_T}")

axes[0].legend()
    
for del_L, del_T in tests_2:
    os.system(f"./build/heat_solver {L} {T} {del_L} {del_T} {PDEL_L} {PDEL_T} finite_diff_scaled temp2.out")
    with open("temp2.out") as file:
        data = np.array([np.fromstring(line, dtype=np.float64, sep=" ") for line in file])

    err = rel_err(data, truth)
    axes[1].plot(t_vals, err, label=f"{del_T}")

axes[1].legend()

plt.show()
