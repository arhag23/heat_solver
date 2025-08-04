import sys

import numpy as np

def rel_err(data: np.ndarray, truth: np.ndarray):
    return np.array([np.max(np.abs(est - act) / act) for est, act in zip(data, truth)])

# with open(data_file) as data, open(truth_file) as truth:
#     for est_line, act_line in zip(data, truth):
#         est = np.fromstring(est_line, dtype=np.float64, sep=" ")
#         act = np.fromstring(act_line, dtype=np.float64, sep=" ")
#
#         err = np.max(np.abs(est - act) / act)
#         errs.append(err)
#
# print(errs)
