import numpy as np

def fourier(L: float, del_L: float, t: float):
    x = np.arange(0.0, 2 * L + del_L / 2, del_L, dtype=np.float128)
    x_cos = np.pi * x / L
    x_comp = np.array([np.cos(n * x_cos) / (n * n)  for n in range(1, 40, 2)])
    t_comp = np.exp(-t * np.array([(n * np.pi / L) ** 2 for n in range(1, 40, 2)], dtype=np.float128))
    return L / 2 + 4 * L / (np.pi * np.pi) * np.sum(x_comp.T * t_comp, axis=1)

def ground_truth_file(L: float, T: float, del_L: float, del_T: float, file_name: str):
    with open(file_name, mode="w") as file:
        lines = [
            " ".join([f"{x:.6f}" for x in fourier(L / 2, del_L, t)]) + "\n" for t in np.arange(del_T, T + del_T / 2, del_T)
        ]
        file.write(" ".join(f"{x:.6f}" for x in (np.abs(-np.arange(0.0, L + del_L / 2, del_L) + L / 2))) + "\n")
        file.writelines(lines)

def ground_truth(L: float, T: float, del_L: float, del_T: float):
        init = np.abs(-np.arange(0.0, L + del_L / 2, del_L) + L / 2)
        data = np.array([fourier(L / 2, del_L, t) for t in np.arange(del_T, T + del_T / 2, del_T)])
        return np.vstack([[init], data])
