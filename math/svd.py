import numpy as np

x = np.array([[1, 2], [3, 4], [5, 6]])
rank_diff = abs(x.shape[0] - x.shape[-1])
u, s, v = np.linalg.svd(x)

sigma = np.zeros_like(x).astype(float)
di = np.diag_indices(len(s))
sigma[di] = s


u @ sigma @ v.T
