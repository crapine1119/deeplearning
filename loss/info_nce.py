import matplotlib.pyplot as plt
import numpy as np

# assume for all other labels
query = np.random.randn(4, 128)
# key = np.random.randn(4, 128)
key = query.copy()

query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)

plt.plot(query[0:1].T, label="raw")
plt.plot(query_norm[0:1].T, label="norm")
plt.legend()

key_norm = key / np.linalg.norm(key, axis=1, keepdims=True)
temperature = 0.1

sim_matrix = np.exp(query_norm @ key_norm.T / temperature)

each_loss = np.diagonal(sim_matrix) / (sim_matrix.sum() + 1e-8)
info_nce = -(np.log(each_loss)).sum()
