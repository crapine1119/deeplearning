import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

a = np.arange(20)
b = -(a**3) + 13 * a - 50

dot = a @ b

a_norm = (a**2).sum() ** 0.5  # np.linalg.norm(a)
b_norm = (b**2).sum() ** 0.5

sim = dot / a_norm / b_norm
print(sim)

cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
