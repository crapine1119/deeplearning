import numpy as np

a = np.random.randn(3, 4)
l1_norm = np.abs(a).sum()
l2_norm = np.sqrt((a**2).sum())
l2_norm == np.linalg.norm(a, "fro")
