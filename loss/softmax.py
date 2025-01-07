import numpy as np

x = np.random.rand(8, 64)

x_exp = np.exp(x)

softmax_result = x_exp / x_exp.sum()
