import numpy as np


x = np.random.randint(1, 10, (3, 3))
print("matrix:")
print(x)
print("Inv:")
print(np.linalg.inv(x))

x @ np.linalg.inv(x)

x.diagonal()
print("Tril:")
print(np.tril(x))
