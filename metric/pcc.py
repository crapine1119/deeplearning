import numpy as np
from scipy.stats import pearsonr

a = np.random.rand(100)
b = a**3 + 10

cov = ((a - a.mean()) * (b - b.mean())).sum()

# std_a = ((a - a.mean()) ** 2).sum() ** 0.5
std_a = (a.var() * len(a)) ** 0.5
# std_b = ((b - b.mean()) ** 2).sum() ** 0.5
std_b = (b.var() * len(b)) ** 0.5
r = cov / (std_a * std_b)
r * np.sqrt((len(a) - 2) / (1 - r**2))

print("Cal:   ", r)
print("Scipy: ", pearsonr(a, b))
