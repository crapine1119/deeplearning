import numpy as np

min_mi = float("inf")
max_mi = float("-inf")

for _ in range(100):
    m = np.random.rand(3, 4)
    m /= m.sum()

    # given x, y probability matrix
    hxy = (m * np.log2(m)).sum()

    px = m.sum(axis=0)
    py = m.sum(axis=1)

    hx = (px * np.log2(px)).sum()
    hy = (py * np.log2(py)).sum()

    mi = hxy - hx - hy
    min_mi = min(mi, min_mi)
    max_mi = max(mi, max_mi)
print(min_mi)
print(max_mi)
