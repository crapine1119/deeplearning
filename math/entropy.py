import numpy as np

# Pred PD
q1 = np.random.rand(16)
q1 /= q1.sum()

# True PD 1
p1 = np.random.rand(16)
p1 /= p1.sum()
# True PD 2
p2 = np.eye(16)[np.random.randint(16)]
p2 /= p2.sum()


# shannon entropy
def entropy(p: np.ndarray, q: np.ndarray, scalar: bool = True) -> np.ndarray:
    eps = 1e-8
    ent = -(p * np.log2(q + eps))
    if scalar:
        return ent.sum().round(4)
    return ent


hq1 = entropy(q1, q1)
hp1 = entropy(p1, p1)
hp2 = entropy(p2, p2)
print("Entropy: ")
print("pred (q1): ", hq1)
print("gt1 (p1, random): ", hp1)
print("gt2 (p2, onehot): ", hp2)

# cross entropy
hp1_q1 = entropy(p1, q1)
hp2_q1 = entropy(p2, q1)
print("Cross Entropy: ")
print("gt1 (p1, random): ", hp1_q1)
print("gt2 (p2, onehot): ", hp2_q1)

# kl divergence (p1: random true, p2: onehot true, q: pred)
kl_p1_q1 = (entropy(p1, q1, scalar=False) - entropy(p1, p1, scalar=False)).sum().round(4)
kl_p2_q1 = (entropy(p2, q1, scalar=False) - entropy(p2, p2, scalar=False)).sum().round(4)

print("KL Divergence: ")
print("gt1 (p1, random): ", kl_p1_q1)
print("gt2 (p2, onehot): ", kl_p2_q1)
