import numpy as np

# Pred PD
q = np.random.rand(16)
q /= q.sum()

# True PD 1
p1 = np.random.rand(16)
p1 /= p1.sum()
# True PD 2 (Onehot)
p2 = np.eye(16)[np.random.randint(16)]
p2 /= p2.sum()


# shannon entropy
def entropy(p: np.ndarray, q: np.ndarray, scalar: bool = True) -> np.ndarray:
    eps = 1e-8
    ent = -(p * np.log2(q + eps))
    if scalar:
        return ent.sum().round(4)
    return ent


hq = entropy(q, q)
hp1 = entropy(p1, p1)
hp2 = entropy(p2, p2)
print("\nEntropy: ")
print("pred (q): ", hq)
print("gt1 (p1, random): ", hp1)
print("gt2 (p2, onehot): ", hp2)

# cross entropy
hp1_q = entropy(p1, q)
hq_p1 = entropy(q, p1)
hp2_q = entropy(p2, q)
hq_p2 = entropy(q, p2)
print("\nCross Entropy: ")
print("CE {random gt1, pred} (p1): ", hp1_q)
print("CE {pred, random gt1} (p1): ", hq_p1, "[Asym]")
print("CE {onehot gt2, pred} (p2): ", hp2_q)
print("CE {pred, onehot gt2} (p2): ", hq_p2, "[Asym]")


def kl_divergence(p, q):
    return (entropy(p, q, scalar=False) - entropy(p, p, scalar=False)).sum().round(4)


# kl divergence (p1: random true, p2: onehot true, q: pred)
kl_p1_q = kl_divergence(p1, q)
kl_q_p1 = kl_divergence(q, p1)

kl_p2_q = kl_divergence(p2, q)
kl_q_p2 = kl_divergence(q, p2)

print("\nKL Divergence: ")
print("KL {random gt1, pred} (p1): ", kl_p1_q)
print("KL {pred, random gt1} (p1): ", kl_q_p1, "[Asym]")
print("KL {onehot gt2, pred} (p2): ", kl_p2_q)
print("KL {pred, onehot gt2} (p2): ", kl_q_p2, "[Asym]")

# js divergence
m1 = (p1 + q) / 2
m2 = (p2 + q) / 2
kl_p1_m1 = (entropy(p1, m1, scalar=False) - entropy(p1, p1, scalar=False)).sum().round(4)
kl_q_m1 = (entropy(q, m1, scalar=False) - entropy(q, q, scalar=False)).sum().round(4)
js_m1 = (kl_p1_m1 + kl_q_m1) / 2

kl_p2_m2 = (entropy(p2, m2, scalar=False) - entropy(p2, p2, scalar=False)).sum().round(4)
kl_q_m2 = (entropy(q, m2, scalar=False) - entropy(q, q, scalar=False)).sum().round(4)
js_m2 = (kl_p2_m2 + kl_q_m2) / 2

print("\nJS Divergence: ")
print("JS {random gt1, pred} (p1): ", js_m1)
print("JS {onehot gt2, pred} (p2): ", js_m2)
