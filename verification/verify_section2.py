# GenAI disclosure: Claude (Anthropic, model claude-opus-5-5) wrote this script. I directed the
# work and reviewed it.
"""Numerical checks for the Section 2 answers (self-attention), run on CPU.

2.1   argmax attention on the given keys, values, and query, plus an autograd check that no
      gradient reaches the query or the keys through argmax (softmax for comparison)
2.2.1 copying: alpha_j -> 1 as the query lines up with key j
2.2.2 M = 2 * sum_i a_i a_i^T recovers v_a from the average of v_a and v_b
2.2.3 q = c (k_a + k_b) gives an output that approaches (v_a + v_b) / 2 as c grows
2.3   Monte Carlo check of E[q k^T] and Var[q k^T], and what scaling by sqrt(h) does to softmax

Usage: python verification/verify_section2.py
"""
import numpy as np
import torch

torch.manual_seed(0)
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


# ---------------------------------------------------------------- 2.1
section("2.1  Argmax attention")
K = np.array([[0, 1, 2], [1, 1, 2], [-1, 1, -2], [0, -1, -2]], dtype=float)  # rows are keys
V = np.array([[3, 2, -4], [3, -2, 4], [3, 2, 0], [3, 2, 4]], dtype=float)    # rows are values
q = np.array([-2, 1, 1], dtype=float)
scores = K @ q
print("scores q.k_i      =", scores)
print("scaled by sqrt(3) =", scores / np.sqrt(3))
weights = np.zeros(4)
weights[np.argmax(scores)] = 1
print("argmax weights    =", weights)
print("output            =", weights @ V)
soft = np.exp(scores / np.sqrt(3)) / np.exp(scores / np.sqrt(3)).sum()
print("(scaled softmax weights for comparison =", soft, ", output", soft @ V, ")")

# Gradient check: argmax is built from a one-hot that autograd cannot differentiate through.
qt = torch.tensor(q, requires_grad=True)
Kt = torch.tensor(K, requires_grad=True)
Vt = torch.tensor(V, requires_grad=True)
s = Kt @ qt
w_hard = torch.nn.functional.one_hot(s.argmax(), 4).double()  # no grad_fn: a constant
out = w_hard @ Vt
out.sum().backward()
print("\nargmax attention, d(sum of output)/d q :", qt.grad)
print("argmax attention, d(sum of output)/d K :", Kt.grad)
print("argmax attention, d(sum of output)/d V :\n", Vt.grad.numpy())

qt2 = torch.tensor(q, requires_grad=True)
Kt2 = torch.tensor(K, requires_grad=True)
Vt2 = torch.tensor(V, requires_grad=True)
out2 = torch.softmax(Kt2 @ qt2 / np.sqrt(3), dim=0) @ Vt2
out2.sum().backward()
print("\nsoftmax attention, d(sum of output)/d q :", qt2.grad.numpy())
print("softmax attention, d(sum of output)/d K :\n", Kt2.grad.numpy())

# ---------------------------------------------------------------- 2.2.1
section("2.2.1  Copying: q = c * k_j with orthonormal keys")
h, n = 16, 6
keys = np.linalg.qr(rng.normal(size=(h, n)))[0].T  # n orthonormal keys in R^h
vals = rng.normal(size=(n, h))
j = 2
for c in [1, 4, 8, 16, 32]:
    a = np.exp(keys @ (c * keys[j]) / np.sqrt(h))
    a /= a.sum()
    others = a.sum() - a[j]
    err = np.linalg.norm(a @ vals - vals[j])
    bound = (n - 1) * np.exp(-c / np.sqrt(h))
    print(f"c={c:3d}: alpha_j={a[j]:.6f}  sum of others={others:.6f}  "
          f"bound (n-1)e^(-c/sqrt h)={bound:.6f}  ||out - v_j||={err:.6f}")

# ---------------------------------------------------------------- 2.2.2
section("2.2.2  M = 2 * sum_i a_i a_i^T")
h, m, p = 20, 4, 5
Q = np.linalg.qr(rng.normal(size=(h, m + p)))[0]  # m + p orthonormal columns
A, B = Q[:, :m], Q[:, m:]  # basis of subspace A, basis of subspace B
M = 2 * A @ A.T
print("max |a_j^T b_k|            =", np.abs(A.T @ B).max())
print("max |A^T A - I|            =", np.abs(A.T @ A - np.eye(m)).max())
worst = 0
for _ in range(1000):
    va = A @ rng.normal(size=m)
    vb = B @ rng.normal(size=p)
    vbar = 0.5 * (va + vb)
    worst = max(worst, np.abs(M @ vbar - va).max())
print("max |M vbar - v_a| over 1,000 random (v_a, v_b) pairs =", worst)
print("for contrast, A A^T vbar = v_a / 2:", np.allclose(A @ A.T @ (0.5 * (va + vb)), 0.5 * va))

# ---------------------------------------------------------------- 2.2.3
section("2.2.3  q = c (k_a + k_b) with orthonormal keys")
h, n = 16, 6
keys = np.linalg.qr(rng.normal(size=(h, n)))[0].T
vals = rng.normal(size=(n, h))
ia, ib = 1, 4
target = 0.5 * (vals[ia] + vals[ib])
for c in [1, 4, 8, 16, 32, 64]:
    qv = c * (keys[ia] + keys[ib])
    a = np.exp(keys @ qv / np.sqrt(h))
    a /= a.sum()
    rest = 1 - a[ia] - a[ib]
    formula = (n - 2) / (2 * np.exp(c / np.sqrt(h)) + n - 2)
    print(f"c={c:3d}: alpha_a={a[ia]:.6f} alpha_b={a[ib]:.6f}  others={rest:.6f} "
          f"(formula {formula:.6f})  ||out - (v_a+v_b)/2||={np.linalg.norm(a @ vals - target):.6f}")

# ---------------------------------------------------------------- 2.3
section("2.3  E and Var of q k^T for q, k ~ N(mu, sigma^2 I), independent")
N = 400_000
for h, mu, sigma in [(64, 0.5, 2.0), (16, -1.0, 0.5), (64, 0.0, 1.0)]:
    qs = rng.normal(mu, sigma, size=(N, h))
    ks = rng.normal(mu, sigma, size=(N, h))
    d = (qs * ks).sum(axis=1)
    print(f"h={h:3d} mu={mu:+.1f} sigma={sigma:.1f}: sample mean {d.mean():8.3f} vs h mu^2 = {h * mu**2:8.3f} | "
          f"sample var {d.var():9.2f} vs h sigma^4 + 2 h sigma^2 mu^2 = {h * sigma**4 + 2 * h * sigma**2 * mu**2:9.2f}")

print("\nmu = 0, sigma = 1: spread of raw and scaled dot products, and the softmax over 32 keys")
n_keys, trials = 32, 2000
for h in [4, 16, 64, 256, 1024]:
    qs = rng.normal(size=(trials, 1, h))
    ks = rng.normal(size=(trials, n_keys, h))
    raw = (qs * ks).sum(-1)
    for name, sc in (("raw   ", raw), ("scaled", raw / np.sqrt(h))):
        z = sc - sc.max(axis=1, keepdims=True)
        a = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        # softmax Jacobian diag(a) - a a^T; its Frobenius norm shrinks to 0 as a goes one-hot
        jac = np.array([np.linalg.norm(np.diag(r) - np.outer(r, r)) for r in a[:200]])
        print(f"h={h:5d} {name}: std {sc.std():7.2f}  mean max weight {a.max(axis=1).mean():.3f}  "
              f"mean ||softmax Jacobian||_F {jac.mean():.4f}")

print("\nfloat16 overflow: exp(x) is inf in float16 once x > ~11.09. Largest raw score in 32 keys,"
      " averaged over trials:")
for h in [64, 256, 1024]:
    qs = rng.normal(size=(trials, 1, h))
    ks = rng.normal(size=(trials, n_keys, h))
    raw = (qs * ks).sum(-1)
    big = raw.max(axis=1)
    with np.errstate(over="ignore"):
        overflow = np.isinf(np.exp(big.astype(np.float16))).mean()
    print(f"h={h:5d}: mean max raw score {big.mean():6.1f}, share of trials where exp(max) overflows fp16: {overflow:.2f}")
