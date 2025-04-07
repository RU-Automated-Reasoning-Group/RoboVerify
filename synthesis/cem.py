import numpy as np


def cem_optimize(f, dim, iterations=50, N=100, K=10, init_mu=None, init_std=1.0):
    mu = init_mu if init_mu is not None else np.zeros(dim)
    sigma = np.ones(dim) * init_std

    for _ in range(iterations):
        samples = np.random.randn(N, dim) * sigma + mu
        scores = np.array([f(x) for x in samples])
        elites = samples[scores.argsort()[-K:]]  # Top-K

        mu = elites.mean(axis=0)
        sigma = elites.std(axis=0)

    return mu
