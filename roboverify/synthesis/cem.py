import numpy as np


def cem_optimize(
    f, dim, iterations=50, N=100, K=10, init_mu=None, init_std=0.1
) -> tuple[float, np.ndarray]:
    mu = init_mu if init_mu is not None else np.zeros(dim)
    sigma = np.ones(dim) * init_std

    mu_list = []
    score_list = []
    # evaluate current mu
    mu_list.append(mu)
    score_list.append(f(mu))

    for _ in range(iterations):


        samples = np.random.randn(N, dim) * sigma + mu
        scores = np.array([f(x) for x in samples])
        print("scores", sorted(scores))
        print("average of best k", np.mean(sorted(scores)[-K:]))
        print("best mu", samples[scores.argsort()[-1]])
        elites = samples[scores.argsort()[-K:]]  # Top-K

        mu = elites.mean(axis=0)
        print("next mu:", mu)
        sigma = elites.std(axis=0)

        # evaluate current mu
        mu_list.append(mu)
        score_list.append(f(mu))

    max_idx = np.argmax(score_list)
    return score_list[max_idx], mu_list[max_idx]
