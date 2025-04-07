import torch
from sklearn.mixture import GaussianMixture


class TorchGMM:
    def __init__(self, gmm: GaussianMixture):
        self.gmm = gmm
        # Convert parameters to torch tensors
        self.weights = torch.tensor(gmm.weights_, dtype=torch.float32)
        self.means = torch.tensor(gmm.means_, dtype=torch.float32)
        self.covariances = torch.tensor(gmm.covariances_, dtype=torch.float32)
        self.n_components = gmm.n_components
        self.covariance_type = gmm.covariance_type

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-likelihood of each sample in x."""
        x_np = x.detach().cpu().numpy()
        log_probs = self.gmm.score_samples(x_np)
        return torch.tensor(log_probs, dtype=torch.float32, device=x.device)

    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples from the GMM."""
        samples, _ = self.gmm.sample(n_samples)
        return torch.tensor(samples, dtype=torch.float32)


def estimate_gmm_from_states_torch(
    states: torch.Tensor,
    n_components: int = 5,
    covariance_type: str = "diag",
    random_state: int = 42,
) -> TorchGMM:
    """
    Fit a GMM to the given states (PyTorch tensor) and return a Torch-compatible wrapper.

    Args:
        states (torch.Tensor): Tensor of shape [num_samples, state_dim].
        n_components (int): Number of Gaussian components.
        covariance_type (str): Covariance type: 'full', 'tied', 'diag', or 'spherical'.
        random_state (int): Random seed.

    Returns:
        TorchGMM: A PyTorch-compatible wrapper for the fitted GMM.
    """
    states_np = states.detach().cpu().numpy()
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=200,
        random_state=random_state,
    )
    gmm.fit(states_np)
    return TorchGMM(gmm)


def kl_divergence_gmm(gmm_p, gmm_q, n_samples=10000):
    """
    Estimate KL divergence KL(P || Q) between two GMMs using Monte Carlo sampling.

    Args:
        gmm_p (TorchGMM): First GMM (from which we sample).
        gmm_q (TorchGMM): Second GMM.
        n_samples (int): Number of samples to use for approximation.

    Returns:
        float: Estimated KL divergence.
    """
    # Sample from P
    samples = gmm_p.sample(n_samples)  # shape: [n_samples, dim]

    # Evaluate log probabilities
    log_p = gmm_p.log_prob(samples)
    log_q = gmm_q.log_prob(samples)

    # KL(P || Q) = E_P[log P - log Q]
    kl_estimate = torch.mean(log_p - log_q)
    return kl_estimate.item()
