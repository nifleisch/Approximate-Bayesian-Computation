import numpy as np


def monte_carlo(sampler, N) -> float:
    """
    Estimate the mean of a distribution using Monte Carlo method.

    Parameters:
    sampler (function): The sampler.
    N (int): The number of samples to be generated.

    Returns:
    float: The mean of the samples.
    """
    samples = [sampler() for i in range(N)]
    return np.mean(samples)