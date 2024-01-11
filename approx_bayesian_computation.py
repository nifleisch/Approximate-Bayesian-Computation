import numpy as np
from tqdm import tqdm
from arviz import ess


def abc_rejection(N, model, prior, rho, epsilon) -> np.ndarray:
    """
    Performs simple ABC rejection sampling. Print acceptance rate.

    Parameters:
    N (int): The number of samples to be generated.
    model (class): The model distribution.
    prior (class): The prior distribution.
    rho (function): The distance function between the simulated and the observed data.
    epsilon (float): The tolerance.

    Returns:
    array: The accepted samples.
    dict: Statistics about the run. 
    """
    sample = []
    n_attempts = 0

    pbar = tqdm(total=N)
    while len(sample) < N:
        n_attempts += 1
        theta = prior.rvs()
        D_star = model.rvs(theta=theta)
        if rho(D_star) < epsilon:
            pbar.update(1)
            sample.append(theta)
    pbar.close()
    p = N / n_attempts
    print("Basic ABC rejection algorithm")
    print(f"epsilon = {epsilon}, N = {N}")
    print(f"Acceptance rate: {p:.5f} +- {np.sqrt(p * (1 - p) / n_attempts) * 1.96:.5f} (95% confidence interval)")
    print(
        "-------------------------------------------------------------------------------------------------"
    )
    stats = {"acceptance_rate": p, 
             "acceptance_rate_ci": [p - 1.96 * np.sqrt(p * (1 - p) / len(sample)), p + 1.96 * np.sqrt(p * (1 - p) / len(sample))],
            }
    return np.array(sample), stats


def abc_mcmc_step(theta, model, prior, proposal, rho, epsilon) -> np.ndarray:
    theta_star = proposal.rvs(theta)
    D_star = model.rvs(theta=theta_star)
    accepted = False
    if rho(D_star) < epsilon:
        accepted = True
        proposal_ratio = proposal.pdf(theta_star, theta) / proposal.pdf(
            theta, theta_star
        )
        prior_ratio = prior.pdf(theta_star) / prior.pdf(theta)
        alpha = min(1, prior_ratio * proposal_ratio)
        if np.random.uniform() < alpha:
            theta = theta_star
    return theta, accepted
    

def abc_mcmc(N, model, prior, proposal, rho, epsilon, theta_0, use_eff_N=False, adaptive_proposal=False) -> np.ndarray:
    """
    Performs ABC-MCMC. Print acceptance rate.

    Parameters:
    N (int): Whether to use N_eff or not.
    model (class): The model distribution.
    prior (class): The prior distribution.
    proposal (class): The proposal distribution.
    rho (function): The distance function between the simulated and the observed data.
    epsilon (float): The tolerance.
    theta_0 (float): The initial value of theta.
    use_eff_N (int): The effective sample size.
    adaptive_proposal (bool): Whether to use adaptive proposal or not.

    Returns:
    array: The accepted samples.
    dict: Statistics about the run. 
    """
    sample = []
    n_accepted = 0

    pbar = tqdm(total=N)
    theta = theta_0
    sample_size = 0
    while sample_size < N:
        theta, accepted = abc_mcmc_step(theta, model, prior, proposal, rho, epsilon)
        sample.append(theta)
        n_accepted += int(accepted)

        if use_eff_N:
            sample_size = int(round(ess(np.array(sample)))) if len(sample) > 4 else len(sample)
        else:
            sample_size = len(sample)

        if adaptive_proposal and len(sample) % 50 == 0:
            proposal.update(theta)
        pbar.n = sample_size
        pbar.refresh()
    pbar.close()
    p = n_accepted / len(sample)
    print("ABC-MCMC algorithm")
    print(f"epsilon = {epsilon}, N = {N}")
    print(
        f"Acceptance rate: {p} +- {np.sqrt(p * (1 - p) / len(sample)) * 1.96} (95% confidence interval)"
    )
    print(
        "-------------------------------------------------------------------------------------------------"
    )
    stats = {"acceptance_rate": p, 
             "acceptance_rate_ci": [p - 1.96 * np.sqrt(p * (1 - p) / len(sample)), p + 1.96 * np.sqrt(p * (1 - p) / len(sample))],
             "N": len(sample)}
    return np.array(sample), stats
