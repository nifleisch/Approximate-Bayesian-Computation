import numpy as np
from stochastic_simulation.utils import plot_samples_with_posterior, kl_divergence, plot_distance, trace_plot, autocorrelation_plot, plot_estimates, ess_plot, plot_kl_divergence
from stochastic_simulation.distribution import (
    NormalPrior,
    MixedGaussianPosterior,
    TwoPopulationsModel,
    NormalProposal,
)
from stochastic_simulation.approx_bayesian_computation import abc_rejection, abc_mcmc
from stochastic_simulation.discreptency_metric import mean_abs_difference
from argparse import ArgumentParser
from scipy.stats import gaussian_kde
import pandas as pd
from functools import partial
from pathlib import Path


parser = ArgumentParser(
    description="Run ABC rejection or ABC-MCMC on the academic example."
)
parser.add_argument(
    "--method",
    type=str,
    default="abc_rejection",
    help="The method to be used. Either abc_rejection or abc_mcmc.",
)
parser.add_argument(
    "--x_bar",
    type=float,
    default=0.0,
    help="The sample mean of the data.",
)
parser.add_argument(
    "--epsilons",
    nargs="+",
    type=float,
    default=[0.7, 0.25, 0.1, 0.025],
    help="The tolerances to be used.",
)
parser.add_argument(
    "--nu",
    type=float,
    default=0.1,
    help="The nu value to be used.",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=3,
    help="The standard deviation of the normal prior.",
)
parser.add_argument(
    "--sigma_1",
    type=float,
    default=0.1,
    help="The standard deviation of the gaussians of the likelihood.",
)
parser.add_argument(
    "--M",
    type=int,
    default=100,
    help="The number of observations to be generated.",
)
parser.add_argument(
    "--N",
    type=int,
    default=500,
    help="The number of samples to be generated.",
)
parser.add_argument(
    "--theta_0",
    type=float,
    default=0.0,
    help="The inital state for the ABC-MCMC algorithm.",
)
parser.add_argument(
    "--plot_distance",
    type=bool,
    default=False,
    help="Whether to plot the distance function.",
)


if __name__ == "__main__":
    # -----------------------------------------
    # Parse arguments
    args = parser.parse_args()

    # -----------------------------------------
    # Set the seed
    np.random.seed(0)

    # -----------------------------------------
    # Set parameters
    method = args.method
    M = args.M
    sigma = np.sqrt(args.sigma)
    sigma_1 = np.sqrt(args.sigma_1)
    a = 1
    x_bar = args.x_bar
    epsilons = args.epsilons
    N = args.N
    rho = partial(mean_abs_difference, x_bar=x_bar)

    # MCMC specific parameters
    nu = args.nu
    theta_0 = args.theta_0

    # -----------------------------------------
    # Define the distributions
    prior = NormalPrior(mu=0, sigma=sigma)
    model = TwoPopulationsModel(a=a, sigma_1=sigma_1, M=M)
    posterior = MixedGaussianPosterior(M=M, x_bar=x_bar, a=a, sigma=sigma, sigma_1=sigma_1)

    plot_folder = Path("plots")
    plot_folder.mkdir(exist_ok=True, parents=True)
    method_folder = plot_folder / method
    method_folder.mkdir(exist_ok=True, parents=True)
    nu_postfix = "_nu_" + str(nu).replace('.', '_') if method == "abc_mcmc" else ""

    # -----------------------------------------
    # Visualize the distance function
    if args.plot_distance:
        distance_plot_path = plot_folder / "distance.png"
        plot_distance(prior, model, rho, epsilon=0.7, filepath=distance_plot_path)

    # -----------------------------------------
    # Run the algorithms
    estimate_dict = {}
    kl_dict = {}
    result_list = []  
    for epsilon in epsilons:
        discreptancy_folder = method_folder / str(epsilon).replace('.', '_')
        discreptancy_folder.mkdir(exist_ok=True, parents=True)
        # -----------------------------------------
        # 1) Simple ABC Rejection Algorithm
        if method == "abc_rejection":
            samples, result = abc_rejection(N, model, prior, rho, epsilon)

        # -----------------------------------------
        # 3) ABC-MCMC
        if method == "abc_mcmc":
            proposal = NormalProposal(sigma=nu)
            samples, result = abc_mcmc(
                N=N,
                model=model,
                prior=prior,
                proposal=proposal,
                rho=rho,
                epsilon=epsilon,
                theta_0=theta_0,
                use_eff_N=True,
            )
            trace_plot(samples=samples, filepath=discreptancy_folder / f"trace_plot{nu_postfix}.png")
            autocorrelation_plot(samples=samples, filepath=discreptancy_folder / f"autocorrelation_plot{nu_postfix}.png")
            ess_plot(samples=samples, filepath=discreptancy_folder / f"ess_plot{nu_postfix}.png")

        # -----------------------------------------
        # Plot results
        plot_samples_with_posterior(
                thetas=samples,
                posterior=posterior,
                filepath=discreptancy_folder / f"samples{nu_postfix}.png",
            )
        
        # -----------------------------------------
        # Compute the KL divergence between the true posterior and the approximate posterior
        kde = gaussian_kde(samples, bw_method = 0.05)
        kl = kl_divergence(kde=kde, p=posterior.pdf, lower_bound=-1.5, upper_bound=0.5)
        print(f"KL divergence: {kl:.5f}")

        estimate_dict[epsilon] = kde
        kl_dict[epsilon] = kl

        # -----------------------------------------
        # Collect run statistics in a dictionary
        result["method"] = method
        result["epsilon"] = epsilon
        result["M"] = M
        result["N"] = N
        result["x_bar"] = x_bar
        if method == "abc_mcmc":
            result["nu"] = nu
            result["theta_0"] = theta_0
        else:
            result["nu"] = None
            result["theta_0"] = None
        result["kl_divergence"] = kl

        cols = [
            "method",
            "epsilon",
            "M",
            "N",
            "x_bar",
            "nu",
            "theta_0",
            "acceptance_rate",
            "acceptance_rate_ci",
            "kl_divergence",
        ]
        result = {col: result[col] for col in cols}
        result_list.append(result)
    
    # -----------------------------------------
    plot_estimates(estimate_dict=estimate_dict, prior=prior.pdf, posterior=posterior, filepath=method_folder / f"estimates{nu_postfix}.png")
    plot_kl_divergence(kl_dict=kl_dict, filepath=method_folder / f"kl_divergence{nu_postfix}.png")

    run_df = pd.DataFrame.from_records(result_list)
    try:
        df = pd.read_csv("academic_example_results.csv", index_col=False)
    except FileNotFoundError:
        df = pd.DataFrame(columns=cols)
    df = pd.concat([df, run_df]).reset_index(drop=True)
    df.to_csv("academic_example_results.csv", index=False)
