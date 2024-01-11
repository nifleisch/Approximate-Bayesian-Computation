from stochastic_simulation.distribution import (
    TeophyllineConcentration, 
    TeophyllinePrior,
    MultivariateNormalProposal
)
from stochastic_simulation.utils import plot_concentration_curves, plot_distance, trace_plot, autocorrelation_plot, plot_estimates, ess_plot, plot_pair_distributions
from stochastic_simulation.approx_bayesian_computation import abc_mcmc
from stochastic_simulation.discreptency_metric import PharmacokineticDiscreptancy
import numpy as np
from pathlib import Path
from arviz import ess
from scipy.stats import gaussian_kde
from functools import partial
import pandas as pd


if __name__ == "__main__":
    # -----------------------------------------
    # Set the seed
    np.random.seed(0)

    # -----------------------------------------
    # Define model
    drug_dose = 4
    X_0 = 0
    theta = np.array([0.08, 1.5, 0.04, 0.2])
    theta_0 = np.array([0.07, 1.15, 0.05, 0.33])
    sampling_times = np.array([0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12])
    model = TeophyllineConcentration(
        drug_dose=drug_dose, X_0=X_0, sampling_times=sampling_times, N=20
    )

    # -----------------------------------------
    # Define prior distribution ([mean, std])
    K_e_prior = [-2.7, 0.6]
    K_a_prior = [0.14, 0.4]
    Cl_prior = [-3, 0.8]
    sigma_prior = [-1.1, 0.3]
    prior = TeophyllinePrior(K_e=K_e_prior, K_a=K_a_prior, Cl=Cl_prior, sigma=sigma_prior)
    
    N = 10000
    tolerances = [0.25, 0.7, 1]


    # -----------------------------------------
    # 4) generat synthetic data with parameters theta
    D = model.rvs(theta=theta)

    # -----------------------------------------
    # Define a more fine grained model to plot a typical concentration curve
    sample_list = []
    for _ in range(1000):
        sample_list.append(model.rvs(theta=theta))

    plot_folder = Path("plots") / "pharmacokinetic_model"
    plot_folder.mkdir(exist_ok=True, parents=True)
    plot_concentration_curves(D, sampling_times, sample_list, filepath=plot_folder / "concentration_curves.png")
    
    # Prepare discreptency metric by generating data and fitting the regression model
    p = 1000
    rho = PharmacokineticDiscreptancy(D=D, model=model, prior=prior, p=p, theta_0=theta_0)
    # try to predict theta based on D using the fitted linear regression model
    theta_pred = rho.S(D)[0]

    # plot distance function
    plot_distance(prior, model, rho, epsilon=0.7, filepath=plot_folder / "distance.png", n_samples=10000)

    # -----------------------------------------
    # 5) ABC-MCMC
    variances = [K_e_prior[1]**2, K_a_prior[1]**2, Cl_prior[1]**2, sigma_prior[1]**2]
    result_list = []
    for adaptive_proposal in [True, False]:
        adaptive_proposal_postfix = "_adaptive" if adaptive_proposal else ""
        K_e_estimate_dict = {}
        K_a_estimate_dict = {}
        Cl_estimate_dict = {}
        sigma_estimate_dict = {}
        for epsilon in tolerances:
            proposal = MultivariateNormalProposal(variances)
            samples, result = abc_mcmc(
                N=N,
                model=model,
                prior=prior,
                proposal=proposal,
                rho=rho,
                epsilon=epsilon,
                theta_0=theta,
                adaptive_proposal=adaptive_proposal
            )
            result["epsilon"] = epsilon
            result["N"] = N
            result["adaptive_proposal"] = adaptive_proposal

            K_e, K_a, Cl, sigma = [np.array(param) for param in zip(*samples)]
            result['K_e_ess'] = ess(K_e)
            result['K_a_ess'] = ess(K_a)
            result['Cl_ess'] = ess(Cl)
            result['sigma_ess'] = ess(sigma)

            result['K_e_mean'] = np.exp(np.mean(np.log(K_e)))
            result['K_a_mean'] = np.exp(np.mean(np.log(K_a)))
            result['Cl_mean'] = np.exp(np.mean(np.log(Cl)))
            result['sigma_mean'] = np.exp(np.mean(np.log(sigma)))

            result['K_e_ci_lower'] = np.exp(np.mean(np.log(K_e)) - 1.96 * np.std(np.log(K_e)))
            result['K_e_ci_upper'] = np.exp(np.mean(np.log(K_e)) + 1.96 * np.std(np.log(K_e)))
            result['K_a_ci_lower'] = np.exp(np.mean(np.log(K_a)) - 1.96 * np.std(np.log(K_a)))
            result['K_a_ci_upper'] = np.exp(np.mean(np.log(K_a)) + 1.96 * np.std(np.log(K_a)))
            result['Cl_ci_lower'] = np.exp(np.mean(np.log(Cl)) - 1.96 * np.std(np.log(Cl)))
            result['Cl_ci_upper'] = np.exp(np.mean(np.log(Cl)) + 1.96 * np.std(np.log(Cl)))
            result['sigma_ci_lower'] = np.exp(np.mean(np.log(sigma)) - 1.96 * np.std(np.log(sigma)))
            result['sigma_ci_upper'] = np.exp(np.mean(np.log(sigma)) + 1.96 * np.std(np.log(sigma)))
            result_list.append(result)

            sample_df = pd.DataFrame({"K_e": K_e, "K_a": K_a, "Cl": Cl, "sigma": sigma})
            epsilon_folder = plot_folder / str(epsilon).replace('.', '_')
            epsilon_folder.mkdir(exist_ok=True, parents=True)
            plot_pair_distributions(sample_df, theta, theta_pred, folderpath=epsilon_folder, postfix=adaptive_proposal_postfix)

            def log_kde_wrapper(x, kde):
                return kde(np.log(x))
            K_e_estimate_dict[epsilon] = partial(log_kde_wrapper, kde=gaussian_kde(np.log(K_e)))
            K_a_estimate_dict[epsilon] = partial(log_kde_wrapper, kde=gaussian_kde(np.log(K_a)))
            Cl_estimate_dict[epsilon] = partial(log_kde_wrapper, kde=gaussian_kde(np.log(Cl)))
            sigma_estimate_dict[epsilon] = partial(log_kde_wrapper, kde=gaussian_kde(np.log(sigma)))

            K_e_folder = plot_folder / str(epsilon).replace('.', '_') / "K_e"
            K_e_folder.mkdir(exist_ok=True, parents=True)
            trace_plot(samples=K_e, filepath=K_e_folder / f"trace_plot{adaptive_proposal_postfix}.png", ylabel=r"$K_e$")
            autocorrelation_plot(samples=K_e, filepath=K_e_folder / f"autocorrelation_plot{adaptive_proposal_postfix}.png")
            ess_plot(samples=K_e, filepath=K_e_folder / f"ess_plot{adaptive_proposal_postfix}.png")

            K_a_folder = plot_folder / str(epsilon).replace('.', '_') / "K_a"
            K_a_folder.mkdir(exist_ok=True, parents=True)
            trace_plot(samples=K_a, filepath=K_a_folder / f"trace_plot{adaptive_proposal_postfix}.png", ylabel=r"$K_a$")
            autocorrelation_plot(samples=K_a, filepath=K_a_folder / f"autocorrelation_plot{adaptive_proposal_postfix}.png")
            ess_plot(samples=K_a, filepath=K_a_folder / f"ess_plot{adaptive_proposal_postfix}.png")

            Cl_folder = plot_folder / str(epsilon).replace('.', '_') / "Cl"
            Cl_folder.mkdir(exist_ok=True, parents=True)
            trace_plot(samples=Cl, filepath=Cl_folder / f"trace_plot{adaptive_proposal_postfix}.png", ylabel=r"$Cl$")
            autocorrelation_plot(samples=Cl, filepath=Cl_folder / f"autocorrelation_plot{adaptive_proposal_postfix}.png")
            ess_plot(samples=Cl, filepath=Cl_folder / f"ess_plot{adaptive_proposal_postfix}.png")

            sigma_folder = plot_folder / str(epsilon).replace('.', '_') / "sigma"
            sigma_folder.mkdir(exist_ok=True, parents=True)
            trace_plot(samples=sigma, filepath=sigma_folder / f"trace_plot{adaptive_proposal_postfix}.png", ylabel=r"$\sigma$")
            autocorrelation_plot(samples=sigma, filepath=sigma_folder / f"autocorrelation_plot{adaptive_proposal_postfix}.png")
            ess_plot(samples=sigma, filepath=sigma_folder / f"ess_plot{adaptive_proposal_postfix}.png")
        
        K_e_interval = [np.exp(K_e_prior[0] - K_e_prior[1] * 2.807), np.exp(K_e_prior[0] + K_e_prior[1] * 2.807)]
        K_e_prior_func = partial(prior.marginal_pdf, parameter="K_e")
        plot_estimates(K_e_estimate_dict, K_e_prior_func, theta=theta[0], theta_pred=theta_pred[0],
                        filepath=plot_folder / f"K_e_estimates{adaptive_proposal_postfix}.png", xlabel=r"$K_e$", xlim=K_e_interval)

        K_a_interval = [np.exp(K_a_prior[0] - K_a_prior[1] * 2.807), np.exp(K_a_prior[0] + K_a_prior[1] * 2.807)]
        K_a_prior_func = partial(prior.marginal_pdf, parameter="K_a")
        plot_estimates(K_a_estimate_dict, K_a_prior_func, theta=theta[1], theta_pred=theta_pred[1],
                        filepath=plot_folder / f"K_a_estimates{adaptive_proposal_postfix}.png", xlabel=r"$K_a$", xlim=K_a_interval)

        Cl_interval = [np.exp(Cl_prior[0] - Cl_prior[1] * 2.807), np.exp(Cl_prior[0] + Cl_prior[1] * 2.807)]
        Cl_prior_func = partial(prior.marginal_pdf, parameter="Cl")
        plot_estimates(Cl_estimate_dict, Cl_prior_func, theta=theta[2], theta_pred=theta_pred[2],
                        filepath=plot_folder / f"Cl_estimates{adaptive_proposal_postfix}.png", xlabel=r"$Cl$", xlim=Cl_interval)

        sigma_interval = [np.exp(sigma_prior[0] - sigma_prior[1] * 2.807), np.exp(sigma_prior[0] + sigma_prior[1] * 2.807)]
        sigma_prior_func = partial(prior.marginal_pdf, parameter="sigma")
        plot_estimates(sigma_estimate_dict, sigma_prior_func, theta=theta[3], theta_pred=theta_pred[3],
                        filepath=plot_folder / f"sigma_estimates{adaptive_proposal_postfix}.png", xlabel=r"$\sigma$", xlim=sigma_interval)

    df = pd.DataFrame(result_list)
    df.to_csv("pharmacokinetic_model_results", index=False)
