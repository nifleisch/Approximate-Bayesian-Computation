from statsmodels.tsa.stattools import acf as autocorr
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import seaborn as sns
import pandas as pd
import warnings
from arviz import ess
from itertools import combinations


def kl_divergence(kde, p, lower_bound, upper_bound):
    """
    Computes the KL divergence between the true posterior and the posterior.

    Parameters:
    kde (function): The approximate posterior density.
    p (function): The true posterior density.
    lower_bound (float): The lower bound of the domain of the posterior.
    upper_bound (float): The upper bound of the domain of the posterior.
    """
    def integrand(x):
        px = p(x)
        qx = kde(x)[0]
        return qx * np.log(qx / px)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = quad(integrand, lower_bound, upper_bound)[0]
    return result


def plot_samples_with_posterior(thetas, posterior, filepath) -> None:
    """
    Plots the histogram of the samples and the posterior density.

    Parameters:
    thetas (array): The samples from the approximate posterior.
    posterior (function): The true posterior density.
    title (string): The title of the plot.
    filepath (path): The filename of the plot.
    """
    theta_grid = np.linspace(-1.5, 0.5, 1000)
    posterior_values = posterior.pdf(theta_grid)

    fig, ax = plt.subplots()
    ax.hist(thetas, bins=20, density=True, color="grey", label="Samples")
    ax.plot(theta_grid, posterior_values, color="red", label="Posterior")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(filepath)
    fig.clear()


def plot_distance(prior, model, rho, epsilon, filepath, n_samples=100000) -> None:
    """
    Plots the distribution of the distance.

    Parameters:
    prior (class): The prior distribution.
    model (class): The model distribution.
    rho (function): The distance function.
    epsilon (float): The tolerance.
    filepath (path): The filename of the plot.
    n_samples (int): The number of samples to be generated.
    """
    distance_sample =[]
    for _ in range(n_samples):
        theta = prior.rvs()
        D_star = model.rvs(theta=theta)
        distance_sample.append(rho(D_star))
    kde = gaussian_kde(distance_sample, bw_method=0.05)
    x = np.linspace(0, 8, 3000)
    y = kde(x)
    fig, ax = plt.subplots()
    ax.plot(x, y, color="black")
    ax.fill_between(x, y, where=x < epsilon, color="navy")
    ax.fill_between(x, y, where=x > epsilon, color="0.9")
    kde_epsilon = kde(epsilon)[0]
    ax.plot([epsilon, epsilon], [0, 1.2 * max(y)], color="black", linestyle="--")
    ax.annotate(f"$\epsilon={epsilon:.1f}$", xy=(epsilon, kde_epsilon), xytext=(epsilon+ 0.05, kde_epsilon + 0.05))
    ax.set_ylim(0, 1.2 * max(y))
    ax.set_xlabel(r"$\rho(D^*, D)$")
    ax.set_ylabel("Density")

    fig.savefig(filepath)
    fig.clear()


def trace_plot(samples, filepath, ylabel=r"$\theta$") -> None:
    """
    Plots the trace plot of the samples.

    Parameters:
    samples (array): The samples.
    filepath (path): The filename of the plot.
    """
    fig, ax = plt.subplots()
    ax.plot(samples)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    fig.savefig(filepath)
    fig.clear()


def autocorrelation_plot(samples, filepath) -> None:
    """
    Plots the autocorrelation plot of the samples.

    Parameters:
    samples (array): The samples.
    filepath (path): The filename of the plot.
    """
    fig, ax = plt.subplots()
    ax.stem(autocorr(samples, nlags=100))
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    fig.savefig(filepath)
    fig.clear()


def plot_estimates(estimate_dict, prior, filepath, posterior=None, theta=None, theta_pred=None, xlabel=r"$\theta$", xlim=[-2, 1]) -> None:
    """
    Plots the estimates of the posterior density.

    Parameters:
    estimate_dict (dict): The estimates for different values of epsilon.
    prior (class): The prior distribution.
    filepath (path): The filename of the plot.
    posterior (function): The true posterior density.
    theta (float): The true value of theta.
    theta_pred (float): The predicted value of theta.
    xlabel (string): The label of the x-axis.
    """
    theta_grid = np.linspace(xlim[0], xlim[1], 1000)
    prior_values = prior(theta_grid)
    linestyle = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(10, 5))
    if posterior is not None:
        posterior_values = posterior.pdf(theta_grid)
        ax.plot(theta_grid, posterior_values, color="red", label="Posterior")
    if theta is not None:
        ax.plot(theta, 0, color="black", marker="x", markersize=5, label="True value")
    if theta_pred is not None:
        ax.plot(theta_pred, 0, color="blue", marker="x", markersize=5, label="Predicted value")
    ax.plot(theta_grid, prior_values, color="grey", label="Prior")
    for i, (eps, kde) in enumerate(estimate_dict.items()):
        ax.plot(theta_grid, kde(theta_grid), label=f"$\epsilon={eps}$", linestyle=linestyle[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    fig.savefig(filepath)
    fig.clear()


def ess_plot(samples, filepath):
    """
    Plots the effective sample size of the samples.

    Parameters:
    samples (array): The samples.
    filepath (path): The filename of the plot.
    """
    ess_list = []
    for i in range(1, len(samples)):
        ess_list.append(ess(samples[:i]))
    fig, ax = plt.subplots()
    ax.plot(ess_list)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Effective sample size")
    fig.savefig(filepath)
    fig.clear()


def plot_kl_divergence(kl_dict, filepath):
    """
    Plots the KL divergence for different values of epsilon.

    Parameters:
    kl_dict (dict): KL divergence for different values of epsilon.
    filepath (path): The filename of the plot.
    """
    fig, ax = plt.subplots()
    ax.plot(list(kl_dict.keys()), list(kl_dict.values()))
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("KL divergence")
    fig.savefig(filepath)
    fig.clear()


def plot_concentration_curves(D, sampling_times, sample_list, filepath):
    """
    Plots a typical concentration curve.

    Parameters:
    sample_list (list): list of multiple concentration samples.
    filepath (path): The filename of the plot.
    """
    sampling_times = [0] + sampling_times.tolist()
    sample_list = [[0] + sample.tolist() for sample in sample_list]
    D = [0] + D.tolist()
    fig, ax = plt.subplots()
    for sample in sample_list:
        ax.plot(sampling_times, sample, color=".8", alpha=0.005, linewidth=1)
    ax.plot(sampling_times, D, color="navy", label="D", marker="o", linestyle="-", markersize=2)
    ax.set_xlabel("$t$ [in h]")
    ax.set_ylabel("Theophylline concentration $X_t$")
    fig.savefig(filepath)
    fig.clear()


def plot_pair_distributions(sample_df, theta, theta_hat, folderpath, postfix=""):
    param_names = ["K_e", "K_a", "Cl", "sigma"]
    param_names_latex = [r"$K_e$", r"$K_a$", r"$Cl$", r"$\sigma$"]
    combis = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    ax_combis = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)]
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for combi, ax_combi in zip(combis, ax_combis):
        param1, param2 = combi
        ax = axs[ax_combi[0], ax_combi[1]]
        param1_name = param_names[param1]
        param2_name = param_names[param2]
        ax.set_facecolor('black')
        sns.kdeplot(data=sample_df, x=param1_name, y=param2_name, ax=ax,
                        fill=True, thresh=0, levels=100, cmap='mako')
        ax.scatter(theta[param1], theta[param2], color="red", marker="x", s=15, label=r"$\theta$")
        ax.scatter(theta_hat[param1], theta_hat[param2], color="blue", marker="x", s=15, label=r"$\hat{\theta}$")
        ax.set_xlabel(param_names_latex[param1])
        ax.set_ylabel(param_names_latex[param2])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axs[0, 1].set_visible(False)
    axs[0, 2].set_visible(False)
    axs[1, 2].set_visible(False)
    fig.savefig(folderpath / f"pairplot{postfix}.png")
    fig.clear()
