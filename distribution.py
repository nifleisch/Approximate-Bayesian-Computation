import scipy.stats as stats
import numpy as np
from typing import Union
import numpy as np


np.random.seed(42)


class NormalPrior:
    def __init__(self, mu, sigma) -> None:
        """
        The normal prior distribution.

        Parameters:
        mu (float): The mean of the normal distribution.
        sigma (float): The standard deviation of the normal distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def rvs(self) -> float:
        """
        Returns a random sample from a normal distribution with mean mu and standard deviation sigma.
        """
        return stats.norm.rvs(loc=self.mu, scale=self.sigma)

    def pdf(self, x) -> Union[float, np.ndarray]:
        """
        Returns the probability density of the normal distribution at x.

        Parameters:
        x (float|array): The value at which the density is evaluated.
        """
        return stats.norm.pdf(x=x, loc=self.mu, scale=self.sigma)


class MixedGaussianPosterior:
    def __init__(self, M, x_bar, a, sigma, sigma_1) -> None:
        """
        The posterior density of a mixture of two normal distributions with means theta and
        theta + a and standard deviation sigma_1 with a normal prior with mean 0 and standard deviation
        sigma.

        Parameters:
        D (array): The observations.
        a (float): The shift of the mean of the second normal distribution.
        sigma (float): The std of the normal prior.
        sigma_1 (float): The std of the normal distriburions of the mixture.
        """
        self.M = M
        self.x_bar = x_bar
        self.a = a
        self.sigma = sigma
        self.sigma_1 = sigma_1

    def pdf(self, x) -> Union[float, np.ndarray]:
        """
        Computes the posterior density of a mixture of two normal distributions with means theta and
        theta + a and standard deviation sigma with a normal prior with mean 0 and standard deviation
        sigma.

        Parameters:
        x (array): The values of theta at which the posterior density is evaluated.

        Returns:
        float: The posterior density of the mixture of two normal distributions evaluated at theta.
        """
        mu_1 = (
            (self.sigma**2)
            / (self.sigma**2 + self.sigma_1**2 / self.M)
            * self.x_bar
        )
        mu_2 = (
            (self.sigma**2)
            / (self.sigma**2 + self.sigma_1**2 / self.M)
            * (self.x_bar - self.a)
        )
        std = np.sqrt(
            self.sigma_1**2 / (self.M + self.sigma_1**2 / self.sigma**2)
        )
        density_1 = stats.norm.pdf(x=x, loc=mu_1, scale=std)
        density_2 = stats.norm.pdf(x=x, loc=mu_2, scale=std)

        alpha = 1 / (
            1
            + np.exp(
                self.a
                * (self.x_bar - self.a / 2)
                * self.M
                / (self.M * self.sigma**2 + self.sigma_1**2)
            )
        )
        return alpha * density_1 + (1 - alpha) * density_2


class TwoPopulationsModel:
    def __init__(self, a, sigma_1, M) -> None:
        """
        The mixture of two normal distributions.

        Parameters:
        a (float): The shift of the mean of the second normal distribution.
        sigma (float): The standard deviation of the normal distributions.
        """
        self.a = a
        self.sigma_1 = sigma_1
        self.M = M

    def rvs(self, theta) -> np.ndarray:
        """
        Returns samples from a mixture of two normal distributions with means theta and
        theta + a and standard deviation sigma.

        Parameters:
        theta (float): The mean of the first normal distribution.

        Returns:
        array: The samples from the mixture of two normal distributions.
        """
        mu = theta if stats.bernoulli.rvs(0.5, size=1) == 0 else theta + self.a
        return stats.norm.rvs(loc=mu, scale=self.sigma_1, size=self.M)


class NormalProposal:
    def __init__(self, sigma) -> None:
        """
        The proposal distribution.

        Parameters:
        sigma (float): The standard deviation of the normal distribution.
        """
        self.sigma = sigma

    def rvs(self, theta) -> float:
        """
        Returns a random sample from a normal distribution with mean theta and standard deviation sigma.
        """
        return stats.norm.rvs(loc=theta, scale=self.sigma)

    def pdf(self, theta, x) -> float:
        """
        Returns the probability density of the normal distribution at x.

        Parameters:
        x (float): The value at which the density is evaluated.
        """
        return stats.norm.pdf(x=x, loc=theta, scale=self.sigma)


class TeophyllineConcentration:
    def __init__(self, drug_dose, X_0, sampling_times, N=20) -> None:
        """
        Theophylline concentration model modelled with stochastic differential equation that is
        discretized with the Euler-Maruyama method.

        Parameters:
        D (float): Known drug oral dose received by a subject.
        X_0 (float): Initial concentration of theophylline in the blood.
        sampling_times (array): The time points at which the concentration of theophylline is measured.
        N (int): Number of subintervals for Euler-Maruyama method between two consecutive samples.
        """
        self.D = drug_dose
        self.X_0 = X_0
        self.sampling_times = [0] + sampling_times.tolist()
        self.N = N

    def rvs(self, theta) -> np.ndarray:
        """
        Returns samples from the theophylline concentration model.
        
        Parameters:
        theta (array): The parameters of the model.

        Returns:
        array: The samples from the theophylline concentration model.
        """
        n = len(self.sampling_times)
        sample = np.zeros(n)
        sample[0] = self.X_0
        for i in range(n-1):
            t = self.sampling_times[i]
            t_next = self.sampling_times[i+1]
            X = sample[i]
            sample[i+1] = self.next_sample(theta, t, t_next, X)
        return sample[1:]
    
    def next_sample(self, theta, t_current, t_next, X_current) -> float:
        """
        Returns the next sample from the theophylline concentration model.

        Parameters:
        theta (array): The parameters of the model.
        t_current (float): The current time.
        t_next (float): The next time.
        X_current (float): The current concentration of theophylline in the blood.

        Returns:
        float: The next sample from the theophylline concentration model.
        """
        N = self.N
        t = np.linspace(t_current, t_next, N + 1)
        dt = (t_next - t_current) / N
        dW = stats.norm.rvs(loc=0, scale=np.sqrt(dt), size=N)
        X = np.zeros(N+1, )
        X[0] = X_current
        for i in range(1, N+1):
            X[i] = self.drift(theta, X[i-1], t[i-1], dt) + self.diffusion(theta, dW[i-1])
        return X[-1]

    def rvs_variance_reduction(self, theta) -> np.ndarray:
        pass
    
    def drift(self, theta, X, t, dt):
        K_e, K_a, Cl, _ = theta
        return X + (self.D*K_a*K_e/Cl * np.exp(-K_a*t) - K_e*X) * dt
    
    def diffusion(self, theta, dW):
        _, _, _, sigma = theta
        return sigma * dW
    

class TeophyllinePrior:
    def __init__(self, K_e, K_a, Cl, sigma) -> None:
        """
        The prior distribution of the parameters of the theophylline concentration model.

        Parameters:
        mu (float): The mean of the normal distribution.
        sigma (float): The standard deviation of the normal distribution.
        """
        self.K_e_mu, self.K_e_sigma = K_e
        self.K_a_mu, self.K_a_sigma = K_a
        self.Cl_mu, self.Cl_sigma = Cl
        self.sigma_mu, self.sigma_sigma = sigma

    def rvs(self) -> float:
        K_e = np.exp(stats.norm.rvs(loc=self.K_e_mu, scale=self.K_e_sigma))
        K_a = np.exp(stats.norm.rvs(loc=self.K_a_mu, scale=self.K_a_sigma))
        Cl = np.exp(stats.norm.rvs(loc=self.Cl_mu, scale=self.Cl_sigma))
        sigma = np.exp(stats.norm.rvs(loc=self.sigma_mu, scale=self.sigma_sigma))
        return np.array([K_e, K_a, Cl, sigma])
        
    def pdf(self, x) -> Union[float, np.ndarray]:
        """
        Returns the probability density of the normal distribution at x.

        Parameters:
        x (float|array): The value at which the density is evaluated.
        """
        K_e, K_a, Cl, sigma = x
        density_K_e = stats.norm.pdf(x=np.log(K_e), loc=self.K_e_mu, scale=self.K_e_sigma)
        density_K_a = stats.norm.pdf(x=np.log(K_a), loc=self.K_a_mu, scale=self.K_a_sigma)
        density_Cl = stats.norm.pdf(x=np.log(Cl), loc=self.Cl_mu, scale=self.Cl_sigma)
        density_sigma = stats.norm.pdf(x=np.log(sigma), loc=self.sigma_mu, scale=self.sigma_sigma)
        return density_K_e * density_K_a * density_Cl * density_sigma
    
    def marginal_pdf(self, x, parameter) -> Union[float, np.ndarray]:
        """
        Returns the probability density of the normal distribution at x.

        Parameters:
        x (float|array): The value at which the density is evaluated.
        parameter (str): The parameter for which the marginal density is computed.

        Returns:
        float: The marginal density of the parameter.
        """
        if parameter == "K_e":
            return stats.norm.pdf(x=np.log(x), loc=self.K_e_mu, scale=self.K_e_sigma)
        elif parameter == "K_a":
            return stats.norm.pdf(x=np.log(x), loc=self.K_a_mu, scale=self.K_a_sigma)
        elif parameter == "Cl":
            return stats.norm.pdf(x=np.log(x), loc=self.Cl_mu, scale=self.Cl_sigma)
        elif parameter == "sigma":
            return stats.norm.pdf(x=np.log(x), loc=self.sigma_mu, scale=self.sigma_sigma)


class MultivariateNormalProposal:
    def __init__(self, variances, scale=1.0) -> None:
        """
        The proposal distribution for MCMC, based on a multivariate normal distribution.
        This class assumes unlogarithmized inputs and outputs.

        Parameters:
        variances (list[float]): List of variances for the log-normal priors of the parameters.
        scale (float): Scaling factor for the covariance matrix.
        """
        self.variances = variances
        self.scale = scale

        self.C = np.diag(np.array(variances) * scale)
        self.x_bar = np.zeros(len(self.variances))
        self.t = 0

    def rvs(self, theta) -> np.ndarray:
        """
        Returns a random sample from a multivariate normal distribution centered at the logarithm of theta.
        The output is unlogarithmized.

        Parameters:
        theta (np.ndarray): Current state of the parameters.
        """
        log_theta = np.log(theta)
        log_theta_proposed = stats.multivariate_normal.rvs(mean=log_theta, cov=self.C)
        return np.exp(log_theta_proposed)

    def pdf(self, theta, x) -> float:
        """
        Returns the probability density of the multivariate normal distribution at x,
        relative to the logarithm of theta. Both inputs and outputs are unlogarithmized.

        Parameters:
        x (np.ndarray): Value at which the density is evaluated.
        """
        log_theta = np.log(theta)
        log_x = np.log(x)
        return stats.multivariate_normal.pdf(x=log_x, mean=log_theta, cov=self.C)
    
    def update(self, theta):
        """
        Updates the covariance matrix of the proposal distribution.

        Parameters:
        theta (np.ndarray): Current sample used for the update.
        """
        x_bar_old = self.x_bar
        self.x_bar = self.x_bar_update(self.t, self.x_bar, np.log(theta))
        self.C = self.C_update(self.t, self.C, self.x_bar, x_bar_old, np.log(theta))
        self.t += 1

    @staticmethod
    def x_bar_update(t, x_bar, x) -> np.ndarray:
        """
        Computes the sample mean of the parameters.

        Parameters:
        t (int): Current iteration.
        x_bar (np.ndarray): Current sample mean.
        x (np.ndarray): Current sample.

        Returns:
        np.ndarray: Updated sample mean.
        """
        return (x_bar * t + x) / (t + 1)
    
    @staticmethod
    def C_update(t, C, x_bar, x_bar_old, x) -> np.ndarray:
        """
        Updates the covariance matrix of the proposal distribution.

        Parameters:
        t (int): Current iteration.
        C (np.ndarray): Current covariance matrix.
        x_bar (np.ndarray): Current sample mean.
        x_bar_old (np.ndarray): Previous sample mean.
        x (np.ndarray): Current sample.

        Returns:
        np.ndarray: Updated covariance matrix.
        """
        if t < 2:
            return C
        d = len(x)
        s_d = (2.4**2) / d # after Gelman et al. (1996)
        eps = 1e-6
        return (t - 1)/t * C + s_d/t * (t * np.outer(x_bar_old, x_bar_old) - (t + 1) * np.outer(x_bar, x_bar) + np.outer(x, x) + eps * np.eye(d))
