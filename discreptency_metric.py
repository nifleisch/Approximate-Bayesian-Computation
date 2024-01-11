import numpy as np
from statsmodels.api import OLS, add_constant


def mean_abs_difference(D_star, x_bar):
    """
    Computes the absolute difference between the mean of the observations D_star and the mean of
    the observations D.

    Parameters:
    D_star (array): The generated observations.
    x_bar: The sample mean of the observations D.

    Returns:
    float: The absolute difference between the mean of the observations D_star and the mean of
    the observations D.
    """
    return np.abs(np.mean(D_star) - x_bar)


class PharmacokineticDiscreptancy():
    def __init__(self, D, model, prior, p, theta_0) -> None:
        """
        The pharmacokinetic discreptancy metric.

        Parameters:
        model (class): The model distribution.
        prior (class): The prior distribution.
        p (int): The number of samples to be generated.
        theta_0 (float): The initial value of theta.
        """
        self.D = D
        self.model = model
        self.prior = prior
        self.theta_0 = theta_0
        Y, X = self.generate_data(p)
        self.regression_model = self.fit_model(Y, X)

    def __call__(self, D_star) -> float:
        """
        Computes the pharmacokinetic discreptancy metric.

        Args:
            D_star (ndarray): t-dimensional array containing sampled measurements from pharmacokinetic model
            D (ndarray): t-dimensional array containing sampled measurements from pharmacokinetic model

        Returns:
            float: The pharmacokinetic discreptancy metric.
        """
        return self.weighted_euclidean_norm(self.S(D_star) - self.S(self.D))

    def generate_data(self, p) -> tuple:
        """
        Sample p thetas from the prior and simulate corresponding measurements from the pharmacokinetic model.

        Args:
            p (int): The number of samples to be generated.

        Returns:
            tuple: The sampled thetas and the corresponding measurements.
        """
        theta_list = []
        sample_list = []
        for _ in range(p):
            theta = self.prior.rvs()
            theta_list.append(theta)
            sample = self.model.rvs(theta=theta)
            sample_list.append(sample)
        return np.array(theta_list), np.array(sample_list)

    def fit_model(self, Y, X) -> OLS:
        """
        Fits a linear regression model to the data.

        Args:
            Y (ndarray): p x 4 array containing sampled thetas
            X (ndarray): p x t array containing sampled measurements from pharmacokinetic model

        Returns:
            OLS: The fitted linear regression model.
        """
        X = add_constant(X)
        regression_model = OLS(Y, X).fit()
        return regression_model
    
    def S(self, D) -> np.ndarray:
        """
        Predicts theta based on the measurements D using the fitted linear regression model.

        Args:
            D (ndarray): t-dimensional array containing sampled measurements from pharmacokinetic model

        Returns:
            ndarray: 4-dimensional array containing predicted theta.
        """
        D = np.array([1] + list(D))
        return self.regression_model.predict(D)

    def weighted_euclidean_norm(self, theta) -> float:
        """
        Computes the weighted euclidean norm of theta weighted by theta_0.

        Args:
            theta (ndarray): 4 x 1 array containing parameters of pharmacokinetic model

        Returns:
            float: The weighted euclidean norm of theta weighted by theta_0.
        """
        return np.sum(theta**2 / self.theta_0**2)