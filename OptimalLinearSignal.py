import pandas as pd
import numpy as np
from scipy.optimize import minimize 

class OptimalLinearSignal:
    def __init__(self, prices: pd.Series, lambda_reg: float = 0, l1_reg: float=0) -> None:
        """
        Initialize the model with the given parameters.
        :param prices: DataFrame containing historical price data.
        :param lambda_reg: The overall regularization parameter (non-negative).
        :param l1_reg: Weight for L1 regularization term, should be in [0, 1].
        """
        # Initialize the class variables
        self.prices = prices.copy()  # Copy the price data
        self.lambda_l1reg = lambda_reg * l1_reg  # L1 regularization term
        self.lambda_l2reg = lambda_reg * (1 - l1_reg)  # L2 regularization term

    def __h_operator__(self, X_i: pd.Series) -> pd.Series:
        """
        Transform individual time series based on historical price data.
        :param X_i: A single time-series data.
        :return: Transformed time-series.
        """
        # This function applies the h-operator to transform the features
        return X_i.multiply(self.prices.loc[X_i.index], axis=0).shift(1).multiply(self.prices.loc[X_i.index].pct_change(), axis=0)

    def __get_optimal_beta__(self, mu: np.array, sigma: np.array) -> np.array:
        """
        Conducts optimization to determine the optimal beta coefficients.
        Parameters:
        mu (np.array): The mean vector of the transformed features.
        sigma (np.array): The covariance matrix of the transformed features.
        Returns:
        np.array: Optimal beta coefficients.
        """

        # Apply L2 regularization if specified
        if self.lambda_l2reg: sigma += self.lambda_l2reg * np.eye(len(sigma))

        # Verify if Sigma matrix is invertible
        if np.linalg.det(sigma) == 0: raise ValueError("Sigma matrix must be invertible.")

        # Compute optimal beta (or starting point if L1 regularization)
        sigma_inv = np.linalg.inv(sigma)
        beta_0 = sigma_inv.dot(mu) / np.sqrt(mu.dot(sigma_inv.dot(mu)))

        # Apply L1 regularization if specified
        if self.lambda_l1reg:
            loss = lambda beta: - beta.dot(mu) + self.lambda_l1reg * np.abs(beta).sum()
            grad_loss = lambda beta: (sigma.dot(beta) - mu) + self.lambda_l1reg * np.sign(beta).sum()
            constraint = {'type': 'eq', 'fun': lambda beta: beta.dot(sigma.dot(beta)) - 1}
            return  minimize(fun=loss, jac=grad_loss, constraints=constraint, x0=beta_0, method='SLSQP').x
        else: return beta_0  # If L1 regularization is not applied, return beta_0

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the model based on the input features X.
        :param X: DataFrame containing feature data with index aligned to price data.
        """
        X_local = X.copy()
        X_local.loc[:, 'Intercept Serie'] = 1  # Add a constant term for the intercept

        # Transform features using h-operator
        hX = pd.DataFrame(index=X_local.index)
        for col in X_local.columns : hX[col] = self.__h_operator__(X_local[col])

        # Compute mean and covariance of transformed features
        self.mu = np.array(hX.mean())
        self.sigma = np.array(hX.cov())

        # Get the optimal beta values
        self.beta = self.__get_optimal_beta__(self.mu, self.sigma)

    def predict_optimal_signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions on new data.
        :param X: DataFrame containing new feature data.
        :return: DataFrame containing the predicted signals.
        """
        X_local = X.copy()
        X_local.loc[:, 'Intercept Serie'] = 1  # Add a constant term for the intercept
        return X_local.dot(self.beta)  # Compute optimal signals

    def predict_optimal_pnl(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions on new data.
        :param X: DataFrame containing new feature data.
        :return: DataFrame containing the predicted pnl (profit and loss).
        """
        X_local = X.copy()
        X_local.loc[:, 'Intercept Serie'] = 1  # Add a constant term for the intercept

        # Transform features and compute PnL (profit and loss)
        hX = pd.DataFrame(index=X_local.index)
        for col in X_local.columns : hX[col] = self.__h_operator__(X_local[col])
        return hX.dot(self.beta)  # Compute optimal PnL

    def get_weight(self):
        """
        Retrieve the optimized weight vector.
        :return: Optimized beta values.
        """
        return self.beta  # Return the optimized beta values

    def evaluate(self): 
        """
        Evaluate the quality of the model based on the optimized beta values.
        """
        sharpe_ratio = lambda beta, mu, sigma : beta.dot(mu) / np.sqrt(beta.dot(sigma.dot(beta)))
        return sharpe_ratio(self.beta, self.mu, self.sigma)  # Evaluate the model based on the objective function