import pandas as pd
import numpy as np
from scipy.optimize import minimize 
from scipy.stats import t
from sklearn.decomposition import PCA
import heapq  

class OptimalLinearSignal:
    def __init__(self, pivot: pd.Series, 
                 l2_reg: float = 0, l1_reg: float = 0, 
                 k_principal_components:int=0, 
                 p_value_threshold:float=0.001
                 ) -> None:
        """
        Initializes the model with specified pivot and regularization parameters 
        :param pivot: Series containing pivot data for feature transformation.
        :param l2_reg: L2 regularization parameter.
        :param l1_reg: L1 regularization parameter.
        :param k_principal_components: Number of principal components for PCA.
        :param p_value_threshold: Threshold for statistical significance in regularization.
        """
        self.pivot = pivot.copy()     
        self.set_params(l2_reg, l1_reg, k_principal_components, p_value_threshold)
        self.beta_neutral = None

    ##### ##### ##### ##### ##### ##### #####
    ##### #####   PUBLIC METHODS  ##### #####
    ##### ##### ##### ##### ##### ##### #####

    def set_params(self, l2_reg: float = 0, l1_reg: float = 0, k_principal_components:int=0, p_value_threshold:float=0.001,) -> None:
        """
        Sets parameters for the model including regularization and PCA components.
        :param l2_reg: L2 regularization parameter.
        :param l1_reg: L1 regularization parameter.
        :param k_principal_components: Number of principal components for PCA.
        :param p_value_threshold: Threshold for p-value in statistical significance regularization.
        """
        self.lambda_l2 = l2_reg # L2 regularization term
        self.lambda_l1 = l1_reg # L1 regularization term
        self.pca = PCA(n_components=k_principal_components) if k_principal_components > 0 else None # Set a PCA if k_principal_components is specified
        self.p_val_threshold = p_value_threshold # Set threshold for p_value for statistic significance regularization

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the model based on the input features X.
        :param X: DataFrame containing feature data with index aligned to pivot data.
        """
        
        self.features = list(X.columns) + ['Intercept Serie']  # Store feature names for consistency checks.

        self.training_size = len(X) # Compute training size

        X_tilde = self.__transform(X) # Transform features using __transform method

        if self.pca: X_tilde = self.__apply_pca(X_tilde) # Apply pca if specificied
        
        self.mu, self.sigma = np.array(X_tilde.mean()), np.array(X_tilde.cov()) # Compute mean and covariance of transformed features
        
        if self.lambda_l2: self.__apply_l2_reg(self.lambda_l2) # Apply L2 regularization if specified

        if self.beta_neutral: self.beta_neutral(X_tilde)
        
        self.alpha = self.__get_optimal_alpha(self.mu, self.sigma) # Get the optimal alpha values

        if self.lambda_l1: self.__apply_l1_reg(self.lambda_l1) # Apply L1 regularization if specified
        
        if self.p_val_threshold: self.__stat_significance_regularization(self.p_val_threshold) #Apply statistical signficance regularization if specified 

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions of the Optimal Linear Signal on new data.
        :param X: DataFrame containing new feature data.
        :return: DataFrame containing the predicted signals.
        """

        if not hasattr(self, "alpha"): raise ValueError("Model must be fit before prediction")
        if set(list(X.columns) + ['Intercept Serie']) != set(self.features): raise ValueError("The feature names should match those that were passed during fit.")
        
        X_local = X.copy()
        X_local.loc[:, 'Intercept Serie'] = 1  # Add a constant term for the intercept
        
        if self.pca: X_local = self.__apply_pca(X_local) # Apply PCA that has been fitted on train data

        return X_local.dot(self.alpha) # Compute and return optimal signals
    
    ##### ##### ##### ##### ##### ##### #####
    ##### #####  PRIVATE METHODS  ##### #####
    ##### ##### ##### ##### ##### ##### #####

    def __transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input features for model fitting or prediction.
        :param X: DataFrame of input features.
        :return: Transformed DataFrame.
        """

        X_local = X.copy()
        X_local.loc[:, 'Intercept Serie'] = 1  # Add an intercept term to the features.
        X_tilde = X_local.shift(1).multiply(self.pivot.loc[X_local.index], axis=0) #X_tilde_t = X_(t-1) * (pivot_t - pivot_(t-1))
        
        return X_tilde  
    
    def __apply_pca(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Applies PCA to DataFrame 'X'. Fits PCA if not already done, then transforms 'X' using PCA components.
        Direct matrix multiplication is used instead of standard pca.transform to avoid data scaling.
        """

        if not hasattr(self.pca, "components_"): self.pca.fit(X.ffill().fillna(0)) # Fit the PCA if it has not been fit yet
        
        return pd.DataFrame(X.dot(self.pca.components_.T), index=X.index)
        # Instead of the standard pca.transform method, direct matrix multiplication with PCA components is used. 
        # This approach is chosen because the standard pca.transform method scales the data before projection, 
        # which is not desired in this specific context. This detail is crucial as scaling can significantly alter 
        # the data characteristics, leading to different results.

    def __apply_l2_reg(self, l2_param:float) -> None:
        """
        Applies L2 regularization to the covariance matrix.
        Parameters:
        l2_param: L2 regularization parameter.
        Modifies sigma in place to include L2 regularization effect.
        """
        self.sigma += l2_param * (np.linalg.norm(self.sigma, ord='fro') + 1e-8) / len(self.sigma) * np.eye(len(self.sigma))
        self.sigma /= (1 + l2_param)

    def __get_optimal_alpha(self, mu: np.array, sigma: np.array) -> np.array:
        """
        Conducts optimization to determine the optimal alpha coefficients.
        Parameters:
        mu (np.array): The mean vector of the transformed features.
        sigma (np.array): The covariance matrix of the transformed features.
        Returns:
        np.array: Optimal alpha coefficients.
        """

        # Verify if Sigma matrix is invertible
        if np.linalg.det(sigma) == 0: raise ValueError("Cov matrix must be invertible, try increasing lambda_reg.")

        # Compute optimal alpha 
        sigma_inv = np.linalg.inv(sigma)
        alpha_hat = sigma_inv.dot(mu) / np.sqrt(mu.dot(sigma_inv.dot(mu)))

        return alpha_hat  
    
    def __apply_l1_reg(self, l1_param:float) -> None:
        """
        Applies L1 regularization to the alpha coefficients using an optimization approach.
        Parameters:
        l1_param: L1 regularization parameter.
        Updates alpha using L1 loss minimization.
        """

        alpha = self.alpha
        mu = self.mu
        sigma = self.sigma 
        lambda_l1_true = l1_param * alpha.dot(mu)

        loss = lambda alpha: - alpha.dot(mu) + lambda_l1_true * np.abs(alpha).sum()
        grad_loss = lambda alpha: (sigma.dot(alpha) - mu) + lambda_l1_true * np.sign(alpha).sum()
        constraint = {'type': 'eq', 'fun': lambda alpha: alpha.dot(sigma.dot(alpha)) - 1}

        self.alpha = minimize(fun=loss, jac=grad_loss, constraints=constraint, x0=alpha, method='SLSQP').x 
    
    def __stat_significance_regularization(self, p_value_threshold:float):
        """
        Applies t-test regularization to adjust the insignifican alpha coefficients to 0. 

        Note: 
        - The method mutates 'self.alpha' directly.
        """

        #A lambda function 'p_val' is defined to calculate the two-tailed p-value for a t-distribution. 
        #It uses the cumulative distribution function (CDF) of the t-distribution.
        p_val = lambda val, k: 2 * min(t.cdf(val, k), 1 - t.cdf(val, k))

        #The 'alpha_test' variable is calculated by normalizing the alpha coefficients      
        alpha_test =  self.alpha * self.alpha.dot(self.mu) * np.sqrt(self.training_size)

        # The alpha coefficients are then iteratively checked against the p-value threshold. 
        # If the p-value for a coefficient (calculated using 'alpha_test') is greater than the threshold, 
        # that coefficient is set to zero, indicating it is statistically insignificant
        for i, alpha_test_i in enumerate(alpha_test): 
            if p_val(alpha_test_i, self.training_size - 1) <= p_value_threshold: self.alpha[i]=0

    ##### ##### ##### ##### ##### ##### #####
    ##### #####   ADD. METHODS    ##### #####
    ##### ##### ##### ##### ##### ##### #####

    def correct(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions of the Optimal Linear PnL on new data.
        :param X: DataFrame containing new feature data.
        :return: DataFrame containing the predicted signals.
        """      

        if not hasattr(self, "alpha"): raise ValueError("Model must be fit before prediction")
        if set(list(X.columns) + ['Intercept Serie']) != set(self.features): raise ValueError("The feature names should match those that were passed during fit.")

        X_tilde = self.__transform(X)
    
        if self.pca: X_tilde = self.__apply_pca(X_tilde)

        return np.sign(X_tilde.dot(self.alpha)) 
    
    def make_beta_neutral(self, orthogonal_pivot:pd.Series):

        self.orthogonal_pivot = orthogonal_pivot.copy()

        def __apply_beta_neutral(self, X:pd.DataFrame):
            self.beta = X.multiply(self.orthogonal_pivot.loc[X.index], axis=0).mean() #beta = mean(X_t * orthogonal_pivot_t)
            
            if np.linalg.det(self.sigma) == 0: raise ValueError("Cov matrix must be invertible, try increasing lambda_reg.")

            sigma_inv_dot_beta = np.linalg.inv(self.sigma).dot(self.beta) #
            mu_sca_beta = self.mu.dot(sigma_inv_dot_beta) / self.beta.dot(sigma_inv_dot_beta)
            self.mu = self.mu - mu_sca_beta * self.beta # mu = mu - <mu, beta> beta 

        self.beta_neutral = lambda X: __apply_beta_neutral(self, X)

    def get_k_best(self, k) -> list[str]:
        """
        Identify the 'k' largest absolute values in the optimized alpha vector and return their indices.
        :param k: The number of largest elements to identify.
        :return: List of indices corresponding to the 'k' largest absolute values in the alpha vector.
        """
        
        if not hasattr(self, "alpha"): raise ValueError("Model must be fit before prediction")

        if self.pca: raise ValueError("The method get_k_best is not applicable when PCA is applied, as PCA already performs feature reduction.")

        # Use heapq.nlargest to get the 'k' largest absolute values. 
        k_largest_elements = heapq.nlargest(k, enumerate(self.alpha), key=lambda x: np.abs(x[1]))
        
        # Extract the indices of the elements from k_largest_elements.
        indexs_k_bests = [index for index, _ in k_largest_elements]

        return self.features[indexs_k_bests]  # Return the list of features corresponding to indices.
    
    def summary(self):
        return {'alpha': self.alpha,
                'mu':self.mu,
                'sigma':self.sigma,
                'PCA':self.pca.components_ if self.pca else None,
                'beta':self.beta if self.beta_neutral else None,
                }
    
    def fit_non_linear(self, X:pd.DataFrame, activation:str='basis'):

        self.fit(X)

        X_local = X.copy()
        X_local.loc[:, 'Intercept Serie'] = 1 
        if self.pca: X_local = self.__apply_pca(X_local) # Apply PCA that has been fitted on train data
        
        f_act = {'basis': lambda x:np.tanh(x)}
        delta_f_act = {'basis': lambda x:1}

        self.f_activation = f_act[activation]

        def loss_and_gradLoss(X:pd.DataFrame, pivot:pd.Series, f:callable, delta_f:callable)->tuple[callable]:
            a = lambda alpha : pivot.multiply(f(alpha.dot(X.T)))
            mu = lambda alpha : a(alpha).mean()
            sigma = lambda alpha : a(alpha).std()
            loss = lambda alpha: np.array(mu(alpha)/sigma(alpha))

            b = lambda alpha : pivot.multiply(delta_f(alpha.dot(X.T)))
            grad_mu = lambda alpha : X.multiply(b(alpha)).mean()
            grad_sigma = lambda alpha: (X.multiply(b(alpha)).multiply(a(alpha)).mean() - grad_mu(alpha)*mu(alpha)) / sigma(alpha)
            grad_loss = lambda alpha : np.array((grad_mu(alpha) - (grad_sigma(alpha)*mu(alpha) / sigma(alpha)**2) ) / sigma(alpha))

            return loss, grad_loss

        loss, grad_loss = loss_and_gradLoss(X_local, self.pivot.loc[X_local.index], f_act[activation], delta_f_act[activation])
        #print(loss(self.alpha), grad_loss(self.alpha))
        self.alpha = minimize(fun=loss, 
                              #jac=grad_loss, 
                              x0=self.alpha, 
                              method='SLSQP').x 
    
    def predict_generalized(self, X:pd.DataFrame):

        X_local = X.copy()
        X_local.loc[:, 'Intercept Serie'] = 1  # Add a constant term for the intercept
        
        if self.pca: X_local = self.__apply_pca(X_local) # Apply PCA that has been fitted on train data

        return - self.f_activation(self.alpha.dot(X_local.T))

