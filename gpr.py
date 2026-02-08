"""
Gaussian Process Regressor implementation.

This module provides the core GPR class for Bayesian regression using
covariance functions (kernels) to define prior distributions over functions.

Corresponds to Cells 11-12 of the original notebook.
"""

import numpy as np
from typing import Dict, Any, Optional
from kernels import Kernel


def cov_matrix(x1: np.ndarray, x2: np.ndarray, cov_function: Kernel) -> np.ndarray:
    """
    Compute the covariance matrix between two sets of points.
    
    Corresponds to notebook Cell 11.
    
    Args:
        x1: First set of points, shape (n,) or (n, d)
        x2: Second set of points, shape (m,) or (m, d)
        cov_function: Kernel function to compute covariances
        
    Returns:
        Covariance matrix of shape (m, n)
    """
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    return np.array([[cov_function(a, b) for a in x1] for b in x2])


class GPR:
    """
    Gaussian Process Regressor.
    
    Implements GP regression with arbitrary kernels, computing posterior mean
    and variance for predictions at new input locations.
    
    Corresponds to notebook Cell 12.
    
    Mathematical background:
    Given training points x_1,...,x_n with values y_1,...,y_n and noise N(0, σ²),
    the posterior distribution at test points is:
    
        N(K_* K^{-1} y, K_{**} - K_* K^{-1} K_*^T)
    
    where:
        K = (k(x_i, x_j))_{i,j≤n} + σ² I_n  (training covariance + noise)
        K_* = (k(x_i, x_j))_{i>n, j≤n}      (cross-covariance)
        K_{**} = (k(x_i, x_j))_{i,j>n}      (test covariance)
    """
    
    # Jitter for numerical stability (slightly larger than machine epsilon)
    JITTER = 3e-7
    
    def __init__(
        self,
        data_x: np.ndarray,
        data_y: np.ndarray,
        covariance_function: Kernel,
        white_noise_sigma: float = 0.0
    ):
        """
        Initialize the GPR with training data and kernel.
        
        Args:
            data_x: Training input locations, shape (n,) or (n, d)
            data_y: Training output values, shape (n,)
            covariance_function: Kernel instance for computing covariances
            white_noise_sigma: Standard deviation of observation noise
        """
        self.data_x = np.atleast_1d(data_x)
        self.data_y = np.atleast_1d(data_y)
        self.noise = white_noise_sigma
        self.covariance_function = covariance_function
        
        # Precompute and store the inverse of the training covariance matrix
        # Adding jitter + noise variance to diagonal for numerical stability
        K = cov_matrix(self.data_x, self.data_x, covariance_function)
        noise_variance = self.JITTER + self.noise ** 2
        K_reg = K + noise_variance * np.identity(len(self.data_x))
        
        # Store for diagnostics
        self._K_train = K
        self._K_train_regularized = K_reg
        
        # Compute inverse using stable method
        try:
            self._inverse_of_covariance_matrix_of_input = np.linalg.inv(K_reg)
            self._condition_number = np.linalg.cond(K_reg)
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse if matrix is singular
            self._inverse_of_covariance_matrix_of_input = np.linalg.pinv(K_reg)
            self._condition_number = np.inf
        
        # Memory for storing last prediction results
        self._memory: Optional[Dict[str, Any]] = None
    
    def predict(self, at_values: np.ndarray) -> np.ndarray:
        """
        Predict mean and variance at new input locations.
        
        Args:
            at_values: Test input locations, shape (m,) or (m, d)
            
        Returns:
            Mean predictions at test locations, shape (m,)
            
        Side effects:
            Stores mean, covariance matrix, and variance in self._memory
        """
        at_values = np.atleast_1d(at_values)
        
        # K_* : cross-covariance between test and training points
        k_lower_left = cov_matrix(self.data_x, at_values, self.covariance_function)
        
        # K_** : covariance among test points
        k_lower_right = cov_matrix(at_values, at_values, self.covariance_function)
        
        # Posterior mean: K_* K^{-1} y
        mean_at_values = np.dot(
            k_lower_left,
            np.dot(self.data_y, self._inverse_of_covariance_matrix_of_input.T).T
        ).flatten()
        
        # Posterior covariance: K_** - K_* K^{-1} K_*^T
        cov_at_values = k_lower_right - np.dot(
            k_lower_left,
            np.dot(self._inverse_of_covariance_matrix_of_input, k_lower_left.T)
        )
        
        # Add jitter to ensure positive semi-definiteness
        cov_at_values = cov_at_values + self.JITTER * np.eye(cov_at_values.shape[0])
        
        # Extract diagonal (variance at each point)
        var_at_values = np.diag(cov_at_values)
        
        # Ensure non-negative variances (numerical issue protection)
        var_at_values = np.maximum(var_at_values, 0)
        
        # Store results
        self._memory = {
            'mean': mean_at_values,
            'covariance_matrix': cov_at_values,
            'variance': var_at_values
        }
        
        return mean_at_values
    
    def sample(self, at_values: np.ndarray, n_samples: int = 1, 
               random_state: Optional[int] = None) -> np.ndarray:
        """
        Draw random function samples from the posterior distribution.
        
        Corresponds to notebook Cell 17 (random function drawing).
        
        Args:
            at_values: Test input locations
            n_samples: Number of function samples to draw
            random_state: Random seed for reproducibility
            
        Returns:
            Function samples, shape (n_samples, len(at_values))
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Ensure prediction has been run
        if self._memory is None or len(self._memory['mean']) != len(at_values):
            self.predict(at_values)
        
        mean = self._memory['mean']
        cov = self._memory['covariance_matrix']
        
        # Draw samples from multivariate normal
        samples = np.random.multivariate_normal(mean, cov, size=n_samples)
        
        return samples
    
    @property
    def train_kernel_matrix(self) -> np.ndarray:
        """Return the training kernel matrix (before regularization)."""
        return self._K_train
    
    @property
    def condition_number(self) -> float:
        """Return the condition number of the regularized kernel matrix."""
        return self._condition_number
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return diagnostic information about the GPR.
        
        Returns:
            Dictionary with condition number, matrix shape, etc.
        """
        return {
            'n_train_points': len(self.data_x),
            'condition_number': self._condition_number,
            'noise_variance': self.noise ** 2,
            'jitter': self.JITTER,
            'kernel_name': self.covariance_function.name() if hasattr(
                self.covariance_function, 'name') else 'Unknown'
        }
