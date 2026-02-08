"""
Kernel implementations for Gaussian Process Regression.

This module provides various covariance functions (kernels) for use with GPR,
including base kernels and composition operators for combining kernels.

Corresponds to Cell 8 of the original notebook, extended with additional kernels.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import numpy as np


class Kernel(ABC):
    """Abstract base class for all kernels."""
    
    @abstractmethod
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute covariance between two points."""
        pass
    
    @classmethod
    @abstractmethod
    def latex_formula(cls) -> str:
        """Return LaTeX representation of the kernel formula."""
        pass
    
    @classmethod
    @abstractmethod
    def param_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Return parameter information for UI generation.
        
        Returns:
            Dict mapping parameter names to dicts with keys:
            - 'default': default value
            - 'min': minimum value
            - 'max': maximum value
            - 'step': slider step
            - 'description': human-readable description
        """
        pass
    
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return human-readable kernel name."""
        pass


class SquaredExponentialKernel(Kernel):
    """
    Squared Exponential (RBF) kernel - the most commonly used kernel.
    
    Produces infinitely differentiable (very smooth) functions.
    Original implementation from notebook Cell 8.
    """
    
    def __init__(self, sigma_f: float = 1.0, length: float = 1.0):
        """
        Args:
            sigma_f: Signal standard deviation (controls output variance)
            length: Lengthscale (controls smoothness/correlation distance)
        """
        self.sigma_f = sigma_f
        self.length = length
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dist_sq = np.linalg.norm(x1 - x2) ** 2
        return float(self.sigma_f ** 2 * np.exp(-dist_sq / (2 * self.length ** 2)))
    
    @classmethod
    def latex_formula(cls) -> str:
        return r"k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)"
    
    @classmethod
    def param_info(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'sigma_f': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Signal std dev (σf)'
            },
            'length': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Lengthscale (ℓ)'
            }
        }
    
    @classmethod
    def name(cls) -> str:
        return "Squared Exponential (RBF)"


class Matern32Kernel(Kernel):
    """
    Matérn 3/2 kernel - produces once-differentiable functions.
    
    Less smooth than RBF, often more realistic for physical processes.
    """
    
    def __init__(self, sigma_f: float = 1.0, length: float = 1.0):
        self.sigma_f = sigma_f
        self.length = length
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dist = np.linalg.norm(x1 - x2)
        sqrt3_dist_l = np.sqrt(3) * dist / self.length
        return float(self.sigma_f ** 2 * (1 + sqrt3_dist_l) * np.exp(-sqrt3_dist_l))
    
    @classmethod
    def latex_formula(cls) -> str:
        return r"k(x, x') = \sigma_f^2 \left(1 + \frac{\sqrt{3}\|x - x'\|}{\ell}\right) \exp\left(-\frac{\sqrt{3}\|x - x'\|}{\ell}\right)"
    
    @classmethod
    def param_info(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'sigma_f': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Signal std dev (σf)'
            },
            'length': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Lengthscale (ℓ)'
            }
        }
    
    @classmethod
    def name(cls) -> str:
        return "Matérn 3/2"


class Matern52Kernel(Kernel):
    """
    Matérn 5/2 kernel - produces twice-differentiable functions.
    
    Popular choice balancing smoothness and flexibility.
    """
    
    def __init__(self, sigma_f: float = 1.0, length: float = 1.0):
        self.sigma_f = sigma_f
        self.length = length
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dist = np.linalg.norm(x1 - x2)
        sqrt5_dist_l = np.sqrt(5) * dist / self.length
        return float(self.sigma_f ** 2 * (1 + sqrt5_dist_l + sqrt5_dist_l ** 2 / 3) * 
                     np.exp(-sqrt5_dist_l))
    
    @classmethod
    def latex_formula(cls) -> str:
        return r"k(x, x') = \sigma_f^2 \left(1 + \frac{\sqrt{5}\|x - x'\|}{\ell} + \frac{5\|x - x'\|^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}\|x - x'\|}{\ell}\right)"
    
    @classmethod
    def param_info(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'sigma_f': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Signal std dev (σf)'
            },
            'length': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Lengthscale (ℓ)'
            }
        }
    
    @classmethod
    def name(cls) -> str:
        return "Matérn 5/2"


class RationalQuadraticKernel(Kernel):
    """
    Rational Quadratic kernel - equivalent to infinite mixture of RBF kernels.
    
    Can model functions with varying lengthscales.
    """
    
    def __init__(self, sigma_f: float = 1.0, length: float = 1.0, alpha: float = 1.0):
        self.sigma_f = sigma_f
        self.length = length
        self.alpha = alpha
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dist_sq = np.linalg.norm(x1 - x2) ** 2
        return float(self.sigma_f ** 2 * 
                     (1 + dist_sq / (2 * self.alpha * self.length ** 2)) ** (-self.alpha))
    
    @classmethod
    def latex_formula(cls) -> str:
        return r"k(x, x') = \sigma_f^2 \left(1 + \frac{\|x - x'\|^2}{2\alpha\ell^2}\right)^{-\alpha}"
    
    @classmethod
    def param_info(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'sigma_f': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Signal std dev (σf)'
            },
            'length': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Lengthscale (ℓ)'
            },
            'alpha': {
                'default': 1.0, 'min': 0.01, 'max': 10.0, 'step': 0.1,
                'description': 'Scale mixture (α)'
            }
        }
    
    @classmethod
    def name(cls) -> str:
        return "Rational Quadratic"


class PeriodicKernel(Kernel):
    """
    Periodic kernel - for modeling periodic/seasonal patterns.
    
    Useful for time series with known periodicity.
    """
    
    def __init__(self, sigma_f: float = 1.0, length: float = 1.0, period: float = 1.0):
        self.sigma_f = sigma_f
        self.length = length
        self.period = period
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dist = np.abs(np.linalg.norm(x1 - x2))
        sin_term = np.sin(np.pi * dist / self.period)
        return float(self.sigma_f ** 2 * np.exp(-2 * sin_term ** 2 / self.length ** 2))
    
    @classmethod
    def latex_formula(cls) -> str:
        return r"k(x, x') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi|x - x'|/p)}{\ell^2}\right)"
    
    @classmethod
    def param_info(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'sigma_f': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Signal std dev (σf)'
            },
            'length': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Lengthscale (ℓ)'
            },
            'period': {
                'default': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1,
                'description': 'Period (p)'
            }
        }
    
    @classmethod
    def name(cls) -> str:
        return "Periodic"


class LinearKernel(Kernel):
    """
    Linear kernel - for modeling linear relationships.
    
    Equivalent to Bayesian linear regression.
    """
    
    def __init__(self, sigma_f: float = 1.0, center: float = 0.0):
        self.sigma_f = sigma_f
        self.center = center
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        x1_centered = np.atleast_1d(x1) - self.center
        x2_centered = np.atleast_1d(x2) - self.center
        return float(self.sigma_f ** 2 * np.dot(x1_centered, x2_centered))
    
    @classmethod
    def latex_formula(cls) -> str:
        return r"k(x, x') = \sigma_f^2 (x - c)(x' - c)"
    
    @classmethod
    def param_info(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'sigma_f': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Signal std dev (σf)'
            },
            'center': {
                'default': 0.0, 'min': -10.0, 'max': 10.0, 'step': 0.1,
                'description': 'Center (c)'
            }
        }
    
    @classmethod
    def name(cls) -> str:
        return "Linear"


class PolynomialKernel(Kernel):
    """
    Polynomial kernel - for modeling polynomial relationships.
    
    Generalizes linear kernel to higher-order polynomials.
    """
    
    def __init__(self, sigma_f: float = 1.0, center: float = 0.0, degree: int = 2):
        self.sigma_f = sigma_f
        self.center = center
        self.degree = degree
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        x1_centered = np.atleast_1d(x1) - self.center
        x2_centered = np.atleast_1d(x2) - self.center
        inner = np.dot(x1_centered, x2_centered) + 1
        return float(self.sigma_f ** 2 * inner ** self.degree)
    
    @classmethod
    def latex_formula(cls) -> str:
        return r"k(x, x') = \sigma_f^2 ((x - c)(x' - c) + 1)^d"
    
    @classmethod
    def param_info(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'sigma_f': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Signal std dev (σf)'
            },
            'center': {
                'default': 0.0, 'min': -10.0, 'max': 10.0, 'step': 0.1,
                'description': 'Center (c)'
            },
            'degree': {
                'default': 2, 'min': 1, 'max': 5, 'step': 1,
                'description': 'Degree (d)'
            }
        }
    
    @classmethod
    def name(cls) -> str:
        return "Polynomial"


class ExponentialKernel(Kernel):
    """
    Exponential kernel (Matérn 1/2) - produces rough, non-differentiable functions.
    
    Corresponds to Ornstein-Uhlenbeck process.
    """
    
    def __init__(self, sigma_f: float = 1.0, length: float = 1.0):
        self.sigma_f = sigma_f
        self.length = length
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dist = np.linalg.norm(x1 - x2)
        return float(self.sigma_f ** 2 * np.exp(-dist / self.length))
    
    @classmethod
    def latex_formula(cls) -> str:
        return r"k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|}{\ell}\right)"
    
    @classmethod
    def param_info(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'sigma_f': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Signal std dev (σf)'
            },
            'length': {
                'default': 1.0, 'min': 0.01, 'max': 5.0, 'step': 0.01,
                'description': 'Lengthscale (ℓ)'
            }
        }
    
    @classmethod
    def name(cls) -> str:
        return "Exponential (Matérn 1/2)"


# =============================================================================
# Kernel Composition Classes
# =============================================================================

class SumKernel(Kernel):
    """
    Additive combination of two kernels.
    
    k_sum(x, x') = k1(x, x') + k2(x, x')
    
    Use case: Modeling multiple independent sources of variation.
    """
    
    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return self.kernel1(x1, x2) + self.kernel2(x1, x2)
    
    @classmethod
    def latex_formula(cls) -> str:
        return r"k_{sum}(x, x') = k_1(x, x') + k_2(x, x')"
    
    def instance_latex_formula(self) -> str:
        """Return LaTeX formula with actual kernel formulas substituted."""
        k1_name = self.kernel1.name()
        k2_name = self.kernel2.name()
        return f"k(x, x') = k_{{\\text{{{k1_name}}}}}(x, x') + k_{{\\text{{{k2_name}}}}}(x, x')"
    
    @classmethod
    def param_info(cls) -> Dict[str, Dict[str, Any]]:
        return {}  # Parameters come from constituent kernels
    
    @classmethod
    def name(cls) -> str:
        return "Sum"
    
    def instance_name(self) -> str:
        return f"{self.kernel1.name()} + {self.kernel2.name()}"


class ProductKernel(Kernel):
    """
    Multiplicative combination of two kernels.
    
    k_prod(x, x') = k1(x, x') × k2(x, x')
    
    Use case: Modulating one kernel's behavior by another (e.g., decaying periodicity).
    """
    
    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return self.kernel1(x1, x2) * self.kernel2(x1, x2)
    
    @classmethod
    def latex_formula(cls) -> str:
        return r"k_{prod}(x, x') = k_1(x, x') \cdot k_2(x, x')"
    
    def instance_latex_formula(self) -> str:
        """Return LaTeX formula with actual kernel formulas substituted."""
        k1_name = self.kernel1.name()
        k2_name = self.kernel2.name()
        return f"k(x, x') = k_{{\\text{{{k1_name}}}}}(x, x') \\cdot k_{{\\text{{{k2_name}}}}}(x, x')"
    
    @classmethod
    def param_info(cls) -> Dict[str, Dict[str, Any]]:
        return {}  # Parameters come from constituent kernels
    
    @classmethod
    def name(cls) -> str:
        return "Product"
    
    def instance_name(self) -> str:
        return f"{self.kernel1.name()} × {self.kernel2.name()}"


# =============================================================================
# Kernel Registry
# =============================================================================

# Dictionary mapping kernel names to classes for easy lookup
KERNEL_REGISTRY: Dict[str, type] = {
    'Squared Exponential (RBF)': SquaredExponentialKernel,
    'Matérn 3/2': Matern32Kernel,
    'Matérn 5/2': Matern52Kernel,
    'Rational Quadratic': RationalQuadraticKernel,
    'Periodic': PeriodicKernel,
    'Linear': LinearKernel,
    'Polynomial': PolynomialKernel,
    'Exponential (Matérn 1/2)': ExponentialKernel,
}


def get_kernel_names() -> List[str]:
    """Return list of available kernel names."""
    return list(KERNEL_REGISTRY.keys())


def create_kernel(name: str, **params) -> Kernel:
    """
    Create a kernel instance by name with given parameters.
    
    Args:
        name: Kernel name (must be in KERNEL_REGISTRY)
        **params: Kernel-specific parameters
        
    Returns:
        Instantiated kernel
    """
    if name not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel: {name}. Available: {list(KERNEL_REGISTRY.keys())}")
    return KERNEL_REGISTRY[name](**params)
