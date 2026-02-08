# Gaussian Process Regression

Interactive application for exploring single-objective Gaussian Process regression with various kernel functions.

## Overview

This tool visualizes how Gaussian Processes model uncertainty over functions. Given a set of observed data points, the GP computes a posterior distribution over possible functions, providing both predictions and confidence intervals.

Key features:
- 8 kernel types (RBF, Matern, Periodic, Linear, Polynomial, etc.)
- Kernel composition via addition or multiplication
- Adjustable noise and hyperparameters
- Custom data input or CSV upload
- Posterior sampling visualization
- Covariance matrix diagnostics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit application |
| `kernels.py` | Kernel implementations and composition |
| `gpr.py` | Gaussian Process Regressor class |
| `requirements.txt` | Python dependencies |

## Kernels

| Kernel | Use Case |
|--------|----------|
| Squared Exponential (RBF) | Smooth, infinitely differentiable functions |
| Matern 3/2 | Once-differentiable, realistic for physical processes |
| Matern 5/2 | Twice-differentiable, common default choice |
| Rational Quadratic | Functions with varying lengthscales |
| Periodic | Seasonal or cyclic patterns |
| Linear | Linear trends |
| Polynomial | Polynomial relationships |
| Exponential | Rough, non-differentiable functions |

Kernels can be combined using `+` (additive) or `×` (multiplicative) operators.

## References

- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006)
- Original notebook implementation adapted for interactive use
