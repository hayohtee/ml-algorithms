"""Second-Order Optimization Algorithms.

This module provides implementations of second-order optimization methods that utilize
curvature information (Hessian matrix / second derivatives) alongside gradient information
(first derivatives) to iteratively find the minimum of an objective function.

Algorithms:
    - Newton's Method: Uses a second-order Taylor series quadratic approximation of the
      objective function to take curvature-adjusted descent steps with quadratic convergence near minima.
"""

from collections.abc import Callable

import numpy as np
from autograd import grad, hessian
from numpy.typing import NDArray


def newton_method(
        fn: Callable[[NDArray[np.float64], ...], np.float64],
        w: NDArray[np.float64],
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Optimizes linear regression parameters using Newton's method.

    Newton's method uses second-order curvature information (the Hessian matrix) to
    take curvature-adjusted steps. Because the least squares loss is quadratic with
    respect to the linear model parameters, Newton's method finds the exact analytic
    minimum in a single step:
        w_{k+1} = w_k - [nabla^2 J(w_k)]^(-1) nabla J(w_k)

    Equivalently, this is solved via the regularized linear system:
        A * w_{next} = A * w - b
    where A = nabla^2 J(w) + eps * I (regularized Hessian) and b = nabla J(w) (gradient).
    A small diagonal perturbation eps * I is added for numerical stability and to ensure
    positive-definiteness and invertibility.

    Args:
        fn: The objective function to minimize.
        w: Initial parameter vector of shape (n_features + 1, 1).
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples, 1) or (n_samples,).
        eps: Small positive constant added to the Hessian diagonal (regularization/damping)
            to guarantee invertibility and numerical stability. Defaults to 1e-8.

    Returns:
        NDArray[np.float64]: Optimized parameter vector of shape (n_features + 1, 1).
    """
    # Create gradient and Hessian evaluation functions via automatic differentiation
    gradient_func = grad(fn, 0)
    hessian_func = hessian(fn, 0)

    # Evaluate gradient and Hessian at the current parameter vector
    grad_eval = gradient_func(w, X, y)
    hess_eval = hessian_func(w, X, y)

    # Ensure the Hessian matrix is properly shaped as a square 2D array (N, N)
    hess_eval.shape = (
        int((np.size(hess_eval)) ** 0.5),
        int((np.size(hess_eval)) ** 0.5),
    )

    # Regularize the Hessian with diagonal perturbation for numerical stability and invertibility
    A = hess_eval + eps * np.eye(w.size)
    b = grad_eval
    # Solve the regularized Newton system A * w = A * w - b
    w = np.linalg.solve(A, np.dot(A, w) - b)
    loss = fn(w, X, y)
    print(f"Loss: {loss:.4f}")

    return w
