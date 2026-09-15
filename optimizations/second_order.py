"""Second-Order Optimization Algorithms.

This module provides implementations of second-order optimization methods that utilize
curvature information (Hessian matrix / second derivatives) alongside gradient information
(first derivatives) to iteratively find the minimum of an objective function.

Functions:
    - newton: Fits model parameters using second-order Newton's method with Hessian
      regularization for numerical stability.
"""

from collections.abc import Callable

import numpy as np
from autograd import grad, hessian
from numpy.typing import NDArray


def newton(
    fn: Callable[
        [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], np.float64
    ],
    w: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    epochs: int = 1000,
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Optimizes model parameters using second-order Newton's method.

    Newton's method incorporates local curvature information (the Hessian matrix)
    to take curvature-adjusted descent steps:
        w_{k+1} = w_k - [nabla^2 J(w_k)]^(-1) nabla J(w_k)

    For quadratic objectives (such as Ordinary Least Squares in linear regression),
    the quadratic approximation is exact and Newton's method reaches the global analytic
    minimum in a single step (epochs=1). For non-quadratic convex objectives (such as
    cross-entropy or softmax loss in logistic regression), Newton's method iteratively
    takes Newton-Raphson steps with quadratic local convergence.

    To guarantee invertibility and numerical stability when the Hessian is singular
    or near-singular, a regularized linear system is solved at each iteration:
        A * w_{k+1} = A * w_k - b
    where A = nabla^2 J(w_k) + eps * I (regularized Hessian) and b = nabla J(w_k) (gradient).

    Args:
        fn: Objective function to minimize, with signature `fn(w, X, y) -> loss`.
            Must be compatible with `autograd.grad` and `autograd.hessian`.
        w: Initial parameter vector of shape (n_features + 1, 1).
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples, 1) or (n_samples,).
        epochs: Maximum number of optimization iterations to run. Defaults to 1000.
        eps: Small positive constant added to the Hessian diagonal (regularization/damping)
            to guarantee positive-definiteness, invertibility, and numerical stability.
            Defaults to 1e-8.

    Returns:
        NDArray[np.float64]: Optimized parameter vector of shape (n_features + 1, 1).
    """
    # Create gradient and Hessian evaluation functions via automatic differentiation
    gradient_func = grad(fn, 0)
    hessian_func = hessian(fn, 0)

    for epoch in range(1, epochs + 1):
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

        if epoch % 5 == 0:
            current_loss = fn(w, X, y)
            print(f"Epoch {epoch}: Loss: {current_loss:.4f}")

    final_loss = fn(w, X, y)
    print(f"Final loss: {final_loss:.4f}")

    return w
