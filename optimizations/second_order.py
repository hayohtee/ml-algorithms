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


def newtons_method(
        fn: Callable[[np.ndarray], np.float64],
        w: np.ndarray,
        max_iter: int,
        eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Minimizes an objective function using Newton's method optimization.

    Newton's method uses a second-order Taylor series quadratic approximation of the objective
    function around the current point:
        q(w + Δw) ≈ f(w) + ∇f(w)ᵀ Δw + ½ Δwᵀ ∇²f(w) Δw

    Minimizing this quadratic model yields the classic Newton step:
        Δw = - [∇²f(w)]⁻¹ ∇f(w)
    leading to the parameter update:
        w_{k+1} = w_k - [∇²f(w_k)]⁻¹ ∇f(w_k)

    Equivalently, this update can be expressed as solving the linear system:
        A w_{k+1} = A w_k - b
    where A = ∇²f(w_k) + εI is the regularized Hessian matrix and b = ∇f(w_k) is the gradient vector.

    A small regularization term `eps` (εI) is added along the diagonal of the Hessian matrix
    (Levenberg-Marquardt style damping) to guarantee that A is symmetric positive-definite,
    invertible, and well-conditioned, preventing numerical instabilities near saddle points or flat regions.

    Gradients and Hessians are computed automatically via automatic differentiation using
    `autograd.grad` and `autograd.hessian`.

    Args:
        fn: Objective function to minimize, mapping a weight vector to a scalar value.
            Must be compatible with `autograd`.
        w: Initial weight vector (starting point).
        max_iter: Maximum number of optimization iterations to run.
        eps: Regularization parameter (damping factor) added to the diagonal of the Hessian
            matrix (eps * I) to ensure numerical stability and positive-definiteness.
            Defaults to 1e-8.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - weights_history: Array of visited weight vectors at each step (shape: (max_iter + 1, N)).
            - cost_history: Array of function evaluations at each step (shape: (max_iter + 1,)).
    """
    # Compute the gradient and Hessian functions via automatic differentiation
    gradient = grad(fn)
    hess = hessian(fn)

    # Record initial position and its corresponding cost
    weights_history = [w.copy()]
    cost_history = [fn(w)]

    for k in range(max_iter):
        # Evaluate the gradient and Hessian at the current position
        grad_eval = gradient(w)
        hess_eval = hess(w)

        # Ensure the Hessian matrix is properly shaped as a square 2D array (N, N)
        hess_eval.shape = (int((np.size(hess_eval)) ** 0.5), int((np.size(hess_eval)) ** 0.5))

        # Regularize the Hessian with diagonal perturbation for numerical stability and invertibility
        A = hess_eval + eps * np.eye(w.size)
        b = grad_eval

        # Solve the linear system A * w_next = A * w - b for the updated position
        w = np.linalg.solve(A, np.dot(A, w) - b)

        # Record updated position and its corresponding cost
        weights_history.append(w.copy())
        cost_history.append(fn(w))

    return np.array(weights_history), np.array(cost_history)

