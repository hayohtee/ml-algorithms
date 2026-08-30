"""First-Order Optimization Algorithms.

This module provides implementations of first-order optimization methods that utilize
gradient information (first derivatives) to iteratively find the minimum of an objective function.

Algorithms:
    - Gradient Descent: Iteratively steps in the direction of steepest descent (opposite to the gradient).
"""

from collections.abc import Callable

from autograd import grad
import numpy as np


def gradient_descent(
    fn: Callable[[np.ndarray], np.float64], 
    w: np.ndarray, 
    max_iter: int, 
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimizes an objective function using gradient descent optimization.

    At each iteration, computes the gradient of the objective function evaluated at the
    current parameter vector using automatic differentiation (`autograd.grad`), and updates
    the parameters by stepping in the opposite direction of the gradient scaled by the learning rate `alpha`.

    Args:
        fn: Objective function to minimize, mapping a weight vector to a scalar value.
            Must be compatible with `autograd`.
        w: Initial weight vector (starting point).
        max_iter: Maximum number of optimization iterations to run.
        alpha: Step length / learning rate multiplier.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - weights_history: Array of visited weight vectors at each step (shape: (max_iter + 1, N)).
            - cost_history: Array of function evaluations at each step (shape: (max_iter + 1,)).
    """
    # Compute the gradient function via automatic differentiation
    gradient = grad(fn)

    # Record initial position and its corresponding cost
    weights_history = [w]
    cost_history = [fn(w)]
    
    for k in range(max_iter):
        # Evaluate the gradient at the current position
        grad_eval = gradient(w)

        # Step in the direction of steepest descent (negative gradient)
        w = w - alpha * grad_eval

        # Record updated position and its corresponding cost
        weights_history.append(w)
        cost_history.append(fn(w))

    return np.array(weights_history), np.array(cost_history)
