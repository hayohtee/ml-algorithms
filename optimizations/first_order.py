"""First-Order Optimization Algorithms.

This module provides implementations of first-order optimization methods that utilize
gradient information (first derivatives) to iteratively find the minimum of an objective function.

Algorithms:
    - Gradient Descent: Iteratively steps in the direction of steepest descent (opposite to the gradient).
    - Momentum: Accelerates gradient descent by incorporating an exponential moving average of past gradients.
"""

from collections.abc import Callable

from autograd import grad
from autograd import value_and_grad
import numpy as np


def gradient_descent(
        fn: Callable[[np.ndarray], np.float64],
        w: np.ndarray,
        max_iter: int,
        alpha: float | None = None,
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
        alpha: Step length / learning rate multiplier. If None, uses a diminishing step
            length rule (setting alpha = 1 / k at iteration k). Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - weights_history: Array of visited weight vectors at each step (shape: (max_iter + 1, N)).
            - cost_history: Array of function evaluations at each step (shape: (max_iter + 1,)).
    """
    # Compute the gradient function via automatic differentiation
    gradient = grad(fn)

    diminishing_steplength = False
    if alpha is None:
        diminishing_steplength = True

    # Record initial position and its corresponding cost
    weights_history = [w.copy()]
    cost_history = [fn(w)]

    for k in range(1, max_iter + 1):
        if diminishing_steplength:
            alpha = 1 / k

        # Evaluate the gradient at the current position
        grad_eval = gradient(w)

        # Step in the direction of steepest descent (negative gradient)
        w = w - alpha * grad_eval

        # Record updated position and its corresponding cost
        weights_history.append(w.copy())
        cost_history.append(fn(w))

    return np.array(weights_history), np.array(cost_history)


def momentum(
        fn: Callable[[np.ndarray], np.float64],
        w: np.ndarray,
        max_iter: int,
        beta: float,
        alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Minimizes an objective function using momentum-accelerated gradient descent.

    Accelerates standard gradient descent by maintaining an exponential moving average
    of past gradient vectors (momentum), which dampens oscillations across steep directions
    and accelerates motion along flat, consistent descent directions.

    Args:
        fn: Objective function to minimize, mapping a weight vector to a scalar value.
            Must be compatible with `autograd`.
        w: Initial weight vector (starting point).
        max_iter: Maximum number of optimization iterations to run.
        beta: Momentum decay rate parameter (typically between 0 and 1, e.g., 0.9),
            controlling the weighting between historical momentum and the current gradient.
        alpha: Step length / learning rate multiplier.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - weights_history: Array of visited weight vectors at each step (shape: (max_iter + 1, N)).
            - cost_history: Array of function evaluations at each step (shape: (max_iter + 1,)).
    """
    # Compute the value and gradient function via automatic differentiation
    gradient = value_and_grad(fn)

    # Record initial position and its corresponding cost
    cost_eval, grad_eval = gradient(w)
    weights_history = [w.copy()]
    cost_history = [cost_eval]

    # Initialize momentum direction with the initial gradient
    d = grad_eval.copy()

    for k in range(1, max_iter + 1):
        # Update position along the momentum direction
        w = w - alpha * d

        # Evaluate cost and gradient at the new position
        cost_eval, grad_eval = gradient(w)

        # Record updated position and its corresponding cost
        weights_history.append(w.copy())
        cost_history.append(cost_eval)

        # Update momentum direction for the next step via exponential moving average
        d = (beta * d) + (1 - beta) * grad_eval

    return np.array(weights_history), np.array(cost_history)
