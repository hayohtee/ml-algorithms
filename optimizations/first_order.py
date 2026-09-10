"""First-Order Optimization Algorithms.

This module provides implementations of first-order optimization methods that utilize
gradient information (first derivatives) to iteratively find the minimum of an objective function.

Algorithms:
    - Gradient Descent: Iteratively steps in the direction of steepest descent (opposite to the gradient).
    - Momentum: Accelerates gradient descent by incorporating an exponential moving average of past gradients.
    - Normalized Gradient Descent: Normalizes the gradient to unit length to maintain a fixed step size regardless of gradient magnitude.
    - Component-Wise Normalized Gradient Descent: Normalizes each gradient component independently by its absolute value (sign).
"""

from collections.abc import Callable

import numpy as np
from autograd import grad, value_and_grad
from numpy.linalg import norm


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


def normalized_gradient_descent(
        fn: Callable[[np.ndarray], np.float64],
        w: np.ndarray,
        max_iter: int,
        alpha: float,
        e: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Minimizes an objective function using normalized gradient descent optimization.

    At each iteration, computes the gradient of the objective function evaluated at the
    current parameter vector using automatic differentiation (`autograd.grad`), normalizes
    it by its Euclidean (L2) norm, and updates the parameters by stepping in the direction
    of the unit negative gradient scaled by the step length `alpha`.

    Normalizing the gradient decouples the step size from the gradient magnitude, ensuring
    a consistent step size of `alpha` regardless of how steep or flat the objective surface is.

    Args:
        fn: Objective function to minimize, mapping a weight vector to a scalar value.
            Must be compatible with `autograd`.
        w: Initial weight vector (starting point).
        max_iter: Maximum number of optimization iterations to run.
        alpha: Step length / learning rate multiplier.
        e: Small positive constant (epsilon) added to the norm denominator for numerical
            stability to prevent division by zero. Defaults to 1e-8.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - weights_history: Array of visited weight vectors at each step (shape: (max_iter + 1, N)).
            - cost_history: Array of function evaluations at each step (shape: (max_iter + 1,)).
    """
    # Compute the gradient function via automatic differentiation
    gradient = grad(fn)

    # Record initial position and its corresponding cost
    weights_history = [w.copy()]
    cost_history = [fn(w)]

    for k in range(max_iter):
        # Evaluate the gradient at the current position
        grad_eval = gradient(w)

        # Step in the normalized direction of steepest descent (unit negative gradient)
        w = w - alpha * (grad_eval / (norm(grad_eval) + e))

        # Record updated position and its corresponding cost
        weights_history.append(w.copy())
        cost_history.append(fn(w))

    return np.array(weights_history), np.array(cost_history)


def component_wise(
        fn: Callable[[np.ndarray], np.float64],
        w: np.ndarray,
        max_iter: int,
        alpha: float,
        eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Minimizes an objective function using component-wise normalized gradient descent.

    Unlike standard normalized gradient descent which normalizes the full gradient vector
    by its Euclidean (L2) norm, component-wise normalization divides each coordinate of the
    gradient by its absolute value (its sign). This ensures that every coordinate moves by
    a fixed step length `alpha`, effectively optimizing within an L-infinity ball and taking
    equal step sizes along each dimension.

    Args:
        fn: Objective function to minimize, mapping a weight vector to a scalar value.
            Must be compatible with `autograd`.
        w: Initial weight vector (starting point).
        max_iter: Maximum number of optimization iterations to run.
        alpha: Step length / learning rate multiplier.
        eps: Safety threshold. Components with absolute gradient less than or equal to
            `eps` are treated as zero to prevent division by zero and noise near stationary
            points. Defaults to 1e-8.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - weights_history: Array of visited weight vectors at each step (shape: (max_iter + 1, N)).
            - cost_history: Array of function evaluations at each step (shape: (max_iter + 1,)).
    """
    # Compute the gradient function via automatic differentiation
    gradient = grad(fn)

    # Record initial position and its corresponding cost
    weights_history = [w.copy()]
    cost_history = [fn(w)]

    for k in range(max_iter):
        # Evaluate the gradient at the current position
        grad_eval = gradient(w)

        # Normalize each coordinate by its sign; zero out components below safety threshold
        coord_norm = np.where(np.abs(grad_eval) > eps, np.sign(grad_eval), 0.0)

        # Step in the component-wise normalized descent direction
        w = w - alpha * coord_norm

        # Record updated position and its corresponding cost
        weights_history.append(w.copy())
        cost_history.append(fn(w))

    return np.array(weights_history), np.array(cost_history)
