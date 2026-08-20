"""Zero-Order (Derivative-Free) Optimization Algorithms.

This module provides implementations of zero-order optimization methods that minimize
an objective function without requiring gradient or Hessian information. These algorithms
are particularly useful for non-differentiable, noisy, or black-box optimization problems.

Algorithms:
    - Random Search: Explores candidate solutions by sampling random directions on a hypersphere.
    - Coordinate Search: Evaluates candidate steps simultaneously along all standard coordinate axes.
    - Coordinate Descent: Sequentially updates each coordinate axis one by one within each iteration.
"""

from collections.abc import Callable

import numpy as np

rng = np.random.default_rng(42)


def random_search(
        fn: Callable[[np.ndarray], np.float64],
        w: np.ndarray,
        max_iter: int,
        num_samples: int,
        alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimizes an objective function using random search optimization.

    At each iteration, samples multiple random unit directions uniformly distributed
    on the unit hypersphere, generates candidate points scaled by step size `alpha`,
    and updates the current position if a lower cost is found.

    Args:
        fn: Objective function to minimize, mapping a weight vector to a scalar value.
        w: Initial weight vector (starting point).
        max_iter: Maximum number of iterations to run.
        num_samples: Number of random candidate directions sampled per iteration.
        alpha: Step length / learning rate multiplier. If None, uses a diminishing step
            length rule (setting alpha = 1 / k at iteration k). Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - weights_history: Array of visited weight vectors at each step (shape: (max_iter + 1, N)).
            - cost_history: Array of function evaluations at each step (shape: (max_iter + 1,)).
    """
    weights_history = []
    cost_history = []

    vanishing_steplength = False
    if alpha is None:
        vanishing_steplength = True

    for k in range(1, max_iter + 1):
        # Record current position and its corresponding cost
        weights_history.append(w.copy())
        cost_history.append(fn(w))

        if vanishing_steplength:
            alpha = 1 / k

        N = np.size(w)

        # Sample random unit directions on the N-dimensional sphere
        raw_directions = rng.standard_normal((num_samples, N))
        norms = np.linalg.norm(raw_directions, axis=1, keepdims=True)
        unit_directions = raw_directions / norms

        # Generate candidate points along sampled directions
        w_candidates = w + alpha * unit_directions
        evals = np.array([fn(w_eval) for w_eval in w_candidates])
        min_idx = np.argmin(evals)

        # Greedy update: step only if the best candidate improves the cost
        if evals[min_idx] < cost_history[-1]:
            w = w_candidates[min_idx]

    # Record final position and cost
    weights_history.append(w.copy())
    cost_history.append(fn(w))

    return np.array(weights_history), np.array(cost_history)


def coordinate_search(
        fn: Callable[[np.ndarray], np.float64],
        w: np.ndarray,
        max_iter: int,
        alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimizes an objective function using coordinate search optimization.

    At each iteration, simultaneously generates candidate points along all positive and
    negative standard coordinate directions (±e_1, ±e_2, ..., ±e_N) scaled by step size `alpha`,
    and greedily updates the current position if a candidate yields a lower cost.

    Args:
        fn: Objective function to minimize, mapping a weight vector to a scalar value.
        w: Initial weight vector (starting point).
        max_iter: Maximum number of iterations to run.
        alpha: Step length / learning rate multiplier. If None, uses a diminishing step
            length rule (setting alpha = 1 / k at iteration k). Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - weights_history: Array of visited weight vectors at each step (shape: (max_iter + 1, N)).
            - cost_history: Array of function evaluations at each step (shape: (max_iter + 1,)).
    """
    # Construct standard coordinate basis directions (+/- unit basis vectors)
    directions_plus = np.eye(np.size(w), np.size(w))
    directions_minus = -np.eye(np.size(w), np.size(w))
    directions = np.concatenate((directions_plus, directions_minus), axis=0)

    weights_history = []
    cost_history = []

    vanishing_steplength = False
    if alpha is None:
        vanishing_steplength = True

    for k in range(1, max_iter + 1):
        # Record current position and its corresponding cost
        weights_history.append(w.copy())
        cost_history.append(fn(w))

        if vanishing_steplength:
            alpha = 1 / k

        # Generate candidate points along coordinate directions
        w_candidates = w + alpha * directions
        evals = np.array([fn(w_eval) for w_eval in w_candidates])
        min_idx = np.argmin(evals)

        # Greedy update: step only if the best candidate improves the cost
        if evals[min_idx] < cost_history[-1]:
            w = w_candidates[min_idx]

    # Record final position and cost
    weights_history.append(w.copy())
    cost_history.append(fn(w))

    return np.array(weights_history), np.array(cost_history)


def coordinate_descent(
        fn: Callable[[np.ndarray], np.float64],
        w: np.ndarray,
        max_iter: int,
        alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimizes an objective function using coordinate descent optimization.

    Unlike coordinate search (which evaluates all coordinate directions simultaneously
    and takes a single step), coordinate descent sequentially sweeps through each coordinate
    axis (1, 2, ..., N), immediately updating the parameter vector after each axis evaluation
    if an improvement is found.

    Args:
        fn: Objective function to minimize, mapping a weight vector to a scalar value.
        w: Initial weight vector (starting point).
        max_iter: Maximum number of full sweeps through all coordinates.
        alpha: Step length / learning rate multiplier. If None, uses a diminishing step
            length rule (setting alpha = 1 / k at iteration k). Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - weights_history: Array of visited weight vectors at each step (shape: (max_iter + 1, N)).
            - cost_history: Array of function evaluations at each step (shape: (max_iter + 1,)).
    """
    N = np.size(w)

    weights_history = []
    cost_history = []

    vanishing_steplength = False
    if alpha is None:
        vanishing_steplength = True

    for k in range(1, max_iter + 1):
        # Record current position and its corresponding cost
        weights_history.append(w.copy())
        cost_history.append(fn(w))

        if vanishing_steplength:
            alpha = 1 / k

        # Sequentially optimize each coordinate axis
        for i in range(N):
            # Positive and negative coordinate unit vectors along axis i
            direction_plus = np.zeros(N)
            direction_minus = np.zeros(N)

            direction_plus[i] = 1
            direction_minus[i] = -1

            # Candidate points along positive and negative axis i
            w_plus = w + alpha * direction_plus
            w_minus = w + alpha * direction_minus

            # Evaluate candidates
            current_cost = fn(w)
            plus_cost = fn(w_plus)
            minus_cost = fn(w_minus)

            # Greedily update position if an improvement is found
            if plus_cost < current_cost and plus_cost <= minus_cost:
                w = w_plus
            elif minus_cost < current_cost:
                w = w_minus

    # Record final position and cost
    weights_history.append(w.copy())
    cost_history.append(fn(w))

    return np.array(weights_history), np.array(cost_history)
