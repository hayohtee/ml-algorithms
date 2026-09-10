"""Linear Regression.

This module provides an implementation of Ordinary Least Squares (OLS) Linear
Regression from scratch using Autograd for automatic differentiation. It supports
parameter optimization via both first-order (Gradient Descent) and second-order
(Newton's Method) optimization techniques.

Classes:
    - LinearRegression: Scikit-learn style linear regression estimator supporting
      gradient descent and Newton's method optimization.

Functions:
    - model: Computes linear predictions given input features and parameter vector.
    - least_squares: Evaluates the mean squared error (least squares) cost function.
    - gradient_descent: Fits regression parameters using first-order gradient descent.
    - newton_method: Fits regression parameters using second-order Newton's method.
"""

from typing import Optional, Self

from autograd import grad, hessian
import autograd.numpy as np
from numpy.typing import NDArray


class LinearRegression:
    """Ordinary Least Squares Linear Regression model.

    Models the linear relationship between input features X and continuous target y:
        y_hat = w_0 + X @ w_{1:} = w_0 + sum_{j=1}^D (x_j * w_j)

    The model parameters (bias w_0 and weights w_{1:}) are learned by minimizing the
    mean squared error (least squares cost) using either first-order gradient descent
    or second-order Newton's method with automatic differentiation via Autograd.

    Attributes:
        learning_rate (float): Step size / learning rate for gradient descent.
        epochs (int): Number of optimization iterations.
        optimizer (str): Optimization algorithm used ('gradient' or 'newton').
        weights_ (Optional[NDArray[np.float64]]): Learned feature weights / coefficients
            of shape (n_features, 1) after fitting.
        bias_ (Optional[np.float64]): Learned intercept / bias term after fitting.
        n_features_in_ (Optional[int]): Number of input features seen during `fit`.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        optimizer: str = "gradient",
        eps: float = 1e-8,
    ):
        """Initializes the LinearRegression model.

        Args:
            learning_rate: Step size multiplier for gradient updates. Used when
                `optimizer='gradient'`. Defaults to 0.01.
            epochs: Maximum number of training epochs / optimization iterations.
                Defaults to 1000.
            optimizer: Optimization method to use for parameter estimation.
                Must be either 'gradient' (Gradient Descent) or 'newton'
                (Newton's Method). Defaults to 'gradient'.
            eps: Small positive constant added to the Hessian diagonal (regularization/damping)
            to guarantee invertibility and numerical stability. Defaults to 1e-8.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.eps = eps
        self.optimizer = optimizer
        self.weights_: Optional[NDArray[np.float64]] = None
        self.bias_: Optional[np.float64] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Self:
        """Fits the linear regression model to training data.

        Initializes parameter vector w of shape (n_features + 1, 1) uniformly at
        random from [0, 1), where w[0] corresponds to the bias term and w[1:]
        correspond to feature weights. Optimizes w using the configured optimizer
        to minimize the least squares cost function, then stores the learned bias
        and weights.

        Args:
            X: Training input samples of shape (n_samples, n_features).
            y: Target values of shape (n_samples, 1) or (n_samples,).

        Returns:
            Self: Returns the fitted estimator instance.

        Raises:
            ValueError: If `self.optimizer` is not 'gradient' or 'newton'.
        """
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        w = np.random.rand(n_features + 1, 1)

        match self.optimizer:
            case "gradient":
                w = gradient_descent(w, X, y)
                self.weights_ = w[1:]
                self.bias_ = w[0]
            case "newton":
                w = newton_method(w, X, y, eps=self.eps)
                self.weights_ = w[1:]
                self.bias_ = w[0]
            case _:
                raise ValueError("Unknown optimizer type.")

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predicts target values using the linear model.

        Args:
            X: Input samples of shape (n_samples, n_features).

        Returns:
            NDArray[np.float64]: Predicted target values of shape (n_samples, 1).

        Raises:
            ValueError: If the model has not been fitted yet (`weights_` or `bias_` is None).
            ValueError: If the number of features in `X` does not match `n_features_in_`.
        """
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model must be fitted before prediction.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have same number of features as model.")

        return model(X, np.array([self.bias_, self.weights_.flatten()]))


def model(X: NDArray[np.float64], w: NDArray[np.float64]):
    """Computes linear model predictions given inputs and a parameter vector.

    Evaluates the affine transformation:
        y_hat = w_0 + X @ w_{1:}
    where w_0 is the scalar bias (intercept) and w_{1:} are the feature weights.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        w: Parameter vector of shape (n_features + 1, 1) or (n_features + 1,),
            where w[0] represents the bias and w[1:] represent feature weights.

    Returns:
        NDArray[np.float64]: Predicted values of shape (n_samples, 1) or (n_samples,).
    """
    return w[0] + np.dot(X, w[1:])


def least_squares(
    w: NDArray[np.float64], X: NDArray[np.float64], y: NDArray[np.float64]
) -> float:
    """Computes the Mean Squared Error (least squares) loss.

    The cost function measures average squared deviation between model predictions
    and ground truth targets:
        J(w) = (1 / P) * sum_{p=1}^P (model(x_p, w) - y_p)^2

    Args:
        w: Parameter vector of shape (n_features + 1, 1) including bias and weights.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values array of shape (n_samples, 1) or (n_samples,).

    Returns:
        float: The mean squared error loss value.
    """
    cost = np.sum((model(X, w) - y) ** 2)
    return cost / np.float64(y.size)


def gradient_descent(
    w: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    learning_rate: float = 0.01,
    epochs: int = 1000,
) -> NDArray[np.float64]:
    """Optimizes linear regression parameters using gradient descent.

    Computes the gradient of the least squares objective function with respect to
    parameters w using automatic differentiation (`autograd.grad`), and updates
    w in the negative gradient direction scaled by the learning rate:
        w_{k+1} = w_k - alpha * grad_w J(w_k)

    Args:
        w: Initial parameter vector of shape (n_features + 1, 1), containing bias
            at index 0 and initial feature weights at subsequent indices.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples, 1) or (n_samples,).
        learning_rate: Step size multiplier for gradient updates. Defaults to 0.01.
        epochs: Maximum number of gradient descent iterations. Defaults to 1000.

    Returns:
        NDArray[np.float64]: Optimized parameter vector of shape (n_features + 1, 1).
    """
    # Create the gradient function via automatic differentiation
    gradient_func = grad(least_squares, 0)

    for epoch in range(epochs):
        # Compute gradients with respect to parameter vector w
        gradients = gradient_func(w, X, y)
        # Update parameters in the direction of steepest descent
        w = w - learning_rate * gradients

        if epoch % 10 == 0:
            current_loss = least_squares(w, X, y)
            print(f"Epoch {epoch}: Loss: {current_loss:.4f}")

    return w


def newton_method(
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
        w: Initial parameter vector of shape (n_features + 1, 1).
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples, 1) or (n_samples,).
        eps: Small positive constant added to the Hessian diagonal (regularization/damping)
            to guarantee invertibility and numerical stability. Defaults to 1e-8.

    Returns:
        NDArray[np.float64]: Optimized parameter vector of shape (n_features + 1, 1).
    """
    # Create gradient and Hessian evaluation functions via automatic differentiation
    gradient_func = grad(least_squares, 0)
    hessian_func = hessian(least_squares, 0)

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
    loss = least_squares(w, X, y)
    print(f"Loss: {loss:.4f}")

    return w
