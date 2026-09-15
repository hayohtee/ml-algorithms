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
"""

from typing import Self

import autograd.numpy as np
from numpy.typing import NDArray

from optimizations.first_order import gradient_descent
from optimizations.second_order import newton


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
        eps (float): Small positive constant added to the Hessian diagonal
            for numerical stability and invertibility in Newton's method.
        weights_ (NDArray[np.float64] | None): Learned feature weights / coefficients
            of shape (n_features, 1) after fitting.
        bias_ (np.float64 | None): Learned intercept / bias term after fitting.
        n_features_in_ (int | None): Number of input features seen during `fit`.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        optimizer: str = "gradient",
        eps: float = 1e-8,
    ) -> None:
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
        self.weights_: NDArray[np.float64] | None = None
        self.bias_: np.float64 | None = None
        self.n_features_in_: int | None = None

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
        n_features = X.shape[1]
        self.n_features_in_ = n_features

        w = np.random.rand(n_features + 1, 1)

        match self.optimizer:
            case "gradient":
                w = gradient_descent(
                    least_squares,
                    w,
                    X,
                    y,
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                )
                self.weights_ = w[1:]
                self.bias_ = w[0]
            case "newton":
                w = newton(least_squares, w, X, y, epochs=1, eps=self.eps)
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


def model(X: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
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
    w: NDArray[np.float64], 
    X: NDArray[np.float64], 
    y: NDArray[np.float64],
) -> np.float64:
    """Computes the Mean Squared Error (least squares) loss.

    The cost function measures average squared deviation between model predictions
    and ground truth targets:
        J(w) = (1 / P) * sum_{p=1}^P (model(x_p, w) - y_p)^2

    Args:
        w: Parameter vector of shape (n_features + 1, 1) including bias and weights.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values array of shape (n_samples, 1) or (n_samples,).

    Returns:
        np.float64: The mean squared error loss value.
    """
    cost = np.sum((model(X, w) - y) ** 2)
    return cost / np.float64(y.size)

