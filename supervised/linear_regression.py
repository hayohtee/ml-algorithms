from autograd import grad, hessian
import autograd.numpy as np
from numpy.typing import NDArray
from typing import Optional


class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, optimizer: str = "gradient"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.weights_: Optional[NDArray[np.float64]] = None
        self.bias_: Optional[np.float64] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]):
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        w = np.random.rand(n_features + 1, 1)

        match self.optimizer:
            case "gradient":
                w = gradient_descent(w, X, y)
                self.weights_ = w[1:]
                self.bias_ = w[0]
            case "newton":
                w = newton_method(w, X, y)
                self.weights_ = w[1:]
                self.bias_ = w[0]
            case _:
                raise ValueError("Unknown optimizer type.")

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model must be fitted before prediction.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have same number of features as model.")

        return model(X, np.array([self.bias_, self.weights_.flatten()]))


def model(X: NDArray[np.float64], w: NDArray[np.float64]):
    return w[0] + np.dot(X, w[1:])


def least_squares(w: NDArray[np.float64], X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    cost = np.sum((model(X, w) - y) ** 2)
    return cost / np.float64(y.size)


def gradient_descent(
        w: NDArray[np.float64],
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        learning_rate: float = 0.01,
        epochs: int = 1000
) -> NDArray[np.float64]:
    gradient_func = grad(least_squares, 0)

    for epoch in range(epochs):
        gradients = gradient_func(w, X, y)
        w = w - learning_rate * gradients

        if epoch % 10 == 0:
            current_loss = least_squares(w, X, y)
            print(f"Epoch {epoch}: Loss: {current_loss:.4f}")

    return w


def newton_method(
        w: NDArray[np.float64],
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int = 100,
        eps: float = 1e-8
) -> NDArray[np.float64]:
    gradient_func = grad(least_squares, 0)
    hessian_func = hessian(least_squares, 0)

    grad_eval = gradient_func(w, X, y)
    hess_eval = hessian_func(w, X, y)

    # Ensure the Hessian matrix is properly shaped as a square 2D array (N, N)
    hess_eval.shape = (int((np.size(hess_eval)) ** 0.5), int((np.size(hess_eval)) ** 0.5))

    # Regularize the Hessian with diagonal perturbation for numerical stability and invertibility
    A = hess_eval + eps * np.eye(w.size)
    b = grad_eval
    w = np.linalg.solve(A, np.dot(A, w) - b)
    loss = least_squares(w, X, y)
    print(f"Loss: {loss:.4f}")

    return w
