"""Logistic Regression.

This module provides an implementation of binary Logistic Regression from scratch
using Autograd for automatic differentiation. It supports both {0, 1} label encoding
via Binary Cross-Entropy loss and {-1, 1} label encoding via Softmax (logistic loss),
with parameter optimization via both first-order (Gradient Descent) and second-order
(Newton's Method) optimization techniques.

Classes:
    - LogisticRegression: Scikit-learn style binary logistic regression classifier
      supporting gradient descent and Newton's method optimization.

Functions:
    - model: Computes linear model predictions (logits) given input features and weights.
    - sigmoid: Evaluates the logistic sigmoid activation function.
    - cross_entropy: Evaluates binary cross-entropy cost for labels in {0, 1}.
    - softmax: Evaluates the softmax (logistic) cost function for labels in {-1, 1}.
"""

from typing import Self

import autograd.numpy as np
from numpy.typing import NDArray

from optimizations.first_order import gradient_descent
from optimizations.second_order import newton


class LogisticRegression:
    """Binary Logistic Regression classifier.

    Models the probability of binary outcomes using the logistic sigmoid function applied
    to a linear combination of input features:
        p(y = positive_class | X) = sigmoid(w_0 + X @ w_{1:})

    Supports two standard binary labeling conventions:
        - Labels in {0, 1}: Minimized using Binary Cross-Entropy loss.
        - Labels in {-1, 1}: Minimized using the Softmax (logistic) loss.

    The model parameters (bias w_0 and weights w_{1:}) are learned by minimizing the
    selected cost function using either first-order gradient descent or second-order
    Newton's method with automatic differentiation via Autograd.

    Attributes:
        learning_rate (float): Step size / learning rate for gradient descent updates.
        epochs (int): Number of optimization iterations.
        optimizer (str): Optimization algorithm used ('gradient' or 'newton').
        threshold (float): Decision threshold for classifying positive versus negative class.
        weights_ (NDArray[np.float64] | None): Learned feature weights of shape (n_features, 1)
            after fitting. Initialized to None.
        bias_ (np.float64 | None): Learned intercept / bias term after fitting.
            Initialized to None.
        positive_class_ (int | None): Label corresponding to the positive class (1).
            Initialized to None.
        negative_class_ (int | None): Label corresponding to the negative class (0 or -1).
            Initialized to None.
        n_features_in_ (int | None): Number of input features seen during `fit`.
            Initialized to None.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        optimizer: str = "gradient",
        threshold: float = 0.5,
    ) -> None:
        """Initializes the LogisticRegression classifier.

        Args:
            learning_rate: Step size multiplier for gradient updates. Used when
                `optimizer='gradient'`. Defaults to 0.01.
            epochs: Maximum number of training epochs / optimization iterations.
                Defaults to 1000.
            optimizer: Optimization method to use for parameter estimation.
                Must be either 'gradient' (Gradient Descent) or 'newton'
                (Newton's Method). Defaults to 'gradient'.
            threshold: Probability threshold for predicting the positive class.
                Samples with predicted probability >= `threshold` are assigned
                to `positive_class_`. Defaults to 0.5.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.weights_: NDArray[np.float64] | None = None
        self.bias_: np.float64 | None = None
        self.positive_class_: int | None = None
        self.negative_class_: int | None = None
        self.threshold = threshold
        self.n_features_in_: int | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Self:
        """Fits the logistic regression model to training data.

        Validates that target labels are binary ({0, 1} or {-1, 1}), selects the
        appropriate cost function (cross-entropy for {0, 1}, softmax for {-1, 1}),
        initializes parameter vector w uniformly at random, and minimizes the cost
        function using the specified optimizer.

        Args:
            X: Training input samples of shape (n_samples, n_features).
            y: Binary target labels of shape (n_samples, 1) or (n_samples,).
                Values must be either {0, 1} or {-1, 1}.

        Returns:
            Self: Returns the fitted estimator instance.

        Raises:
            ValueError: If `y` does not contain exactly two unique classes.
            ValueError: If labels in `y` are not in {0, 1} or {-1, 1}.
            ValueError: If `self.optimizer` is not 'gradient' or 'newton'.
        """
        if np.unique(y).size != 2:
            raise ValueError("y must contains only two class")

        n_features = X.shape[1]
        self.n_features_in_ = n_features

        w = np.random.rand(n_features + 1, 1)

        self.positive_class_ = y.max()
        self.negative_class_ = y.min()

        if self.negative_class_ == 0 and self.positive_class_ == 1:
            cost = cross_entropy
        elif self.negative_class_ == -1 and self.positive_class_ == 1:
            cost = softmax
        else:
            raise ValueError("y must be element of {0, 1} or {-1, 1}")

        match self.optimizer:
            case "gradient":
                w = gradient_descent(
                    cost, w, X, y, epochs=self.epochs, learning_rate=self.learning_rate
                )
                self.bias_ = w[0]
                self.weights_ = w[1:]
            case "newton":
                w = newton(cost, w, X, y, epochs=self.epochs)
                self.bias_ = w[0]
                self.weights_ = w[1:]
            case _:
                raise ValueError("Unknown optimizer type")

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predicts binary class labels for input samples.

        Computes predicted probabilities via the sigmoid function and maps them
        to `positive_class_` or `negative_class_` based on `self.threshold`.

        Args:
            X: Input samples of shape (n_samples, n_features).

        Returns:
            NDArray[np.float64]: Predicted class labels of shape (n_samples, 1)
                matching the encoding seen during `fit` ({0, 1} or {-1, 1}).

        Raises:
            ValueError: If the model has not been fitted yet (`weights_` or `bias_` is None).
            ValueError: If the number of features in `X` does not match `n_features_in_`.
        """
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model must be fitted before prediction.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features as the model.")

        weights = np.array([self.bias_, self.weights_.flatten()])
        predicted_probs = sigmoid(model(X, weights))
        predicted_labels = np.where(
            predicted_probs >= self.threshold,
            self.positive_class_,
            self.negative_class_,
        )

        return predicted_labels


def model(X: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes linear model predictions (logits) given inputs and a parameter vector.

    Evaluates the affine transformation:
        z = w_0 + X @ w_{1:}
    where w_0 is the scalar bias (intercept) and w_{1:} are the feature weights.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        w: Parameter vector of shape (n_features + 1, 1) or (n_features + 1,),
            where w[0] represents the bias and w[1:] represent feature weights.

    Returns:
        NDArray[np.float64]: Linear combination values (logits) of shape (n_samples, 1)
            or (n_samples,).
    """
    return w[0] + np.dot(X, w[1:])


def sigmoid(t: NDArray[np.float64] | np.float64 | float) -> NDArray[np.float64]:
    """Computes the logistic sigmoid activation function.

    Evaluates:
        sigma(t) = 1 / (1 + exp(-t))

    Args:
        t: Input array or scalar value.

    Returns:
        NDArray[np.float64]: Element-wise sigmoid activation values bounded in (0, 1).
    """
    return 1 / (1 + np.exp(-t))


def cross_entropy(
    w: NDArray[np.float64], 
    X: NDArray[np.float64], 
    y: NDArray[np.float64],
) -> np.float64:
    """Computes the Binary Cross-Entropy loss for labels in {0, 1}.

    Evaluates average negative log-likelihood for binary targets:
        J(w) = - (1 / P) * sum_{p=1}^P [ y_p * log(a_p) + (1 - y_p) * log(1 - a_p) ]
    where a_p = sigmoid(model(x_p, w)) and P is the total number of samples.

    Args:
        w: Parameter vector of shape (n_features + 1, 1) including bias and weights.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target binary label vector of shape (n_samples, 1) or (n_samples,)
            with elements in {0, 1}.

    Returns:
        np.float64: The binary cross-entropy loss value.
    """
    a = sigmoid(model(X, w))

    # Compute cost of label 0 points
    ind = np.argwhere(y == 0)[:, 1]
    cost = -1 * np.sum(np.log(1 - a[:, ind]))

    # Add cost of label 1 points
    ind = np.argwhere(y == 1)[:, 1]
    cost -= np.sum(np.log(a[:, ind]))

    return cost / np.float64(y.size)


def softmax(
    w: NDArray[np.float64], 
    X: NDArray[np.float64], 
    y: NDArray[np.float64],
) -> np.float64:
    """Computes the Softmax (logistic) loss for labels in {-1, 1}.

    Evaluates the smooth approximation to the 0-1 classification loss for binary targets
    encoded as {-1, 1}:
        J(w) = (1 / P) * sum_{p=1}^P log(1 + exp(-y_p * model(x_p, w)))
    where P is the total number of samples.

    Args:
        w: Parameter vector of shape (n_features + 1, 1) including bias and weights.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target binary label vector of shape (n_samples, 1) or (n_samples,)
            with elements in {-1, 1}.

    Returns:
        np.float64: The softmax (logistic) loss value.
    """
    cost = np.sum(np.log(1 + np.exp(-y * model(X, w))))
    return cost / np.float64(y.size)

