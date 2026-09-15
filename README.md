# Machine Learning from Scratch

A clean, educational repository dedicated to implementing machine learning algorithms and mathematical optimization techniques from scratch using Python, NumPy, and Autograd.

The primary goal of this project is to build core ML models and optimizers from first principles to deeply understand their mathematical mechanics, convergence behaviors, and algorithmic details without relying on high-level framework abstractions.

---

## Repository Structure

```text
machine-learning/
├── optimizations/
│   ├── first_order.py           # First-order optimization (Gradient Descent, Momentum, Normalized GD, Component-Wise Normalized GD)
│   ├── second_order.py          # Second-order optimization (Newton's Method)
│   └── zero_order.py            # Zero-order optimization (Random Search, Coordinate Search, Coordinate Descent)
├── supervised/
│   ├── linear_regression.py     # Linear Regression (Gradient Descent, Newton's Method)
│   └── logistic_regression.py   # Logistic Regression (Binary Cross-Entropy, Softmax Loss via GD & Newton's Method)
├── pyproject.toml               # Project dependencies and packaging configuration
├── uv.lock                      # Lockfile for reproducible environment setup
└── README.md
```

---

## Implemented Algorithms

### Optimization

#### Zero-Order Optimization (Derivative-Free)
Implemented in [`optimizations/zero_order.py`](optimizations/zero_order.py):
- **Random Search**:
  - Explores the parameter space by sampling uniform random directions on an $N$-dimensional hypersphere.
  - Candidate steps are scaled by step size $\alpha$ and accepted greedily if they decrease the objective cost.
  - Supports constant or diminishing step length rules ($\alpha = 1 / k$).
- **Coordinate Search**:
  - Evaluates candidate steps simultaneously along all standard positive and negative coordinate axes ($\pm e_1, \pm e_2, \dots, \pm e_N$).
  - Greedily takes the single step that yields the greatest cost reduction.
  - Supports constant or diminishing step length rules ($\alpha = 1 / k$).
- **Coordinate Descent**:
  - Sequentially sweeps through each coordinate axis ($1, 2, \dots, N$) and immediately updates the position after each axis evaluation if improved.
  - Supports constant or diminishing step length rules ($\alpha = 1 / k$).

#### First-Order Optimization (Gradient-Based)
Implemented in [`optimizations/first_order.py`](optimizations/first_order.py):
- **Gradient Descent**:
  - Leverages automatic differentiation via `autograd` to compute exact first-order gradients.
  - Iteratively updates parameter weights in the direction of steepest descent (negative gradient) scaled by learning rate $\alpha$: $w_{k+1} = w_k - \alpha \nabla_w J(w_k; X, y)$.
- **Momentum**:
  - Accelerates gradient descent by incorporating an exponentially decaying moving average of past gradients with decay parameter $\beta$.
  - Dampens oscillations in steep directions and accelerates progress along flat, consistent descent directions: $d_k = \beta d_{k-1} + (1 - \beta) \nabla f(w_{k-1})$, $w_k = w_{k-1} - \alpha d_k$.
- **Normalized Gradient Descent**:
  - Normalizes the gradient vector by its Euclidean ($L_2$) norm with a numerical stability term $\epsilon$: $w_{k+1} = w_k - \alpha \frac{\nabla J(w_k)}{\|\nabla J(w_k)\|_2 + \epsilon}$.
  - Decouples step length from gradient magnitude, ensuring a consistent step size $\alpha$ across steep valleys and flat plateaus.
- **Component-Wise Normalized Gradient Descent**:
  - Normalizes each coordinate of the gradient vector independently by its sign / absolute value with a safety threshold $\epsilon$: $w_k = w_{k-1} - \alpha \cdot \text{sign}(\nabla f(w_{k-1}))$.
  - Moves along the vertices of an $L_\infty$ unit ball, ensuring equal step lengths along all active dimensions.

#### Second-Order Optimization (Hessian-Based)
Implemented in [`optimizations/second_order.py`](optimizations/second_order.py):
- **Newton's Method**:
  - Leverages curvature information using automatic differentiation via `autograd.grad` and `autograd.hessian` to fit a second-order Taylor series quadratic approximation: $w_{k+1} = w_k - [\nabla^2 J(w_k)]^{-1} \nabla J(w_k)$.
  - Adds diagonal regularization $\epsilon I$ (damping) to guarantee positive-definiteness, invertibility, and numerical stability of the Hessian matrix.
  - Solves the linear system $(H + \epsilon I) w_{k+1} = (H + \epsilon I) w_k - \nabla J(w_k)$ to take curvature-adjusted descent steps.
  - Converges to the global optimum in a single step ($epochs=1$) for quadratic objectives (e.g. least squares), and iteratively with quadratic convergence for non-quadratic convex objectives (e.g. cross-entropy / softmax).

### Supervised Learning

#### Regression Models
Implemented in [`supervised/linear_regression.py`](supervised/linear_regression.py):
- **Linear Regression**:
  - Models the relationship between input features $X$ and continuous target $y$ via affine transformation: $\hat{y} = w_0 + X w_{1:}$.
  - Minimizes the Mean Squared Error (MSE / least squares) cost function $J(w) = \frac{1}{P} \sum_{p=1}^P (\hat{y}_p - y_p)^2$ using Autograd automatic differentiation.
  - Supports two optimization backends:
    - **Gradient Descent**: Iteratively updates parameters in the negative gradient direction with configurable learning rate $\alpha$ and epochs.
    - **Newton's Method**: Leverages exact Hessian second-order curvature with diagonal regularization ($\epsilon I$) to converge to the optimal solution in a single step.
  - Provides a Scikit-Learn style interface (`fit`, `predict`) with learned attributes `weights_` and `bias_`.

#### Classification Models
Implemented in [`supervised/logistic_regression.py`](supervised/logistic_regression.py):
- **Logistic Regression**:
  - Models the probability of binary classes via the logistic sigmoid activation: $p(y = 1 \mid X) = \sigma(w_0 + X w_{1:}) = \frac{1}{1 + e^{-(w_0 + X w_{1:})}}$.
  - Supports two standard label encoding schemes:
    - **$\{0, 1\}$ Labels**: Minimized using Binary Cross-Entropy loss: $J(w) = -\frac{1}{P} \sum_{p=1}^P [y_p \log(a_p) + (1 - y_p) \log(1 - a_p)]$.
    - **$\{-1, 1\}$ Labels**: Minimized using Softmax (logistic) loss: $J(w) = \frac{1}{P} \sum_{p=1}^P \log(1 + e^{-y_p (w_0 + x_p^\top w_{[1:]})})$.
  - Supports two optimization backends:
    - **Gradient Descent**: Iteratively optimizes weights and bias via gradient updates with configurable `learning_rate` and `epochs`.
    - **Newton's Method**: Uses second-order regularized Hessian solves across epochs for fast local quadratic convergence.
  - Provides a Scikit-Learn style interface (`fit`, `predict`) with configurable decision `threshold` (default 0.5) and learned attributes `weights_`, `bias_`, `positive_class_`, and `negative_class_`.

---

## Getting Started

### Prerequisites
- Python `>= 3.14`
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`

### Installation

Clone the repository:
```bash
git clone https://github.com/your-username/machine-learning.git
cd machine-learning
```

#### Using `uv` (Recommended)
```bash
# Sync dependencies and create the virtual environment
uv sync
```

#### Using `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Quick Usage Examples

### Running Zero-Order Optimizers

```python
import numpy as np
from optimizations.zero_order import random_search, coordinate_search, coordinate_descent

# Define an objective function to minimize
def objective_fn(w: np.ndarray) -> np.float64:
    return np.tanh(4 * w[0] + 4 * w[1]) + np.max([0.4 * w[1]**2, 1]) + 1

# Initial weight vector
w_init = np.array([2.0, 2.0])

# 1. Run Random Search
rs_weights, rs_costs = random_search(
    fn=objective_fn,
    w=w_init,
    max_iter=10,        # Number of iterations
    num_samples=1000,   # Directions sampled per iteration
    alpha=1.0           # Step size (or None for diminishing alpha = 1/k)
)

print(f"Random Search - Optimal weights: {rs_weights[-1]}, Min cost: {rs_costs[-1]:.4f}")

# 2. Run Coordinate Search
cs_weights, cs_costs = coordinate_search(
    fn=objective_fn,
    w=w_init,
    max_iter=10,        # Number of iterations
    alpha=0.5           # Step size along coordinate axes
)

print(f"Coordinate Search - Optimal weights: {cs_weights[-1]}, Min cost: {cs_costs[-1]:.4f}")

# 3. Run Coordinate Descent
cd_weights, cd_costs = coordinate_descent(
    fn=objective_fn,
    w=w_init,
    max_iter=10,        # Number of full coordinate sweeps
    alpha=0.5           # Step size along coordinate axes
)

print(f"Coordinate Descent - Optimal weights: {cd_weights[-1]}, Min cost: {cd_costs[-1]:.4f}")
```

### Running First-Order Optimizers

```python
import autograd.numpy as np
from optimizations.first_order import (
    gradient_descent,
    normalized_gradient_descent,
    momentum,
    component_wise,
)
from supervised.linear_regression import least_squares

# 1. Supervised loss optimization (Gradient Descent & Normalized GD)
np.random.seed(42)
X = np.random.randn(100, 2)
y = 1.0 + 2.0 * X[:, :1] - 1.5 * X[:, 1:] + 0.05 * np.random.randn(100, 1)
w_init = np.zeros((3, 1))

# Gradient Descent
w_gd = gradient_descent(least_squares, w_init.copy(), X, y, learning_rate=0.05, epochs=300)
print(f"Gradient Descent - Optimized weights:\n{w_gd.flatten()}")

# Normalized Gradient Descent
w_ngd = normalized_gradient_descent(least_squares, w_init.copy(), X, y, learning_rate=0.05, epochs=300)
print(f"Normalized GD - Optimized weights:\n{w_ngd.flatten()}")

# 2. General function optimization (Momentum & Component-Wise)
def objective_fn(w: np.ndarray) -> np.float64:
    return np.tanh(4 * w[0] + 4 * w[1]) + np.maximum(0.4 * w[1]**2, 1.0) + 1.0

w_scalar_init = np.array([2.0, 2.0])

# Momentum-Accelerated Gradient Descent
mom_weights, mom_costs = momentum(objective_fn, w_scalar_init, max_iter=50, beta=0.9, alpha=0.1)
print(f"Momentum - Optimal weights: {mom_weights[-1]}, Min cost: {mom_costs[-1]:.4f}")

# Component-Wise Normalized Gradient Descent
cw_weights, cw_costs = component_wise(objective_fn, w_scalar_init, max_iter=50, alpha=0.1)
print(f"Component-Wise - Optimal weights: {cw_weights[-1]}, Min cost: {cw_costs[-1]:.4f}")
```

### Running Second-Order Optimizers (Newton's Method)

```python
import autograd.numpy as np
from optimizations.second_order import newton
from supervised.linear_regression import least_squares

# Generate synthetic dataset
np.random.seed(42)
X = np.random.randn(50, 2)
y = 3.0 + 1.5 * X[:, :1] - 2.0 * X[:, 1:] + 0.05 * np.random.randn(50, 1)

# Initial parameter vector: [bias, w1, w2]
w_init = np.zeros((X.shape[1] + 1, 1))

# Run Newton's Method (for least squares, exact minimum is found in 1 step)
w_opt = newton(
    fn=least_squares,
    w=w_init,
    X=X,
    y=y,
    epochs=1,
    eps=1e-8
)

print(f"Newton's Method - Optimal parameters:\n{w_opt.flatten()}")
```

### Running Linear Regression

```python
import numpy as np
from supervised.linear_regression import LinearRegression

# Generate synthetic linear dataset: y = 2.5 * x1 - 1.5 * x2 + 4.0 + noise
np.random.seed(42)
X = np.random.randn(100, 2)
true_weights = np.array([[2.5], [-1.5]])
true_bias = 4.0
y = true_bias + np.dot(X, true_weights) + 0.1 * np.random.randn(100, 1)

# 1. Fit using Gradient Descent
reg_gd = LinearRegression(learning_rate=0.01, epochs=1000, optimizer="gradient")
reg_gd.fit(X, y)
predictions_gd = reg_gd.predict(X)
print(f"Gradient Descent - Bias: {reg_gd.bias_}, Weights: {reg_gd.weights_.flatten()}")

# 2. Fit using Newton's Method
reg_newton = LinearRegression(optimizer="newton")
reg_newton.fit(X, y)
predictions_newton = reg_newton.predict(X)
print(f"Newton's Method  - Bias: {reg_newton.bias_}, Weights: {reg_newton.weights_.flatten()}")
```

### Running Logistic Regression

```python
import numpy as np
from supervised.logistic_regression import LogisticRegression

# Generate synthetic binary dataset: boundary 2.0 * x1 - 3.0 * x2 + 0.5 > 0
np.random.seed(42)
X = np.random.randn(100, 2)
y = (2.0 * X[:, :1] - 3.0 * X[:, 1:] + 0.5 > 0).astype(np.float64)

# 1. Fit using Gradient Descent (Binary Cross-Entropy)
clf_gd = LogisticRegression(learning_rate=0.1, epochs=1000, optimizer="gradient")
clf_gd.fit(X, y)
preds_gd = clf_gd.predict(X)
print(f"Logistic Regression (GD) - Accuracy: {np.mean(preds_gd == y) * 100:.1f}%")

# 2. Fit using Newton's Method
clf_newton = LogisticRegression(epochs=20, optimizer="newton")
clf_newton.fit(X, y)
preds_newton = clf_newton.predict(X)
print(f"Logistic Regression (Newton) - Accuracy: {np.mean(preds_newton == y) * 100:.1f}%")
```

---

## Roadmap / Planned Implementations

- [x] **Zero-Order Optimization**
  - [x] Random Search
  - [x] Coordinate Search
  - [x] Coordinate Descent
- [x] **First-Order Optimization**
  - [x] Gradient Descent
  - [x] Momentum
  - [x] Normalized Gradient Descent
  - [x] Component-Wise Normalized Gradient Descent
- [ ] **Second-Order Optimization**
  - [x] Newton's Method
  - [ ] Quasi-Newton Methods (BFGS / L-BFGS)
- [ ] **Supervised Learning**
  - [x] Linear Regression (Gradient Descent & Newton's Method)
  - [x] Logistic Regression & Softmax Regression (Gradient Descent & Newton's Method)
  - [ ] Decision Trees & Random Forests
  - [ ] Support Vector Machines (SVM)
  - [ ] K-Nearest Neighbors (KNN)
- [ ] **Unsupervised Learning**
  - [ ] K-Means Clustering
  - [ ] Principal Component Analysis (PCA)
- [ ] **Neural Networks**
  - [ ] Multi-Layer Perceptron (MLP) with Backpropagation

---

## License

This project is licensed under the MIT License.
