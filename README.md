# Machine Learning from Scratch

A clean, educational repository dedicated to implementing machine learning algorithms and mathematical optimization techniques from scratch using Python, NumPy, and Autograd.

The primary goal of this project is to build core ML models and optimizers from first principles to deeply understand their mathematical mechanics, convergence behaviors, and algorithmic details without relying on high-level framework abstractions.

---

## Repository Structure

```text
machine-learning/
├── optimizations/
│   ├── first_order.py           # First-order optimization (Gradient Descent, Momentum, Normalized GD, Component-Wise Normalized GD)
│   └── zero_order.py            # Zero-order optimization (Random Search, Coordinate Search, Coordinate Descent)
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
  - Candidate steps are scaled by learning rate $\alpha$ and accepted greedily if they decrease the objective cost.
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
  - Iteratively updates parameter weights in the direction of steepest descent (negative gradient) scaled by learning rate $\alpha$.
  - Supports constant or diminishing step length rules ($\alpha = 1 / k$).
- **Momentum**:
  - Accelerates gradient descent by incorporating an exponentially decaying moving average of past gradients with decay parameter $\beta$.
  - Dampens oscillations in steep directions and accelerates progress along flat, consistent descent directions.
- **Normalized Gradient Descent**:
  - Normalizes the gradient vector by its Euclidean ($L_2$) norm with a numerical stability term $\epsilon$  
  - Decouples step length from gradient magnitude, ensuring a consistent step size $\alpha$ across steep valleys and flat plateaus.
- **Component-Wise Normalized Gradient Descent**:
  - Normalizes each coordinate of the gradient vector independently by its sign / absolute value with a safety threshold $\varepsilon$
  - Moves along the vertices of an $L_\infty$ unit ball, ensuring equal step lengths along all active dimensions.

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

## Quick Usage Example

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

print(f"Random Search - Optimal weights: {rs_weights[-1]}, Min cost: {rs_costs[-1]}")

# 2. Run Coordinate Search
cs_weights, cs_costs = coordinate_search(
    fn=objective_fn,
    w=w_init,
    max_iter=10,        # Number of iterations
    alpha=0.5           # Step size along coordinate axes
)

print(f"Coordinate Search - Optimal weights: {cs_weights[-1]}, Min cost: {cs_costs[-1]}")

# 3. Run Coordinate Descent
cd_weights, cd_costs = coordinate_descent(
    fn=objective_fn,
    w=w_init,
    max_iter=10,        # Number of full coordinate sweeps
    alpha=0.5           # Step size along coordinate axes
)

print(f"Coordinate Descent - Optimal weights: {cd_weights[-1]}, Min cost: {cd_costs[-1]}")
```

### Running First-Order Optimizers

```python
import autograd.numpy as anp
from optimizations.first_order import (
    gradient_descent,
    momentum,
    normalized_gradient_descent,
    component_wise,
)

# Define an autograd-compatible objective function
def objective_fn(w: anp.ndarray):
    return anp.tanh(4 * w[0] + 4 * w[1]) + anp.maximum(0.4 * w[1]**2, 1.0) + 1.0

# Initial weight vector
w_init = anp.array([2.0, 2.0])

# 1. Run Gradient Descent
gd_weights, gd_costs = gradient_descent(
    fn=objective_fn,
    w=w_init,
    max_iter=50,        # Number of iterations
    alpha=0.1           # Step size / learning rate (or None for diminishing alpha = 1/k)
)

print(f"Gradient Descent - Optimal weights: {gd_weights[-1]}, Min cost: {gd_costs[-1]}")

# 2. Run Momentum-Accelerated Gradient Descent
mom_weights, mom_costs = momentum(
    fn=objective_fn,
    w=w_init,
    max_iter=50,        # Number of iterations
    beta=0.9,           # Momentum decay rate
    alpha=0.1           # Step size / learning rate
)

print(f"Momentum - Optimal weights: {mom_weights[-1]}, Min cost: {mom_costs[-1]}")

# 3. Run Normalized Gradient Descent
ngd_weights, ngd_costs = normalized_gradient_descent(
    fn=objective_fn,
    w=w_init,
    max_iter=50,        # Number of iterations
    alpha=0.1           # Step size
)

print(f"Normalized GD - Optimal weights: {ngd_weights[-1]}, Min cost: {ngd_costs[-1]}")

# 4. Run Component-Wise Normalized Gradient Descent
cw_weights, cw_costs = component_wise(
    fn=objective_fn,
    w=w_init,
    max_iter=50,        # Number of iterations
    alpha=0.1,          # Step size along each coordinate
    eps=1e-8            # Safety threshold
)

print(f"Component-Wise Normalized GD - Optimal weights: {cw_weights[-1]}, Min cost: {cw_costs[-1]}")
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
  - [ ] Newton's Method
  - [ ] Quasi-Newton Methods (BFGS / L-BFGS)
- [ ] **Supervised Learning**
  - [ ] Linear Regression (Analytical & Gradient Descent)
  - [ ] Logistic Regression & Softmax Regression
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
