# Machine Learning from Scratch

A clean, educational repository dedicated to implementing machine learning algorithms and mathematical optimization techniques from scratch using Python and NumPy.

The primary goal of this project is to build core ML models and optimizers from first principles to deeply understand their mathematical mechanics, convergence behaviors, and algorithmic details without relying on high-level framework abstractions.

---

## 📂 Repository Structure

```text
machine-learning/
├── optimizations/
│   └── zero_order.py            # Zero-order optimization (Random Search, Coordinate Search)
├── practice.ipynb               # Interactive experiments and visualization notebook
├── pyproject.toml               # Project dependencies and packaging configuration
├── uv.lock                      # Lockfile for reproducible environment setup
└── README.md
```

---

## 🚀 Implemented Algorithms

### Optimization

#### Zero-Order Optimization (Derivative-Free)
Implemented in [`optimizations/zero_order.py`](optimizations/zero_order.py):
- **Random Search**:
  - Explores the parameter space by sampling uniform random directions on an $N$-dimensional hypersphere.
  - Candidate steps are scaled by learning rate $\alpha$ and accepted greedily if they decrease the objective cost.
  - Supports constant or diminishing step length rules ($\alpha = 1 / k$).
- **Coordinate Search**:
  - Evaluates candidate steps along the standard positive and negative coordinate axes ($\pm e_1, \pm e_2, \dots, \pm e_N$).
  - Greedily updates to the coordinate step that yields the lowest objective value.
  - Supports constant or diminishing step length rules ($\alpha = 1 / k$).

---

## 🛠️ Getting Started

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

## 💡 Quick Usage Example

### Running Zero-Order Optimizers

```python
import numpy as np
from optimizations.zero_order import random_search, coordinate_search

# Define an objective function to minimize
def objective_fn(w: np.ndarray) -> np.float64:
    return np.tanh(4 * w[0] + 4 * w[1]) + np.max([0.4 * w[1]**2, 1]) + 1

# Initial weight vector
w_init = np.array([2.0, 2.0])

# 1. Run Random Search
rs_weights, rs_costs = random_search(
    fn=objective_fn,
    w=w_init,
    alpha=1.0,          # Step size
    max_iter=10,        # Number of iterations
    num_samples=1000    # Directions sampled per iteration
)

print(f"Random Search - Optimal weights: {rs_weights[-1]}, Min cost: {rs_costs[-1]}")

# 2. Run Coordinate Search
cs_weights, cs_costs = coordinate_search(
    fn=objective_fn,
    w=w_init,
    alpha=0.5,          # Step size along coordinate axes
    max_iter=10         # Number of iterations
)

print(f"Coordinate Search - Optimal weights: {cs_weights[-1]}, Min cost: {cs_costs[-1]}")
```

---

## 🗺️ Roadmap / Planned Implementations

- [x] **Zero-Order Optimization**
  - [x] Random Search
  - [x] Coordinate Search
  - [ ] Coordinate Descent
- [ ] **First-Order Optimization**
  - [ ] Gradient Descent (Batch, Mini-batch, Stochastic)
  - [ ] Momentum & Nesterov Accelerated Gradient
  - [ ] AdaGrad, RMSprop, Adam
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

## 📜 License

This project is licensed under the MIT License.
