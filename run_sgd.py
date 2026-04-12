"""
SGD Algorithm: Prototype & Preliminary Results

Author: Edwin Fabricio Quizhpe Aguilar

  Every gradient computation is explicit and corresponds
  directly to the mathematical formulas in the report.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

#  Reproducibility

SEED = 42
np.random.seed(SEED)

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


#  1.  FROM-SCRATCH SGD — LINEAR REGRESSION


class SGDLinearRegression:
    """
    Stochastic Gradient Descent for Linear Regression.

    Loss:     MSE(w) = (1/2n) * sum_i (y_i - w^T x_i)^2
    Gradient (single sample i):
              grad_i = (w^T x_i - y_i) * x_i
    Update:   w_{t+1} = w_t - eta_t * grad_i
    """

    def __init__(self, learning_rate: float = 0.01,
                 epochs: int = 50,
                 lr_schedule: str = "constant",
                 decay: float = 0.01):
        self.lr0          = learning_rate
        self.epochs       = epochs
        self.lr_schedule  = lr_schedule
        self.decay        = decay
        self.weights_     = None
        self.bias_        = None
        self.loss_history = []

    # ----------------------------------------------------------
    def _get_lr(self, t: int) -> float:
        """
        Three learning-rate schedules:
          constant   : eta_t = eta_0
          time_decay : eta_t = eta_0 / (1 + kappa * t)
                       satisfies Robbins-Monro conditions:
                         sum eta_t = inf, sum eta_t^2 < inf
          step_decay : eta_t = eta_0 * gamma^(floor(t / s))
        """
        if self.lr_schedule == "constant":
            return self.lr0
        elif self.lr_schedule == "time_decay":
            return self.lr0 / (1.0 + self.decay * t)
        elif self.lr_schedule == "step_decay":
            drop_every = 10   # steps between drops
            gamma      = 0.5  # multiplicative factor
            return self.lr0 * (gamma ** (t // drop_every))
        return self.lr0

    # ----------------------------------------------------------
    def _mse_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Full-dataset MSE — used only for monitoring, NOT for training."""
        preds = X @ self.weights_ + self.bias_
        return float(np.mean((y - preds) ** 2))

    # ----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDLinearRegression":
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)   # w_0 = 0
        self.bias_    = 0.0
        t = 0  # global step counter

        for epoch in range(self.epochs):
            # Shuffle once per epoch to avoid cyclic patterns
            idx = np.random.permutation(n_samples)

            for i in idx:
                xi, yi = X[i], y[i]
                lr = self._get_lr(t)

                # ── Forward pass ──────────────────────────────
                pred_i = np.dot(xi, self.weights_) + self.bias_

                # ── Gradient of MSE w.r.t. single sample ──────
                #    dL/dw = (pred - y) * x
                #    dL/db = (pred - y)
                error    = pred_i - yi
                grad_w   = error * xi
                grad_b   = error

                # ── SGD update: w_{t+1} = w_t - eta * grad ────
                self.weights_ -= lr * grad_w
                self.bias_    -= lr * grad_b
                t += 1

            # Record full-dataset loss once per epoch
            self.loss_history.append(self._mse_loss(X, y))

        return self

    # ----------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights_ + self.bias_


#  2.  FROM-SCRATCH SGD — LOGISTIC REGRESSION


class SGDLogisticRegression:
    """
    SGD for Binary Logistic Regression.

    Loss (Binary Cross-Entropy):
      Q(w) = -(1/n) sum_i [ y_i log(p_i) + (1-y_i) log(1-p_i) ]
    where p_i = sigmoid(w^T x_i + b)

    Gradient (single sample i):
      grad_w = (p_i - y_i) * x_i
      grad_b = (p_i - y_i)

    This gradient is derived by taking dQ_i/dw through the chain rule:
      dQ_i/dw = dQ_i/dp_i * dp_i/dz * dz/dw
              = (-(y_i/p_i) + (1-y_i)/(1-p_i)) * p_i*(1-p_i) * x_i
              = (p_i - y_i) * x_i       <-- elegant simplification
    """

    def __init__(self, learning_rate: float = 0.1, epochs: int = 50):
        self.lr           = learning_rate
        self.epochs       = epochs
        self.weights_     = None
        self.bias_        = None
        self.loss_history = []

    # ----------------------------------------------------------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid:
          For z >= 0: sigma = 1 / (1 + exp(-z))
          For z <  0: sigma = exp(z) / (1 + exp(z))
        Avoids overflow in exp(-z) for large negative z.
        """
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z))
        )

    # ----------------------------------------------------------
    def _bce_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Binary Cross-Entropy over full dataset — monitoring only."""
        z   = X @ self.weights_ + self.bias_
        p   = self._sigmoid(z)
        eps = 1e-15  # numerical stability: avoid log(0)
        return float(-np.mean(
            y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)
        ))

    # ----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDLogisticRegression":
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_    = 0.0

        for epoch in range(self.epochs):
            idx = np.random.permutation(n_samples)

            for i in idx:
                xi, yi = X[i], float(y[i])

                # ── Forward pass ──────────────────────────────
                z     = np.dot(xi, self.weights_) + self.bias_
                p_hat = self._sigmoid(z)         # predicted probability

                # ── Gradient of BCE w.r.t. single sample ──────
                error  = p_hat - yi              # (p_i - y_i)
                grad_w = error * xi
                grad_b = error

                # ── SGD update ────────────────────────────────
                self.weights_ -= self.lr * grad_w
                self.bias_    -= self.lr * grad_b

            self.loss_history.append(self._bce_loss(X, y))

        return self

    # ----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)



#  3.  DATASET GENERATORS


def make_linear_dataset(n=500, d=5, noise=0.5):
    """Synthetic linear regression dataset: y = X @ w_true + noise."""
    X      = np.random.randn(n, d)
    w_true = np.array([1.5, -2.0, 0.8, 3.1, -0.5])[:d]
    y      = X @ w_true + noise * np.random.randn(n)
    return X, y, w_true


def make_binary_dataset(n=500, d=4):
    """
    Synthetic binary classification dataset.
    Class 0: centered at -1  (for each feature)
    Class 1: centered at +1
    """
    half  = n // 2
    X0    = np.random.randn(half, d) - 1.0
    X1    = np.random.randn(n - half, d) + 1.0
    X     = np.vstack([X0, X1])
    y     = np.hstack([np.zeros(half), np.ones(n - half)])
    # shuffle
    idx   = np.random.permutation(n)
    return X[idx], y[idx]


def train_test_split_manual(X, y, test_ratio=0.2):
    """Simple train/test split — no sklearn."""
    n      = len(y)
    n_test = int(n * test_ratio)
    idx    = np.random.permutation(n)
    X_tr, y_tr = X[idx[n_test:]], y[idx[n_test:]]
    X_te, y_te = X[idx[:n_test]], y[idx[:n_test]]
    return X_tr, X_te, y_tr, y_te


def normalize(X_train, X_test):
    """
    Z-score normalization using training statistics only.
    Prevents data leakage from test set.
    """
    mu    = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, (X_test - mu) / sigma


#  4.  EXPERIMENTS

EPOCHS = 50

# ── Experiment 1: Linear Regression ─────────────────────────

print("=" * 60)
print("EXPERIMENT 1 — Linear Regression (from-scratch SGD)")
print("=" * 60)

X_lin, y_lin, w_true = make_linear_dataset(n=500, d=5)
X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split_manual(X_lin, y_lin)
X_tr_l, X_te_l = normalize(X_tr_l, X_te_l)

schedules = {
    "Constant  (η=0.01)":    ("constant",   0.01,  0.01),
    "Time-Decay (η₀=0.05)":  ("time_decay", 0.05,  0.005),
    "Step-Decay (η₀=0.05)":  ("step_decay", 0.05,  0.01),
}

lin_models = {}
for label, (sched, lr, decay) in schedules.items():
    m = SGDLinearRegression(
        learning_rate=lr, epochs=EPOCHS,
        lr_schedule=sched, decay=decay
    )
    m.fit(X_tr_l, y_tr_l)
    lin_models[label] = m
    preds    = m.predict(X_te_l)
    test_mse = np.mean((y_te_l - preds) ** 2)
    print(f"  {label:<30}  Test MSE: {test_mse:.4f}")

# ── Experiment 2: Logistic Regression ───────────────────────

print()
print("=" * 60)
print("EXPERIMENT 2 — Logistic Regression (from-scratch SGD)")
print("=" * 60)

X_cls, y_cls = make_binary_dataset(n=500, d=4)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split_manual(X_cls, y_cls)
X_tr_c, X_te_c = normalize(X_tr_c, X_te_c)

lr_values = [0.01, 0.1, 0.5]
log_models = {}
for lr in lr_values:
    label = f"η = {lr}"
    m = SGDLogisticRegression(learning_rate=lr, epochs=EPOCHS)
    m.fit(X_tr_c, y_tr_c)
    log_models[label] = m
    preds    = m.predict(X_te_c)
    accuracy = np.mean(preds == y_te_c)
    print(f"  {label:<12}  Test Accuracy: {accuracy * 100:.1f}%")


#  5.  FIGURE GENERATION


COLORS = ["#1f77b4", "#d62728", "#2ca02c"]

# ── Figure 1: Linear Regression Loss Curves ─────────────────

fig, ax = plt.subplots(figsize=(6, 4))
for (label, model), color in zip(lin_models.items(), COLORS):
    ax.plot(range(1, EPOCHS + 1), model.loss_history,
            label=label, color=color, linewidth=1.8)

ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("SGD Convergence — Linear Regression")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("figures/linear_loss.png")
plt.close(fig)
print("\n[✓] Saved figures/linear_loss.png")

# ── Figure 2: Logistic Regression Loss Curves ───────────────

fig, ax = plt.subplots(figsize=(6, 4))
for (label, model), color in zip(log_models.items(), COLORS):
    ax.plot(range(1, EPOCHS + 1), model.loss_history,
            label=label, color=color, linewidth=1.8)

ax.set_xlabel("Epoch")
ax.set_ylabel("Binary Cross-Entropy Loss")
ax.set_title("SGD Convergence — Logistic Regression")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("figures/logistic_loss.png")
plt.close(fig)
print("[✓] Saved figures/logistic_loss.png")

# ── Figure 3: Learning Rate Schedules Comparison ────────────

T_steps = np.arange(1, 201)
eta_constant   = np.full_like(T_steps, 0.05, dtype=float)
eta_time_decay = 0.05 / (1.0 + 0.005 * T_steps)
eta_step_decay = 0.05 * (0.5 ** (T_steps // 10))

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(T_steps, eta_constant,   label="Constant",    color=COLORS[0], lw=1.8)
ax.plot(T_steps, eta_time_decay, label="Time-Decay",  color=COLORS[1], lw=1.8)
ax.plot(T_steps, eta_step_decay, label="Step-Decay",  color=COLORS[2], lw=1.8)
ax.set_xlabel("Step $t$")
ax.set_ylabel(r"Learning rate $\eta_t$")
ax.set_title("Learning Rate Schedules")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("figures/lr_schedules.png")
plt.close(fig)
print("[✓] Saved figures/lr_schedules.png")

# ── Figure 4: SGD Trajectory on 2D Quadratic ────────────────

def f(w):      return w[0]**2 + 5*w[1]**2
def grad_f(w): return np.array([2*w[0], 10*w[1]])

def noisy_grad_f(w, noise_std=2.0):
    """Simulate single-sample stochastic gradient."""
    return grad_f(w) + noise_std * np.random.randn(2)

w0  = np.array([3.0, 2.0])
eta = 0.08
steps = 60

# SGD trajectory
w_sgd  = [w0.copy()]
w_curr = w0.copy()
for _ in range(steps):
    w_curr = w_curr - eta * noisy_grad_f(w_curr)
    w_sgd.append(w_curr.copy())
w_sgd = np.array(w_sgd)

# GD (full gradient) trajectory
w_gd   = [w0.copy()]
w_curr = w0.copy()
for _ in range(steps):
    w_curr = w_curr - eta * grad_f(w_curr)
    w_gd.append(w_curr.copy())
w_gd = np.array(w_gd)

# Contour plot
w1_grid = np.linspace(-3.5, 3.5, 300)
w2_grid = np.linspace(-2.5, 2.5, 300)
W1, W2  = np.meshgrid(w1_grid, w2_grid)
Z       = W1**2 + 5*W2**2

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, traj, label, color in zip(
        axes,
        [w_gd, w_sgd],
        ["Gradient Descent (GD)", "Stochastic GD (SGD)"],
        [COLORS[0], COLORS[1]]):

    ax.contour(W1, W2, Z, levels=20, cmap="Blues", alpha=0.6)
    ax.plot(traj[:, 0], traj[:, 1], "o-",
            color=color, markersize=3, linewidth=1.2,
            label=label, alpha=0.85)
    ax.plot(*w0, "k^", markersize=8, label="Start")
    ax.plot(0, 0, "r*", markersize=12, label="Optimum")
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    ax.set_title(label)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

fig.suptitle(r"Optimization Trajectory on $f(w_1,w_2) = w_1^2 + 5w_2^2$",
             fontsize=12)
fig.tight_layout()
fig.savefig("figures/trajectory.png")
plt.close(fig)
print("[✓] Saved figures/trajectory.png")

print()
print("=" * 60)
print("All figures saved to ./figures/")
print("All outputs in the report are reproducible via: python run.py")
print("=" * 60)
