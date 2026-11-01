"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
import numpy as np

def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Step 3 - compute_feature_stats
import numpy as np


def compute_feature_stats(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)   # population std (divide by n)
    std = np.where(std == 0, 1.0, std)
    return mean, std

# Step 4 - standardize_features
import numpy as np


def standardize_features(X, mean, std):
    X = np.asarray(X, dtype=float)
    return (X - mean) / std

# Step 5 - add_bias_column
import numpy as np


def add_bias_column(X):
    X = np.asarray(X, dtype=float)
    return np.hstack([np.ones((X.shape[0], 1)), X])

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    """Standardize features, then prepend a bias column of ones.

    Returns
    -------
    np.ndarray, shape (n, d+1)
    """
    X_std = standardize_features(X, mean, std)
    return add_bias_column(X_std)

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights."""
    return X @ weights

# Step 8 - mse_loss
import numpy as np

def mse_loss(y_true, y_pred):
    """Mean squared error: (1/n) * sum((y_true - y_pred)^2)."""
    residuals = y_true - y_pred
    return float(np.mean(residuals ** 2))

# Step 9 - mse_gradient
import numpy as np

def mse_gradient(X, y_true, y_pred):
    """Gradient of MSE w.r.t. weights: (2/n) * X^T (y_pred - y_true)."""
    n = X.shape[0]
    return (2.0 / n) * (X.T @ (y_pred - y_true))

# Step 10 - normal_equation
import numpy as np

def normal_equation(X, y):
    """Closed-form least squares: solve (X^T X) w = X^T y."""
    return np.linalg.solve(X.T @ X, X.T @ y)

# Step 11 - initialize_weights
import numpy as np

def initialize_weights(n_features, seed=None):
    """Return (n_features,) weights sampled i.i.d. from N(0, 0.01^2)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n_features) * 0.01

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights."""
    y_pred = predict_linear(X, weights)
    grad = mse_gradient(X, y, y_pred)
    return weights - lr * grad

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights."""
    train_loss = mse_loss(y_train, predict_linear(X_train, weights))
    val_loss = mse_loss(y_val, predict_linear(X_val, weights))
    return train_loss, val_loss

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    """Update early-stopping state after one validation evaluation.

    Returns
    -------
    (best_val_loss, wait, best_weights, should_stop)
    """
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = weights.copy()
        wait = 0
    else:
        wait += 1
    should_stop = wait >= patience
    return best_val_loss, wait, best_weights, should_stop

# Step 15 - init_training_state
import numpy as np

def init_training_state(n_features, seed=None):
    """Build the initial training-state dict for the GD epoch loop."""
    weights = initialize_weights(n_features, seed=seed)
    return {
        "weights": weights,
        "best_weights": weights.copy(),
        "best_val_loss": np.inf,
        "wait": 0,
        "train_losses": [],
        "val_losses": [],
        "stopped": False,
    }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state."""
    state["weights"] = gd_step(X_train, y_train, state["weights"], lr)

    train_loss, val_loss = epoch_train_val_losses(
        X_train, y_train, X_val, y_val, state["weights"]
    )
    state["train_losses"].append(train_loss)
    state["val_losses"].append(val_loss)

    (state["best_val_loss"],
     state["wait"],
     state["best_weights"],
     state["stopped"]) = update_early_stop_state(
        val_loss,
        state["best_val_loss"],
        state["wait"],
        state["weights"],
        state["best_weights"],
        patience,
    )
    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    """Full-batch GD with early stopping.

    Returns
    -------
    (best_weights, train_losses, val_losses)
    """
    state = init_training_state(X_train.shape[1], seed=seed)

    for _ in range(epochs):
        state = run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)
        if state["stopped"]:
            break

    return state["best_weights"], state["train_losses"], state["val_losses"]

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

