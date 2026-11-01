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

# Step 18 - mean_absolute_error
import numpy as np

def mean_absolute_error(y_true, y_pred):
    """Mean absolute error: (1/n) * sum(|y_true - y_pred|)."""
    return float(np.mean(np.abs(y_true - y_pred)))

# Step 19 - root_mean_squared_error
import numpy as np

def root_mean_squared_error(y_true, y_pred):
    """RMSE: sqrt((1/n) * sum((y_true - y_pred)^2))."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# Step 20 - r_squared
import numpy as np

def r_squared(y_true, y_pred):
    """R^2 = 1 - SS_res / SS_tot, with SS_tot taken about mean(y_true)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    """Bundle MAE, RMSE, and R^2 into one metrics dictionary."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }

# Step 22 - learning_curve_data
def learning_curve_data(train_losses, val_losses):
    """Return (epochs, train_losses, val_losses) as plain Python lists.

    Epoch indices start at 1.
    """
    train = [float(v) for v in train_losses]
    val = [float(v) for v in val_losses]
    epochs = list(range(1, len(train) + 1))
    return epochs, train, val

# Step 23 - weights_l2_distance
import numpy as np

def weights_l2_distance(w_gd, w_closed):
    """Euclidean distance ||w_gd - w_closed||_2."""
    return float(np.linalg.norm(w_gd - w_closed))

# Step 24 - create_lr_model
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    """Build the initial (unfitted) LinearRegressionGD-style model dict."""
    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "seed": seed,
        "weights": None,
        "normal_weights": None,
        "mean": None,
        "std": None,
        "train_losses": [],
        "val_losses": [],
    }

# Step 25 - fit_lr_model
import numpy as np

def fit_lr_model(model, X_train, y_train, X_val, y_val):
    """Fit the model dict in place using train-only statistics, GD, and the normal equation."""
    mean, std = compute_feature_stats(X_train)

    A_train = prepare_design_matrix(X_train, mean, std)
    A_val = prepare_design_matrix(X_val, mean, std)

    weights, train_losses, val_losses = train_batch_gd(
        A_train, y_train, A_val, y_val,
        model["learning_rate"], model["epochs"], model["patience"],
        seed=model["seed"],
    )

    try:
        normal_weights = normal_equation(A_train, y_train)
    except np.linalg.LinAlgError:
        # Rank-deficient design (collinear or constant features):
        # fall back to the minimum-norm least-squares solution.
        normal_weights = np.linalg.lstsq(A_train, y_train, rcond=None)[0]

    model["mean"] = mean
    model["std"] = std
    model["weights"] = weights
    model["normal_weights"] = normal_weights
    model["train_losses"] = train_losses
    model["val_losses"] = val_losses
    return model

# Step 26 - predict_lr_model
def predict_lr_model(model, X):
    """Predict targets for raw X using the fitted model's train stats and GD weights."""
    A = prepare_design_matrix(X, model["mean"], model["std"])
    return predict_linear(A, model["weights"])

# Step 27 - score_lr_model
def score_lr_model(model, X, y):
    """Score a fitted model on raw held-out features and targets."""
    y_pred = predict_lr_model(model, X)
    return evaluate_regression(y, y_pred)

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

