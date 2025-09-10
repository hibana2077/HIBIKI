"""
HIBIKI Toy Experiments

Implements the toy-experiment design from abs.md:
- Datasets: make_classification, make_moons, load_breast_cancer
- Baselines: Least-Squares Classifier (LSC), Logistic, LDA, Linear SVM (hinge), Perceptron
- Metrics: Accuracy, AUC, Brier; HIBIKI-Score; Subspace Angle to PCA subspace
- Visuals: HIBIKI-plot (alpha histogram by class + alpha vs kernel-PC1 scatter)
- Robustness: kernel-space noise invariance check

Outputs:
- Plots: HIBIKI/out/{dataset}_{model}_hibiki.png
- Results CSV: HIBIKI/out/results.csv
"""

from __future__ import annotations

import os
import math
import json
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import subspace_angles

from sklearn.datasets import make_classification, make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
)


# --------- Utilities ---------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into train/val/test with stratification.
    val_size is with respect to the original full set (not the train remainder).
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # compute val fraction relative to remaining temp set
    val_frac_of_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac_of_temp, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_w_b(model, kind: str) -> Tuple[np.ndarray, float]:
    if kind == "lsc":  # LinearRegression
        w = model.coef_.ravel()
        b = float(model.intercept_)
    elif kind in {"logreg", "lda", "linsvm", "perceptron"}:
        w = model.coef_.ravel()
        b = float(model.intercept_.ravel()[0])
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return w, b


def decision_scores_and_probs(model, kind: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (scores, probs) where scores are decision_function-like and probs are in [0,1].
    For models without predict_proba, apply a logistic transform to scores for pseudo-probabilities.
    """
    if kind in {"logreg", "lda"}:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
        else:
            # For some implementations, decision_function may be absent; use log-odds from proba
            proba = model.predict_proba(X)[:, 1]
            eps = 1e-15
            scores = np.log(np.clip(proba, eps, 1 - eps) / np.clip(1 - proba, eps, 1 - eps))
        probs = model.predict_proba(X)[:, 1]
    elif kind == "lsc":
        scores = model.predict(X)
        # clip to [0,1] as pseudo-probabilities
        probs = np.clip(scores, 0.0, 1.0)
    elif kind in {"linsvm", "perceptron"}:
        scores = model.decision_function(X)
        # logistic squashing to get pseudo-probabilities for Brier
        probs = 1.0 / (1.0 + np.exp(-scores))
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    # Ensure 1-D
    scores = np.asarray(scores).ravel()
    probs = np.asarray(probs).ravel()
    return scores, probs


def hibiki_score(alpha: np.ndarray, y: np.ndarray) -> float:
    """
    HS = Var(E[alpha | y]) / Var(alpha)
    For binary y in {0,1}.
    """
    alpha = np.asarray(alpha).ravel()
    y = np.asarray(y).ravel()
    var_alpha = np.var(alpha, ddof=1)
    if var_alpha <= 1e-15:
        return 0.0
    mu = np.mean(alpha)
    classes = np.unique(y)
    between = 0.0
    for c in classes:
        idx = (y == c)
        if np.any(idx):
            p = idx.mean()
            mu_c = alpha[idx].mean()
            between += p * (mu_c - mu) ** 2
    return float(between / var_alpha)


def subspace_angle_deg(w_hat: np.ndarray, basis: np.ndarray) -> float:
    """
    Principal angle (in degrees) between span(w_hat) and subspace spanned by columns of basis.
    basis shape: (n_features, k)
    """
    w_hat = np.asarray(w_hat).ravel()
    w_hat = w_hat / (np.linalg.norm(w_hat) + 1e-12)
    A = w_hat.reshape(-1, 1)
    B = np.asarray(basis)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    angles = subspace_angles(A, B)
    angle_rad = float(angles[0])
    return float(np.degrees(angle_rad))


def alpha_and_kernel_pc1(X: np.ndarray, w: np.ndarray, center: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute alpha = (x · w_hat), residual r = x - alpha w_hat, and the first PC direction of residuals.
    Returns (alpha, r, pc1_dir) where pc1_dir is in feature space (unit vector).
    """
    if center:
        # already standardized upstream; keep center=False to avoid shifting
        pass
    w = np.asarray(w).ravel()
    w_hat = w / (np.linalg.norm(w) + 1e-12)
    alpha = X @ w_hat
    R = X - np.outer(alpha, w_hat)
    # PCA on residuals to get first PC direction in feature space
    # Fit PCA to R and map back to feature-space direction via components_
    pca = PCA(n_components=1, random_state=0)
    pca.fit(R)
    pc1_dir = pca.components_[0]  # already unit-length
    return alpha, R, pc1_dir


def kernel_noise(X: np.ndarray, w: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add Gaussian noise only in the kernel subspace orthogonal to w.
    sigma is the standard deviation of isotropic noise before projection.
    """
    w_hat = w / (np.linalg.norm(w) + 1e-12)
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    proj = (noise @ w_hat)[:, None] * w_hat[None, :]
    noise_kernel = noise - proj
    return X + noise_kernel


# --------- Plotting ---------


def hibiki_plot(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    title: str,
    out_path: str,
) -> None:
    """
    Create a two-panel HIBIKI plot:
    - Left: histogram of alpha by class, with decision threshold line.
    - Right: scatter of alpha vs kernel PC1 coordinate, colored by class.
    """
    alpha, R, pc1_dir = alpha_and_kernel_pc1(X, w)
    # kernel PC1 coordinates per sample
    pc1_scores = R @ pc1_dir

    w_norm = np.linalg.norm(w) + 1e-12
    alpha_thr = -b / w_norm  # where w_norm*alpha + b = 0

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: histogram by class
    bins = 30
    classes = np.unique(y)
    palette = sns.color_palette("Set1", n_colors=len(classes))
    for i, c in enumerate(classes):
        sns.histplot(alpha[y == c], bins=bins, kde=False, stat="density", color=palette[i], alpha=0.4, ax=axes[0], label=f"y={c}")
    axes[0].axvline(alpha_thr, color="k", linestyle="--", linewidth=1.5, label="decision boundary")
    axes[0].set_xlabel(r"$\alpha = \hat{w} \cdot x$")
    axes[0].set_ylabel("density")
    axes[0].set_title("Alpha histogram by class")
    axes[0].legend()

    # Right: scatter alpha vs kernel PC1 score
    for i, c in enumerate(classes):
        idx = (y == c)
        axes[1].scatter(alpha[idx], pc1_scores[idx], s=12, alpha=0.7, color=palette[i], label=f"y={c}")
    axes[1].axvline(alpha_thr, color="k", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("kernel PC1 score")
    axes[1].set_title("Alpha vs kernel-PC1")
    axes[1].legend()

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------- Datasets ---------


def load_dataset(name: str, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, str]:
    name = name.lower()
    if name == "classification":
        X, y = make_classification(
            n_samples=2000,
            n_features=10,
            n_informative=1,
            n_redundant=7,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            flip_y=0.01,
            class_sep=1.0,
            random_state=random_state,
        )
        return X.astype(float), y.astype(int), "make_classification"
    elif name == "moons":
        X, y = make_moons(n_samples=2000, noise=0.3, random_state=random_state)
        return X.astype(float), y.astype(int), "make_moons"
    elif name == "breast_cancer":
        data = load_breast_cancer()
        return data.data.astype(float), data.target.astype(int), "breast_cancer"
    else:
        raise ValueError(f"Unknown dataset {name}")


# --------- Main experiment pipeline ---------


@dataclass
class ModelSpec:
    name: str
    kind: str
    model: object


def make_baselines(random_state: int = 42) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    # Least Squares Classifier
    specs.append(ModelSpec(name="LSC", kind="lsc", model=LinearRegression()))
    # Logistic Regression
    specs.append(
        ModelSpec(
            name="Logistic",
            kind="logreg",
            model=LogisticRegression(solver="lbfgs", max_iter=1000, random_state=random_state),
        )
    )
    # LDA
    specs.append(ModelSpec(name="LDA", kind="lda", model=LinearDiscriminantAnalysis()))
    # Linear SVM (hinge)
    specs.append(ModelSpec(name="LinearSVM", kind="linsvm", model=LinearSVC(random_state=random_state, max_iter=5000)))
    # Perceptron
    specs.append(ModelSpec(name="Perceptron", kind="perceptron", model=Perceptron(random_state=random_state, max_iter=1000)))
    return specs


def evaluate_all(random_state: int = 42) -> pd.DataFrame:
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    ensure_dir(out_dir)

    results: List[Dict] = []
    datasets = ["classification", "moons", "breast_cancer"]
    rng = np.random.default_rng(random_state)

    for ds in datasets:
        X, y, ds_name = load_dataset(ds, random_state=random_state)

        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, test_size=0.2, val_size=0.2, random_state=random_state
        )

        # Standardize (fit on train only)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        # PCA on train for subspace angle computation
        pca_k = 2
        pca = PCA(n_components=pca_k, random_state=random_state)
        pca.fit(X_train_s)
        pca_basis = pca.components_.T  # (n_features, k)

        for spec in make_baselines(random_state=random_state):
            # Fit model
            model = spec.model
            # Special-case LSC target as float
            y_train_fit = y_train.astype(float) if spec.kind == "lsc" else y_train
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_s, y_train_fit)

            # Extract w, b
            w, b = get_w_b(model, spec.kind)
            w_hat = w / (np.linalg.norm(w) + 1e-12)

            # Scores and probs on test
            scores_test, probs_test = decision_scores_and_probs(model, spec.kind, X_test_s)
            y_pred_test = (probs_test >= 0.5).astype(int)

            # Metrics
            acc = accuracy_score(y_test, y_pred_test)
            try:
                auc = roc_auc_score(y_test, scores_test)
            except Exception:
                auc = np.nan
            try:
                brier = brier_score_loss(y_test, np.clip(probs_test, 0.0, 1.0))
            except Exception:
                brier = np.nan

            # HIBIKI quantities on test set
            alpha_test = X_test_s @ w_hat
            hs = hibiki_score(alpha_test, y_test)
            angle_deg = subspace_angle_deg(w_hat, pca_basis)

            # HIBIKI plot (on validation set for visualization) to avoid peeking into test too much
            title = f"{ds_name} | {spec.name} | HS={hs:.3f} | angle_to_PCA{pca_k}={angle_deg:.1f}°"
            plot_path = os.path.join(out_dir, f"{ds_name}_{spec.name}_hibiki.png")
            hibiki_plot(X_val_s, y_val, w, b, title, plot_path)

            # Kernel-noise robustness on test set
            # noise scale based on residual std dev per feature from train set
            alpha_tr, R_tr, _ = alpha_and_kernel_pc1(X_train_s, w)
            # Use a scalar sigma as median feature std of residuals
            sigma = float(np.median(R_tr.std(axis=0)))
            X_test_noised = kernel_noise(X_test_s, w, sigma=sigma, rng=rng)
            scores_noised, probs_noised = decision_scores_and_probs(model, spec.kind, X_test_noised)
            y_pred_noised = (probs_noised >= 0.5).astype(int)
            acc_noised = accuracy_score(y_test, y_pred_noised)
            try:
                auc_noised = roc_auc_score(y_test, scores_noised)
            except Exception:
                auc_noised = np.nan
            try:
                brier_noised = brier_score_loss(y_test, np.clip(probs_noised, 0.0, 1.0))
            except Exception:
                brier_noised = np.nan

            results.append(
                dict(
                    dataset=ds_name,
                    model=spec.name,
                    acc=acc,
                    auc=auc,
                    brier=brier,
                    hibiki_score=hs,
                    angle_to_pca_k_deg=angle_deg,
                    acc_kernel_noise=acc_noised,
                    auc_kernel_noise=auc_noised,
                    brier_kernel_noise=brier_noised,
                )
            )

    df = pd.DataFrame(results)
    csv_path = os.path.join(os.path.dirname(__file__), "out", "results.csv")
    df.to_csv(csv_path, index=False)
    return df


def main():
    df = evaluate_all(random_state=42)
    # Also dump a brief JSON summary with mean metrics per model
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    summary = (
        df.groupby("model")[["acc", "auc", "brier", "hibiki_score", "angle_to_pca_k_deg"]]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Results written to:")
    print(os.path.join(out_dir, "results.csv"))
    print(os.path.join(out_dir, "summary.json"))
    print(os.path.join(out_dir, "<dataset>_<model>_hibiki.png"))


if __name__ == "__main__":
    main()
