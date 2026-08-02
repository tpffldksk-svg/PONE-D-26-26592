"""
Firth Penalized Logistic Regression
====================================
Custom implementation used for all Firth-penalized logistic regression
analyses in:

"Predicting Red Breast Syndrome After ADM-Assisted Breast Reconstruction:
An Exploratory Scoring Model Using Routine Hematologic Markers"

This module implements:
  1. Firth penalized logistic regression (IRLS with Jeffreys-prior penalty)
  2. Penalized log-likelihood ratio test (LRT) p-values
  3. Profile-likelihood 95% confidence intervals

Dependencies: numpy, scipy

Reference:
  Firth D. Bias reduction of maximum likelihood estimates. Biometrika. 1993;80(1):27-38.
  Heinze G, Schemper M. A solution to the problem of separation in logistic
  regression. Stat Med. 2002;21(16):2409-19.
"""

import numpy as np
from scipy.stats import chi2
from scipy.optimize import brentq


def firth_fit(X, y, max_iter=500, tol=1e-10):
    """
    Fit a Firth penalized logistic regression model via iteratively
    reweighted least squares (IRLS) with a Jeffreys-prior penalty term.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix (include a column of ones for the intercept).
    y : ndarray, shape (n_samples,)
        Binary outcome vector (0/1).
    max_iter : int
        Maximum number of IRLS iterations.
    tol : float
        Convergence tolerance on the maximum absolute coefficient update.

    Returns
    -------
    beta : ndarray, shape (n_features,)
        Estimated (penalized) regression coefficients.
    Finv : ndarray, shape (n_features, n_features)
        Inverse Fisher information matrix at convergence (for Wald SEs).
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = mu * (1 - mu)
        XtWX = X.T @ (X * W[:, None])
        try:
            Finv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            Finv = np.linalg.pinv(XtWX)

        # Hat matrix diagonal (leverage), needed for the Firth correction term
        Wsq = np.sqrt(W)
        H = Wsq[:, None] * X @ Finv @ X.T * Wsq[None, :]
        h = np.diag(H)

        # Firth-modified score equation: U*(beta) = X'(y - mu + h*(0.5 - mu))
        Ustar = X.T @ (y - mu + h * (0.5 - mu))
        delta = Finv @ Ustar
        beta = beta + delta

        if np.max(np.abs(delta)) < tol:
            break
    return beta, Finv


def penalized_log_likelihood(beta, X, y):
    """
    Penalized log-likelihood used by Firth's method:
        l*(beta) = l(beta) + 0.5 * log|I(beta)|
    where I(beta) is the Fisher information matrix.
    """
    eta = np.clip(X @ beta, -30, 30)
    mu = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    W = mu * (1 - mu)
    F = X.T @ (X * W[:, None])
    _, logdet = np.linalg.slogdet(F)
    ll = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
    return ll + 0.5 * logdet


def lrt_pvalue(X_full, y, X_reduced):
    """
    Penalized likelihood ratio test comparing a full model (X_full) to a
    reduced model (X_reduced, typically with one covariate omitted).

    Returns the p-value from a chi-squared(1) reference distribution.
    """
    beta_full, _ = firth_fit(X_full, y)
    beta_reduced, _ = firth_fit(X_reduced, y)
    ll_full = penalized_log_likelihood(beta_full, X_full, y)
    ll_reduced = penalized_log_likelihood(beta_reduced, X_reduced, y)
    stat = max(2 * (ll_full - ll_reduced), 0)
    return chi2.sf(stat, df=1)


def profile_likelihood_ci(X, y, beta_hat, idx, level=0.95):
    """
    Compute the profile-likelihood confidence interval for coefficient
    beta[idx] by inverting the penalized likelihood-ratio statistic.

    For each candidate value b of beta[idx], the remaining coefficients are
    re-optimized (profiled out) via an offset model; the CI bounds are the
    values of b for which twice the drop in penalized log-likelihood equals
    the chi-squared(1, level) quantile.

    Parameters
    ----------
    X : ndarray
        Full design matrix (including intercept).
    y : ndarray
        Binary outcome vector.
    beta_hat : ndarray
        Firth-fitted coefficient vector for the full model.
    idx : int
        Column index of the coefficient of interest.
    level : float
        Confidence level (default 0.95).

    Returns
    -------
    (lower, upper) : tuple of float
        Profile-likelihood confidence bounds on the coefficient scale.
        Exponentiate to obtain the OR-scale interval.
    """
    ll_hat = penalized_log_likelihood(beta_hat, X, y)
    cutoff = chi2.ppf(level, df=1) / 2

    mask = [j for j in range(X.shape[1]) if j != idx]
    X_rest = X[:, mask]

    def firth_offset(X_r, y, offset, max_iter=500, tol=1e-10):
        """Firth fit for the remaining covariates with a fixed linear offset."""
        n, p = X_r.shape
        beta_r = np.zeros(p)
        for _ in range(max_iter):
            eta = X_r @ beta_r + offset
            mu = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
            W = mu * (1 - mu)
            XtWX = X_r.T @ (X_r * W[:, None])
            try:
                Finv = np.linalg.inv(XtWX)
            except np.linalg.LinAlgError:
                Finv = np.linalg.pinv(XtWX)
            Wsq = np.sqrt(W)
            H = Wsq[:, None] * X_r @ Finv @ X_r.T * Wsq[None, :]
            h = np.diag(H)
            Ustar = X_r.T @ (y - mu + h * (0.5 - mu))
            delta = Finv @ Ustar
            beta_r = beta_r + delta
            if np.max(np.abs(delta)) < tol:
                break
        return beta_r

    def profile_diff(b_fixed):
        offset = X[:, idx] * b_fixed
        beta_r = firth_offset(X_rest, y, offset)
        beta_full = np.zeros(X.shape[1])
        beta_full[idx] = b_fixed
        beta_full[mask] = beta_r
        ll_profile = penalized_log_likelihood(beta_full, X, y)
        return ll_hat - ll_profile - cutoff

    # Search bounds seeded from the Wald standard error
    eta_hat = X @ beta_hat
    mu_hat = 1 / (1 + np.exp(-eta_hat))
    W_hat = mu_hat * (1 - mu_hat)
    se = np.sqrt(abs(np.linalg.pinv(X.T @ (X * W_hat[:, None]))[idx, idx]))
    b0 = beta_hat[idx]

    try:
        lower = brentq(profile_diff, b0 - 10 * se, b0, xtol=1e-6, maxiter=200)
    except ValueError:
        lower = b0 - 3 * se
    try:
        upper = brentq(profile_diff, b0, b0 + 10 * se, xtol=1e-6, maxiter=200)
    except ValueError:
        upper = b0 + 3 * se

    return lower, upper


def fit_and_summarize(X, y, var_names, level=0.95):
    """
    Convenience wrapper: fits the full model and returns a summary table
    of OR, profile-likelihood 95% CI (on the OR scale), and the penalized
    LRT p-value for each covariate (excluding the intercept, column 0).

    Returns a list of dicts: {name, OR, CI_lower, CI_upper, p_value}
    """
    beta_hat, _ = firth_fit(X, y)
    results = []
    for i, name in enumerate(var_names):
        if i == 0:
            continue  # skip intercept
        OR = np.exp(beta_hat[i])
        lo, hi = profile_likelihood_ci(X, y, beta_hat, i, level=level)

        mask = [j for j in range(X.shape[1]) if j != i]
        X_reduced = X[:, mask]
        p_value = lrt_pvalue(X, y, X_reduced)

        results.append({
            "name": name,
            "OR": OR,
            "CI_lower": np.exp(lo),
            "CI_upper": np.exp(hi),
            "p_value": p_value,
        })
    return results


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Example usage (illustrative; replace with study data)
    # -----------------------------------------------------------------
    rng = np.random.default_rng(42)
    n = 47
    wbc = rng.normal(7.5, 4.5, n)
    plt_ = rng.normal(190, 60, n)
    hb = rng.normal(10.9, 1.1, n)
    ki67_high = rng.integers(0, 2, n).astype(float)

    # Simulate an outcome correlated with the predictors for demonstration
    linpred = 0.2 * (wbc - wbc.mean()) + 0.01 * (plt_ - plt_.mean()) \
              - 0.4 * (hb - hb.mean()) + 1.0 * ki67_high - 1.5
    prob = 1 / (1 + np.exp(-linpred))
    y = rng.binomial(1, prob)

    X = np.column_stack([np.ones(n), wbc, plt_, hb, ki67_high])
    var_names = ["Intercept", "WBC", "PLT", "Hb", "Ki67_High"]

    summary = fit_and_summarize(X, y, var_names)
    print(f"{'Variable':<12}{'OR':>8}{'95% CI':>20}{'p-value':>10}")
    for r in summary:
        ci = f"{r['CI_lower']:.3f}-{r['CI_upper']:.3f}"
        print(f"{r['name']:<12}{r['OR']:>8.3f}{ci:>20}{r['p_value']:>10.4f}")
