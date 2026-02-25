"""
regimes/hmm.py
==============
Gaussian Hidden Markov Model implementation.

Implements the full Baum-Welch Expectation-Maximisation algorithm for
Gaussian-emission HMMs — from scratch using numpy/scipy.

The HMM models:
  * K latent states (regimes)
  * Gaussian emissions: P(x_t | z_t = k) = N(x_t; μ_k, Σ_k)
  * Markov transitions: P(z_t | z_{t-1} = j) = A[j, k]

Outputs per date:
  * `state`       : int — argmax of the posterior
  * `p_0..p_{K-1}`: float — posterior state probabilities γ(z_t = k)

References:
  Rabiner (1989), "A Tutorial on Hidden Markov Models"
  Murphy (2012), "Machine Learning: A Probabilistic Perspective", Ch. 17
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Numerical constants ───────────────────────────────────────────────────────
LOG_EPS   = -1e300   # log(0) substitute
MIN_COVAR = 1e-6     # regularisation floor for diagonal covariance


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class HMMResult:
    """Regime inference outputs for a date range."""
    dates           : pd.DatetimeIndex
    state           : np.ndarray          # shape (T,) int — argmax posterior
    posteriors      : np.ndarray          # shape (T, K) float — γ(z_t=k)
    log_likelihood  : float
    n_iter          : int
    converged       : bool
    transition_matrix: np.ndarray         # shape (K, K)
    state_means     : np.ndarray          # shape (K, D)
    state_covars    : np.ndarray          # shape (K, D) — diagonal variances
    initial_probs   : np.ndarray          # shape (K,)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert posteriors + argmax state to a labelled DataFrame."""
        K   = self.posteriors.shape[1]
        cols = {f"p_{k}": self.posteriors[:, k] for k in range(K)}
        cols["state"] = self.state
        return pd.DataFrame(cols, index=self.dates)

    def regime_summary(self, feature_names: list[str]) -> pd.DataFrame:
        """
        Per-state mean table for diagnostics.

        Returns
        -------
        pd.DataFrame : shape (K, D), rows = states, columns = features
        """
        return pd.DataFrame(
            self.state_means,
            columns=feature_names,
            index=[f"State_{k}" for k in range(len(self.state_means))],
        )


# ── Gaussian emission helpers ─────────────────────────────────────────────────

def _log_gaussian(X: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """
    Compute log-likelihoods under diagonal Gaussian.

    Parameters
    ----------
    X      : (T, D) observations
    mu     : (K, D) state means
    sigma2 : (K, D) state diagonal variances

    Returns
    -------
    log_B  : (T, K) log emission probabilities
    """
    T, D = X.shape
    K    = mu.shape[0]
    log_B = np.zeros((T, K))

    for k in range(K):
        diff  = X - mu[k]                  # (T, D)
        s2    = np.maximum(sigma2[k], MIN_COVAR)
        log_det = np.sum(np.log(s2))
        maha  = np.sum(diff ** 2 / s2, axis=1)   # (T,)
        log_B[:, k] = -0.5 * (D * np.log(2 * np.pi) + log_det + maha)

    return log_B


# ── Log-sum-exp (numerically stable) ─────────────────────────────────────────

def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out   = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if axis is not None:
        out = out.squeeze(axis=axis)
    return out


# ── Core HMM (Baum-Welch EM) ─────────────────────────────────────────────────

class GaussianHMM:
    """
    Diagonal-covariance Gaussian Hidden Markov Model.

    Parameters
    ----------
    n_states       : Number of hidden states (regimes).
    n_iter         : Maximum EM iterations.
    tol            : Log-likelihood convergence tolerance.
    covariance_type: 'diag' (only option currently; full in future).
    random_state   : NumPy random seed.
    reg_covar      : Covariance regularisation (added to diagonal).
    """

    def __init__(
        self,
        n_states        : int   = 4,
        n_iter          : int   = 200,
        tol             : float = 1e-4,
        covariance_type : str   = "diag",
        random_state    : int   = 42,
        reg_covar       : float = 1e-4,
    ) -> None:
        self.n_states        = n_states
        self.n_iter          = n_iter
        self.tol             = tol
        self.covariance_type = covariance_type
        self.random_state    = random_state
        self.reg_covar       = reg_covar

        # Model parameters (set during fit)
        self._pi      : np.ndarray | None = None   # (K,) initial
        self._A       : np.ndarray | None = None   # (K, K) transition
        self._mu      : np.ndarray | None = None   # (K, D) means
        self._sigma2  : np.ndarray | None = None   # (K, D) variances
        self._fitted  : bool = False

    # ── Parameter initialisation ──────────────────────────────────────────

    def _init_params(self, X: np.ndarray) -> None:
        """Initialise parameters using k-means-like random assignment."""
        T, D = X.shape
        K    = self.n_states
        rng  = np.random.default_rng(self.random_state)

        # Initial state distribution: uniform
        self._pi = np.ones(K) / K

        # Transition matrix: slightly sticky diagonal
        self._A = np.ones((K, K)) * 0.05
        np.fill_diagonal(self._A, 0.85)
        self._A /= self._A.sum(axis=1, keepdims=True)

        # Emission parameters: initialise from random subset
        idx = rng.choice(T, size=K, replace=False)
        self._mu = X[idx].copy()

        # Variance: global variance as starting point
        self._sigma2 = np.tile(X.var(axis=0), (K, 1)) + self.reg_covar

    # ── Forward algorithm (log-space) ─────────────────────────────────────

    def _forward(self, log_B: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Forward algorithm in log-space.

        Parameters
        ----------
        log_B : (T, K) log emission probabilities.

        Returns
        -------
        log_alpha : (T, K) log forward variables
        log_likelihood : scalar
        """
        T, K = log_B.shape
        log_A  = np.log(self._A + 1e-300)
        log_pi = np.log(self._pi + 1e-300)

        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_B[0]

        for t in range(1, T):
            # log_alpha[t, k] = log Σ_j exp(log_alpha[t-1, j] + log_A[j, k]) + log_B[t, k]
            log_alpha[t] = (
                _logsumexp(log_alpha[t - 1, :, np.newaxis] + log_A, axis=0)
                + log_B[t]
            )

        log_likelihood = float(_logsumexp(log_alpha[-1], axis=0).squeeze())
        return log_alpha, log_likelihood

    # ── Backward algorithm (log-space) ───────────────────────────────────

    def _backward(self, log_B: np.ndarray) -> np.ndarray:
        """
        Backward algorithm in log-space.

        Returns
        -------
        log_beta : (T, K) log backward variables
        """
        T, K  = log_B.shape
        log_A = np.log(self._A + 1e-300)

        log_beta = np.zeros((T, K))
        # log_beta[T-1] = 0 (= log(1))

        for t in range(T - 2, -1, -1):
            log_beta[t] = _logsumexp(
                log_A + log_B[t + 1] + log_beta[t + 1], axis=1
            )

        return log_beta

    # ── E-step: compute posteriors ─────────────────────────────────────────

    def _e_step(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        E-step: compute γ(z_t = k) and ξ(z_{t-1}, z_t) from forward-backward.

        Returns
        -------
        gamma  : (T, K) posterior marginals
        xi_sum : (K, K) expected transition counts (summed over t)
        log_lik: scalar log-likelihood
        """
        log_B     = _log_gaussian(X, self._mu, self._sigma2)
        log_alpha, log_lik = self._forward(log_B)
        log_beta  = self._backward(log_B)

        T, K = X.shape[0], self.n_states
        log_A = np.log(self._A + 1e-300)

        # γ(z_t = k) = α_t(k) β_t(k) / Σ_k α_t(k) β_t(k)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1, )[:, np.newaxis]
        gamma = np.exp(log_gamma)

        # ξ(z_{t-1}=j, z_t=k) — summed over t=1..T-1
        xi_sum = np.zeros((K, K))
        for t in range(T - 1):
            log_xi_t = (
                log_alpha[t, :, np.newaxis]
                + log_A
                + log_B[t + 1, np.newaxis, :]
                + log_beta[t + 1, np.newaxis, :]
            )
            log_xi_t -= _logsumexp(log_xi_t.ravel(), axis=0).squeeze()
            xi_sum += np.exp(log_xi_t)

        return gamma, xi_sum, log_lik

    # ── M-step: update parameters ─────────────────────────────────────────

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi_sum: np.ndarray) -> None:
        """
        M-step: update π, A, μ, σ² from posteriors.

        Updates are the standard closed-form Baum-Welch maximisers.
        """
        T, D = X.shape
        K    = self.n_states

        # Initial state probabilities
        self._pi = gamma[0] / gamma[0].sum()

        # Transition matrix
        alpha = 1e-2  # small smoothing parameter
        row_sums = xi_sum.sum(axis=1, keepdims=True)
        self._A = (xi_sum + alpha) / (row_sums + alpha * K)
        #self._A = np.where(np.isfinite(self._A), self._A, 1.0 / K)

        # Emission means
        denom       = gamma.sum(axis=0)           # (K,)
        self._mu    = (gamma.T @ X) / denom[:, np.newaxis]   # (K, D)

        # Emission diagonal variances
        self._sigma2 = np.zeros((K, D))
        for k in range(K):
            diff          = X - self._mu[k]        # (T, D)
            w             = gamma[:, k]             # (T,)
            self._sigma2[k] = (w[:, np.newaxis] * diff ** 2).sum(axis=0) / denom[k]
        self._sigma2 = np.maximum(self._sigma2 + self.reg_covar, MIN_COVAR)

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray | pd.DataFrame) -> "GaussianHMM":
        """
        Fit the HMM via Baum-Welch EM.

        Parameters
        ----------
        X : (T, D) observation matrix (already z-scored).

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)

        X = np.asarray(X, dtype=float)
        T, D = X.shape
        assert T > self.n_states, f"Need T > K ({T} > {self.n_states})"

        self._init_params(X)

        prev_ll = -np.inf
        converged = False

        for i in range(self.n_iter):
            gamma, xi_sum, ll = self._e_step(X)
            self._m_step(X, gamma, xi_sum)

            delta = ll - prev_ll
            logger.debug("HMM EM iter %d | LL=%.4f Δ=%.6f", i, ll, delta)

            if i > 0 and abs(delta) < self.tol:
                logger.info("HMM converged after %d iterations (LL=%.4f)", i + 1, ll)
                converged = True
                break
            prev_ll = ll

        if not converged:
            logger.warning("HMM did not converge in %d iterations", self.n_iter)

        self._fitted = True
        self._last_ll = float(prev_ll)
        self._converged = converged
        self._n_iter_done = i + 1
        return self

    # ── Predict (Viterbi or posterior) ───────────────────────────────────

    def predict_proba(
        self, X: np.ndarray | pd.DataFrame
    ) -> np.ndarray:
        """
        Return posterior state probabilities γ(z_t = k) for each t.

        Parameters
        ----------
        X : (T, D) observation matrix (must be z-scored with SAME scaler as fit).

        Returns
        -------
        np.ndarray : (T, K) posterior probabilities.
        """
        assert self._fitted, "Must call fit() first."
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)
        gamma, _, _ = self._e_step(X)
        return gamma

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return argmax state (most likely regime) for each t."""
        return self.predict_proba(X).argmax(axis=1)

    def viterbi(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Viterbi algorithm: globally most probable state sequence.

        Returns
        -------
        np.ndarray : (T,) integer state sequence.
        """
        assert self._fitted, "Must call fit() first."
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)

        T, _  = X.shape
        K     = self.n_states
        log_B = _log_gaussian(X, self._mu, self._sigma2)
        log_A = np.log(self._A + 1e-300)

        delta    = np.full((T, K), LOG_EPS)
        psi      = np.zeros((T, K), dtype=int)
        delta[0] = np.log(self._pi + 1e-300) + log_B[0]

        for t in range(1, T):
            for k in range(K):
                scores     = delta[t - 1] + log_A[:, k]
                psi[t, k]  = int(np.argmax(scores))
                delta[t, k] = scores[psi[t, k]] + log_B[t, k]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def infer(
        self,
        X      : pd.DataFrame,
        method : str = "posterior",
    ) -> HMMResult:
        """
        Run full inference and return HMMResult.

        Parameters
        ----------
        X      : Feature matrix (DataFrame with DatetimeIndex).
        method : 'posterior' (default) or 'viterbi' for state sequence.
        """
        assert self._fitted, "Must call fit() first."
        Xarr = X.values.astype(float)

        posteriors = self.predict_proba(Xarr)
        if method == "viterbi":
            state = self.viterbi(Xarr)
        else:
            state = posteriors.argmax(axis=1)

        gamma_unused, _, ll = self._e_step(Xarr)

        return HMMResult(
            dates            = pd.DatetimeIndex(X.index),
            state            = state,
            posteriors       = posteriors,
            log_likelihood   = ll,
            n_iter           = self._n_iter_done,
            converged        = self._converged,
            transition_matrix= self._A.copy(),
            state_means      = self._mu.copy(),
            state_covars     = self._sigma2.copy(),
            initial_probs    = self._pi.copy(),
        )

    # ── Multi-restart fit (avoid local optima) ───────────────────────────

    def fit_best_of(
        self,
        X        : np.ndarray | pd.DataFrame,
        n_restarts: int = 5,
    ) -> "GaussianHMM":
        """
        Fit with multiple random restarts; keep the best LL solution.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(float)

        best_ll  = -np.inf
        best_pi  = None; best_A = None; best_mu = None; best_s2 = None

        for restart in range(n_restarts):
            self.random_state = 42 + restart
            try:
                self.fit(X)
                if not np.isfinite(self._last_ll):
                    logger.warning("Restart %d produced non-finite LL (%s); skipping.", restart, self._last_ll)
                    continue
                if self._last_ll > best_ll:
                    best_ll = self._last_ll
                    best_pi = self._pi.copy()
                    best_A  = self._A.copy()
                    best_mu = self._mu.copy()
                    best_s2 = self._sigma2.copy()
                    logger.debug("New best LL=%.4f at restart %d", best_ll, restart)
            except Exception as e:
                logger.warning("Restart %d failed: %s", restart, e)

        assert best_mu is not None, "All restarts failed."
        self._pi     = best_pi
        self._A      = best_A
        self._mu     = best_mu
        self._sigma2 = best_s2
        self._last_ll = best_ll
        self._fitted  = True
        logger.info("HMM best-of-%d | final LL=%.4f", n_restarts, best_ll)
        return self

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        assert self._fitted
        return {
            "n_states"  : self.n_states,
            "pi"        : self._pi.tolist(),
            "A"         : self._A.tolist(),
            "mu"        : self._mu.tolist(),
            "sigma2"    : self._sigma2.tolist(),
            "last_ll"   : self._last_ll,
            "converged" : self._converged,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("HMM saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "GaussianHMM":
        with open(path) as f:
            d = json.load(f)
        model = cls(n_states=d["n_states"])
        model._pi      = np.array(d["pi"])
        model._A       = np.array(d["A"])
        model._mu      = np.array(d["mu"])
        model._sigma2  = np.array(d["sigma2"])
        model._last_ll = d["last_ll"]
        model._converged = d["converged"]
        model._n_iter_done = 0
        model._fitted  = True
        return model
