#!/usr/bin/env python3
# Author: Jishnu
"""
Baseline Correction Module
==========================

This module provides functions for baseline correction of signals using Asymmetrically Reweighted Penalized Least Squares (ARPLS) smoothing.

Functions
---------

### arpls_baseline

Baseline correction using ARPLS smoothing.

### adaptive_arpls

Adaptive ARPLS baseline correction with noise estimation.

"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve



def arpls_baseline(y: np.ndarray, lam: float = 1e4, ratio: float = 0.05, itermax: int = 100) -> np.ndarray:
    """
    Baseline correction using Asymmetrically Reweighted Penalized Least Squares smoothing

    Parameters:
        y : array_like
            Input signal
        lam : float, optional
            Smoothness Higher values make the baseline smoother.
            Typically 1e2 < lam < 1e7
        ratio : float, optional
            Smaller values allow less negative deviations.
        itermax : int, optional
            Maximum number of iterations

    Returns:
        baseline : ndarray
            The estimated baseline
    """
    N = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(N, N - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(N)
    z = y.copy()  # Initialize z with the input signal
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        Z = W + D
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1.0 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z


def adaptive_arpls(y: np.ndarray, lam: float = 1e4, ratio: float = 0.05, itermax: int = 100) -> np.ndarray:
    """
    Adaptive ARPLS baseline correction with noise estimation

    Parameters:
        y : array_like
            Input signal
        lam : float, optional
            Initial smoothness parameter
        ratio : float, optional
            Asymmetry parameter
        itermax : int, optional
            Maximum number of iterations

    Returns:
        baseline : ndarray
            The estimated baseline
    """
    diff = np.diff(y)
    noise_level = np.median(np.abs(diff)) / 0.6745

    lam_adjusted = float(lam * (noise_level**2))

    baseline = arpls_baseline(y, lam=lam_adjusted, ratio=ratio, itermax=itermax)

    return baseline
