"""
Pruning and noise-calibration helpers for DP-BART-PR+.
Adapted from reference implementation for inference use.
"""

from math import sqrt

import mpmath
import torch
from mpmath import mp


def calibrateAnalyticGaussianMechanism_precision(epsilon, delta, GS, tol=1.e-12):
    """
    Calibrates analytic Gaussian mechanism (Balle & Wang, 2018).
    Returns noise standard deviation for (epsilon, delta)-DP.
    """
    if epsilon <= 1000:
        mp.dps = 500  # works for epsilon = 1000
    elif epsilon <= 2500 and epsilon > 1000:
        mp.dps = 1100  # works for epsilon = 2500
    else:
        mp.dps = 2200  # works for epsilon = 5000

    def Phi(t):
        return 0.5*(1.0 + mpmath.erf(t/mpmath.sqrt(2.0)))

    def caseA(epsilon, s):
        return Phi(mpmath.sqrt(epsilon*s)) - mpmath.exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon, s):
        return Phi(-mpmath.sqrt(epsilon*s)) - mpmath.exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while (not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while (not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s: caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s: caseA(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s: mpmath.sqrt(1.0 + s/2.0) - mpmath.sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s: caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s: caseB(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s: mpmath.sqrt(1.0 + s/2.0) + mpmath.sqrt(s/2.0)

        predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    sigma = alpha*GS/mpmath.sqrt(2.0*epsilon)

    return float(sigma)


def load_neurons_for_pruning(in_path='.'):
    """Load the saved tensor of neuron indices to prune."""
    k_prune_neurons = torch.load(in_path)
    return k_prune_neurons


def non_intersection(tensor1, tensor2):
    """Return the symmetric difference (elements in exactly one of the inputs)."""
    combined = torch.cat((tensor1, tensor2))
    uniques, counts = combined.unique(return_counts=True)
    non_intersection = uniques[counts == 1]
    return non_intersection
