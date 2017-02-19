#!/usr/bin/env python

import math
from random import random

"""
Defines the main class for Bayesian Optimization with Neural Networks.
"""

EPSILON = 1e-8

def pdf(x):
    # Probability distriution function for the standard normal distriution
    return math.exp(-(x**2) / 2.0) / math.sqrt(2*math.pi)

def cdf(x):
    # Cumulative distribution function for the standard normal distribution
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


class Bo(object):

    def __init__(self, exp):
        self.exp = exp

    def fit(self, exp=None):
        raise NotImplementedError()


    def sample(self, exp=None):
        raise NotImplementedError()


class Bonn(Bo):

    def __init__(self, exp, num_trials=128, layers=(5, 5, 5), opt=None,
                 loss=None, cuda=False):
        self.exp = exp
        self.num_trials = num_trials

    def fit(self, exp=None):
        if exp is None:
            exp = self.exp

        res_min = None
        for res in exp.all_results():
            if res_min is None or res.value < res_min:
                res_min = res.value
        self.current_min = res_min


    def sample(self, exp=None):
        if exp is None:
            exp = self.exp

        best_ei = None
        best_params = None
        # Sample num_trials parameters
        for _ in range(self.num_trials):
            params = exp.sample_all_params()
            ei = self._ei(params)
            if best_ei < ei:
                best_ei = ei
                best_params = params
        return best_params


    def _ei(self, sol):
        mean, std = self._predict(sol)
        var = std**2
        gamma = (self.current_min - mean) / (var + EPSILON)
        return var * (gamma*cdf(gamma) + pdf(gamma))

    def _predict(self, sol):
        return random(), random()
