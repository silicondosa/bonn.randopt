#!/usr/bin/env python

import math
from random import random

import torch as th
from .nn_utils import FCNetwork, get_opt, get_loss, TData, DataLoader, V

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

    def __init__(self, exp, num_trials=128, layers=(5, 5, 5), num_epochs=10, 
                 lr=0.01, cuda=False):
        self.exp = exp
        self.num_trials = num_trials
        self.num_epochs = num_epochs
        self.cuda = cuda
        self.net = FCNetwork(num_in=len(exp.params), layers=layers)
        self.opt = get_opt(self.net.parameters(), lr)
        self.loss = get_loss()
        self.param_keys = self.exp.params.keys()

    def fit(self, exp=None):
        if exp is None:
            exp = self.exp

        params, results = [], []
        res_min = None
        # Gather the data
        for res in exp.all_results():
            params.append(self._extract_features(res.params))
            results.append(res.value)
            if res_min is None or res.value < res_min:
                res_min = res.value
        self.current_min = res_min

        # Fit the data
        dataset = DataLoader(TData(params, results), shuffle=True, num_workers=4)
        for epoch in range(self.num_epochs):
            for X, y in dataset:
                X, y = V(X), V(y)
                self.opt.zero_grad()
                pred = self.net.forward(X)
                loss = self.loss(pred, y)
                loss.backward()
                self.opt.step()


    def sample(self, exp=None):
        if exp is None:
            exp = self.exp

        best_ei = None
        best_params = None
        # Sample num_trials parameters
        for _ in range(self.num_trials):
            params = exp.sample_all_params()
            sol = self._extract_features(params)
            ei = self._ei(sol)
            if best_ei is None or best_ei < ei:
                best_ei = ei
                best_params = params
        return best_params


    def _ei(self, sol):
        mean, std = self._predict(sol)
        var = std**2
        gamma = (self.current_min - mean) / (var + EPSILON)
        return var * (gamma*cdf(gamma) + pdf(gamma))

    def _predict(self, sol):
        X = V(th.FloatTensor([sol]))
        preds = []
        for _ in range(10):
            preds.append(self.net.forward(X).data.storage()[0])
        p = th.FloatTensor(preds)
        return p.mean(), p.std()

    def _extract_features(self, params):
        return [params[k] for k in self.param_keys]
