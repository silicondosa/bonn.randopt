#!/usr/bin/env python

"""
Defines the main class for Bayesian Optimization with Tree of Parzen Estimators.
"""


from .bonn import Bo


class Botpe(Bo):

    def __init__(self, exp):
        self.exp = exp

    def fit(self, exp=None):
        if exp is None:
            exp = self.exp


    def sample(self, exp=None):
        if exp is None:
            exp = self.exp
