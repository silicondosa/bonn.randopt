#!/usr/bin/env python

"""
Defines the main class for Bayesian Optimization with Gaussian Processes.
"""


from .bonn import Bo


class Bogp(Bo):

    def __init__(self, exp):
        self.exp = exp

    def fit(self, exp=None):
        if exp is None:
            exp = self.exp


    def sample(self, exp=None):
        if exp is None:
            exp = self.exp


