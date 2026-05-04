from packages.LaplaceApproximation import LaplaceApproximation
import jax.numpy as jnp
import jax
import numpy as onp
from jax import random
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as snb

from scipy.optimize import minimize

from scipy.stats import multivariate_normal as mvn
from scipy.stats import beta as beta_dist
from scipy.stats import binom as binom_dist
from scipy.stats import norm as norm_dist

probit = lambda x: norm_dist.cdf(x)
sigmoid = lambda x: 1./(1 + jnp.exp(-x))

class PosteriorPredictiveDistribution(object):
    
    def  __init__(self, model):
        self.model = model
        self.feature_transformation = model.feature_transformation
        self.laplace = LaplaceApproximation(model)

    def posterior_f(self, xstar_):
        """ computes the mean and variance of f^* = w^T x^* """
        xstar = self.model.preprocess(xstar_)
        m = xstar@self.laplace.posterior_mean
        v = jnp.diag(xstar@self.laplace.posterior_cov@xstar.T)
        return m, v

    def plugin_approx(self, xstar_):
        """ implements the plugin approximation for p(y^*|y, x^*) using w_MAP. If xstar has shape (M, D), then the shape of the output p must be (M,) """
        xstar = self.model.preprocess(xstar_)
        p = self.model.predict(xstar, self.model.w_MAP)

        assert p.shape == (len(xstar_),), f"Expected the shape of the output from the Monte Carlo approximation to be ({len(xstar)},), but the received shape was {p.shape}"
        return p
    
    def montecarlo(self, xstar, num_samples=1000, seed=0):
        """ implements the Monte Carlo estimator for p(y^*|y, x^*). If xstar has shape (M, D), then the shape of the output p must be (M,) """
        m, v = self.posterior_f(xstar)

        ##############################################
        # Your solution goes here
        ##############################################
        
        key = random.PRNGKey(seed)
        f = m  + jnp.sqrt(v)*random.normal(key, shape=(num_samples, len(xstar)))
        p = sigmoid(f).mean(0)
        
        ##############################################
        # End of solution
        ##############################################

        assert p.shape == (len(xstar),), f"Expected the shape of the output from the Monte Carlo approximation to be ({len(xstar)},), but the received shape was {p.shape}"
        return p

    def probit_approx(self, xstar):
        """ implements the probit approximation for p(y^*|y, x^*). If xstar has shape (M, D), then the shape of the output p must be (M,) """
        m, v = self.posterior_f(xstar)

        ##############################################
        # Your solution goes here
        ##############################################
        
        p = probit(m/jnp.sqrt(8/jnp.pi + v))
        
        ##############################################
        # End of solution
        ##############################################

        assert p.shape == (len(xstar),), f"Expected the shape of the output from the Monte Carlo approximation to be ({len(xstar)},), but the received shape was {p.shape}"
        return p