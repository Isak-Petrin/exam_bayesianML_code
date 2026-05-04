
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

from mpl_toolkits.axes_grid1 import make_axes_locatable

snb.set_style('darkgrid')
snb.set_theme(font_scale=1.)

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

jax.config.update("jax_enable_x64", True)

sigmoid = lambda x: 1./(1 + jnp.exp(-x))
log_npdf = lambda x, m, v: -(x-m)**2/(2*v) - 0.5*jnp.log(2*jnp.pi*v)

class LogisticRegression(object):

    def __init__(self, X, y, feature_transformation=lambda x: x, alpha=1.):
        
        # store data and hyperparameters
        self.X0 = X
        self.y = y
        self.alpha = alpha
        self.feature_transformation = feature_transformation

        # apply feature transformation and standardize
        self.X = feature_transformation(self.X0)
        self.X_mean = jnp.mean(self.X, 0)
        self.X_std = jnp.std(self.X, 0)
        self.X_std.at[self.X_std == 0].set(1)
        self.X = self.preprocess(X)

        # store number of training data and number of features
        self.N, self.D = self.X.shape
        
        # get MAP by optimization
        self.w_MAP = self.get_MAP()

    def preprocess(self, X_):
        X = self.feature_transformation(X_)
        return (X - self.X_mean)/self.X_std
        
    def predict(self, X, w):
        """ evaluates sigma(f(X)) """
        f = w@X.T
        return sigmoid(f)
    
    def log_joint(self, w):
        """
            evaluates log joint, i.e. log p(y, w), for each row in w.
            w is expected to be of shape [M, D], where D is the number of parameters in the model and M is the number of points to evaluated
        """
        p = self.predict(self.X, w)
        log_prior = jnp.sum(log_npdf(w, 0, 1./self.alpha), axis=1)
        log_lik = binom_dist.logpmf(self.y, p=p, n=1)
        log_joint = log_prior + log_lik.sum(axis=1)

        return log_joint
    
    def hessian(self, w):
        """ Returns hessian of log joint evaluated at w 
            Input:   w       (shape: [1, D])
            Returns: H       (shape: [D, D])            """
        
        ##############################################
        # Your solution goes here
        ##############################################
        
        p = self.predict(self.X, w)
        v = p*(1-p)
        H = -self.X.T @ jnp.diag(v) @ self.X -self.alpha*jnp.identity(self.D)
        
        ##############################################
        # End of solution
        ##############################################
        
        assert H.shape == (self.D, self.D), f"The shape of the Hessians appears to be wrong. Expected shape ({self.D}, {self.D}), but received {H.shape}. Check your implementation"
        print("This is H",H)
        return H

    def grad(self, w):
        """ Returns gradient of log joint evaluated at w 
            Input:   w          (shape: [1, D])
            Returns: grad       (shape: [1, D])            """
        
        ##############################################
        # Your solution goes here
        ##############################################
        
        p = self.predict(self.X, w)
        err = p - self.y
        grad = -jnp.sum(err.T*self.X, axis=0) -self.alpha*w
        
        ##############################################
        # End of solution
        ##############################################

        assert grad.shape == (1, self.D), f"The shape of the gradient appears to be wrong. Expected shape (1, {self.D}), but received {grad.shape}. Check your implementation"
        print("This is grad",grad)
        return grad
  
    def get_MAP(self):
        """ returns MAP estimate obtained by maximizing the log joint """
        init_w = jnp.zeros(self.D)
        results = minimize(lambda x: -self.log_joint(x[None, :]), jac=lambda x: -self.grad(x[None, :]).flatten(), x0=init_w)
        if not results.success:
            print(results)
            raise ValueError('Optization failed')
        
        w_MAP = results.x 
        return w_MAP
    
        
