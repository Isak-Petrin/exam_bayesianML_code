
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as snb

from scipy.optimize import minimize
from jax import value_and_grad
from jax import hessian
from jax import random
import numpy as np

# for plotting
from matplotlib.colors import ListedColormap

# for manipulating images


from jax import config
config.update("jax_enable_x64", True)

# style stuff
snb.set_theme(font_scale=1.25)
snb.set_style('darkgrid')
colors = ['r', 'g', 'b', 'y']



def log_npdf(x, m, v):
    return -0.5*(x-m)**2/v - 0.5*jnp.log(2*jnp.pi*v)

# convert from class label to one-hot encoding
def to_onehot(y, num_classes):
    return jnp.column_stack([1.0*(y==value) for value in jnp.arange(num_classes)])


# softmax transformation
def softmax(a_, axis=1):
    max_val = jnp.max(a_, axis=axis)                # get maximum value along axis
    a = a_ - jnp.expand_dims(max_val, axis=axis)    # subtract max value for numerical stability
    exp_a = jnp.exp(a)                                
    return exp_a/jnp.sum(exp_a, axis=axis)[:, None]

class BayesianLinearSoftmax(object):
    """ Bayesian linear softmax classifier with i.i.d. Gaussian priors """

    
    def __init__(self, X, y, alpha=1.):
        
        # data and prior
        self.X, self.y  = X, y
        self.N, self.D = self.X.shape
        self.alpha = alpha
        
        # num classes, num parameters and one-hot encoding
        self.num_classes = len(jnp.unique(y))
        self.num_params = self.num_classes * self.D
        self.y_onehot = to_onehot(self.y, self.num_classes)
        
        # fit
        self.compute_laplace_approximation()

    def log_prior(self, w_flat):
        """ Evaluates the log prior, i.e. log p(W). 
            The function accepts the argument w_flat, which is a flattened version of W, such that the shape of w_flat is (T,), where T = num_classes x D is the total number of parameters.
            The return value of the function must be a scalar.
        """
        log_prior_val = jnp.sum(log_npdf(w_flat, 0, 1./self.alpha))  

        # check dimensions and return
        assert log_prior_val.shape == ()
        return log_prior_val
        
    def log_likelihood(self, w_flat):
        """ Evaluates the log likelihood for dataset (self.X, self.y) using a Categorical distribution with softmax inverse link function
            The function accepts the argument w_flat, which is a flattened version of W, such that the shape of w_flat is (T,), where T = num_classes x D is the total number of parameters.
            The return value of the function must be a scalar.
        """
        
        # reshape from flat vector to matrix of size num_classes by D
        W = w_flat.reshape((self.num_classes, self.D))

        ##############################################
        # Your solution goes here
        ##############################################
        
        # compute values for each latent function
        y_all = self.X@W.T

        # normalize using softmax
        p_all = softmax(y_all)
        
        # evaluate 
        loglik_val =  jnp.sum(self.y_onehot*jnp.log(p_all))
        
        ##############################################
        # End of solution
        ##############################################

        # check dimensions and return
        assert loglik_val.shape == ()
        return loglik_val
        
    def log_joint(self, w_flat):
        return self.log_prior(w_flat) + self.log_likelihood(w_flat)
    
    def compute_laplace_approximation(self):
        """ computes Laplace approximation of model """

        w_init_flat = jnp.zeros(self.num_params)
        cost_fun = lambda W: -self.log_joint(W)
        result = minimize(value_and_grad(cost_fun), w_init_flat, jac=True)

        if result.success:
            w_MAP = result.x
            self.m_flat = w_MAP[:, None]    
            self.A_flat = hessian(cost_fun)(w_MAP)
            self.S_flat = jnp.linalg.inv(self.A_flat)
            return self.m_flat, self.S_flat
        else:
            print('Warning optimization failed')
            return None, None
    
    def predict_f(self, X_star, w_given = None):
        """ computes the posterior distribution of f_i(x, w) = w_i^T phi(x^*) for all K classes

            Arguments:
            X_star            --         PxD prediction points

            Returns
            mu_f_all_classes  --         posterior mean of f for all classes (shape: P x K)
            var_f_all_classes  --        posterior variance of f for all classes (shape: P x K)
         """
        
        # get relevant part for each of the K linear models

        mi = self.m_flat.reshape((self.num_classes, self.D))
        Si = [self.S_flat[i*self.D:(i+1)*self.D, i*self.D:(i+1)*self.D] for i in range(self.num_classes)]
    
        # compute mean and variance for each function
        mu_f_all_classes = X_star@mi.T
        var_f_all_classes = jnp.squeeze(jnp.stack([jnp.diag(X_star@Si[i]@X_star.T) for i in range(self.num_classes)], axis=1))

        return mu_f_all_classes, var_f_all_classes
    
    def generate_samples_f(self, X_star, num_samples=500, seed=456):
        key = random.PRNGKey(123)
        """ generates samples from the posterior distribution p(f^*|y, x^*) based on the Laplace approximation
            
            Arguments:
            X_star            --         PxD prediction points
            num_samples       --         number of Monte Carlo samples to use
            seed              --         seed for random number generator

            Returns
            f_samples         --         posterior samples of f^*, shape: P x num_classes x num_samples         
        """

        # generate samples (shape: num_samples x total_params)
        w_samples_flat = random.multivariate_normal(key, self.m_flat.ravel(), self.S_flat, shape=num_samples)

        # reshape (shape: num_classes x D x num_samples)
        W_samples = w_samples_flat.T.reshape((self.num_classes, self.D, num_samples))

        # compute samples of f_star for all classes (shape: num_classes x P x num_samples)
        f_samples = X_star@W_samples

        # swap ax for convenience  (shape: P x num_classes x num_samples)
        f_samples = jnp.swapaxes(f_samples, 0, 1)

        return f_samples
                
    def predict_y(self, X_star, W_given = None,num_samples=500, seed=123):
        """ computes and returns p(y^*=k|y, x^*) using Monte Carlo sampling
         
            Arguments:
            X_star            --         PxD prediction points
            num_samples       --         number of Monte Carlo samples to use
            seed              --         seed for random number generator

            Returns
            p_all             --         Post. pred. probabilities for each point in X_star for each class, shape: PxK array, where K is the number of classes
        """
        if W_given is not None:
            f_samples = X_star @ W_given.T
        else:
            # generate posterior samples of f* (shape: num_classes x P x num_samples)
            f_samples = self.generate_samples_f(X_star, num_samples, seed)
        
        ##############################################
        # Your solution goes here
        ##############################################
        
        # compute softmax for all individual samples (shape: P x num_classes x num_samples)
        p_all_samples = softmax(f_samples, axis=1)
        
        # compute mean over Monte Carlo samples  (shape: P x num_classes)
        p_all = p_all_samples.mean(2) if W_given is None else p_all_samples
        
        ##############################################
        # End of solution
        ##############################################
        
        assert p_all.shape == (len(X_star), self.num_classes), f"The shape of p_all was expected to be ({len(X_star)}, {self.num_classes}), but the actual shape was {p_all.shape}. Please check the code"
        return p_all
    