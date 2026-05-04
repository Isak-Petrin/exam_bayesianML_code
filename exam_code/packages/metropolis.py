
import jax.numpy as jnp
from jax import value_and_grad
from jax import grad
from jax import random
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as snb
from scipy.stats import multivariate_normal

snb.set_theme(font_scale=1.25)

class metropolis(object):
    
    def __init__(self, log_target, num_params, tau, num_iter, theta_init=None, seed=0, dis_prop = 0.5):
        
        # store data and hyperparameters
        self.log_target = log_target
        self.num_params = num_params
        self.tau = tau
        self.num_iter = num_iter
        self.theta_init = theta_init
        self.seed = seed
        self.dis_prop = dis_prop
        
        self.thetas = self.metropolis(self.log_target, self.num_params, self.tau, self.num_iter, self.theta_init, self.seed)
        self.post_warmup_thetas = self.thetas[int(self.num_iter*self.dis_prop):]

        #Statistics
        self.mean_thetas, self.var_thetas = jnp.mean(self.post_warmup_thetas), jnp.var(self.post_warmup_thetas)
        
    def metropolis(self, log_target, num_params, tau, num_iter, theta_init=None, seed=0):    
        """ Runs a Metropolis-Hastings sampler 
        
            Arguments:
            log_target:         function for evaluating the log target distribution, i.e. log \tilde{p}(theta). The function expect a parameter of size num_params.
            num_params:         number of parameters of the joint distribution (integer)
            tau:                standard deviation of the Gaussian proposal distribution (positive real)
            num_iter:           number of iterations (integer)
            theta_init:         vector of initial parameters (np.array with shape (num_params) or None)        
            seed:               seed (integer)

            returns
            thetas              np.array with MCMC samples (np.array with shape (num_iter+1, num_params))
        """ 
        
        # set initial key
        key = random.PRNGKey(seed)

        if theta_init is None:
            theta_init = jnp.zeros((num_params))
        
        # prepare lists 
        thetas = [theta_init]
        accepts = []
        log_p_theta = log_target(theta_init)
        
        for k in range(num_iter):

            # update keys: key_proposal for sampling proposal distribution and key_accept for deciding whether to accept or reject.
            key, key_proposal, key_accept = random.split(key, num=3)

            ##############################################
            # Your solution goes here
            ##############################################
            

            # get the last value for theta and generate new proposal candidate
            theta_cur = thetas[-1]
            theta_star = theta_cur + tau*random.normal(key_proposal, shape=(num_params, ))
            
            # evaluate the log density for the candidate sample
            log_p_theta_star = log_target(theta_star)

            # compute acceptance probability
            log_r = log_p_theta_star - log_p_theta
            A = min(1, jnp.exp(log_r))
            
            # accept new candidate with probability A
            if random.uniform(key_accept) < A:
                theta_next = theta_star
                log_p_theta = log_p_theta_star
                accepts.append(1)
            else:
                theta_next = theta_cur
                accepts.append(0)


            
            ##############################################
            # End of solution
            ##############################################
                
            thetas.append(theta_next)


            
        print('Acceptance ratio: %3.2f' % jnp.mean(jnp.array(accepts)))
            
        # return as np.array
        thetas = jnp.stack(thetas)

        # check dimensions and return
        assert thetas.shape == (num_iter+1, num_params), f'The shape of thetas was expected to be ({num_iter+1}, {num_params}), but the actual shape was {thetas.shape}. Please check your code.'
        return thetas
    
    
    def plot_trace(self):
        fig, axes = plt.subplots(1, self.num_params, figsize=(20, 4))
        if self.num_params == 1:
            axes.plot(self.post_warmup_thetas)
            axes.set_xlabel('Iteration')
            axes.set_ylabel('Parameter $\\theta$')
            axes.set_title('Trace of parameter $\\theta$', fontweight='bold')
        else:
            for i,theta in enumerate(self.post_warmup_thetas.T):
                axes[i].plot(theta)
                axes[i].set_xlabel('Iteration')
                axes[i].set_ylabel('Parameter $\\theta$')
                axes[i].set_title('Trace of parameter $\\theta$', fontweight='bold')
            
    def credability_interval(self, p):
        return [jnp.quantile(theta, q = p, axis = 0) for theta in self.post_warmup_thetas.T]
# sanity check: estimate the mean and variance of a N(x|1,3) Gaussian distribution
#p_target = lambda x: log_npdf(x, 1., 3.)

# run sampler
#thetas = metropolis(p_target, 1, 2., 20000, theta_init=jnp.array([0]))

# estimate the mean and variance of p_target and relative errors


