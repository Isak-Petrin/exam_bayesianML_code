import jax.numpy as jnp

sigmoid = lambda x: 1 / (1+jnp.exp(-x))

log_npdf = lambda x, m, v: -0.5*jnp.log(2*jnp.pi*v) -0.5*(x-m)**2/v 