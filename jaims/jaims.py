import jax
import tqdm
import numpy
from jaims import Chain
from functools import partial
from typing import Callable, Optional, Union, Tuple

__all__ = ['Sampler']

#jarray = Union[jax.numpy.ndarray, jax.numpy.DeviceArray] 
jarray = Union[jax.numpy.ndarray, jax.Array] 

class Sampler:
    '''
    A class for sampling 
    '''
    
    def __init__(self, likelihood: Callable,
                       walkers: jarray,
                       steps: int=1,
                       burn_in: Union[float, int]=0):
        '''
        Args:
            likelihood: a callable that computes the log likelihood or the log posterior
            walkers: a jax.numpy array of shape (num_walkers, ndim) that contain blob of 
                    position around the initial guess.
            steps: an int that tells us the number of steps of sampling to perform
            burn_in:  fraction or percentage to burn-in
            #key: a jax.random.PRNGkey value to be used as a key
        '''

        self.chain = None
        self.likelihood = likelihood
        self.walkers = walkers
        self.steps = steps
        self.ndim = self.walkers.shape[1]
        self.num_walkers = self.walkers.shape[0]
        self.burn_in = burn_in

        self._a = 2
        self._size = self.num_walkers//2
        self._ids = (jax.numpy.arange(self._size), jax.numpy.arange(self._size) + self._size)



    @partial(jax.jit, static_argnums=(0,))
    def _proposal(self, sample: jarray,
                        complement: jarray,
                        key: jarray) -> Tuple[jarray, jarray]: 
        '''
        A proposed set of walkers based on parallel stretch move of dfm++
        Args:
            sample: a primary set of walkers of array size (num_walkers//2, ndim)
            complement: a complementary set of walkers of size (num_walkers//2, ndim)
            key: a jax.random.PRNGkey value to be used as a key
        Returns:
            a proposed set of walkers of size (num_walkers//2, ndim)
        '''

        key, subkey = jax.random.split(key)
        z = (self._a - 1.0) * (jax.random.uniform(subkey, shape=(self._size,), minval=0, maxval=1) + 1) ** 2.0 / self._a
        factors = (self.ndim - 1.0) * jax.numpy.log(z)
        key, subkey = jax.random.split(key)
        rint = jax.random.randint(subkey, shape=(self._size,), minval=0, maxval=self._size)
        return complement[rint,:] + (sample - complement[rint,:]) * z[:,None], factors
    

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, sample: jarray, 
                     complement: jarray, 
                     keys: jarray) -> jarray:
        '''
        An accepted set of next move for sample
        Args:
            sample: a primary set of walkers of array size (num_walkers//2, ndim)
            complement: a complementary set of walkers of size (num_walkers//2, ndim)
            key: a jax.random.PRNGkey value to be used as a key
        Returns:
            an accepted set of walkers of size (num_walkers//2, ndim)
        '''        
        q, factors = self._proposal(sample, complement, keys[0])
        likq = self.likelihood(q)
        likq = jax.numpy.where(jax.numpy.isnan(likq), -jax.numpy.inf, likq)
        accepted = jax.numpy.where(factors + likq - self.likelihood(sample) >
                jax.numpy.log(jax.random.uniform(keys[1], shape=(self._size,), minval=0, maxval=1)), 1, 0)
        sample = jax.numpy.where(accepted[:,None], q, sample)
        return sample


    def run_mcmc(self, key: jarray) -> jarray:
        '''
        A function to perform MCMC sampling
        Args:
            key: a jax.random.PRNGkey value to be used as a key
        Returns:
            a jax.numpy array of shape(steps*num_walkers, ndim) containing proposed samples
        '''
        #TODO re-write this more optimally using fori_loop, perhaps!
        samples = []
        for _ in tqdm.tqdm(range(self.steps)):
            key, *subkey = jax.random.split(key, 5)
            jax.random.permutation(key, self.walkers)
            sample, complement = self.walkers[self._ids[0],:], self.walkers[self._ids[1],:]
            sample = self.sample(sample, complement, subkey[:2])
            complement = self.sample(complement, sample, subkey[2:])
            self.walkers = jax.numpy.concatenate([sample, complement], axis=0)
            samples.append(self.walkers)
        self.chain = Chain(jax.numpy.stack(samples), burn_in=self.burn_in)
        return self.chain


    def reset(self, likelihood: Callable,
                       walkers: jarray,
                       steps: int=1,
                       shuffle: bool=True, 
                       key: jarray=jax.random.PRNGKey(43)):
        '''A function to reset the sampler although the recommended way is to create a new instance'''

        self.__init__(likelihood, walkers, steps, shuffle, key)

    def get_chain(self):

        return self.chain

    def unflatten(self):

        return self.chain.reshape(self.steps, self.num_walkers, self.ndim) 


