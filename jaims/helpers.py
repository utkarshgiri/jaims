import jax
import numpy
from typing import Union, Optional

jarray = Union[jax.numpy.ndarray, jax.numpy.DeviceArray] 

def initialize_random_walkers(nwalkers: int, loc: jarray, upper_bound: jarray, lower_bound: jarray, scale: Optional[Union[int, jarray]]=1e-10):
    '''
    A function to generate random blob of walkers

    Args:
        nwalkers: number of walkers
        loc: array that stores initial parameter guess
        upper_bound: array with upper bounds for the walkers
        lower_bound: array with lower bounds for the walkers
        scale: an int or an array with scale for random normal distribution

    Returns:
        a jax.numpy array of shape (nwalkers, ndim) which stores a random blob of walkers around initial guess
    '''

    initial_walkers = numpy.zeros((nwalkers, loc.size))

    for i in range(nwalkers):
        while numpy.all(initial_walkers[i,:] == 0):
            walker = numpy.random.multivariate_normal(loc, numpy.diag(numpy.full_like(loc, scale)))
            if numpy.all((lower_bound < walker) & (walker < upper_bound)):
                initial_walkers[i,:] = walker[:]

    return numpy.array(initial_walkers)


