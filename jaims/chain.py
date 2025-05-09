import jax
import numpy
import joblib
import pathlib
import logging
from typing import Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ['Chain']

#jarray = Union[numpy.ndarray, jax.numpy.DeviceArray]
jarray = Union[numpy.ndarray, jax.Array]

class Chain:

    def __init__(self, chain: jarray=None, burn_in: Union[float, int]=0):
        
        self.chain = chain if chain.ndim == 2 else chain.reshape(-1, chain.shape[-1])
        
        if burn_in:
            self.chain = self.burn(burn_in)

        self._param_names = [str(i) for i in range(self.num_parameters)]

    def __array__(self):
        return self.chain
    
    def __getitem__(self, index:int):
        return Chain(self.chain.__getitem__(index))

    def __repr__(self):
        return ('''Chain({})'''.format(self.chain))

    @property
    def param_names(self):
        return self._param_names

    @param_names.setter
    def param_names(self, names):
        assert isinstance(names, Union[tuple, list]), 'Argument `names` must be a sequence of string'
        assert len(names) == self.num_parameters, 'Length of `names` is not compatiple with number of parameters'
        self._param_names = names

    def as_numpy(self):
        return numpy.asarray(self.chain)

    def flatten(self):
        """A helper function to flatten tthe stack of chain to 
        the shape (nsamples, ndim)
        """
        return self.chain.reshape(-1, self.chain.shape[-1])
    
    @property
    def num_parameters(self):
        """Returns total number of parameters"""
        return self.chain.shape[-1]

    @property
    def total_samples(self):
        """Returns total number of samples in the chain"""
        return self.chain.shape[0]

    def last(self, n: int=1):
        '''A function that returns a Chain object with last n samples of the instance
        Args:
            n: number of samples from the end to return. By default it returns the last sample
        Returns:
            a Chain objec with last n samples
        '''
        logger.debug(f'Returning last {n} samples from the chain')
        return Chain(self.chain[-n:,:])

    def burn(self, x: Union[float, int]=0):
        '''Returns a Chain after throwing away samples from the begining'''

        if 0 < x < 1:
            logger.info('Interpreting argument as a fraction. Throwing away {100*x} % samples')
            burn_in = int(x*self.total_samples)
        elif 1 <= x < 100:
            logger.info('Interpreting argument as a percentage. Throwing away {x} % samples')
            burn_in = (x*self.total_samples/100)
        else:
            raise ValueError('x must be between in the range (0,100)')

        return Chain(self.chain[burn_in:,:])


    def median(self, lastn: int=0):
        '''A helper function to compute the median of the Chain'''
        return jax.numpy.median(self.chain[-lastn:], axis=0)

    def mean(self, lastn: int=0):
        '''A helper function to compute the median of the Chain'''
        return jax.numpy.mean(self.chain[-lastn:], axis=0)
    
    def std(self, lastn: int=0):
        '''A helper function to compute the standard deviation of the Chain'''
        return jax.numpy.std(self.chain[-lastn:], axis=0)

    def save_parameters(self, filename, lastn: int=0):
        '''A helper function to save parameters to disk'''
        jax.numpy.save(filename, self.last(lastn).median())

    def save(self, filename):
        """A helper method to save the instance
        Args:
            filename ([type]): Name of the file s
        """
        logger.info(f'Saving file to {filename}.chain')
        joblib.dump(self, pathlib.Path(filename).with_suffix('.chain'))
    
    @classmethod
    def load(cls, filename):
        """A class method to load map2D object from disk
        Args:
            filename ([type]): Name of the file to load from disk
        Returns:
            Chain: The loaded map2D object
        """
        filename = pathlib.Path(str(filename))
        assert filename.exists()
        logger.info(f'Loading file {filename}.chain')
        return joblib.load(pathlib.Path(filename).with_suffix('.chain'))


    def triangle_plot(self, filename=None, lastn=0, labels=None):
        '''A function to plot trace/corner/triangle plot of samples'''
        try:
            from getdist import MCSamples, plots
        except ImportError:
            return ('getdist not found. Please install getdist')

        if labels is None:
            labels = self.param_names
        if filename is None:
            filename = 'figure'
        
        samples = MCSamples(samples=numpy.array(self.chain), names=labels, labels=labels,
                            settings={'fine_bins_2D':32, 'smooth_scale_1D':0.3})
        
        g = plots.get_subplot_plotter()
        g.triangle_plot([samples], filled=True)
        g.export(f'{filename}.pdf')
