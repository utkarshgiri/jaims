import jax
import chex
import numpy
from jaims import Chain

array = jax.numpy.arange(1000).reshape(10,10,10)
chain = Chain(array)
assert isinstance(chain, Chain)
chex.assert_shape(chain.chain, (100,10))
chex.assert_shape(chain.as_numpy(), (100,10))

assert numpy.allclose(numpy.mean(array.reshape(-1,10)[-12:,:], axis=0), chain.last(12).mean())
assert numpy.allclose(numpy.median(array.reshape(-1,10)[-100:,:], axis=0), chain.last(100).median())
assert numpy.allclose(numpy.std(array.reshape(-1,10), axis=0), chain.std())
