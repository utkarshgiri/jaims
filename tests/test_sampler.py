import jax
import chex
import jaims
import numpy

key = jax.random.PRNGKey(42)
mean=4; sigma=5
steps = 20000; nwalkers=200

def log_pdf(mu):
    return -0.5*jax.numpy.sum((mu-mean)**2/sigma**2, axis=1)

walkers = jax.numpy.array(numpy.random.normal(loc=2, scale=1e-4, size=(nwalkers,1)))
walkers = jaims.helpers.initialize_random_walkers(nwalkers, numpy.array([2]), numpy.array([200]), numpy.array([-200]), 1)
sampler =  jaims.Sampler(log_pdf, walkers, steps=steps)
sampler.run_mcmc(key)

sample_mean = sampler.chain.last(1000000).mean()
sample_std = sampler.chain.last(1000000).std()
assert numpy.allclose(mean, sample_mean, atol=0.1), f'sample mean :{sample_mean}, true mean: {mean}'
assert numpy.allclose(sigma, sample_std, atol=0.1), f'sample std :{sample_std}, true std: {sigma}'
chex.assert_shape(sampler.chain.chain, (steps*nwalkers, sampler.chain.num_parameters))
sampler.chain.last(200000).triangle_plot()




