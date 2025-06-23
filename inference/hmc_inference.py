from numpyro.infer import MCMC, NUTS
from inference.base_inference import InferenceEngine

class HMCInference(InferenceEngine):
    """
    Hamiltonian Monte Carlo inference using NUTS.
    """

    def run(self, rng_key, **data):
        """
        Runs HMC inference.

        Args:
            rng_key: JAX random key.
            data : Dictionary containing X and Y arrays.

        Returns:
            Posterior samples.
        """

        X, Y = data["X"], data["Y"]

        assert X.shape[0] == Y.shape[0], "Mismatch in number of samples between X and Y"
        
        
        # # Define Hamiltonian Monte Carlo (HMC) kernel
        # # NUTS = "No-U-Turn Sampler" (https://arxiv.org/abs/1111.4246), gives HMC an adaptive step size
        kernel = NUTS(self.model)
        
        mcmc = MCMC(kernel, num_warmup=self.args.num_warmup,
                    num_samples=self.args.num_samples,
                    num_chains=self.args.num_chains)
        
        mcmc.run(rng_key, X, Y)
        
        mcmc.print_summary()
        
        samples = mcmc.get_samples()
        
        assert samples is not None and len(samples) > 0, "No samples returned from HMC"
        
        return mcmc, samples
