from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
import optax
from inference.base_inference import InferenceEngine


class SVIInference(InferenceEngine):
    """
    Stochastic Variational Inference engine.
    """

    def run(self, rng_key, **data):
        """
        Runs SVI inference.

        Args:
            rng_key: JAX random key.
            data : Dictionary containing X and Y arrays.

        Returns:
            Guide and learned parameters.
        """
        # Data needed for the model's __call__ method (X, Y)
        # BNNModel also implicitly uses N from X.shape
        model_args = {k: v for k, v in data.items() if k in ['X', 'Y']}

        guide = AutoDiagonalNormal(self.model) # AutoDiagonalNormal is another option
        # Consider increasing learning_rate for BNNs or using a schedule
        optimizer = optax.adam(self.args.learning_rate)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO())

        print(f"Starting SVI for {self.args.num_steps} steps...")
        # Pass model_args to svi.run
        result = svi.run(rng_key, self.args.num_steps, **model_args)
        print("SVI finished.")

        assert result.params is not None, "SVI did not return parameters"
        return guide, result.params
