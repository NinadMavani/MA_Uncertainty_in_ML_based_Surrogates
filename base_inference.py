from abc import ABC, abstractmethod

class InferenceEngine(ABC):
    """
    Abstract base class for inference engines.
    """

    def __init__(self, model, args):
        """
        Initialize the inference engine.

        Args:
            model: The probabilistic model to use.
            args: Configuration arguments.
        """
        self.model = model
        self.args = args

    @abstractmethod
    def run(self, rng_key, X, Y):
        """
        Run the inference algorithm.

        Args:
            rng_key: JAX random key.
            X: Input features.
            Y: Target values.

        Returns:
            Inference results.
        """
        pass
