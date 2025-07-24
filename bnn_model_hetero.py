import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def get_activation(name):
    """
    Returns the activation function corresponding to the given name.
    Args:
        name (str): Name of the activation function.
    Returns:
        function: Activation function.
    """
    if name == "relu":
        return lambda x: jnp.maximum(0, x)
    elif name == "tanh":
        return jnp.tanh
    elif name == "sigmoid":
        return lambda x: 1 / (1 + jnp.exp(-x))
    elif name == "gelu":
        return lambda x: 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))
    else:
        raise ValueError(f"Unsupported activation function: {name}")


class BNNModel_Hetero:
    """
    Bayesian Neural Network model class with configurable prior mean and variance.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, activation, var_ale, prior_mean=0.0, prior_std=1.0):
        """
        Initialize the BNN model.

        Args:
            input_dim (int) : Number of input dimensions
            hidden_dim (int): Number of hidden units per layer.
            num_layers (int): Number of hidden layers.
            output_dim (int) : Number of output dimensions
            
            activation (str): Activation function name.
            var_ale (float or None) : Variance of Aleatoric uncertainty (known/unknown)
            prior_mean (float): Mean of the prior distribution.
            prior_std (float): Standard deviation of the prior distribution.
        """
        assert input_dim > 0, "Input Dimension must be positive"
        assert hidden_dim > 0, "Hidden dimension must be positive"
        assert num_layers > 0, "Number of layers must be positive"
        assert output_dim > 0, "Output dimension must be positive"
        assert prior_std > 0, "Prior standard deviation must be positive"
        #assert var_ale >= 0, "Aleataoric Variance"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.activation = get_activation(activation)
        self.var_ale = var_ale
        self.prior_mean = prior_mean
        self.prior_std = prior_std

    def __call__(self, X, Y=None):
        """
        Defines the probabilistic model for the BNN.

        Args:
            X (jax.numpy.ndarray): Input features.
            Y (jax.numpy.ndarray): Target values.
        """
        #assert X.ndim == 2, "Input X must be a 2D array"
        if Y is not None:
            assert Y.shape[0] == X.shape[0], "Y must have the same number of samples as X"

        N, _ = X.shape
        layers = [self.input_dim] + [self.hidden_dim] * self.num_layers

        z = X

        for i in range(len(layers) - 1):
            w = numpyro.sample(
                f"w{i}",
                dist.Normal(self.prior_mean, self.prior_std).expand([layers[i], layers[i + 1]])
            )
            b = numpyro.sample(
                f"b{i}",
                dist.Normal(self.prior_mean, self.prior_std).expand([layers[i + 1]])
            )
            z = self.activation(jnp.matmul(z, w) + b)
            assert z.shape == (N, self.hidden_dim), f"Layer {i} output shape mismatch"

        w_out = numpyro.sample(
            "w_out",
            dist.Normal(self.prior_mean, self.prior_std).expand([self.hidden_dim, 2 * self.output_dim])
        )
        b_out = numpyro.sample(
            "b_out",
            dist.Normal(self.prior_mean, self.prior_std).expand([2 * self.output_dim])
        )
        z_out = jnp.matmul(z, w_out) + b_out
        
        assert z_out.shape == (N, 2 * self.output_dim), "Output layer shape mismatch"

        if not isinstance(self.var_ale, float):

            #std_ale = jnp.sqrt(self.var_ale)

            mean, log_var = jnp.split(z_out, 2, axis=-1)
            std = jnp.exp(0.5 * log_var)

            if Y is not None:
                # observe data    
                with numpyro.plate("data", N):
                    # note we use to_event(1) because each observation has shape (1,) 
                    # # Check implementation # #
                    numpyro.sample("Y", dist.Normal(z_out, std_ale).to_event(1), obs=Y)
            
            else:
                numpyro.deterministic("mean", mean)
                numpyro.deterministic("std", std)
