import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import Predictive
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroSample
import matplotlib.pyplot as plt
from tqdm.auto import trange
from matplotlib.offsetbox import AnchoredText
# Set random seed for reproducibility
np.random.seed(42)

# Define the Gaussian mixture parameters
means = [-4, 0, 4]
stds = [np.sqrt(2/5), np.sqrt(0.9), np.sqrt(2/5)]
weights = [1/3, 1/3, 1/3]

# Sample 750 values from the mixture of Gaussians
n_samples = 750
components = np.random.choice([0, 1, 2], size=n_samples, p=weights)
x_obs = np.array([np.random.normal(loc=means[c], scale=stds[c]) for c in components])

# Generate heteroscedastic noise
epsilon = np.random.normal(0, 1, size=n_samples)
y_obs = 7 * np.sin(x_obs) + 3 * np.abs(np.cos(x_obs / 2)) * epsilon

x_true = np.linspace(-6,6,2000)
y_exp = 7 * np.sin(x_true) #+ 3 * np.abs(np.cos(x_true / 2)) 

xlims = [-6, 6]
ylims = [-1.5, 2.5]

print("Checkpoint 1")
# Plotting
# plt.figure(figsize=(10, 6))
# plt.scatter(x_obs, y_obs, alpha=0.6, s=10, label='Samples')
# plt.plot(x_true, y_true, 'b', linewidth=2, label="True function")
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Heteroscedastic Regression: $y = 7\\sin(x) + 3|\\cos(x/2)|\\epsilon$')
# plt.legend()
# plt.grid(True)

# plt.savefig('Heteroscedastic.png')

class DynamicBNN(PyroModule):
    def __init__(self, input_dim=1, output_dim=1, hidden_dims=[10, 10], prior_scale=[10.0, 10.0]):
        super().__init__()
        
        self.activation = nn.Tanh()  # Can also be passed as an argument
        
        self.hidden_layers = nn.ModuleList()
        self.prior_scale = prior_scale

        # Input to first hidden layer
        self.hidden_layers.append(PyroModule[nn.Linear](input_dim, hidden_dims[0]))
        self.hidden_layers[0].weight = PyroSample(
            dist.Normal(0., self._scale(0)).expand([hidden_dims[0], input_dim]).to_event(2)
        )
        self.hidden_layers[0].bias = PyroSample(
            dist.Normal(0., self._scale(0)).expand([hidden_dims[0]]).to_event(1)
        )

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layer = PyroModule[nn.Linear](hidden_dims[i - 1], hidden_dims[i])
            layer.weight = PyroSample(
                dist.Normal(0., self._scale(i)).expand([hidden_dims[i], hidden_dims[i - 1]]).to_event(2)
            )
            layer.bias = PyroSample(
                dist.Normal(0., self._scale(i)).expand([hidden_dims[i]]).to_event(1)
            )
            self.hidden_layers.append(layer)

        # Output layer
        self.output_layer = PyroModule[nn.Linear](hidden_dims[-1], output_dim)
        self.output_layer.weight = PyroSample(
            dist.Normal(0., self._scale(-1)).expand([output_dim, hidden_dims[-1]]).to_event(2)
        )
        self.output_layer.bias = PyroSample(
            dist.Normal(0., self._scale(-1)).expand([output_dim]).to_event(1)
        )

    def _scale(self, i):
        """Get prior scale per layer, supports single float or list of floats."""
        if isinstance(self.prior_scale, list):
            if i < len(self.prior_scale):
                return self.prior_scale[i]
            else:
                return self.prior_scale[-1]
        return self.prior_scale

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        # Forward through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        mu = self.output_layer(x).squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1.0))

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        
        return mu


#pyro.clear_param_store()
#
#
# SVI Inference Method
#
#

model_BNN = DynamicBNN(input_dim=1, output_dim=1, hidden_dims=[5], prior_scale=[5.0, 5.0])

# Set Pyro random seed
pyro.set_rng_seed(42)


# Convert data to PyTorch tensors
x_train = torch.from_numpy(x_obs).float()
y_train = torch.from_numpy(y_obs).float()

mean_field_guide = AutoDiagonalNormal(model_BNN)
optimizer = pyro.optim.Adam({"lr": 0.01})

svi = SVI(model_BNN, mean_field_guide, optimizer, loss=Trace_ELBO())
pyro.clear_param_store()

num_epochs = 10000
progress_bar = trange(num_epochs)

for epoch in progress_bar:
    loss = svi.step(x_train, y_train)
    progress_bar.set_postfix(loss=f"{loss / x_train.shape[0]:.3f}")

x_test = torch.linspace(xlims[0], xlims[1], 10000)

predictive_svi = Predictive(model=model_BNN, guide=mean_field_guide, num_samples=50)
preds_svi = predictive_svi(x_test)


def plot_predictions(preds):
    y_pred = preds['obs'].T.detach().numpy().mean(axis=1)
    y_std = preds['obs'].T.detach().numpy().std(axis=1)

    fig, ax = plt.subplots()
    xlims = [-6, 6]
    ylims = [-10, 10]
    plt.xlim(xlims)
    plt.ylim(ylims)
    ax.set_xlabel("X", fontsize=30)
    ax.set_ylabel("Y", fontsize=30)

    ax.plot(x_true, y_exp, 'b-', linewidth=3, label="True function")
    ax.plot(x_obs, y_obs, 'ko', markersize=4, label="observations")
    ax.plot(x_obs, y_obs, 'ko', markersize=3)
    #ax.plot(x_test, y_pred, '-', linewidth=3, color="#408765", label="predictive mean")
    ax.plot(x_test, y_pred, '-', color="#408765" ,  label="Predictive Mean ")
    ax.fill_between(x_test, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.6, color='#86cfac', zorder=5)

    plt.legend(loc=4, fontsize=15, frameon=False)
    plt.show()


if __name__=="__main__":

    plot_predictions(preds_svi)
