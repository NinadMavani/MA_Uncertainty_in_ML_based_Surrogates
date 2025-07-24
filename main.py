import os
from config import get_args
from bnn_model import BNNModel
from bnn_model_hetero import BNNModel_Hetero
from base_inference import InferenceEngine
from hmc_inference import HMCInference
import numpyro
import jax.random as random
from jax import vmap
import jax.numpy as jnp
from numpyro import handlers
from numpyro.infer import Predictive
from data import *
import seaborn as sns

from utils import *

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))



def predict(model, rng_key, samples, X):
    """
    Predict output using a seeded and substituted model.
    """
    substituted_model = handlers.substitute(handlers.seed(model, rng_key), samples)
    trace = handlers.trace(substituted_model).get_trace(X=X, Y=None)
    return trace["Y_observed"]["value"] # Set this to Y_observed for correct shape

def main():
    """
    Main entry point for training and evaluating the Bayesian Neural Network.
    """
    args = get_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    # Random keys
    #rng_key_data, rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    rng_key_data, rng_key, rng_key_predict = random.split(random.PRNGKey(0), 3)

    # Load dataset
    
    X_obs, Y_obs, X_test, X_GRID, Y_GRID = load_objective_func_1(args.num_data, rng_key_data)

    data = {
        "X": X_obs,
        "Y": Y_obs
    }

    # Initialize model
    model = BNNModel(1, args.num_data, args.num_layers, 1, args.activation, args.var_ale, args.prior_mean, args.prior_std)#, save_model=True)
    model = model.__call__

    
    # Select inference engine
    if args.inference == "hmc":
        engine = HMCInference(model, args)
        mcmc, samples = engine.run(rng_key, **data)
        
        save_inference_data(mcmc, model_name="bnn_hmc", save_path="bnn_mcmc.nc")

        # Vectorized predictions using vmap
        num_samples = args.num_samples * args.num_chains
        rng_keys = random.split(rng_key_predict, num_samples)

        # predict Y_test at inputs X_test
        vmap_args = (
            samples,
            random.split(rng_key_predict, args.num_samples * args.num_chains),
        )
        predictions = vmap(
            lambda samples, rng_key: predict(model, rng_key, samples, X_test)
        )(*vmap_args)

        preds = predictions[..., 0]  # Shape (n_samples*n_chains, X_test.shape[0])

        print(preds.shape)

        plot_predictions_hmc(X_obs, Y_obs, X_test, preds, args.var_ale, filename="bnn_hmc_plot.pdf")
            
        
    elif args.inference == "svi":
        engine = SVIInference(model, args)
        from numpyro.diagnostics import print_summary
        guide, params = engine.run(rng_key, **data)
        
        for k in params.keys():
            print(f"- {k}: {params[k].shape}")

        posterior_predictive = Predictive(model = model, guide = guide, params = params, num_samples=100, return_sites=['Y_observed', '_y_mean'])

        preds_svi = posterior_predictive(rng_key=random.PRNGKey(1), X=X_test, Y=None)

        print(preds_svi["Y_observed"].shape) # Shape must be (n_samples, X_test.shape[0])
        print(preds_svi["_y_mean"].shape)    # Shape must ne (n_samples, X_test.shape[0])

        plot_predictions_svi(X_obs, Y_obs, X_test, preds_svi, args.var_ale, filename="bnn_svi_plot.pdf")


    elif args.inference == "alpha":
        engine = AlphaBBVIInference(model, args)
        guide, params = engine.run(rng_key, X, Y)
        predictions = guide.sample_posterior(random.PRNGKey(1), params, sample_shape=(1000,))
        preds = predictions["Y"]
                
        idata = az.from_dict(posterior={k: v for k, v in posterior_samples.items() if k != "Y"})

        # Optionally add observed data
        idata.add_groups(observed_data={"Y": Y})

        # Save to NetCDF
        az.to_netcdf(idata, "alpha_bb_inference.nc")


    else:
        raise ValueError("Unsupported inference method")

    
if __name__ == "__main__":
    main()
