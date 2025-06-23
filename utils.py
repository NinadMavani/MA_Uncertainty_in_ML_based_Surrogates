import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import arviz as az
from config import get_args

def save_inference_data(mcmc, model_name="bnn_model", save_path="bnn_inference.nc"):
    """
    Convert NumPyro posterior samples to ArviZ InferenceData and save to NetCDF.
    Works for HMC Posteriors only !!!
    Args:
        samples (dict): Posterior from NumPyro.
        observed_data (dict): Observed data (e.g., {"Y": Y}).
        model_name (str): Name of the model.
        save_path (str): Path to save the NetCDF file.
    """
    idata = az.from_numpyro(posterior=mcmc)
    idata.attrs["model"] = model_name
    az.to_netcdf(idata, save_path)
    print(f"Inference data saved to {save_path}")


def plot_predictions_hmc( X, Y, X_test, preds, var_ale, filename="svi_plot.pdf"):
    """
    Plot training data, mean predictions, and confidence intervals.

    Args:
        X: Training inputs.
        Y: Training targets.
        X_test: Test inputs.
        predictions: Posterior predictive samples.
        filename: Output filename for the plot.
    """
    args = get_args()

    if isinstance(args.var_ale, float): # Aleatoric uncertainty known 
        
        var_ale = args.var_ale
        std_ale = jnp.sqrt(args.var_ale)

        var_epistemic_model = jnp.var(preds, axis=0)
        #var_epistemic_model = y_std ** 2

        var_total_model = var_ale + var_epistemic_model

        #mean_prediction = jnp.mean(preds, axis=0)
        #print(mean_prediction.shape, "First")
        std_ale = jnp.tile(std_ale, (100 ))

    #
    if not isinstance(args.var_ale, float):
            
        means = preds["mean"]
        stds  = preds["std"]

        # Aleatoric uncertainty : expectation over predicted variance
        var_aleatoric_model  = jnp.mean(stds ** 2, axis = 0)

        # Epistemic uncertainty : variance of predicted means 
        var_epistemic_model = jnp.var(means, axis = 0)

        var_total_model = var_aleatoric_model + var_epistemic_model
        
        mean_prediction = jnp.mean(means, axis = 0)

    
    std_total_model = np.sqrt(var_total_model)

    mean_prediction = jnp.mean(preds, axis=0) # Shape must be (X_test[test,])
    print(mean_prediction.shape, "shape mean \n")
    percentiles = np.percentile(preds, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    ax.plot(X[:, 1], Y[:, 0], "kx", label="Training data")

    ax.fill_between(X_test[:, 1], percentiles[0, :].squeeze(), percentiles[1, :].squeeze(), color="lightblue", label="95% CI")
    
    ax.plot(X_test[:, 1], mean_prediction, "blue", lw=2.0, label="Mean prediction")

    lower_band_ale = mean_prediction - 2 * std_ale
    upper_band_ale = mean_prediction + 2 * std_ale 
    #print((mean_prediction - 2 * std_ale).shape)
    
    ax.fill_between(X_test[:, 1],  lower_band_ale,  upper_band_ale, label="Aleatoric Uncertainty")
    ax.set(xlabel="X", ylabel="Y", title="BNN Predictions")
    
    ax.legend()
    plt.savefig(filename)
    plt.close()


def plot_predictions_svi( X, Y, X_test, predictive_samples, var_ale, filename="svi_plot.pdf"):
    """
    Plot training data, mean predictions, and confidence intervals.

    Args:
        X: Training inputs.
        Y: Training targets.
        X_test: Test inputs.
        predictions: Posterior predictive samples.
        filename: Output filename for the plot.
    """
    args = get_args()

    mu_pred_samples = predictive_samples['Y_observed']
    epi_mu_pred_y_mean = predictive_samples['_y_mean']

    # Calculate mean and 95% CI for 'mu_pred' (Total)
    mean_mu_pred = jnp.mean(mu_pred_samples, axis=0)

    percentiles = np.percentile(mu_pred_samples, [2.5, 97.5], axis=0)

    #hdi_mu_pred = az.hdi(mu_pred_samples, hdi_prob=0.95)
    #lower_mu_pred = jnp.percentile(mu_pred_samples, 2.5, axis=0)
    #upper_mu_pred = jnp.percentile(mu_pred_samples, 97.5, axis=0)

    # Calculate mean and 95% CI for 'Y' (Total)
    #mean_y_pred = jnp.mean(y_pred_samples, axis=0)
    #hdi_y_pred = az.hdi(y_pred_samples, hdi_prob=0.95)
    #lower_y_pred = jnp.percentile(y_pred_samples, 2.5, axis=0)
    #upper_y_pred = jnp.percentile(y_pred_samples, 97.5, axis=0)


    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'o', ms=4, alpha=0.5, label='Observed Data', color='red')
    #plt.plot(X_data, Y_true, '--', color='black', label='True Function')

    # Plot Epistemic Uncertainty (from mu_pred)
    plt.plot(X_test[:,1], mean_mu_pred, color='blue', label='Mean Prediction ')
    plt.fill_between(X_test[:,1], percentiles[0, :].squeeze(), percentiles[1, :].squeeze(),
                        color='blue', alpha=0.2, label='95% CI')

    # Plot Total Uncertainty (from Y_pred)
    #plt.plot(X_data, mean_y_pred, color='green', linestyle=':', label='Mean Prediction (Total)') # Should be very close to mean_mu_pred
    #plt.fill_between(X_data.flatten(), lower_y_pred, upper_y_pred ,
    #                    color='green', alpha=0.1, label='95% HDI (Total)') # Wider band

    plt.title('BNN Predictive Uncertainty: Epistemic vs. Total')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    plt.close()