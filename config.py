import argparse

def get_args():
    """
    Parses command-line arguments for configuring the Bayesian Neural Network.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Bayesian Neural Network Configuration")

    # Data-specific
    parser.add_argument("--num-data", type=int, default=20, help="Number of training data points")
    #parser.add_argument("--variance-data", type=float, default= 0.05 ** 2, help="Variance of data generation (Aleatoric Uncertainty)")
    parser.add_argument("--var-ale", type=int, default= 0.5 * 0.5)

    # Inference settings
    parser.add_argument("--inference", type=str, default="hmc",
                        choices=["hmc", "svi", "alpha"],
                        help="Inference method to use")

    # Model architecture
    parser.add_argument("--num-hidden", type=int, default=100, help="Number of hidden units per layer")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "tanh", "sigmoid", "gelu"],
                        help="Activation function to use")

                        
    parser.add_argument("--prior-mean", type=int, default=0.0)
    parser.add_argument("--prior-std", type=int, default=1.0)


    # HMC-specific
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-warmup", type=int, default=100)
    parser.add_argument("--num-chains", type=int, default=5)

    # SVI-specific
    parser.add_argument("--num-steps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    
    parser.add_argument("--num-particles", type=int, default=10000, help="Particles for alpha-divergence")

    # Device
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])

    return parser.parse_args()
