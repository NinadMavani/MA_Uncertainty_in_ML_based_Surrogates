import jax.numpy as jnp
import numpy as np


def load_objective_func_1(num_points, rng_key_data):
    """
    Generates synthetic training and test data.
    Returns:
        tuple: (X_train, Y_train, X_test, X_GRID, Y_GRID)
    """
    
    rng_seed_list = rng_key_data.tolist()
    rng = np.random.default_rng(rng_seed_list)
    
    X_GRID = np.linspace(-1, 1, num_points).reshape(-1, 1)
    Y_GRID = 2.0 * X_GRID[:, 0]**3 + 0.05 * np.random.randn()
    Y_GRID = Y_GRID.reshape(-1, 1)
    
    X_test = np.linspace(-1.3, 1.3, 100).reshape(-1, 1)

    # Arrays and func used in the paper BNN for UQ in data-driven materials modeling
    # https://www.sciencedirect.com/science/article/abs/pii/S0045782521004102  

    X_obs = np.array([-0.02519606, -0.29152739, -0.60474655, 0.31944225, -0.08100553, -0.24830156, 0.57461577,
               0.50232181, 0.60433894, -0.02046175, 0.53479088, -0.65367602, -0.06110107, 0.46652892,
               -0.66163461, 0.26793157, 0.20481661, -0.24144274, -0.42398829, -0.52080597]).reshape((-1, 1))

    Y_obs = np.array([0.04928457864952569, -0.11915410490457669, -0.405097551770553, 0.029554098140267056,
               -0.013086956159543405, -0.017770100521146612, 0.42280077037504055, 0.1944984572601308,
               0.4534092801344878, -0.05744532400253988, 0.27416952296635494, -0.6450129511010473,
               -0.00434618253501617, 0.16330603887330705, -0.5274704221475347, 0.02189741180766931,
               0.012647796994763167, 0.08367359752673682, -0.10875986459325471,
               -0.2964629150726794]).reshape((-1, 1))


    X_obs = jnp.array(X_obs)
    Y_obs = jnp.array(Y_obs)
    X_test = jnp.array(X_test)

    assert X_obs.shape[0] == num_points, "Mismatch in expected number of training samples"
    assert X_obs.ndim == 2 and Y_obs.ndim == 2, "X and Y must be 2D arrays"
    return X_obs, Y_obs, X_test, X_GRID, Y_GRID



def load_objective_func_2(num_points, variance_data, rng_key_data):
    """
    Generates synthetic training and test data.
    Returns:
        tuple: (X_train, Y_train, X_test, X_GRID, Y_GRID)
    """
    
    # Set random seed for reproducibility
    rng_seed_list = rng_key_data.tolist()
    rng = np.random.default_rng(rng_key_data)

    # Define the Gaussian mixture parameters
    means = [-4, 0, 4]
    stds = [np.sqrt(2/5), np.sqrt(0.9), np.sqrt(2/5)]
    weights = [1/3, 1/3, 1/3]

    components = np.random.choice([0, 1, 2], size=num_points, p=weights)

    X_obs = np.array([np.random.normal(loc=means[c], scale=stds[c]) for c in components])
    
    # Generate heteroscedastic noise
    epsilon = np.random.normal(0, jnp.sqrt(variance_data), size=num_points)
    
    Y_obs = (7 * np.sin(X_obs) + 3 * np.abs(np.cos(X_obs / 2)) * epsilon)
    
    ## For the given functin y = f(x) 
    # Here grid represents the expectation of the function E(y)

    X_GRID = np.linspace(-6, 6, 10000).reshape(-1, 1)
    Y_GRID =  7 * np.sin(X_GRID)
    Y_GRID = Y_GRID.reshape(-1, 1)
    
    X_test = np.linspace(-6, 6, 100).reshape(-1, 1)
        
    #x_true = np.linspace(-6,6,10000)
    #y_exp = 7 * np.sin(x_true) # + 3 * np.abs(np.cos(x_true / 2)) 
    #epsilon_true = np.random.normal(0, 0.5, size = 10000)

    #y_true = y_exp + 3 * np.abs(np.cos(x_true / 2)) * epsilon_true
    #
   
    X_obs = jnp.array(X_obs.reshape(-1,1))
    Y_obs = jnp.array(Y_obs.reshape(-1,1))
    X_test = jnp.array(X_test)

    assert X_obs.shape[0] == num_points, "Mismatch in expected number of training samples"
    assert X_obs.ndim == 2 and Y_obs.ndim == 2, "X and Y must be 2D arrays"
    return X_obs, Y_obs, X_test, X_GRID, Y_GRID