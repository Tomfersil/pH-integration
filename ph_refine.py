"""
This will become a module of MDRefine for the refinement with pH-dependent data from multiple MD simulations
at constant pH.

Fixing the protonation state, we have sampled every canonical ensemble (protonated / deprotonated,
for instance) multiple times and we want to refine these ensembles by comparing their average values of
selected observables with the experimental values, measured at multiple pH values. To this aim, we exploit
the grand canonical statistics.
"""

import jax.numpy as np
import numpy as numpy

from MDRefine import compute_new_weights

class Manage_indices():
    """
    Manage experimental value indices relative to observable and pH indices.

    This class relies both on `numpy` and `jax.numpy` because of filling matrices, which is not plainly
    allowed in Jax.

    This class provides utility methods for working with a 2D matrix of experimental values for example
    (`my_exp_values`), where:
    * The **first index** corresponds to the observable index.
    * The **second index** corresponds to the pH index.

    ### Methods
    - **build_legend(my_exp_values)** → `(legend_matrix, legend_row)`  
    Builds:
    - `legend_matrix`: a 2D array where each element stores the index of the corresponding
        value in a flattened list (`-1` if the value is absent).  
    - `legend_row`: a 1D array storing the ending index of each observable row in the flattened data.  

    - **flatten(my_exp_values)** → `flat_mat`  
    Flattens `my_exp_values` into a 1D array of non-NaN values according to `legend_matrix`.  

    - **flat_to_matrix(flat_mat, legend_matrix)** → `my_exp_values`  
    Reconstructs the original `my_exp_values` matrix from its flattened form and the legend.  

    ### Example
    ```python
    my_exp_values = np.array([
        [0.1, 0.2, 0.1, np.nan, np.nan],
        [np.nan, 0.4, 0.5, np.nan, 0.7],
        [np.nan, np.nan, np.nan, 0.2, np.nan],
        [0.3, 0.5, 0.8, np.nan, 0.1],
        [np.nan, np.nan, 0.3, 0.3, np.nan]
    ])

    # Build legend
    legend_matrix, legend_row = ManageIndices.build_legend(my_exp_values)

    # Flatten and reconstruct
    flat_mat = ManageIndices.flatten(my_exp_values)
    mat = ManageIndices.flat_to_matrix(flat_mat, legend_matrix)

    # Select values for a specific observable (row i)
    i = 3
    values_for_obs = flat_mat[legend_row[i] : legend_row[i + 1]]

    # Example usage in a correction computation:
    correction_lambdas = (1 / alphas[j]) * np.einsum(
        'ki,ij,kt->jt',
        lambdas[legend_row[i] : legend_row[i + 1]],
        ph_weights[j],
        g
    )
    ```

    This workflow is useful for handling non-trivial sums over observable and pH indices when
    a simple 1D representation of lambdas is insufficient.
    """
    def build_legend(my_exp_values):
        a, b = my_exp_values.shape

        legend_matrix = numpy.full((a, b), 0)
        legend_row = [0]

        tot = 0

        for i in range(a):
            for j in range(b):
                if not np.isnan(my_exp_values[i, j]):
                    legend_matrix[i, j] = tot
                    tot += 1
                else:
                    legend_matrix[i, j] = -1

            legend_row.append(tot)

        legend_matrix = np.int32(legend_matrix)
        legend_row = np.int32(legend_row)

        return legend_matrix, legend_row

    def flatten(my_exp_values):
        flat_mat = np.ravel(my_exp_values)
        flat_mat = flat_mat[~np.isnan(flat_mat)]
        return flat_mat

    def flat_to_matrix_old(flat_mat, legend_matrix):
        """ It uses also NumPy, not only Jax """

        mat = numpy.full(legend_matrix.shape, np.nan)

        whs = np.argwhere(legend_matrix >= 0)
        for wh in whs: mat[wh[0], wh[1]] = flat_mat[legend_matrix[wh[0], wh[1]]]

        mat = np.array(mat)

        return mat
    
    def flat_to_matrix(flat_mat, legend_matrix):
        """
        Reconstructs a matrix from its flattened representation and legend_matrix,
        using only JAX (no Python-side assignments).
        """

        # Initialize with NaNs
        mat = np.full(legend_matrix.shape, np.nan)

        # Indices where legend_matrix >= 0
        whs = np.argwhere(legend_matrix >= 0)

        # Values to insert: flat_mat[legend_matrix[idx]]
        vals = flat_mat[legend_matrix[whs[:, 0], whs[:, 1]]]

        # Update using JAX's immutable update API
        mat = mat.at[whs[:, 0], whs[:, 1]].set(vals)

        return mat

def ph_gamma(lambdas, legend_matrix, gs, g_exp, weights_ref, alphas, ph_weights):
    """
    Compute the Gamma function for the pH refinement.

    Parameters:
    ----------
    
    lambdas : 1-D array-like
        Numpy 1-dimensional array, each element corresponds to the lambda value for an experimental observable
        at a certain pH value; this correspondence is given by `Manage_indices.flatten` (from table of values
        to 1d array) and `Manage_indices.flat_to_matrix` (from 1d array to table of values).
    
    legend_matrix :
        Numpy 2-dimensional array used to map `lambdas` 1d array into the table of values by
        `Manage_indices.flat_to_matrix` (first index is for the observable, second index for the pH).

    gs : List of 2-D array-like
        List (one element for each protonation state) of Numpy 2-dimensional arrays (M x N);
        `g[i, j]` is the j-th observable computed in the i-th frame.
    
    g_exp : 2-D array-like
        Numpy 2-dimensional array (N x 2); `g_exp[j, 0]` is the experimental value of the j-th observable,
        `g_exp[j, 1]` is the associated experimental uncertainty.
    
    weights_ref : List of 1-D array-like
        List (one element for each protonation state) of Numpy 1-dimensional arrays, each of them is the
        set of weights for the reference ensemble.
    
    alphas : 1-D array-like
        Numpy 1-dimensional array for the values of the alpha hyperparameters.
    
    ph_weights : 2-D array-like
        Numpy 2-dimensional array, `ph_weights[i, j]` is the probability of the protonation state `j` at pH `i`,
        normalized over `j` for every `i`.

    Return
    ----------

    gamma : float
        Value of the pH_Gamma function (analogous to the Gamma function for the pH application).
    """
    # if len(alphas) == 1:
    # then just a single hyperparameter alpha (so, optimize over a single hyperparameter)
    logZs = []
    
    n_ph = len(weights_ref)

    table_lambdas = Manage_indices.flat_to_matrix(lambdas, legend_matrix)
    table_lambdas = np.nan_to_num(table_lambdas)  # put nan to zero
    
    for j in range(n_ph):
        # print(fake_lambdas.shape, ph_weights.shape, gs[j].shape)
        # print(np.einsum('ki,i,lk', fake_lambdas, ph_weights[:, j], gs[j]))
        correction_lambdas = 1/alphas[j]*np.einsum('ki,i,tk->t', table_lambdas, ph_weights[:, j], gs[j])
        log_Z_lambda = compute_new_weights(weights_ref[j], correction_lambdas)[1]
        logZs.append(log_Z_lambda)

    logZs = np.array(logZs)
    
    gamma = 1/2*np.sum((lambdas*g_exp[:, 1])**2) + np.dot(lambdas, g_exp[:, 0]) + np.sum(logZs)

    return gamma

