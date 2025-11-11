"""
This will become a module of MDRefine for the refinement with pH-dependent data from multiple MD simulations
at constant pH.

Fixing the protonation state, we have sampled every canonical ensemble (protonated / deprotonated,
for instance) multiple times and we want to refine these ensembles by comparing their average values of
selected observables with the experimental values, measured at multiple pH values. To this aim, we exploit
the grand canonical statistics.
"""

import numpy as numpy
import pandas
import jax
import jax.numpy as np  # consistently with MDRefine, np for jax.numpy
from scipy.optimize import minimize

from bussilab import coretools
from MDRefine import compute_new_weights

class Ph_data(coretools.Result):
    def __init__(self, g_exp : np.ndarray, sigma_exp : np.ndarray, legend_matrix : np.ndarray,
                 gs : dict, weights : dict, legend_weights : dict, ref_ph : float, pi0 : np.ndarray,
                 log_fugacities : np.ndarray, ns_prot : np.ndarray, obs_names : list, ph_vals : list):
        """ Class with the fixed quantities that are required to evaluate the loss function `ph_loss`. """
        
        super().__init__()

        self.g_exp = g_exp
        """ 1-D array-like with experimental values """

        self.sigma_exp = sigma_exp
        """ 1-D array-like with experimental uncertainties """

        self.legend_matrix = legend_matrix
        """ `legend_matrix` returned by `Manage_indices`, needed to correctly map 1-D arrays `g_exp` and
        `sigma_exp` to corresponding observable and pH value """
        
        self.gs = gs
        """ Dict of 2-D array-like; each item correspond to a protonation state and its value is
        the 2-D array (M x N) of observables computed from MD simulations, with M the total n. of frames
        at given protonation state from all the simulations at constant pH and N the n. of observables """

        self.p0s = weights
        """ Dict of 1-D array-like with the reference normalized weights for each protonation state,
        given by the collection of all the sampled configurations at that protonation state from the
        simulations at constant pH """

        self.legend_weights = legend_weights
        """ Dict with lists of indices to map back the NumPy arrays with weights and observables from a unique
        array at fixed protonation state to the contributions from multiple constant-pH simulations """

        self.ref_ph = ref_ph
        """ Reference value for the pH among the possible ones """

        self.pi0 = pi0
        """ 1-D array-like with the original (namely, initial hypothesis) probabilities to be at the
        j-th protonation state for the reference pH value """

        self.log_fugacities = log_fugacities
        """ 1-D array-like with the logarithm of the fugacity factors, namely the values of
        $\beta \Delta \mu_i$ (length given by the n. of pH values) """

        self.ns_prot = ns_prot
        """ 1-D array-like with the numbers of protonation (length given by the n. of protonation states) """
        
        self.obs_names = obs_names
        """ List with the names of the observables (length given by the total n. of observables) """

        self.ph_vals = ph_vals
        """ List with the pH values (redundant since it can be got from `log_fugacities` and `ref_ph`) """

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

def load_ph_data(path = 'Simulation-data', mol_name = 'A5mer', obs_names = ['chi', 'eRMSD'],
                 ph_vals = [3.50, 4.00, 4.50], ref_ph = 4.5, g_exp = None, sigma_exp = None):
    """
    Load pH data from multiple simulations at constant pH.

    Parameters:
    ----------
    
    path : str
        String with the name of the folder containing the directory `mol_name` with the data from
        constant-pH simulations.
    
    mol_name : str
        String with the name of the molecular system, corresponding to the name of the directory in `path`.
    
    obs_names : list
        List of strings with the names of the observables, corresponding to the selected columns of the txt files
        with observables and weights.
    
    ph_vals : list
        List of float values corresponding to the pH values (they should also match with the values required
        for the names of the data files).
    
    ref_ph : float
        Float for the reference value of pH, used for reference populations in protonation states and fugacities
        (`pi0` and `log_fugacities` attributes of `Ph_data`).
        Suggestion: take the pH value to be the closest one to the pKa, namely, take the one such that the
        populations of protonated/deprotonated states are as similar as possible (avoid that one is much less
        than the other).

    g_exp, sigma_exp : numpy.ndarray
        Numpy 2-dimensional arrays with measured values and corresponding uncertainties; they are structured
        as a table with rows corresponding to the observables and columns corresponding to the pH values, as
        listed in `obs_names` and `ph_vals`.
        If None, it will return the average values and standard deviations on the mean.
    
    Return
    ----------

    ph_data : Ph_data
        Instance of the `Ph_data` class.
    """

    assert ref_ph in ph_vals, 'ref_ph not valid'

    # 1. read files and get ns, gs, weights, and ns_prot

    path = path + '/' + mol_name + '/'

    ns = {}
    gs = {}
    weights = {}

    for ph in ph_vals:
        # ns[ph] = np.array(pandas.read_csv(path + 'A5mer_pH0%.2f.occ' % ph, header=None).iloc[:, 0])
        ns[ph] = numpy.loadtxt(path + mol_name + '_pH0%.2f.occ' % ph)

        # df = pandas.read_csv(path + 'COLVAR_REWEIGHT_0%.2f' % ph, header=3, sep=' ').iloc[:, :4]
        # df.columns = list(pandas.read_csv(path + 'COLVAR_REWEIGHT_0%.2f' % ph, nrows=0, sep=' '))[2:]
        df = pandas.read_csv(path + 'COLVAR_REWEIGHT_0%.2f_weighted' % ph, header=0, sep=' ', comment='#')  # [1:]  # .iloc[:, :1]

        gs[ph] = df[obs_names]
        weights[ph] = np.array(df['weight'])

    ns_prot = numpy.sort(numpy.unique([ns[ph] for ph in ph_vals]))
    ns_prot = ns_prot.astype(int)

    # 2. put together values from different pH at the same protonation state

    my_gs = {}
    my_ws = {}
    my_legend = {}

    for n_prot in ns_prot:
        # n_prot = int(n_prot)
        my_gs[n_prot] = []
        my_ws[n_prot] = []
        my_legend[n_prot] = [0]

        for ph in ph_vals:
            # my_gs[n_prot].append(np.array(gs[ph].iloc[ns[ph] == n_prot].loc[0][obs_names]))
            # my_gs[n_prot].append(gs[ph][ns[ph].to_numpy() == n_prot])
            my_gs[n_prot].append(np.array(gs[ph])[ns[ph] == n_prot])
            my_ws[n_prot].append(weights[ph][ns[ph] == n_prot])
            my_legend[n_prot].append(len(my_ws[n_prot][-1]))
        
        my_gs[n_prot] = np.vstack(my_gs[n_prot])
        my_ws[n_prot] = np.hstack(my_ws[n_prot])
        my_ws[n_prot] /= np.sum(my_ws[n_prot])
        my_legend[n_prot] = numpy.cumsum(my_legend[n_prot])

    # 3. given a ref. value for the pH, compute reference populations and log_fugacities

    pop = []

    for n_prot in ns_prot:
        pop.append(np.sum(weights[ref_ph][ns[ref_ph] == n_prot]))

    pop = np.array(pop)/np.sum(weights[ref_ph])

    delta_ph = np.array(ph_vals) - ref_ph
    log_fugacities = - delta_ph*np.log(10)
    
    # 4. from tables of experimental values and uncertainties, make a 1d array
    # if tables are absent, use average values and standard deviations on the mean

    if g_exp is None:

        table_av = numpy.zeros((len(obs_names), len(ph_vals)))
        table_std = + table_av

        for i, obs_name in enumerate(obs_names):
            for j, ph in enumerate(ph_vals):
                # first index observable, second index ph
                if obs_name in gs[ph].columns:
                    table_av[i, j] = np.mean(np.array(gs[ph][obs_name]))
                    table_std[i, j] = np.std(np.array(gs[ph][obs_name]))/np.sqrt(len(gs[ph][obs_name]))

        g_exp = table_av
        sigma_exp = table_std

    legend_matrix = Manage_indices.build_legend(g_exp)[0]
    g_exp = Manage_indices.flatten(g_exp)
    sigma_exp = Manage_indices.flatten(sigma_exp)

    return Ph_data(g_exp, sigma_exp, legend_matrix, my_gs, my_ws, my_legend, ref_ph, pop, log_fugacities,
                   ns_prot, obs_names, ph_vals)

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

ph_gamma_gradient_fun = jax.grad(ph_gamma, argnums=0)

def ph_gamma_and_grad(lambdas, legend_matrix, gs, g_exp, weights_ref, alphas, ph_weights):
    args = (lambdas, legend_matrix, gs, g_exp, weights_ref, alphas, ph_weights)
    gamma = ph_gamma(*args)
    grad = ph_gamma_gradient_fun(*args)
    return gamma, grad

def compute_ph_weights(log_pi, log_fugacity, ns):
    ph_weights = np.exp(log_pi + log_fugacity*ns)
    ph_weights /= np.sum(ph_weights)
    return ph_weights

def ph_loss(log_pi_vec, lambdas, is_lambdas_fixed, alpha_pi, alphas, ph_data):
    """
    This is the loss function $\tilde{\mathcal L}(\log\pi_j)$ depending on `log_pi_vec`.

    It does not depend on `lambdas` in the sense that:
    - if `is_lambdas_fixed` is True, then the optimal lambdas are determined by minimizing the `ph_gamma`
        function at given `log_pi_vec` with input `lambdas` used only as a starting point for the minimization;
    - else, we suppose the input `lambdas` are already the optimal ones and we just compute corresponding
        `ph_gamma` value; this is useful when we have already minimized over $\lambda$ and we want to compute
        the derivatives of $\mathcal L$ with respect to $\vec\pi$, since in this case the partial derivatives
        of $\mathcal L$ with respect to $\vec\lambda$ are zero and we can use `jax.grad` to get the gradient
        of $\mathcal L(\vec\pi)$ (otherwise `jax.grad` will consider also the minimization process!)
    
    Parameters:
    ----------

    log_pi_vec : 1-D array-like
        The variables $\log\pi_j$ to optimize (probabilities of the protonation states).
    
    lambdas : 1-D array-like
        The `lambdas` variables.

    is_lambdas_fixed : Bool
        Boolean variable, `True` if we do not want to minimize over $\vec\lambda$.

    alpha_pi : float
        The hyperparameter for the regularization of $\vec\pi$.

    alphas : 1-D array-like
        The hyperparameters for the regularization over $P_j$; you should include also the possibility for
        equal value for all the $P_j$.

    ph_data : object of class Ph_data
        This object includes all the fixed variables upon which you minimize the loss function based on
        reweighting.
        (it includes: $\sigma_{ki,exp},\,g_{ki,exp},\,g_k(x)$ with $x$ in each protonation state,
        $P_{0j}(x)$, $\vec\pi_0$, the fugacity factors $e^{\beta\Delta\mu_i}$ related to pH values, the protonation numbers $N_j$ and also the `legend_matrix`)
    
    Return
    ----------

    loss : float
        Value of the `ph_loss` function $\mathcal L$, corresponding to $\mathcal L$ since we are in the minimum
        over $\vec\lambda$.
    """
    
    log_pi_vec -= np.mean(log_pi_vec)  # enforce zero-mean gauge
    
    # 1. compute ph_weights from pi_vec (ph_weights is a 2d array, `ph_weights[i, j]` is the probability
    # of the protonation state `j` at pH `i`, normalized over `j` for every `i`)
    ph_weights = []

    for log_fug in ph_data.log_fugacities:
        weights = compute_ph_weights(log_pi_vec, log_fug, ph_data.n_prot)
        weights /= np.sum(weights)
        ph_weights.append(weights)
    
    ph_weights = np.array(ph_weights)
    
    # 2. minimize Gamma function at given ph_weights (or evaluate Gamma at optimal lambdas)
    exp_values = np.vstack((ph_data.g_exp, ph_data.sigma_exp))
    args = (ph_data.legend_matrix, ph_data.gs, exp_values, ph_data.weights_ref, alphas, ph_weights)

    if not is_lambdas_fixed:
        mini = minimize(ph_gamma_and_grad, lambdas, args=args, method='BFGS', jac=True)  # , options={'gtol': gtol})
        gamma = mini.fun
    else:
        gamma = ph_gamma(lambdas, *args)
    
    # 3. add dkl value to compute the total loss value
    pi_vec = np.exp(log_pi_vec - np.max(log_pi_vec))
    pi_vec /= np.sum(pi_vec)
    
    dkl = np.sum(np.exp(log_pi_vec)*(log_pi_vec - ph_data.log_pi_ref))
    loss = - gamma + alpha_pi*dkl

    return loss

ph_loss_gradient_fun = jax.grad(ph_loss, argnums=0)

def ph_loss_and_grad(log_pi_vec, lambdas, is_lambdas_fixed, alpha_pi, alphas, ph_data):

    args = (log_pi_vec, lambdas, is_lambdas_fixed, alpha_pi, alphas, ph_data)
    loss = ph_loss(*args)
    grad = ph_loss_gradient_fun(*args)
    return loss, grad

