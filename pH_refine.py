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
from typing import Union, Optional, List

from bussilab import coretools
from MDRefine import compute_new_weights

class PHData(coretools.Result):
    def __init__(self, g_exp : np.ndarray, sigma_exp : np.ndarray, legend_matrix : np.ndarray,
                 gs : dict, weights : dict, legend_weights : dict, ref_pH : float, ref_pops : np.ndarray,
                 pops : dict, log_fugacities : np.ndarray, ns_prot : np.ndarray, obs_names : list,
                 pH_vals : list):
        """ Class with the fixed quantities that are required to evaluate the loss function `pH_loss`. """
        
        super().__init__()

        assert len(g_exp) == len(sigma_exp), 'mismatch between experimental values and uncertainties'

        self.g_exp = g_exp
        """ 1-D array-like with experimental values """

        self.sigma_exp = sigma_exp
        """ 1-D array-like with experimental uncertainties """

        self.legend_matrix = legend_matrix
        """ `legend_matrix` returned by `ManageIndices`, needed to correctly map 1-D arrays `g_exp` and
        `sigma_exp` to corresponding observable and pH value """
        
        self.gs = gs
        """ Dict of 2-D array-like; each item correspond to a protonation state and its value is
        the 2-D array (M x N) of observables computed from MD simulations, with M the total n. of frames
        at given protonation state from all the simulations at constant pH and N the n. of observables """

        for k in weights.keys():
            if np.sum(weights[k]) != 1.0:
                weights[k] /= np.sum(weights[k])
        
        self.p0s = weights
        """ Dict of 1-D array-like with the reference normalized weights for each protonation state,
        given by the collection of all the sampled configurations at that protonation state from the
        simulations at constant pH """

        self.legend_weights = legend_weights
        """ Dict with lists of indices to map back the NumPy arrays with weights and observables from a unique
        array at fixed protonation state to the contributions from multiple constant-pH simulations.
        This attribute is not required to evaluate and minimize the `pH_loss` but for `compute_pH_consistency`. """

        self.ref_pH = ref_pH
        """ Reference value for the pH among the possible ones """

        if np.sum(ref_pops) != 1.0:
            pops /= np.sum(pops)
        
        self.ref_pops = ref_pops
        """ 1-D array-like with the populations of the protonation states at reference pH value `ref_pH` """

        self.pops = pops
        """ Dict with 1-D array-like with the original (namely, initial hypothesis) probabilities to be at
        the j-th protonation state for the each pH value. This information is lost when we collect multiple
        simulations at constant pH values at fixed protonation state. """

        self.log_fugacities = log_fugacities
        """ 1-D array-like with the logarithm of the fugacity factors, namely the values of
        $\beta \Delta \mu_i$ (length given by the n. of pH values) """

        self.ns_prot = ns_prot
        """ 1-D array-like with the numbers of protonation (length given by the n. of protonation states) """
        
        self.obs_names = obs_names
        """ List with the names of the observables (length given by the total n. of observables) """

        self.pH_vals = pH_vals
        """ List with the pH values (redundant since it can be got from `log_fugacities` and `ref_pH`) """

class ManageIndices():
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

class PHConsistency(coretools.Result):
    def __init__(self, ph_fraction : dict, eff_n_frames : dict, s : list, s_CG : list, s_CG_within : list,
                 avs : dict, stds : dict, zs : dict, bins : dict, hists : dict, hist_dkls : dict, hist_ress : dict,
                 pops : dict, pops_dkl : dict):
        """ Class with the fixed quantities that are required to evaluate the loss function `ph_loss`. """
    
        super().__init__()

        self.ph_fraction = ph_fraction
        """ Dict with the fraction of weighted frames coming from each pH for a given protonation state """

        self.eff_n_frames = eff_n_frames
        """ Dict with the effective n. of frames for each protonation state and pH value """
    
        self.s = s
        """ List of the entropy for the distribution at each protonation state """
    
        self.s_CG = s_CG
        """ List of the entropy for the clustered / coarse-grained (accordingly to the pH) distribution
        at each protonation state """
    
        self.s_CG_within = s_CG_within
        """ List of the 1d numpy.ndarray with the entropies of the clusters given by the pH at each
        protonation state """
    
        self.avs = avs
        """ Dict with the average values of the observables at different pH values and protonation states """
    
        self.stds = stds
        """ Dict with the error on-the-mean values of the observables at different pH values and protonation
        states; they are computed by dividing the standard deviation by the square root of the effective number
        of frames """
    
        self.zs = zs
        """ Dict with the z values comparing averages at different pH values """
    
        self.bins = bins
        """ Dict with the bins used for the 1d histograms of the observables (same bin for multiple pH values) """
    
        self.hists = hists
        """ Dict with the 1d histograms of the observables """

        self.hist_dkls = hist_dkls
        """ Dict with the values of the Kullback-Leibler divergence between 1d histograms of the observables `hists` """
    
        self.hist_ress = hist_ress
        """ Dict with the values of the residues (as defined in `compute_dkl`) between 1d histograms of the
        observables `hists` """
    
        self.pops = pops
        """ Dict with the populations of multiple protonation states at each pH value beyond the reference one
        `data.ref_pH`, as computed by `compute_pH_weights` following the grand-canonical statistics (correspondence
        between pH differences and fugacities) """

        self.pops_dkl = pops_dkl
        """ Dict with the values of Kullback-Leibler divergence between `pops` and corresponding values
        resulting from constant-pH MD simulations """

class Lambdas:
    def __init__(self, value, is_lambdas_fixed):
        self.value = value
        self.is_fixed = is_lambdas_fixed

class PHResult(coretools.Result):
    def __init__(self, gamma : float, check_gamma : float, log_ps, avs, avs_ph, rel_diff, chi2, dkl_p, dkl_pi, loss):
        """ Class with the results of `pH_loss`. """
        
        super().__init__()

        self.gamma = gamma

        self.check_gamma = check_gamma

        self.log_ps = log_ps

        self.avs = avs

        self.avs_ph = avs_ph

        self.rel_diff = rel_diff

        self.chi2 = chi2

        self.dkl_p = dkl_p
        
        self.dkl_pi = dkl_pi

        self.loss = loss

def load_pH_data(path = 'Simulation-data', mol_name = 'A5mer', obs_names = ['chi', 'eRMSD'],
                 pH_vals = [3.50, 4.00, 4.50], ref_pH = 4.5, g_exp = None, sigma_exp = None):
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
    
    pH_vals : list
        List of float values corresponding to the pH values (they should also match with the values required
        for the names of the data files).
    
    ref_pH : float
        Float for the reference value of pH, used for reference populations in protonation states and fugacities
        (`pi0` and `log_fugacities` attributes of `PHData`).
        Suggestion: take the pH value to be the closest one to the pKa, namely, take the one such that the
        populations of protonated/deprotonated states are as similar as possible (avoid that one is much less
        than the other).

    g_exp, sigma_exp : numpy.ndarray
        Numpy 2-dimensional arrays with measured values and corresponding uncertainties; they are structured
        as a table with rows corresponding to the observables and columns corresponding to the pH values, as
        listed in `obs_names` and `pH_vals`.
        If None, it will return the average values and standard deviations on the mean.
    
    Return
    ----------

    ph_data : PHData
        Instance of the `PHData` class.
    """

    assert ref_pH in pH_vals, 'ref_pH not valid'

    # 1. read files and get ns, gs, weights, and ns_prot

    path = path + '/' + mol_name + '/'

    ns = {}
    gs = {}
    weights = {}

    for ph in pH_vals:
        # ns[ph] = np.array(pandas.read_csv(path + 'A5mer_pH0%.2f.occ' % ph, header=None).iloc[:, 0])
        ns[ph] = numpy.loadtxt(path + mol_name + '_pH0%.2f.occ' % ph)

        # df = pandas.read_csv(path + 'COLVAR_REWEIGHT_0%.2f' % ph, header=3, sep=' ').iloc[:, :4]
        # df.columns = list(pandas.read_csv(path + 'COLVAR_REWEIGHT_0%.2f' % ph, nrows=0, sep=' '))[2:]
        df = pandas.read_csv(path + 'COLVAR_REWEIGHT_0%.2f_weighted' % ph, header=0, sep=' ', comment='#')  # [1:]  # .iloc[:, :1]

        gs[ph] = df[obs_names]
        weights[ph] = np.array(df['weight'])

    ns_prot = numpy.sort(numpy.unique([ns[ph] for ph in pH_vals]))
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

        for ph in pH_vals:
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

    pops = {}

    for ph in pH_vals:
        pops[ph] = []

        for n_prot in ns_prot:
            pops[ph].append(np.sum(weights[ph][ns[ph] == n_prot]))

        pops[ph] = np.array(pops[ph])/np.sum(weights[ph])

    delta_ph = np.array(pH_vals) - ref_pH
    log_fugacities = - delta_ph*np.log(10)
    
    # 4. from tables of experimental values and uncertainties, make a 1d array
    # if tables are absent, use average values and standard deviations on the mean

    if g_exp is None:

        table_av = numpy.zeros((len(obs_names), len(pH_vals)))
        table_std = + table_av

        for i, obs_name in enumerate(obs_names):
            for j, ph in enumerate(pH_vals):
                # first index observable, second index ph
                if obs_name in gs[ph].columns:
                    table_av[i, j] = np.mean(np.array(gs[ph][obs_name]))
                    table_std[i, j] = np.std(np.array(gs[ph][obs_name]))/np.sqrt(len(gs[ph][obs_name]))

        g_exp = table_av
        sigma_exp = table_std

    legend_matrix = ManageIndices.build_legend(g_exp)[0]
    g_exp = ManageIndices.flatten(g_exp)
    sigma_exp = ManageIndices.flatten(sigma_exp)

    return PHData(g_exp, sigma_exp, legend_matrix, my_gs, my_ws, my_legend, ref_pH, pops, log_fugacities,
                   ns_prot, obs_names, pH_vals)

def entropy_fun(p):
    p = p[p != 0]
    entropy = np.sum(p*np.log(p))
    return entropy

def compute_dkl(p, p0, if_zero = False):
    """
    Compute the Kullback-Leibler divergence between `p` and `p0`.
    If `if_zero` is True, then remove from `p` and `p0` the points with `p0 = 0` so that no `inf` value
    will be returned. To check that this modification is just due to statistical fluctuation, return also
    the total removed probability.
    """
    
    p0 = p0[p != 0]
    p = p[p != 0]

    if if_zero:
        tot = np.sum(p[p0 == 0])
        p = p[p0 != 0]
        p0 = p0[p0 != 0]
    
    dkl = np.sum(p*np.log(p/p0))
    
    if if_zero: return dkl, tot
    else: return dkl

def _fun_zeta_val(mu1, mu2, sigma1, sigma2):
    z = (mu1 - mu2)/np.sqrt(sigma1**2 + sigma2**2)
    return z

def compute_pH_weights(log_weights, log_fugacity, ns_prot):
    """ Compute the weights of the protonation states at a given pH value, determined by `log_fugacity`. """
    
    log_weights = +log_weights # needed to avoid external modifications of the input (reference-type)
    log_weights += log_fugacity*ns_prot
    log_weights -= np.max(log_weights)

    weights = np.exp(log_weights)
    weights /= np.sum(weights)

    return weights

def compute_weights_pH(logW, ns_prot = None, pH = None):
    """
    This function computes normalized weights of each frame at arbitrary pH,
    starting from 
    """

    logW = +logW  # needed to avoid external modifications of the input (reference-type)

    if ns_prot is not None:
        logW -= np.log(10)*pH*ns_prot
        assert pH is not None, 'error: pH is None'

    logW -= np.max(logW)
    weights = np.exp(logW)
    weights /= np.sum(weights)

    return weights

def compute_pH_consistency(data):
    """
    1. Compute total weight of each pH value to each protonation state and the effective number of frames.

    Notice that the contribution of each pH value to each protonation state is not a physical quantity
    and can be arbitrarily modified (for example, one can count more a given pH value by doing a longer
    MD simulation). The point is that each pH value at fixed protonation state is a sampling from the
    canonical ensemble and they should be all consistent with each other.

    Notice also that, when multiple samplings at different pH values corresponding to the same protonation
    state are merged, the effective number of frames is not the sum of those corrresponding to each sampling.
    However, the entropy fulfills an equation in the same spirit of the ANOVA for the variances.

    2. Compute the entropy with and without clusters: multiple samplings of the canonical ensemble are put
    together (from simulations at different pH values) and we compute the entropy of the whole distribution `s`,
    the entropy of the clustered distribution `s_CG` and the entropies of the distributions inside each cluster
    `s_CG_within`. A relation similar to ANOVA holds between these values of entropy.
    
    3. Compute average values `avs` and standard errors on the mean `stds` (given by the standard deviation
    divided by the square root of the effective n. of frames), for each protonation state, pH value and
    observable. Then, compare values for different pH values, are they compatible with each other?
    This agreement is quantified by the `z` values.

    4. Comparing average values is not enough to claim that there is agreement: let's consider also 1-dimensional
    histograms `my_bins`, `my_hists` and compute the KL divergence among them `my_dkls`, as indicated in
    `compute_dkl`, so we also have the "residues" `my_ress`.

    5. Compare the computation of the population of protonation states at multiple pH values with the values
    from MD simulations.
    """

    #### part 1

    my_tot = {}
    split_weights = {}
    split_gs = {}
    eff_n_frames = {}

    for n_prot in data.ns_prot:

        my_tot[n_prot] = []
        split_weights[n_prot] = []
        split_gs[n_prot] = []
        eff_n_frames[n_prot] = []

        indices = data.legend_weights[n_prot]
        
        for i in range(len(indices) - 1):
            my_tot[n_prot].append(np.sum(data.p0s[n_prot][indices[i] : indices[i + 1]]))
            split_weights[n_prot].append(data.p0s[n_prot][indices[i] : indices[i + 1]]/my_tot[n_prot][-1])
            split_gs[n_prot].append(data.gs[n_prot][indices[i] : indices[i + 1]])
        
            eff_n_frames[n_prot].append(1/np.sum(split_weights[n_prot][-1]**2))

        my_tot[n_prot] = np.array(my_tot[n_prot])

    #### part 2

    s = []  # list with the entropy of the i-th protonation state from merging multiple samplings at different pH
    s_CG = []  # list with the entropy of the clustered distribution (i-th protonation state)
    s_CG_within = []  # list with the entropies for each cluster

    for n_prot in data.ns_prot:

        s.append(entropy_fun(data.p0s[n_prot]))
        s_CG.append(entropy_fun(my_tot[n_prot]))

        my_vec = [my_tot[n_prot][i]*entropy_fun(split_weights[n_prot][i]) for i in range(len(data.pH_vals))]
        s_CG_within.append(np.array(my_vec))

    #### part 3

    avs = {}
    stds = {}

    for n_prot in data.ns_prot:
        avs[n_prot] = []
        stds[n_prot] = []

        for i_ph in range(len(data.pH_vals)):
            avs[n_prot].append(np.average(split_gs[n_prot][i_ph], axis=0, weights=split_weights[n_prot][i_ph]))
            variance = np.average((split_gs[n_prot][i_ph] - avs[n_prot][-1])**2, axis=0, weights=split_weights[n_prot][i_ph])
            stds[n_prot].append(np.sqrt(variance/eff_n_frames[n_prot][i_ph]))

    zs = {}

    for n_prot in data.ns_prot:
        zs[n_prot] = []
        
        for i1_ph in range(len(data.pH_vals)):
            for i2_ph in range(i1_ph + 1, len(data.pH_vals)):
                zs[n_prot].append(_fun_zeta_val(avs[n_prot][i1_ph], avs[n_prot][i2_ph], stds[n_prot][i1_ph], stds[n_prot][i2_ph]))

    #### part 4

    my_bins = {}
    my_hists = {}
    my_dkls = {}
    my_ress = {}

    for i_obs, name_obs in enumerate(data.obs_names):
        my_bins[name_obs] = []
        my_hists[name_obs] = []
        my_dkls[name_obs] = []
        my_ress[name_obs] = []

        for n_prot in range(len(data.ns_prot)):

            hists = []

            i_ph = 0
            hist, bins = np.histogram(split_gs[n_prot][i_ph][:, i_obs], weights=split_weights[n_prot][i_ph], bins=50, density=True)
            hists.append(hist)

            for i_ph in range(1, len(data.pH_vals)):
                hist = np.histogram(split_gs[n_prot][i_ph][:, i_obs], weights=split_weights[n_prot][i_ph], bins=bins, density=True)
                hists.append(hist[0])

            dkl = []
            res = []

            for i1_ph in range(len(data.pH_vals)):
                for i2_ph in range(i1_ph + 1, len(data.pH_vals)):
                    out = compute_dkl(hists[i1_ph], hists[i2_ph], True)
                    dkl.append(out[0])
                    res.append(out[1])

            my_bins[name_obs].append(bins)
            my_hists[name_obs].append(hists)
            my_dkls[name_obs].append(dkl)
            my_ress[name_obs].append(res)

    ### part 5

    pi0 = data.ref_pops  # [data.ref_pH]
    my_pH_vals = set(data.pH_vals) - set([data.ref_pH])

    pops = {}
    pops_dkl = {}

    for i, ph in enumerate(my_pH_vals):
        pops[ph] = compute_pH_weights(np.log(pi0), data.log_fugacities[i], data.ns_prot)
        pops_dkl[ph] = compute_dkl(pops[ph], data.pops[ph])

    return PHConsistency(my_tot, eff_n_frames, s, s_CG, s_CG_within, avs, stds, zs, my_bins, my_hists,
                          my_dkls, my_ress, pops, pops_dkl)

def pH_loss(lambdas : np.ndarray, pis : np.ndarray, data : PHData, alphas : Union[float, List[float]] = 1.,
            alpha_pi : float = 1.):
    """
    Function to compute the loss function for pH refinement, defined as in documentation (1/2 chi2 + reg. terms).

    Parameters
    ----------
    
    lambdas : 1-D array-like
        Numpy 1-dimensional array, each element corresponds to the lambda value for an experimental observable
        at a certain pH value; this correspondence is given by `ManageIndices.flatten` (from table of values
        to 1d array) and `ManageIndices.flat_to_matrix` (from 1d array to table of values).
    
    pis : 1-D array-like
        Numpy 1-dimensional array for the (normalized) populations of each protonation state at reference pH `data.ref_pH`.

    data : PHData
        Instance of the `PHData` class with all the quantities (from experiments and MD simulations) required to
        evaluate and minimize the loss function `pH_loss`.

    alphas : float or list of floats
        Values of the hyperparameters for each canonical ensemble corresponding to a protonation state; by default,
        `alphas = 1.`, that means `alphas = np.ones(len(data.ns_prot))`.

    alpha_pi : float
        Value of the hyperparameter for the `pis` probability distribution (populations of each protonation state at
        reference pH).

    Return
    ----------

    result : PHResult
        Instance of the `PHResult` class with all the quantities computed to evaluate the loss function.
    """

    assert (np.all(np.array(alphas) > 0)), 'error on alphas, it must be positive!'
    if isinstance(alphas, (int, float)):
        alphas = alphas*np.ones(len(data.ns_prot))
    else:
        assert len(alphas) == len(data.ns_prot), 'error on alphas, it must have the same length as data.ns_prot'

    exp_values = np.vstack((data.g_exp, data.sigma_exp)).T

    ph_weights = []

    for i in range(len(data.pH_vals)):
        w = compute_pH_weights(np.log(pis), data.log_fugacities[i], data.ns_prot)
        ph_weights.append(w)

    ph_weights = np.array(ph_weights)

    gamma, logZs, corrections = pH_gamma(lambdas, data.legend_matrix, data.gs, exp_values, data.p0s, alphas, ph_weights)

    log_ps = []
    avs = []
    avs_correction = []

    for j in range(len(data.ns_prot)):

        log_ps.append(np.log(data.p0s[j]) - logZs[j] - corrections[j])

        p = np.exp(log_ps[j])  # normalized by definition of log_ps
        avs.append(np.dot(p, data.gs[j]))
        avs_correction.append(np.dot(p, corrections[j]))

    avs = np.vstack(avs)  # (N. prot. states x N. obs) matrix

    avs_ph = np.dot(ph_weights, avs).T
    # avs_ph is a full matrix (N. obs x N. pH), but it may happen that only some of its cells have a
    # corresponding experimental value

    exp_vals = ManageIndices.flat_to_matrix(data.g_exp, data.legend_matrix)
    exp_errs = ManageIndices.flat_to_matrix(data.sigma_exp, data.legend_matrix)

    rel_diff = np.where(np.isnan(exp_vals) | np.isnan (exp_errs), np.nan, (avs_ph - exp_vals)/exp_errs)

    chi2 = np.sum(rel_diff**2)

    loss = 1/2*chi2

    dkl_p = []

    for j in range(len(data.ns_prot)):
        dkl_p.append(-logZs[j] - avs_correction[j])

    dkl_p = np.array(dkl_p)

    dkl_pi = compute_dkl(pis, data.ref_pops)  # [data.ref_pH])

    loss += np.dot(alphas, dkl_p) + alpha_pi*dkl_pi

    check_gamma = 1/2*chi2 + np.dot(alphas, dkl_p) + gamma

    return PHResult(gamma, check_gamma, log_ps, avs, avs_ph, rel_diff, chi2, dkl_p, dkl_pi, loss)

def pH_gamma(lambdas, legend_matrix, gs, g_exp, weights_ref, alphas, ph_weights):
    """
    Compute the Gamma function for the pH refinement.

    Parameters
    ----------
    
    lambdas : 1-D array-like
        Numpy 1-dimensional array, each element corresponds to the lambda value for an experimental observable
        at a certain pH value; this correspondence is given by `ManageIndices.flatten` (from table of values
        to 1d array) and `ManageIndices.flat_to_matrix` (from 1d array to table of values).
    
    legend_matrix :
        Numpy 2-dimensional array used to map `lambdas` 1d array into the table of values by
        `ManageIndices.flat_to_matrix` (first index is for the observable, second index for the pH).

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

    logZs : 1-D array-like
        Numpy 1-dimensional array for the logarithms of the partition function at each protonation state.

    corrections : list
        List of 1-dimensional arrays with the corrections to the reference ensemble at each protonation state.
    """
    # if len(alphas) == 1:
    # then just a single hyperparameter alpha (so, optimize over a single hyperparameter)
    logZs = []
    corrections = []
    
    n_ph = len(weights_ref)

    table_lambdas = ManageIndices.flat_to_matrix(lambdas, legend_matrix)
    table_lambdas = np.nan_to_num(table_lambdas)  # put nan to zero
    
    for j in range(n_ph):
        # print(fake_lambdas.shape, ph_weights.shape, gs[j].shape)
        # print(np.einsum('ki,i,lk', fake_lambdas, ph_weights[:, j], gs[j]))
        correction_lambdas = 1/alphas[j]*np.einsum('ki,i,tk->t', table_lambdas, ph_weights[:, j], gs[j])
        log_Z_lambda = compute_new_weights(weights_ref[j], correction_lambdas)[1]

        corrections.append(correction_lambdas)
        logZs.append(log_Z_lambda)

    logZs = np.array(logZs)
    
    gamma = 1/2*np.sum((lambdas*g_exp[:, 1])**2) + np.dot(lambdas, g_exp[:, 0]) + np.sum(logZs)

    return gamma, logZs, corrections

def _pH_gamma_only(lambdas, legend_matrix, gs, g_exp, weights_ref, alphas, ph_weights):
    gamma = pH_gamma(lambdas, legend_matrix, gs, g_exp, weights_ref, alphas, ph_weights)[0]
    return gamma

pH_gamma_gradient_fun = jax.grad(_pH_gamma_only, argnums=0)

def pH_gamma_and_grad(lambdas, legend_matrix, gs, g_exp, weights_ref, alphas, ph_weights):
    args = (lambdas, legend_matrix, gs, g_exp, weights_ref, alphas, ph_weights)
    gamma = _pH_gamma_only(*args)
    grad = pH_gamma_gradient_fun(*args)
    return gamma, grad

def pH_tilde_loss(log_pi_vec : np.ndarray, ph_data : PHData, lambdas : Optional[Lambdas] = None, alpha_pi : float = 1.,
                  alphas : Union[float, List[float]] = 1.):
    """
    This is the loss function L̃(log(pi_j)) depending on `log_pi_vec`.

    It does not depend on `lambdas` in the sense that:
    - if `is_lambdas_fixed` is False, then the optimal lambdas are determined by minimizing the `ph_gamma`
        function at given `log_pi_vec` with input `lambdas` used only as a starting point for the minimization;
    - else, we suppose the input `lambdas` are already the optimal ones and we just compute corresponding
        `pH_gamma` value; this is useful when we have already minimized over $\lambda$ and we want to compute
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

    ph_data : object of class PHData
        This object includes all the fixed variables upon which you minimize the loss function based on
        reweighting.
        (it includes: $\sigma_{ki,exp},\,g_{ki,exp},\,g_k(x)$ with $x$ in each protonation state,
        $P_{0j}(x)$, $\vec\pi_0$, the fugacity factors $e^{\beta\Delta\mu_i}$ related to pH values, the protonation numbers $N_j$ and also the `legend_matrix`)
    
    Return
    ----------

    loss : float
        Value of the `pH_loss` function $\mathcal L$, corresponding to $\mathcal L$ since we are in the minimum
        over $\vec\lambda$.
    """

    if lambdas is None: lambdas = Lambdas(np.zeros(len(ph_data.g_exp)), False)
    
    if alphas == 1: alphas = np.ones(len(ph_data.ns_prot))
    log_pi_ref = np.log(ph_data.ref_pops)  # [ph_data.ref_pH])

    log_pi_vec -= np.mean(log_pi_vec)  # enforce zero-mean gauge
    
    # 1. compute ph_weights from pi_vec (ph_weights is a 2d array, `ph_weights[i, j]` is the probability
    # of the protonation state `j` at pH `i`, normalized over `j` for every `i`)
    ph_weights = []

    for log_fug in ph_data.log_fugacities:
        weights = compute_pH_weights(log_pi_vec, log_fug, ph_data.ns_prot)
        ph_weights.append(weights)
    
    ph_weights = np.array(ph_weights)
    
    # 2. minimize Gamma function at given ph_weights (or evaluate Gamma at optimal lambdas)
    exp_values = np.vstack((ph_data.g_exp, ph_data.sigma_exp)).T
    args = (ph_data.legend_matrix, ph_data.gs, exp_values, ph_data.p0s, alphas, ph_weights)

    if not lambdas.is_fixed:
        mini = minimize(pH_gamma_and_grad, lambdas.value, args=args, method='BFGS', jac=True)  # , options={'gtol': gtol})
        gamma = mini.fun
        lambdas.value = mini.x  # update value of lambdas
    else:
        gamma = _pH_gamma_only(lambdas.value, *args)
    
    # 3. add dkl value to compute the total loss value
    pi_vec = np.exp(log_pi_vec - np.max(log_pi_vec))
    pi_vec /= np.sum(pi_vec)
    
    dkl = np.sum(pi_vec*(np.log(pi_vec) - log_pi_ref))
    
    # wrong because log_pi must be normalized!!
    ## dkl = np.sum(np.exp(log_pi_vec)*(log_pi_vec - log_pi_ref))
    
    loss = - gamma + alpha_pi*dkl

    return loss

pH_tilde_loss_gradient_fun = jax.grad(pH_tilde_loss, argnums=0)

def pH_tilde_loss_and_grad(log_pi_vec, ph_data, lambdas, alpha_pi : float = 1.,
                           alphas : Union[float, List[float]] = 1., is_verbose = True):
    """ Here lambdas are starting values for lambdas """

    assert not lambdas.is_fixed, 'error: lambdas is fixed'
    args = (log_pi_vec, ph_data, lambdas, alpha_pi, alphas)
    loss = pH_tilde_loss(*args)

    lambdas.is_fixed = True
    # in this way, the gradient is computed without looking at the derivative of lambdas w.r.t. log_pi_vec,
    # which is zero, since we are at the optimal lambdas for that log_pi_vec value
    
    grad = pH_tilde_loss_gradient_fun(*args)

    lambdas.is_fixed = False

    if is_verbose:
        print('loss, grad: ', loss, grad)
        
        pi_vec = np.exp(log_pi_vec)
        pi_vec /= np.sum(pi_vec)
        print('pi_vec: ', pi_vec)
        print('\n')

    return loss, grad

