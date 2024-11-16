# S4Like: CMB-S4 single field r likelihood pipeline. 
# Author: Shamik Ghosh, 2024.

import numpy as np
import emcee as mc 
import pygtc as gtc
import camb_wrapper as cw
import os 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import splrep, splev
import sys

try:
    nthreads = int(os.environ['OMP_NUM_THREADS'])
except:
    print("WARNING: using single thread for parameter fit."); sys.stdout.flush()
    nthreads = 10

default_prior = {
    'r' : {
        'fix'  : None, 
        'min'  : 0.,
        'max'  : 0.01,
        'latex': r'$r$'
    },
    'A_lens' : {
        'fix'  : None,
        'min'  : 0.,
        'max'  : 0.1,
        'latex': r'$A_{\rm lens}$'
    }
}

orange = ['#ff7f00', '#cd6600', '#8b4500']
coral  = ['#ff7256', '#cd5b45', '#8b3e2f']
green  = ['#bcee68', '#a2cd5a', '#6e8b3d']
blue   = ['#00b2ee', '#009acd', '#00688b']
purple = ['#ab82ff', '#8968cd', '#5d478b']


def g_func(x):
    """
    g_func(x) computes the function:

    .. math::
        f(x) = sgn(x-1) \sqrt{2 (x - \log(x) - 1)}

    Parameters
    ----------
    x : float
        Input of the function

    Returns
    -------
    float
        Output of the function

    Notes
    -----
    This function is used in the likelihood calculation for the CMB-S4 r likelihood.
    """
    func = np.sign(x - 1.) * np.sqrt(2. * (x - np.log(x) - 1.))
    if np.any(np.isnan(func)):
        print(f"NaN for x:{x}, sign: {np.sign(x - 1.)}, logx: {np.log(x)}, x-logx-1: {(x - np.log(x) - 1.)}")
    return func

def compute_cov(Cl_arr):
    """
    Compute covariance matrix from a given array of Cl's.

    Parameters
    ----------
    Cl_arr : 2D numpy array
        Array of Cl values with shape (nsims, nbins)

    Returns
    -------
    cov : 2D numpy array
        Covariance matrix of the input Cl array
    """
    nbins = Cl_arr.shape[1]
    cov = np.zeros((nbins, nbins))

    for i in range(nbins):
        for j in range(i, i+2):
            if i == j: 
                cov[i,i] = np.var(Cl_arr[:,i])
            elif (j == i+1) and (i < nbins) and (j < nbins):
                cov[i,j] = cov[j,i] = np.cov(Cl_arr[:,i], Cl_arr[:,j])[0,1] 
    return cov 

def log_prior_uniform(par, min, max):
    """
    Compute the log prior probability for a parameter with a uniform distribution.

    Parameters
    ----------
    par : float
        The parameter value for which the log prior is calculated.
    min : float
        The minimum value of the parameter's prior distribution.
    max : float
        The maximum value of the parameter's prior distribution.

    Returns
    -------
    float
        The log prior probability. Returns 0. if the parameter is within the
        specified range [min, max), otherwise returns infinity.
    """
    if min <= par < max:
        return 0.
    else:
        return np.inf
    
def results_summary(chain, log_prob):
    # Maximum a-posteriori:
    """
    Compute and return MAP, MMSE and percentiles of a posterior distribution
    from a Markov Chain sample.

    Parameters
    ----------
    chain : array
        The Markov Chain samples, shape (n_samples, n_dim).
    log_prob : array
        The log probability of each sample, shape (n_samples).

    Returns
    -------
    theta_MAP : array
        The maximum a posteriori estimate of the parameters, shape (n_dim).
    theta_MMSE : array
        The minimum mean squared estimate of the parameters, shape (n_dim).
    theta_percentiles : array
        The 16th, 50th, 68th, 84th and 95th percentiles of the parameters, shape (n_dim, 5).
    """
    theta_MAP = chain[np.argmax(log_prob, axis=0)]
    # print(theta_MAP.shape)
    # Minimum Mean-Squared Error:
    theta_MMSE = np.mean(chain, axis=0)
    theta_percentiles = np.percentile(chain, [16, 50, 68, 84, 95], axis=0)

    return theta_MAP, theta_MMSE, theta_percentiles

def smoothed_histogram(samples, nbins=30, sigma=1., minmax=None):
    """
    Smooth a histogram of samples using a Gaussian filter and interpolate to a grid.

    Parameters
    ----------
    samples : array_like
        The samples to be used for the histogram.
    nbins : int, optional
        The number of bins to use in the histogram. Default is 30.
    sigma : float, optional
        The standard deviation of the Gaussian filter to use. Default is 1.
    minmax : list of float, optional
        The minimum and maximum values to use for the histogram. If not specified,
        the minimum and maximum values of the input array are used.

    Returns
    -------
    x_grid, spline : tuple
        The first element is the grid of x values, and the second element is the
        interpolated histogram.
    """
    if not isinstance(minmax, list):
        bmin = min(0., np.min(samples))
        bmax = np.max(samples)
    else:
        bmin = minmax[0]
        bmax = minmax[1]
    hist, bin_edges = np.histogram(samples, bins=nbins, range=(bmin, bmax), density=False)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.

    # Convolve the data with a gaussian filtering kernel
    smoothed_hist_data = gaussian_filter1d((bin_centers, hist), sigma=sigma)
    
    # Spline the filtered data and interpolate to a grid
    x_grid = np.linspace(bmin, bmax, num=200)
    knots = splrep(smoothed_hist_data[0], smoothed_hist_data[1], s=0)
    spline = splev(x_grid, knots, ext=0)

    # Normalize:
    spline /= np.trapz(spline, x=x_grid)
    
    return (x_grid, spline)

def smoothed_histogram2d(samples_x, samples_y, nbins=30, sigma=1., ncontours=2, sigma_levels=False, minmax_x=None, minmax_y=None):
    """
    Smooth a 2D histogram using a Gaussian filter and generate a set of
    contours for a given confidence level.

    Parameters
    ----------
    samples_x, samples_y : 1D arrays
        The samples to be histogrammed.
    nbins : int or list of int
        The number of bins to use in both the x and y directions. If a list,
        the first element specifies the number of bins in the x direction and
        the second element specifies the number of bins in the y direction.
    sigma : float
        The standard deviation of the Gaussian filter to use.
    ncontours : int
        The number of contours to generate. Contours are generated for the
        68%, 95%, and 99% confidence levels by default, but this can be
        changed by setting sigma_levels=True.
    sigma_levels : bool
        If True, generate contours for the 1, 2, and 3 sigma confidence
        levels rather than the 68%, 95%, and 99% confidence levels.
    minmax_x, minmax_y : list of float
        The minimum and maximum values to use for the x and y directions.
        If not specified, the minimum and maximum values of the input arrays
        are used.

    Returns
    -------
    bin_centers_x, bin_centers_y, smoothed_hist2d, contour_levels : tuple
        The first two elements are the bin centers in the x and y directions,
        the third element is the smoothed 2D histogram, and the fourth element
        is a list of the contour levels.
    """
    if not isinstance(minmax_x, list):
        bmin_x = min(0., np.min(samples_x))
        bmax_x = np.max(samples_x)
    else:
        bmin_x = minmax_x[0]
        bmax_x = minmax_x[1]

    if not isinstance(minmax_y, list):
        bmin_y = min(0., np.min(samples_y))
        bmax_y = np.max(samples_y)
    else:
        bmin_y = minmax_y[0]
        bmax_y = minmax_y[1]

     # Create 2d histogram
    hist2d, xedges, yedges = np.histogram2d(samples_x, samples_y, bins=nbins, range=[[bmin_x, bmax_x],[bmin_y, bmax_y]])
    # image extent, needed below for contour lines
    bin_centers_x = (xedges[1:] + xedges[:-1]) / 2.
    bin_centers_y = (yedges[1:] + yedges[:-1]) / 2.
    # Normalize
    hist2d = hist2d/np.sum(hist2d)

    # Cumulative 1d distribution
    histOrdered = np.sort(hist2d.flat)
    histCumulative = np.cumsum(histOrdered)

    # 2d contour levels: (68%, 95%, 99%) or sigma (39%, 86%, 99%)
    confLevels = (.3173, .0455, .0027)
    if sigma_levels:
        confLevels = (.6065, .1353, .0111)

    contour_levels = np.ones((ncontours+1,))

    if isinstance(nbins, (list, np.ndarray)):
        nBinsFlat = np.linspace(0., nbins[0]*nbins[1], nbins[0]*nbins[1])
    else:
        nBinsFlat = np.linspace(0., nbins**2, nbins**2)
    
    for i_contour in range(ncontours): 
        temp = np.interp(confLevels[i_contour], histCumulative, nBinsFlat)
        # Find "height" of contour level
        contour_levels[ncontours - 1 - i_contour] = np.interp(temp, nBinsFlat, histOrdered)

    smoothed_hist2d = gaussian_filter(hist2d.T, sigma=sigma)

    return (bin_centers_x, bin_centers_y, smoothed_hist2d, contour_levels)

def plot_contour(ax, xedges, yedges, hist2d, contour_levels, colors=coral, alpha=1., lw=0.1, do_filled=True, do_outlined=True):
    """
    Plot a 2D histogram as contours on a given axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis on which to plot the contours.
    xedges, yedges : array_like
        Edges of the 2D binning.
    hist2d : array_like
        2D histogram data.
    contour_levels : array_like
        Levels of the contours to plot.
    colors : array_like or str, optional
        Colors to use for the contours. Default is ``coral``.
    alpha : float, optional
        Transparency of the contours. Default is 1.
    lw : float, optional
        Line width of the contours. Default is 0.1.
    do_filled : bool, optional
        If ``True``, fill the contours. Default is ``True``.
    do_outlined : bool, optional
        If ``True``, draw the contours as outlines. Default is ``True``.

    Returns
    -------
    ax : matplotlib axis
        Modified axis with the contours plotted.
    """
    if not ( do_filled and do_outlined): print("WARNING: Both countour outline and fill are set ti false. There will be not contours!")
    if do_filled  : ax.contourf(xedges, yedges, hist2d, contour_levels, colors=colors, cmap=None, alpha=alpha)
    if do_outlined: ax.contour( xedges, yedges, hist2d, contour_levels, colors=colors, linewidths=lw, ls='-')

    return ax

class s4like:
    def __init__(self, lmax, Cl_est, nmt_bins, Cl_arr=None, Cl_f=None, mc_idx=None, Nl=None, beam=None, bin_cuts=None, prior_dict=default_prior): 
        """
        Initialize a s4like object.

        Parameters
        ----------
        lmax : int
            maximum multipole
        Cl_est : array_like
            estimated power spectrum. (Power spectrum to fit.)
        nmt_bins : NmtBin object
            NmtBin object used to bin the power spectrum
        Cl_arr : array_like, optional
            array of power spectra used to compute the covariance matrix. 
            If not provided, the covariance matrix is computed from the Knox formula.
        Cl_f : array_like, optional
            fiducial power spectrum. Only needed for HL likelihood.
        mc_idx : int, optional
            index of the power spectrum to be used as the 'fiducial'. 
            Only needed when computing the covariance matrix from simulations, and 
            leaves the current simulation out.
        Nl : array_like, optional
            power spectrum of the noise. (Can include noise frorm lensing template 
            or overall 'noise' term including lensing)
        beam : array_like, optional
            beam window function. Only needed when fitting power spectra that is not debeamed.
        bin_cuts : array_like, optional
            bin cuts to be applied to the power spectrum. 
            Provided as [low_cut_bin, high_cut_bin] list.
        prior_dict : dict, optional
            dictionary containing the prior parameters

        Notes
        -----
        If Cl_arr is provided, the covariance matrix is computed from it. If Cl_arr is not provided, the covariance matrix is computed from the Knox formula.
        If Cl_f is not provided, it is set to 0.
        If Nl is not provided, it is set to 0.
        If beam is not provided, it is set to 1.
        If bin_cuts is not provided, the entire power spectrum is used.
        If prior_dict is not provided, the default prior parameters are used.
        """
        self.Cl_hat = Cl_est

        if isinstance(Cl_f, np.ndarray): 
            self.Cl_fid = Cl_f
        else:
            self.Cl_fid = 0.

        if isinstance(Nl, np.ndarray): 
            self.Nl = Nl
        else:
            self.Nl = 0.

        if isinstance(Cl_arr, np.ndarray):
            if mc_idx is not None:
                Cl4cov = np.delete(Cl_arr, mc_idx, axis=0) - self.Cl_fid
            else:
                Cl4cov = Cl_arr - self.Cl_fid
            
            cov = compute_cov(Cl4cov)
        else:
            # Define covariance matrix from Knox formula
            pass

        self.cov_mat_inv = np.linalg.inv(cov)
        self.cov_mat_det = np.linalg.det(cov)

        ells = np.arange(lmax+1)
        Dell_factor = ells * (ells + 1) / 2. / np.pi 
        
        Cl_lens = cw.get_lensed_scalar(lmax=lmax)[:,2]
        Cl_tens = cw.get_tensor(lmax=lmax)[:,2]

        if isinstance(beam, np.ndarray): 
            Cl_lens *= beam[:lmax+1]**2.
            Cl_tens *= beam[:lmax+1]**2.

        Cl_lens[2:] /= Dell_factor[2:] 
        Cl_tens[2:] /= Dell_factor[2:] 

        if isinstance(bin_cuts, (np.ndarray, list)): 
            self.Cl_lens = nmt_bins.bin_cell(Cl_lens)[bin_cuts[0]:bin_cuts[1]]
            self.Cl_tens = nmt_bins.bin_cell(Cl_tens)[bin_cuts[0]:bin_cuts[1]]
        else:
            self.Cl_lens = nmt_bins.bin_cell(Cl_lens)
            self.Cl_tens = nmt_bins.bin_cell(Cl_tens)

        self.prior_params = prior_dict

        self.rng = np.random.default_rng()

        self.params = []
        self.par_tex = []
        for par in list(self.prior_params.keys()):
            if self.prior_params[par]['fix'] == None:
                self.params.append(par)
                self.par_tex.append(self.prior_params[par]['latex'])
        # print(self.params)

    def log_prior(self, r, A_lens=0.05):
        """
        Calculate the log prior for the given parameters `r` and `A_lens`.

        The function computes the log prior probability based on the uniform 
        distribution for the parameters `r` and `A_lens` according to the 
        specified prior ranges in `self.prior_params`. If a parameter is fixed, 
        it is not included in the computation.

        Parameters
        ----------
        r : float
            The tensor-to-scalar ratio parameter.
        A_lens : float, optional
            The lensing amplitude parameter, default is 0.05.

        Returns
        -------
        float
            The negative log prior value.
        """
        lnp = 0.
        if 'r' in self.prior_params:
            if self.prior_params['r']['fix'] == None: lnp += log_prior_uniform(r, self.prior_params['r']['min'], self.prior_params['r']['max'])
        if 'A_lens' in self.prior_params:
            if self.prior_params['A_lens']['fix'] == None: lnp += log_prior_uniform(A_lens, self.prior_params['A_lens']['min'], self.prior_params['A_lens']['max'])

        return -lnp
               
    # def HL_likelihood(self, theta):
    #     if (self.prior_params['r']['fix'] == None) and (self.prior_params['A_lens']['fix'] == None):
    #         r, A_lens = theta
        
    #     if self.prior_params['r']['fix'] != None: r = self.prior_params['r']['fix']; A_lens = theta
    #     if self.prior_params['A_lens']['fix'] != None: A_lens = self.prior_params['A_lens']['fix']; r = theta

        
    #     logPrior = self.log_prior(r, A_lens)

    #     if not np.isfinite(logPrior):
    #         return - np.inf
        
    #     Cl = r * self.Cl_tens + A_lens * self.Cl_lens

    #     g_Cl = np.reshape(g_func(self.Cl_hat / Cl) * self.Cl_fid,(len(Cl), 1))
        
    #     logLike = -0.5 * np.matmul(g_Cl.T, np.matmul(self.cov_mat_inv, g_Cl))

    #     if np.isnan(logLike):
    #         print(f"Is NaN for r={r} and A_lens={A_lens}. NaN in gCl? {np.any(np.isnan(g_Cl))}")
    #     return logLike + logPrior
    def HL_likelihood(self, theta):
        """
        Computes the Hamimeche-Lewis likelihood for given parameters.

        Parameters
        ----------
        theta : array-like
            A tuple or list containing the parameters 'r' (tensor-to-scalar ratio) 
            and 'A_lens' (lensing amplitude) to be evaluated. If 'r' or 'A_lens' 
            have fixed values in prior_params, they will be set accordingly.

        Returns
        -------
        float
            The log-likelihood value. If the log prior is not finite or the data-to-model 
            ratio is negative, returns negative infinity. If log-likelihood calculation 
            results in NaN, it prints a warning message for the given parameters.
        """
        if (self.prior_params['r']['fix'] == None) and (self.prior_params['A_lens']['fix'] == None):
            r, A_lens = theta
        
        if self.prior_params['r']['fix'] != None: r = self.prior_params['r']['fix']; A_lens = theta
        if self.prior_params['A_lens']['fix'] != None: A_lens = self.prior_params['A_lens']['fix']; r = theta

        
        logPrior = self.log_prior(r, A_lens)

        if not np.isfinite(logPrior):
            return - np.inf
        
        Cl = r * self.Cl_tens + A_lens * self.Cl_lens + self.Nl

        data2model_ratio = self.Cl_hat / Cl

        if np.any(data2model_ratio < 0.):
            return -np.inf
        

        g_Cl = np.reshape(g_func(data2model_ratio) * self.Cl_fid,(len(Cl), 1))
        
        logLike = -0.5 * np.matmul(g_Cl.T, np.matmul(self.cov_mat_inv, g_Cl))

        if np.isnan(logLike):
            print(f"Is NaN for r={r} and A_lens={A_lens}. NaN in gCl? {np.any(np.isnan(g_Cl))}")
        return logLike + logPrior
    
    def gaussian_likelihood(self, theta):
        """
        Calculate the Gaussian log-likelihood for given parameters `r` and `A_lens`.

        This function computes the log-likelihood based on a Gaussian model for 
        the parameters `r` (tensor-to-scalar ratio) and `A_lens` (lensing amplitude),
        using the estimated power spectrum and the covariance matrix. The parameters 
        are adjusted according to their fixed values specified in `prior_params`.

        Parameters
        ----------
        theta : tuple or float
            A tuple containing the parameters `r` and `A_lens` if neither is fixed. 
            Otherwise, it is a float representing the unfixed parameter.

        Returns
        -------
        float
            The log-likelihood value, incorporating the log prior. If the log prior 
            is not finite, returns negative infinity. If log-likelihood calculation 
            results in NaN, it prints the value of `r`.
        """
        if (self.prior_params['r']['fix'] == None) and (self.prior_params['A_lens']['fix'] == None):
            r, A_lens = theta
        if self.prior_params['r']['fix'] != None: r = self.prior_params['r']['fix']; A_lens = theta
        if self.prior_params['A_lens']['fix'] != None: A_lens = self.prior_params['A_lens']['fix']; r = theta

        logPrior = self.log_prior(r, A_lens)

        if not np.isfinite(logPrior):
            return - np.inf
        
        Cl = r * self.Cl_tens + A_lens * self.Cl_lens + self.Nl

        d_Cl = self.Cl_hat - Cl
        
        logLike = -0.5 * (np.matmul(d_Cl.T, np.matmul(self.cov_mat_inv, d_Cl)) + np.log(self.cov_mat_det))

        if np.isnan(logLike):
            print(r)
        return logLike + logPrior
    
    def run_sampler(self, likelihood='HL', nsamples=5000, progress=False):
        """
        Runs the MCMC sampler to estimate the parameters of the model using the chosen likelihood.
        The MCMC sampling is based on emcee implementation based on the affine-invariant ensemble MCMC algorithm.
        
        Parameters
        ----------
        likelihood : str, optional
            The likelihood function to use. Options are 'HL' (Hamiltonian likelihood)
            and 'gauss' (Gaussian likelihood). Default is 'HL'.
        nsamples : int, optional
            The number of samples to draw from the posterior distribution. Default is 5000.
        progress : bool, optional
            If True, prints the progress of the sampler. Default is False.
        
        Attributes
        ----------
        chain : array_like
            The MCMC chain containing the samples from the posterior distribution.
        results : dict
            A dictionary containing the results of the estimation, including the MAP estimate,
            MMSE estimate, and percentiles.
        """
        nwalkers = nthreads 

        params_init = []
        for par in self.params:
            # print(par, type(self.prior_params[par]['min']),type(self.prior_params[par]['max']), type(nwalkers))
            if par == 'r': 
                params_init.append(self.rng.uniform(np.max([self.prior_params[par]['min'], 0.]), np.min([self.prior_params[par]['max'], 0.01]), nwalkers))
            else:
                params_init.append(self.rng.uniform(self.prior_params[par]['min'], self.prior_params[par]['max'], nwalkers))
        
        params_init = np.array(params_init).T
        par_dim = len(self.params)

        if likelihood == 'HL': sampler = mc.EnsembleSampler(nwalkers, par_dim, self.HL_likelihood)
        if likelihood == 'gauss': sampler = mc.EnsembleSampler(nwalkers, par_dim, self.gaussian_likelihood)

        # print(params_init.shape, nsamples, par_dim, nwalkers)
        sampler.run_mcmc(params_init, nsamples, progress=progress)
        
        log_prob = sampler.get_log_prob(discard=300, thin=10, flat=True)
        self.chain = sampler.get_chain(discard=300, thin=10, flat=True)

        MAP_est, MMSE_est, percentiles = results_summary(self.chain, log_prob)

        self.results = {}
        for i, par in enumerate(self.params):
            self.results[par] = {'MAP': MAP_est[i], 'MMSE': MMSE_est[i], 'percentiles': percentiles[:,i]}

    def create_corner(self, fileout, truth='result', truth_labels=None, par_range='priors'):
        """
        Creates a corner plot with the posterior distribution from the MCMC chain
        
        Parameters
        ----------
        fileout : str
            The output filename for the corner plot
        truth : str, optional
            If 'result', uses the MAP and MMSE estimates from the MCMC chain as the truth values.
            If None, does not plot any truth values. Default is 'result'.
        truth_labels : list, optional
            List of labels for the truth values. Default is ['MAP', 'MMSE'].
        par_range : str or list, optional
            If 'priors', uses the prior ranges for the parameters. If a list, uses the given ranges.
            Default is 'priors'.
        """
        plt.rcParams['figure.dpi'] = 300

        if truth == None: truth2d = None; truth_labels = None

        if truth == 'result':
            truth_labels = ['MAP', 'MMSE']

            truth2d = []
            for label in truth_labels:
                truth_j = []
                for i,  par in enumerate(self.params):
                    truth_j.append(self.results[par][label])
                truth2d.append(truth_j)

            truth2d = np.array(truth2d)
            # print(truth2d.shape)

        if par_range == 'priors':
            par_range = []
            for par in self.params:
                par_range.append([self.prior_params[par]['min'], self.prior_params[par]['max']]) 
    

        gtc.plotGTC(chains=[self.chain],
            paramNames=self.par_tex, 
            truths=truth2d,
            truthLabels=truth_labels,
            paramRanges=par_range,
            figureSize=3.5,
            filledPlots=True,
            customLabelFont={'family':'sans-serif', 'size':10.},
            customLegendFont={'family':'sans-serif', 'size':9.},
            customTickFont={'family':'sans-serif', 'size':7.},
            plotName=fileout)
