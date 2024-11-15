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
    func = np.sign(x - 1.) * np.sqrt(2. * (x - np.log(x) - 1.))
    if np.any(np.isnan(func)):
        print(f"NaN for x:{x}, sign: {np.sign(x - 1.)}, logx: {np.log(x)}, x-logx-1: {(x - np.log(x) - 1.)}")
    return func

def compute_cov(Cl_arr):
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
    if min <= par < max:
        return 0.
    else:
        return np.inf
    
def results_summary(chain, log_prob):
    # Maximum a-posteriori:
    theta_MAP = chain[np.argmax(log_prob, axis=0)]
    # print(theta_MAP.shape)
    # Minimum Mean-Squared Error:
    theta_MMSE = np.mean(chain, axis=0)
    theta_percentiles = np.percentile(chain, [16, 50, 68, 84, 95], axis=0)

    return theta_MAP, theta_MMSE, theta_percentiles

def smoothed_histogram(samples, nbins=30, sigma=1., minmax=None):
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
    if not ( do_filled and do_outlined): print("WARNING: Both countour outline and fill are set ti false. There will be not contours!")
    if do_filled  : ax.contourf(xedges, yedges, hist2d, contour_levels, colors=colors, cmap=None, alpha=alpha)
    if do_outlined: ax.contour( xedges, yedges, hist2d, contour_levels, colors=colors, linewidths=lw, ls='-')

    return ax

class s4like:
    def __init__(self, lmax, Cl_est, Cl_arr, Cl_f, mc_idx, nmt_bins, Nl=None, beam=None, bin_cuts=None, prior_dict=default_prior): 
        self.Cl_hat = Cl_est
        self.Cl_fid = Cl_f
        if isinstance(Nl, np.ndarray): 
            self.Nl = Nl
        else:
            self.Nl = 0.
        Cl4cov = np.delete(Cl_arr, mc_idx, axis=0) - Cl_f

        cov = compute_cov(Cl4cov)
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
        if (self.prior_params['r']['fix'] == None) and (self.prior_params['A_lens']['fix'] == None):
            r, A_lens = theta
        if self.prior_params['r']['fix'] != None: r = self.prior_params['r']['fix']; A_lens = theta
        if self.prior_params['A_lens']['fix'] != None: A_lens = self.prior_params['A_lens']['fix']; r = theta

        logPrior = self.log_prior(r, A_lens)

        if not np.isfinite(logPrior):
            return - np.inf
        
        Cl = r * self.Cl_tens + A_lens * self.Cl_lens

        d_Cl = self.Cl_hat - Cl
        
        logLike = -0.5 * (np.matmul(d_Cl.T, np.matmul(self.cov_mat_inv, d_Cl)) + np.log(self.cov_mat_det))

        if np.isnan(logLike):
            print(r)
        return logLike + logPrior
    
    def run_sampler(self, likelihood='HL', nsamples=5000, progress=False):
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
