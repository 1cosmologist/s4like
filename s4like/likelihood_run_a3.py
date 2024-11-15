
import numpy as np 
import healpy as hp
import matplotlib.pyplot as plt
from mpi4py import MPI
import skytools as st
import camb_wrapper as cw
import os 
import sys
import pymaster as nmt
import s4like as lik
import sys

import warnings
warnings.filterwarnings('ignore')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
totProc = comm.Get_size()

np.set_printoptions(threshold=sys.maxsize)

if rank == 0: print(f'Total processes: {totProc}')

nside = 512
lmax_o = 250
nsims = 50
nbins = 50
lmax_low_cut = 30
lmax_high_cut = 250
nticks = 9

noisy = True
years_of_survey = 20.
Alens = 0.08
# fg_comps = ['d10', 's5']#, 's4'], 'co3']#]
fg_comps = ['d12', 's7', 'a2']
optics_mode = 'mixed'   # wide or split or mixed

r_fid = 0.
Alens_fid = Alens

priors = {
    'r' : {
        'fix'  : None, 
        'min'  : -0.01,
        'max'  : 0.01,
        'latex': r'$r$'
    },
    'A_lens' : {
        'fix'  : Alens_fid,      # None or Alens_fid
        'min'  : 0.04,
        'max'  : 0.16,
        'latex': r'$A_{\rm lens}$'
    }
}
use_like = 'HL' # 'HL' or 'gauss
nsamples = 2500
model_Nl = True
save_corner = False

skytools_data = os.environ['SKYTOOLS_DATA']

output_path = f'../output/{optics_mode}/fg_{fg_comps[0]}-{fg_comps[1]}/'
os.makedirs(f'{output_path}corner/', exist_ok=True)


casefix = f'a3sat-{optics_mode}-AL{Alens}-{years_of_survey}yr-'
if isinstance(fg_comps, list):
    for fg_i in fg_comps:
        casefix = casefix+fg_i
if not noisy: casefix = casefix+'_noiseless'
# casefix = f'{casefix}-field4'

case_title = f"{optics_mode} bands; {fg_comps[0]}+{fg_comps[1]}; {years_of_survey} yr survey"


def divide2procs(num2divide):
    if not type(num2divide) in [int, np.int8, np.int16, np.int32, np.int64]:
        print("ERROR: Cannot divide a non-integer number")

    slab_min = num2divide // totProc      # Q
    iter_remains = np.mod(num2divide, totProc)    # R 

    #  SZ = R x (Q + 1) + (P-R) x Q 
    slab_per_Proc = np.zeros((totProc,), dtype=np.int64)  # P = len(zslab_per_Proc)

    slab_per_Proc[0:int(iter_remains)] = slab_min + 1     # R procs together get (Q+1)xR z slabs
    slab_per_Proc[int(iter_remains):]  = slab_min 

    return slab_per_Proc

# Run MCMC for each sample
sim4proc = divide2procs(nsims)

bin_nmt_cl = nmt.NmtBin.from_lmax_linear(lmax_o, is_Dell=False, nlb=31)
leff = bin_nmt_cl.get_effective_ells()
low_cut_bin = np.min(np.where(leff >= lmax_low_cut))
high_cut_bin = np.max(np.where(leff <= lmax_high_cut))+1
# print(low_cut_bin, high_cut_bin, leff[low_cut_bin], leff[high_cut_bin-1])

mc_loc = np.empty((sim4proc[rank],), dtype=np.int32)
mc_idx = None
if rank == 0: mc_idx = np.arange(nsims, dtype=np.int32)
comm.Scatterv([mc_idx, sim4proc, MPI.INT32_T], mc_loc, root=0)

# if rank != 0: Cl_nmt = None; Nl_nmt = None
# Cl_nmt = comm.bcast(Cl_nmt, root=0)
# Nl_nmt = comm.bcast(Nl_nmt, root=0)
Cl_nmt = np.load(f'{output_path}spectra/Cl_nmt_Nl_nmt_{casefix}.npz')['Cl_nmt'][:, low_cut_bin:high_cut_bin]
Nl_nmt = np.load(f'{output_path}spectra/Cl_nmt_Nl_nmt_{casefix}.npz')['Nl_nmt'][:, low_cut_bin:high_cut_bin]
if model_Nl: Nl_nmt = np.mean(Nl_nmt, axis=0)

ells = np.arange(lmax_o+1) 
Dell_factor = ells * (ells + 1.) / 2. / np.pi
Cl_fid = cw.get_total(lmax=lmax_o, r=r_fid, Alens=Alens_fid)[:,2]
Cl_fid[2:] /= Dell_factor[2:]
Cl_fid = bin_nmt_cl.bin_cell(Cl_fid)[low_cut_bin:high_cut_bin]

if model_Nl: 
    casefix = f'{casefix}_wNlmd'
    Cl_fid += Nl_nmt
else:
    casefix = f'{casefix}_wNldb'

if priors['A_lens']['fix'] == None: casefix = f'{casefix}-fitAL'

npctl = 5
r_mmse_loc = np.zeros((sim4proc[rank],), dtype=np.float64)
r_map_loc  = np.zeros((sim4proc[rank],), dtype=np.float64)
r_pctl_loc = np.zeros((sim4proc[rank], npctl), dtype=np.float64)
Alens_pctl_loc = np.zeros((sim4proc[rank], npctl), dtype=np.float64)
if priors['A_lens']['fix'] == None: Alens_mmse_loc = np.zeros((sim4proc[rank],), dtype=np.float64)
for i_mc, mc in enumerate(mc_loc):
    print(f'Rank={rank}: Running {mc} of {nsims}...'); sys.stdout.flush()
    if model_Nl:
        Cl_est = Cl_nmt[mc]
    else:
        Nl_nmt_mc = np.delete(Nl_nmt, mc, axis=0)
        Cl_est = Cl_nmt[mc] - np.mean(Nl_nmt_mc, axis=0)

    if model_Nl:
        likelihood = lik.s4like(lmax_o, Cl_est, Cl_nmt, Cl_fid, mc, bin_nmt_cl, bin_cuts=[low_cut_bin, high_cut_bin], Nl=Nl_nmt, beam=None, prior_dict=priors)
    else:
        likelihood = lik.s4like(lmax_o, Cl_est, Cl_nmt, Cl_fid, mc, bin_nmt_cl, bin_cuts=[low_cut_bin, high_cut_bin], Nl=None, beam=None, prior_dict=priors)
    likelihood.run_sampler(likelihood=use_like, nsamples=nsamples, progress=False)
    if save_corner: likelihood.create_corner(f'{output_path}corner/corner_{casefix}_mc{mc:{0}>4}.png')

    if mc == mc_loc[0]: chains_loc = np.zeros((sim4proc[rank], likelihood.chain.shape[0], likelihood.chain.shape[1]), dtype=np.float64)
        
    r_mmse_loc[i_mc] = likelihood.results['r']['MMSE']
    r_map_loc[i_mc]  = likelihood.results['r']['MAP']
    r_pctl_loc[i_mc] = np.array(likelihood.results['r']['percentiles'])
    if priors['A_lens']['fix'] == None: 
        Alens_mmse_loc[i_mc] = likelihood.results['A_lens']['MMSE']
        Alens_pctl_loc[i_mc] = np.array(likelihood.results['A_lens']['percentiles'])
    
    chains_loc[i_mc] = np.array(likelihood.chain, dtype=np.float64)

comm.barrier()

chain_shape = chains_loc.shape
# print(chain_shape)

r_mmse = None; r_map = None; r_pctl = None; Alens_mmse = None; Alens_pctl = None;# chains = None
chains = np.empty((nsims, chain_shape[1], chain_shape[2]), dtype=np.float64)
if rank == 0:
    r_mmse = np.empty((nsims,), dtype=np.float64)
    r_map  = np.empty((nsims,), dtype=np.float64)
    r_pctl = np.empty((nsims, npctl), dtype=np.float64)
    # chains = np.empty((nsims, chain_shape[1], chain_shape[2]), dtype=np.float64)
    if priors['A_lens']['fix'] == None: 
        Alens_mmse = np.empty((nsims,), dtype=np.float64)
        Alens_pctl = np.empty((nsims, npctl), dtype=np.float64)


comm.Gatherv([r_mmse_loc, sim4proc[rank],   MPI.DOUBLE], r_mmse, root=0)
comm.Gatherv([r_map_loc , sim4proc[rank],   MPI.DOUBLE], r_map , root=0)
comm.Gatherv([r_pctl_loc, MPI.DOUBLE], [r_pctl, sim4proc*npctl, MPI.DOUBLE], root=0)  
                               #                                
if priors['A_lens']['fix'] == None: 
    comm.Gatherv([Alens_mmse_loc, sim4proc[rank], MPI.DOUBLE], Alens_mmse, root=0)
    comm.Gatherv([Alens_pctl_loc, MPI.DOUBLE], [Alens_pctl, sim4proc*npctl, MPI.DOUBLE], root=0)

comm.Gatherv([chains_loc, sim4proc[rank]*chain_shape[1]*chain_shape[2], MPI.DOUBLE], [chains, sim4proc*chain_shape[1]*chain_shape[2], MPI.DOUBLE], root=0)

if rank == 0:
    print(f"MMSE r estimate: mean={np.mean(r_mmse)}, median={np.median(r_mmse)}, std={np.std(r_mmse)}"); sys.stdout.flush()
    print(f"MAP r estimate: mean={np.mean(r_map)}, median={np.median(r_map)}, std={np.std(r_map)}"); sys.stdout.flush()
    print(f"r {[16, 50, 68, 84, 95]} percentile estimates: mean={np.mean(r_pctl, axis=0)}, median={np.median(r_pctl, axis=0)}, std={np.std(r_pctl, axis=0)}"); sys.stdout.flush()
    print(f"MMSE r estimate from all chains: {np.mean(chains[:,:,0])}")
    if priors['A_lens']['fix'] == None:
        print(f"MMSE A_lens estimate: mean={np.mean(Alens_mmse)}, median={np.median(Alens_mmse)}, std={np.std(Alens_mmse)}"); sys.stdout.flush()
        print(f"A_lens {[16, 50, 68, 84, 95]} percentile estimates: mean={np.mean(Alens_pctl, axis=0)}, median={np.median(Alens_pctl, axis=0)}, std={np.std(Alens_pctl, axis=0)}"); sys.stdout.flush()

if rank == 0:
    hist_data_arr = []
    plt.figure(figsize=(3., 3.5), dpi=300)
    # fno, fig, ax = cf.make_plotaxes(res='print', shape='sq')
    for i in range(nsims):
        # print(i)
        hist_data = lik.smoothed_histogram(chains[i,:,0], nbins=nbins, sigma=1.4, minmax=[priors['r']['min'], priors['r']['max']])
        plt.plot(hist_data[0] * 1e3, hist_data[1]/np.max(hist_data[1]), '-', c='C1', lw=0.5, alpha=0.7)
        hist_data_arr.append(hist_data)

        del hist_data 
    
    hist_data = np.mean(np.array(hist_data_arr), axis=0)
    r_map_meandist = hist_data[0, np.argmax(hist_data[1]/np.max(hist_data[1]))]
    print(f"MAP r estimate from all chains: {r_map_meandist}")
    plt.plot(hist_data[0] * 1e3, hist_data[1]/np.max(hist_data[1]), 'k-', lw=1., alpha=0.7)


    plt.axvline(x=np.mean(r_mmse) * 1e3, ls='--', c='k', lw=0.7, alpha=0.7, label='Mean MMSE')
    plt.axvline(x=np.mean(r_map_meandist) * 1e3,  ls='-',  c='k', lw=0.7, alpha=0.7, label='Mean MAP')

    plt.ylabel(r'$P/P_{\rm peak}$')
    plt.xlabel(r'$r$ ($\times 10^{-3}$)')
    plt.xlim(priors['r']['min'] * 1e3, priors['r']['max'] * 1e3)
    plt.ylim(0.,1.1)
    plt.yticks(ticks=[0., 0.5, 1.0], fontsize='x-small')
    plt.title(f"{case_title} \n $r_{{MMSE}} = $ {np.mean(r_mmse)*1.e3:.3}x$10^{{-3}}$, $\sigma(r) = $ +{(np.mean(r_pctl, axis=0)[3] - np.mean(r_mmse))*1.e3:.2}x$10^{{-3}}$ -{(np.mean(r_mmse) - np.mean(r_pctl, axis=0)[0])*1.e3:.2}x$ 10^{{-3}}$", fontdict={'fontsize':'x-small'})
    tick_labels = list(np.linspace(priors['r']['min'], priors['r']['max'], nticks) * 1e3)
    plt.xticks(ticks=tick_labels, fontsize='x-small') # [0., 0.002, 0.004, 0.006, 0.008, 0.010]
    plt.legend(loc='best', frameon=False, fontsize='small')
    plt.savefig(f'{output_path}corner/joint_r_posterior_{casefix}.png', bbox_inches='tight', pad_inches=0.1)

    if priors['A_lens']['fix'] == None:
        hist_data_arr = []
        plt.figure(figsize=(3., 3.5), dpi=300)
        # fno, fig, ax = cf.make_plotaxes(res='print', shape='sq')
        for i in range(nsims):
            # print(i)
            hist_data = lik.smoothed_histogram(chains[i,:,1], nbins=nbins, sigma=1.4, minmax=[0.04, 0.16])
            plt.plot(hist_data[0], hist_data[1]/np.max(hist_data[1]), '-', c='C1', lw=0.5, alpha=0.7)
            hist_data_arr.append(hist_data)

            del hist_data 
        
        hist_data = np.mean(np.array(hist_data_arr), axis=0)
        plt.plot(hist_data[0], hist_data[1]/np.max(hist_data[1]), 'k-', lw=1., alpha=0.7)


        plt.axvline(x=np.mean(Alens_mmse), ls='--', c='k', lw=0.7, alpha=0.7, label='Mean MMSE')

        plt.ylabel(r'$P/P_{\rm peak}$')
        plt.xlabel(r'$A_{\rm lens}$')
        plt.xlim(priors['A_lens']['min'], priors['A_lens']['max'])
        plt.ylim(0.,1.1)
        plt.yticks(ticks=[0., 0.5, 1.0], fontsize='small')
        plt.title(f"{case_title} \n $A_{{lens, MMSE}} = $ {np.mean(Alens_mmse):.3}, $\sigma(A_{{lens}}) = $ +{np.mean(Alens_pctl, axis=0)[3] - np.mean(Alens_mmse):.2} -{np.mean(Alens_mmse) - np.mean(Alens_pctl, axis=0)[0]:.2}", fontdict={'fontsize':'x-small'})
        plt.xticks(ticks=[0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16], fontsize='small')
        plt.legend(loc='best', frameon=False, fontsize='small')
        plt.savefig(f'{output_path}corner/joint_Alens_posterior_{casefix}.png', bbox_inches='tight', pad_inches=0.1)

        fig, ax = plt.subplots(figsize=(4., 4.3), dpi=300)
        xedges, yedges, hist2d, contourlevels = lik.smoothed_histogram2d(chains[:,:,0].flatten(), chains[:,:,1].flatten(), nbins=nbins, minmax_x=[0., 0.01], minmax_y=[0.04, 0.16])
        ax = lik.plot_contour(ax, xedges, yedges, hist2d, contourlevels, colors=lik.orange, alpha=0.65, lw=0.5)
        ax.axhline(y=np.mean(Alens_mmse), ls='--', c='k', lw=0.7, alpha=0.7, label='Mean MMSE')
        ax.axvline(x=np.mean(r_mmse), ls='--', c='k', lw=0.7, alpha=0.7, label=None)
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r'$A_{\rm lens}$')
        ax.set_ylim(priors['A_lens']['min'], priors['A_lens']['max'])
        ax.set_xlim(priors['r']['min'], priors['r']['max'])
        ax.set_yticks(ticks=[0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16])
        ax.set_xticks(ticks=[0., 0.002, 0.004, 0.006, 0.008,0.010])
        ax.legend(loc='best', frameon=False, fontsize='small')
        ax.set_title(f"{case_title}", fontsize='small')
        plt.savefig(f'{output_path}corner/joint_r-Alens_contour_{casefix}.png', bbox_inches='tight', pad_inches=0.1)