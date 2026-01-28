import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gstatsMCMC import Topography
# from gstatsMCMC import MCMC
from gstatsMCMC import MCMC
import gstatsim as gs
from sklearn.preprocessing import QuantileTransformer
import skgstat as skg
import scipy as sp
from copy import deepcopy
import time
import multiprocessing as mp
from pathlib import Path


def largeScaleChain_mp(n_chains, n_workers, largeScaleChain, rf, initial_beds, rng_seeds, n_iters):
    '''
    function to run multiple large scale chain using multiprocessing

    Parameters
    ----------
    n_chains (int): the number of chains this multiprocessing choose to run
    n_workers (int): the number of processes the multiprocessing function create. Must be less than the number of CPUs you have
    largeScaleChain (MCMC.chain_crf): an existing large scale chain that has already been set-up
    rf (MCMC.RandField): an existing RandField instance that has already been set-up
    initial_beds (list): a list of subglacial topography (each of which are a 2D numpy array) used to initialize each chain
    rng_seeds (list): a list of int used to initialize the random number generator of each chain
    n_iters (int): a list of number of iterations runned for each chain

    Returns
    -------
    result: a list of results from all the chains runned.

    '''

    tic = time.time()

    params = []
    # retrive parameters from the existing chain / RandField
    example_chain = largeScaleChain.__dict__
    example_RF = rf.__dict__
    run_param = {}  # a dictionary of parameters passed in the run() function

   # modify some of the parameters based on the input rng_seeds, initial_beds, and n_iters
    for i in range(n_chains):
        chain_param = deepcopy(example_chain)
        chain_param['rng_seed'] = rng_seeds[i]
        chain_param['initial_bed'] = initial_beds[i]
        run_param['n_iter'] = n_iters[i]
        # some display parameters are fixed.
        run_param['only_save_last_bed'] = True
        run_param['info_per_iter'] = 1000
        run_param['plot'] = False
        run_param['progress_bar'] = False
        params.append([deepcopy(chain_param), deepcopy(
            example_RF), deepcopy(run_param)])

    # the multiprocessing step
    with mp.Pool(n_workers) as pool:
        result = pool.starmap(lsc_run_wrapper, params)

    for i, r in enumerate(result):
        beds, loss_mc, loss_data, loss, steps, resampled_times, blocks_used = r
        # TODO: code for save data

    toc = time.time()
    print(f'{toc-tic} seconds')

    return result


def lsc_run_wrapper(param_chain, param_rf, param_run):
    # a function used to initialize chain by input parameters and run the chains

    chain = MCMC.init_lsc_chain_by_instance(param_chain)
    rf1 = MCMC.initiate_RF_by_instance(param_rf)
    result = chain.run(n_iter=param_run['n_iter'], RF=rf1, only_save_last_bed=param_run['only_save_last_bed'],
                       info_per_iter=param_run['info_per_iter'], plot=param_run['plot'], progress_bar=param_run['progress_bar'])

    return result


def smallScaleChain_mp(n_chains, n_workers, smallScaleChain, initial_beds, rng_seeds, n_iters):
    '''
    function to run multiple small scale chain using multiprocessing

    Parameters
    ----------
    n_chains (int): the number of chains this multiprocessing choose to run
    n_workers (int): the number of processes the multiprocessing function create. Must be less than the number of CPUs you have
    smallScaleChain (MCMC.chain_sgs): an existing small scale chain that has already been set-up
    initial_beds (list): a list of subglacial topography (each of which are a 2D numpy array) used to initialize each chain
    rng_seeds (list): a list of int used to initialize the random number generator of each chain
    n_iters (int): a list of number of iterations runned for each chain

    Returns
    -------
    result: a list of results from all the chains runned.

    '''

    tic = time.time()

    params = []
    # retrive parameters from the existing chain
    example_chain = smallScaleChain.__dict__
    run_param = {}

    # modify some of the parameters based on the input rng_seeds, initial_beds, and n_iters
    for i in range(n_chains):
        chain_param = deepcopy(example_chain)
        chain_param['rng_seed'] = rng_seeds[i]
        chain_param['initial_bed'] = initial_beds[i]
        run_param['n_iter'] = n_iters[i]
        # some display parameters are fixed.
        run_param['only_save_last_bed'] = True
        run_param['info_per_iter'] = 10
        run_param['plot'] = False
        run_param['progress_bar'] = False
        params.append([deepcopy(chain_param), deepcopy(run_param)])

    # the multiprocessing step
    with mp.Pool(n_workers) as pool:
        result = pool.starmap(msc_run_wrapper, params)

    for i, r in enumerate(result):
        beds, loss_mc, loss_data, loss, steps, resampled_times, blocks_used = r
        # TODO: code for save data

    toc = time.time()
    print(f'{toc-tic} seconds')

    return result


def msc_run_wrapper(param_chain, param_run):
    # a function used to initialize chain by input parameters and run the chains

    chain = MCMC.init_msc_chain_by_instance(param_chain)
    result = chain.run(n_iter=param_run['n_iter'], only_save_last_bed=param_run['only_save_last_bed'],
                       info_per_iter=param_run['info_per_iter'], plot=param_run['plot'], progress_bar=param_run['progress_bar'])

    return result


if __name__ == '__main__':

    # load compiled bed elevation measurements
    df = pd.read_csv('DenmanDataGridded.csv')

    rng_seed = 23198104

    # create a grid of x and y coordinates
    x_uniq = np.unique(df.x)
    y_uniq = np.unique(df.y)

    xmin = np.min(x_uniq)
    xmax = np.max(x_uniq)
    ymin = np.min(y_uniq)
    ymax = np.max(y_uniq)

    cols = len(x_uniq)
    rows = len(y_uniq)

    resolution = 500

    xx, yy = np.meshgrid(x_uniq, y_uniq)

    # load other data
    dhdt = df['dhdt'].values.reshape(xx.shape)
    smb = df['smb'].values.reshape(xx.shape)
    velx = df['velx'].values.reshape(xx.shape)
    vely = df['vely'].values.reshape(xx.shape)
    bedmap_mask = df['bedmap_mask'].values.reshape(xx.shape)
    bedmachine_thickness = df['bedmachine_thickness'].values.reshape(xx.shape)
    bedmap_surf = df['bedmap_surf'].values.reshape(xx.shape)
    highvel_mask = df['highvel_mask'].values.reshape(xx.shape)
    bedmap_bed = df['bedmap_bed'].values.reshape(xx.shape)

    # notice that this didn't recover bedmachine bed for ocean bathymetry or sub-ice-shelf bathymetry
    bedmachine_bed = bedmap_surf - bedmachine_thickness

    # create conditioning data
    # bed elevation measurement in grounded ice region, and bedmachine bed topography elsewhere
    cond_bed = np.where(
        bedmap_mask == 1, df['bed'].values.reshape(xx.shape), bedmap_bed)
    df['cond_bed'] = cond_bed.flatten()

    # create a mask of conditioning data
    data_mask = ~np.isnan(cond_bed)

    initial_bed = np.loadtxt('Denman_bed_599000.txt')
    thickness = bedmap_surf - initial_bed
    # make sure every topography in the grounded ice region is below ice surface
    initial_bed = np.where((thickness <= 0) & (
        bedmap_mask == 1), bedmap_surf-1, initial_bed)

    # sigma here control the smoothness of the trend
    trend = sp.ndimage.gaussian_filter(initial_bed, sigma=10)

    # normalize the conditioning bed data, saved to df['Nbed']
    df['cond_bed_residual'] = df['cond_bed'].values-trend.flatten()
    data = df['cond_bed_residual'].values.reshape(-1, 1)
    # data used to evaluate the distribution. We use all data in the initial bed
    data_for_distribution = (initial_bed - trend).reshape((-1, 1))
    nst_trans = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                    subsample=None, random_state=rng_seed).fit(data_for_distribution)
    # normalize all data in df['cond_bed_residual']
    transformed_data = nst_trans.transform(data)
    df['Nbed_residual'] = transformed_data

    # randomly drop out 50% of coordinates. Decrease this value if you have a lot of data and it takes a long time to run
    df_sampled = df.sample(frac=0.5, random_state=rng_seed)
    df_sampled = df_sampled[df_sampled["cond_bed_residual"].isnull() == False]
    df_sampled = df_sampled[df_sampled["bedmap_mask"] == 1]

    # compute experimental (isotropic) variogram
    coords = df_sampled[['x', 'y']].values
    values = df_sampled['Nbed_residual']

    maxlag = 20000      # maximum range distance
    # num of bins (try decreasing if this is taking too long)
    n_lags = 70

    # compute variogram
    V1 = skg.Variogram(coords, values, bin_func='even',
                       n_lags=n_lags, maxlag=maxlag, normalize=False,
                       model='matern')

    # extract variogram values
    xdata = V1.bins
    ydata = V1.experimental

    # Notice: because we randomly drop out some data, the calculation of V1_p won't always obtain the same result
    # To ensure reproducibility of your work, please use a consistent set of V1_p throughout different chains
    V1_p = V1.parameters

    grounded_ice_mask = (bedmap_mask == 1)

    # initialize the small scale chain to be used as an example to initialize other small scale chain
    smallScaleChain = MCMC.chain_sgs(xx, yy, initial_bed, bedmap_surf, velx,
                                     vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution)
    # set the update region
    smallScaleChain.set_update_region(True, highvel_mask)

    # get mass flux residuals for bedmachien as a reference
    mc_res_bm = Topography.get_mass_conservation_residual(
        bedmachine_bed, bedmap_surf, velx, vely, dhdt, smb, resolution)

    # in multiprocessing, we choose to only use mass flux residual loss in squared sum (Gaussian distribution)
    smallScaleChain.set_loss_type(sigma_mc=5, massConvInRegion=True)

    # set up the block sizes
    min_block_x = 5
    max_block_x = 20
    min_block_y = 5
    max_block_y = 20
    smallScaleChain.set_block_sizes(
        min_block_x, max_block_x, min_block_y, max_block_y)

    # set up normal score transformation, the trend, the variogram parameters, and the sgs paramters
    smallScaleChain.set_normal_transformation(nst_trans, do_transform=True)

    smallScaleChain.set_trend(trend=trend, detrend_map=True)

    smallScaleChain.set_variogram(
        'Matern', V1_p[0], V1_p[1], 0, isotropic=True, vario_smoothness=V1_p[2])

    smallScaleChain.set_sgs_param(48, 30e3, sgs_rand_dropout_on=False)

    # set up the random generator used in the chain
    # in multiprocessing, the random generator in here will be replaced by rng_seeds later
    smallScaleChain.set_random_generator(rng_seed=rng_seed)

    n_iter = 100

    n_chains = 4
    n_workers = 4

    # fill in a list of initial_beds to be used for each chain
    # the list length should be equal to number of chains
    initial_beds = np.array(
        [initial_bed, initial_bed, initial_bed, initial_bed])

# =============================================================================
#     with open(Path('../200_seeds.txt'), 'r') as f:
#         lines = f.readlines()
#     rng_seeds = []
#     for line in lines:
#         rng_seeds.append(int(line.strip()))
# =============================================================================

    # fill in a list of rng_seeds to be used for each chain
    # the list length should be equal to number of chains
    example_seed = rng_seed
    rng_seeds = []
    for i in range(n_chains):
        rng_seeds.append(example_seed)

    # rng_seeds = [12312,18578,64830,85058]

    # number of iterations used to run each chain
    n_iters = [n_iter]*n_chains

    result = smallScaleChain_mp(n_chains, n_workers, 
        smallScaleChain, initial_beds, rng_seeds, n_iters)

    # beds, loss_mc, loss_data, loss, steps, resampled_times, blocks_used  = smallScaleChain.run(n_iter=100, info_per_iter=10, only_save_last_bed=False)
