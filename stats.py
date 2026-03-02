#%%
from SAXS_Cylinder import *
import numpy as np
from PaleoBSM import compute_trackSpectra, olivineObj
import emcee as emcee
import scipy.optimize as spo
from functools import partial
import matplotlib.pyplot as plt
import functions as fct


def logL_binned_faster(v, theta_centrals, sig_js, q_range, q_resolution, xt_bins, dRdx_dm, dRdx_bkg, I_dm_th, I_bkg_th, bins = 51):
    rho_sl_p_def = 4000e-27  # kg/nm^3
    R_dm, rho_sl_0_dm, R_bkg, rho_sl_0_bkg = v[0:4]
    theta_js = v[4:]
    if len(theta_centrals) != len(sig_js):
        raise Exception("Given central values for nuisance parameters is not equal in length to given errors")

    I_dm_meas, _ = cylinder_binned(dRdx_dm, xt_bins, q_range, q_resolution, R_dm, rho_sl_p_def, rho_sl_0_dm, bin_number=bins)

    I_bkg_meas, _ = cylinder_binned(dRdx_bkg, xt_bins, q_range, q_resolution, R_bkg, rho_sl_p_def, rho_sl_0_dm, bin_number=bins)

    I_total_th = np.asarray(I_dm_th) + np.asarray(I_bkg_th)
    I_total_meas = np.asarray(I_dm_meas) + np.asarray(I_bkg_meas)


    return -0.5 * (np.sum(((I_total_meas - I_total_th) ** 2 / (I_total_th))) + np.sum((theta_js-theta_centrals)**2/sig_js**2))


def log_prior(v, theta_centrals, rho_sl_0_frac = 0.90, R_dm_def = 0.1, R_bkg_def = 0.3):
    if (len(theta_centrals) != len(v[4:])):
        raise Exception("More/less nuisance parameters than given centrals")

    rho_sl_p_def = 4000e-27
    rho_sl_0_def = rho_sl_p_def * rho_sl_0_frac

    R_dm, rho_sl_0_dm, R_bkg, rho_sl_0_bkg = v[:4]

    # hard bounds – reject unphysical proposals
    if (R_dm <= 0) or (R_bkg <= 0) or (rho_sl_0_dm <= 0) or (rho_sl_0_bkg <= 0):
        return -np.inf

    lp  = -0.5 * ((np.log(R_dm) - np.log(R_dm_def))/1.0)**2    # R1 ~ log-normal(10 nm, σ=1 dex)
    lp += -0.5 * ((np.log(R_bkg) - np.log(R_bkg_def))/1.0)**2    # R2 ~ log-normal(30 nm, σ=1 dex)
    lp += -0.5 * ((np.log(rho_sl_0_dm) - np.log(rho_sl_0_def))/1.0)**2 # Δρ priors, centre & width as you like
    lp += -0.5 * ((np.log(rho_sl_0_bkg) - np.log(rho_sl_0_def))/1.0)**2

    for theta_j, theta_central in zip(v[4:], theta_centrals):
        lp += -0.5 * ((np.log(theta_j) - np.log(theta_central))/1.0)**2

    return lp

def log_post_test(v, theta_centrals, sig_js, q_range, q_resolution, x_bins, dRdx_dm, dRdx_bkg, I_dm_th, I_bkg_th):
    prior = log_prior(v, theta_centrals)
    if not np.isfinite(prior):
        return -np.inf
    return prior + logL_binned_faster(v, theta_centrals, sig_js, q_range, q_resolution, x_bins, dRdx_dm, dRdx_bkg, I_dm_th, I_bkg_th)


def get_bkg_total(mineral, x_range, resolution_x, C = 1e-9, bin_number = 100):

    neutrino_data_path = 'PaleoBSM/neutrino_fluxes/'
    solar_sources = ['pp', 'hep', 'B', 'F', 'Be862', 'Be384', 'N13', 'O15', 'pep']

    dRdx_solar = 0
    solar_flux = 0
    for i, source in enumerate(solar_sources):
        solar_flux_i = np.load(neutrino_data_path + f'extrapolate_solar_{source}_fluxes.npy')
        result, _ = compute_trackSpectra.get_SM_neutrino_Nbins(x_range, mineral, solar_flux_i, \
                                                               resolution=resolution_x, number_of_bins=bin_number)
        dRdx_solar += result
        solar_flux += solar_flux_i[1]

    atm_flux = np.load(neutrino_data_path + f'extrapolate_atm_fluxes.npy')
    DSNB_flux = np.load(neutrino_data_path + f'extrapolate_DSNB_fluxes.npy')
    GSNB_flux = np.load(neutrino_data_path + f'extrapolate_GSNB_fluxes.npy')

    dRdx_atm, x_bin_edges = compute_trackSpectra.get_SM_neutrino_Nbins(x_range, mineral, atm_flux, resolution=resolution_x, number_of_bins=bin_number)
    dRdx_DSNB, _ = compute_trackSpectra.get_SM_neutrino_Nbins(x_range, mineral, DSNB_flux, resolution=resolution_x, number_of_bins=bin_number)
    dRdx_GSNB, _ = compute_trackSpectra.get_SM_neutrino_Nbins(x_range, mineral, GSNB_flux, resolution=resolution_x, number_of_bins=bin_number)

    dRdx_neutron, _ = compute_trackSpectra.get_neutron_Nbins(C, x_range, mineral, resolution=resolution_x, number_of_bins=bin_number)
    dRdx_thorium, _ = compute_trackSpectra.get_Th_Nbins(C, x_range, resolution=resolution_x, number_of_bins=bin_number)

    dRdx_bkg = dRdx_solar + dRdx_atm + dRdx_DSNB + dRdx_GSNB + dRdx_neutron + dRdx_thorium
    fluxes = [np.sum(solar_flux), np.sum(DSNB_flux[1]), np.sum(GSNB_flux[1]), np.sum(atm_flux[1])]
    return dRdx_bkg, fluxes, x_bin_edges

def get_bkg_arrays_binned(mineral, x_range, resolution_x, C = 1e-9, bin_number = 100):

    neutrino_data_path = 'PaleoBSM/neutrino_fluxes/'
    solar_sources = ['pp', 'hep', 'B', 'F', 'Be862', 'Be384', 'N13', 'O15', 'pep']

    dRdx_solar = 0
    solar_flux = 0
    for i, source in enumerate(solar_sources):
        solar_flux_i = np.load(neutrino_data_path + f'extrapolate_solar_{source}_fluxes.npy')
        result, _ = compute_trackSpectra.get_SM_neutrino_Nbins(x_range, mineral, solar_flux_i, resolution=resolution_x, number_of_bins = bin_number)
        dRdx_solar += result
        solar_flux += solar_flux_i[1]

    atm_flux = np.load(neutrino_data_path + f'extrapolate_atm_fluxes.npy')
    DSNB_flux = np.load(neutrino_data_path + f'extrapolate_DSNB_fluxes.npy')
    GSNB_flux = np.load(neutrino_data_path + f'extrapolate_GSNB_fluxes.npy')

    dRdx_atm, x_bin_edges = compute_trackSpectra.get_SM_neutrino_Nbins(x_range, mineral, atm_flux, resolution=resolution_x, number_of_bins = bin_number)
    dRdx_DSNB, _ = compute_trackSpectra.get_SM_neutrino_Nbins(x_range, mineral, DSNB_flux, resolution=resolution_x, number_of_bins = bin_number)
    dRdx_GSNB, _ = compute_trackSpectra.get_SM_neutrino_Nbins(x_range, mineral, GSNB_flux, resolution=resolution_x, number_of_bins = bin_number)

    dRdx_neutron, _ = compute_trackSpectra.get_neutron_Nbins(C, x_range, mineral, resolution=resolution_x, number_of_bins = bin_number)
    dRdx_thorium, _ = compute_trackSpectra.get_Th_Nbins(C, x_range, resolution=resolution_x, number_of_bins = bin_number)

    dRdx_bkg = np.array([dRdx_solar, dRdx_atm, dRdx_DSNB, dRdx_GSNB, dRdx_neutron, dRdx_thorium])
    fluxes = np.array([np.sum(solar_flux), np.sum(DSNB_flux[1]), np.sum(GSNB_flux[1]), np.sum(atm_flux[1])])

    return dRdx_bkg, fluxes


def get_bkg_arrays(mineral, x_range, C = 1e-9):

    neutrino_data_path = 'PaleoBSM/neutrino_fluxes/'
    solar_sources = ['pp', 'hep', 'B', 'F', 'Be862', 'Be384', 'N13', 'O15', 'pep']

    dRdx_solar = 0
    solar_flux = 0
    for i, source in enumerate(solar_sources):
        solar_flux_i = np.load(neutrino_data_path + f'extrapolate_solar_{source}_fluxes.npy')
        result = compute_trackSpectra.get_SM_Neutrino_dRdx_mol(x_range, mineral, solar_flux_i)
        dRdx_solar += result
        solar_flux += solar_flux_i[1]

    atm_flux = np.load(neutrino_data_path + f'extrapolate_atm_fluxes.npy')
    DSNB_flux = np.load(neutrino_data_path + f'extrapolate_DSNB_fluxes.npy')
    GSNB_flux = np.load(neutrino_data_path + f'extrapolate_GSNB_fluxes.npy')

    dRdx_atm = compute_trackSpectra.get_SM_Neutrino_dRdx_mol(x_range, mineral, atm_flux)
    dRdx_DSNB = compute_trackSpectra.get_SM_Neutrino_dRdx_mol(x_range, mineral, DSNB_flux)
    dRdx_GSNB = compute_trackSpectra.get_SM_Neutrino_dRdx_mol(x_range, mineral, GSNB_flux)

    dRdx_neutron = compute_trackSpectra.get_neutron_dRdx_mol(x_range, mineral, C)

    data = np.load('PaleoBSM/SRIM_derived_data/Backgrounds/Th/Tables/dist_at_72keV_01ppb.npy')
    dRdx_table = data * (C / 1e-10)
    dRdx_thorium = np.interp(x_range, np.logspace(-2, 3, 500), dRdx_table)

    dRdx_bkg = np.array([dRdx_solar, dRdx_atm, dRdx_DSNB, dRdx_GSNB, dRdx_neutron[0, :], dRdx_thorium])
    fluxes = np.array([np.sum(solar_flux), np.sum(DSNB_flux[1]), np.sum(GSNB_flux[1]), np.sum(atm_flux[1])])

    return dRdx_bkg, fluxes

def mcmc_WIMPs(init, mineral, sig_x, m_x, resolution_x, resolution_q, nwalkers, samples, R_def=0.1, rho_dm_frac=0.95,
               rho_bkg_frac=0.90):
    rho_sl_p = 4000e-27
    x_range = np.logspace(np.log10(resolution_x / 2), 3, 100)
    q_range = np.logspace(np.log10(resolution_q / 2), 10, 100)
    dRdx_dm, _ = compute_trackSpectra.get_wimps_Nbins(sig_x, x_range, m_x, mineral, resolution=resolution_x)
    _, x_bin_edges, I_dm_th, _ = cylinder_WIMPs_binned(mineral, m_x, sig_x, x_range, resolution_x, q_range, resolution_q, R_def,
                                                 rho_sl_p, rho_sl_p * rho_dm_frac)

    dRdx_bkg, fluxes = get_bkg_total(mineral, x_range, resolution_x)
    I_bkg_th, _ = cylinder_binned(dRdx_bkg, x_bin_edges, q_range, resolution_q, R_def, rho_sl_p, rho_sl_p * rho_bkg_frac)

    theta_centrals_nonNeu = [1e-10, 1, 100]
    theta_centrals = np.append(fluxes, theta_centrals_nonNeu)
    sig_j = np.array([0.14, 1, 1, 1, 0.01, 0.05, 0.01 / 100])

    init = np.append(init, theta_centrals * sig_j)
    ndim = len(init)
    p0 = init * (1 + 1e-2 * np.random.randn(nwalkers, ndim))  # small scatter

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_post_test,
        args=(theta_centrals, sig_j, q_range, resolution_q, x_bin_edges, \
              dRdx_dm, dRdx_bkg, I_dm_th, I_bkg_th)
    )

    print("Running MCMC …")
    sampler.run_mcmc(p0, samples, progress=True)
    return sampler

def logL_binned_WIMPs(v, theta_centrals, sig_js, q_range, resolution_q, resolution_x, dRdx_bkg, I_dm_th, I_bkg_th, bins = 51):
    rho_sl_p_def = 4000e-27  # kg/nm^3
    rho_sl_0_def = 0.90*rho_sl_p_def
    R_dm = 1e-10
    R_bkg = 1e-10

    x_range = np.logspace(np.log10(resolution_x / 2), 3, 100)
    m_x, sig_x = v[0:2]
    theta_js = v[2:]

    if len(theta_centrals) != len(sig_js):
        raise Exception("Given central values for nuisance parameters is not equal in length to given errors")

    dRdx_dm, xt_bins = compute_trackSpectra.get_wimps_Nbins(sig_x, x_range, m_x, mineral, resolution=resolution_x)
    I_dm_meas, _ = cylinder_binned(dRdx_dm, xt_bins, q_range, resolution_q, R_dm, rho_sl_p_def, rho_sl_0_def, bin_number=bins)
    I_bkg_meas, _ = cylinder_binned(dRdx_bkg, xt_bins, q_range, resolution_q, R_bkg, rho_sl_p_def, rho_sl_0_def, bin_number=bins)

    I_total_th = np.asarray(I_dm_th) + np.asarray(I_bkg_th)
    I_total_meas = np.asarray(I_dm_meas) + np.asarray(I_bkg_meas)
 
    return -0.5 * (np.sum(((I_total_meas - I_total_th) ** 2 / (I_total_th))) + np.sum((theta_js-theta_centrals)**2/sig_js**2))

def profileL_ratio_old(m_x, sig_x, theta_centrals, sig_js, q_range, q_resolution, x_resolution, dRdx_bkg, I_dm_th, I_bkg_th):
    x0 = np.append([m_x, sig_x], theta_centrals)
    fun = lambda x: logL_binned_WIMPs(x, theta_centrals, sig_js, q_range, q_resolution, x_resolution, dRdx_bkg, I_dm_th, I_bkg_th)
    L_denom = sp.optimize.minimize(fun, x0, method='L-BFGS-B')
    def fun2(theta_js):
        return logL_binned_WIMPs(np.append([m_x, sig_x], theta_js), theta_centrals, sig_js, q_range, q_resolution, x_resolution, dRdx_bkg, I_dm_th, I_bkg_th)
    L_num = sp.optimize.minimize(fun2, theta_centrals, method='L-BFGS-B')
    return L_num.fun - L_denom.fun

def profileL_old(m_x, sig_x, sig_js, q_resolution, resolution_x):
    rho_sl_p = 4000e-27
    R_def = 1e-10

    x_range = np.logspace(np.log10(resolution_x / 2), 3, 100)
    q_range = np.logspace(np.log10(q_resolution / 2), np.log10(3), 100)
    q_bin_number = int((max(q_range)-min(q_range))/(2.533*q_resolution))
    dRdx_dm, _ = compute_trackSpectra.get_wimps_Nbins(sig_x, x_range, m_x, mineral, resolution=resolution_x)
    _, x_bin_edges, I_dm_th, _ = cylinder_WIMPs_binned(mineral, m_x, sig_x, x_range, resolution_x, q_range, q_resolution, R_def,
                                                       rho_sl_p, rho_sl_p * 0.9)
    dRdx_bkg, fluxes, _ = get_bkg_total(mineral, x_range, resolution_x, \
                                        C = 1e-9, bin_number = q_bin_number)
    I_bkg_th, _ = cylinder_binned(dRdx_bkg, x_bin_edges, q_range, q_resolution, R_def, rho_sl_p, rho_sl_p * 0.9, \
                                  bin_number = q_bin_number)
    theta_centrals_nonNeu = [1e-10, 1, 100]
    theta_centrals = np.append(fluxes, theta_centrals_nonNeu)
    return profileL_ratio_old(m_x, sig_x, theta_centrals, sig_js, q_range, q_resolution, resolution_x, dRdx_bkg, I_dm_th, I_bkg_th)

def profileL_test(m_xs, sig_xs, sig_js, q_resolution,\
                  test = False, m = 0.1, t = 1e3, rho_sl_p = 1, rho_sl_0 = 1, R_def = 1,\
                  q_bin_number = None, type = 'Cylinder'):

    x_range = np.logspace(-2, 3, 200)
    q_range = np.logspace(np.log10(q_resolution / 2), np.log10(3), 200)
    if q_bin_number == None:
        q_bin_number = int((max(q_range)-min(q_range))/(2.533*q_resolution))

    dRdx_dm_grid = np.empty((len(m_xs), len(sig_xs), len(x_range)))
    for i, m_x in enumerate(m_xs):
        dRdx_dm_grid[i, 0] = compute_trackSpectra.get_wimps_dRdx(m_x, sig_xs[0], x_range, mineral)
        for j, sig_x in enumerate(sig_xs[1:]):
            dRdx_dm_grid[i, j+1] = sig_x/sig_xs[0] * dRdx_dm_grid[i, 0]

    dRdx_dm_grid = dRdx_dm_grid * m * t
    I_dm, bins = cylinder_binned_mesh(dRdx_dm_grid, x_range, q_range, q_resolution, R_def, rho_sl_p, rho_sl_0, \
                                      bin_number = q_bin_number, m = m/1000, type = type)

    dRdx_bkgs, fluxes = get_bkg_arrays(mineral, x_range, C=1e-9)
    dRdx_bkgs = dRdx_bkgs * m * t
    I_bkg_th = np.empty((dRdx_bkgs.shape[0], len(bins)-1))
    for i, dRdx_bkg in enumerate(dRdx_bkgs):
        if i == 3:
            I_bkg_th[i] = cylinder_binned_quick(dRdx_bkg, x_range, q_range, q_resolution, R_def, rho_sl_p, rho_sl_0,\
                                            bin_number = q_bin_number, m = m/1000, alpha_0 = 9, alpha_f = np.deg2rad(11),
                                                type = type)[0]
        else:
            I_bkg_th[i] = cylinder_binned_quick(dRdx_bkg, x_range, q_range, q_resolution, R_def, rho_sl_p, rho_sl_0,\
                                            bin_number =q_bin_number, m = m/1000,
                                                type = type)[0]
    I_meas = np.sum(I_bkg_th, axis=0)
    drho = rho_sl_p - rho_sl_0
    mx_grid, sigx_grid = np.meshgrid(m_xs, sig_xs, indexing = 'ij')
    theta_centrals_nonNeu = [1e-10, m, t, drho]
    theta_centrals = np.append(fluxes, theta_centrals_nonNeu)
    sig_js = sig_js*theta_centrals
    bounds = np.empty((len(sig_js), 2))
    bounds[:, 0] = theta_centrals - sig_js
    bounds[:, 1] = theta_centrals + sig_js

    if test:
        return I_dm, I_bkg_th, bins

    from functools import partial
    def profile_one(i, j):
        nll_fun = partial(logL_binned_one,
                          theta_centrals=theta_centrals,
                          sig_js=sig_js,
                          I_meas=I_meas,
                          I_dm=I_dm[i, j],
                          I_bkg=I_bkg_th,
                          drho_central = drho)

        x0_nuis = theta_centrals + sig_js/2
        v_fixed = np.array([mx_grid[i, j], sigx_grid[i, j]])

        res = sp.optimize.minimize(
            lambda x: nll_fun(np.concatenate([v_fixed, x])),
            x0=x0_nuis,
            bounds= bounds,
            method='Nelder-Mead',
            options={'fatol':1e-90, 'disp':False})
        return res.fun

    # print("Starting Profile Likelihood Denominator Calculation")
    # L_denom = compute_denominator(theta_centrals, sig_js,
    #                             x_range, q_range, q_bin_number,
    #                             I_bkg_th,
    #                             mineral, resolution_q,
    #                               m = m, t = t, rho_sl_0 = rho_sl_0, rho_sl_p=rho_sl_p)
    # print(L_denom)
    L_denom = 0
    L_num = np.empty_like(mx_grid)
    import time
    start_time = time.time()
    print("Starting Profile Likelihood Calculation")
    for i, m in enumerate(m_xs):
        for j, sig in enumerate(sig_xs):
            L_num[i, j] = profile_one(i, j)
        print("Row %d: %s seconds" % (i, time.time() - start_time))
        start_time = time.time()
    return -(L_num - L_denom)

def logL_binned_one(v, theta_centrals, sig_js, I_meas, I_dm, I_bkg, drho_central):

    m_x, sig_x = v[0:2]
    theta_js = v[2:]
    C_js = v[6]
    theta_js_fluxes = np.append(v[2:7], C_js)
    C_central = theta_centrals[4]
    theta_central_fluxes = np.append(theta_centrals[:5], C_central)
    if len(theta_centrals) != len(sig_js):
        raise Exception("Given central values for nuisance parameters is not equal in length to given errors")
    drho_ratio = theta_js[-1]/drho_central
    I_th = drho_ratio**2 * (I_dm + np.sum((theta_js_fluxes/theta_central_fluxes)[:, None] * I_bkg, axis = 0))
    I_meas = drho_ratio**2 * I_meas
    return (np.sum(((I_meas - I_th) ** 2 / (I_th))) + np.sum(
        (theta_js - theta_centrals) ** 2 / sig_js ** 2))

def sig_test(m_x, sig_xs):
    x_range = np.logspace(np.log10(10/ 2), 3, 10)
    dR_dm_test = np.empty((1, 100))
    for sig_x in sig_xs:
        temp = compute_trackSpectra.get_wimps_Nbins(sig_x, x_range, m_x, olivineObj, resolution=10)[0]
        dR_dm_test = np.vstack((dR_dm_test, temp))
    dR_dm_test2 = (compute_trackSpectra.get_wimps_Nbins(sig_xs[0], x_range, m_x, olivineObj, resolution=10)[0])[np.newaxis, :]
    for sig_x in sig_xs[1:]:
        temp = sig_x/sig_xs[0] * dR_dm_test2[0]
        dR_dm_test2 = np.vstack((dR_dm_test2, temp))
    return dR_dm_test, dR_dm_test2

def chi2_full(p, theta_c, sigma_abs,                        # 7-vec
              x_range, q_range, q_bin_number,
              I_bkg_templates,                              # (6, Nbin)
              mineral, resolution_q,
              R_def, rho_sl_p, rho_sl_0, \
              m = 0.1, t = 1e3):

    m_chi, sig_chi = p[:2]
    theta          = p[2:]

    dRdx_dm = compute_trackSpectra.get_wimps_dRdx(
                    m_chi, sig_chi, x_range, mineral)

    dRdx_dm = dRdx_dm * m * t

    I_dm, _ = cylinder_binned_quick(dRdx_dm, x_range, q_range,
                                    resolution_q, R_def,
                                    rho_sl_p, rho_sl_0,
                                    bin_number = q_bin_number)

    theta_flux   = np.array([theta[0],theta[1],theta[2],theta[3],
                             theta[4],theta[4]])
    theta_c_flux = np.array([theta_c[0],theta_c[1],theta_c[2],theta_c[3],
                             theta_c[4],theta_c[4]])
    scale = (theta_flux/theta_c_flux)[:,None]
    rho_ratio = theta[-1]/(rho_sl_p-rho_sl_0)

    I_th  = (I_dm + (scale * I_bkg_templates).sum(axis=0)) * rho_ratio**2
    I_meas = I_bkg_templates.sum(axis=0) * rho_ratio**2

    chi2_bins  = np.sum((I_meas - I_th)**2 / I_th)
    chi2_pulls = np.sum((theta - theta_c)**2 / sigma_abs**2)

    return chi2_bins + chi2_pulls     # = −2 ln L

def I_bkg_binned(q_resolution, m = 0.1, t = 1e3,
                 rho_sl_p = 1, rho_sl_0 = 1, R_def = 1, C = 1e-9, \
                 q_bin_number = None, q_max = 1e3, dRdx_bkg = None):

    x_range = np.logspace(-2, 3, 200)
    q_range = np.logspace(np.log10(q_resolution / 2), np.log10(q_max), 200)
    if q_bin_number == None:
        q_bin_number = int((max(q_range) - min(q_range)) / (2.533 * q_resolution))
    if dRdx_bkg is None:
        dRdx_bkg = get_bkg_arrays(mineral, x_range, C = C)[0] * m * t
    I_bkg = np.empty((6, q_bin_number))
    q_bins = 0

    for i, dRdx_bkg in enumerate(dRdx_bkg):
        if i == 3:
            I_bkg[i], q_bins = cylinder_binned_quick(dRdx_bkg, x_range, q_range, q_resolution, R_def, rho_sl_p, rho_sl_0, \
                                                bin_number=q_bin_number, m=m / 1000, alpha_0=0,
                                                alpha_f=np.deg2rad(3.4))
        else:
            I_bkg[i] = cylinder_binned_quick(dRdx_bkg, x_range, q_range, q_resolution, R_def, rho_sl_p, rho_sl_0,\
                                                bin_number = q_bin_number, m = m / 1000)[0]
    return I_bkg, q_bins

def find_qmins(q_resolution, R_range, m = 0.1, t = 1e3,
          rho_sl_p = 1, rho_sl_0 = 1, C = 1e-9):
    x_range = np.logspace(-2, 3, 200)
    dRdx_bkg = get_bkg_arrays(mineral, x_range, C=C)[0] * m * t
    q_mins = []
    for R in R_range:
        I_bk, q_bins = I_bkg_binned(q_resolution, m = m, t = t,
                                    rho_sl_p = rho_sl_p, rho_sl_0 = rho_sl_0,
                                    R_def = R, C = C, dRdx_bkg = dRdx_bkg)
        I_bk = np.sum(I_bk, axis=0)
        min = I_bk[0]
        for i, I in enumerate(I_bk[1:]):
            if I <= min:
                min = I
            else:
                q_mins = np.append(q_mins, (q_bins[i] + q_bins[i-1])/2)
                break

    return q_mins


def compute_denominator(theta_centrals, sigma_abs,
                        x_range, q_range, q_bin_number,
                        I_bkg_templates,
                        mineral, resolution_q,
                        R_def=1e-10, rho_sl_p= 1, rho_sl_0 = 1,
                        m = 0.1, t = 1e3):

    # initial guess
    p0 = np.concatenate([
            [np.median([0.1, 1e3]),   # mχ = middle
             1e-44],                  # σχ = somewhere in the middle
            theta_centrals])

    bnds = [(0, None), (0, None)] + [(0, None)]*5 + [(0, None)]*3

    obj = partial(chi2_full, theta_c=theta_centrals,
                  sigma_abs=sigma_abs,
                  x_range=x_range, q_range=q_range,
                  I_bkg_templates=I_bkg_templates,
                  mineral=mineral,
                  resolution_q=resolution_q,
                  R_def=R_def, rho_sl_p=rho_sl_p, rho_sl_0=rho_sl_0,
                  m = m, t = t, q_bin_number=q_bin_number)

    res = spo.minimize(obj, p0, bounds=bnds, method='Nelder-Mead')
    if not res.success:
        raise RuntimeError(res.message)

    return res.fun          #  χ²_min  (denominator of Eq. 31)
#%%
if False:
    mineral = olivineObj
    A_to_m = 1e-10  # 1^-10 meters / 1 angstrum
    resolution_q = 0.01
    sig_j = np.array([0.14, 1, 1, 1, 0.01, 0.05, 0.01 / 100, 0.50])
    rho_bulk = 1.7e-24 # kg/nm^3
    M_weight = 0.1533 # kg/mol
    rho_sl_p =  rho_xray(mineral, rho_bulk, M_weight)
    rho_sl_0 = rho_sl_p * 0.95
    sig_xs = np.logspace(np.log10(1e-43), np.log10(1e-31), 110)
    m_xs = np.logspace(np.log10(5e-1), np.log10(1e3), 110)
    types = ['Cylinder', 'Hollow', 'Capped', 'Sphere', 'Ellipsoid']
    import matplotlib.pyplot as plt
    from itertools import cycle
    import functions as fct
    fct.plotSet(r'$\lambda(m_\chi,\sigma_\chi)\,<\,-2.71$', r'$m_\chi$ [GeV]',
                r'$\sigma_\chi$ [cm$^{2}$]', 0, 10, 0, 10)
    lines = ["-", "--", "-.", ":", (0, (5, 10))]
    linecycler = cycle(lines)
    for type in types:

        I_profile_100g = profileL_test(m_xs, sig_xs, sig_j, resolution_q, \
                                  rho_sl_p = rho_sl_p, rho_sl_0 = rho_sl_0, test = False, m = 100,
                                       type = type)
        I_profile_10mg = profileL_test(m_xs, sig_xs, sig_j, resolution_q, \
                                rho_sl_p = rho_sl_p, rho_sl_0 = rho_sl_0, test = False, m = 0.01,
                                       type = type)

        #%%
        sigx_grid, m_xgrid = np.meshgrid(sig_xs, m_xs, indexing='ij')
        # -- indices of the first crossing per column (from the code you pasted) ---
        mask       = I_profile_100g.T < -2.71
        first_true = np.argmax(mask, axis=0)           # row index for each col
        has_true   = mask.any(axis=0)                  # bool mask
        first_true = np.where(has_true, first_true, -1)

        # -- convert indices → physical values -------------------------------------
        valid_cols     = first_true >= 0               # columns that do cross
        mchi_curve_100g     = m_xs[valid_cols]              # x-axis (GeV)
        sig_curve_100g      = sig_xs[first_true[valid_cols]]# y-axis (cm²)

        # -- indices of the first crossing per column (from the code you pasted) ---
        mask       = I_profile_10mg.T < -2.71
        first_true = np.argmax(mask, axis=0)           # row index for each col
        has_true   = mask.any(axis=0)                  # bool mask
        first_true = np.where(has_true, first_true, -1)

        # -- convert indices → physical values -------------------------------------
        valid_cols     = first_true >= 0               # columns that do cross
        mchi_curve_10mg     = m_xs[valid_cols]              # x-axis (GeV)
        sig_curve_10mg      = sig_xs[first_true[valid_cols]]# y-axis (cm²)
        ls = next(linecycler)
        # -- log–log plot -----------------------------------------------------------
        plt.plot(mchi_curve_100g, sig_curve_100g, linestyle = ls, \
                 lw=2, label='100g ' + type, color = 'red')
        plt.plot(mchi_curve_10mg, sig_curve_10mg, linestyle = ls, \
                 lw=2, label='10mg ' + type, color = 'black')

    plt.ylim(min(sig_xs), max(sig_xs))
    plt.xlim(min(m_xs), max(m_xs))
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(which='both', ls=':')
    plt.show()
    # #%%
    # m_xs = [mchi_curve_100g[0], mchi_curve_100g[10]]
    # sig_xs = [sig_curve_100g[0]*1.5, sig_curve_100g[0]]
    # I_dm, I_th, bins = profileL_test(m_xs, sig_xs, sig_j, resolution_q, \
    #                                  test=True, rho_sl_p=rho_sl_p, rho_sl_0=rho_sl_0)
    # #%%
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.stairs(I_dm[1, 1], bins, label = 'Right on line')
    # plt.stairs(I_dm[1, 0], bins, label = 'Above line')
    # plt.stairs(np.sum(I_th, axis=0), bins, label = 'Background')
    # plt.legend()
    # plt.show()

if True:
    #%%
    R = 1
    mineral = olivineObj
    rho_bulk = 1.7e-24 # kg/nm^3
    M_weight = 0.1533 # kg/mol
    rho_sl_p =  rho_xray(mineral, rho_bulk, M_weight)
    rho_sl_0 = rho_sl_p * 0.95
    sig_x = 1e-42
    m_x = 1
    m = 100
    t = 1e3
    x_resolution = 1
    x_range = np.logspace(np.log10(x_resolution / 2), 3, 200)


    rate_bkg = np.sum(get_bkg_arrays(mineral, x_range)[0], axis=0) * m * t
    rate_dm_low = compute_trackSpectra.get_wimps_dRdx(m_x, sig_x, x_range, mineral) * m * t
    m_x = 5
    sig_x = 1e-45
    rate_dm_med = compute_trackSpectra.get_wimps_dRdx(m_x, sig_x, x_range, mineral) * m * t
    m_x = 100
    sig_x = 1e-47
    rate_dm_high = compute_trackSpectra.get_wimps_dRdx(m_x, sig_x, x_range, mineral) * m * t
#%%
    fct.plotSet("Recoil Spectrum: m = 100g, t = 1Gyr", "x [nm]",
                r"$dR/dX \cdot m \cdot t$", min(x_range), max(x_range),
                0, 10)
    plt.plot(x_range, rate_bkg, color = 'black', label = "Background")
    plt.plot(x_range, rate_dm_low[0], color = 'green', label = r"$m_{\chi} = 1GeV, \sigma_{\chi} = 10^{-42}cm^2$")
    plt.plot(x_range, rate_dm_med[0], color = 'blue', label = r"$m_{\chi} = 5GeV, \sigma_{\chi} = 10^{-45}cm^2$")
    plt.plot(x_range, rate_dm_high[0], color = 'red', label = r"$m_{\chi} = 100GeV, \sigma_{\chi} = 10^{-47}cm^2$")
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(0.5e0, 10e13)
    plt.xlim(0.5e0, 1e3)
    plt.legend()
    plt.show()
#%%
    q_resolution = 0.01
    q_range = np.logspace(np.log10(q_resolution / 2), np.log10(3), 500)
    q_bin_number = int((max(q_range) - min(q_range)) / (2.533 * q_resolution))
    I_bkg, bin_edges_q = cylinder_binned_quick(rate_bkg, x_range, q_range, q_resolution, \
                                       R, rho_sl_p, rho_sl_0, m = m, bin_number = q_bin_number)
    I_dm_low, _ =  cylinder_binned_quick(rate_dm_low, x_range, q_range, q_resolution, \
                                       R, rho_sl_p, rho_sl_0, m = m, bin_number = q_bin_number)
    I_dm_med, _ = cylinder_binned_quick(rate_dm_med, x_range, q_range, q_resolution, \
                                         R, rho_sl_p, rho_sl_0, m=m, bin_number=q_bin_number)

    I_dm_high, _ = cylinder_binned_quick(rate_dm_high, x_range, q_range, q_resolution, \
                                         R, rho_sl_p, rho_sl_0, m=m, bin_number=q_bin_number)
#%%
    fct.plotSet(r"Intensity Spectrum Cylinder: R = 1nm, $\rho_{sl, 0} = 0.95 \cdot\rho_{sl, p}$, m = 100g, t = 1Gyr",
                r"q $[\AA^{-1}]$",
                r"I(q) $[cm^{-1}]$",
                min(x_range), max(x_range),
                0, 10)
    plt.stairs(I_bkg * 1e7, bin_edges_q * 0.1, color = 'black', label = 'Background')
    plt.stairs(I_dm_low * 1e7, bin_edges_q * 0.1, color = 'green', label = r"$m_{\chi} = 1GeV, \sigma_{\chi} = 10^{-42}cm^2$")
    plt.stairs(I_dm_med * 1e7, bin_edges_q * 0.1, color = 'blue', label = r"$m_{\chi} = 5GeV, \sigma_{\chi} = 10^{-45}cm^2$")
    plt.stairs(I_dm_high * 1e7, bin_edges_q * 0.1, color = 'red', label = r"$m_{\chi} = 100GeV, \sigma_{\chi} = 10^{-47}cm^2$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.autoscale()
    plt.xlim(q_resolution / 1.95*0.1, 3*0.1)
    plt.show()
#%%
if False:
    mineral = olivineObj
    A_to_m = 1e-10  # 1^-10 meters / 1 angstrum
    x_resolution = 10
    q_resolution = 0.01
    sig_js = np.array([0.14, 1, 1, 1, 0.01, 0.05, 0.01 / 100, 0.10])
    rho_bulk = 1.7e-24 # kg/nm^3
    M_weight = 0.1533 # kg/mol
    rho_sl_p =  rho_xray(mineral, rho_bulk, M_weight)
    rho_sl_0 = rho_sl_p * 0.9
    sig_xs = [1e-42, 1e-45, 1e-47]
    m_xs = [1, 5, 100]
    I_dm, I_th, bins = profileL_test(m_xs, sig_xs, sig_js, q_resolution, \
                  test=True, rho_sl_p = rho_sl_p, rho_sl_0 = rho_sl_0)

    #%%
    plt.xscale('log')
    plt.yscale('log')
    plt.stairs(I_th[3], bins)
    plt.show()

if True:
    #%%
    R_range = np.logspace(1e-2, 1e1, 100)
    mineral = olivineObj
    rho_bulk = 1.7e-24 # kg/nm^3
    M_weight = 0.1533 # kg/mol
    rho_sl_p =  rho_xray(mineral, rho_bulk, M_weight)
    rho_sl_0 = rho_sl_p * 0.9
    m = 100
    t = 1e3
    q_resolution = 0.01

    q_mins = find_qmins(q_resolution, R_range, m = m, t = t,
                        rho_sl_p = rho_sl_p, rho_sl_0 = rho_sl_0)
    #%%
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(R_range, q_mins, 'bo', markersize = 3)
    plt.show()

#%%
if True:
    #%%
    R_range = [0.01, 0.1, 1, 10]
    mineral = olivineObj
    rho_bulk = 1.7e-24 # kg/nm^3
    M_weight = 0.1533 # kg/mol
    rho_sl_p =  rho_xray(mineral, rho_bulk, M_weight)
    rho_sl_0 = rho_sl_p * 0.9
    m = 100
    t = 1e3
    q_resolution = 0.01

    x_range = np.logspace(-2, 3, 200)
    dRdx_bkg = get_bkg_arrays(mineral, x_range)[0] * m * t
    I_bkg = []  # or better, use a list of rows

    for R in R_range:
        I_bk, q_bins = I_bkg_binned(q_resolution, m=m, t=t,
                                    rho_sl_p=rho_sl_p, rho_sl_0=rho_sl_0,
                                    R_def=R, dRdx_bkg=dRdx_bkg)
        I_bkg.append(np.sum(I_bk, axis=0))

    I_bkg = np.array(I_bkg)
    #%%
    fct.plotSet(r"Intensity Spectrum Cylinder: Background, $\rho_{sl, 0} = 0.95 \cdot\rho_{sl, p}$, m = 100g, t = 1Gyr",
                r"q $[\AA^{-1}]$",
                r"I(q) $[cm^{-1}]$",
                min(x_range), max(x_range),
                0, 10)
    for i, I in enumerate(I_bkg):
        plt.stairs(I*1e7, q_bins*0.1, label = 'R = ' + str(R_range[i]) + "nm")
    plt.xscale('log')
    plt.yscale('log')
    plt.axvline(0.3, ls='--', color='black', label = r"SAXS $q_{max} = 0.3 \AA^{-1}$")
    plt.legend()
    plt.autoscale()
    plt.xlim(q_resolution / 1.95*0.1, max(q_bins*0.1))
    plt.show()
#%%
# # -- log–log plot -----------------------------------------------------------
# plt.plot(sig_xs, I_profile[1], '-k', \
#          lw=2, label='10nm, 10mg, $m_\chi$ = ' + str(int(m_xs[5])) + "GeV")
# plt.xlim(min(sig_xs), max(sig_xs))
# plt.xscale('log')
# plt.yscale('symlog')
# plt.ylim(None, 0.5e0)
# plt.legend()
# plt.xlabel(r'$\sigma_\chi$ [cm$^{2}$]')
# plt.ylabel(r'$\lambda$')
# plt.title(r'$\lambda(m_\chi,\sigma_\chi)$')
# plt.grid(which='both', ls=':')
# plt.show()


# #%%
# nwalkers = 32
# samples = 500
# init = np.array([1, 4000*1e-27*0.85, 30, 4000*1e-27*0.9])
# sampler = mcmc_WIMPs(init, mineral, sig_x, m_x, resolution_x, resolution_q, nwalkers, samples)
# # -- discard burn-in, flatten, and inspect -------------------------------
# burn  = 100
# thin  = 10
# flat_chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
# # quick summaries
# #%%
# import corner
# test = flat_chain[:, 0:4]
# fig = corner.corner(flat_chain[:, 0:4],
#               labels=[r"$R_1$", r"$\rho_{sl1}$", r"$R_2$", r"$\rho_{sl2}$"]);
# fig.show()
