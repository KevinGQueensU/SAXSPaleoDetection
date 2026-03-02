import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy import special
from .wimps.compute_wimps import get_drder_one
from .light_mediators.compute_neutrino_spectra import get_dRdE_nu, get_dRdE_nu_SM, get_dRdE_nu_BSMcorrection
import time 

def normal_dist_simple(x):
    return np.exp(-x**2/2)/(np.sqrt(2*np.pi))

def norm_cdf(x):
    return (special.erf((x)/(np.sqrt(2)))+1)/2

def skew_normal_dist(a, loc, scale, x):
    y = (x-loc)/scale
    cdf = norm_cdf(y*a)
    return 2* normal_dist_simple(y)*cdf/scale

def fit_alpha(E, data_path):
    # in log
    skewnorm_alpha = np.load(data_path+'Tables/skewnorm_alpha.npy')
    m = skewnorm_alpha[0]
    c = skewnorm_alpha[1]
    y = skewnorm_alpha[2]
    lnE = np.log(E)
    return np.exp(m*lnE + c)+y

def fit_mean(E, data_path):

    skewnorm_params = np.load(data_path+'Tables/skewnorm_params.npy')
    E_data = skewnorm_params[0]
    mean_data = skewnorm_params[2]
    fitted_mean = interp1d(E_data, mean_data, fill_value='extrapolate')(E)
    return fitted_mean*(fitted_mean>0)

def fit_var(E, data_path):
    skewnorm_params = np.load(data_path+'Tables/skewnorm_params.npy')
    E_data = skewnorm_params[0]
    var_data = skewnorm_params[3]
    fitted_var = interp1d(E_data, var_data, fill_value='extrapolate')(E)
#     quartic = a*E**4 + b*E**3 + c*E**2 +d*E + e
    return fitted_var* (fitted_var>0)

def get_dRdx_general_one_target(xt_range, dRdE_table, data_path):
    
    # dRdE table for incoming parrticle;  Erange should be universal : np.logspace(-3,3, 700)
    Erange, dRdE_val = dRdE_table
    drder = dRdE_val[..., np.newaxis] # always add a new axis for x pts # extra dimensions go first

    # load SRIM track data points
    Elist = np.load(data_path+'Tables/distTabE.npy')
    xlist = np.load(data_path+'Tables/distTabx.npy')
    data_grid = np.load(data_path+'Tables/distTab2d.npy')
    Prob0track_data = np.loadtxt(data_path+'Tables/Prob_ObservableTrack.txt')
    Prob0track = np.interp(xt_range, Prob0track_data[:,0], Prob0track_data[:,1], left=0., right=1.)

    # interpolate x,E ranges from grid
    finterp = RegularGridInterpolator((Elist, xlist),data_grid, fill_value=0, bounds_error=False)
    Egrid, xgrid = np.meshgrid(Erange, xt_range, indexing='ij', sparse=True)

    interp_vals = finterp((Egrid, xgrid))
    norm = np.trapz(interp_vals,xgrid , axis=-1)
    idx0 = np.where(norm == 0)[0]
    norm[idx0] = 1

    interp_distribution = interp_vals/norm[:,np.newaxis]

    integrand = drder * interp_distribution # trailing dimension already matched, no new axis needed

    drdx_one = np.trapz(integrand, Erange, axis=-2) # E always in second last axis

    return drdx_one*Prob0track # trailing dimension matched in all cases


def get_wimps_dRdx_one(mx, sigma, xt_range, A, data_path, frac=0, spin='SI'):

    E_universal = np.logspace(-3,3, 700)
    dRdE_table = get_drder_one(E_universal, A, mx, sigma, frac=frac, spin=spin)

    dRdx = get_dRdx_general_one_target(xt_range, dRdE_table, data_path)

    return dRdx


def get_wimps_dRdx(mx, sigma, xt_range, mineral, frac=0, spin='SI'):

    dRdx_dm = 0
    for i in range(4):
        A = mineral.atomic_masses[i]
        data_path =mineral.derived_data_path[i]
        fraction = mineral.atomic_fractions[i]

        dRdx_dm += get_wimps_dRdx_one(mx, sigma, xt_range, A, data_path, frac=frac, spin=spin)*fraction
    return dRdx_dm

def get_neutron_dRdx_one(xt_range, data_path, molecule_mass,C):


    try:
        extraDim = len(C)
    except TypeError:
        extraDim = 0
    
    if extraDim == 0:
        C = np.array([C])
    
    E_universal = np.logspace(-3,3, 700)
    
    dRdE_load = np.load(data_path+'Tables/neutron_recoil.npy')

    dRdE_interp = interp1d(dRdE_load[0], dRdE_load[1][np.newaxis, ...]*(C[..., np.newaxis]/1e-10)*(molecule_mass/238.05), \
                           axis=-1 ,fill_value=0, bounds_error=False)(E_universal)
    
    dRdE_table = (E_universal, dRdE_interp)

    dRdx = get_dRdx_general_one_target(xt_range, dRdE_table, data_path)

    return dRdx


def get_neutron_dRdx_mol(xt_range, mineral, C):

    number = mineral.number_of_elements
    num_frac = np.array(mineral.number_fractions)
    atomic_mass = np.array(mineral.atomic_masses)
    molecule_mass = np.sum(num_frac*atomic_mass)

    dRdx_neutron = 0
    for i in range(number):
        data_path = mineral.derived_data_path[i]
        fraction = mineral.atomic_fractions[i]

        dRdx_neutron += get_neutron_dRdx_one(xt_range, data_path, molecule_mass,C)*fraction
    return dRdx_neutron

def get_Neutrino_dRdx_one(xt_range, data_path, A, Z, Sn, flux_table,mscalar=None, g_vs=None, mvector=None, g_vz=None, mpseuvec=None, g_va=None, kappa=1):
    
    # Sn = 0 #for SM bkg
    E_universal = np.logspace(-3,3, 700)
    
    dRdE = get_dRdE_nu(E_universal, A, Z, Sn, flux_table,mscalar=mscalar, g_vs=g_vs, mvector=mvector, g_vz=g_vz, mpseuvec=mpseuvec, g_va=g_va, kappa=kappa) #[coupling, E]
    dRdE_table = (E_universal, dRdE)

    dRdx = get_dRdx_general_one_target(xt_range, dRdE_table, data_path) # collapse the coupling dimension

    return dRdx


def get_Neutrino_dRdx_mol(xt_range, mineral, flux_table,mscalar=None, g_vs=None, mvector=None, g_vz=None, mpseuvec=None, g_va=None, kappa=1):
    
    number = mineral.number_of_elements

    dRdx_neutrino = 0
    for i in range(number):
        data_path = mineral.derived_data_path[i]
        A = mineral.atomic_masses[i]
        Z = mineral.atomic_number[i]
        Sn = mineral.nuclear_spin[i]
        fraction = mineral.atomic_fractions[i]
        spin_fraction  = mineral.spin_isotopic_fractions[i]
        
        if mpseuvec == None:
            dRdx_neutrino += get_Neutrino_dRdx_one(xt_range, data_path, A, Z, Sn, flux_table,\
                mscalar=mscalar, g_vs=g_vs, mvector=mvector, g_vz=g_vz, mpseuvec=mpseuvec, g_va=g_va, kappa=kappa)*fraction
        else:
            dRdx_sm = get_Neutrino_dRdx_one(xt_range, data_path, A, Z, Sn, flux_table)
            dRdx_bsm = (get_Neutrino_dRdx_one(xt_range, data_path, A, Z, Sn, flux_table,\
                mscalar=mscalar, g_vs=g_vs, mvector=mvector, g_vz=g_vz, mpseuvec=mpseuvec, g_va=g_va, kappa=kappa) - dRdx_sm )* spin_fraction
            dRdx_neutrino += (dRdx_sm + dRdx_bsm) * fraction

    return dRdx_neutrino


def get_SM_Neutrino_dRdx_one(xt_range, data_path, A, Z, flux_table):
    
    Sn = 0 #for SM bkg
    E_universal = np.logspace(-3,3, 700)
    dRdE = get_dRdE_nu_SM(E_universal, A, Z, flux_table) #[coupling, E]
    dRdE_table = (E_universal, dRdE)

    dRdx = get_dRdx_general_one_target(xt_range, dRdE_table, data_path) # collapse the coupling dimension

    return dRdx


def get_SM_Neutrino_dRdx_mol(xt_range, mineral, flux_table):

    number = mineral.number_of_elements

    dRdx_neutrino = 0
    for i in range(number):
        data_path = mineral.derived_data_path[i]
        A = mineral.atomic_masses[i]
        Z = mineral.atomic_number[i]
        fraction = mineral.atomic_fractions[i]

        dRdx_neutrino += get_SM_Neutrino_dRdx_one(xt_range, data_path, A, Z, flux_table)*fraction

    return dRdx_neutrino

def get_BSM_Neutrino_dRdx_corr_one(xt_range, data_path, A, Z, Sn,flux_table, extra_term,mscalar=None, g_vs=None, mvector=None, g_vz=None, mpseuvec=None, g_va=None, kappa=1):
    
    E_universal = np.logspace(-3,3, 700)

    
    dRdE = get_dRdE_nu_BSMcorrection(E_universal, A, Z, Sn, flux_table, extra_term=extra_term, \
        mscalar=mscalar, g_vs=g_vs, mvector=mvector, g_vz=g_vz, mpseuvec=mpseuvec, g_va=g_va, kappa=kappa)
    dRdE_table = (E_universal, dRdE)

    dRdx = get_dRdx_general_one_target(xt_range, dRdE_table, data_path) # collapse the coupling dimension

    return dRdx

def get_BSM_Neutrino_dRdx_corr_mol(xt_range, mineral, flux_table, extra_term, mscalar=None, g_vs=None, mvector=None, g_vz=None, mpseuvec=None, g_va=None, kappa=1):
    
    number = mineral.number_of_elements

    dRdx_bsm_neutrino = 0
    for i in range(number):
        data_path = mineral.derived_data_path[i]
        A = mineral.atomic_masses[i]
        Z = mineral.atomic_number[i]
        S = mineral.nuclear_spin[i]
        fraction = mineral.atomic_fractions[i]
        spin_fraction = mineral.spin_isotopic_fractions[i]
        
        if mpseuvec == None:
            dRdx_bsm_neutrino += get_BSM_Neutrino_dRdx_corr_one(xt_range, data_path, A, Z, S,flux_table, extra_term=extra_term, \
                        mscalar=mscalar, g_vs=g_vs, mvector=mvector, g_vz=g_vz, mpseuvec=mpseuvec, g_va=g_va, kappa=kappa)*fraction
        else:
            dRdx_bsm_neutrino += get_BSM_Neutrino_dRdx_corr_one(xt_range, data_path, A, Z, S,flux_table, extra_term=extra_term, \
                        mscalar=mscalar, g_vs=g_vs, mvector=mvector, g_vz=g_vz, mpseuvec=mpseuvec, g_va=g_va, kappa=kappa)*fraction*spin_fraction

    return dRdx_bsm_neutrino

def window_function(xT, xa, xb, res):
    left_smearing = special.erf((xT-xa)/(np.sqrt(2)*res))
    right_smearing = special.erf((xT-xb)/(np.sqrt(2)*res))
    return 0.5 * (left_smearing - right_smearing)   


def get_binned_track_spec_method1(drdx, x, binwidth=1, integration_res=500):
    # drdx is numerical, so there's only a finite number of points, so first interpolate drdx first then pick the values at bin edges
    # track lengths are always in the last axis, rearrange all extra axis to the front
    if binwidth != 0:
        drdx_interp = interp1d(x, drdx, axis=-1,fill_value=0, bounds_error=False)
        xmin = x[0]
        xmax = x[-1]

        # xrange = np.linspace(xmin, xmax, int((xmax-xmin)/binwidth))
        R = []
        # for xi in range(len(xrange)-1):
        #     R.append(integrate.quad(drdx_interp, xrange[xi], xrange[xi+1])[0])
        for xi in range(len(x)-1):
            intRange = np.linspace(x[xi], x[xi+1],int(integration_res))
            R.append(np.trapz(drdx_interp(intRange), intRange, axis=-1))
    else:
        return
    R = np.array(R)
    R = np.moveaxis(R,0,-1)
    return (R, x)

def get_binned_track_spec(drdx, x,resolution=15, number_of_bins=100, window=True, logspace=False):
    # drdx is numerical, so there's only a finite number of points, so first interpolate drdx first then pick the values at bin edges
    # track lengths are always in the _=last axis, rearrange all extra axis to the front
    if logspace==True:
        bin_edges = np.logspace(np.log10(resolution/2), 3, number_of_bins+1)
    else:
        bin_edges = np.linspace(resolution/2, 1000, number_of_bins+1)

    interp_region = np.logspace(np.log10(resolution/2), 3, 700)

    # drdx_interp = np.exp(np.interp(np.log(interp_region), np.log(x), np.log(drdx), left=0, right=0))
    drdx_interp = np.interp(interp_region, x, drdx)
    binned_rate =[]
    if window == True:
        for i in range(number_of_bins):
            # convolved_rate = window_function(interp_region, bin_edges[i], bin_edges[i+1], resolution)
            convolved_rate = window_function(interp_region, bin_edges[i], bin_edges[i+1], resolution)
            integrand = drdx_interp*convolved_rate
            integral = np.trapz(integrand, interp_region, axis=-1)
            binned_rate.append(integral)
    else:
        for i in range(number_of_bins):
            int_range = np.logspace(np.log10(bin_edges[i]), np.log10(bin_edges[i+1]), 100)
            integrand = np.interp(int_range, interp_region, drdx_interp, left=0, right=0)
            integral = np.trapz(integrand, int_range)
            binned_rate.append(integral)
        
    binned_rate = np.array(binned_rate)
    binned_rate = np.moveaxis(binned_rate,0,-1)
    return (binned_rate, bin_edges)

def get_wimps_Nbins(sigma, xt_range, mx, mineral, resolution=15, number_of_bins=100, frac=0, spin='SI', window=True):

    dRdx = get_wimps_dRdx(mx, sigma, xt_range, mineral, frac=frac, spin=spin)[0]
    # rate, bin_edges = get_binned_track_spec(dRdx, xt_range, binwidth=binwidth, integration_res=integration_res)
    rate, bin_edges = get_binned_track_spec(dRdx, xt_range, resolution=resolution, number_of_bins=number_of_bins, window=window)
    return rate, bin_edges
def get_neutron_Nbins(C, xt_range, mineral, resolution=15, number_of_bins=100, window=True):
    
    dRdx = get_neutron_dRdx_mol(xt_range, mineral, C)[0]
    # rate, bin_edges = get_binned_track_spec(dRdx, xt_range, binwidth=binwidth, integration_res=integration_res)
    rate, bin_edges = get_binned_track_spec(dRdx, xt_range, resolution=resolution, number_of_bins=number_of_bins, window=window)

    return rate, bin_edges

def get_Th_Nbins(C, xt_range, resolution=15, number_of_bins=100, window=True):

    data = np.load('PaleoBSM/SRIM_derived_data/Backgrounds/Th/Tables/dist_at_72keV_01ppb.npy')
    dRdx_table = data *(C/1e-10)
    
    dRdx = np.interp(xt_range, np.logspace(-2,3,500), dRdx_table)

    rate, bin_edges = get_binned_track_spec(dRdx, xt_range, resolution=resolution, number_of_bins=number_of_bins, window=window)
    return rate, bin_edges

def get_neutrino_Nbins(xt_range, mineral, flux_table, mscalar=None, g_vs=None, mvector=None, g_vz=None, mpseuvec=None, g_va=None, resolution=15, number_of_bins=100, kappa=1, window=True):

    dRdx = get_Neutrino_dRdx_mol(xt_range, mineral, flux_table, \
            mscalar=mscalar, g_vs=g_vs, mvector=mvector, g_vz=g_vz, mpseuvec=mpseuvec, g_va=g_va, kappa=kappa)
    # rate, bin_edges = get_binned_track_spec(dRdx, xt_range, binwidth=binwidth, integration_res=integration_res)
    rate, bin_edges = get_binned_track_spec(dRdx, xt_range, resolution=resolution, number_of_bins=number_of_bins, window=window)

    return rate, bin_edges

def get_SM_neutrino_Nbins(xt_range, mineral, flux_table, resolution=15, number_of_bins=100):

    dRdx = get_SM_Neutrino_dRdx_mol(xt_range, mineral, flux_table)
    # rate, bin_edges = get_binned_track_spec(dRdx, xt_range, binwidth=binwidth, integration_res=integration_res)
    rate, bin_edges = get_binned_track_spec(dRdx, xt_range, resolution=resolution, number_of_bins=number_of_bins)

    return rate, bin_edges



def get_BSM_neutrino_Nbins_corr(xt_range, mineral, flux_table, extra_term, mscalar=None, g_vs=None, mvector=None, g_vz=None, mpseuvec=None, g_va=None, resolution=15, number_of_bins=100, kappa=1):

    dRdx = get_BSM_Neutrino_dRdx_corr_mol(xt_range, mineral, flux_table, extra_term=extra_term, \
            mscalar=mscalar, g_vs=g_vs, mvector=mvector, g_vz=g_vz, mpseuvec=mpseuvec, g_va=g_va, kappa=kappa)
    # rate, bin_edges = get_binned_track_spec(dRdx, xt_range, binwidth=binwidth, integration_res=integration_res)
    rate, bin_edges = get_binned_track_spec(dRdx, xt_range, resolution=resolution, number_of_bins=number_of_bins)

    return rate, bin_edges

# , window=True_all_cntrl)*mineral_mass*t_age
    return N_all_cntrl


# --------------------------- SP only ---------------------------------------------

def get_dRdx_general_spOnly_one_target(xt_range, dRdE_table, composition):

    SP_files = ['OOli_SRIM.txt', 'SiOli_SRIM.txt', 'FeOli_SRIM.txt', 'MgOli_SRIM.txt']
    index = SP_files.index(composition+'Oli_SRIM.txt')
    baum_fraction = np.array([0.41743753, 0.18319438, 0.14570679, 0.2536613 ])

    dEdx = np.loadtxt(f'/Users/szechingaudreyfung/paleo/paleoSpec/Olivine/Ranges/{SP_files[index]}')
    dEdx = np.moveaxis(dEdx, 0,1)
    dEdx_interp = interp1d(dEdx[2]/10, dEdx[1]/baum_fraction[index]*10, fill_value=0, bounds_error=False)(xt_range)
    Erange_interp = interp1d(dEdx[2]/10, dEdx[0], fill_value=0, bounds_error=False)(xt_range)
    dRdE_interp = interp1d(dRdE_table[0], dRdE_table[1], fill_value=0, bounds_error=False)(Erange_interp)

    dRdx_sp = dRdE_interp * dEdx_interp

    return dRdx_sp

def get_wimps_spOnly_dRdx_one(mx, sigma, xt_range, A, composition, frac=0, spin='SI'):

    E_universal = np.logspace(-3,3, 700)
    dRdE_table = get_drder_one(E_universal, A, mx, sigma, frac=frac, spin=spin)

    dRdx_sp = get_dRdx_general_spOnly_one_target(xt_range, dRdE_table, composition)

    return dRdx_sp 

def get_wimps_spOnly_dRdx(mx, sigma, xt_range, mineral, frac=0, spin='SI'):

    dRdx_dm = 0
    for i in range(4):
        A = mineral.atomic_masses[i]
        composition =mineral.composition[i]
        fraction = mineral.atomic_fractions[i]

        dRdx_dm += get_wimps_spOnly_dRdx_one(mx, sigma, xt_range, A, composition, frac=0, spin='SI')*fraction
    return dRdx_dm

def get_neutron_spOnly_dRdx_one(xt_range, data_path, composition, molecule_mass,C):


    try:
        extraDim = len(C)
    except TypeError:
        extraDim = 0
    
    if extraDim == 0:
        C = np.array([C])
    
    E_universal = np.logspace(-3,3, 700)
    
    dRdE_load = np.load(data_path+'Tables/neutron_recoil.npy')

    dRdE_interp = interp1d(dRdE_load[0], dRdE_load[1][np.newaxis, ...]*(C[..., np.newaxis]/1e-10)*(molecule_mass/238.05), \
                           axis=-1 ,fill_value=0, bounds_error=False)(E_universal)
    
    dRdE_table = (E_universal, dRdE_interp)

    dRdx = get_dRdx_general_spOnly_one_target(xt_range, dRdE_table, composition)

    return dRdx


def get_neutron_spOnly_dRdx_mol(xt_range, mineral, C):

    number = mineral.number_of_elements
    num_frac = np.array(mineral.number_fractions)
    atomic_mass = np.array(mineral.atomic_masses)
    molecule_mass = np.sum(num_frac*atomic_mass)
    

    dRdx_neutron = 0
    for i in range(number):
        composition = mineral.composition[i]
        data_path = mineral.derived_data_path[i]
        fraction = mineral.atomic_fractions[i]
        dRdx_neutron += get_neutron_spOnly_dRdx_one(xt_range, data_path, composition, molecule_mass,C)*fraction
    return dRdx_neutron

def get_Th_spOnly_dRdx_mol(x, C, center=26.135851995198294, epsilon=1e-6):
    # There is no dRdx_one for Thorium as simulations are for the entire olivine
    absdiff = abs(x-center)
    min_idx = np.where(absdiff == absdiff.min())[0][0]
    new_center = x[min_idx]
    normalisation = 10**7 * C/1e-10
    func = np.exp(-((x - new_center) ** 2) / (2 * epsilon**2)) / (np.sqrt(2 * np.pi) * epsilon)
    return func/np.trapz(func, x) * normalisation

def get_SM_Neutrino_spOnly_dRdx_one(xt_range, composition, A, Z, flux_table):
    
    Sn = 0 #for SM bkg
    E_universal = np.logspace(-3,3, 700)
    dRdE = get_dRdE_nu(E_universal, A, Z, Sn, flux_table) #[coupling, E]
    dRdE_table = (E_universal, dRdE)

    dRdx = get_dRdx_general_spOnly_one_target(xt_range, dRdE_table, composition) # collapse the coupling dimension

    return dRdx



def get_SM_Neutrino_spOnly_dRdx_mol(xt_range, mineral, flux_table):

    number = mineral.number_of_elements

    dRdx_neutrino = 0
    for i in range(number):
        composition = mineral.composition[i]
        A = mineral.atomic_masses[i]
        Z = mineral.atomic_number[i]
        fraction = mineral.atomic_fractions[i]

        dRdx_neutrino += get_SM_Neutrino_spOnly_dRdx_one(xt_range, composition, A, Z, flux_table)*fraction

    return dRdx_neutrino


def get_wimps_spOnly_Nbins(sigma, xt_range, mx, mineral, resolution=15, number_of_bins=100):

    dRdx = get_wimps_spOnly_dRdx(mx, sigma, xt_range, mineral, frac=0, spin='SI')[0]
    # rate, bin_edges = get_binned_track_spec(dRdx, xt_range, binwidth=binwidth, integration_res=integration_res)
    rate, bin_edges = get_binned_track_spec(dRdx, xt_range, resolution=resolution, number_of_bins=number_of_bins)
    return rate, bin_edges
def get_neutron_spOnly_Nbins(C, xt_range, mineral, resolution=15, number_of_bins=100):
    
    dRdx = get_neutron_spOnly_dRdx_mol(xt_range, mineral, C)[0]
    # rate, bin_edges = get_binned_track_spec(dRdx, xt_range, binwidth=binwidth, integration_res=integration_res)
    rate, bin_edges = get_binned_track_spec(dRdx, xt_range, resolution=resolution, number_of_bins=number_of_bins)

    return rate, bin_edges

def get_Th_spOnly_Nbins(C, xt_range, resolution=15, number_of_bins=100):
    
    dRdx = get_Th_spOnly_dRdx_mol(xt_range, C)
    # rate, bin_edges = get_binned_track_spec(dRdx, xt_range, binwidth=binwidth, integration_res=integration_res)
    rate, bin_edges = get_binned_track_spec(dRdx, xt_range, resolution=resolution, number_of_bins=number_of_bins)

    return rate, bin_edges

def get_neutrino_spOnly_Nbins(xt_range, mineral, flux_table , resolution=15, number_of_bins=100):

    dRdx = get_SM_Neutrino_spOnly_dRdx_mol(xt_range, mineral, flux_table)
    # rate, bin_edges = get_binned_track_spec(dRdx, xt_range, binwidth=binwidth, integration_res=integration_res)
    rate, bin_edges = get_binned_track_spec(dRdx, xt_range, resolution=resolution, number_of_bins=number_of_bins)

    return rate, bin_edges


# def get_N_all_spOnly_cntrl_array(sigma, mx, xt_range,mineral, mineral_mass=1, t_age=1 ,C_cntrl=1e-10, resolution=15, number_of_bins=100, window=True):
    
#     N_wimps_ctrl = get_wimps_spOnly_Nbins(sigma, xt_range, mx, mineral,resolution=resolution, number_of_bins=number_of_bins)[0]
#     N_neutron_ctrl = get_neutron_spOnly_Nbins(C_cntrl, xt_range, mineral,resolution=resolution, number_of_bins=number_of_bins)[0]

#     N_all_cntrl = [N_wimps_ctrl]
#     N_solar_neutrino_cntrl = 0
#     for i, source in enumerate(sources[:-3]):
#         flux_i = np.load(neutrino_data_path + f'extrapolate_{source}_fluxes.npy')
#         N_solar_neutrino_cntrl += get_neutrino_spOnly_Nbins(xt_range, mineral, flux_i, resolution=resolution, number_of_bins=number_of_bins)[0]
        
#     N_all_cntrl.append(N_solar_neutrino_cntrl)
#     for i, source in enumerate(sources[-3:]):
#         flux_i = np.load(neutrino_data_path + f'extrapolate_{source}_fluxes.npy')
#         N_neutrino = get_neutrino_spOnly_Nbins(xt_range, mineral, flux_i, resolution=resolution, number_of_bins=number_of_bins)[0]
#         N_all_cntrl.append(N_neutrino)
    
#     N_all_cntrl.append(N_neutron_ctrl)
#     N_all_cntrl = np.array(N_all_cntrl)*mineral_mass*t_age
#     return N_all_cntrl
