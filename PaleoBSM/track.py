from WIMPS.compute_wimps import get_drder, get_drder_one
from minerals_def import olivineObj
from compute_trackSpectra import get_wimps_Nbins, get_neutron_Nbins, get_neutrino_Nbins, get_wimps_dRdx_one, get_SM_Neutrino_dRdx_mol, get_Neutrino_dRdx_mol
from compute_trackSpectra import get_wimps_dRdx, get_neutron_dRdx_mol, get_wimps_Nbins, get_neutron_Nbins
from light_mediators import get_dRdE_nu
import numpy as np
from scipy.interpolate import interp1d

# test new spOnly function
if False:
    mx = 5
    sigma = 1e-45
    # xt_range = np.logspace(-2,3,500)
    xt_range = np.logspace(np.log10(7.5),3,71)
    mineral = olivineObj
    neutrino_data_path = '/Users/szechingaudreyfung/PaleoBSMwithTRIM/neutrino_fluxes/'
    flux_table = np.load(neutrino_data_path + f'extrapolate_solar_pp_fluxes.npy')

    # data_path= mineral.derived_data_path[0]
    # composition= mineral.composition[0]
    # A = mineral.atomic_masses[0]
    # Z = mineral.atomic_number[0]

    # drdx = get_SM_Neutrino_dRdx_one(xt_range, data_path, A, Z, flux_table)
    # drdx_sp = get_SM_Neutrino_spOnly_dRdx_one(xt_range, composition, A, Z, flux_table)
    # drdx_sp = get_wimps_spOnly_dRdx(mx, 1e-45, xt_range, mineral, frac=0, spin='SI')
    rate= get_neutrino_Nbins(xt_range, mineral, flux_table)[0]
    rate_sp= get_neutrino_spOnly_Nbins(xt_range, mineral, flux_table)[0]
    print(rate.shape, rate_sp.shape)
    # print(drdx_sp.shape)
    # np.save('/Users/szechingaudreyfung/PaleoBSMwithTRIM/test_results/spOnly_wimps_test_1e45', drdx_sp)

# test new tracks function 

if False:
    mx = 5
    sigma = 1e-45
    C = 1e-10
    mineral = olivineObj
    resolution = 15
    xt_range = np.logspace(np.log10(resolution/2),3,71)

    drdx = get_wimps_dRdx(mx, sigma, xt_range, mineral, frac=0, spin='SI')
    # N_neutron, be = get_neutron_Nbins(C, xt_range, mineral)

    # neutrino_data_path = '/Users/szechingaudreyfung/PaleoBSMwithTRIM/neutrino_fluxes/'
    # solar_flux_i = np.load(neutrino_data_path + f'extrapolate_solar_pp_fluxes.npy')
    R1, bin_edges1 = get_binned_track_spec_method1(drdx, xt_range)
    R2, bin_edges2 = get_binned_track_spec_method2(drdx, xt_range, resolution=resolution)


    print(R1.shape, bin_edges1.shape, R2.shape, bin_edges2.shape)
    np.save('/Users/szechingaudreyfung/PaleoBSMwithTRIM/test_results/R1_test', R1)
    np.save('/Users/szechingaudreyfung/PaleoBSMwithTRIM/test_results/R1_edges_test', bin_edges1)
    np.save('/Users/szechingaudreyfung/PaleoBSMwithTRIM/test_results/R2_test', R2)
    np.save('/Users/szechingaudreyfung/PaleoBSMwithTRIM/test_results/R2_edges_test', bin_edges2)

# test new dRdE function
if False:
    neutrino_data_path = '/Users/szechingaudreyfung/PaleoBSMwithTRIM/neutrino_fluxes/'
    solar_sources = ['pp','hep', 'B', 'F', 'Be862', 'Be384', 'N13', 'O15', 'pep']

    dRdx_new = 0
    dRdx_old = 0

    j = 0
    AT = olivineObj.atomic_masses[j]
    ZT = olivineObj.atomic_number[j]
    Sn = olivineObj.nuclear_spin[j]
    data_path = olivineObj.derived_data_path[j]
    xt_range = np.logspace(-2,3,500)


    for i, source in enumerate(solar_sources):
        solar_flux_i = np.load(neutrino_data_path + f'extrapolate_solar_{source}_fluxes.npy')

        dRdx_new += get_SM_Neutrino_dRdx_one(xt_range, data_path, AT, ZT, solar_flux_i)[0]* olivineObj.atomic_fractions[j]
        dRdx_old += get_bkgNeutrino_dRdx_one(xt_range, data_path, AT, ZT, solar_flux_i)* olivineObj.atomic_fractions[j]

    print(dRdx_new.shape)
    print(dRdx_old.shape)
    print(np.all(dRdx_new == dRdx_old))


    # np.save('/Users/szechingaudreyfung/PaleoBSMwithTRIM/test_results/dRdE_split_solar_neutrino_sources', dRdE)
        




# binned rates

# binwidth = 1
# xt_range = np.logspace(-2,3,500)
# mineral = olivineObj
# mx = 0.5

# sigma = 1e-43

# dRdx DM with SRIM

if False:
    xt_range = np.logspace(-2,3,500)
    mx = 5
    sigma = np.array([1e-43])
    # N_dist = []
    # sigma_range = np.linspace(3e-47/1e-45, 3e-45/1e-45,20)*1e-45
    # for sigma in sigma_range:
    #     N, bin_edges = get_wimps_Nbins(sigma, xt_range, mx, olivineObj)
    #     bin_cntrl = (bin_edges[1:] + bin_edges[-1:])/2
    #     N_dist.append(np.trapz(N, bin_cntrl))

    dRdx_dm = 0
    for i in range(4):
        A = olivineObj.atomic_masses[i]
        data_path =olivineObj.derived_data_path[i]
        fraction = olivineObj.atomic_fractions[i]

        dRdx_dm += get_wimps_dRdx_one(mx, sigma, xt_range, A, data_path)*fraction
    
    
    # np.save('/Users/szechingaudreyfung/PaleoBSMwithTRIM/test_results/N_total_vs_sig_debug', N_dist)
    # np.save(f'/Users/szechingaudreyfung/Desktop/Dark Interactions/arrays_to_plot/drdx_olivine_{mx}GeV_5e-47', dRdx_dm)
    # print(len(N_dist))
    print(f'DM shape: {dRdx_dm.shape}')

# light mediators neutrino dRdx with srim
if True:
    xt_range = np.logspace(-2,3,500)
    mineral=olivineObj
    neutrino_data_path = '/Users/szechingaudreyfung/PaleoBSMwithTRIM/neutrino_fluxes/'
    solar_sources = ['solar_pp', 'solar_B', 'solar_F', 'solar_Be862', 'solar_Be384', 'solar_N13', \
               'solar_O15', 'solar_pep', 'solar_hep'] # only solar sources are included
    
    # mpseuvec = 1# GeV
    # g_va = 5e-2
    # mscalar = 1
    # g_vs = 1e-3
    print(olivineObj.spin_isotopic_fractions)
    dRdx = 0
    for i, source in enumerate(solar_sources):
        flux_i = np.load(neutrino_data_path + f'extrapolate_{source}_fluxes.npy')
        dRdx += get_Neutrino_dRdx_mol(xt_range, mineral, flux_i)
        np.save('/Users/szechingaudreyfung/paleo/saved_results/drdx_solar_nu_', dRdx)


# binned tracks
if False:
    mx = 5
    sigma = np.array([1e-43])
    xt_range = np.logspace(-2,3,500)
    # rate, bin_edges = get_wimps_Nbins(sigma, xt_range, mx, olivineObj, resolution=10, number_of_bins=100)
    R_neu, bin_neu = get_neutron_Nbins(1e-11, xt_range, olivineObj, resolution=10, number_of_bins=100)
    # np.save(f'/Users/szechingaudreyfung/PaleoBSMwithTRIM/test_results/N_binned_{mx}', rate)
    # np.save(f'/Users/szechingaudreyfung/PaleoBSMwithTRIM/test_results/bin_edges_{mx}', bin_edges)

    np.save('/Users/szechingaudreyfung/PaleoBSMwithTRIM/test_results/N_neu_binned', R_neu)
    np.save('/Users/szechingaudreyfung/PaleoBSMwithTRIM/test_results/bin_neu', bin_neu)

    

# neutron dRdx with srim
if False:
    xt_range = np.logspace(-2,3,500)
    # dRdx_neutron = 0
    # for i in range(4):
    #     data_path = olivineObj.derived_data_path[i]
    #     num_frac = olivineObj.number_fractions[i]

    #     dRdx_neutron += get_neutron_dRdx_one(xt_range, data_path, 153.31,1e-10)
    dRdx_neutron = get_neutron_dRdx_mol(xt_range, olivineObj,1e-10)
    
    np.save('/Users/szechingaudreyfung/paleo/saved_results/drdx_bkg_neutron_01ppb', dRdx_neutron)

# neutrino dRdx with srim
if False:
    xt_range = np.logspace(-2,3,500)
    mineral=olivineObj
    neutrino_data_path = '/Users/szechingaudreyfung/PaleoBSMwithTRIM/neutrino_fluxes/'
    sources = ['solar_pp', 'solar_B', 'solar_F', 'solar_Be862', 'solar_Be384', 'solar_N13', \
               'solar_O15', 'solar_pep', 'solar_hep', 'DSNB','ATM', 'GSNB'] # hep is removed, is not optimzied
    # sources = ['DSNB']
    dRdx_bkgNu = 0
    for i, source in enumerate(sources[-1:]):
        flux_i = np.load(neutrino_data_path + f'extrapolate_{source}_fluxes.npy')
        dRdx_bkgNu += get_SM_Neutrino_dRdx_mol(xt_range, mineral, flux_i)
    np.save('/Users/szechingaudreyfung/Desktop/Dark Interactions/arrays_to_plot/drdx_bkg_neutrino_GSNB', dRdx_bkgNu)


# test with Stopping power
if False:
    xt_range = np.logspace(-2,3,500)
    SP_files = ['OOli_SRIM.txt', 'SiOli_SRIM.txt', 'FeOli_SRIM.txt', 'MgOli_SRIM.txt']
    baum_fraction = np.array([0.41743753, 0.18319438, 0.14570679, 0.2536613 ])
    neutrino_data_path = '/Users/szechingaudreyfung/PaleoBSMwithTRIM/neutrino_fluxes/'
    sources = ['solar_pp', 'solar_B', 'solar_F', 'solar_Be862', 'solar_Be384', 'solar_N13', \
               'solar_O15', 'solar_pep', 'solar_hep']
    # sources = ['DSNB','ATM', 'GSNB']
    if False:
        mx = 100
        sigma  = np.array([1e-43])
    if True:
        mpseuvec = 1e-2 # GeV
        g_va = 1e-4

    # dRdx_sp = 0
    # dRdx_sp_neutron = 0
    dRdx_sp_neutrino = 0
    # dRdE_DM = []
    for i in range(4):

        data_path =olivineObj.derived_data_path[i]
        fraction = olivineObj.atomic_fractions[i]
        A = olivineObj.atomic_masses[i]
        Z = olivineObj.atomic_number[i]
        Sn = olivineObj.nuclear_spin[i]
        spin_fraction = olivineObj.spin_isotopic_fractions[i]
        Erange = np.logspace(-3,3, 700)

        # neutrino
        if True:
            dRdE_neutrino = 0
            for source in sources:
                flux_j = np.load(neutrino_data_path + f'extrapolate_{source}_fluxes.npy')
                dRdE_neutrino += get_dRdE_nu(Erange, A, Z, Sn, flux_j, mpseuvec=mpseuvec, g_va=g_va)
        # neutron
        if False:
            C = 1e-10
            dRdE_neutron_load = np.load(data_path+'Tables/neutron_recoil.npy')
        # DM
        if False:
            dRdE = get_drder_one(Erange, A, mx, sigma)[1]
        
        dEdx = np.loadtxt(f'/Users/szechingaudreyfung/paleo/paleoSpec/Olivine/Ranges/{SP_files[i]}')
        dEdx = np.moveaxis(dEdx, 0,1)
        dEdx_interp = interp1d(dEdx[2]/10, dEdx[1]/baum_fraction[i]*10, fill_value=0, bounds_error=False)(xt_range)
        Erange_interp = interp1d(dEdx[2]/10, dEdx[0], fill_value=0, bounds_error=False)(xt_range)

        # dRdE_interp_neutron = interp1d(dRdE_neutron_load[0], dRdE_neutron_load[1], fill_value=0, bounds_error=False)(Erange_interp)*153.31/238.051*(C/1e-10)
        dRdE_interp_neutrino = interp1d(Erange, dRdE_neutrino, fill_value=0, bounds_error=False)(Erange_interp)
        # dRdE_interp = interp1d(Erange, dRdE, fill_value=0, bounds_error=False)(Erange_interp)

        # dRdE_DM.append(dRdE)

        # dRdx_sp_neutron += dRdE_interp_neutron * dEdx_interp * fraction
        dRdx_sp_neutrino += dRdE_interp_neutrino * dEdx_interp * fraction * spin_fraction
        # dRdx_sp += dRdE_interp * dEdx_interp * fraction
    # dRdE_DM = np.array(dRdE_DM)

    # np.save(f'/Users/szechingaudreyfung/Desktop/Dark Interactions/arrays_to_plot/drdE_olivine_{mx}GeV_1e-45', dRdE_DM)
    # np.save(f'/Users/szechingaudreyfung/Desktop/Dark Interactions/arrays_to_plot/drdx_olivine_{mx}GeV_5e-47_sp', dRdx_sp)
    # np.save(f'/Users/szechingaudreyfung/paleo/saved_results/drdx_bkg_neutron_01ppb_sp', dRdx_sp_neutron)
    # np.save(f'/Users/szechingaudreyfung/paleo/saved_results/drdx_neutrino_SM_GSNB_sp', dRdx_sp_neutrino)
    np.save('/Users/szechingaudreyfung/paleo/saved_results/drdx_mediators_axialVec_1keV_1e-4_sp', dRdx_sp_neutrino)
    # print(f'SP shape: {dRdx_sp.shape}')

