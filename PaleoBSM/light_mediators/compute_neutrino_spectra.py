import numpy as np
from scipy.interpolate import interp1d

##### temporary functions for debugging ########################################

def load_Aarons_neutrinoflux_data():
    # Allfluxes[source][rows,column]
    # raw data: col 0 = energy[MeV] ; col 1 fluxes [1/(cm^2 s MeV)]
    # output: col0 in keV; col 1 in [1/(cm^2 s keV)]
    neutrino_dir = '/Users/szechingaudreyfung/PaleoBSMwithTRIM/neutrino_fluxes/solar_neutrinos/nuSpectra/'
    
    normalization = np.array([5.98e10,8.04e3,5.58e6,5.52e6,5.00e9*0.9,5.00e9*0.1,2.96e8,2.23e8,1.44e8])
    # normalization = np.array([5.941e10, 3e4, 5.2e6, 5.51e7, 4.93e9*0.9, 4.93e9*0.1, 3.48e8, 2.53e8, 1.421e8]) # from 2311.16226
    pp = np.genfromtxt(neutrino_dir+'solarspectrum0.txt')
    pp[:,1] = pp[:,1]*1e-3
    pp[:,0] = pp[:,0]*1e3

    hep = np.genfromtxt(neutrino_dir+'solarspectrum1.txt')
    hep[:,1] = hep[:,1]*1e-3
    hep[:,0] = hep[:,0]*1e3 

    B = np.genfromtxt(neutrino_dir+'B8_neutrino_winters.txt')
    B[:,1] = B[:,1]/1000*1e-3
    B[:,0] = B[:,0]*1e3

    F = np.genfromtxt(neutrino_dir+'solarspectrum5.txt')
    F[:,1] = F[:,1]*1e-3
    F[:,0] = F[:,0]*1e3

    Be862 = np.genfromtxt(neutrino_dir+'solarspectrum6.txt')
    Be862[:,1] = Be862[:,1]*1e-3
    Be862[:,0] = Be862[:,0]*1e3

    Be384 = np.genfromtxt(neutrino_dir+'solarspectrum7.txt')
    Be384[:,1] = Be384[:,1]*1e-3
    Be384[:,0] = Be384[:,0]*1e3

    N13 = np.genfromtxt(neutrino_dir+'solarspectrum3.txt')
    N13[:,1] = N13[:,1]*1e-3
    N13[:,0] = N13[:,0]*1e3

    O15 = np.genfromtxt(neutrino_dir+'solarspectrum4.txt')
    O15[:,1] = O15[:,1]*1e-3
    O15[:,0] = O15[:,0]*1e3

    pep = np.genfromtxt(neutrino_dir+'solarspectrum8.txt')
    pep[:,1] = pep[:,1]*1e-3
    pep[:,0] = pep[:,0]*1e3

    Allfluxes = [pp,hep, B, F, Be862, Be384, N13, O15, pep]
    normed_fluxes = []
    
    for i, flux in enumerate(Allfluxes):
        normed_flux = flux
        normed_flux[:,1] = flux[:,1]/np.trapz(flux[:,1], flux[:,0]) * normalization[i]
        normed_fluxes.append(normed_flux)
    return normed_fluxes


def extrapolate_solar_neutrino_fluxes_except(flux_factor_solar=1,exclude=100, Npts=3000):
    # Ev in keV
    # output in [source, normalized flux, E(keV)]
    Ev = np.logspace(0,6,Npts)
    data = load_Aarons_neutrinoflux_data()
    if type(flux_factor_solar) == int:
        flux_factor_solar = np.ones((len(data)))
    extrapolated_data = 0
    for i, source in enumerate(data):
        if i != exclude:
            E_above = np.where(Ev>source[-1,0])[0][0]
            E_below = np.where(Ev>source[0,0])[0][0]
            if E_above == E_below:
                total_flux = np.trapz(source[:,1], source[:,0])
                extrap_flux = np.zeros_like(Ev)
                extrap_flux[E_above] = total_flux
                extrapolated_data += extrap_flux

            else:
                logflux = interp1d(np.log(source[:,0]), np.log(source[:,1]), \
                            fill_value=(-1000, -1000), bounds_error=False)(np.log(Ev))
                extrapolated_data += np.exp(logflux)*flux_factor_solar[i]
        else:
            print(f'data {i} is excluded')
               
    return np.array([Ev, extrapolated_data])

def extrapolate_solar_neutrino_fluxes_single(include, Npts=3000):
    # Ev in keV
    # output in [source, normalized flux, E(keV)]
    Ev = np.logspace(0,6,Npts)

    data = load_Aarons_neutrinoflux_data()[include]
    extrapolated_data = 0
    E_above = np.where(Ev>data[-1,0])[0][0]
    E_below = np.where(Ev>data[0,0])[0][0]
    if E_above == E_below:
        total_flux = np.trapz(data[:,1], data[:,0])
        extrap_flux = np.zeros_like(Ev)
        extrap_flux[E_above] = total_flux
        extrapolated_data += extrap_flux

    else:
        logflux = interp1d(np.log(data[:,0]), np.log(data[:,1]), \
                    fill_value=(-1000, -1000), bounds_error=False)(np.log(Ev))
        extrapolated_data += np.exp(logflux)
            
        
    return np.array([Ev, extrapolated_data])



################################################################################



def Fn2SI(Er_kev,A):
    # calculate form factor squared
    # Input Er in keV
    # from 0608035 eq 9, 10, 11
    # M: nuclear mass, R1: effective nuclear radius, s: nuclear skin thickness

    # convert Er from keV to MeV
    Er = Er_kev*(1e-3)

    # calcualte nuclear radius and skin thickness
    s = 0.9 # fm
    R1 = ((1.23*A**(1/3)-0.6)**2+(7/3)*np.pi**2*0.52**2 - 5*s**2)**0.5 # in fm

    #convert fm to MeV
    hbarc = (0.1973269804)*(1e-6)*(1e9) #MeV*fm
    R1_mev = R1/hbarc # [MeV^-1]
    s_mev = s/hbarc #[MeV^-1]

    # put things together
    M = 931.5*A # in MeV
    q = (2*M*Er)**(0.5) # in MeV
    # print(Er[:10])
    j1 = np.sin(q*R1_mev)/(q*R1_mev)**2 - np.cos(q*R1_mev)/(q*R1_mev)
    Fn2 = ((np.exp(-q**2*s_mev**2/2))*(3*j1/(q*R1_mev)))**2

    # normalisation
    nanidx = np.where(np.isnan(Fn2))
    Fn2[nanidx] = 1
    return Fn2

def calc_dsigdEr_neu_sm(Ev_kev, Er_kev, An, Zn):
    # to be consistent with other sources of dRdE, use energies in kev
    # Er[keV], Ev[keV]
    # An = mass number (weighted average over isotopes)
    # Zn = proton number (integer)
    # out shape: (len(Er), len(Ev))
    # out unit = cm^2/kev
    # Ev_min = sqrt(atomic_mass*Er/2)

    Ev_kev = np.array(Ev_kev)
    Er_kev = np.array(Er_kev)

    sin2w = 0.2387# from 1604.01025 pg 3 or 0409169 pg 18 # 0.23121 from paleoSpec 
    GF_gev = 1.1663787e-5
    hbarc = 6.582e-16 * 3e10/1e9 #GeV s * cm/s

    Qw = (An-Zn) - (1-4*sin2w)*Zn
    F2 = Fn2SI(Er_kev,An)[:,None]

    # unit conversion, add dimension
    Ev_gev = (Ev_kev*1e-6)[None,:]
    Er_gev = (Er_kev*1e-6)[:,None]
    atomic_mass = An*931.49432/1e3 #amu to GeV

    dsigdE = Qw**2*atomic_mass*(1-atomic_mass*Er_gev/(2*Ev_gev**2))*F2
    # mask = atomic_mass*Er_gev/(2*Ev_gev**2)
    dsigdE = GF_gev**2/4/np.pi*dsigdE*hbarc**2*1e-6

    return dsigdE*(dsigdE>0)

def calc_dsigdE_scalar(Ev_kev, Er_kev, An, Zn, mscalar, g_vs, extra_term):
    # to be consistent with other sources of dRdE, use energies in kev
    # Er[keV], Ev[keV]
    # An = mass number (weighted average over isotopes)
    # Zn = proton number (integer)
    # out shape: (len(g_vs), len(Er), len(Ev))
    # out unit = cm^2/kev
    # Ev_min = sqrt(atomic_mass*Er/2)
    # g_vs can be an array
    # m_new in GeV
    # accept only 1 mass at a time
    # create dimensions
    try:
        g_length = len(g_vs)
    except:
        g_vs = [g_vs]

    if (g_vs[0] == None) & (mscalar == None):

        return np.zeros((len(Er_kev),len(Ev_kev)))

    g_vs = np.array(g_vs)[:, None, None]

    g_qs = g_vs
    Qsprime = (14*An+1.1*Zn)*g_vs*g_qs
    hbarc = 6.582e-16 * 3e10 #eV s * cm/s

    # convert to GeV
    Ev_kev = np.array(Ev_kev)
    Er_kev = np.array(Er_kev)
    Ev_gev = (Ev_kev*1e-6)[None, None,:]
    Er_gev = (Er_kev*1e-6)[None, :,None]
    atomic_mass = An*931.49432/1e3 # amu to GeV


    dsigdE = Qsprime**2 * atomic_mass**2 * Er_gev/(4*np.pi* Ev_gev**2 * (2*Er_gev*atomic_mass + mscalar**2)**2)
#     mask = atomic_mass*Er_gev/(2*Ev_gev**2)
    dsigdE = dsigdE*hbarc**2*1e-24

    if extra_term == 1:
        return dsigdE[0] # for now we only pass in one g, so collapse one dimension
    else: 
        return 0
    

def calc_dsigdE_vector(Ev_kev, Er_kev, An, Zn, mvector, g_vz, extra_term, kappa=1):
    # to be consistent with other sources of dRdE, use energies in kev
    # Er[keV], Ev[keV]
    # An = mass number (weighted average over isotopes)
    # Zn = proton number (integer)
    # In units: m_new in GeV
    # out shape: (len(coupling), len(Er), len(Ev))
    # out unit = cm^2/kev
    # Ev_min = sqrt(atomic_mass*Er/2)
    try:
        g_length = len(g_vz)
    except:
        g_vz = [g_vz]

    if (g_vz[0] == None) & (mvector == None):
        return np.zeros((len(Er_kev),len(Ev_kev)))

    g_vz = np.array(g_vz)[:, None, None]
    g_qz = g_vz
    # constants
    sin2theta = 0.2387 # from 1604.01025 pg 3
    Qvprime = 3*An*g_vz*g_qz
    Qv = (An-Zn) - (1-4*sin2theta)*Zn
    GF_gev = 1.1663787e-5
    hbarc = 6.582e-16 * 3e10 #eV s * cm/s

    # unit conversions
    Ev_kev = np.array(Ev_kev)
    Er_kev = np.array(Er_kev)
    Ev_gev = (Ev_kev*1e-6)[None, None,:]
    Er_gev = (Er_kev*1e-6)[None, :,None]
    atomic_mass = An*931.49432/1e3 # amu to GeV


    if extra_term == 1:
        dsigdE = -kappa*GF_gev*atomic_mass*Qv*Qvprime*(2*Ev_gev**2 - Er_gev* atomic_mass)/\
            (8**0.5*np.pi*Ev_gev**2*(2*Er_gev*atomic_mass + mvector**2))
        
    
    elif extra_term == 2:
        dsigdE = kappa**2*Qvprime**2 * atomic_mass * (2*Ev_gev**2 - Er_gev*atomic_mass)/\
            (4*np.pi*Ev_gev**2 * (2*Er_gev * atomic_mass + mvector **2)**2)
    else: 
        dsigdE = np.zeros_like(Ev_gev)
    
    dsigdE = dsigdE*hbarc**2/1e24

    return dsigdE[0]


def calc_dsigdE_pseuvec(Ev_kev, Er_kev, An, Zn, Sn, mpseuvec, g_va, extra_term):
    # to be consistent with other sources of dRdE, use energies in kev
    # Er[keV], Ev[keV]
    # An = mass number (weighted average over isotopes)
    # Zn = proton number (integer)
    # Sn = nucleus spin
    # out shape: (len(Er), len(Ev))
    # out unit = cm^2/kev
    # Ev_min = sqrt(atomic_mass*Er/2)
    try:
        g_length = len(g_va)
    except:
        g_va = [g_va]

    if (g_va[0] == None) & (mpseuvec == None):
        return np.zeros((len(Er_kev),len(Ev_kev)))


    g_va = np.array(g_va)[:, None, None]

    g_qa = g_va
    GF_gev = 1.1663787e-5
    hbarc = 6.582e-16 * 3e10 #eV s * cm/s
    Qa = 1.3*Sn
    Qaprime = 0.3*Sn*g_va*g_qa
    sin2theta = 0.2387 # from 1604.01025 pg 3
    Qv = (An-Zn) - (1-4*sin2theta)*Zn

    Ev_kev = np.array(Ev_kev)
    Er_kev = np.array(Er_kev)
    Ev_gev = (Ev_kev*1e-6)[None, None,:]
    Er_gev = (Er_kev*1e-6)[None, :,None]
    atomic_mass = An*931.49432/1e3 # amu to GeV


    if extra_term == 1:
        dsigdE = GF_gev*atomic_mass*Qa*Qaprime*(2*Ev_gev**2 + atomic_mass*Er_gev)/ \
                (8**0.5*np.pi*Ev_gev**2*(2*Er_gev*atomic_mass + mpseuvec**2)) \
                - GF_gev*atomic_mass * Qv*Qaprime *Ev_gev*Er_gev/ \
                (2**0.5*np.pi*Ev_gev**2 * (2*Er_gev*atomic_mass + mpseuvec**2)) 
        
    elif extra_term == 2:
        dsigdE = Qaprime**2 * atomic_mass*(2*Ev_gev**2 + Er_gev*atomic_mass)/ \
                (4*np.pi*Ev_gev**2*(2*Er_gev*atomic_mass + mpseuvec**2)**2)
        
    else:
        dsigdE = np.zeros_like(Er_gev)
    
    # mask = atomic_mass*Er_gev/(2*Ev_gev**2)
    dsigdE = dsigdE*hbarc**2/1e24

    return dsigdE[0]



def get_dRdE_nu(Er_kev, AT, ZT, Sn, flux_table ,mscalar=None, g_vs=None, mvector=None, g_vz=None, mpseuvec=None, g_va=None, kappa=1):
    # in: Energy [keV], masses[GeV]
    # neuflux should be a tuple of (Enu, flux)
    # # flux factor order: [pp,hep, B, F, Be862, Be384, N13, O15, pep, DSNB, ATM, GSNB]
    # out: dRdE in 1/Myr/kg/keV
    # out shape: [len(coupling g), len(Er)], Ev integrated in the last axis

    kgperamu = 1.661e-27
    kevperamu = 931.5e3 #keV
    mT = AT*kevperamu
    Ev_kev = np.logspace(0, 6, 3000)

    if np.all(Ev_kev == flux_table[0]):
        flux = flux_table[1]
    else:
        flux = np.interp(Ev_kev, flux_table[0], flux_table[1])
    # load default neufluxes
    
    # neuflux shape = (3000,)

    dsigdE = calc_dsigdEr_neu_sm(Ev_kev, Er_kev, AT, ZT) \
    + calc_dsigdE_scalar(Ev_kev, Er_kev, AT, ZT, mscalar, g_vs, extra_term=1)\
    + calc_dsigdE_vector(Ev_kev, Er_kev, AT, ZT, mvector, g_vz, extra_term=1, kappa=kappa)\
    + calc_dsigdE_vector(Ev_kev, Er_kev, AT, ZT, mvector, g_vz, extra_term=2)\
    + calc_dsigdE_pseuvec(Ev_kev, Er_kev, AT, ZT, Sn, mpseuvec, g_va, extra_term=1)\
    + calc_dsigdE_pseuvec(Ev_kev, Er_kev, AT, ZT, Sn, mpseuvec, g_va, extra_term=2)

    # Ev,min to induce a nuclear recoil = sqrt(0.5*m*Er)
    # repeat_Er = np.repeat(Er_kev[...,None], len(Ev_kev),axis=1)
    # repeat_Ev = np.repeat(Ev_kev[None, ...],len(Er_kev), axis=0)

    repeat_Ev, repeat_Er = np.meshgrid(Ev_kev ,Er_kev, sparse=True)

    # _minEcutoff = (repeat_Ev>(0.5*mT*repeat_Er)**0.5)*1
    minEcutoff = (repeat_Ev>(0.5*(repeat_Er+np.sqrt(repeat_Er**2+2*mT*repeat_Er))))
    # minEcutoff shape = [Er.shpae, Ev.shape] = (Er, 1000)


    # minEcutoff = np.repeat(_minEcutoff[None,...], len(coupling), axis=0)

    # for now, only pass in on coupling, so collapse one dimension below
    integrand = dsigdE * flux[None,:] #* minEcutoff[None,:]
    # integrand = integrand*(integrand>0)
    integrand = integrand*minEcutoff
    # integrand = np.where(integrand<0, 0, integrand)
    dRdEi = np.trapz(integrand, Ev_kev,axis=-1)


    # unit conversion [1/Myr/kg/keV]
    dRdEi = dRdEi/(AT*kgperamu)*(1e6*365*24*60*60)

    return dRdEi

def get_dRdE_nu_SM(Er_kev, AT, ZT, flux_table):
    # in: Energy [keV], masses[GeV]
    # neuflux should be a tuple of (Enu, flux)
    # # flux factor order: [pp,hep, B, F, Be862, Be384, N13, O15, pep, DSNB, ATM, GSNB]
    # out: dRdE in 1/Myr/kg/keV
    # out shape: [len(coupling g), len(Er)], Ev integrated in the last axis

    kgperamu = 1.661e-27
    kevperamu = 931.5e3 #keV
    mT = AT*kevperamu
    Ev_kev = np.logspace(0, 6, 3000)

    if np.all(Ev_kev == flux_table[0]):
        flux = flux_table[1]
    else:
        flux = np.interp(Ev_kev, flux_table[0], flux_table[1])
    # load default neufluxes

    # 2D (Er, Ev)
    dsigdE = calc_dsigdEr_neu_sm(Ev_kev, Er_kev, AT, ZT)

    repeat_Ev, repeat_Er = np.meshgrid(Ev_kev ,Er_kev, sparse=True)

    # 2D, same size as dsigdE
    minEcutoff = (repeat_Ev>(0.5*(repeat_Er+np.sqrt(repeat_Er**2+2*mT*repeat_Er))))
    
    # for now, only pass in on coupling, so collapse one dimension below
    integrand = dsigdE * flux[None,:] #* minEcutoff[None,:]
    # integrand = integrand*(integrand>0)
    integrand = integrand*minEcutoff
    # integrand = np.where(integrand<0, 0, integrand)
    dRdEi = np.trapz(integrand, Ev_kev,axis=-1)


    # unit conversion [1/Myr/kg/keV]
    dRdEi = dRdEi/(AT*kgperamu)*(1e6*365*24*60*60)

    return dRdEi

def get_dRdE_nu_BSMcorrection(Er_kev, AT, ZT, Sn, flux_table , extra_term ,mscalar=None, g_vs=None, mvector=None, g_vz=None, mpseuvec=None, g_va=None, kappa=1):
    # this is the BSM correction to SM rate, these rate alone makes no sense
    # in: Energy [keV], masses[GeV]
    # neuflux should be a tuple of (Enu, flux)
    # # flux factor order: [pp,hep, B, F, Be862, Be384, N13, O15, pep, DSNB, ATM, GSNB]
    # out: dRdE in 1/Myr/kg/keV
    # out shape: [len(coupling g), len(Er)], Ev integrated in the last axis

    # to add:
    # add extra input: extra_term

    kgperamu = 1.661e-27
    kevperamu = 931.5e3 #keV
    mT = AT*kevperamu
    Ev_kev = np.logspace(0, 6, 3000)

    if np.all(Ev_kev == flux_table[0]):
        flux = flux_table[1]
    else:
        flux = np.interp(Ev_kev, flux_table[0], flux_table[1])
    # load default neufluxes
    
    
    
    # neuflux shape = (3000,)

    # to add:
    # for each calc_dsigE_BSM, add extra input: extra_term=extra_term

    dsigdE = calc_dsigdE_scalar(Ev_kev, Er_kev, AT, ZT, mscalar, g_vs, extra_term=extra_term)\
    + calc_dsigdE_vector(Ev_kev, Er_kev, AT, ZT, mvector, g_vz, extra_term=extra_term, kappa=kappa)\
    + calc_dsigdE_pseuvec(Ev_kev, Er_kev, AT, ZT, Sn, mpseuvec, g_va, extra_term=extra_term)

    # Ev,min to induce a nuclear recoil = sqrt(0.5*m*Er)
    # repeat_Er = np.repeat(Er_kev[...,None], len(Ev_kev),axis=1)
    # repeat_Ev = np.repeat(Ev_kev[None, ...],len(Er_kev), axis=0)

    repeat_Ev, repeat_Er = np.meshgrid(Ev_kev ,Er_kev, sparse=True)

    # _minEcutoff = (repeat_Ev>(0.5*mT*repeat_Er)**0.5)*1
    minEcutoff = (repeat_Ev>(0.5*(repeat_Er+np.sqrt(repeat_Er**2+2*mT*repeat_Er))))
    # minEcutoff shape = [Er.shpae, Ev.shape] = (Er, 1000)


    # minEcutoff = np.repeat(_minEcutoff[None,...], len(coupling), axis=0)

    # for now, only pass in on coupling, so collapse one dimension below
    integrand = dsigdE * flux[None,:] #* minEcutoff[None,:]
    # integrand = integrand*(integrand>0)
    integrand = integrand*minEcutoff
    # integrand = np.where(integrand<0, 0, integrand)
    dRdEi_corr = np.trapz(integrand, Ev_kev,axis=-1)


    # unit conversion [1/Myr/kg/keV]
    dRdEi_corr = dRdEi_corr/(AT*kgperamu)*(1e6*365*24*60*60)

    return dRdEi_corr