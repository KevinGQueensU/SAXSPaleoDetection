import numpy as np
from datetime import datetime
from scipy import special, integrate

def vearthwrtsun(when=datetime.now()):
    # calcuate velocity of the earth today
    # from 1-s2.0-S0927650596000473-main Appendix B
    # input when has to be in a format of np.datetime(year, month, day, hour, min, s) with optional precision

    #calcualte fractional day from 31/12/1999 noon
    d0= datetime(1999,12,31,12,0)
    d1 = when
    n = (d1-d0).days+((d1-d0).seconds)/(24*60*60)

    L = 280.460 + 0.9856474*n
    g = 357.528 + 0.9856003*n
    l = L + 1.915*np.sin(np.deg2rad(g)) + 0.020*np.sin(np.deg2rad(2*g))
    ue = 29.79*(1-0.016722*np.sin(np.deg2rad(l-13)))

    # calculate Earth's orbital velocity relative to the sun
    uex = ue*np.cos(np.deg2rad(-5.5303))*np.sin(np.deg2rad(l-266.141))
    uey = ue*np.cos(np.deg2rad(59.575))*np.sin(np.deg2rad(l-13.3485))
    uez = ue*np.cos(np.deg2rad(29.812))*np.sin(np.deg2rad(l-179.3212))

    return np.array((uex, uey, uez))

def earth2galaxy(v_det,theta,phi, when=datetime.now()):
    # calculate the velocity of DM in galactic frame
    v0 = 233 # km/s from PhysRevD.99.023012 Table 1

    # create array shape


    # earth's motion relative to milky way
    vlsr = np.array((0,v0,0)) # from McCabe_2014_J._Cosmol._Astropart._Phys._2014_027 intro # v of lsr wrt galaxy
    vpec = np.array((11.1, 12.2, 7.3)) # from McCabe_2014_J._Cosmol._Astropart._Phys._2014_027 intro # v of sun wrt lsr
    ue = np.array((vearthwrtsun(when)[0], vearthwrtsun(when)[1], vearthwrtsun(when)[2])) # v of earth wrt sun
    u = vlsr + vpec+ ue


    # calcualte dark matter velocity in galactic rest frame
    v_galx = v_det*np.sin(theta)*np.sin(phi)+u[0]
    v_galy = v_det*np.sin(theta)*np.cos(phi)+u[1]
    v_galz = v_det*np.cos(theta)+u[2]

    return np.array([v_galx, v_galy, v_galz])


def etar(vmin):
    #v0 = 235
    sig = 166
    v0 = 2**0.5*sig
    vesc = 550
    vlsr = 248
    Nresc = special.erf(vesc/v0) - (2/np.pi)**0.5 * (vesc/sig)*np.exp(-vesc**2/v0**2)
    A = 1/((2*np.pi*sig**2)**1.5*Nresc)

    # Nesc = erf(vesc/v0)-2./np.sqrt(np.pi)*(vesc/v0)*np.exp(-(vesc/v0)**2)
    # A = 1./( 2.*vlsr*Nesc )

    #coeff = 1/(2*vlsr*Nresc)
    int_res1 = np.pi*A*v0**2/vlsr*(np.pi**0.5/2*v0*(special.erf((vmin+vlsr)/v0) - special.erf((vmin-vlsr)/v0)) - np.exp(-vesc**2/v0**2)*2*vlsr)
    int_res2 = np.pi*A*v0**2/vlsr*(np.pi**0.5/2*v0*(special.erf(vesc/v0) - special.erf((vmin-vlsr)/v0)) - np.exp(-vesc**2/v0**2)*(vesc+vlsr-vmin))


    return int_res1*(vmin<np.abs(vlsr-vesc)) +int_res2*(vmin>np.abs(vlsr-vesc))*(vmin<(vlsr+vesc))

def f_s(vmin, vtheta, vphi):
    # calculate the velocity distribution of gaia sausage in galactic frame

    # v in galactic frame
    v0 = 233 # km/s from PhysRevD.99.023012 Table 1
    vesc = 528 # km/s from PhysRevD.99.023012 Table 1

    # coefficients and constants fs
    beta = 0.9 # from PhysRevD.99.023012 Sec III
    sigma_r2 = 3*v0**2/(2*(3-2*beta))
    sigma_theta2 = 3*v0**2*(1-beta)/(2*(3-2*beta))
    Nsesc = special.erf(vesc/(sigma_r2*2)**(0.5))- ((1-beta)/beta)**(0.5)*np.exp(-vesc**2/(2*sigma_theta2))\
            *special.erfi(vesc*beta**(0.5)/((1-beta)*2*sigma_r2)**0.5) # from PhysRevD.99.023012 Eq 7

    A = ((2*np.pi)**(3/2)*sigma_r2**0.5*sigma_theta2*Nsesc)**(-1)

    H = (np.sin(vtheta)**2*np.cos(vphi)**2*(1/(2*sigma_r2)-(1/(2*sigma_theta2))))+1/(2*sigma_theta2)
    vmin_gal3 = earth2galaxy(vmin,vtheta,vphi)
    vmin_gal = np.linalg.norm(vmin_gal3,axis=0)
    res = (A/(2*H))*(np.exp(-H*vmin_gal**2)-np.exp(-H*vesc**2))

    truncate = ((np.exp(-H*vmin_gal**2)-np.exp(-H*vesc**2))>=0)*1
    truncated_res = res*np.sin(vtheta)*truncate

    return np.abs(truncated_res)

def etas(vmin):


    #integrate over theta and phi
    theta = np.linspace(0,np.pi,40)
    phi = np.linspace(0,2*np.pi,80)
    v_mesh = np.array(np.meshgrid(vmin,theta,phi)) # v_mesh.shape = (3,40,len(vmin), 80)

    #integrate over phi


    phi_range = np.repeat(np.repeat(phi[np.newaxis,:],len(vmin),axis=0)[np.newaxis,:,:],len(theta),axis=0)
    int_phi = integrate.simps(f_s(v_mesh[0],v_mesh[1],v_mesh[2]),phi_range, axis=-1)

    #integrate over theta
    theta_range = np.repeat(theta[:,np.newaxis],len(vmin), axis=-1)
    int_theta = integrate.simps(int_phi,theta_range,axis=0)


    return int_theta

def etax(Er_kev, mx_gev, A, frac):
    # Er_kev is an array
    # return eta_tot in km/s
    Er = Er_kev*1e-3 # convert from keV to MeV
    mx = mx_gev*1e3 # convert from GeV to MeV
    mn = A*931.5 # in MeV
    un = mn*mx/(mn+mx) #in MeV
    vmin = 3e5*(mn*Er/(2*un**2))**0.5
    if frac != 0:
        eta_tot = (1-frac)*etar(vmin)+frac*etas(vmin)
    else:
        eta_tot = etar(vmin)

    return eta_tot

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

def get_drder_one(Er, A,mx,sigma, frac=0,spin='SI'):
    # Er in keV
    # mx in GeV
    #Er = np.logspace(-3,3,2201)
    eV2J = 1.602e-19
    rho_0 = 0.3 # GeV/cm^3 2106.06559
    mp = 0.93828 # GeV/c^2
    mN = A*931.5
    reduced_mass = mp*mx/(mp+mx) #GeV
    etax_cm = etax(Er, mx,A, frac)/(1e5) # convert from s/km to s/cm

    try:
        len(sigma)
    except TypeError:
        sigma = np.array([sigma])

    try:
        mx_length = len(mx)
    except TypeError:
        mx_length = 1
    if mx_length != 1:
        print('can only accept one DM mass')
        return
    
    result_bare = (A**2)*sigma[:,np.newaxis]*(Fn2SI(Er,A)*rho_0*etax_cm/(2*reduced_mass**2*mx))[np.newaxis,:]


    # units conversion
    result = result_bare * (3e10)**2 # in 1/(GeV^2 s)
    # Convert unit to rate/mg/keV
    rate = result*(3e8)**2/(1e6 * 1e9 * eV2J) # rate per sec per kg per keV
    rate_1my = rate*(1e6*365*24*60*60) # rate per 1 Myr per kg per keV
    #rate_1gy_10mg = rate_1my*1000*(10e-3/1e3)

    return Er, rate_1my

def get_drder(Er, A, atomic_fraction,mx,sigma,frac=0,spin='SI'):
    Num_types = 1
    try:
        Num_types = len(A)
        Num_types = len(atomic_fraction)
        print('entered try')
    except:
        A = [A]
        atomic_fraction = [atomic_fraction]
        print('entered except')
    else:
        Num_types = len(atomic_fraction)

    if len(A) != len(atomic_fraction):
        print('A and composition should have the same length')
        return
    drder_weighted = 0
    for i in range(Num_types):
        print('comp length', len(atomic_fraction))

        drder_weighted += get_drder_one(Er, A[i],mx,sigma, frac=0,spin='SI')*atomic_fraction[i]
    return drder_weighted