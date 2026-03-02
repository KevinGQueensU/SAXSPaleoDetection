import scipy as sp
import numpy as np
from PaleoBSM import compute_trackSpectra, olivineObj

def P_ellipsoid(q, Re,  Rp, alpha_n = 100):
    Re, q, Rp = np.meshgrid(Re, q, Rp, indexing='ij')
    alpha = np.linspace(0, np.pi / 2, alpha_n)[:, None, None, None]
    r = np.sqrt((Re*np.sin(alpha))**2+(Rp*np.cos(alpha))**2)
    F = 3*(np.sin(q*r) - q*r*np.cos(q*r))/(q*r)**3
    mask = np.isnan(F)
    F[mask] = 1
    P = np.trapz(F**2*np.sin(alpha), alpha[:, 0, 0, 0], axis=0)
    return P

def P_sphere(q, L, R):
    L, q, R = np.meshgrid(L, q, R, indexing='ij')
    F = (np.sin(q*L)-q * L * np.cos(q*L))/((q*L)**3)
    mask = np.isnan(F)
    F[mask] = 1/3
    return F**2 * 36*np.pi

def P_capped_cylinder(q, L, r, R, alpha_n = 100):
    h = -np.sqrt(R**2 - r**2)
    ts = np.linspace(-h/R, 1, alpha_n)[:, None, None, None, None]
    L, q, R = np.meshgrid(L, q, R, indexing='ij')
    alpha = np.linspace(0, np.pi/2, alpha_n)[None, :, None, None, None]

    def fct(t, a):
        y = q * R * np.sin(a) * np.sqrt(1-t**2)
        y = sp.special.jn(1, y)/y
        nan_mask = np.isnan(y)
        y[nan_mask] = 1/2
        return y

    f1 = lambda a: np.sinc(0.5*q*L*np.cos(a)) * 2 * sp.special.jn(1, q*r*np.sin(a))/(q*r*np.sin(a))
    f2 = lambda t, a: 4*np.pi * R**3 * np.cos(q*np.cos(a)*(R*t+h+0.5*L)) *(1-t**2)*fct(t, a)
    y1 = f1(alpha)
    nan_mask = np.isnan(y1)
    y1[nan_mask] = 1
    y1 = y1 * np.pi * r**2 * L
    y2 = f2(ts, alpha)

    A_q = y1 + np.trapz(y2, ts[:, 0, 0, 0, 0], axis = 0)
    y = A_q * np.sin(alpha)
    P_q = np.trapz(y, alpha[0, :, 0, 0, 0], axis = 1)

    return P_q[0]
def P_hollow_cylinder(q, L, R1, R2, x_n = 100):
    L, q, R1 = np.meshgrid(L, q, R1, indexing='ij')
    R2 = 0.9*R1
    x = np.linspace(0, 1, x_n)[:, None, None, None]

    fct = (1/(1-(R2/R1)**2))**2
    f1 = lambda x, R: (2*sp.special.jn(1, q*R*np.sqrt(1-x**2)))/(q*R*np.sqrt(1-x**2))
    f2 = lambda x: (sp.special.sinc(q*L*x/2))**2
    f = lambda x: fct * (f1(x, R1) - (R2/R1)**2 * f1(x, R2)) * f2(x)

    y_vals =  np.nan_to_num(f(x), copy=False)

    I_grid = np.trapz(y_vals, x[:, 0, 0, 0], axis = 0)
    return I_grid

# Computes the orientally averaged form factor for a cylinder with meshgrid inputs
# q, L, and R can be meshgrids
# alpha_n denotes the resolution of the alpha integration axis, eps is for limit convergence
def P_cylinder(q, L, R, alpha_n = 100, eps=1e-12, mesh = False, alpha_0 = 0, alpha_f = np.pi/2):
    if mesh:
        L, q, R = np.meshgrid(L, q, R, indexing = 'ij')
    # Create a new axis to integrate along
    alpha = np.linspace(alpha_0, alpha_f, alpha_n)[:, None, None, None]

    # Form factor equation as a function of oreintation angle alpha
    F = lambda a: (
            2
            * np.sin(0.5 * q * L * np.cos(a)) / (0.5 * q * L * np.cos(a))
            * sp.special.jn(1, q * R * np.sin(a)) / (q * R * np.sin(a))
    )

    # Oreientational average integrand
    func   = lambda a: F(a) ** 2 * np.sin(a)

    # Ignoring divisions that can blow up
    with np.errstate(divide='ignore', invalid='ignore'):
        y_vals = func(alpha)

    # Fix nans resulting from sin(qr)/qr
    nan_mask = np.isnan(y_vals)

    if np.any(nan_mask):
        # Checking for the sinc function of sin(q*L*cos(alpha))/q*L*cos(alpha)
        s_arg = 0.5 * q * L * np.cos(alpha)          # same broadcasting as original
        sinc  = np.ones_like(s_arg)                  # lim_{x->0} sin(x)/x = sinc(x) = 1
        nz    = np.abs(s_arg) > eps                  # check if numerator is smaller than eps
        sinc[nz] = np.sin(s_arg[nz]) / s_arg[nz]

        # Do the same thing for the Bessel function over x
        j_arg = q * R * np.sin(alpha)
        j1_over_x = np.full_like(j_arg, 0.5)         # lim_{x→0} J1(x)/x = 1/2
        nz2   = np.abs(j_arg) > eps
        j1_over_x[nz2] = sp.special.jn(1, j_arg[nz2]) / j_arg[nz2]

        # Creates an array with safe values for Nnans
        y_safe = (2.0 * sinc * j1_over_x) ** 2 * np.sin(alpha)

        # replace only the troublesome points
        y_vals[nan_mask] = y_safe[nan_mask]

    # Integrate along alpha, returns an intensity grid
    I_grid = np.trapz(y_vals, alpha[:, 0, 0, 0], axis = 0)

    return I_grid

# Returns the intensity for a cylindrical shape in a solution
# q is the momentrum transfer, L is the length, R is radius
# rho_sl_p and rho_sl_0 are the average densities of the solution and cylinder respectively
def cylinder(q, L, R, rho_sl_p, rho_sl_0, scale = 1, type = 'Hollow'):
    if type == 'Hollow':
        R2 = R*0.9
        P = P_hollow_cylinder(q, L, R, R2)
        V_p = np.pi * L * (R2**2 - R**2)
    elif type == 'Capped':
        P = P_capped_cylinder(q, L, R, R)

    else:
        P = P_cylinder(q, L, R) # Orientally averaged intensity
        V_p = np.pi * R**2 * L # Volume of cylinder
    del_rho = rho_sl_p-rho_sl_0 # Density difference
    return scale * (del_rho)**2 * (V_p) * P # intensity in [unit volume^-1]

# Returns an intensity grid for a cylindrical shape in a solution
# q_range and x_range are used to create a meshgrid, R is the cylindrical raidus
# rho_sl_p and rho_sl_0 are the average densities of the solution and cylinder respectively
def cylinder_mesh_qx(q_range, x_range, R, rho_sl_p, rho_sl_0, \
                     alpha_0 = 0, alpha_f = np.pi/2, type = 'Cylinder'):
    if type == 'Hollow':
        R2 = R*0.9
        P = P_hollow_cylinder(q_range, x_range, R, R2)
        V_p = np.pi * x_range * (R2**2 - R**2)
    elif type == 'Capped':
        P = P_capped_cylinder(q_range, x_range, R, R)
        V_p = np.pi*x_range*R**2 + 2*np.pi*(2/3*R**3)
    elif type == 'Sphere':
        P = P_sphere(q_range, x_range, R)
        V_p = 4*np.pi*R**3/3
    elif type == 'Ellipsoid':
        P = P_ellipsoid(q_range, x_range, R)
        V_p = (4/3)*np.pi*R*x_range**2
    else:
        P = P_cylinder(q_range, x_range, R, mesh = True, alpha_0 = alpha_0, alpha_f = alpha_f) # Feed in grids
        V_p = np.pi * R ** 2 * x_range # Volume of cylinder
    del_rho = rho_sl_p-rho_sl_0 # Density difference
    return (del_rho)**2 * (V_p)**2 * P # intensity grid in [unit volume^-1]

# Takes in a length range for the cylinder and averages over it, returns the intensity
def cylinder_tracks(dRdx, q_range, x_range, R, rho_sl_p, rho_sl_0, \
                    normalize = False, alpha_0 = 0, alpha_f = np.pi/2, type = 'Cylinder'):
    # Turning dRdx into a probability distribution
    if normalize:
        norm = np.trapz(dRdx, x_range, axis= -1)
        dRdx = dRdx / norm

    # Get the intensity grid, integrate it against dRdx
    I_grid = cylinder_mesh_qx(q_range, x_range, R, rho_sl_p, rho_sl_0, alpha_0 = alpha_0, alpha_f = alpha_f, type=type)[:, :, 0]
    if(dRdx.shape[0] == 1):
        dRdx = dRdx[0, :]
    dRdx = dRdx[:, None]
    result = np.trapz(dRdx*I_grid, x_range, axis=0)

    return result

# Returns an intensity distribution averaged over the recoil spectrum for dark matter
# m_x and sig_x are the dark matter mass and cross-section respectively for a WIMP
# x_range should be given in nanometers
def cylinder_WIMPs(mineral, m_x, sig_x, x_range, q_range, R, rho_sl_p, rho_sl_0, scale = 1):
    # Compute the track length distribution for dark matter
    dRdx_dm = 0
    for i in range(len(mineral.atomic_masses)):
        A = mineral.atomic_masses[i]
        data_path = mineral.derived_data_path[i]
        fraction = mineral.atomic_fractions[i]
        dRdx_dm += compute_trackSpectra.get_wimps_dRdx_one(m_x, sig_x, x_range, A, data_path) * fraction

    # Return the intensities
    return cylinder_tracks(dRdx_dm, q_range, x_range, R, rho_sl_p, rho_sl_0)

# Returns a binned intensity distribution as a function of q
# Beta is the fractional error associated with the intensity measurement
# Beta = 0.3 is quoted from https://pmc.ncbi.nlm.nih.gov/articles/PMC2931518/pdf/1471-2105-11-429.pdf
def cylinder_binned(binned_dRdx, x_bin_edges, q_range, q_resolution, R, rho_sl_p, rho_sl_0, \
                    bin_number = 51, log = True):
    # Creating the bin edges
    if log:
        bin_edges = np.logspace(np.log10(q_resolution / 2), np.log10(max(q_range)), bin_number + 1)

    else:
        bin_edges = np.linspace((q_resolution / 2), max(q_range), bin_number + 1)

    # Calculating intensity and error
    I_i = cylinder_tracks(binned_dRdx, q_range, x_bin_edges[:-1], R, rho_sl_p, rho_sl_0, normalize = False)
    interp_region = np.logspace(np.log10(q_resolution / 2), np.log10(max(q_range)), 700)
    I = np.interp(interp_region, q_range, I_i)
    q_resolution = q_resolution * interp_region
    # Binning using window function convolution
    binned_rate = []
    for i in range(bin_number):
        window = compute_trackSpectra.window_function(
            interp_region, bin_edges[i], bin_edges[i + 1], q_resolution)

        window /= np.trapz(window, interp_region)
        mean_I = np.trapz(I * window, interp_region)
        binned_rate.append(mean_I)
    return binned_rate, bin_edges

# Returns a binned intensity distribution using a binned WIMP recoil spectrum distribution
# resolution_x and resolution_q determine the size and number of bins
def cylinder_WIMPs_binned(mineral, m_x, sig_x, x_range, resolution_x, q_range, resolution_q, R, rho_sl_p, rho_sl_0,
                          t = 1e3, m = 0.1):
    rate_dm, bin_edges_x = compute_trackSpectra.get_wimps_Nbins(sig_x, x_range, m_x, mineral, \
                                                                resolution=resolution_x)
    rate_dm *= t * m
    rate_I, bin_edges_q = cylinder_binned(rate_dm, bin_edges_x, q_range, resolution_q, R, rho_sl_p, rho_sl_0)
    return rate_dm, bin_edges_x, rate_I, bin_edges_q,

def scale_dRdx_grid(dRdx_grid, rho_mineral, m):
    N_cyl = np.sum(dRdx_grid, axis = 2)
    V_tot = m/rho_mineral
    return N_cyl / V_tot

def scale_dRdx(dRdx, rho_mineral, m):
    N_cyl = np.sum(dRdx, axis = -1)
    V_tot = m/rho_mineral
    return N_cyl / V_tot

# Takes in a length range for the cylinder and averages over it, returns the intensity
def cylinder_tracks_mesh(dRdx_grid, q_range, x_range, R, rho_sl_p, rho_sl_0, normalize = False, type = 'Cylinder'):
    # Turning dRdx into a probability distribution
    if normalize:
        norm = np.trapz(dRdx_grid, x_range, axis= -1)[:, :, np.newaxis]
        dRdx_grid = dRdx_grid / norm

    # Get the intensity grid, integrate it against dRdx
    I_grid = cylinder_mesh_qx(q_range, x_range, R, rho_sl_p, rho_sl_0, type = type)[:, :, 0]
    dRdx_grid = dRdx_grid[..., np.newaxis]
    I_grid = I_grid[None, None, :, :]
    result = np.trapz(dRdx_grid * I_grid, x_range, axis=2)
    return result

# Returns a binned intensity distribution as a function of q
# Beta is the fractional error associated with the intensity measurement
# Beta = 0.3 is quoted from https://pmc.ncbi.nlm.nih.gov/articles/PMC2931518/pdf/1471-2105-11-429.pdf
def cylinder_binned_mesh(dRdx_grid, x_range, q_range, q_resolution, R, rho_sl_p, rho_sl_0, \
                         bin_number = 51, log = False, m = 1e-5, t = 1e3, type = 'Cylinder'):
    # Creating the bin edges
    if log:
        bin_edges = np.sort(np.logspace(np.log10(q_resolution / 2), np.log10(max(q_range)), bin_number + 1))
    else:
        bin_edges = np.linspace((q_resolution / 2), max(q_range), bin_number + 1)

    # Calculating intensity and error
    n = scale_dRdx_grid(dRdx_grid, 3210e-27, m)
    I_i = cylinder_tracks_mesh(dRdx_grid, q_range, x_range, R, rho_sl_p, rho_sl_0, normalize = False,
                               type = type) * n[:, :, None]
    interp_region = np.logspace(np.log10(q_resolution / 2), np.log10(max(q_range)), 400)
    I = sp.interpolate.interp1d(q_range, I_i, axis=2, kind='linear',
             bounds_error=False, fill_value='extrapolate')
    q_resolution = q_resolution * interp_region
    # Binning using window function convolution
    window = window_function_mesh(interp_region, bin_edges, q_resolution)
    temp = np.trapz(window, interp_region, axis = -1)[..., None]
    window /= temp
    integrand = I(interp_region)[:, :, None, :] * window [None, None, :, :]
    binned_rates = np.trapz(integrand, interp_region, axis = 3)
    return binned_rates, bin_edges

def window_function_mesh(xT, bin_edges, res):
    res = res[None, :]
    xT = xT[None, :]
    xa = bin_edges[:-1, None]
    xb = bin_edges[1:, None]
    left_smearing = sp.special.erf((xT-xa)/(np.sqrt(2)*res))
    right_smearing = sp.special.erf((xT-xb)/(np.sqrt(2)*res))
    return 0.5 * (left_smearing - right_smearing)

def cylinder_binned_quick(dRdx, x_range, q_range, q_resolution, R, rho_sl_p, rho_sl_0, \
                          bin_number = 51, log = False, m = 1e-5, alpha_0 = 0, alpha_f = np.pi/2, type = 'Cylinder'):
    # Creating the bin edges
    if log:
        bin_edges = np.sort(np.logspace(np.log10(q_resolution / 2), np.log10(max(q_range)), bin_number + 1))

    else:
        bin_edges = np.linspace((q_resolution / 2), max(q_range), bin_number + 1)

    # Calculating intensity and error
    n = scale_dRdx(dRdx, 3210e-27, m)
    I_i = cylinder_tracks(dRdx, q_range, x_range, R, rho_sl_p, rho_sl_0, normalize = False, alpha_0 = alpha_0, alpha_f = alpha_f,
                          type = type) * n
    interp_region = np.logspace(np.log10(q_resolution / 2), np.log10(max(q_range)), 700)
    I = np.interp(interp_region, q_range, I_i)
    q_resolution = q_resolution * interp_region

    # Binning using window function convolution
    window = window_function_mesh(interp_region, bin_edges, q_resolution)
    temp = np.trapz(window, interp_region, axis = -1)[..., None]
    # window /= temp
    integrand = I[None, :] * window
    binned_rates = np.trapz(integrand, interp_region, axis = 1)
    return binned_rates, bin_edges

# r_e is in nanometers, electron radius
def rho_xray(mineral, rho_bulk, M_weight, r_e = 2.82e-6, NA = 6.022e23):
    V_m = M_weight/(rho_bulk*NA)
    b = np.empty(len(mineral.atomic_number))
    for i, Z in enumerate(mineral.atomic_number):
        b[i] = Z * r_e
    return np.sum(b)/V_m

#%%
if True:
    mineral = olivineObj
    #%% Background Testing
    neutrino_data_path = 'PaleoBSM/neutrino_fluxes/'
    solar_sources = ['pp', 'hep', 'B', 'F', 'Be862', 'Be384', 'N13', 'O15', 'pep']

    dRdx_solar = 0
    xt_range = np.logspace(-2, 3, 500)

    for i, source in enumerate(solar_sources):
        solar_flux_i = np.load(neutrino_data_path + f'extrapolate_solar_{source}_fluxes.npy')

        dRdx_solar += compute_trackSpectra.get_SM_Neutrino_dRdx_mol(xt_range, mineral, solar_flux_i)

    atm_flux = np.load(neutrino_data_path + f'extrapolate_atm_fluxes.npy')
    dRdx_atm = compute_trackSpectra.get_SM_Neutrino_dRdx_mol(xt_range, mineral, atm_flux)
    DSNB_flux = np.load(neutrino_data_path + f'extrapolate_DSNB_fluxes.npy')
    dRdx_DSNB = compute_trackSpectra.get_SM_Neutrino_dRdx_mol(xt_range, mineral, DSNB_flux)
    GSNB_flux = np.load(neutrino_data_path + f'extrapolate_GSNB_fluxes.npy')
    dRdx_GSNB = compute_trackSpectra.get_SM_Neutrino_dRdx_mol(xt_range, mineral, GSNB_flux)
    dRdx_neutron = compute_trackSpectra.get_neutron_dRdx_mol(xt_range, mineral, 0.5e-9)
    dRdx_thorium = np.load('PaleoBSM/SRIM_derived_data/Backgrounds/Th/Tables/dist_at_72keV_01ppb.npy')

    # %%
    n_x = 500
    n_q = 200
    rho_sl_p = 4000 * 1e-27 # kg/nm^3
    rho_sl_0 = rho_sl_p * 0.90
    R = 2e-1 # nm
    xt_range = np.logspace(-2, 3, n_x) # nm
    q_range = np.logspace(-2, 1, n_q)  # inverse nm
    m_xs = np.array([1, 5, 100]) # GeV
    sig_xs = np.array([1e-42, 1e-45, 1e-47]) # /cm^-3

    # %%
    rows = [
        cylinder_WIMPs(mineral, m_x, sig_x,
                       xt_range, q_range, R, rho_sl_p, rho_sl_0)
        for m_x, sig_x in zip(m_xs, sig_xs)
    ]

    I_dm = np.vstack(rows)
    #%%
    I_solar = cylinder_tracks(dRdx_solar[np.newaxis, :], q_range, xt_range, R, rho_sl_p, rho_sl_0)
    I_atm = cylinder_tracks(dRdx_atm[np.newaxis, :], q_range, xt_range, R, rho_sl_p, rho_sl_0)
    I_DSNB = cylinder_tracks(dRdx_DSNB[np.newaxis, :], q_range, xt_range, R, rho_sl_p, rho_sl_0)
    I_GSNB = cylinder_tracks(dRdx_GSNB[np.newaxis, :], q_range, xt_range, R, rho_sl_p, rho_sl_0)
    I_neutron = cylinder_tracks(dRdx_neutron, q_range, xt_range, R, rho_sl_p, rho_sl_0)
    #%%
    xt_range = np.logspace(-2, 3, 500)
    I_thorium = cylinder_tracks(dRdx_thorium[np.newaxis, :], q_range, xt_range, R, rho_sl_p, rho_sl_0)
    I_background = I_solar + I_atm + I_GSNB + I_neutron + I_DSNB + I_thorium
    I_normal = cylinder(q_range, 1, R, rho_sl_p, rho_sl_0)[0, 0, :]
    # %%
    fct.plotSet(r"Intensity Spectrum Olivine: WIMPs",
                r"q $[nm^{-1}]$", r"I(q) $[nm^{-3}]$",
                1e-1, 1e-2,
                0, 10)
    plt.plot(q_range, I_dm[0],
             label="$m_x$ = " + "{:d}GeV".format(int(m_xs[0])) + r", $\sigma_{SI}$ = " + "{:.0e}".format(
                 sig_xs[0]) + r"$cm^2$")
    plt.plot(q_range, I_dm[1],
             label="$m_x$ = " + "{:d}GeV".format(int(m_xs[1])) + r", $\sigma_{SI}$ = " + "{:.0e}".format(
                 sig_xs[1]) + r"$cm^2$")
    plt.plot(q_range, I_dm[2],
             label="$m_x$ = " + "{:d}GeV".format(int(m_xs[2])) + r", $\sigma_{SI}$ = " + "{:.0e}".format(
                 sig_xs[2]) + r"$cm^2$")
    plt.plot(q_range, I_background , label="Background")
    plt.plot(q_range, I_normal, label = "Normal Radius")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.ylim(1e-57, 1e-43)
    plt.xlim(1e-2, 1e1)
    plt.show()
    # %%
    fct.plotSet(r"Intensity Spectrum Olivine: Background",
                r"q $[nm^{-1}]$", r"I(q) $[nm^{-3}]$",
                1e-1, 1e-2,
                0, 10)
    plt.plot(q_range, I_solar, label=r"Solar $\nu$")
    plt.plot(q_range, I_atm, label=r"ATM")
    plt.plot(q_range, I_DSNB, label=r"DSNB")
    plt.plot(q_range, I_GSNB, label=r"GSNB")
    plt.plot(q_range, I_neutron, label=r"Neutron")
    plt.plot(q_range, I_thorium, label=r"Thorium")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.ylim(1e-57, 1e-43)
    plt.xlim(1e-2, 1e1)
    plt.show()
    # %%
    m_x = 5
    sig_x = 1e-45
    resolution_x = 10
    resolution_q = 0.001
    n_x = 1000
    n_q = 200
    rho_sl_p = 4000 * 1e-27 # kg/nm^3
    rho_sl_0 = rho_sl_p * 0.90
    R = 1e-1 # nm
    xt_range = np.logspace(np.log10(resolution_x / 2), 3, n_x)
    q_range = np.logspace(np.log10(resolution_q / 2), 1, n_q)
    _, _, rate, edges = cylinder_WIMPs_binned(mineral, m_x, sig_x, xt_range, resolution_x, q_range, resolution_q, R, rho_sl_p, rho_sl_0)
    plt.stairs(np.asarray(rate) * 1000, edges)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    print(rate[:5])
    print(rate[-5:])












