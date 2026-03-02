import numpy as np


def convert_ppm_to_gpg(ppm,molecular_mass_target, molecular_mass_source=238.05):
    return ppm*1e-6 *(molecular_mass_target/molecular_mass_source)
def convert_gpg_to_ppm(C, molecular_mass_target, molecular_mass_source=238.05):
    return C*1e6*(molecular_mass_source/molecular_mass_target)
    
def get_logL_nuissance(theta, theta_central, error_width):
    # return (C,)
    # try:
    #     extraDim = len(theta)
    # except TypeError:
    #     extraDim = 0
    
    # if extraDim == 0:
    #     theta = np.array([theta)
    res = -0.5*((theta-theta_central)/error_width)**2
    return res
def get_logL_poisson(N_data, N_th, sys_err=0):
    # N_th.shape = (sigma, C, bins), 
    # y = number of bins
    # N_data.shape = (bins,)
    return np.sum(-0.5*(N_data - N_th)**2/(N_data + (sys_err*N_data)**2), axis=-1)
    # return np.sum(N_data*np.log(N_th) - N_th, axis=-1) # scalar

def get_logL_total_one(logL_poisson, logL_nuissance_list):
    
    if type(logL_nuissance_list) == list:
        # logL_nuissance_list = (nuis, C,)
        logL_nuissance = np.sum(logL_nuissance_list, axis=0)
    else:
        logL_nuissance =logL_nuissance_list #scalar

    return logL_poisson + logL_nuissance 

def logL_nD(x, xerr, N_central, N_data):
    # N_central = array of central values of all the track lengths components
    # N_data = the total counts, summed over all components
    # x = [solar neutrino, DSNB, GSNB, ATM, neutron, t_age, mineral_mass]
    # N_central = [DM, solar_nu, DSNB, GSNB, ATM, neutron, Th234] # not considering: , Th234, single_alpha]
    # x = np.array(x[:,np.newaxis])
    N_DM = N_central[0]
    N_solar_nu = x[0]*N_central[1]
    N_DSNB = x[1]*N_central[2]
    N_GSNB = x[2]*N_central[3]
    N_ATM= x[3]*N_central[4]
    N_neutron = x[4]*(N_central[5] + N_central[6]) # neutron and Th234 share the same uncertainty
    # N_Th234 = x[4]*N_central[6]
    # N_alpha = x[4]*N_central[7]
    N_tb_tested = (N_DM + N_solar_nu + N_DSNB + N_GSNB + N_ATM + N_neutron)*x[-1]*x[-2]
    
    # print(f'all input shapes: {x[1:4].shape,N_central[2:5].shape, N_other_nu.shape, N_tb_tested.shape}')
    logL_nuissance_list = []
    for i, err in enumerate(xerr):
        logL_nuissance_list.append(get_logL_nuissance(x[i],1,err))
    logL_poisson = get_logL_poisson(N_data, N_tb_tested)

    log_L = get_logL_total_one(logL_poisson, logL_nuissance_list)

    return -log_L

