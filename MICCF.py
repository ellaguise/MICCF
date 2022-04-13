import scipy.interpolate
import numpy as np

def M_ICCF(x1, y1, x2, y2, lags):
    
    """
    This function computes a modified interpolated cross correlation of light curves, 
    where only one light curve is interpolated and the epochs matching that of the 
    second light curve are used to compute the cross correlation function for a 
    different range of lags

    PARAMETERS:
	x1:    An n-element vector containing the epochs of the first light curve 
		(i.e. the interpolated light curve)
	y1:    An n-element vector containing the fluxes of the first light curve 
		(i.e. the interpolated light curve)
	x2:    An n-element vector containing the epochs of the second light curve
	y2:    An n-element vector containing the fluxes of the second light curve
	LAGS:  An n-element vector containing the range of lags being explored

    """

    n_lags = len(lags)
    n_obs = len(x2)
    n_interp = len(x1)

    obs_jd_arr = np.full([n_lags, n_obs], x2)
    min_obs_jd = np.transpose(np.full([n_obs,n_lags],np.amin(obs_jd_arr, axis=1)))
    obs_flux_arr = np.full([n_lags, n_obs], y2)
    
    tau = np.full([n_interp, n_lags], lags)
    interp_jd_arr = np.full([n_lags, n_interp], x1)
    interp_jd_arr = np.add(interp_jd_arr, np.transpose(tau))
    min_interp_jd = np.transpose(np.full([n_obs,n_lags],np.amin(interp_jd_arr, axis=1)))
    
    flux_arr = np.full([n_lags, n_interp], y1)
  
    subbed_obs_jd = np.subtract(obs_jd_arr, min_interp_jd)
    subbed_obs_jd_list = subbed_obs_jd.flatten()
    
    lags_transp = np.transpose(np.full([n_obs,n_lags],lags))
    lags_transp_list = lags_transp.flatten()
    
    flux_arr_list = flux_arr[0]
    
    flux_arr_hlp = np.zeros_like(obs_flux_arr)
    
    for i in range(len(flux_arr)):
        interp = np.interp(obs_jd_arr[i], interp_jd_arr[i], flux_arr[i])
        flux_arr_hlp[i] = interp

    interp_flux_arr = flux_arr_hlp
  
    int_jd_min = np.min(interp_jd_arr, axis=1)
    int_jd_max = np.max(interp_jd_arr, axis=1)
    int_jd_min_arr = np.transpose(np.full([n_obs, n_lags], int_jd_min))
    int_jd_max_arr = np.transpose(np.full([n_obs, n_lags], int_jd_max))
    
    idx = np.where(obs_jd_arr >= int_jd_min_arr, 1, np.nan)
    idx2 = np.where(obs_jd_arr <= int_jd_max_arr, 1, np.nan)
    does_match_obs = idx*idx2
    
    obs_flux_arr *= does_match_obs
    interp_flux_arr *= does_match_obs
    
    sig_interp = np.nanstd(interp_flux_arr, axis=1)
    sig_obs = np.nanstd(obs_flux_arr, axis=1)
    interp_mean = np.nanmean(interp_flux_arr, axis=1)
    obs_mean = np.nanmean(obs_flux_arr, axis=1)
    
    interp_mean = np.transpose(np.full([n_obs, n_lags], interp_mean))
    obs_mean = np.transpose(np.full([n_obs, n_lags], obs_mean))
    
    ccf = np.nansum(np.subtract(obs_flux_arr, obs_mean)*np.subtract(interp_flux_arr,interp_mean), axis=1)
    ccf /= np.nansum(does_match_obs, axis=1)
    ccf /= (sig_obs*sig_interp)
    
    return ccf
