FUNCTION MICCF, ep_x, f_x, ep_y, f_y, lags, reverse=reverse

;  NAME:
;       MICCF
;
; PURPOSE:
;       This function computes a modified interpolated cross correlation of light curves
;
; CALLING SEQUENCE:
;       Result = MICCF(ep_x, f_x, ep_y, f_y, lags)
;
; INPUTS:
;    EP_X:    An n-element vector containing the epochs of the first light curve
;
;     F_X:    An n-element vector containing the fluxes of the first light curve
;
;    EP_Y:    An n-element vector containing the epochs of the second light curve
;
;     F_Y:    An n-element vector containing the fluxes of the second light curve
;
;    LAGS:    An n-element vector containing the range of lags being explored
;
; KEYWORD PARAMETERS:
;     REVERSE:        If set to a non-zero value, the responding light curve has been
;                     interpolated. If set to zero then the driving light curve has been
;                     interpolated.


  ; set up the necessary arrays


    n_lags = n_elements(lags)

    n_obs = n_elements(f_y)
    n_interp_jd = n_elements(f_x)

    obs_jd_arr = rebin(ep_y, [n_elements(ep_y), n_lags])
    tau = rebin(lags, [n_lags, n_interp_jd])

    min_obs_jd = transpose(rebin(min(obs_jd_arr,dim=1),[n_lags,n_obs]))

    obs_flux_arr = rebin(f_y, [n_elements(ep_y), n_lags])
    interp_jd_arr = rebin(ep_x, [n_interp_jd, n_lags]) 
    interp_jd_arr += transpose(tau) 
    flux_arr = rebin(f_x, [n_interp_jd, n_lags])
    min_interp_jd = transpose(rebin(min(interp_jd_arr,dim=1),[n_lags,n_obs]))
    flux_arr_hlp = interpolate(flux_arr,(obs_jd_arr-min_interp_jd)[*],(transpose(rebin(lags,[n_lags,n_obs])))[*])
    interp_flux_arr = dblarr(n_obs,n_lags)
    interp_flux_arr[*] = flux_arr_hlp

    ; find overlapping ranges
    int_jd_min = min(interp_jd_arr,dim=1)
    int_jd_max = max(interp_jd_arr,dim=1)
    does_match_obs = dblarr(n_obs,n_lags) + !values.d_nan
    idx = where(obs_jd_arr ge transpose(rebin(int_jd_min,[n_lags,n_obs])) $
                    and obs_jd_arr le transpose(rebin(int_jd_max,[n_lags,n_obs])))
    does_match_obs[idx] = 1.

    ; make sure non-overlapping data are nan
    obs_flux_arr *= does_match_obs
    interp_flux_arr *= does_match_obs

    ; Note: ONLY USE OVERLAPPING DATA FOR THIS!
    sig_interp = stddev(interp_flux_arr,dim=1,/nan)
    sig_obs = stddev(obs_flux_arr,dim=1,/nan)
    interp_mean = mean(interp_flux_arr,dim=1,/nan)
    obs_mean = mean(obs_flux_arr,dim=1,/nan)
    interp_mean = transpose(rebin(interp_mean,[n_lags,n_obs]))
    obs_mean = transpose(rebin(obs_mean,[n_lags,n_obs]))

    ; cross correlation
    ccf = total((obs_flux_arr-obs_mean)*(interp_flux_arr-interp_mean),1,/nan)

    ; normalisation
    ccf /= total(does_match_obs,1,/nan)
    ccf /= (sig_obs*sig_interp)


  return, ccf

end
