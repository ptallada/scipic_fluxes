import extinction
import numpy as np
import os
import pandas as pd
import pkg_resources

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy import units as u
from astropy.units import cds
from pyspark import Row
from scipy.integrate import trapz
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import minimize_scalar

from scipic.mocks import spark_utils, utils
from scipic.mocks.base import Galaxy

C_A = u.cds.c.to(u.Angstrom / u.s)
C_m = u.cds.c.to(u.m / u.s)

class Fluxes:
    def __init__(
        self, common_wvls, cosmology, ref_filter_spectra, sed_spectra, ext_spectra=None
    ):
        """
        @param wvls: All spectra data will be rebinned using this lambda (in Angstroms)
        @type wvls: np.array<float>
        @param cosmology: Cosmology used for computing the luminosity distance
        @type cosmology: astropy.cosmology
        @param reference_filter: Reference filter response for flux normalization
        @type reference_filter: pd.Series<lambda: float -> response: float>
        @param sed_spectra: Galaxy SEDs spectra
        @type sed_spectra: pd.Series<idx: float, lambda: float -> response: float>
        @param ext_spectra: Extinction curves spectra
        @type ext_spectra: pd.Series<idx: float, lambda: float -> response: float>
        
        """
        self._common_wvls = common_wvls
        self._cosmology = cosmology
        self._ref_filter_spectra = ref_filter_spectra
        self._sed_spectra = sed_spectra
        self._ext_spectra = ext_spectra
        
        self._initialized = False
    
    def _rebin_wvls(self, wvls, transmission):
        return np.interp(
            self._common_wvls, xp=wvls, fp=transmission,
            left=0., right=0.,
        )
    
    def _build_spectra_interp(self, spectra):
        indexes = []
        transmissions = []
        
        for idx, s in spectra.groupby(level=0):
            s.reset_index(level=0, drop=True, inplace=True)
            indexes.append(idx)
            transmissions.append(
                self._rebin_wvls(
                    s.index.values, s.values
                )
            )
        
        return RegularGridInterpolator([indexes], transmissions, bounds_error=False, fill_value=None)
    
    def initialize(self):
        self._rebin_wvls_vec = np.vectorize(
            self._rebin_wvls, signature='(i),(i)->(i)'
        )
        
        self._ref_filter_trans = self._rebin_wvls(
            self._ref_filter_spectra.index.values, self._ref_filter_spectra.values
        )
        self._ref_filter_int = trapz(
            self._ref_filter_trans / self._common_wvls, self._common_wvls
        ) * C_A
        
        self._sed_spectra_interp = self._build_spectra_interp(self._sed_spectra)
        if self._ext_spectra is not None:
            self._ext_spectra_interp = self._build_spectra_interp(self._ext_spectra)
        
        self._initialized = True
    
    def _ensure_initialized(f):
        def wrapped(self, *args, **kwargs):
            if not self._initialized:
                self.initialize()
            
            return f(self, *args, **kwargs)
        return wrapped
    
    @staticmethod
    def _gaussian(x, mu, sig, norm):
        """
        This function creates a gaussian profile used to model the emission lines

        :param x: wavelength positions
        :param mu: central wavelength of emission line
        :param sig: emission line dispersion
        :param norm: normalization factor
        :return:
        """
        flux = np.exp(-np.power((x[None, None, :] - mu[:, :, None]) / sig[:, :, None], 2.) / 2.)
        
        return norm[:, :, None] * flux
    
    @_ensure_initialized
    def filter_flux(self, filter_spectra, a_v=1.0, r_v=3.1, kind=Galaxy.Kind.CENTRAL.value):
        filter_transmission = self._rebin_wvls(filter_spectra.index.values, filter_spectra.values)
        kind = Galaxy.Kind(kind)

        def f(sed, z_obs, z_true, abs_mag_ref, el=None, mw_ebv=None, ext=None, ebv=None):
            """\
            Return the flux, possible including the contribution of the emission lines
            and the milky way extinction.
            """
            # Alias
            _wvls = self._common_wvls
            
            # Get interpolated SED spectra
            spectra = self._sed_spectra_interp(sed)
            
            # Apply extinction
            if ext is not None or ebv is not None:
                spectra *= np.power(
                    np.nan_to_num(
                        self._ext_spectra_interp(ext) / self._ext_spectra_interp([0]),
                        nan=1
                    ),
                    (ebv[np.newaxis] / 0.2).T
                )
            
            # Apply redshift
            obs_spectra = self._rebin_wvls_vec(
                _wvls * (1.0 + z_obs[:, None]),
                spectra / (1.0 + z_obs[:, None])
            )
            
            # Average flux densities
            if kind in [Galaxy.Kind.CENTRAL, Galaxy.Kind.SATELLITE]:
                fnu_restframe = trapz(
                    spectra * self._ref_filter_trans * _wvls, _wvls
                ) / self._ref_filter_int
                fnu_observed = trapz(
                    obs_spectra * self._ref_filter_trans * _wvls, _wvls
                ) / self._ref_filter_int
                kcorr = 2.5 * np.log10(
                    fnu_restframe / fnu_observed
                )
            else: # [Galaxy.Kind.QSO, Galaxy.Kind.HIGH_Z]
                fnu_observed = trapz(
                    spectra * self._ref_filter_trans * _wvls, _wvls
                ) / trapz(
                    self._ref_filter_trans * _wvls, _wvls
                ) * (1 + z_obs)
                kcorr = - 2.5 * np.log10(1 + z_obs)
            
            # Flux scaling
            lum_dist = self._cosmology.luminosity_distance(z_true).to(
                u.parsec
            ).value
            
            app_mag_reconstructed = (
                abs_mag_ref + (
                    5 * (
                        np.log10(
                            lum_dist
                        ) - 1
                    )
                ) + kcorr
            )
            fnu_correct = np.power(
                10., -0.4 * (app_mag_reconstructed + 48.6)
            )
            
            if kind == Galaxy.Kind.QSO:
                fnu_correct *= C_A / (1450**2)
            elif kind == Galaxy.Kind.HIGH_Z:
                fnu_correct *= C_A / (1500**2)
            
            # Normalization
            obs_spectra *= (
                fnu_correct / fnu_observed
            )[np.newaxis].T
            
            # Add emission lines
            if el is not None:
                el_lambda = el.columns.values
                el_sigma_kms = np.power(10, (
                    (-0.10 + 0.01 * z_obs) * (abs_mag_ref - 3.0) - 0.05 * z_obs )
                ).clip(50.) # Santi enforces a minimum sigma_kms of 50 km/s at restframe
                
                el_sigma = el_lambda * el_sigma_kms[:, None] * 1000 / C_m
                el_norm = el.values / (el_sigma * np.sqrt(2 * np.pi))
                
                # sample EL spectra WITH APPLIED REDSHIFT
                obs_spectra += self._gaussian(
                    _wvls,
                    el_lambda * (1 + z_obs[:, None]),
                    el_sigma * (1 + z_obs[:, None]),
                    el_norm / (1 + z_obs[:, None]),
                ).sum(axis=1)
            
            # Apply MW extinction
            if mw_ebv is not None:
                obs_spectra /= np.power(
                    10.,
                    0.4 * (mw_ebv * r_v / a_v)[:, None] * extinction.odonnell94(
                        _wvls, a_v, r_v, unit='aa', out=None
                    )
                )
            
            # Flux
            fnu = trapz(
                obs_spectra * filter_transmission * _wvls, _wvls
            ) / trapz(
                filter_transmission / _wvls, _wvls
            ) / C_A
            
            return fnu
        
        return f

