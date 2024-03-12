# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: mocks
#     language: python
#     name: mocks
# ---

# %% [markdown]
# ## Fluxes

# %%
import pkg_resources

from scipic.mocks.fluxes import Fluxes, IGM


# %%
# Auxiliary functions to read SED and extinction curve spectra, and filter transmission curves
def read_spectra(resource):
    with pkg_resources.resource_stream(
        'scipic.mocks.fluxes', resource
    ) as fd:
        return pd.read_csv(
            fd, names=['lambda', 'spectra'],
            comment = '#', index_col=0
        ).squeeze("columns")

def read_spectra_dir(resource_dir, basename_as_index=False):
    files = pkg_resources.resource_listdir(
        'scipic.mocks.fluxes', resource_dir
    )

    return pd.concat(
        {
            os.path.splitext(os.path.basename(f))[0]
            if basename_as_index
            else idx : read_spectra(os.path.join(resource_dir, f))
            for idx, f in enumerate(sorted(files))
        },
        names = ['idx']
    )


# %%
# Reference filter for flux normalization
reference_filter = read_spectra('filters/sdss_r01.csv')
# Galaxy SEDs
sed_spectra = read_spectra_dir('galaxy_seds')
# Galaxy extinction curves
ext_spectra = read_spectra_dir('galaxy_extincts')
# Filters
filter_spectra = read_spectra_dir('filters', basename_as_index=True)

# Binnings
common_wvls = np.arange(600., 24000., 1.)
igm_wvls = np.arange(100., 10000., 2.)
igm_z = np.arange(0, 4+0.02, 0.02)

# Kernels
fluxes = Fluxes(common_wvls, cosmology, reference_filter, sed_spectra, ext_spectra)
igm = IGM(igm_wvls, igm_z)

# %%
config_set = {
    'broad' : {
        # Apparent magnitudes in atmosphere
        'app_air_filter_list' : [
            '2mass_h', '2mass_j', '2mass_ks',
            'blanco_decam_g', 'blanco_decam_i', 'blanco_decam_r', 'blanco_decam_u', 'blanco_decam_z',
            'cfht_megacam_r', 'cfht_megacam_u',
            'cfht_megacam_uprime', 'cfht_megacam_ustar',
            'cfht_wircam_h', 'cfht_wircam_ks',
            'jedis_g',
            'kids_g', 'kids_i', 'kids_r', 'kids_u',
            'lsst_g', 'lsst_i', 'lsst_r', 'lsst_u', 'lsst_y', 'lsst_z',
            'pan_starrs_g', 'pan_starrs_i', 'pan_starrs_r', 'pan_starrs_z',
            'sdss_g', 'sdss_i', 'sdss_r', 'sdss_u', 'sdss_z',
            'sdss_r01',
            'subaru_b', 'subaru_g', 'subaru_i', 'subaru_r', 'subaru_v', 'subaru_y', 'subaru_z',
            'subaru_hsc_g', 'subaru_hsc_i', 'subaru_hsc_i2', 'subaru_hsc_r', 'subaru_hsc_r2', 'subaru_hsc_y', 'subaru_hsc_z',
            'subaru_nb1010', 'subaru_nb387', 'subaru_nb816', 'subaru_nb921',
            'subaru_ib427', 'subaru_ib445', 'subaru_ib464', 'subaru_ib484',
            'subaru_ib505', 'subaru_ib527', 'subaru_ib550', 'subaru_ib574',
            'subaru_ib598', 'subaru_ib624', 'subaru_ib651', 'subaru_ib679',
            'subaru_ib709', 'subaru_ib738', 'subaru_ib767', 'subaru_ib797',
            'subaru_ib827', 'subaru_ib856',
            'ukirt_wfcam_h', 'ukirt_wfcam_j', 'ukirt_wfcam_k',
            'vista_h', 'vista_j', 'vista_ks', 'vista_y',
        ],
        # Absolute magnitudes in atmosphere
        'abs_air_filter_list' : [
            'cfht_u', # IA required
            'd4000_n_blue', 'd4000_n_red',
            'jkc_b', 'jkc_i', 'jkc_r', 'jkc_u', 'jkc_v',
            'lsst_g', 'lsst_i', 'lsst_r', 'lsst_u', 'lsst_y', 'lsst_z',
            'sdss_g', 'sdss_i', 'sdss_r', 'sdss_u', 'sdss_z',
            'subaru_r', # IA required
            'vista_h', 'vista_j', 'vista_ks', 'vista_y',
        ],
        # Apparent magnitudes in vacuum
        'app_vacuum_filter_list' : [
            'euclid_nisp_h', 'euclid_nisp_j', 'euclid_nisp_y',
            'euclid_vis',
            'gaia_bp', 'gaia_g', 'gaia_rp',
            # 'irac_ch1', 'irac_ch2', # Disabled as they return NULL
            # 'wise_w1', 'wise_w2', # Cannot be simulated with current common lambda
        ],
        # Absolute magnitudes in vacuum
        'abs_vacuum_filter_list' : [
            'euclid_nisp_h', 'euclid_nisp_j', 'euclid_nisp_y',
            'euclid_vis',
            'galex_fuv', 'galex_nuv',
        ],
        'sed_e_bins' : np.sort(
            np.concatenate((
                sed_spectra.index.levels[0].values,
                ((sed_spectra.index.levels[0].values[:-1] + sed_spectra.index.levels[0].values[1:])/2.)
            ))
        ),
        'ext_e_bins' : ext_spectra.index.levels[0].values,
        'ebv_e_bins' : np.linspace(0., 0.5, 50),
        'z_obs_e_bins' : np.arange(0.0, 3.0+0.02, 0.02),    # extend to FS2 max redshift
        'abs_mag_e_bins' : np.linspace(-25, -12, 100),
        'el_z_obs_e_bins' : np.arange(0.0, 3.0+0.002, 0.002),
        'el_abs_mag_e_bins' : np.linspace(-25, -12, 100),
    },
    'narrow' : {
        'app_air_filter_list' : [
        ],
        'abs_air_filter_list' : [
        ],
        'app_vacuum_filter_list' : [
        ],
        'abs_vacuum_filter_list' : [
        ],
        'sed_e_bins' : np.sort(
            np.concatenate((
                sed_spectra.index.levels[0].values,
                ((sed_spectra.index.levels[0].values[:-1] + sed_spectra.index.levels[0].values[1:])/2.)
            ))
        ),
        'ext_e_bins' : ext_spectra.index.levels[0].values,
        'ebv_e_bins' : np.linspace(0., 0.5, 51),
        'z_obs_e_bins' : np.arange(0.0, 3.0+0.005, 0.005),
        'abs_mag_e_bins' : np.linspace(-25, -12, 100),
        'el_z_obs_e_bins' : np.arange(0.0, 3.0+0.001, 0.001),
        'el_abs_mag_e_bins' : np.linspace(-25, -12, 100),
    }
}

# %%
# Build the set of apparent and absolute fluxes that need to be computed, depending on the configuration above
app_filter_list = set()
abs_filter_list = set()

for s in config_set:
    cfg = config_set[s]

    assert not (set(cfg['app_air_filter_list']) & set(cfg['app_vacuum_filter_list']))
    assert not (set(cfg['abs_air_filter_list']) & set(cfg['abs_vacuum_filter_list']))

    cfg['sed_c_bins'] = ((cfg['sed_e_bins'][:-1] + cfg['sed_e_bins'][1:])/2.)
    cfg['ext_c_bins'] = cfg['ext_e_bins']
    cfg['ebv_c_bins'] = ((cfg['ebv_e_bins'][:-1] + cfg['ebv_e_bins'][1:])/2.)
    cfg['z_obs_c_bins'] = ((cfg['z_obs_e_bins'][:-1] + cfg['z_obs_e_bins'][1:])/2.)
    cfg['abs_mag_c_bins'] = ((cfg['abs_mag_e_bins'][:-1] + cfg['abs_mag_e_bins'][1:])/2.)
    cfg['el_z_obs_c_bins'] = ((cfg['el_z_obs_e_bins'][:-1] + cfg['el_z_obs_e_bins'][1:])/2.)
    cfg['el_abs_mag_c_bins'] = ((cfg['el_abs_mag_e_bins'][:-1] + cfg['el_abs_mag_e_bins'][1:])/2.)


    cfg['sed_a_bins'] = np.sort(np.concatenate((cfg['sed_e_bins'], cfg['sed_c_bins'])))
    cfg['ext_a_bins'] = cfg['ext_e_bins']
    cfg['ebv_a_bins'] = np.sort(np.concatenate((cfg['ebv_e_bins'], cfg['ebv_c_bins'])))
    cfg['z_obs_a_bins'] = np.sort(np.concatenate((cfg['z_obs_e_bins'], cfg['z_obs_c_bins'])))
    cfg['abs_mag_a_bins'] = np.sort(np.concatenate((cfg['abs_mag_e_bins'], cfg['abs_mag_c_bins'])))
    cfg['el_z_obs_a_bins'] = np.sort(np.concatenate((cfg['el_z_obs_e_bins'], cfg['el_z_obs_c_bins'])))
    cfg['el_abs_mag_a_bins'] = np.sort(np.concatenate((cfg['el_abs_mag_e_bins'], cfg['el_abs_mag_c_bins'])))

    app_filter_list |= set(cfg['app_air_filter_list'])
    app_filter_list |= set(cfg['app_vacuum_filter_list'])

    abs_filter_list |= set(cfg['abs_air_filter_list'])
    abs_filter_list |= set(cfg['abs_vacuum_filter_list'])

# %% [markdown]
# #### Hotfix for SPARK-12717

# %%
import threading

class BroadcastPickleRegistry(threading.local):
    """ Thread-local registry for broadcast variables that have been pickled
    """

    def __init__(self):
        self.__dict__.setdefault("_registry", set())

    def __iter__(self):
        for bcast in self._registry:
            yield bcast

    def add(self, bcast):
        self._registry.add(bcast)

    def clear(self):
        self._registry.clear()

if not local_mode.value:
    sc._pickled_broadcast_vars = BroadcastPickleRegistry()

# %% [markdown]
# ### Interpolators

# %%
import operator
import astropy.units as u

from functools import reduce
from multiprocessing.pool import ThreadPool
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar
from scipic.mocks.utils import cache

# %%
# Number of concurrent matrices that can be computed
pool = ThreadPool(8)


# %%
# Define a function, whose results are cached, that computes a grid of "true" fluxes corresponding to a filter
@cache(cache_path)
def compute_binned_fluxes(
    bins, fluxes, filter_spectrum, flux_interpolator=None
):
    return spark_utils.compute_binned_fluxes(
        bins,
        fluxes, filter_spectrum, flux_interpolator, sc,
        chunk_size=500, kind=Galaxy.Kind.CENTRAL.value
    )


# %%
# Build the interpolation matrix from the grid of grid
def compute_flux_interpolator(
    bins, fluxes, filter_spectrum
):
    df = compute_binned_fluxes(
        bins, fluxes, filter_spectrum
    )

    data = df['flux_integ'].values.reshape(
        tuple(len(x[1]) for x in bins),
    )

    return RegularGridInterpolator(
        [v for _,v in bins],
        data, bounds_error=False, fill_value=None
    )


# %%
# Queue the computation of all flux interpolation matrices for every configured filter
flux_interpolators_results = {}
for name, cfg in config_set.items():
    flux_interpolators_results[name] = pool.map_async(
        lambda args: (
            args['band'],
            compute_flux_interpolator(
                args['bins'], fluxes, filter_spectra.loc[args['band']]
            ),
        ),
        [
            {
                'band' : band,
                'bins' : [
                    ('sed',  cfg['sed_e_bins']),
                    ('ext',  cfg['ext_e_bins']),
                    ('ebv',  cfg['ebv_e_bins']),
                    ('z_obs',  cfg['z_obs_e_bins']),
                ]
            }
            for band in (
                set(cfg['app_air_filter_list']) | set(cfg['abs_air_filter_list']) |
                set(cfg['app_vacuum_filter_list']) | set(cfg['abs_vacuum_filter_list'])
            )
        ]
    )

# %%
# Check if all flux interpolation matrices are ready
reduce(operator.and_, [result.ready() for result in flux_interpolators_results.values()])

# %%
# Retrieve flux interpolation matrices  
flux_interpolators = dict(
    {band:interpolator for result in flux_interpolators_results.values() for band, interpolator in result.get()}
)

# %%
# Queue the computation of all IGM interpolation matrices for every configured filter
igm_interpolators_results = {}
for name, cfg in config_set.items():
    igm_interpolators_results[name] = pool.map_async(
        lambda args: (
            args['band'],
            igm(filter_spectra.loc[args['band']])
        ),
        [
            {
                'band' : band,
            }
            for band in (
                set(cfg['app_air_filter_list']) | set(cfg['app_vacuum_filter_list'])
            )
        ]
    )

# %%
# Retrieve IGM interpolation matrices  
igm_interpolators = dict(
    {band:interpolator for result in igm_interpolators_results.values() for band, interpolator in result.get()}
)

# %%
# Broadcast interpolation matrices to all nodes
if local_mode.value:
    flux_interpolators_bc = type('',(object,),{"value": flux_interpolators})()
    igm_interpolators_bc = type('',(object,),{"value": igm_interpolators})()
else:
    flux_interpolators_bc = spark.sparkContext.broadcast(flux_interpolators)
    igm_interpolators_bc = spark.sparkContext.broadcast(igm_interpolators)

# %%
# Use redshift at 10 parsec as absolute magnitude reference
fun = lambda z_true: abs(fluxes._cosmology.luminosity_distance(z_true).to(u.parsec).value - 10.)
z_true = minimize_scalar(fun, method='golden', tol=1e-15).x


# %%
# Flux pipeline: add a column for each apparent and absolute flux
def p_fluxes(app_filter_list, abs_filter_list):
    def _p_fluxes(df):
        # Add redshift at 10 parsecs
        df['restframe_redshift_gal'] = z_true

        # Compute apparent fluxes
        for band in sorted(app_filter_list):
            flux_interpolator = flux_interpolators_bc.value[band]
            igm_interpolator = igm_interpolators_bc.value[band]

            # Compute observed flux
            df[band] = (
                (
                    (
                        flux_interpolator(
                            df[['sed_cosmos_1', 'ext_curve_cosmos_1', 'ebv_cosmos_1', 'observed_redshift_gal']]
                        ) * np.power(
                            10.,
                            -0.4 * (
                                20.44
                                + df['abs_mag_r01'].values
                                + 5 * (
                                    np.log10(
                                        # Luminosity distance
                                        df['r_gal'] * (1 + df['true_redshift_gal']) * 1e6 # Mpc -> pc
                                    )
                                    - 1
                                )
                            )
                        )
                    ) * df['frac_cosmos_1'] +
                    (
                        flux_interpolator(
                            df[['sed_cosmos_2', 'ext_curve_cosmos_2', 'ebv_cosmos_2', 'observed_redshift_gal']]
                        ) * np.power(
                            10.,
                            -0.4 * (
                                20.44
                                + df['abs_mag_r01'].values
                                + 5 * (
                                    np.log10(
                                        # Luminosity distance
                                        df['r_gal'] * (1 + df['true_redshift_gal']) * 1e6 # Mpc -> pc
                                    )
                                    - 1
                                )
                            )
                        )
                    ) * (1 - df['frac_cosmos_1'])
                ) * igm_interpolator(df['true_redshift_gal'])
            ).astype('f4')

        # Compute absolute fluxes
        for band in sorted(abs_filter_list):
            flux_interpolator = flux_interpolators_bc.value[band]

            df[band+'_abs'] = (
                (
                    flux_interpolator(
                        df[['sed_cosmos_1', 'ext_curve_cosmos_1', 'ebv_cosmos_1', 'restframe_redshift_gal']]
                    ) * np.power(
                        10.,
                        -0.4 * (
                            20.44
                            + df['abs_mag_r01'].values
                            # Commented, as the normalization is 0 at 10 parsec
                            # + 5 * (
                            #     np.log10(
                            #         # Luminosity distance
                            #         df['r_gal'] * (1 + df['true_redshift_gal']) * 1e6 # Mpc -> pc
                            #     )
                            #     - 1
                            # )
                        )
                    )
                ) * df['frac_cosmos_1'] +
                (
                    flux_interpolator(
                        df[['sed_cosmos_2', 'ext_curve_cosmos_2', 'ebv_cosmos_2', 'restframe_redshift_gal']]
                    ) * np.power(
                        10.,
                        -0.4 * (
                            20.44
                            + df['abs_mag_r01'].values
                            # Commented, as the normalization is 0 at 10 parsec
                            # + 5 * (
                            #     np.log10(
                            #         # Luminosity distance
                            #         df['r_gal'] * (1 + df['true_redshift_gal']) * 1e6 # Mpc -> pc
                            #     )
                            #     - 1
                            # )
                        )
                    )
                ) * (1 - df['frac_cosmos_1'])
            ).astype('f4')

        # Remove extra column
        del df['restframe_redshift_gal']

        return df
    return _p_fluxes


# %%
# Add this step to the pipeline ony if requested
if steps['Fluxes']:
    cat_mock_22 = cat_mock_21.mapPartitionsWithIndex(
        spark_utils.pipeliner(
            p_fluxes(app_filter_list, abs_filter_list)
        ),
        True
    )
else:
    cat_mock_22 = cat_mock_21


# %%
# Show a sample of the results
head(cat_mock_22)
