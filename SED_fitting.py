import os
import sys
import h5py
import numpy as np
import pandas as pd
from prospect.models import SedModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import FastStepBasis
import fsps
from prospect.fitting import fit_model
from prospect.models.transforms import zfrac_to_masses
from prospect.models import priors, transforms
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import traceback
from sedpy.observate import load_filters
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# -------------------- CONFIGURATION --------------------
NUM_CPUS = 25  # Number of CPU cores to use for parallel processing
SED_FILE = os.path.join(os.path.expanduser("~"), "galaxy_seds_9-10.h5")  # Input SED data file
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "galaxy_fit_results_snr30")  # Output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist

# Mass range for galaxy selection
MASS_RANGE = [1e9, 10**9.0005]  # Only process galaxies with current mass between 1e9 and 1e10 solar masses

# -------------------- HELPER FUNCTIONS --------------------
def zfrac_constraint(z_fraction, **kwargs):
    """Constraint function for z_fraction parameter in non-parametric SFH models."""
    z_fraction = np.clip(z_fraction, 0.0, 1.0)
    if np.sum(z_fraction) > 1.0:
        return np.zeros_like(z_fraction)
    return np.maximum(z_fraction, 0.0)


def build_model(mode="dirichlet"):
    """
    Build a Prospector SED model with specified SFH parameterization.
    
    Parameters
    ----------
    mode : str
        SFH parameterization mode:
        - "dirichlet": Non-parametric SFH with Dirichlet prior
        - "delaytau": Parametric delayed-tau SFH
        - "delaytau+burst": Parametric delayed-tau SFH with additional burst
        
    Returns
    -------
    model : SedModel
        Configured Prospector SED model
    """
    if mode == "dirichlet":
        # Non-parametric SFH with Dirichlet prior
        model_params = TemplateLibrary["dirichlet_sfh"]

        # Define age bins for SFH reconstruction
        agebins = np.array([
            [0.0, 7.6990], [7.6990, 8.0], [8.0, 8.1761], [8.1761, 8.3010],
            [8.3010, 8.3979], [8.3979, 8.4771], [8.4771, 9.8482], [9.8482, 10.1399]
        ])
        n_bins = len(agebins)

        # Configure mass parameters
        model_params["mass"]["N"] = n_bins
        model_params["mass"]["init"] = np.full(n_bins, 1e8)

        # Configure age bins
        model_params["agebins"]["N"] = n_bins
        model_params["agebins"]["init"] = agebins

        # Configure z_fraction parameters with Dirichlet prior
        model_params["z_fraction"]["N"] = n_bins - 1
        model_params["z_fraction"]["init"] = np.full(n_bins - 1, 1.0 / (n_bins - 1))
        alpha = np.full(n_bins - 1, 5.0)
        model_params["z_fraction"]["prior"] = priors.Beta(
            alpha=alpha,
            beta=alpha,
            mini=np.zeros(n_bins - 1),
            maxi=np.ones(n_bins - 1),
        )
        model_params["z_fraction"]["depends_on"] = zfrac_constraint

        # Configure total mass parameter
        model_params["total_mass"] = {
            "N": 1,
            "isfree": True,
            "init": 1e9,
            "units": r"M$_\odot$",
            "prior": priors.LogUniform(mini=1e8, maxi=1e11),
        }

    elif mode in ["delaytau", "delaytau+burst"]:
        # Start from parametric SSP template
        model_params = TemplateLibrary["ssp"]

        # Configure delayed-tau SFH
        model_params["sfh"]["init"] = 4
        model_params["tau"] = {
            "N": 1,
            "isfree": True,
            "init": 1.0,
            "units": "Gyr",
            "prior": priors.LogUniform(mini=0.1, maxi=30.0),
        }

        # Configure stellar mass parameter
        model_params["mass"] = {
            "N": 1,
            "isfree": True,
            "init": 1e9,
            "units": r"M$_\odot$",
            "prior": priors.LogUniform(mini=1e8, maxi=1e11),
        }

        if mode == "delaytau+burst":
            # Add burst parameters for burst mode
            fage_burst = {
                "N": 1,
                "isfree": True,
                "init": 0.7,
                "units": "time at which burst happens (fraction of tage)",
                "prior": priors.TopHat(mini=0.5, maxi=1.0),
            }

            tburst = {
                "N": 1,
                "isfree": False,
                "init": 0.0,
                "units": "Gyr",
                "prior": None,
                "depends_on": transforms.tburst_from_fage,
            }

            fburst = {
                "N": 1,
                "isfree": True,
                "init": 0.1,
                "units": "fraction of total mass formed in the burst",
                "prior": priors.TopHat(mini=0.0, maxi=0.5),
            }

            model_params.update({
                "tburst": tburst,
                "fburst": fburst,
                "fage_burst": fage_burst,
            })

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Common parameters for all models
    model_params["lumdist"] = {"N": 1, "isfree": False, "init": 1e-5, "units": "Mpc"}
    model_params["zred"]["init"] = 0.0
    model_params["dust2"]["init"] = 0.0
    model_params["dust2"]["isfree"] = False

    return SedModel(model_params)


def build_sps():
    """Build the Stellar Population Synthesis (SPS) object for Prospector."""
    sp = fsps.StellarPopulation(
        zcontinuous=1,
        imf_type=2,
        sfh=0,  
    )
    return FastStepBasis(ssp=sp, sfh_smoothing=0.01)


def build_obs(wave, spec, phot_maggies, phot_noise, filters, use_spectrum=True, snr=30):
    """
    Build observation dictionary for Prospector fitting.
    
    Parameters
    ----------
    wave : array-like
        Wavelength array of the spectrum
    spec : array-like
        Spectral flux array
    phot_maggies : array-like
        Photometric fluxes in maggies
    phot_noise : array-like
        Photometric uncertainties
    filters : list
        List of filter objects
    use_spectrum : bool
        Whether to include spectral data in fitting
    snr : float
        Signal-to-noise ratio to scale the spectral uncertainties
        
    Returns
    -------
    obs : dict
        Observation dictionary compatible with Prospector
    """
    obs = {
        "filters": filters,
        "maggies": phot_maggies,
        "maggies_unc": phot_noise,
        "phot_mask": np.ones(len(filters), dtype=bool),
        "phot_wave": np.array([f.wave_effective for f in filters])
    }
    
    if use_spectrum:
        wave = np.asarray(wave)
        obs.update({
            "wavelength": wave,
            "spectrum": spec,
            'unc': (spec / snr) * (1 + np.random.normal(0, 0.05, size=spec.shape)),
            'mask': ((wave >= 3600) & (wave <= 9800)).astype(bool)
        })
    else:
        obs.update({
            "wavelength": None,
            "spectrum": None,
            "unc": None,
            "mask": None
        })
    
    return obs


def fit_galaxy(galaxy_id, hdf_file, mode="dirichlet", use_spectrum=True, snr=30):
    """
    Fit a single galaxy's SED data using Prospector.
    
    Parameters
    ----------
    galaxy_id : int
        ID of the galaxy to fit
    hdf_file : h5py.File
        HDF5 file containing galaxy data
    mode : str
        SFH parameterization mode
    use_spectrum : bool
        Whether to use spectral data in fitting
    snr : float
        Signal-to-noise ratio for spectral uncertainties
        
    Returns
    -------
    result : dict or None
        Dictionary containing fitting results, or None if fitting failed
    """
    try:
        # Load galaxy data from HDF5 file
        grp = hdf_file[f"galaxy_{galaxy_id}"]
        wave = grp["wavelength"][()]
        spec = grp["spectrum"][()]
        phot_maggies = grp["phot_maggies"][()]
        phot_noise = grp["phot_noise"][()]
        
        # Build observation dictionary
        obs = build_obs(wave, spec, phot_maggies, phot_noise, FILTERS,
                        use_spectrum=use_spectrum, snr=snr)
        
        # Build model and SPS objects
        model = build_model(mode=mode)
        sps = build_sps()
        
        # Configure nested sampling parameters
        nested_params = {
            "dynesty": True,
            "nested_bound": "multi",
            "nested_sample": "rwalk",
            "nested_dlogz": 0.05,  
            "nested_walks": 200,
            "output_posterior": True,
            "nested_print_progress": False
        }
        
        # Perform SED fitting
        output = fit_model(obs, model, sps, **nested_params)
        theta_labels = model.theta_labels()
        dynesty_samples = output["sampling"][0]["samples"]
        n_samples = dynesty_samples.shape[0]
        
        # Extract basic parameters
        total_mass_idx = theta_labels.index("mass")
        logzsol_idx = theta_labels.index("logzsol")
        formed_mass = np.median(dynesty_samples[:, total_mass_idx])
        logzsol = np.median(dynesty_samples[:, logzsol_idx])
        
        # Initialize result dictionary
        result = {
            "galaxy_id": galaxy_id,
            "formed_mass": formed_mass,
            "logzsol": logzsol
        }
        
        # Process results based on SFH mode
        if mode == "dirichlet":
            # Non-parametric SFH results
            zfrac_indices = [i for i, name in enumerate(theta_labels) if name.startswith("z_fraction_")]
            agebins_init = np.array(model.params["agebins"])
            
            # Calculate SFR in each age bin
            sfr_all = np.zeros((n_samples, len(agebins_init))) 
            for i in range(n_samples):
                total_mass_i = dynesty_samples[i, total_mass_idx]
                z_frac_i = dynesty_samples[i, zfrac_indices]
                masses_i = transforms.zfrac_to_masses(
                    total_mass=total_mass_i,
                    z_fraction=z_frac_i,
                    agebins=agebins_init
                )
                time_span = 10**agebins_init[:, 1] - 10**agebins_init[:, 0]
                sfr_i = masses_i / time_span
                sfr_all[i] = sfr_i
            
            # Calculate median SFR and uncertainties
            sfr_median = np.median(sfr_all, axis=0)
            sfr_16 = np.percentile(sfr_all, 16, axis=0)
            sfr_84 = np.percentile(sfr_all, 84, axis=0)
            sfr_err = (sfr_84 - sfr_16) / 2
            
            # Calculate current stellar mass
            theta_best = np.median(dynesty_samples, axis=0)
            _, _, pfrac = model.mean_model(theta_best, obs=obs, sps=sps)
            current_mass = formed_mass * pfrac
            
            # Update result dictionary
            result.update({
                "sfr2": sfr_median,
                "sfr_err": sfr_err,
                "current_mass": current_mass
            })
        
        elif mode in ["delaytau", "delaytau+burst"]:
            # Parametric SFH results
            param_names = ["mass", "logzsol", "tage", "tau"]
            if mode == "delaytau+burst":
                param_names += ["fburst", "fage_burst"]
            
            # Extract parameter values and uncertainties
            for pname in param_names:
                idx = theta_labels.index(pname)
                samples = dynesty_samples[:, idx]
                result[f"{pname}_median"] = np.median(samples)
                result[f"{pname}_err"] = (np.percentile(samples, 84) - np.percentile(samples, 16)) / 2
            
            # Calculate current stellar mass
            theta_best = np.median(dynesty_samples, axis=0)
            _, _, pfrac = model.mean_model(theta_best, obs=obs, sps=sps)
            result["current_mass"] = formed_mass * pfrac
            
            # Additional parameters for burst mode
            if mode == "delaytau+burst":
                tburst_val = model.params["tburst"]
                # Convert lookback time to cosmic time
                tburst_cosmic = COSMIC_AGE - tburst_val
                result["tburst_cosmic"] = tburst_cosmic
                
                # Calculate burst mass
                fburst_idx = theta_labels.index("fburst")
                fburst_samples = dynesty_samples[:, fburst_idx]
                burst_mass = formed_mass * np.median(fburst_samples)
                result["burst_mass"] = burst_mass
        
        return result
    
    except Exception as e:
        print(f"Error fitting galaxy {galaxy_id}: {str(e)}")
        traceback.print_exc()
        return None


def process_galaxy_wrapper(args):
    """
    Wrapper function for parallel processing of galaxy fitting.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (galaxy_id, hdf_path, mode, use_spectrum, snr)
        
    Returns
    -------
    result : dict or None
        Fitting results or None if processing failed
    """
    galaxy_id, hdf_path, mode, use_spectrum, snr = args
    try:
        with h5py.File(hdf_path, 'r') as hdf:
            if f"galaxy_{galaxy_id}" in hdf:
                return fit_galaxy(galaxy_id, hdf, mode=mode,
                                  use_spectrum=use_spectrum, snr=snr)
        return None
    except Exception as e:
        print(f"Error processing galaxy {galaxy_id}: {str(e)}")
        return None


def main(mode="dirichlet", use_spectrum=True, snr=30):
    """
    Main function to perform SED fitting for multiple galaxies.
    
    Parameters
    ----------
    mode : str
        SFH parameterization mode
    use_spectrum : bool
        Whether to use spectral data in fitting
    snr : float
        Signal-to-noise ratio for spectral uncertainties
    """
    start_time = time.time()
    
    global FILTERS, COSMIC_AGE
    
    # Load LSST filters
    FILTER_NAMES = ['lsst_baseline_{0}'.format(b) for b in ['u', 'g', 'r', 'i', 'z', 'y']]
    FILTERS = load_filters(FILTER_NAMES)
    
    # Get cosmic age from HDF5 file
    with h5py.File(SED_FILE, 'r') as hdf:
        COSMIC_AGE = hdf.attrs["cosmic_age"]
    
    # Get galaxy IDs and filter by mass range
    galaxy_ids = []
    with h5py.File(SED_FILE, 'r') as hdf:
        for key in hdf.keys():
            if key.startswith("galaxy_"):
                galaxy_id = int(key.split("_")[1])
                # Check if galaxy mass is within specified range
                current_mass = hdf[key].attrs.get("current_mass", 0)
                if MASS_RANGE[0] <= current_mass <= MASS_RANGE[1]:
                    galaxy_ids.append(galaxy_id)
    
    galaxy_ids = sorted(galaxy_ids)
    
    print(f"Found {len(galaxy_ids)} galaxies in mass range {MASS_RANGE} in {SED_FILE}")
    
    # Prepare tasks for parallel processing
    tasks = [(gid, SED_FILE, mode, use_spectrum, snr) for gid in galaxy_ids]
    all_results = []
    print(f"Starting fitting with {NUM_CPUS} CPUs...")
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_galaxy_wrapper, task): task[0] for task in tasks}
        
        # Process results with progress bar
        with tqdm(total=len(futures), desc="Fitting galaxies") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                except Exception as e:
                    galaxy_id = futures[future]
                    print(f"Error processing galaxy {galaxy_id}: {str(e)}")
                pbar.update(1)
    
    # Save all results to a single CSV file
    if all_results:
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save to CSV
        output_filename = f"all_galaxies_{mode}_fit_results.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        results_df.to_csv(output_path, index=False)
        print(f"Saved all results to {output_path}")
    else:
        print("No results to save.")
    
    end_time = time.time()
    print(f"Completed fitting in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Global settings
    MODE = "delaytau+burst"  # SFH parameterization mode: "dirichlet", "delaytau", or "delaytau+burst"
    USE_SPECTRUM = False     # Whether to use spectral data in fitting
    SPEC_SNR = 30            # Signal-to-noise ratio for spectral uncertainties
    
    # Run main function
    main(mode=MODE, use_spectrum=USE_SPECTRUM, snr=SPEC_SNR)