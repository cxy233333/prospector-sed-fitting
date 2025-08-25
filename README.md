# Galaxy SED Fitting with Prospector

This repository contains a pipeline for fitting galaxy spectral energy distributions (SEDs) using the **[Prospector](https://prospect.readthedocs.io/)** framework and **FSPS**.\
The fitting supports multiple star formation history (SFH) parameterizations (non-parametric Dirichlet, delay-τ, delay-τ+burst), and can run in parallel on multiple CPU cores.

---

## Installation

### 1. Install system dependencies

This project requires **gfortran** to compile FSPS (Flexible Stellar Population Synthesis).\
Please install it before proceeding.

- **Ubuntu/Debian**

  ```bash
  sudo apt-get update
  sudo apt-get install gfortran
  ```

- **macOS** (via Homebrew)

  ```bash
  brew install gcc
  ```

  (Note: Homebrew's `gcc` includes `gfortran`.)

- **CentOS/RedHat**

  ```bash
  sudo yum install gcc-gfortran
  ```

Verify installation:

```bash
gfortran --version
```

---

### 2. Clone the repository

```bash
git clone https://github.com/cxy233333/prospector-sed-fitting.git
cd prospector-sed-fitting
```

---

### 3. Create a Python environment

It is recommended to use **Python 3.10+** and a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate   # (Linux/macOS)
venv\Scripts\activate      # (Windows PowerShell)
```

---

### 4. Install Python dependencies

All required Python packages are listed in `requirements.txt`.\
To install:

```bash
pip install -r requirements.txt
```

This will install:

- `prospector` (from GitHub)
- `sedpy`
- `fsps`
- `h5py`
- `numpy`, `pandas`, `tqdm`
- and other supporting libraries.

---

## Input Data

The input galaxy SEDs are stored in an **HDF5 file** (example: `galaxy_seds_9-10.h5`).\
Each galaxy is stored in a group named `galaxy_{id}`, with the following datasets:

- `wavelength` : rest-frame wavelength array
- `spectrum`   : spectral flux (if available)
- `phot_maggies` : photometric fluxes (maggies)
- `phot_noise`   : photometric uncertainties

Additionally, each group has attributes:

- `current_mass` : current stellar mass of the galaxy

The HDF5 file also has a global attribute:

- `cosmic_age` : cosmic age of the snapshot (Gyr)

---

## Output

The code saves results into a **CSV file** under the output directory (e.g., `~/galaxy_fit_results_snr30/`).

Columns include:

- `galaxy_id` : galaxy identifier
- `formed_mass` : total stellar mass formed
- `logzsol` : stellar metallicity (log(Z/Z☉))
- `current_mass` : present stellar mass
- SFH-dependent parameters:
  - For **Dirichlet**: `sfr2`, `sfr_err` (median and uncertainty per age bin)
  - For **delaytau/delaytau+burst**: `tage`, `tau`, `fburst`, `tburst_cosmic`, etc.

---

## Running the Code

Modify the parameters at the bottom of `fit_galaxies.py`:

```python
if __name__ == "__main__":
    MODE = "delaytau+burst"  # "dirichlet", "delaytau", or "delaytau+burst"
    USE_SPECTRUM = False     # Whether to use spectral data in fitting
    SPEC_SNR = 30            # SNR used for spectral uncertainties
    
    main(mode=MODE, use_spectrum=USE_SPECTRUM, snr=SPEC_SNR)
```

Then run:

```bash
python SED_fitting.py
```

The script will:

1. Load the input HDF5 file (`SED_FILE` in the script).
2. Select galaxies within the specified mass range (`MASS_RANGE`).
3. Perform SED fitting in parallel (using `NUM_CPUS`).
4. Save results to a CSV file in the output directory.

---

## Notes

- The fitting can be **computationally intensive**. Please adjust `NUM_CPUS` based on your server.
- For large galaxy samples, consider running on an HPC cluster.
- If you only want photometric fitting, set `USE_SPECTRUM = False`.
- The `requirements.txt` installs `prospector` from a pinned GitHub commit for reproducibility.

---

## Authors

- Original code: Xingyu Chen
- Based on [Prospector](https://prospect.readthedocs.io/) and [FSPS](https://github.com/cconroy20/fsps).

