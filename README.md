# Spatial Spaxel-Binning Pipeline

This repository contains a fast, Python-based spaxel-binning algorithm optimized for Integral Field Spectroscopy (IFS) data cubes. The pipeline supports any dataset that can be represented as an MPDAF Cube with shape (nwave, ny, nx), a valid WCS, and a variance cube. The code is designed to aggregate spaxels based on spatial/radial coordinates to achieve a target Signal-to-Noise Ratio (S/N) across multiple emission lines simultaneously, minimizing the loss of radial information.

This implementation is based on the algorithm outlined by Carton et al. (2017), optimized significantly for computational efficiency by utilizing rapid window-based mean flux and variance calculations instead of full iterative profile fitting when running in fallback mode.

If you use this code in your research, please cite our paper:
> Chougule et al. (2026), Astronomy & Astrophysics (in prep).
> Zenodo DOI: https://doi.org/10.5281/zenodo.XXXXXXX

---

## Features

* **Multi-Line Thresholding:** Evaluates S/N limits across up to 4 distinct emission lines dynamically based on redshift (Balmer lines: H-alpha, H-beta, H-gamma; Forbidden lines: [O II] 3727, [O III] 5007, [S II] 6717, 6731).
* **Geometry-Aware Binning:** Uses de-projected galactocentric radius (r) and azimuthal angle (theta) coordinates derived from galaxy center, inclination, and position angle (PA).
* **High Performance:** Replaces slow, iterative non-linear spectral fitting inside the spatial loops with rapid, window-based mean flux and variance calculations, reducing execution time from hours to a few minutes.

---

## Critical Requirement: MPDAF Data Format

CRITICAL: This code strictly expects input data cubes to be in MPDAF format (MUSE Python Data Analysis Framework). Installing MPDAF is mandatory; the pipeline will not execute without it.

If your data arrays are currently stored as raw numpy arrays or standard FITS files, you must convert them into an MPDAF Cube object before feeding them into the pipeline. You can perform this conversion using the following snippet:

```python
import numpy as np
from astropy import units as u
from mpdaf.obj import Cube, WCS, WaveCoord

# Set up coordinates
# The supplied WCS must correctly describe the spatial coordinates of the cube,
# since the pipeline uses MPDAF WCS transformations (sky2pix) to locate spaxels.
wcs1 = WCS(crval=0, cdelt=0.2)
wave1 = WaveCoord(cdelt=1.25, crval=4000.0, cunit=u.angstrom) # Example wavelength

# Your raw numpy arrays (flux and variance)
# Shape must be (nwave, ny, nx)
MyData = np.ones((400, 30, 30)) # Example data
MyVar = np.ones((400, 30, 30)) # Example variance; 

# Create the MPDAF cube required by the code
# var must contain variance, not standard deviation or inverse variance.
mycube = Cube(data=MyData, var=MyVar, wcs=wcs1, wave=wave1)
```
---

## Execution Modes: Mode A vs. Mode B

The pipeline dynamically detects your environment at runtime and will execute in one of two modes depending on whether `pyplatefit` is installed. 

### Core Commonality
Regardless of the mode chosen, the actual spatial binning process is identical: **both modes utilize rapid, window-based mean flux and variance calculations to compute S/N inside the spatial loops**, completely bypassing slow, iterative profile fitting for the bins.

### The Key Difference: Parameter Initialization
The only difference between the two modes lies in how the target line wavelengths and window sizes are determined:

* **Mode A: Full Profile Initialization (Requires pyplatefit)**
  The pipeline automatically extracts the central spectrum and globally integrated spectrum of the galaxy and fits them using `pyplatefit`. It dynamically identifies the two Balmer lines and two forbidden lines with the highest S/N from the globally integrated spectrum, and extracts their precise observed wavelengths and FWHM values to define the spectral windows automatically using the central spectrum.
  
* **Mode B: Fast Window Fallback (No pyplatefit required)**
  The pipeline prints a warning and skips the automated central and globally integrated spectrum profile fitting. Instead, the user must manually provide the emission line list, their respective observed wavelengths, and FWHM. To extract the globally integrated and central spectrum from an MPDAF data cube, you can do it as follows:
```python
# Get globally integrated spectrum
global_spectrum = cube.sum(axis=(1,2))
# Get the x,y coordinate of the central spaxel
y_center,x_center = cube.wcs.sky2pix((dec_center,ra_center),nearest=True)[0]
# Extract the central spectrum
sp_center = cube[:,y_center,x_center]
```

---

## Required Input Parameters

The pipeline requires the following galaxy, structural, and execution metadata inputs:

### Core Configuration (Required for both modes)
* **ra_center, dec_center :** Coordinate center of the galaxy.
* **pa :** Position Angle of the galaxy on the sky.
* **inc :** Inclination of the galaxy disk.
* **rd :** Disc scale length.
* **redshift :** redshift of the galaxy.
* **sn_threshold :** Target minimum Signal-to-Noise Ratio for bin acceptance.
* **n_annuli :** Number of concentric radial rings used to initialize the grid, effectively controlling the initial radial resolution. The code sets up a polar mesh where the number of azimuthal sectors increases by 6 for each outward ring (6, 12, 18, etc.). 
  * *Note on Trade-off:* A lower value significantly speeds up execution time but results in a smaller number of thicker bins with more spaxels each, compromising the final radial resolution. A higher value provides a finer starting mesh.

### Line Window Configuration (Required for Mode B only)

* **emission_line_list :** List of target emission lines.
* **observed_wavelengths :** List of precise observed wavelength values corresponding to each line in your emission list.
* **observed_FWHM :** List of the measured line FWHM values (used to dynamically compute individual line window sizes).

CRITICAL FORMATTING RULE: The inputs for line configurations must be provided as Python lists. To ensure alignment, **all elements across these lists must be sorted in strictly ascending order based on their observed wavelengths.**  

**Example of properly aligned and sorted inputs:**
```python
emission_line_list = ['OII3727b', 'HGAMMA', 'HBETA', 'OIII5007']
observed_wavelengths = [5831.577351076927, 6789.997283646275, 7604.799174797293, 7833.207195372561]
observed_FWHM = [2.943685832664074, 4.893285634910402, 5.335917188248487, 3.1565230133550717]
```
---

## Repository Structure

* `execute_binning.ipynb` - A comprehensive Jupyter Notebook example demonstrating the full end-to-end pipeline logic.
* `Main_binning_code.py` - Contains the logic for the iterative "grow-and-lock" spatial loops and the accretion phase.
* `binning_functions.py` - Calculates galactocentric coordinates, bin positions and S/N. 
* `spectrum_functions.py` - Extract and fit central and global spectra to get required line info. Extract spectra based on window sizes.
* `bin_plots.py` - Visualization tools to map and evaluate the final bin configurations.
* `MHDFS-0003-cube.tar.gz` - Example data cube file provided to test the pipeline (see `execute_binning.ipynb`).

---

## Installation & Setup

### Prerequisites
Ensure you have a Python 3 environment active (Conda is highly recommended).

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/yourusername/repo-name.git](https://github.com/yourusername/repo-name.git)
   cd repo-name
    ```
2. Install the required dependencies (including MPDAF):
   ```
   pip install -r requirements.txt
   ```
## Usage & Quick Start Example
To get started immediately, we have provided an example notebook (execute_binning.ipynb) using the `MHDFS-0003-cube.fits` dataset. 

Launch Jupyter and open `execute_binning.ipynb`. The notebook is entirely self-contained and pre-configured with the precise central coordinates, target lines, observed wavelengths, and FWHM values for the `MHDFS-0003` galaxy.

Running this notebook directly showcases how the code dynamically operates with or without an active `pyplatefit` installation.

### Generated Outputs
Upon successful completion, the pipeline outputs:
* `binmap.png` - A spatial plot visualizing the generated bin network.
* `bin_spaxel_map.txt` - The final mapping of Bin IDs to spaxel coordinates. Each row contains the Bin ID followed by a bracketed list of [x,y] spaxel coordinates belonging to that bin (e.g., 10 [13,12],[13,13],[14,12]). Bins explicitly assigned as unbinned/rejected (-1) are automatically omitted.

### License & Attribution
This project is licensed under the MIT License - see the `LICENSE` file for details.
If you use this code, please cite the Zenodo archive DOI associated with this repository as well as the primary journal reference:
```Code snippet
  @ARTICLE{Chougule2026,
         author = {{Chougule}, Abhishek and et al.},
          title = "{Your Paper Title Here}",
        journal = {A&A},
           year = 2026,
           note = {Code available via Zenodo: [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)}
  }
```
