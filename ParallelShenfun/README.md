## Usage

- Ensure that `Shenfun` and required packages are installed.
- Edit simulation parameters in driver script: `qgmhd_shenfun.py`
- Ensure to be in `Shenfun` virtual environment by running:
  - for conda installation: `$ conda activate shenfun`
  - for Compute Canada fix (on submit script): `source ~/<path-to>/shenfun/bin/activate`
- To run simulation: 
  - in serial: `$ python3 qgmhd_shenfun.py`. 
  - in parallel on Compute Canada (on submit script): `$ srun python3 qgmhd_shenfun.py`
  - in parallel on other architectures: `$ mpiexec -np X python3 qgmhd_shenfun.py`
- To visualize results:
  - run `read_diagnostics.py` to plot timeseries of the diagnostics (scalars)
  - run `read_fields.py` to make animations of the fields
  - refer to `AdditionalDiagnostics` for additional outputs

## Directory

`qgmhd_shenfun.py`: Driver script. Sets parameters and starts simulation. Outputs to directory `output-qgmhd` a `fields` and a `diagnostics` file, both in `.h5`.

`read_diagnostics.py`: After running driver script, run to plot diagnostics.

`read_fields.py`: After running driver script, run to create animation of flow.

`library`: Contains source code for the simulation and function definitions.

- `data_output.py`: Handles how to output data from the model, also some formatting functions.
- `grid.py`: Defines the grid, wavenumbers for the pseudo-spectral method, filter and random-field-generators.
- `initialize.py`: Initilializes model, builds function-spaces, scatters initial conditons to multiple cores.
- `operators.py`: Defines an inner and cross product, along with a Bickley Jet background field.
- `qg_physics.py`: Physical parameters, flux computation (governing equations)
- `time_stepping.py`: Applies AB3 scheme, using flux to evolve equations.

`AdditionalDiagnostics`: Other post-processing codes to compare multiple runs, compute energy spectra and spectral budget.

- `generate_diagnostics.py`: To debug. Use when wanting to recompute a particular diagnostic without re-running simulation.
- `snapshot_fields_F.py`: Creates a figure of field-snapshots for multiple simulations at various times $t$ and values of $F$ (fixed $M$).
- `snapshot_fields_M.py`: Creates a figure of field-snapshots for multiple simulations at various times $t$ and values of $M$ (fixed $F$).
- `spectral_flux.py`: Computes the spectral energy flux for a single simulation. Temporal averaging available.
- `spectral_slope.py`: Computes the spectral energy slopes for a single simulation. Temporal averaging available.
- `spectral_transfer.py`: Computes the spectral energy transfers for a single simulation. Temporal averaging available.
- `summary_dianostics.py`: Creates a figure of diagnostics as a function of time for multiple simulations (different $M$, fixed $F$).
- `summary_slope.py`: Computes the spectral energy slopes for multiple simulations (different $M$, fixed $F$). Temporal averaging available.
- `summary_spectral.py`: Computes the spectral energy fluxes and transfers for multiple simulations (different $M$, fixed $F$). Temporal averaging available.
- `summary_spectral_decomp.py`: Same as `summary_spectral.py` but plots the total and component fluxes/transfers on the same panel. Temporal averaging available.
