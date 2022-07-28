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

## Directory

`qgmhd_shenfun.py`: Driver script. Sets parameters and starts simulation. Outputs to directory `output-qgmhd` a `fields` and a `diagnostics` file, both in `.h5`.

`read_diagnostics.py`: After running driver script, run to compute diagnostics.

`read_fields.py`: After running driver script, run to create animation.

`library`: Contains source code for the simulation and function definitions.

`AdditionalDiagnostics`: Other post-processing codes to compare multiple runs, compute energy spectra and spectral budget.
