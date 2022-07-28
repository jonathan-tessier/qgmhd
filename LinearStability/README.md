## Directory

`LSAQGMHD-cheb.ipynb`: Jupyter notebook outlining the Linear Stability Problem and solving the system at fixed $F,M$ for various wavenumbers $k$. Requires GUI. Usage: `$ jupyter notebook LSAQGMHD-cheb.ipynb`

`LSAQGMHD-cheb_mpi.py`: Same as `LSAQGMHD-cheb.ipynb` but can be run in parallel. Use for high-resolution runs, or in the absence of a GUI. Usage: `$ mpirun -n X python3 LSAQGMHD-cheb_mpi.py`

`ContourPlots`: Contains generalized versions of the above to also loop over other parameters (such as $F,M$) and plots the largest growth rate on a contour plots. Usage is the same as `LSAQGMHD-cheb_mpi.py`.
- `lsa-F.py`: Largest growth rate contours as a function of $F,k$.
- `lsa-FM.py`: Largest growth rate contours as a function of $F,M$.
- `lsa-M.py`: Largest growth rate contours as a function of $M,k$.
- `lsa-beta.py`: Largest growth rate contours as a function of $\beta,k$, (if one wanted to study a non-constant Coriolis parameter).
