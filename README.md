# derivative-tinygp 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edwarddramirez/derivative-tinygp/HEAD)

Summary notebooks implementing derivative gaussian processes with `tinygp`. Building from the `tinygp` tutorial on [Derivative Observations & Pytree Data][1], we implement a 2D derivative gaussian process, perform SVI with 1D derivative gaussian processes, and perform SVI using 2D derivative gaussian processes. 

**Note:** The first three notebooks are instructional. The latter two are inefficient and unrefined. But, they were added anyways for reference. Better versions of those two notebooks may be added in the future.

# Notebooks
1. `01_1d_deriv_gp.ipynb`: 1D Derivative Gaussian Process
2. `02_2d_deriv_gp.ipynb`: 2D Derivative Gaussian Process (Bonus: Polar GP Implementation)
3. `03_svi_1d_deriv_gp.ipynb`: SVI with 1D Derivative Gaussian Process Prior
4. `04_svi_2d_deriv_gp.ipynb`: SVI with 2D Derivative Gaussian Process Prior (Cartesian GP, Polar GP)
5. `05_svi_2d_sparse_deriv_gp.ipynb`: SVI with 2D Sparse Derivative Gaussian Process Prior (Cartesian GP)

# Installation
Run the `environment.yml` file by running the following command on the main repo directory:
```
conda env create
```
The installation works for `conda==4.12.0`. This will install all packages needed to run the code on a CPU with `jupyter`. 

If you want to run this code with a CUDA GPU, you will need to download the appropriate `jaxlib==0.4.13` version. For example, for my GPU running on `CUDA==12.3`, I would run:
```
pip install jaxlib==0.4.13+cuda12.cudnn89
```
The key to using this code directly would be to retain the `jax` and `jaxlib` versions. 

<!-- ### References  -->
[1]: <https://tinygp.readthedocs.io/en/latest/tutorials/derivative.html> "Derivative Observations & Pytree Data"