# derivative-tinygp
Summary notebooks implementing derivative gaussian processes with `tinygp`. Building from the `tinygp` tutorial on [Derivative Observations & Pytree Data][1], we implement a 2D derivative gaussian process, perform SVI with 1D derivative gaussian processes, and perform SVI using 2D derivative gaussian processes. 

**Note:** The first three notebooks are instructional. The latter two are inefficient and unrefined. But, they were added anyways for reference. Better versions of those two notebooks may be added in the future.

# Notebooks
1. `01_1d_deriv_gp.ipynb`: 1D Derivative Gaussian Process
2. `02_2d_deriv_gp.ipynb`: 2D Derivative Gaussian Process (Bonus: Polar GP Implementation)
3. `03_svi_1d_deriv_gp.ipynb`: SVI with 1D Derivative Gaussian Process Prior
4. `04_svi_2d_deriv_gp.ipynb`: SVI with 2D Derivative Gaussian Process Prior (Cartesian GP, Polar GP)
5. `05_svi_2d_sparse_deriv_gp.ipynb`: SVI with 2D Sparse Derivative Gaussian Process Prior (Cartesian GP)

<!-- ### References  -->
[1]: <https://tinygp.readthedocs.io/en/latest/tutorials/derivative.html> "Derivative Observations & Pytree Data"