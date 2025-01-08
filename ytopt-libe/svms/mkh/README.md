# Using ytopt to autotuning the mixed-kernel-heterojunction based on the original Heterojunction mixed-kernel SVM simulation from the GitHub repo: https://github.com/JennyMa0517/mixed-kernel-heterojunction.


 Follow the following instructions to install ytopt from the link https://github.com/ytopt-team/ytopt-libensemble to create the conda env ytune and install the needed packages. The packages are self-contained.

# Directory

ytopt:  use ytopt to autotune three parameters: mixed ratio (p0), sigmoid_ratio (p1), and gaussian_ratio (p2) for the best accuracy
	parameter space is defined in problem.py

ytopt-libe: use ytopt to autotune five parameters: mixed ratio (p0), sigmoid_ratio (p1), and gaussian_ratio (p2), coef0 parameter (p3), and C parameter (P4) in parallel
	parameter space is defined in run_ytopt.py

To run either of them under the conda environment ytune, just use runs.sh
