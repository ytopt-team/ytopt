# Using ytopt to autotune the mixed-kernel SVM simulations for Smart Pixel Dataset (https://doi.org/10.5281/zenodo.7331128) based on the orignal SVM simulations from Tupendra Oli (HEP at ANL).


 Follow the following instructions to install ytopt from the link https://github.com/ytopt-team/ytopt-libensemble to create the conda env ytune and install the needed packages. The packages are self-contained.

# Directory

postive-charge: Download the dataset from Smart Pixel Dataset (https://doi.org/10.5281/zenodo.7331128)
	ytopt-libe/dlp.py: change the path = '/Users/xingfu/research/tmp/ytune/ytopt-libensemble/ytopt-libe-svms/hep/positive-charge/' to this path.

ytopt-libe: use ytopt to autotune five parameters: mixed ratio (p0), sigmoid_ratio (p1), and gaussian_ratio (p2), coef0 parameter (p3), and C parameter (P4) in parallel

	parameter space is defined in run_ytopt.py

To run either of them under the conda environment ytune, just use runs.sh
