# This package leverages and tailors ytopt-libe to generate power grid scenarios then do TSI analysis in parallel.

 This ytopt-libe-uqgrid requires ytopt and UQGrid from https://github.com/dmaldona/uqgrid. It is self-contained.

# On LLNL Dane: 

1. Loading the modules: 
```
  Currently Loaded Modules:
  1) jobutils/1.0       3) StdEnv           (S)   5) mvapich2/2.3.7         7) petsc/3.18.3
  2) texlive/20220321   4) gcc/13.3.1-magic       6) hdf5-parallel/1.14.0
```

2. Install miniconda

3. Create conda envirment ytune 

```
conda create --name ytune python=3.13
conda activate ytune

mkdir ytune
cd ytune
```

4. Install ytopt (https://github.com/ytopt-team/ytopt)

```
git clone https://github.com/ytopt-team/scikit-optimize.git
cd scikit-optimize
pip install -e .
cd ..

git clone -b version1 https://github.com/ytopt-team/autotune.git
cd autotune
pip install -e . 
cd ..

git clone -b main https://github.com/ytopt-team/ytopt.git
cd ytopt
pip install -e .
cd ..
```

5. Install uqgrid (https://github.com/dmaldona/uqgrid)
```
git clone https://github.com/dmaldona/uqgrid.git
cd uqgrid
pip install -e .

pip install numdifftools
```

6. go to the package ytopt-libe-uqgrid

```
cd ytopt/ytopt-libe/uqgrid
To run a job on a laptop, use run-laptop.sh to submit a job.
./run-laptop.sh

To run a batch job on LLNL Dane, use run-dane.sh to submit a job.
./run-dane.sh

```
