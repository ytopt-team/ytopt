# Using ytopt-libe for hyperparameter optimization for Julia codes

This package hpo4julia requires PhasorNetworks.jl to run. You have to install [PhasorNetworks.jl](https://github.com/wilkieolin/PhasorNetworks.jl.git), then copy the package under the folder PhasorNetworks.jl to run the package.

* Polaris at Argonne:
```
module use /eagle/EE-ECP/julia_depot/modulefiles/
module load julia/1.11
```

* Installation:

```
git clone https://github.com/wilkieolin/PhasorNetworks.jl.git

cd PhasorNetworks.jl/hpo4julia
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.test()'
```

* Copy the package to the folder PhasorNetworks.jl to run:

```
cp hpo4julia PhasorNetworks.jl
cd PhasorNetworks.jl/hpo4julia

to run the hpo on a laptop:
./runs-laptop.sh

to run the hpo on Polaris at Argonne:
./runs-polaris.sh
```

* Hyperparameters and their range for the Julia code:
```
Currently, the variables are:
learning rate  (Real, 1e-1 to 1e-5)
epochs (Integer, 1 to 50)
optimizer (String, [rmsprop, adam, sgd)

We use the latest [configspace](https://automl.github.io/ConfigSpace/latest/guide/) to define the parameter space by replacing the old configspace format. When the output file results.csv is generated, the folder plots provides the tools to generate the plots and sensitivity analysis.
```
