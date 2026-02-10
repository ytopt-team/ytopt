Hyperparameter Optimization for the Benchmark to minimize the absolute value of the MSE ((Mean
Squared Error) difference between mixture model and Splines using ytopt-libe

This benchmark is about comparison between  Mixture Model vs Splines (Expanded Function Set).

Files:
- kan.py: The main comparison script (code mold) with parameter markers (#P0, #P1, ...)
- plopper.py: Handles parameter substitution and execution
- exe.pl: Runs the generated Python script and extracts the MSE difference
- run_ytopt.py: Defines the TuningProblem and parameter space for ytopt
- ytopt_obj.py: libEnsemble-compatible objective wrapper
- run-laptop.sh: a batch file to configure and run the HPO on a laptop

Folder: 
-plots: visulize the results

To run with libEnsemble, use ytopt_obj.py as the sim function. It is recommand to use run-laptop.sh to test the HPO process on any laptop. Before the run, use the installation instructions from https://github.com/ytopt-team/ytopt to create a conda environment and install the required python packages.

Adapt kan.py, plopper.py, and exe.pl as needed for your actual problem and data.
