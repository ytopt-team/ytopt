LLM training loss benchmark for training using ytopt-libe

This benchmark is modeled after the llm/loss benchmark in ytopt and the HPO/libE benchmarks. It is designed to be used with libEnsemble for hyperparameter optimization of an LLM loss function. See its details in the paper (https://arxiv.org/pdf/2508.06617).

Files:
- dlp.py: The main model/training script with parameter markers (#P0, #P1, ...)
- plopper.py: Handles parameter substitution and execution
- exe.pl: Runs the generated Python script and extracts the loss
- problem.py: Defines the TuningProblem for ytopt
- ytopt_obj.py: libEnsemble-compatible objective wrapper
- run-laptop.sh: a batch file to configure and run the HPO on a laptop

To run with libEnsemble, use ytopt_obj.py as the sim function. It is recommand to use run-laptop.sh to test the HPO process on any laptop. Before the run, use the installation instructions from https://github.com/ytopt-team/ytopt to create a conda environment and install the required python packages.

Adapt dlp.py, plopper.py, and exe.pl as needed for your actual LLM/loss problem and data.
