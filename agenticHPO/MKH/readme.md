# agenticHPO: Agentic AI for Hyperparameter Optimization in AI for Engineering

Hyperparameter Optimization (HPO) for mixed-kernel-heterojunction (MKH) transistors, is an agentic workflow implementation using Academy.

One HPO implementation using ytopt is from the link https://github.com/ytopt-team/ytopt/tree/main/ytopt-libe/svms/mkh/ytopt-libe from the [paper](https://arxiv.org/pdf/2406.18445). This agenticHPO is based on the HPO implementation.

Files:
- hpo.py (thread-level parallelism): a simple agentic workflow for hyperparameter optimization using Academy.
- mkh-hpo.py (process-level parallelism): a simple agentic workflow for hyperparameter optimization using Academy.
- models.txt: a list for the latest LLM models supported by Argonne Argo API.
- The file results.csv is the output results with the format "mixing_ratio, sigmoid_ratio, gaussian_ratio, coef0_param, C_param, avg_accuracy, elapsed_time" for HPO using agenticHPO.
