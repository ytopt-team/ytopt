# agenticHPO: Agentic AI for Hyperparameter Optimization in AI for Science 

Hyperparameter Optimization (HPO) for the Benchmark, which is about comparison between  Mixture Model vs Splines (Expanded Function Set), to minimize the absolute value of the MSE (Mean
Squared Error) difference between mixture model and Splines, is an agentic workflow using Academy.

One HPO implementation using ytopt is from the link https://github.com/ytopt-team/ytopt/tree/main/ytopt-libe/hpo4mse. This agenticHPO is based on the HPO implementation.

Files:
- hpo.py (thread-level parallelism): a simple agentic workflow for hyperparameter optimization with a random search  using Academy (https://github.com/academy-agents/academy). 
- kan-hpo.py (process-level parallelism): a simple agentic workflow for hyperparameter optimization with a random search  using Academy (https://github.com/academy-agents/academy). 
- models.txt: a list for the latest LLM models supported by Argonne Argo API.
- It supports OpenAI GPT5.6 and Google Gemini3.5 flash via Argonne Argo API (https://argo.anl.gov). Check Argonne Argo API document for latest model support. 
- The file *.csv is the output results with the format "m, n, smoothing, mse_diff, elapsed_time" for HPO using agenticHPO.
