This benchmark is about comparison between  Mixture Model vs Splines (Expanded Function Set).

Hyperparameter Optimization (HPO) for the Benchmark to minimize the absolute value of the MSE (Mean
Squared Error) difference between mixture model and Splines, is an agentic workflow using Academy.

One HPO implementation using ytopt is from the link https://github.com/ytopt-team/ytopt/tree/main/ytopt-libe/hpo4mse. This agenticHPO is based on the HPO implementation.

Files:
- kan-hpo.py: a simple agentic workflow for hyperparameter optimization with a random search  using Academy (https://github.com/academy-agents/academy). 
- models.txt: a list for the latest LLM models supported by Argonne Argo API.
- It requires to install academy ''' pip install academy-py '''. It recommands to create a conda environment to run the script on a machine at Argonne network.
- It supports OpenAI GPT5.6, Google Gemini3.5 flash, and Anthropic Claude Opus5.0 via Argonne Argo API (https://argo.anl.gov). Check Argonne Argo API document for latest model support. 
