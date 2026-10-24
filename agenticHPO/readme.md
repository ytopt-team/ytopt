# agenticHPO: Agentic AI for Hyperparameter Optimization in AI for Science and Engineering

agenticHPO is an agentic workflow framework for Hyperparameter Optimization (HPO) using Academy ((https://github.com/academy-agents/academy). It requires to install academy 
``` 
pip install academy-py 
```

Warning: This example requires access to an OpenAI compatible language model API and contains additional dependencies to langchain and langchain-openai:

```
pip install langchain>=1.0
pip install langchain-openai
```

# Folders and Files:
- MKH: This agenticHPO example about mixed-kernel-heterojunction (MKH) transistors is based on the HPO implementation using ytopt from the link https://github.com/ytopt-team/ytopt/tree/main/ytopt-libe/svms/mkh/ytopt-libe.
- KANs: This agenticHPO example about comparison between Mixture Model vs Splines (Expanded Function Set) is based on the HPO implementation using ytopt from the link https://github.com/ytopt-team/ytopt/tree/main/ytopt-libe/hpo4mse. 
- models.txt: a list for the latest LLM models supported by Argonne Argo API.
- It recommands to create a conda environment to run the script on a machine at Argonne network.
- It supports OpenAI GPT5.6 and Google Gemini3.5 flash via Argonne Argo API (https://argo.anl.gov). Check Argonne Argo API document for latest model support. 
