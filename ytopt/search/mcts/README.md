#Contact: Jaehoon Koo <jkoo@anl.gov>

#MCS, ANL

#Nov 10, 2021

## Pubplicaiton
Customized Monte Carlo Tree Search for LLVM/Polly's Composable Loop Optimization Transformations ([paper](https://scwpub21:conf21%2f%2f@conferences.computer.org/scwpub/pdfs/PMBS2021-vSqRXl4nJSV5KT4jWO5cW/111800a082/111800a082.pdf)) 

# Install instructions for mcts 
* Post analysis requires installing ``python =>3.8``, ``pandas``, ``ipython``, ``dtreeviz``
* Install ``python=3.8`` on top of ytopt: 
```
conda install -c anaconda python=3.8
```
* Install dependencies:
```
pip install pandas ipython cairosvg 
```
* Install [dtreeviz](https://github.com/parrt/dtreeviz.git):
```
conda uninstall python-graphviz
conda uninstall graphviz
pip install dtreeviz             # install dtreeviz for sklearn
pip install dtreeviz[xgboost]    # install XGBoost related dependency
pip install dtreeviz[pyspark]    # install pyspark related dependency
pip install dtreeviz[lightgbm]   # install LightGBM related dependency
```

## Quick start
```
CLANG_PREFIX=/path/to/clang13 ./codes/kernels/autotune_mcts.sh  
```

## Tutorials
* [Autotuning PolyBench GEMM Kernel](https://github.com/ytopt-team/ytopt/blob/mcts/tutorial/docs/tutorials/mcts-gemm/tutorial-mcts-gemm.md)

## Reference
* This repository is built based on Loop transformation search space generator [https://github.com/Meinersbur/mctree.git](https://github.com/Meinersbur/mctree.git)
* Decision tree analysis [dtreeviz](https://github.com/parrt/dtreeviz.git)
