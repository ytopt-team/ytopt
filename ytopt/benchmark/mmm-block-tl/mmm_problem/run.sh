#! /usr/bin/env bash
# source activate ytune
export PYTHONWARNINGS="ignore"
cd /Users/jkoo/research/github/ytopt_tutorial/ytopt/ytopt/benchmark/mmm-block-tl/mmm_problem
####
python -m ytopt.search.ambs --evaluator ray --problem problem_s.Problem --max-evals=30 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge #--set-SEED 1234 
mv results.csv results_rf_100.csv
mv ytopt.log ytopt_rf_100.log
###
python -m ytopt.search.ambs --evaluator ray --problem problem_m.Problem --max-evals=30 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge #--set-SEED 3579 
mv results.csv results_rf_200.csv
mv ytopt.log ytopt_rf_200.log
###
python -m ytopt.search.ambs --evaluator ray --problem problem_l.Problem --max-evals=30 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge 
mv results.csv results_rf_300.csv
mv ytopt.log ytopt_rf_300.log
###
python -m ytopt.search.ambs --evaluator ray --problem problem_target.Problem --max-evals=10 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge #--set-SEED 2468 
mv results.csv results_rf_500.csv
mv ytopt.log ytopt_rf_500.log
#### 
python Run_online_TL.py --max_evals 10 --n_refit 10 --target 500 --top 0.5
mv results_sdv.csv results_sdv_500.csv
python Run_online_TL.py --max_evals 10 --n_refit 3 --target 500 --top 0.5
mv results_sdv.csv results_sdv_500_refit.csv