#! /usr/bin/env bash
source activate ytune
export PYTHONWARNINGS="ignore"
cd /gpfs/jlse-fs0/users/jkoo/github/ytune/ytopt_sub/ytopt/benchmark/xsbench-omp-tl/xsbench
rm ytopt.log
####
python -m ytopt.search.ambs --evaluator ray --problem problem_s.Problem --max-evals=100 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge --set-SEED 2468 
mv results.csv results_rf_s_xsbench.csv
mv ytopt.log ytopt_rf_s_xsbench.log
####
python -m ytopt.search.ambs --evaluator ray --problem problem_m.Problem --max-evals=100 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge --set-SEED 2468 
mv results.csv results_rf_m_xsbench.csv
mv ytopt.log ytopt_rf_m_xsbench.log
####
python -m ytopt.search.ambs --evaluator ray --problem problem_l.Problem --max-evals=100 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge --set-SEED 2468 
mv results.csv results_rf_l_xsbench.csv
mv ytopt.log ytopt_rf_l_xsbench.log
####
python -m ytopt.search.ambs --evaluator ray --problem problem_xl.Problem --max-evals=30 --learner RF --set-KAPPA 1.96 --acq-func gp_hedge --set-SEED 2468 
mv results.csv results_rf_xl_xsbench.csv
mv ytopt.log ytopt_rf_xl_xsbench.log
####
python Run_online_TL.py --max_evals 10 --n_refit 10 --target xl --top 0.3
mv results_sdv.csv results_sdv_xl_0.3.csv
python Run_online_TL.py --max_evals 10 --n_refit 3 --target xl --top 0.3
mv results_sdv.csv results_sdv_xl_0.3_refit3.csv




