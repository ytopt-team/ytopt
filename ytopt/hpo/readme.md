# Using ytopt for hyperparameter optimization of a CNN simulation

This directory includes an example for using ytopt for hyperparameter optimization of a CNN simulation. We will demostrate how to use ytopt to conduct the hyperparameter optimization with 7 and 17 hyperparameters.

1. Hyperparameter optimization with 7 hyperparameters
For 7 hyperparameters, replace the code mold dlp.py with dlp.py.7parameters, problem.py with problem.py.7parameters. Then use ytopt to do the autotuning

2. Hyperparameter optimization with 17 hyperparameters
For 17 hyperparameters, replace the code mold dlp.py with dlp.py.17parameters, problem.py with problem.py.17parameters.

Note 1:
The order of the strings used in ytopt is in lexicographic (dictionary) order, not numeric order.
If the number of tuning parameters is 10 or less (p0, p1, ..., p9), its lexicographic string order
is the same as its numeric order. However, if the number of tuning parameters is more than 10 (p0, p1, ..., p16), its lexicographic string order is p0, p1, p10, p11, p12, p13, p14, p15, p16, p2, p3, p4, p5, p6, p7, p8, p9. When you define the output order of the parameters, you have to define them based on the lexicographic string order.

Note 2:
If the number of tuning parameters is more than 10, when you define a parameter in the code mold,
you have to use the following format: 

#P+number+blank space

For example, replace "m=11" with "m=#P11 "; "n=1" with "n=#P1 ". In this way, when ytopt searches the parameter in the code mold to replace p1, it will only replace the one in "n=#P1 ". Otherwise, it causes the error.
