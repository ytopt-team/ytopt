#!/usr/bin/env python
from __future__ import print_function
import math

def convert_to_number(s: str) -> float:
    multipliers = {
        'K': 1e3,
        'M': 1e6,
        'B': 1e9,
        'T': 1e12,
    }
    
    s = s.strip().upper()
    if s[-1] in multipliers:
        num = float(s[:-1])
        return num * multipliers[s[-1]]
    else:
        return float(s)

N = convert_to_number('18.8B') # number of parameters
D = convert_to_number('110.2B') #number of tokens
S = 0.98 #sparsity

#Proposed scaling law coefficients
# e = 1.69
# a = 406.4
# b = 410.7
# c = 93.45
# alpha = 0.34 
# beta = 0.28
# gamma = 0.001

#Hoffman coefficients
# e = 1.69
# a = 406.4
# b = 410.7
# alpha = 0.34 
# beta = 0.28

# Frantar co-efficients
# e = #P0
# _as = #P1
# bs = #P2
# cs = #P3
# bn = #P4
# ad = #P5
# bd = #P6

#Abnar co-efficients
# e = 0.94
# a = 16612.50
# b = 5455.67
# c = 0.4598
# d = 17.26
# alpha = 0.5962
# beta = 0.3954
# _lambda = -0.1666
# delta = 0.1603
# gamma = 0.1595


#Abnar co-efficients
e = #P0
a = #P1
b = #P2
c = #P3
d = #P4
alpha = #P5
beta = #P6
_lambda = #P7
delta = #P8
gamma = #P9

# Frantar scaling law
# term1 = e
# term2 = (_as * math.pow(1 - S, bs)) + cs
# term3 = math.pow(1 / N, bn)
# term4 = math.pow(ad / D, bd)

# loss = term1 + (term2 * term3) + term4

#Hoffman scaling law
# term1 = a / N**alpha
# term2 = b / D**beta

# loss = e + term1 + term2

# Abnar scaling law
term1 = e
term2 = a / math.pow(N, alpha)
term3 = b / math.pow(D, beta)
temp_1 = math.pow(1-S,_lambda)
term4 = c / temp_1
temp_2 = math.pow(1-S, delta)
temp_3 = math.pow(N, gamma)
term5 = d / (temp_2 * temp_3)

loss = term1 + term2 + term3 + term4 + term5


# proposed scaling law
# term1 = e * math.pow(1-S, gamma)
# temp_1 = a * math.pow(1-S, alpha)
# temp_2 = c * S
# term2 = (temp_1 + temp_2) * math.pow(1/N, alpha)
# term3 = b / math.pow(D, beta)

# loss = term1 + term2 + term3

print('Loss:', loss)
