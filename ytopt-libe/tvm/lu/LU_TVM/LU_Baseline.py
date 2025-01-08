import numpy as np
import tvm
from tvm import autotvm

import time
from tvm import te
import logging
import argparse
from datetime import datetime
import json
import scipy

logData = {}

logPath = 'logs/'
resultsPath = 'results/'

def lu_basic(N,dtype):

    A = te.placeholder((N, N), name="A", dtype=dtype)
    At = te.placeholder((N, N), name="At", dtype=dtype)

    k = te.reduce_axis((0, N), name="k")
    C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * At[k, j], axis=k), name="C")

    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    
    #using 8 as the tiling factor
    yo, yi = s[C].split(y, 8) 
    xo, xi = s[C].split(x, 8)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, At, C]



def main(datasize):
    if datasize == 'L':  
        N = 2000
    else:
        N = 4000

    s, arg_bufs = lu_basic(N, "float64")
    func = tvm.build(s,arg_bufs)
    A = np.random.uniform(-1,1,[N,N])
    P,a_np,b_np = scipy.linalg.lu(A)
    c_np = a_np.dot(b_np)
    c_tvm = tvm.nd.empty(c_np.shape, dtype='float64')
    logging.info("Matrix shape{}".format(a_np.shape))
    start = time.time()
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)
    end = time.time()

    runTime = round(end-start,3)
    logging.info("Elpased time = {} secs".format(runTime))
    logData['ElapsedTime'] = runTime
    with open(resultsPath+'Baseline.json', "w") as json_file:
        json.dump(logData, json_file)


if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--size', type=str, choices=['L', 'XL'], help='L or XL')
        args = parser.parse_args()
        size = args.size
        if size == 'L':
            logPath = logPath + 'large/'
            resultsPath = resultsPath + 'large/'
        else:
            logPath = logPath + 'extraLarge/'
            resultsPath = resultsPath + 'extraLarge/'

        logging.basicConfig(filename=logPath+'Baseline_MM.log', level=logging.DEBUG,filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        logging.info('TVM Baseline LU Decomposition - {}'.format(datetime.now()))

        main(size)

    except Exception as e:
        logging.error("Exception occured = ",e)

    