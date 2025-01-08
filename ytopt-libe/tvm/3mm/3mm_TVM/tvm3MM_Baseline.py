import numpy as np
import tvm
from tvm import autotvm

import time
from tvm import te
import logging
from datetime import datetime
import argparse
import json

logData = {}
logPath = 'logs/'
resultsPath = 'results/'

def matmul_basic(N, L, M, O, P, dtype):

    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    
    C = te.placeholder((M, O), name="C", dtype=dtype)
    D = te.placeholder((O, P), name="D", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    l = te.reduce_axis((0, O), name="l")
    m = te.reduce_axis((0, M), name="m")

    E = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="E")
    F = te.compute((M, P), lambda i, j: te.sum(C[i, l] * D[l, j], axis=l), name="F")
    G = te.compute((N, P), lambda i, j: te.sum(E[i, m] * F[m, j], axis=m), name="G")
    
    s1 = te.create_schedule(E.op)
    s2 = te.create_schedule(F.op)
    s3 = te.create_schedule(G.op)

    # schedule
    y, x = s1[E].op.axis
    k = s1[E].op.reduce_axis[0]
    y1, x1 = s2[F].op.axis
    l = s2[F].op.reduce_axis[0]
    y2, x2 = s3[G].op.axis
    m = s3[G].op.reduce_axis[0]
    
    #using 8 as the tiling factor
    yo, yi = s1[E].split(y, 8) 
    xo, xi = s1[E].split(x, 8)
    yo1, yi1 = s2[F].split(y1, 8) 
    xo1, xi1 = s2[F].split(x1, 8)
    yo2, yi2 = s3[G].split(y2, 8) 
    xo2, xi2 = s3[G].split(x2, 8)

    s1[E].reorder(yo, xo, k, yi, xi)
    s2[F].reorder(yo1, xo1, l, yi1, xi1)
    s3[G].reorder(yo2, xo2, m, yi2, xi2)

    return s3, [A, B, C, D, G]



def main(datasize):

    if datasize == 'L':  
        N,L,M,O,P = 800, 900, 1000, 1100, 1200
    else:
        N,L,M,O,P = 1600, 1800, 2000, 2200, 2400

    s, arg_bufs = matmul_basic(N, L, M, O, P,"float64")
    func = tvm.build(s,arg_bufs)
    a_np = np.random.uniform(size=(N, L)).astype(np.float64)
    b_np = np.random.uniform(size=(L, M)).astype(np.float64)
    c_np = np.random.uniform(size=(M, O)).astype(np.float64)
    d_np = np.random.uniform(size=(O, P)).astype(np.float64)

    mm_np = a_np.dot(b_np).dot(c_np.dot(d_np))
    mm_tvm = tvm.nd.empty(mm_np.shape, dtype='float64')

    start = time.time()
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), tvm.nd.array(c_np), tvm.nd.array(d_np), mm_tvm)
    end = time.time()
    runTime = round(end-start,2)
    logging.info("Elpased time = {} secs".format(runTime))
    logData['ElapsedTime'] = runTime
    with open(resultsPath+'Baseline_3MM.json', "w") as json_file:
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

        logging.basicConfig(filename=logPath+'Baseline_3MM.log', level=logging.DEBUG,filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        logging.info('TVM Baseline 3MM - {}'.format(datetime.now()))

        main(size)

    except Exception as e:
        logging.error("Exception occured = ",e)

    