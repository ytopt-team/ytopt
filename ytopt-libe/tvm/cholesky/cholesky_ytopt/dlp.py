import numpy as np
import tvm
from tvm import autotvm

import time
from tvm import te


from datetime import datetime


def cholesky_basic(N,dtype):

    A = te.placeholder((N, N), name="A", dtype=dtype)
    At = te.placeholder((N, N), name="At", dtype=dtype)

    k = te.reduce_axis((0, N), name="k")
    C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * At[k, j], axis=k), name="C")

    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    
    #using 8 as the tiling factor
    yo, yi = s[C].split(y, #P0) 
    xo, xi = s[C].split(x, #P1)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, At, C]

def getPositiveDefinite(N):
    A = np.random.randn(N, N).astype(np.float64)
    return A.dot(A.T)


def main():
    
    N = 2000    
    # N = 4000
    s, arg_bufs = cholesky_basic(N, "float64")
    func = tvm.build(s,arg_bufs)
    a_np = np.linalg.cholesky(getPositiveDefinite(N))
    b_np = a_np.T
    c_np = a_np.dot(b_np)
    c_tvm = tvm.nd.empty(c_np.shape, dtype='float64')
 
    start = time.time()
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)
    end = time.time()

    runTime = round(end-start,3)

    print("Elapsed time = {}".format(runTime))



if __name__ == '__main__':

        main()


    
