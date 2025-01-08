import numpy as np
import tvm
from tvm import autotvm

import time
from tvm import te
import scipy


from datetime import datetime


def LU_basic(N,dtype):

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


def main():
    
    N = 2000    
    # N = 4000
    s, arg_bufs = LU_basic(N, "float64")
    func = tvm.build(s,arg_bufs)
    A = np.random.uniform(-1,1,[N,N])
    P,a_np,b_np = scipy.linalg.lu(A)
    c_np = a_np.dot(b_np)
    c_tvm = tvm.nd.empty(c_np.shape, dtype='float64')
 
    start = time.time()
    func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)
    end = time.time()

    runTime = round(end-start,3)

    print("Elapsed time = {}".format(runTime))



if __name__ == '__main__':

        main()


    