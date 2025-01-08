import numpy as np
import tvm
from tvm import autotvm

import time
from tvm import te
import logging
from datetime import datetime
import argparse


logPath = 'logs/'
resultsPath = 'results/'

@autotvm.template("trial/tvm3matmul_v1") 
def matmul_3mm(N, L, M, O, P, size, dtype):

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

    # 2. get the config object
    cfg = autotvm.get_config()

    if size != 'L':

        cfg.define_knob("tile_y", [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000])
        cfg.define_knob("tile_x", [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 80, 100, 160, 200, 400, 800])
        cfg.define_knob("tile_y1", [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 40, 48, 50, 60, 75, 80, 100, 120, 150, 200, 240, 300, 400, 600, 1200])
        cfg.define_knob("tile_x1", [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000])
        cfg.define_knob("tile_y2", [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 40, 48, 50, 60, 75, 80, 100, 120, 150, 200, 240, 300, 400, 600, 1200])
        cfg.define_knob("tile_x2", [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 80, 100, 160, 200, 400, 800])
        
    else:    

        cfg.define_knob("tile_y", [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000])
        cfg.define_knob("tile_x", [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100, 160, 200, 320, 400, 800, 1600])
        cfg.define_knob("tile_y1", [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 32, 40, 48, 50, 60, 75, 80, 96, 100, 120, 150, 160, 200, 240, 300, 400, 480, 600, 800, 1200, 2400])
        cfg.define_knob("tile_x1", [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000])
        cfg.define_knob("tile_y2", [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 32, 40, 48, 50, 60, 75, 80, 96, 100, 120, 150, 160, 200, 240, 300, 400, 480, 600, 800, 1200, 2400])
        cfg.define_knob("tile_x2", [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100, 160, 200, 320, 400, 800, 1600])


    # schedule
    y, x = s1[E].op.axis
    k = s1[E].op.reduce_axis[0]
    y1, x1 = s2[F].op.axis
    l = s2[F].op.reduce_axis[0]
    y2, x2 = s3[G].op.axis
    m = s3[G].op.reduce_axis[0]
    
    #using 8 as the tiling factor
    yo, yi = s1[E].split(y, cfg["tile_y"].val) 
    xo, xi = s1[E].split(x, cfg["tile_x"].val)
    yo1, yi1 = s2[F].split(y1, cfg["tile_y1"].val) 
    xo1, xi1 = s2[F].split(x1, cfg["tile_x1"].val)
    yo2, yi2 = s3[G].split(y2, cfg["tile_y2"].val) 
    xo2, xi2 = s3[G].split(x2, cfg["tile_x2"].val)

    s1[E].reorder(yo, xo, k, yi, xi)
    s2[F].reorder(yo1, xo1, l, yi1, xi1)
    s3[G].reorder(yo2, xo2, m, yi2, xi2)

    return s3, [A, B, C, D, G]



def main(datasize):

    if datasize == 'L':  
        N,L,M,O,P = 800, 900, 1000, 1100, 1200
    else:
        N,L,M,O,P = 1600, 1800, 2000, 2200, 2400
    
    task = autotvm.task.create("trial/tvm3matmul_v1", args=(N, L, M, O, P, datasize, "float64"), target="llvm")

    measure_option = autotvm.measure_option(builder="local", 
                                        runner=autotvm.LocalRunner(number=1, repeat=1, timeout=200), # timeout=20
                                        )
    

    tuner = autotvm.tuner.XGBTuner(task)
    path = resultsPath + "tvmXGBTuner.json"
    start = time.time()
    tuner.tune(
    n_trial=56,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file(path)]
    )
    end = time.time()

    logging.info("Elpased time = {}".format(end-start))

    with autotvm.apply_history_best(path):
        with tvm.target.Target("llvm"):
            s, arg_bufs = matmul_3mm(N, L, M, O, P, datasize, "float64") 
            func = tvm.build(s, arg_bufs)


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

        logging.basicConfig(filename=logPath+'XGBTuner_3MM.log', level=logging.DEBUG,filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        logging.info('XGB Tuner 3MM - {}'.format(datetime.now()))


        main(size)

    except Exception as e:
        logging.error("Exception occured = ",e)
