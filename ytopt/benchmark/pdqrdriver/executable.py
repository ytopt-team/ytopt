import os
import sys
import argparse

def write_input(params):
    with open("QR.in", "w") as f:
        f.write("%d\n"%(1))
        # f.write("%s, %d, %d, %d, %d, %d, %d, %f\n"%(param[0], param[1], param[2], param[5], param[6], param[9], param[10], param[11]))
        f.write("%s, %d, %d, %d, %d, %d, %d, %f\n"%(
            params['fac'],
            params['m'],
            params['n'],
            params['mb'],
            params['nb'],
            params['p'],
            params['q'],
            params['thresh']))

# ('fac', 'm', 'n', 'nodes',
#'cores', 'mb', 'nb', 'nth', 'nproc', 'p', 'q', 'thresh').
# params = [('QR', m, n, nodes, cores, mb, nb, nth, nproc, p, q, 1.)]
#             0    1  2   3       4     5   6   7    8     9  10 11

def read_output():
    with open('QR.out', 'r') as fout:
        time = sys.float_info.max
        for line in fout.readlines():
            words = line.split()
            # WRITE( NOUT, FMT = 9993 ) 'WALL', M, N, MB, NB, NPROW, NPCOL, WTIME( 1 ), TMFLOPS, PASSED, FRESID
            if (len(words) > 0 and words[0] == "WALL"):
                if (words[9] == "PASSED"):
                    # m  = int(words[1])
                    # n  = int(words[2])
                    # mb = int(words[3])
                    # nb = int(words[4])
                    # p  = int(words[5])
                    # q  = int(words[6])
                    # thresh = float(words[10])
                    time = float(words[7])
                    break
        return time


def create_parser():
    'command line parser for keras'

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('--fac', type=str, default='QR')
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--mb', type=int, default=1)
    parser.add_argument('--nb', type=int, default=1)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--q', type=int, default=1)
    parser.add_argument('--thresh', type=float, default=1.)

    return(parser)

parser = create_parser()
cmdline_args = parser.parse_args()
param_dict = vars(cmdline_args)
write_input(param_dict)

os.system('/projects/datascience/regele/ztune/examples/scalapack-driver/bin/theta/pdqrdriver 2>> QR.err')

time = read_output()
print('DH-OUTPUT: ', time)