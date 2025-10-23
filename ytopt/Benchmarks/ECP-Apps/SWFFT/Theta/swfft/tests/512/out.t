Threads per process: 64
Initializing redistribution using a pencil layout on 512 ranks.
distribution 1D: [512:1:1]
distribution 3D: [8:8:8]
distribution 2z: [32:16:1]
distribution 2x: [1:32:16]
distribution 2y: [32:1:16]

Hex representations of double precision floats
0.000000e+00 = 0
1.000000e+00 = 3ff0000000000000
8589934592.000000 = 4200000000000000


TESTING 0

FORWARD     max 6.631e+00s  avg 5.793e+00s  min 4.874e+00s  dev 3.716e-01s

k-space:
real in [1.000000e+00,1.000000e+00] = [3ff0000000000000,3ff0000000000000]
imag in [0.000000e+00,0.000000e+00] = [0,0]

BACKWARD    max 6.945e+00s  avg 5.885e+00s  min 4.864e+00s  dev 4.102e-01s

r-space:
a[0,0,0] = (8589934592.000000,0.000000) = (4200000000000000,0)
real in [0.000000e+00,0.000000e+00] = [0,0]
imag in [0.000000e+00,0.000000e+00] = [0,0]


TESTING 1

FORWARD     max 6.362e+00s  avg 5.519e+00s  min 4.608e+00s  dev 3.703e-01s

k-space:
real in [1.000000e+00,1.000000e+00] = [3ff0000000000000,3ff0000000000000]
imag in [0.000000e+00,0.000000e+00] = [0,0]

BACKWARD    max 7.096e+00s  avg 5.916e+00s  min 4.927e+00s  dev 4.116e-01s

r-space:
a[0,0,0] = (8589934592.000000,0.000000) = (4200000000000000,0)
real in [0.000000e+00,0.000000e+00] = [0,0]
imag in [0.000000e+00,0.000000e+00] = [0,0]

 Runtime(seconds): 29.819528
Application 25312690 resources: utime ~84033s, stime ~8985s, Rss ~643248, inblocks ~0, outblocks ~184
