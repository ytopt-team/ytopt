Threads per process: 64
Initializing redistribution using a pencil layout on 4096 ranks.
distribution 1D: [4096:1:1]
distribution 3D: [16:16:16]
distribution 2z: [64:64:1]
distribution 2x: [1:64:64]
distribution 2y: [64:1:64]

Hex representations of double precision floats
0.000000e+00 = 0
1.000000e+00 = 3ff0000000000000
68719476736.000000 = 4230000000000000


TESTING 0

FORWARD     max 7.298e+00s  avg 6.221e+00s  min 5.045e+00s  dev 4.407e-01s

k-space:
real in [1.000000e+00,1.000000e+00] = [3ff0000000000000,3ff0000000000000]
imag in [0.000000e+00,0.000000e+00] = [0,0]

BACKWARD    max 7.403e+00s  avg 6.249e+00s  min 4.987e+00s  dev 4.386e-01s

r-space:
a[0,0,0] = (68719476736.000000,0.000000) = (4230000000000000,0)
real in [0.000000e+00,0.000000e+00] = [0,0]
imag in [0.000000e+00,0.000000e+00] = [0,0]


TESTING 1

FORWARD     max 7.049e+00s  avg 6.016e+00s  min 4.856e+00s  dev 4.448e-01s

k-space:
real in [1.000000e+00,1.000000e+00] = [3ff0000000000000,3ff0000000000000]
imag in [0.000000e+00,0.000000e+00] = [0,0]

BACKWARD    max 7.412e+00s  avg 6.266e+00s  min 4.999e+00s  dev 4.386e-01s

r-space:
a[0,0,0] = (68719476736.000000,0.000000) = (4230000000000000,0)
real in [0.000000e+00,0.000000e+00] = [0,0]
imag in [0.000000e+00,0.000000e+00] = [0,0]

 Runtime(seconds): 33.057395
Application 25317588 resources: utime ~692408s, stime ~72588s, Rss ~607692, inblocks ~0, outblocks ~1384
