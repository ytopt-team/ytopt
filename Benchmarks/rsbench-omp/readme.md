This directory includes RSBench (https://github.com/ANL-CESAR/RSBench) which is a mini-app representing a key computational kernel of the Monte Carlo neutron transport algorithm. 
Specifically, RSBench represents the multipole method of perfoming continuous energy macroscopic neutron cross section lookups. 
The mulitpole method is a recently developed strategy for building microscopic cross section data "on-the-fly" that requires orders of magnitude 
less memory storage as compared to traditional methods (e.g., those represented in XSBench). 
RSBench serves as a useful performance stand-in for full neutron transport applications like OpenMC that support multipole cross section representations.

We use the ytopt autotuning framework to autotune the benchmark RSBench.
