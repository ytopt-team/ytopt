/*
 *                 Copyright (C) 2017, UChicago Argonne, LLC
 *                            All Rights Reserved
 *
 *           Hardware/Hybrid Cosmology Code (HACC), Version 1.0
 *
 * Salman Habib, Adrian Pope, Hal Finkel, Nicholas Frontiere, Katrin Heitmann,
 *      Vitali Morozov, Jeffrey Emberson, Thomas Uram, Esteban Rangel
 *                        (Argonne National Laboratory)
 *
 *  David Daniel, Patricia Fasel, Chung-Hsing Hsu, Zarija Lukic, James Ahrens
 *                      (Los Alamos National Laboratory)
 *
 *                               George Zagaris
 *                                 (Kitware)
 *
 *                            OPEN SOURCE LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer. Software changes,
 *      modifications, or derivative works, should be noted with comments and
 *      the author and organization’s name.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. Neither the names of UChicago Argonne, LLC or the Department of Energy
 *      nor the names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior written
 *      permission.
 *
 *   4. The software and the end-user documentation included with the
 *      redistribution, if any, must include the following acknowledgment:
 *
 *     "This product includes software produced by UChicago Argonne, LLC under
 *      Contract No. DE-AC02-06CH11357 with the Department of Energy."
 *
 * *****************************************************************************
 *                                DISCLAIMER
 * THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. NEITHER THE
 * UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR 
 * UCHICAGO ARGONNE, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, 
 * EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE
 * ACCURARY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, DATA, APPARATUS,
 * PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE
 * PRIVATELY OWNED RIGHTS.
 *
 * *****************************************************************************
 */

// C++
#include <iostream>
#include <iomanip>

// C
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// parallelism
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// HACC
#include "complex-type.h"
#include "AlignedAllocator.h"
#include "Error.h"
#include "TimingStats.h"

// HACC DFFT
#include "Distribution.hpp"
#ifndef DFFT_TIMING
#define DFFT_TIMING 1
#endif
#include "Dfft.hpp"

#define ALIGN 16

using namespace hacc;



uint64_t double_to_uint64_t(double d) {
  uint64_t i;
  memcpy(&i, &d, 8);
  return i;
}



void assign_delta_function(Dfft &dfft, complex_t *a)
{
  // location of my rank in r-space
  const int *self = dfft.self_rspace();
  
  // local grid dimensions in r-space
  const int *local_ng = dfft.local_ng_rspace();

  complex_t zero(0.0, 0.0);
  complex_t one(1.0, 0.0);
  size_t local_indx = 0;
  for(size_t i=0; i<(size_t)local_ng[0]; i++) {
    size_t global_i = local_ng[0]*self[0] + i;

    for(size_t j=0; j<(size_t)local_ng[1]; j++) {
      size_t global_j = local_ng[1]*self[1] + j;

      for(size_t k=0; k<(size_t)local_ng[2]; k++) {
	size_t global_k = local_ng[2]*self[2] + k;

	if(global_i == 0 &&
	   global_j == 0 &&
	   global_k == 0)
	  a[local_indx] = one;
	else
	  a[local_indx] = zero;

	local_indx++;
      }
    }
  }
}



void check_kspace(Dfft &dfft, complex_t *a)
{
  double LocalRealMin, LocalRealMax, LocalImagMin, LocalImagMax;
  LocalRealMin = LocalRealMax = std::real(a[1]);
  LocalImagMin = LocalImagMax = std::imag(a[1]);

  size_t local_size = dfft.local_size();
  for(size_t local_indx=0; local_indx<local_size; local_indx++) {
    double re = std::real(a[local_indx]);
    double im = std::imag(a[local_indx]);

    LocalRealMin = re < LocalRealMin ? re : LocalRealMin;
    LocalRealMax = re > LocalRealMax ? re : LocalRealMax;
    LocalImagMin = im < LocalImagMin ? im : LocalImagMin;
    LocalImagMax = im > LocalImagMax ? im : LocalImagMax;
  }

  const MPI_Comm comm = dfft.parent_comm();
  double GlobalRealMin, GlobalRealMax, GlobalImagMin, GlobalImagMax;
  MPI_Allreduce(&LocalRealMin, &GlobalRealMin, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&LocalRealMax, &GlobalRealMax, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&LocalImagMin, &GlobalImagMin, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&LocalImagMax, &GlobalImagMax, 1, MPI_DOUBLE, MPI_MAX, comm);

  int rank;
  MPI_Comm_rank(comm, &rank);
  if(rank == 0) {
    std::cout << std::endl << "k-space:" << std::endl
	      << "real in " << std::scientific
	      << "[" << GlobalRealMin << "," << GlobalRealMax << "]"
	      << " = " << std::hex
	      << "[" << double_to_uint64_t(GlobalRealMin) << ","
	      << double_to_uint64_t(GlobalRealMax) << "]"
	      << std::endl
	      << "imag in " << std::scientific
	      << "[" << GlobalImagMin << "," << GlobalImagMax << "]"
	      << " = " << std::hex
      	      << "[" << double_to_uint64_t(GlobalImagMin) << ","
	      << double_to_uint64_t(GlobalImagMax) << "]"
	      << std::endl << std::endl << std::fixed;
  }
}



void check_rspace(Dfft &dfft, complex_t *a)
{
  // location of my rank in r-space
  const int *self = dfft.self_rspace();
  
  // local grid dimensions in r-space
  const int *local_ng = dfft.local_ng_rspace();

  double LocalRealMin, LocalRealMax, LocalImagMin, LocalImagMax;
  LocalRealMin = LocalRealMax = std::real(a[1]);
  LocalImagMin = LocalImagMax = std::imag(a[1]);

  const MPI_Comm comm = dfft.parent_comm();
  int rank;
  MPI_Comm_rank(comm, &rank);
  if(rank == 0)
    std::cout << std::endl << "r-space:" << std::endl;
  
  size_t local_indx = 0;
  for(size_t i=0; i<(size_t)local_ng[0]; i++) {
    size_t global_i = local_ng[0]*self[0] + i;

    for(size_t j=0; j<(size_t)local_ng[1]; j++) {
      size_t global_j = local_ng[1]*self[1] + j;

      for(size_t k=0; k<(size_t)local_ng[2]; k++) {
	size_t global_k = local_ng[2]*self[2] + k;

	if(global_i == 0 &&
	   global_j == 0 &&
	   global_k == 0) {
	  std::cout << "a[0,0,0] = " << std::fixed << a[local_indx]
		    << std::hex << " = ("
		    << double_to_uint64_t(std::real(a[local_indx]))
		    << ","
		    << double_to_uint64_t(std::imag(a[local_indx]))
		    << ")" << std::endl;
	} else {
	  double re = std::real(a[local_indx]);
	  double im = std::imag(a[local_indx]);
	  LocalRealMin = re < LocalRealMin ? re : LocalRealMin;
	  LocalRealMax = re > LocalRealMax ? re : LocalRealMax;
	  LocalImagMin = im < LocalImagMin ? im : LocalImagMin;
	  LocalImagMax = im > LocalImagMax ? im : LocalImagMax;
	}

	local_indx++;
      }
    }
  }  

  double GlobalRealMin, GlobalRealMax, GlobalImagMin, GlobalImagMax;
  MPI_Allreduce(&LocalRealMin, &GlobalRealMin, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&LocalRealMax, &GlobalRealMax, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&LocalImagMin, &GlobalImagMin, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&LocalImagMax, &GlobalImagMax, 1, MPI_DOUBLE, MPI_MAX, comm);

  if(rank == 0) {
    std::cout << "real in " << std::scientific
	      << "[" << GlobalRealMin << "," << GlobalRealMax << "]"
	      << " = " << std::hex
	      << "[" << double_to_uint64_t(GlobalRealMin) << ","
	      << double_to_uint64_t(GlobalRealMax) << "]"
	      << std::endl
	      << "imag in " << std::scientific
	      << "[" << GlobalImagMin << "," << GlobalImagMax << "]"
	      << " = " << std::hex
      	      << "[" << double_to_uint64_t(GlobalImagMin) << ","
	      << double_to_uint64_t(GlobalImagMax) << "]"
	      << std::endl << std::endl << std::fixed;
  }
}



void test(MPI_Comm comm,
	  size_t repetitions,
	  int const ng[])
{
  // instantiate Distribution object
  Distribution d(comm, ng);

  // instantiate Dfft object
  Dfft dfft(d);

  // fft is out-of-place so we need at least 2 arrays
  std::vector<complex_t, AlignedAllocator<complex_t, ALIGN> > a;
  std::vector<complex_t, AlignedAllocator<complex_t, ALIGN> > b;

  // query dfft object to find out how big local arrays must be
  size_t local_size = dfft.local_size();
  a.resize(local_size);
  b.resize(local_size);

  // have dfft make fftw plans based on allocated arrays
  dfft.makePlans(&a[0],  // forward output
		 &b[0],  // forward scratch
		 &a[0],  // backward input
		 &b[0]); // backward scratch

  // grab a copy of the 3D cartesian communicator that Distribution is using
  MPI_Comm CartComm = d.cart_3d();
  
  int rank;
  MPI_Comm_rank(CartComm, &rank);

  if(rank==0) {
    std::cout << std::endl
	      << "Hex representations of double precision floats"
	      << std::endl;
    double zero = 0.0;
    std::cout << std::scientific << zero << " = " << std::hex
	      << double_to_uint64_t(zero) << std::endl;
    double one = 1.0;
    std::cout << std::scientific << one << " = " << std::hex
	      << double_to_uint64_t(one) << std::endl;
    double Ng = 1.0*(((uint64_t)ng[0])*((uint64_t)ng[1])*((uint64_t)ng[2]));
    std::cout << std::fixed << Ng << " = " << std::hex
	      << double_to_uint64_t(Ng) << std::endl;
    std::cout << std::endl;
  }
  MPI_Barrier(CartComm);

  for(size_t i=0; i<repetitions; i++) {
    if(rank==0) {
      std::cout << std::endl << "TESTING " << i << std::endl << std::endl;
    }
    MPI_Barrier(CartComm);

    double start, stop;

    // set up input array "a"
    assign_delta_function(dfft, &a[0]);

    // execute forward fft from input array "a"
    start = MPI_Wtime();
    dfft.forward(&a[0]);
    stop = MPI_Wtime();
    printTimingStats(CartComm, "FORWARD   ", stop-start);

    // check array contents in k-space
    check_kspace(dfft, &a[0]);

    // execute backward fft into output array "a"
    start = MPI_Wtime();
    dfft.backward(&a[0]);
    stop = MPI_Wtime();
    printTimingStats(CartComm, "BACKWARD  ", stop-start);

    // check array contents in r-space
    check_rspace(dfft, &a[0]);
  }
}



int main(int argc, char *argv[])
{

  if(argc < 3) {
    std::cerr << "USAGE: " << argv[0] << " <n_repetitions> <ngx> [ngy ngz]" << std::endl;
    return -1;
  }
  
  MPI_Init(&argc, &argv);

  size_t repetitions = atol(argv[1]);
  int ng[3];
  ng[2] = ng[1] = ng[0] = atoi(argv[2]);
  if(argc > 4) {
    ng[1] = atoi(argv[3]);
    ng[2] = atoi(argv[4]);
  }
  
  double  time_start, time_end;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  time_start = MPI_Wtime();
  // initialize fftw3 openmp threads if necessary
#ifdef _OPENMP
  if(!fftw_init_threads())
    Error() << "fftw_init_threads() failed!";

  //set the env variables for thread affinity
  //omp_set_num_threads(#P0);
  setenv("OMP_PLACES","#P1",1);
  //system("echo $OMP_PLACES");
  setenv("OMP_PROC_BIND","#P2",1);
  //system("echo $OMP_PROC_BIND");
  setenv("OMP_SCHEDULE","#P3",1);
  //system("echo $OMP_SCHEDULE");

  int omt = omp_get_max_threads();
  fftw_plan_with_nthreads(omt);
  if(rank==0)
    std::cout << "Threads per process: " << omt << std::endl;
#endif

  test(MPI_COMM_WORLD, repetitions, ng);

  time_end = MPI_Wtime();
  if(rank == 0) 
	  std::cout <<  " Runtime(seconds): " << time_end - time_start << std::endl;
  
  MPI_Finalize();
  
  return 0;
}
