//  SW4 LICENSE
// # ----------------------------------------------------------------------
// # SW4 - Seismic Waves, 4th order
// # ----------------------------------------------------------------------
// # Copyright (c) 2013, Lawrence Livermore National Security, LLC. 
// # Produced at the Lawrence Livermore National Laboratory. 
// # 
// # Written by:
// # N. Anders Petersson (petersson1@llnl.gov)
// # Bjorn Sjogreen      (sjogreen2@llnl.gov)
// # 
// # LLNL-CODE-643337 
// # 
// # All rights reserved. 
// # 
// # This file is part of SW4, Version: 1.0
// # 
// # Please also read LICENCE.txt, which contains "Our Notice and GNU General Public License"
// # 
// # This program is free software; you can redistribute it and/or modify
// # it under the terms of the GNU General Public License (as published by
// # the Free Software Foundation) version 2, dated June 1991. 
// # 
// # This program is distributed in the hope that it will be useful, but
// # WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// # conditions of the GNU General Public License for more details. 
// # 
// # You should have received a copy of the GNU General Public License
// # along with this program; if not, write to the Free Software
// # Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA 
#include "EWCuda.h"
#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
using namespace std;

//-----------------------------------------------------------------------
EWCuda::EWCuda( int ndevice, int nstream )
{
   m_ndevice = ndevice;
   m_nstream = nstream;
   m_active_gpu = 0;
#ifdef SW4_CUDA
   if( nstream > 0 )
      m_stream  = new cudaStream_t[nstream];
   else
      m_stream = static_cast<cudaStream_t*>(0);
#endif
}

//-----------------------------------------------------------------------
void EWCuda::reset_gpu()
{
#ifdef SW4_CUDA
   if( m_ndevice > 0 )
      cudaDeviceReset();
#endif
}

//-----------------------------------------------------------------------
void EWCuda::sync_stream( int st )
{
#ifdef SW4_CUDA
   cudaError_t retcode;
   retcode = cudaStreamSynchronize(m_stream[st]);
   if( retcode != cudaSuccess )
      cout << "Error EWCuda::EWCuda, cudaStreamSynchronize no " << st << " returned " <<
	 cudaGetErrorString(retcode) << endl;
#endif
}

int mystrcmp(void const *a, void const *b) { 
  char const *aa = (char const *)a;
  char const *bb = (char const *)b;

  return strcmp(aa, bb);
}

//-----------------------------------------------------------------------
void EWCuda::sync_device( )
{
#ifdef SW4_CUDA
   cudaError_t retcode;
   retcode = cudaDeviceSynchronize();
   if( retcode != cudaSuccess )
      cout << "Error EWCuda::EWCuda, cudaDeviceSynchronize returned " <<
	 cudaGetErrorString(retcode) << endl;
#endif
}

extern "C"
void setupgpu(int verbose)
{
#ifdef SW4_CUDA
  int rank, nrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nrank);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char (*hosts)[MPI_MAX_PROCESSOR_NAME] = (char (*)[MPI_MAX_PROCESSOR_NAME])malloc(nrank*(sizeof *hosts));
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int namelen;
  
  // each rank collects hostname of all nodes
  MPI_Get_processor_name(hostname, &namelen);
  strcpy(hosts[rank], hostname);
  for (int i = 0; i < nrank; i++) {
    MPI_Bcast(hosts[i], MPI_MAX_PROCESSOR_NAME, MPI_CHAR, i, MPI_COMM_WORLD);
  }

  // sort list of names
  qsort(hosts, nrank, MPI_MAX_PROCESSOR_NAME, mystrcmp);

  // assign same color to the same node
  int color = 0;
  for (int i = 0; i < nrank; i++) {
    if (i > 0) {
      if (strcmp(hosts[i-1], hosts[i]) != 0) color++;
    }
    if (strcmp(hosts[i], hostname) == 0) break;
  }
  
  MPI_Comm new_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &new_comm);
  int new_rank;
  MPI_Comm_rank(new_comm, &new_rank);

  int ngpu;
  cudaError_t ierr;
  ierr = cudaGetDeviceCount(&ngpu);
  if (ierr != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(ierr));
    exit(0);
  }

  int igpu = new_rank % ngpu;
  if (verbose) 
    printf("P(%d): %s using gpu id %d\n", rank, hosts[rank], igpu);
  ierr = cudaSetDevice(igpu);
  if (ierr != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(ierr));
    exit(0);
  }

  free(hosts);
#endif
}

//-----------------------------------------------------------------------
void EWCuda::initialize_gpu(int myrank)
{
#ifdef SW4_CUDA
   if( m_ndevice > 0){
     setupgpu(0);
     cudaError_t retcode;
     for (int i = 0; i < m_nstream; i++)
     {
       retcode = cudaStreamCreate(&m_stream[i]);
       if( retcode != cudaSuccess )
         cout << "Error EWCuda::EWCuda, cudaStreamCreate no " << i << " returned " <<
           cudaGetErrorString(retcode) << endl;
     }
     
   }
#endif
}

//-----------------------------------------------------------------------
EWCuda::~EWCuda()
{
#ifdef SW4_CUDA
   for( int s=0 ; s < m_nstream ; s++ )
      cudaStreamDestroy(m_stream[s]);
   if( m_nstream > 0 )
      delete[] m_stream;
#endif
}
