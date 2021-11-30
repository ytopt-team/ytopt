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
#ifndef SW4_EWCUDA
#define SW4_EWCUDA

#ifdef SW4_CUDA
#include <cuda_runtime.h>
#endif

class EWCuda 
{
 public:
   int m_nstream, m_ndevice, m_active_gpu;
#ifdef SW4_CUDA
   cudaStream_t* m_stream;
#endif
   EWCuda( int ndevice, int nstream ); 
   ~EWCuda();
   bool has_gpu(){return m_ndevice>0;}
   void reset_gpu();
   void initialize_gpu( int myrank );
   void sync_stream( int st );
   void sync_device();
};

#define RADIUS 2
#define DIAMETER (2*RADIUS+1)
#define RHS4_BLOCKX 16
#define RHS4_BLOCKY 16
#define ADDSGD4_BLOCKX 24
#define ADDSGD4_BLOCKY 16

#endif
