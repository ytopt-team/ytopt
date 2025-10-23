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
#include "sw4.h"
#include "Sarray.h"
#include "GridPointSource.h"
#include "EWCuda.h"
#include <stdio.h>

#ifdef SW4_CUDA
#include <cuda_runtime.h>

__constant__ float_sw4 dev_acof[384];
__constant__ float_sw4 dev_ghcof[6];
__constant__ float_sw4 dev_bope[48];
__constant__ float_sw4 dev_sbop[5];

#include <iostream>
using namespace std;
//-----------------------------------------------------------------------
void copy_stencilcoefficients( float_sw4* acof, float_sw4* ghcof, float_sw4* bope )
{
   cudaError_t retcode;
   //   if( m_ndevice > 0 )
   {
      retcode = cudaMemcpyToSymbol( dev_acof,  acof, 384*sizeof(float_sw4));
      if( retcode != cudaSuccess )
	 cout << "Error copying acof to device constant memory. Error= "
	      << cudaGetErrorString(retcode) << endl;
      retcode = cudaMemcpyToSymbol( dev_bope,  bope,  48*sizeof(float_sw4));
      if( retcode != cudaSuccess )
	 cout << "Error copying bope to device constant memory. Error= "
	      << cudaGetErrorString(retcode) << endl;
      retcode = cudaMemcpyToSymbol( dev_ghcof, ghcof,  6*sizeof(float_sw4));
      if( retcode != cudaSuccess )
	 cout << "Error copying ghcof to device constant memory. Error= "
	      << cudaGetErrorString(retcode) << endl;
   }
}

//-----------------------------------------------------------------------
void copy_stencilcoefficients1( float_sw4* acof, float_sw4* ghcof, float_sw4* bope , float_sw4* sbop )
{
  cudaError_t retcode;
  {
    retcode = cudaMemcpyToSymbol( dev_acof,  acof, 384*sizeof(float_sw4));
    if( retcode != cudaSuccess )
      cout << "Error copying acof to device constant memory. Error= "
           << cudaGetErrorString(retcode) << endl;
    retcode = cudaMemcpyToSymbol( dev_bope,  bope,  48*sizeof(float_sw4));
    if( retcode != cudaSuccess )
      cout << "Error copying bope to device constant memory. Error= "
           << cudaGetErrorString(retcode) << endl;
    retcode = cudaMemcpyToSymbol( dev_ghcof, ghcof,  6*sizeof(float_sw4));
    if( retcode != cudaSuccess )
      cout << "Error copying ghcof to device constant memory. Error= "
           << cudaGetErrorString(retcode) << endl;
    retcode = cudaMemcpyToSymbol( dev_sbop,  sbop,  5*sizeof(float_sw4));
    if( retcode != cudaSuccess )
      cout << "Error copying bope to device constant memory. Error= "
           << cudaGetErrorString(retcode) << endl;
  }
}



//-----------------------------------------------------------------------
__global__ void pred_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			  float_sw4* up, float_sw4* u, float_sw4* um, float_sw4* lu, float_sw4* fo,
			  float_sw4* rho, float_sw4 dt2, int ghost_points )
{
   //   int myi = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   //   int myj = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   //   int myk = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   //   size_t i = myi-ifirst+(ilast-ifirst+1)*((myj-jfirst)+(jlast-jfirst+1)*(myk-kfirst));

   //   int myi = 2 + threadIdx.x + blockIdx.x*blockDim.x;
   //   int myj = 2 + threadIdx.y + blockIdx.y*blockDim.y;
   //   int myk = 2 + threadIdx.z + blockIdx.z*blockDim.z;
   //   size_t i = myi+(ilast-ifirst+1)*(myj+(jlast-jfirst+1)*myk);


   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   const size_t npts = static_cast<size_t>((ilast-ifirst+1))*(jlast-jfirst+1)*(klast-kfirst+1);

   for (size_t i = myi; i < 3*npts; i += nthreads) 
   {
      //       j = i/3;
      //       float_sw4 dt2orh = dt2/rho[j];
      //       float_sw4 dt2orh = dt2/rho[i/3];
      up[i  ] = 2*u[i  ]-um[i  ] + (dt2/rho[i/3])*(lu[i  ]+fo[i  ]);
   }

   //   size_t i = threadIdx.x + blockIdx.x * blockDim.x;
   //   const size_t nthreads = static_cast<size_t> (gridDim.x)*blockDim.x;
   //   const size_t npts = static_cast<size_t>((ilast-ifirst+1))*(jlast-jfirst+1)*(klast-kfirst+1);
   //   up[i] = 2*u[i] - um[i] + dt2/rho[i/3]*(lu[i]+fo[i]);
   //   i += nthreads;
   //   up[i] = 2*u[i] - um[i] + dt2/rho[i/3]*(lu[i]+fo[i]);
   //   i += nthreads;
   //   if( i < 3*npts )
   //   {
   //      up[i] = 2*u[i] - um[i] + dt2/rho[i/3]*(lu[i]+fo[i]);
   //      i += nthreads;
   //      if( i < 3*npts )
   //	 up[i] = 2*u[i] - um[i] + dt2/rho[i/3]*(lu[i]+fo[i]);
   //   }

   //   float_sw4 dt2orh = dt2/rho[i];
   //   i *= 3;
   //   up[i  ] = 2*u[i  ]-um[i  ] + dt2orh*(lu[i  ]+fo[i  ]);
   //   up[i+1] = 2*u[i+1]-um[i+1] + dt2orh*(lu[i+1]+fo[i+1]);
   //   up[i+2] = 2*u[i+2]-um[i+2] + dt2orh*(lu[i+2]+fo[i+2]);

   //   up[3*i  ] = 2*u[3*i  ]-um[3*i  ] + dt2/rho[i]*(lu[3*i  ]+fo[3*i  ]);
   //   up[3*i+1] = 2*u[3*i+1]-um[3*i+1] + dt2/rho[i]*(lu[3*i+1]+fo[3*i+1]);
   //   up[3*i+2] = 2*u[3*i+2]-um[3*i+2] + dt2/rho[i]*(lu[3*i+2]+fo[3*i+2]);
}

//-----------------------------------------------------------------------
__global__ void corr_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			  float_sw4* up, float_sw4* lu, float_sw4* fo,
			  float_sw4* rho, float_sw4 dt4, int ghost_points )
{
   //   int myi = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   //   int myj = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   //   int myk = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   //   size_t i = myi-ifirst+(ilast-ifirst+1)*((myj-jfirst)+(jlast-jfirst+1)*(myk-kfirst));
   //   float_sw4 dt4i12orh=dt4/(12*rho[i]);
   //   i *= 3;
   //   up[i  ] += dt4i12orh*(lu[i  ]+fo[i  ]);
   //   up[i+1] += dt4i12orh*(lu[i+1]+fo[i+1]);
   //   up[i+2] += dt4i12orh*(lu[i+2]+fo[i+2]);

   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   const size_t npts = static_cast<size_t>((ilast-ifirst+1))*(jlast-jfirst+1)*(klast-kfirst+1);

   for (size_t i = myi; i < 3*npts; i += nthreads) 
   {
      up[i  ] += dt4/(12*rho[i/3])*(lu[i  ]+fo[i  ]);
   }
   //   up[i  ] += dt4/(12*rho[i/3])*(lu[i  ]+fo[i  ]);
   //   i += nthreads;
   //   up[i  ] += dt4/(12*rho[i/3])*(lu[i  ]+fo[i  ]);
   //   i += nthreads;
   //   if( i < 3*npts )
   //   {
   //      up[i  ] += dt4/(12*rho[i/3])*(lu[i  ]+fo[i  ]);
   //      i += nthreads;
   //      if( i < 3*npts )
   //	 up[i  ] += dt4/(12*rho[i/3])*(lu[i  ]+fo[i  ]);
   //   }
}

//-----------------------------------------------------------------------
__global__ void dpdmt_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			  float_sw4* up, float_sw4* u, float_sw4* um,
			  float_sw4* u2, float_sw4 dt2i, int ghost_points )
{
   //   int myi = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   //   int myj = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   //   int myk = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   //   size_t i = 3*(myi-ifirst+(ilast-ifirst+1)*((myj-jfirst)+(jlast-jfirst+1)*(myk-kfirst)));
   //   u2[i  ] = dt2i*(up[i  ]-2*u[i  ]+um[i  ]);
   //   u2[i+1] = dt2i*(up[i+1]-2*u[i+1]+um[i+1]);
   //   u2[i+2] = dt2i*(up[i+2]-2*u[i+2]+um[i+2]);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   const size_t npts = static_cast<size_t>((ilast-ifirst+1))*(jlast-jfirst+1)*(klast-kfirst+1);
   //   u2[i  ] = dt2i*(up[i  ]-2*u[i  ]+um[i  ]);
   //   i += nthreads;
   //   u2[i  ] = dt2i*(up[i  ]-2*u[i  ]+um[i  ]);
   //   i += nthreads;
   //   if( i < 3*npts )
   //   {
   //      u2[i  ] = dt2i*(up[i  ]-2*u[i  ]+um[i  ]);
   //      i += nthreads;
   //      if( i < 3*npts )
   //	 u2[i  ] = dt2i*(up[i  ]-2*u[i  ]+um[i  ]);
   //   }
   for (size_t i = myi; i < 3*npts; i += nthreads) 
      u2[i  ] = dt2i*(up[i  ]-2*u[i  ]+um[i  ]);
}

//-----------------------------------------------------------------------
__global__ void addsgd4_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
		      float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
			     float_sw4 beta, int ghost_points )
{
#define rho(i,j,k) a_rho[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define up(c,i,j,k) a_up[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define u(c,i,j,k) a_u[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define um(c,i,j,k) a_um[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define strx(i) a_strx[(i-ifirst)]
#define dcx(i) a_dcx[(i-ifirst)]
#define cox(i) a_cox[(i-ifirst)]
#define stry(j) a_stry[(j-jfirst)]
#define dcy(j) a_dcy[(j-jfirst)]
#define coy(j) a_coy[(j-jfirst)]
#define strz(k) a_strz[(k-kfirst)]
#define dcz(k) a_dcz[(k-kfirst)]
#define coz(k) a_coz[(k-kfirst)]
   if( beta == 0 )
      return;

   const size_t ni = ilast-ifirst+1;
   const size_t nij = ni*(jlast-jfirst+1);
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
      float_sw4 birho=beta/rho(i,j,k);
      for( int c=0 ; c < 3 ; c++ )
      {
	 up(c,i,j,k) -= birho*( 
		  // x-differences
		   strx(i)*coy(j)*coz(k)*(
       rho(i+1,j,k)*dcx(i+1)*
                   ( u(c,i+2,j,k) -2*u(c,i+1,j,k)+ u(c,i,  j,k))
      -2*rho(i,j,k)*dcx(i)  *
                   ( u(c,i+1,j,k) -2*u(c,i,  j,k)+ u(c,i-1,j,k))
      +rho(i-1,j,k)*dcx(i-1)*
                   ( u(c,i,  j,k) -2*u(c,i-1,j,k)+ u(c,i-2,j,k)) 
      -rho(i+1,j,k)*dcx(i+1)*
                   (um(c,i+2,j,k)-2*um(c,i+1,j,k)+um(c,i,  j,k)) 
      +2*rho(i,j,k)*dcx(i)  *
                   (um(c,i+1,j,k)-2*um(c,i,  j,k)+um(c,i-1,j,k)) 
      -rho(i-1,j,k)*dcx(i-1)*
                   (um(c,i,  j,k)-2*um(c,i-1,j,k)+um(c,i-2,j,k)) ) +
// y-differences
      stry(j)*cox(i)*coz(k)*(
      +rho(i,j+1,k)*dcy(j+1)*
                   ( u(c,i,j+2,k) -2*u(c,i,j+1,k)+ u(c,i,j,  k)) 
      -2*rho(i,j,k)*dcy(j)  *
                   ( u(c,i,j+1,k) -2*u(c,i,j,  k)+ u(c,i,j-1,k))
      +rho(i,j-1,k)*dcy(j-1)*
                   ( u(c,i,j,  k) -2*u(c,i,j-1,k)+ u(c,i,j-2,k)) 
      -rho(i,j+1,k)*dcy(j+1)*
                   (um(c,i,j+2,k)-2*um(c,i,j+1,k)+um(c,i,j,  k)) 
      +2*rho(i,j,k)*dcy(j)  *
                   (um(c,i,j+1,k)-2*um(c,i,j,  k)+um(c,i,j-1,k)) 
      -rho(i,j-1,k)*dcy(j-1)*
                   (um(c,i,j,  k)-2*um(c,i,j-1,k)+um(c,i,j-2,k)) ) +
       strz(k)*cox(i)*coy(j)*(
// z-differences
      +rho(i,j,k+1)*dcz(k+1)* 
                 ( u(c,i,j,k+2) -2*u(c,i,j,k+1)+ u(c,i,j,k  )) 
      -2*rho(i,j,k)*dcz(k)  *
                 ( u(c,i,j,k+1) -2*u(c,i,j,k  )+ u(c,i,j,k-1))
      +rho(i,j,k-1)*dcz(k-1)*
                 ( u(c,i,j,k  ) -2*u(c,i,j,k-1)+ u(c,i,j,k-2)) 
      -rho(i,j,k+1)*dcz(k+1)*
                 (um(c,i,j,k+2)-2*um(c,i,j,k+1)+um(c,i,j,k  )) 
      +2*rho(i,j,k)*dcz(k)  *
                 (um(c,i,j,k+1)-2*um(c,i,j,k  )+um(c,i,j,k-1)) 
      -rho(i,j,k-1)*dcz(k-1)*
                 (um(c,i,j,k  )-2*um(c,i,j,k-1)+um(c,i,j,k-2)) ) 
				);
      }
   }
#undef rho
#undef up
#undef u
#undef um
#undef strx
#undef dcx
#undef cox
#undef stry
#undef dcy
#undef coy
#undef strz
#undef dcz
#undef coz
}

//-----------------------------------------------------------------------
__global__ void addsgd6_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
		      float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
			     float_sw4 beta, int ghost_points )
{
#define rho(i,j,k) a_rho[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define up(c,i,j,k) a_up[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define u(c,i,j,k) a_u[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define um(c,i,j,k) a_um[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define strx(i) a_strx[(i-ifirst)]
#define dcx(i) a_dcx[(i-ifirst)]
#define cox(i) a_cox[(i-ifirst)]
#define stry(j) a_stry[(j-jfirst)]
#define dcy(j) a_dcy[(j-jfirst)]
#define coy(j) a_coy[(j-jfirst)]
#define strz(k) a_strz[(k-kfirst)]
#define dcz(k) a_dcz[(k-kfirst)]
#define coz(k) a_coz[(k-kfirst)]
   const size_t ni = ilast-ifirst+1;
   const size_t nij = ni*(jlast-jfirst+1);
   int i = ifirst + ghost_points + 1 + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + 1 + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + 1 + threadIdx.z + blockIdx.z*blockDim.z;
   if ( (i <= ilast-3) && (j <= jlast-3) && (k <= klast-3) )
   {
   float_sw4 birho=0.5*beta/rho(i,j,k);
   for( int c=0 ; c < 3 ; c++ )
   {
      up(c,i,j,k) += birho*( 
       strx(i)*coy(j)*coz(k)*(
// x-differences
         (rho(i+2,j,k)*dcx(i+2)+rho(i+1,j,k)*dcx(i+1))*(
         u(c,i+3,j,k) -3*u(c,i+2,j,k)+ 3*u(c,i+1,j,k)- u(c,i, j,k) 
      -(um(c,i+3,j,k)-3*um(c,i+2,j,k)+3*um(c,i+1,j,k)-um(c,i, j,k)) )
      -3*(rho(i+1,j,k)*dcx(i+1)+rho(i,j,k)*dcx(i))*(
         u(c,i+2,j,k)- 3*u(c,i+1,j,k)+ 3*u(c,i, j,k)- u(c,i-1,j,k)
      -(um(c,i+2,j,k)-3*um(c,i+1,j,k)+3*um(c,i, j,k)-um(c,i-1,j,k)) )
      +3*(rho(i,j,k)*dcx(i)+rho(i-1,j,k)*dcx(i-1))*(
         u(c,i+1,j,k)- 3*u(c,i,  j,k)+3*u(c,i-1,j,k)- u(c,i-2,j,k) 
      -(um(c,i+1,j,k)-3*um(c,i, j,k)+3*um(c,i-1,j,k)-um(c,i-2,j,k)) )
       - (rho(i-1,j,k)*dcx(i-1)+rho(i-2,j,k)*dcx(i-2))*(
         u(c,i, j,k)- 3*u(c,i-1,j,k)+ 3*u(c,i-2,j,k)- u(c,i-3,j,k) 
      -(um(c,i, j,k)-3*um(c,i-1,j,k)+3*um(c,i-2,j,k)-um(c,i-3,j,k)) )
                 ) +  stry(j)*cox(i)*coz(k)*(
// y-differences
         (rho(i,j+2,k)*dcy(j+2)+rho(i,j+1,k)*dcy(j+1))*(
         u(c,i,j+3,k) -3*u(c,i,j+2,k)+ 3*u(c,i,j+1,k)- u(c,i,  j,k)
      -(um(c,i,j+3,k)-3*um(c,i,j+2,k)+3*um(c,i,j+1,k)-um(c,i,  j,k)) )
      -3*(rho(i,j+1,k)*dcy(j+1)+rho(i,j,k)*dcy(j))*(
         u(c,i,j+2,k) -3*u(c,i,j+1,k)+ 3*u(c,i,  j,k)- u(c,i,j-1,k) 
      -(um(c,i,j+2,k)-3*um(c,i,j+1,k)+3*um(c,i,  j,k)-um(c,i,j-1,k)) )
      +3*(rho(i,j,k)*dcy(j)+rho(i,j-1,k)*dcy(j-1))*(
         u(c,i,j+1,k)- 3*u(c,i, j,k)+ 3*u(c,i,j-1,k)- u(c,i,j-2,k) 
      -(um(c,i,j+1,k)-3*um(c,i, j,k)+3*um(c,i,j-1,k)-um(c,i,j-2,k)) )
       - (rho(i,j-1,k)*dcy(j-1)+rho(i,j-2,k)*dcy(j-2))*(
         u(c,i, j,k)- 3*u(c,i,j-1,k)+  3*u(c,i,j-2,k)- u(c,i,j-3,k) 
      -(um(c,i, j,k)-3*um(c,i,j-1,k)+ 3*um(c,i,j-2,k)-um(c,i,j-3,k)) )
                 ) +  strz(k)*cox(i)*coy(j)*(
// z-differences
         ( rho(i,j,k+2)*dcz(k+2) + rho(i,j,k+1)*dcz(k+1) )*(
         u(c,i,j,k+3)- 3*u(c,i,j,k+2)+ 3*u(c,i,j,k+1)- u(c,i,  j,k) 
      -(um(c,i,j,k+3)-3*um(c,i,j,k+2)+3*um(c,i,j,k+1)-um(c,i,  j,k)) )
      -3*(rho(i,j,k+1)*dcz(k+1)+rho(i,j,k)*dcz(k))*(
         u(c,i,j,k+2) -3*u(c,i,j,k+1)+ 3*u(c,i,  j,k)- u(c,i,j,k-1) 
      -(um(c,i,j,k+2)-3*um(c,i,j,k+1)+3*um(c,i,  j,k)-um(c,i,j,k-1)) )
      +3*(rho(i,j,k)*dcz(k)+rho(i,j,k-1)*dcz(k-1))*(
         u(c,i,j,k+1)- 3*u(c,i,  j,k)+ 3*u(c,i,j,k-1)-u(c,i,j,k-2) 
      -(um(c,i,j,k+1)-3*um(c,i,  j,k)+3*um(c,i,j,k-1)-um(c,i,j,k-2)) )
       - (rho(i,j,k-1)*dcz(k-1)+rho(i,j,k-2)*dcz(k-2))*(
         u(c,i,  j,k) -3*u(c,i,j,k-1)+ 3*u(c,i,j,k-2)- u(c,i,j,k-3)
      -(um(c,i,  j,k)-3*um(c,i,j,k-1)+3*um(c,i,j,k-2)-um(c,i,j,k-3)) )
					     )  );
   }
   }
#undef rho
#undef up
#undef u
#undef um
#undef strx
#undef dcx
#undef cox
#undef stry
#undef dcy
#undef coy
#undef strz
#undef dcz
#undef coz
}
//-----------------------------------------------------------------------
__global__ void addsgd6c_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy, 
		      float_sw4* a_strx, float_sw4* a_stry,
		      float_sw4* a_jac, float_sw4* a_cox,  float_sw4* a_coy,  
		      float_sw4 beta, int ghost_points )
{
#define rho(i,j,k) a_rho[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define up(c,i,j,k) a_up[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define u(c,i,j,k)   a_u[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define um(c,i,j,k) a_um[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define jac(i,j,k) a_jac[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define strx(i) a_strx[(i-ifirst)]
#define dcx(i)  a_dcx[(i-ifirst)]
#define cox(i)  a_cox[(i-ifirst)]
#define stry(j) a_stry[(j-jfirst)]
#define dcy(j)  a_dcy[(j-jfirst)]
#define coy(j)  a_coy[(j-jfirst)]

   if( beta == 0. )
      return;

   const size_t ni = ilast-ifirst+1;
   const size_t nij = ni*(jlast-jfirst+1);
   int i = ifirst + ghost_points + 1 + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + 1 + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + 1 + threadIdx.z + blockIdx.z*blockDim.z;
   if ( (i <= ilast-3) && (j <= jlast-3) && (k <= klast-3) )
   {
   float_sw4 birho=0.5*beta/(rho(i,j,k)*jac(i,j,k));
	       for( int c=0 ; c < 3 ; c++ )
	       {
		 up(c,i,j,k) += birho*( 
       strx(i)*coy(j)*(
// x-differences
      (rho(i+2,j,k)*dcx(i+2)*jac(i+2,j,k)+rho(i+1,j,k)*dcx(i+1)*jac(i+1,j,k))*(
         u(c,i+3,j,k) -3*u(c,i+2,j,k)+ 3*u(c,i+1,j,k)- u(c,i, j,k) 
      -(um(c,i+3,j,k)-3*um(c,i+2,j,k)+3*um(c,i+1,j,k)-um(c,i, j,k)) )
      -3*(rho(i+1,j,k)*dcx(i+1)*jac(i+1,j,k)+rho(i,j,k)*dcx(i)*jac(i,j,k))*(
         u(c,i+2,j,k)- 3*u(c,i+1,j,k)+ 3*u(c,i, j,k)- u(c,i-1,j,k)
      -(um(c,i+2,j,k)-3*um(c,i+1,j,k)+3*um(c,i, j,k)-um(c,i-1,j,k)) )
      +3*(rho(i,j,k)*dcx(i)*jac(i,j,k)+rho(i-1,j,k)*dcx(i-1)*jac(i-1,j,k))*(
         u(c,i+1,j,k)- 3*u(c,i,  j,k)+3*u(c,i-1,j,k)- u(c,i-2,j,k) 
      -(um(c,i+1,j,k)-3*um(c,i, j,k)+3*um(c,i-1,j,k)-um(c,i-2,j,k)) )
      - (rho(i-1,j,k)*dcx(i-1)*jac(i-1,j,k)+rho(i-2,j,k)*dcx(i-2)*jac(i-2,j,k))*(
         u(c,i, j,k)- 3*u(c,i-1,j,k)+ 3*u(c,i-2,j,k)- u(c,i-3,j,k) 
      -(um(c,i, j,k)-3*um(c,i-1,j,k)+3*um(c,i-2,j,k)-um(c,i-3,j,k)) )
		       ) +  stry(j)*cox(i)*( 
// y-differences
     (rho(i,j+2,k)*dcy(j+2)*jac(i,j+2,k)+rho(i,j+1,k)*dcy(j+1)*jac(i,j+1,k))*(
         u(c,i,j+3,k) -3*u(c,i,j+2,k)+ 3*u(c,i,j+1,k)- u(c,i,  j,k)
      -(um(c,i,j+3,k)-3*um(c,i,j+2,k)+3*um(c,i,j+1,k)-um(c,i,  j,k)) )
     -3*(rho(i,j+1,k)*dcy(j+1)*jac(i,j+1,k)+rho(i,j,k)*dcy(j)*jac(i,j,k))*(
         u(c,i,j+2,k) -3*u(c,i,j+1,k)+ 3*u(c,i,  j,k)- u(c,i,j-1,k) 
      -(um(c,i,j+2,k)-3*um(c,i,j+1,k)+3*um(c,i,  j,k)-um(c,i,j-1,k)) )
     +3*(rho(i,j,k)*dcy(j)*jac(i,j,k)+rho(i,j-1,k)*dcy(j-1)*jac(i,j-1,k))*(
         u(c,i,j+1,k)- 3*u(c,i, j,k)+ 3*u(c,i,j-1,k)- u(c,i,j-2,k) 
      -(um(c,i,j+1,k)-3*um(c,i, j,k)+3*um(c,i,j-1,k)-um(c,i,j-2,k)) )
     - (rho(i,j-1,k)*dcy(j-1)*jac(i,j-1,k)+rho(i,j-2,k)*dcy(j-2)*jac(i,j-2,k))*(
         u(c,i, j,k)- 3*u(c,i,j-1,k)+  3*u(c,i,j-2,k)- u(c,i,j-3,k) 
      -(um(c,i, j,k)-3*um(c,i,j-1,k)+ 3*um(c,i,j-2,k)-um(c,i,j-3,k)) )
					     )  );
	       }
      }
#undef rho
#undef up
#undef u
#undef um
#undef jac
#undef strx
#undef dcx
#undef cox
#undef stry
#undef dcy
#undef coy
}
//-----------------------------------------------------------------------
__global__ void addsgd6c_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy, 
		      float_sw4* a_strx, float_sw4* a_stry,
		      float_sw4* a_jac, float_sw4* a_cox,  float_sw4* a_coy,  
		      float_sw4 beta, int ghost_points )
{
#define rho(i,j,k) a_rho[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define up(c,i,j,k) a_up[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define u(c,i,j,k)   a_u[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define um(c,i,j,k) a_um[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define jac(i,j,k) a_jac[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define strx(i) a_strx[(i-ifirst)]
#define dcx(i) a_dcx[(i-ifirst)]
#define cox(i) a_cox[(i-ifirst)]
#define stry(j) a_stry[(j-jfirst)]
#define dcy(j) a_dcy[(j-jfirst)]
#define coy(j) a_coy[(j-jfirst)]

   if( beta == 0. )
      return;

   const size_t ni = ilast-ifirst+1;
   const size_t nij = ni*(jlast-jfirst+1);
   const size_t nijk = nij*(klast-kfirst+1);
   int i = ifirst + ghost_points + 1 + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + 1 + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + 1 + threadIdx.z + blockIdx.z*blockDim.z;
   if( (i <= ilast-3) && (j <= jlast-3) && (k <= klast-3) )
   {
   float_sw4 birho=0.5*beta/(rho(i,j,k)*jac(i,j,k));
	       for( int c=0 ; c < 3 ; c++ )
	       {
		 up(c,i,j,k) += birho*( 
       strx(i)*coy(j)*(
// x-differences
      (rho(i+2,j,k)*dcx(i+2)*jac(i+2,j,k)+rho(i+1,j,k)*dcx(i+1)*jac(i+1,j,k))*(
         u(c,i+3,j,k) -3*u(c,i+2,j,k)+ 3*u(c,i+1,j,k)- u(c,i, j,k) 
      -(um(c,i+3,j,k)-3*um(c,i+2,j,k)+3*um(c,i+1,j,k)-um(c,i, j,k)) )
      -3*(rho(i+1,j,k)*dcx(i+1)*jac(i+1,j,k)+rho(i,j,k)*dcx(i)*jac(i,j,k))*(
         u(c,i+2,j,k)- 3*u(c,i+1,j,k)+ 3*u(c,i, j,k)- u(c,i-1,j,k)
      -(um(c,i+2,j,k)-3*um(c,i+1,j,k)+3*um(c,i, j,k)-um(c,i-1,j,k)) )
      +3*(rho(i,j,k)*dcx(i)*jac(i,j,k)+rho(i-1,j,k)*dcx(i-1)*jac(i-1,j,k))*(
         u(c,i+1,j,k)- 3*u(c,i,  j,k)+3*u(c,i-1,j,k)- u(c,i-2,j,k) 
      -(um(c,i+1,j,k)-3*um(c,i, j,k)+3*um(c,i-1,j,k)-um(c,i-2,j,k)) )
      - (rho(i-1,j,k)*dcx(i-1)*jac(i-1,j,k)+rho(i-2,j,k)*dcx(i-2)*jac(i-2,j,k))*(
         u(c,i, j,k)- 3*u(c,i-1,j,k)+ 3*u(c,i-2,j,k)- u(c,i-3,j,k) 
      -(um(c,i, j,k)-3*um(c,i-1,j,k)+3*um(c,i-2,j,k)-um(c,i-3,j,k)) )
		       ) +  stry(j)*cox(i)*( 
// y-differences
     (rho(i,j+2,k)*dcy(j+2)*jac(i,j+2,k)+rho(i,j+1,k)*dcy(j+1)*jac(i,j+1,k))*(
         u(c,i,j+3,k) -3*u(c,i,j+2,k)+ 3*u(c,i,j+1,k)- u(c,i,  j,k)
      -(um(c,i,j+3,k)-3*um(c,i,j+2,k)+3*um(c,i,j+1,k)-um(c,i,  j,k)) )
     -3*(rho(i,j+1,k)*dcy(j+1)*jac(i,j+1,k)+rho(i,j,k)*dcy(j)*jac(i,j,k))*(
         u(c,i,j+2,k) -3*u(c,i,j+1,k)+ 3*u(c,i,  j,k)- u(c,i,j-1,k) 
      -(um(c,i,j+2,k)-3*um(c,i,j+1,k)+3*um(c,i,  j,k)-um(c,i,j-1,k)) )
     +3*(rho(i,j,k)*dcy(j)*jac(i,j,k)+rho(i,j-1,k)*dcy(j-1)*jac(i,j-1,k))*(
         u(c,i,j+1,k)- 3*u(c,i, j,k)+ 3*u(c,i,j-1,k)- u(c,i,j-2,k) 
      -(um(c,i,j+1,k)-3*um(c,i, j,k)+3*um(c,i,j-1,k)-um(c,i,j-2,k)) )
     - (rho(i,j-1,k)*dcy(j-1)*jac(i,j-1,k)+rho(i,j-2,k)*dcy(j-2)*jac(i,j-2,k))*(
         u(c,i, j,k)- 3*u(c,i,j-1,k)+  3*u(c,i,j-2,k)- u(c,i,j-3,k) 
      -(um(c,i, j,k)-3*um(c,i,j-1,k)+ 3*um(c,i,j-2,k)-um(c,i,j-3,k)) )
					     )  );
	       }
	    }
#undef rho
#undef up
#undef u
#undef um
#undef jac
#undef strx
#undef dcx
#undef cox
#undef stry
#undef dcy
#undef coy
}

//-----------------------------------------------------------------------
__global__ void addsgd4c_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, 
                      float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy, 
                      float_sw4* a_strx, float_sw4* a_stry, 
		      float_sw4* a_jac, float_sw4* a_cox,  float_sw4* a_coy, 
                      float_sw4 beta, int ghost_points )
{
#define rho(i,j,k) a_rho[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define up(c,i,j,k) a_up[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define u(c,i,j,k) a_u[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define um(c,i,j,k) a_um[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define jac(i,j,k) a_jac[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define strx(i) a_strx[(i-ifirst)]
#define dcx(i) a_dcx[(i-ifirst)]
#define cox(i) a_cox[(i-ifirst)]
#define stry(j) a_stry[(j-jfirst)]
#define dcy(j) a_dcy[(j-jfirst)]
#define coy(j) a_coy[(j-jfirst)]
   if( beta == 0. )
      return;

   const size_t ni = ilast-ifirst+1;
   const size_t nij = ni*(jlast-jfirst+1);
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
      float_sw4 irhoj=beta/(rho(i,j,k)*jac(i,j,k));
      for( int c=0 ; c < 3 ; c++ )
      {
         up(c,i,j,k) -= irhoj*( 
         // x-differences
          strx(i)*coy(j)*(
          rho(i+1,j,k)*dcx(i+1)*jac(i+1,j,k)*
          ( u(c,i+2,j,k) -2*u(c,i+1,j,k)+ u(c,i,  j,k))
          -2*rho(i,j,k)*dcx(i)*jac(i,j,k)*
          ( u(c,i+1,j,k) -2*u(c,i,  j,k)+ u(c,i-1,j,k))
          +rho(i-1,j,k)*dcx(i-1)*jac(i-1,j,k)*
          ( u(c,i,  j,k) -2*u(c,i-1,j,k)+ u(c,i-2,j,k)) 
          -rho(i+1,j,k)*dcx(i+1)*jac(i+1,j,k)*
          (um(c,i+2,j,k)-2*um(c,i+1,j,k)+um(c,i,  j,k)) 
          +2*rho(i,j,k)*dcx(i)*jac(i,j,k)*
          (um(c,i+1,j,k)-2*um(c,i,  j,k)+um(c,i-1,j,k)) 
          -rho(i-1,j,k)*dcx(i-1)*jac(i-1,j,k)*
          (um(c,i,  j,k)-2*um(c,i-1,j,k)+um(c,i-2,j,k)) ) +
// y-differences
          stry(j)*cox(i)*(
           +rho(i,j+1,k)*dcy(j+1)*jac(i,j+1,k)*
          ( u(c,i,j+2,k) -2*u(c,i,j+1,k)+ u(c,i,j,  k)) 
           -2*rho(i,j,k)*dcy(j)*jac(i,j,k)*
          ( u(c,i,j+1,k) -2*u(c,i,j,  k)+ u(c,i,j-1,k))
           +rho(i,j-1,k)*dcy(j-1)*jac(i,j-1,k)*
          ( u(c,i,j,  k) -2*u(c,i,j-1,k)+ u(c,i,j-2,k)) 
           -rho(i,j+1,k)*dcy(j+1)*jac(i,j+1,k)*
          (um(c,i,j+2,k)-2*um(c,i,j+1,k)+um(c,i,j,  k)) 
           +2*rho(i,j,k)*dcy(j)*jac(i,j,k)*
          (um(c,i,j+1,k)-2*um(c,i,j,  k)+um(c,i,j-1,k)) 
           -rho(i,j-1,k)*dcy(j-1)*jac(i,j-1,k)*
           (um(c,i,j,  k)-2*um(c,i,j-1,k)+um(c,i,j-2,k)) ) );
	}
    } 
#undef rho
#undef up
#undef u
#undef um
#undef jac
#undef strx
#undef dcx
#undef cox
#undef stry
#undef dcy
#undef coy
}
//-----------------------------------------------------------------------
__global__ void addsgd4c_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, 
                      float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy, 
                      float_sw4* a_strx, float_sw4* a_stry, 
		      float_sw4* a_jac, float_sw4* a_cox,  float_sw4* a_coy, 
                      float_sw4 beta, int ghost_points )
{
#define rho(i,j,k) a_rho[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define up(c,i,j,k) a_up[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define u(c,i,j,k)   a_u[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define um(c,i,j,k) a_um[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define jac(i,j,k) a_jac[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define strx(i) a_strx[(i-ifirst)]
#define dcx(i) a_dcx[(i-ifirst)]
#define cox(i) a_cox[(i-ifirst)]
#define stry(j) a_stry[(j-jfirst)]
#define dcy(j) a_dcy[(j-jfirst)]
#define coy(j) a_coy[(j-jfirst)]
   if( beta == 0. )
      return;

   const size_t ni = ilast-ifirst+1;
   const size_t nij = ni*(jlast-jfirst+1);
   const size_t nijk = nij*(klast-kfirst+1);
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
      float_sw4 irhoj=beta/(rho(i,j,k)*jac(i,j,k));
      for( int c=0 ; c < 3 ; c++ )
      {
         up(c,i,j,k) -= irhoj*( 
         // x-differences
          strx(i)*coy(j)*(
          rho(i+1,j,k)*dcx(i+1)*jac(i+1,j,k)*
          ( u(c,i+2,j,k) -2*u(c,i+1,j,k)+ u(c,i,  j,k))
          -2*rho(i,j,k)*dcx(i)*jac(i,j,k)*
          ( u(c,i+1,j,k) -2*u(c,i,  j,k)+ u(c,i-1,j,k))
          +rho(i-1,j,k)*dcx(i-1)*jac(i-1,j,k)*
          ( u(c,i,  j,k) -2*u(c,i-1,j,k)+ u(c,i-2,j,k)) 
          -rho(i+1,j,k)*dcx(i+1)*jac(i+1,j,k)*
          (um(c,i+2,j,k)-2*um(c,i+1,j,k)+um(c,i,  j,k)) 
          +2*rho(i,j,k)*dcx(i)*jac(i,j,k)*
          (um(c,i+1,j,k)-2*um(c,i,  j,k)+um(c,i-1,j,k)) 
          -rho(i-1,j,k)*dcx(i-1)*jac(i-1,j,k)*
          (um(c,i,  j,k)-2*um(c,i-1,j,k)+um(c,i-2,j,k)) ) +
// y-differences
          stry(j)*cox(i)*(
           +rho(i,j+1,k)*dcy(j+1)*jac(i,j+1,k)*
          ( u(c,i,j+2,k) -2*u(c,i,j+1,k)+ u(c,i,j,  k)) 
           -2*rho(i,j,k)*dcy(j)*jac(i,j,k)*
          ( u(c,i,j+1,k) -2*u(c,i,j,  k)+ u(c,i,j-1,k))
           +rho(i,j-1,k)*dcy(j-1)*jac(i,j-1,k)*
          ( u(c,i,j,  k) -2*u(c,i,j-1,k)+ u(c,i,j-2,k)) 
           -rho(i,j+1,k)*dcy(j+1)*jac(i,j+1,k)*
          (um(c,i,j+2,k)-2*um(c,i,j+1,k)+um(c,i,j,  k)) 
           +2*rho(i,j,k)*dcy(j)*jac(i,j,k)*
          (um(c,i,j+1,k)-2*um(c,i,j,  k)+um(c,i,j-1,k)) 
           -rho(i,j-1,k)*dcy(j-1)*jac(i,j-1,k)*
           (um(c,i,j,  k)-2*um(c,i,j-1,k)+um(c,i,j-2,k)) ) );
	}
    } 
#undef rho
#undef up
#undef u
#undef um
#undef jac
#undef strx
#undef dcx
#undef cox
#undef stry
#undef dcy
#undef coy
}

//-----------------------------------------------------------------------
__global__ void rhs4center_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
				float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
				int ghost_points )
{
 // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+i+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+i+ni*(j)+nij*(k)]
#define u(c,i,j,k)   a_u[base3+c+3*(i)+nic*(j)+nijc*(k)]   
#define lu(c,i,j,k) a_lu[base3+c+3*(i)+nic*(j)+nijc*(k)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define strz(k) a_strz[k-kfirst0]
   const float_sw4 a1   = 0;
   const float_sw4 i6   = 1.0/6;
   //   const float_sw4 i12  = 1.0/12;
   const float_sw4 i144 = 1.0/144;
   const float_sw4 tf   = 0.75;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = 3*base-1;
   const int nic  = 3*ni;
   const int nijc = 3*nij;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof;
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;

   cof = 1.0/(h*h);
   if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
   mux1 = mu(i-1,j,k)*strx(i-1)-
      tf*(mu(i,j,k)*strx(i)+mu(i-2,j,k)*strx(i-2));
   mux2 = mu(i-2,j,k)*strx(i-2)+mu(i+1,j,k)*strx(i+1)+
      3*(mu(i,j,k)*strx(i)+mu(i-1,j,k)*strx(i-1));
   mux3 = mu(i-1,j,k)*strx(i-1)+mu(i+2,j,k)*strx(i+2)+
      3*(mu(i+1,j,k)*strx(i+1)+mu(i,j,k)*strx(i));
   mux4 = mu(i+1,j,k)*strx(i+1)-
      tf*(mu(i,j,k)*strx(i)+mu(i+2,j,k)*strx(i+2));

   muy1 = mu(i,j-1,k)*stry(j-1)-
      tf*(mu(i,j,k)*stry(j)+mu(i,j-2,k)*stry(j-2));
   muy2 = mu(i,j-2,k)*stry(j-2)+mu(i,j+1,k)*stry(j+1)+
      3*(mu(i,j,k)*stry(j)+mu(i,j-1,k)*stry(j-1));
   muy3 = mu(i,j-1,k)*stry(j-1)+mu(i,j+2,k)*stry(j+2)+
      3*(mu(i,j+1,k)*stry(j+1)+mu(i,j,k)*stry(j));
   muy4 = mu(i,j+1,k)*stry(j+1)-
      tf*(mu(i,j,k)*stry(j)+mu(i,j+2,k)*stry(j+2));

   muz1 = mu(i,j,k-1)*strz(k-1)-
      tf*(mu(i,j,k)*strz(k)+mu(i,j,k-2)*strz(k-2));
   muz2 = mu(i,j,k-2)*strz(k-2)+mu(i,j,k+1)*strz(k+1)+
      3*(mu(i,j,k)*strz(k)+mu(i,j,k-1)*strz(k-1));
   muz3 = mu(i,j,k-1)*strz(k-1)+mu(i,j,k+2)*strz(k+2)+
      3*(mu(i,j,k+1)*strz(k+1)+mu(i,j,k)*strz(k));
   muz4 = mu(i,j,k+1)*strz(k+1)-
      tf*(mu(i,j,k)*strz(k)+mu(i,j,k+2)*strz(k+2));
/* xx, yy, and zz derivatives:*/
/* 75 ops */
   r1 = i6*( strx(i)*( (2*mux1+la(i-1,j,k)*strx(i-1)-
               tf*(la(i,j,k)*strx(i)+la(i-2,j,k)*strx(i-2)))*
                              (u(1,i-2,j,k)-u(1,i,j,k))+
           (2*mux2+la(i-2,j,k)*strx(i-2)+la(i+1,j,k)*strx(i+1)+
                3*(la(i,j,k)*strx(i)+la(i-1,j,k)*strx(i-1)))*
                              (u(1,i-1,j,k)-u(1,i,j,k))+ 
           (2*mux3+la(i-1,j,k)*strx(i-1)+la(i+2,j,k)*strx(i+2)+
                3*(la(i+1,j,k)*strx(i+1)+la(i,j,k)*strx(i)))*
                              (u(1,i+1,j,k)-u(1,i,j,k))+
                (2*mux4+ la(i+1,j,k)*strx(i+1)-
               tf*(la(i,j,k)*strx(i)+la(i+2,j,k)*strx(i+2)))*
                (u(1,i+2,j,k)-u(1,i,j,k)) ) + stry(j)*(
                     muy1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                     muy2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                     muy3*(u(1,i,j+1,k)-u(1,i,j,k)) +
                     muy4*(u(1,i,j+2,k)-u(1,i,j,k)) ) + strz(k)*(
                     muz1*(u(1,i,j,k-2)-u(1,i,j,k)) + 
                     muz2*(u(1,i,j,k-1)-u(1,i,j,k)) + 
                     muz3*(u(1,i,j,k+1)-u(1,i,j,k)) +
                     muz4*(u(1,i,j,k+2)-u(1,i,j,k)) ) );

/* 75 ops */
   r2 = i6*( strx(i)*(mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                      mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                      mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                      mux4*(u(2,i+2,j,k)-u(2,i,j,k)) ) + stry(j)*(
                  (2*muy1+la(i,j-1,k)*stry(j-1)-
                      tf*(la(i,j,k)*stry(j)+la(i,j-2,k)*stry(j-2)))*
                          (u(2,i,j-2,k)-u(2,i,j,k))+
           (2*muy2+la(i,j-2,k)*stry(j-2)+la(i,j+1,k)*stry(j+1)+
                     3*(la(i,j,k)*stry(j)+la(i,j-1,k)*stry(j-1)))*
                          (u(2,i,j-1,k)-u(2,i,j,k))+ 
           (2*muy3+la(i,j-1,k)*stry(j-1)+la(i,j+2,k)*stry(j+2)+
                     3*(la(i,j+1,k)*stry(j+1)+la(i,j,k)*stry(j)))*
                          (u(2,i,j+1,k)-u(2,i,j,k))+
                  (2*muy4+la(i,j+1,k)*stry(j+1)-
                    tf*(la(i,j,k)*stry(j)+la(i,j+2,k)*stry(j+2)))*
                          (u(2,i,j+2,k)-u(2,i,j,k)) ) + strz(k)*(
                     muz1*(u(2,i,j,k-2)-u(2,i,j,k)) + 
                     muz2*(u(2,i,j,k-1)-u(2,i,j,k)) + 
                     muz3*(u(2,i,j,k+1)-u(2,i,j,k)) +
                     muz4*(u(2,i,j,k+2)-u(2,i,j,k)) ) );

/* 75 ops */
   r3 = i6*( strx(i)*(mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                      mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                      mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                      mux4*(u(3,i+2,j,k)-u(3,i,j,k))  ) + stry(j)*(
                     muy1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                     muy2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                     muy3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                     muy4*(u(3,i,j+2,k)-u(3,i,j,k)) ) + strz(k)*(
                  (2*muz1+la(i,j,k-1)*strz(k-1)-
                      tf*(la(i,j,k)*strz(k)+la(i,j,k-2)*strz(k-2)))*
                          (u(3,i,j,k-2)-u(3,i,j,k))+
           (2*muz2+la(i,j,k-2)*strz(k-2)+la(i,j,k+1)*strz(k+1)+
                      3*(la(i,j,k)*strz(k)+la(i,j,k-1)*strz(k-1)))*
                          (u(3,i,j,k-1)-u(3,i,j,k))+ 
           (2*muz3+la(i,j,k-1)*strz(k-1)+la(i,j,k+2)*strz(k+2)+
                      3*(la(i,j,k+1)*strz(k+1)+la(i,j,k)*strz(k)))*
                          (u(3,i,j,k+1)-u(3,i,j,k))+
                  (2*muz4+la(i,j,k+1)*strz(k+1)-
                    tf*(la(i,j,k)*strz(k)+la(i,j,k+2)*strz(k+2)))*
		  (u(3,i,j,k+2)-u(3,i,j,k)) ) );


/* Mixed derivatives: */
/* 29ops /mixed derivative */
/* 116 ops for r1 */
/*   (la*v_y)_x */
   r1 = r1 + strx(i)*stry(j)*
                 i144*( la(i-2,j,k)*(u(2,i-2,j-2,k)-u(2,i-2,j+2,k)+
                             8*(-u(2,i-2,j-1,k)+u(2,i-2,j+1,k))) - 8*(
                        la(i-1,j,k)*(u(2,i-1,j-2,k)-u(2,i-1,j+2,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i-1,j+1,k))) )+8*(
                        la(i+1,j,k)*(u(2,i+1,j-2,k)-u(2,i+1,j+2,k)+
                             8*(-u(2,i+1,j-1,k)+u(2,i+1,j+1,k))) ) - (
                        la(i+2,j,k)*(u(2,i+2,j-2,k)-u(2,i+2,j+2,k)+
                             8*(-u(2,i+2,j-1,k)+u(2,i+2,j+1,k))) )) 
/*   (la*w_z)_x */
               + strx(i)*strz(k)*       
                 i144*( la(i-2,j,k)*(u(3,i-2,j,k-2)-u(3,i-2,j,k+2)+
                             8*(-u(3,i-2,j,k-1)+u(3,i-2,j,k+1))) - 8*(
                        la(i-1,j,k)*(u(3,i-1,j,k-2)-u(3,i-1,j,k+2)+
                             8*(-u(3,i-1,j,k-1)+u(3,i-1,j,k+1))) )+8*(
                        la(i+1,j,k)*(u(3,i+1,j,k-2)-u(3,i+1,j,k+2)+
                             8*(-u(3,i+1,j,k-1)+u(3,i+1,j,k+1))) ) - (
                        la(i+2,j,k)*(u(3,i+2,j,k-2)-u(3,i+2,j,k+2)+
                             8*(-u(3,i+2,j,k-1)+u(3,i+2,j,k+1))) )) 
/*   (mu*v_x)_y */
               + strx(i)*stry(j)*       
                 i144*( mu(i,j-2,k)*(u(2,i-2,j-2,k)-u(2,i+2,j-2,k)+
                             8*(-u(2,i-1,j-2,k)+u(2,i+1,j-2,k))) - 8*(
                        mu(i,j-1,k)*(u(2,i-2,j-1,k)-u(2,i+2,j-1,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i+1,j-1,k))) )+8*(
                        mu(i,j+1,k)*(u(2,i-2,j+1,k)-u(2,i+2,j+1,k)+
                             8*(-u(2,i-1,j+1,k)+u(2,i+1,j+1,k))) ) - (
                        mu(i,j+2,k)*(u(2,i-2,j+2,k)-u(2,i+2,j+2,k)+
                             8*(-u(2,i-1,j+2,k)+u(2,i+1,j+2,k))) )) 
/*   (mu*w_x)_z */
               + strx(i)*strz(k)*       
                 i144*( mu(i,j,k-2)*(u(3,i-2,j,k-2)-u(3,i+2,j,k-2)+
                             8*(-u(3,i-1,j,k-2)+u(3,i+1,j,k-2))) - 8*(
                        mu(i,j,k-1)*(u(3,i-2,j,k-1)-u(3,i+2,j,k-1)+
                             8*(-u(3,i-1,j,k-1)+u(3,i+1,j,k-1))) )+8*(
                        mu(i,j,k+1)*(u(3,i-2,j,k+1)-u(3,i+2,j,k+1)+
                             8*(-u(3,i-1,j,k+1)+u(3,i+1,j,k+1))) ) - (
                        mu(i,j,k+2)*(u(3,i-2,j,k+2)-u(3,i+2,j,k+2)+
				     8*(-u(3,i-1,j,k+2)+u(3,i+1,j,k+2))) )) ;

/* 116 ops for r2 */
/*   (mu*u_y)_x */
   r2 = r2 + strx(i)*stry(j)*
                 i144*( mu(i-2,j,k)*(u(1,i-2,j-2,k)-u(1,i-2,j+2,k)+
                             8*(-u(1,i-2,j-1,k)+u(1,i-2,j+1,k))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j-2,k)-u(1,i-1,j+2,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i-1,j+1,k))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j-2,k)-u(1,i+1,j+2,k)+
                             8*(-u(1,i+1,j-1,k)+u(1,i+1,j+1,k))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j-2,k)-u(1,i+2,j+2,k)+
                             8*(-u(1,i+2,j-1,k)+u(1,i+2,j+1,k))) )) 
/* (la*u_x)_y */
              + strx(i)*stry(j)*
                 i144*( la(i,j-2,k)*(u(1,i-2,j-2,k)-u(1,i+2,j-2,k)+
                             8*(-u(1,i-1,j-2,k)+u(1,i+1,j-2,k))) - 8*(
                        la(i,j-1,k)*(u(1,i-2,j-1,k)-u(1,i+2,j-1,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i+1,j-1,k))) )+8*(
                        la(i,j+1,k)*(u(1,i-2,j+1,k)-u(1,i+2,j+1,k)+
                             8*(-u(1,i-1,j+1,k)+u(1,i+1,j+1,k))) ) - (
                        la(i,j+2,k)*(u(1,i-2,j+2,k)-u(1,i+2,j+2,k)+
                             8*(-u(1,i-1,j+2,k)+u(1,i+1,j+2,k))) )) 
/* (la*w_z)_y */
               + stry(j)*strz(k)*
                 i144*( la(i,j-2,k)*(u(3,i,j-2,k-2)-u(3,i,j-2,k+2)+
                             8*(-u(3,i,j-2,k-1)+u(3,i,j-2,k+1))) - 8*(
                        la(i,j-1,k)*(u(3,i,j-1,k-2)-u(3,i,j-1,k+2)+
                             8*(-u(3,i,j-1,k-1)+u(3,i,j-1,k+1))) )+8*(
                        la(i,j+1,k)*(u(3,i,j+1,k-2)-u(3,i,j+1,k+2)+
                             8*(-u(3,i,j+1,k-1)+u(3,i,j+1,k+1))) ) - (
                        la(i,j+2,k)*(u(3,i,j+2,k-2)-u(3,i,j+2,k+2)+
                             8*(-u(3,i,j+2,k-1)+u(3,i,j+2,k+1))) ))
/* (mu*w_y)_z */
               + stry(j)*strz(k)*
                 i144*( mu(i,j,k-2)*(u(3,i,j-2,k-2)-u(3,i,j+2,k-2)+
                             8*(-u(3,i,j-1,k-2)+u(3,i,j+1,k-2))) - 8*(
                        mu(i,j,k-1)*(u(3,i,j-2,k-1)-u(3,i,j+2,k-1)+
                             8*(-u(3,i,j-1,k-1)+u(3,i,j+1,k-1))) )+8*(
                        mu(i,j,k+1)*(u(3,i,j-2,k+1)-u(3,i,j+2,k+1)+
                             8*(-u(3,i,j-1,k+1)+u(3,i,j+1,k+1))) ) - (
                        mu(i,j,k+2)*(u(3,i,j-2,k+2)-u(3,i,j+2,k+2)+
				     8*(-u(3,i,j-1,k+2)+u(3,i,j+1,k+2))) )) ;
/* 116 ops for r3 */
/*  (mu*u_z)_x */
   r3 = r3 + strx(i)*strz(k)*
                 i144*( mu(i-2,j,k)*(u(1,i-2,j,k-2)-u(1,i-2,j,k+2)+
                             8*(-u(1,i-2,j,k-1)+u(1,i-2,j,k+1))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j,k-2)-u(1,i-1,j,k+2)+
                             8*(-u(1,i-1,j,k-1)+u(1,i-1,j,k+1))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j,k-2)-u(1,i+1,j,k+2)+
                             8*(-u(1,i+1,j,k-1)+u(1,i+1,j,k+1))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j,k-2)-u(1,i+2,j,k+2)+
                             8*(-u(1,i+2,j,k-1)+u(1,i+2,j,k+1))) )) 
/* (mu*v_z)_y */
              + stry(j)*strz(k)*
                 i144*( mu(i,j-2,k)*(u(2,i,j-2,k-2)-u(2,i,j-2,k+2)+
                             8*(-u(2,i,j-2,k-1)+u(2,i,j-2,k+1))) - 8*(
                        mu(i,j-1,k)*(u(2,i,j-1,k-2)-u(2,i,j-1,k+2)+
                             8*(-u(2,i,j-1,k-1)+u(2,i,j-1,k+1))) )+8*(
                        mu(i,j+1,k)*(u(2,i,j+1,k-2)-u(2,i,j+1,k+2)+
                             8*(-u(2,i,j+1,k-1)+u(2,i,j+1,k+1))) ) - (
                        mu(i,j+2,k)*(u(2,i,j+2,k-2)-u(2,i,j+2,k+2)+
                             8*(-u(2,i,j+2,k-1)+u(2,i,j+2,k+1))) ))
/*   (la*u_x)_z */
              + strx(i)*strz(k)*
                 i144*( la(i,j,k-2)*(u(1,i-2,j,k-2)-u(1,i+2,j,k-2)+
                             8*(-u(1,i-1,j,k-2)+u(1,i+1,j,k-2))) - 8*(
                        la(i,j,k-1)*(u(1,i-2,j,k-1)-u(1,i+2,j,k-1)+
                             8*(-u(1,i-1,j,k-1)+u(1,i+1,j,k-1))) )+8*(
                        la(i,j,k+1)*(u(1,i-2,j,k+1)-u(1,i+2,j,k+1)+
                             8*(-u(1,i-1,j,k+1)+u(1,i+1,j,k+1))) ) - (
                        la(i,j,k+2)*(u(1,i-2,j,k+2)-u(1,i+2,j,k+2)+
                             8*(-u(1,i-1,j,k+2)+u(1,i+1,j,k+2))) )) 
/* (la*v_y)_z */
              + stry(j)*strz(k)*
                 i144*( la(i,j,k-2)*(u(2,i,j-2,k-2)-u(2,i,j+2,k-2)+
                             8*(-u(2,i,j-1,k-2)+u(2,i,j+1,k-2))) - 8*(
                        la(i,j,k-1)*(u(2,i,j-2,k-1)-u(2,i,j+2,k-1)+
                             8*(-u(2,i,j-1,k-1)+u(2,i,j+1,k-1))) )+8*(
                        la(i,j,k+1)*(u(2,i,j-2,k+1)-u(2,i,j+2,k+1)+
                             8*(-u(2,i,j-1,k+1)+u(2,i,j+1,k+1))) ) - (
                        la(i,j,k+2)*(u(2,i,j-2,k+2)-u(2,i,j+2,k+2)+
				     8*(-u(2,i,j-1,k+2)+u(2,i,j+1,k+2))) )) ;

/* 9 ops */
   lu(1,i,j,k) = a1*lu(1,i,j,k) + cof*r1;
   lu(2,i,j,k) = a1*lu(2,i,j,k) + cof*r2;
   lu(3,i,j,k) = a1*lu(3,i,j,k) + cof*r3;
#undef mu
#undef la
#undef u
#undef lu
#undef strx
#undef stry
#undef strz
}
}
//-----------------------------------------------------------------------
__global__ void rhs4upper_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
       //		       float_sw4* dev_acof, float_sw4* dev_bope, float_sw4* dev_ghcof,
			       float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
			       float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			       int ghost_points )
{
   // For 1 <= k <= 6 if free surface boundary.
#define mu(i,j,k)     a_mu[base+i+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+i+ni*(j)+nij*(k)]
#define u(c,i,j,k)   a_u[base3+c+3*(i)+nic*(j)+nijc*(k)]   
#define lu(c,i,j,k) a_lu[base3+c+3*(i)+nic*(j)+nijc*(k)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
   //#define strz(k) a_strz[k-kfirst0]
#define acof(i,j,k) dev_acof[(i-1)+6*(j-1)+48*(k-1)]
#define bope(i,j) dev_bope[(i-1)+6*(j-1)]
#define ghcof(i) dev_ghcof[(i-1)]
   const float_sw4 a1   = 0;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 i12  = 1.0/12;
   const float_sw4 i144 = 1.0/144;
   const float_sw4 tf   = 0.75;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = 3*base-1;
   const int nic  = 3*ni;
   const int nijc = 3*nij;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   //   const int kfirst0 = kfirst;

   int q, m;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4;//, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof, mucof, mu1zz, mu2zz, mu3zz;
   float_sw4 lap2mu, u3zip2, u3zip1, u3zim1, u3zim2, lau3zx, mu3xz, u3zjp2, u3zjp1, u3zjm1, u3zjm2;
   float_sw4 lau3zy, mu3yz, mu1zx, mu2zy, u1zip2, u1zip1, u1zim1, u1zim2;
   float_sw4 u2zjp2, u2zjp1, u2zjm1, u2zjm2, lau1xz, lau2yz;

   //   return;

   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if( k < 1 || k > 6 )
      return;

   cof = 1.0/(h*h);
   if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
   mux1 = mu(i-1,j,k)*strx(i-1)-
		     tf*(mu(i,j,k)*strx(i)+mu(i-2,j,k)*strx(i-2));
   mux2 = mu(i-2,j,k)*strx(i-2)+mu(i+1,j,k)*strx(i+1)+
		     3*(mu(i,j,k)*strx(i)+mu(i-1,j,k)*strx(i-1));
   mux3 = mu(i-1,j,k)*strx(i-1)+mu(i+2,j,k)*strx(i+2)+
		     3*(mu(i+1,j,k)*strx(i+1)+mu(i,j,k)*strx(i));
   mux4 = mu(i+1,j,k)*strx(i+1)-
		     tf*(mu(i,j,k)*strx(i)+mu(i+2,j,k)*strx(i+2));

   muy1 = mu(i,j-1,k)*stry(j-1)-
		     tf*(mu(i,j,k)*stry(j)+mu(i,j-2,k)*stry(j-2));
   muy2 = mu(i,j-2,k)*stry(j-2)+mu(i,j+1,k)*stry(j+1)+
		     3*(mu(i,j,k)*stry(j)+mu(i,j-1,k)*stry(j-1));
   muy3 = mu(i,j-1,k)*stry(j-1)+mu(i,j+2,k)*stry(j+2)+
		     3*(mu(i,j+1,k)*stry(j+1)+mu(i,j,k)*stry(j));
   muy4 = mu(i,j+1,k)*stry(j+1)-
		     tf*(mu(i,j,k)*stry(j)+mu(i,j+2,k)*stry(j+2));

   r1 = i6*(strx(i)*((2*mux1+la(i-1,j,k)*strx(i-1)-
                       tf*(la(i,j,k)*strx(i)+la(i-2,j,k)*strx(i-2)))*
                              (u(1,i-2,j,k)-u(1,i,j,k))+
           (2*mux2+la(i-2,j,k)*strx(i-2)+la(i+1,j,k)*strx(i+1)+
                        3*(la(i,j,k)*strx(i)+la(i-1,j,k)*strx(i-1)))*
                              (u(1,i-1,j,k)-u(1,i,j,k))+ 
           (2*mux3+la(i-1,j,k)*strx(i-1)+la(i+2,j,k)*strx(i+2)+
                        3*(la(i+1,j,k)*strx(i+1)+la(i,j,k)*strx(i)))*
                              (u(1,i+1,j,k)-u(1,i,j,k))+
                (2*mux4+ la(i+1,j,k)*strx(i+1)-
                       tf*(la(i,j,k)*strx(i)+la(i+2,j,k)*strx(i+2)))*
                (u(1,i+2,j,k)-u(1,i,j,k)) ) + stry(j)*(
                   + muy1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                     muy2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                     muy3*(u(1,i,j+1,k)-u(1,i,j,k)) +
                     muy4*(u(1,i,j+2,k)-u(1,i,j,k)) ) );

		  /* (mu*uz)_z can not be centered */
		  /* second derivative (mu*u_z)_z at grid point z_k */
		  /* averaging the coefficient, */
		  /* leave out the z-supergrid stretching strz, since it will */
		  /* never be used together with the sbp-boundary operator */
   mu1zz = 0;
   mu2zz = 0;
   mu3zz = 0;
   for( q=1; q <= 8; q ++ )
   {
      lap2mu= 0;
      mucof = 0;
      for( m=1 ; m<=8; m++ )
      {
	 mucof  += acof(k,q,m)*mu(i,j,m);
	 lap2mu += acof(k,q,m)*(la(i,j,m)+2*mu(i,j,m));
      }
      mu1zz += mucof*u(1,i,j,q);
      mu2zz += mucof*u(2,i,j,q);
      mu3zz += lap2mu*u(3,i,j,q);
   }
		  /* ghost point only influences the first point (k=1) because ghcof(k)=0 for k>=2*/
   r1 = r1 + (mu1zz + ghcof(k)*mu(i,j,1)*u(1,i,j,0));

   r2 = i6*(strx(i)*(mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                      mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                      mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                      mux4*(u(2,i+2,j,k)-u(2,i,j,k)) )+ stry(j)*(
                  (2*muy1+la(i,j-1,k)*stry(j-1)-
                        tf*(la(i,j,k)*stry(j)+la(i,j-2,k)*stry(j-2)))*
                          (u(2,i,j-2,k)-u(2,i,j,k))+
           (2*muy2+la(i,j-2,k)*stry(j-2)+la(i,j+1,k)*stry(j+1)+
                        3*(la(i,j,k)*stry(j)+la(i,j-1,k)*stry(j-1)))*
                          (u(2,i,j-1,k)-u(2,i,j,k))+ 
           (2*muy3+la(i,j-1,k)*stry(j-1)+la(i,j+2,k)*stry(j+2)+
                        3*(la(i,j+1,k)*stry(j+1)+la(i,j,k)*stry(j)))*
                          (u(2,i,j+1,k)-u(2,i,j,k))+
                  (2*muy4+la(i,j+1,k)*stry(j+1)-
                       tf*(la(i,j,k)*stry(j)+la(i,j+2,k)*stry(j+2)))*
		  (u(2,i,j+2,k)-u(2,i,j,k)) ) );

 /* ghost point only influences the first point (k=1) because ghcof(k)=0 for k>=2 */
   r2 = r2 + (mu2zz + ghcof(k)*mu(i,j,1)*u(2,i,j,0));

   r3 = i6*(strx(i)*(mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                      mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                      mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                      mux4*(u(3,i+2,j,k)-u(3,i,j,k))  ) + stry(j)*(
                     muy1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                     muy2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                     muy3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                     muy4*(u(3,i,j+2,k)-u(3,i,j,k)) ) );
/* ghost point only influences the first point (k=1) because ghcof(k)=0 for k>=2 */
   r3 = r3 + (mu3zz + ghcof(k)*(la(i,j,1)+2*mu(i,j,1))*
			     u(3,i,j,0));

  /* cross-terms in first component of rhs */
/*   (la*v_y)_x */
   r1 = r1 + strx(i)*stry(j)*(
                 i144*( la(i-2,j,k)*(u(2,i-2,j-2,k)-u(2,i-2,j+2,k)+
                             8*(-u(2,i-2,j-1,k)+u(2,i-2,j+1,k))) - 8*(
                        la(i-1,j,k)*(u(2,i-1,j-2,k)-u(2,i-1,j+2,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i-1,j+1,k))) )+8*(
                        la(i+1,j,k)*(u(2,i+1,j-2,k)-u(2,i+1,j+2,k)+
                             8*(-u(2,i+1,j-1,k)+u(2,i+1,j+1,k))) ) - (
                        la(i+2,j,k)*(u(2,i+2,j-2,k)-u(2,i+2,j+2,k)+
				     8*(-u(2,i+2,j-1,k)+u(2,i+2,j+1,k))) ))
/*   (mu*v_x)_y */
               + i144*( mu(i,j-2,k)*(u(2,i-2,j-2,k)-u(2,i+2,j-2,k)+
                             8*(-u(2,i-1,j-2,k)+u(2,i+1,j-2,k))) - 8*(
                        mu(i,j-1,k)*(u(2,i-2,j-1,k)-u(2,i+2,j-1,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i+1,j-1,k))) )+8*(
                        mu(i,j+1,k)*(u(2,i-2,j+1,k)-u(2,i+2,j+1,k)+
                             8*(-u(2,i-1,j+1,k)+u(2,i+1,j+1,k))) ) - (
                        mu(i,j+2,k)*(u(2,i-2,j+2,k)-u(2,i+2,j+2,k)+
				     8*(-u(2,i-1,j+2,k)+u(2,i+1,j+2,k))) )) );
/*   (la*w_z)_x: NOT CENTERED */
   u3zip2=0.;
   u3zip1=0.;
   u3zim1=0.;
   u3zim2=0.;
   for( q=1 ; q <=8 ; q++ )
   {
      u3zip2 += bope(k,q)*u(3,i+2,j,q);
      u3zip1 += bope(k,q)*u(3,i+1,j,q);
      u3zim1 += bope(k,q)*u(3,i-1,j,q);
      u3zim2 += bope(k,q)*u(3,i-2,j,q);
   }
   lau3zx= i12*(-la(i+2,j,k)*u3zip2 + 8*la(i+1,j,k)*u3zip1
	               -8*la(i-1,j,k)*u3zim1 +   la(i-2,j,k)*u3zim2);
   r1 = r1 + strx(i)*lau3zx;
	    /*   (mu*w_x)_z: NOT CENTERED */
   mu3xz=0.;
   for( q=1 ; q<=8 ; q++ )
      mu3xz += bope(k,q)*( mu(i,j,q)*i12*
                  (-u(3,i+2,j,q) + 8*u(3,i+1,j,q)
                   -8*u(3,i-1,j,q) + u(3,i-2,j,q)) );
   r1 = r1 + strx(i)*mu3xz;

/* cross-terms in second component of rhs */
/*   (mu*u_y)_x */
   r2 = r2 + strx(i)*stry(j)*(
                 i144*( mu(i-2,j,k)*(u(1,i-2,j-2,k)-u(1,i-2,j+2,k)+
                             8*(-u(1,i-2,j-1,k)+u(1,i-2,j+1,k))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j-2,k)-u(1,i-1,j+2,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i-1,j+1,k))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j-2,k)-u(1,i+1,j+2,k)+
                             8*(-u(1,i+1,j-1,k)+u(1,i+1,j+1,k))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j-2,k)-u(1,i+2,j+2,k)+
				     8*(-u(1,i+2,j-1,k)+u(1,i+2,j+1,k))) )) 
/* (la*u_x)_y  */
               + i144*( la(i,j-2,k)*(u(1,i-2,j-2,k)-u(1,i+2,j-2,k)+
                             8*(-u(1,i-1,j-2,k)+u(1,i+1,j-2,k))) - 8*(
                        la(i,j-1,k)*(u(1,i-2,j-1,k)-u(1,i+2,j-1,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i+1,j-1,k))) )+8*(
                        la(i,j+1,k)*(u(1,i-2,j+1,k)-u(1,i+2,j+1,k)+
                             8*(-u(1,i-1,j+1,k)+u(1,i+1,j+1,k))) ) - (
                        la(i,j+2,k)*(u(1,i-2,j+2,k)-u(1,i+2,j+2,k)+
				     8*(-u(1,i-1,j+2,k)+u(1,i+1,j+2,k))) )) );
/* (la*w_z)_y : NOT CENTERED */
   u3zjp2=0.;
   u3zjp1=0.;
   u3zjm1=0.;
   u3zjm2=0.;
   for( q=1 ; q <=8 ; q++ )
   {
      u3zjp2 += bope(k,q)*u(3,i,j+2,q);
      u3zjp1 += bope(k,q)*u(3,i,j+1,q);
      u3zjm1 += bope(k,q)*u(3,i,j-1,q);
      u3zjm2 += bope(k,q)*u(3,i,j-2,q);
   }
   lau3zy= i12*(-la(i,j+2,k)*u3zjp2 + 8*la(i,j+1,k)*u3zjp1
			 -8*la(i,j-1,k)*u3zjm1 + la(i,j-2,k)*u3zjm2);

   r2 = r2 + stry(j)*lau3zy;

/* (mu*w_y)_z: NOT CENTERED */
   mu3yz=0.;
   for(  q=1 ; q <=8 ; q++ )
      mu3yz += bope(k,q)*( mu(i,j,q)*i12*
                  (-u(3,i,j+2,q) + 8*u(3,i,j+1,q)
                   -8*u(3,i,j-1,q) + u(3,i,j-2,q)) );

   r2 = r2 + stry(j)*mu3yz;

	    /* No centered cross terms in r3 */
	    /*  (mu*u_z)_x: NOT CENTERED */
   u1zip2=0.;
   u1zip1=0.;
   u1zim1=0.;
   u1zim2=0.;
   for(  q=1 ; q <=8 ; q++ )
   {
      u1zip2 += bope(k,q)*u(1,i+2,j,q);
      u1zip1 += bope(k,q)*u(1,i+1,j,q);
      u1zim1 += bope(k,q)*u(1,i-1,j,q);
      u1zim2 += bope(k,q)*u(1,i-2,j,q);
   }
   mu1zx= i12*(-mu(i+2,j,k)*u1zip2 + 8*mu(i+1,j,k)*u1zip1
                   -8*mu(i-1,j,k)*u1zim1 + mu(i-2,j,k)*u1zim2);
   r3 = r3 + strx(i)*mu1zx;

	    /* (mu*v_z)_y: NOT CENTERED */
   u2zjp2=0.;
   u2zjp1=0.;
   u2zjm1=0.;
   u2zjm2=0.;
   for(  q=1 ; q <=8 ; q++ )
   {
      u2zjp2 += bope(k,q)*u(2,i,j+2,q);
      u2zjp1 += bope(k,q)*u(2,i,j+1,q);
      u2zjm1 += bope(k,q)*u(2,i,j-1,q);
      u2zjm2 += bope(k,q)*u(2,i,j-2,q);
   }
   mu2zy= i12*(-mu(i,j+2,k)*u2zjp2 + 8*mu(i,j+1,k)*u2zjp1
                        -8*mu(i,j-1,k)*u2zjm1 + mu(i,j-2,k)*u2zjm2);
   r3 = r3 + stry(j)*mu2zy;

/*   (la*u_x)_z: NOT CENTERED */
   lau1xz=0.;
   for(  q=1 ; q <=8 ; q++ )
      lau1xz += bope(k,q)*( la(i,j,q)*i12*
                  (-u(1,i+2,j,q) + 8*u(1,i+1,j,q)
		   -8*u(1,i-1,j,q) + u(1,i-2,j,q)) );
   r3 = r3 + strx(i)*lau1xz;

/* (la*v_y)_z: NOT CENTERED */
   lau2yz=0.;
   for(  q=1 ; q <=8 ; q++ )
      lau2yz += bope(k,q)*( la(i,j,q)*i12*
                  (-u(2,i,j+2,q) + 8*u(2,i,j+1,q)
                   -8*u(2,i,j-1,q) + u(2,i,j-2,q)) );
   r3 = r3 + stry(j)*lau2yz;

   lu(1,i,j,k) = a1*lu(1,i,j,k) + cof*r1;
   lu(2,i,j,k) = a1*lu(2,i,j,k) + cof*r2;
   lu(3,i,j,k) = a1*lu(3,i,j,k) + cof*r3;
#undef mu
#undef la
#undef u
#undef lu
#undef strx
#undef stry
#undef strz
#undef acof
#undef bope
#undef ghcof
}
}
//-----------------------------------------------------------------------
__device__ void rhs4lower_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			       int nk,
			       float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
			       float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			       int ghost_points )
{
   // Lower boundary nk-5 <= k <= nk
#define mu(i,j,k)     a_mu[base+i+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+i+ni*(j)+nij*(k)]
#define u(c,i,j,k)   a_u[base3+c+3*(i)+nic*(j)+nijc*(k)]   
#define lu(c,i,j,k) a_lu[base3+c+3*(i)+nic*(j)+nijc*(k)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
   //#define strz(k) a_strz[k-kfirst0]
#define acof(i,j,k) dev_acof[(i-1)+6*(j-1)+48*(k-1)]
#define bope(i,j) dev_bope[i-1+6*(j-1)]
#define ghcof(i) dev_ghcof[i-1]
   
   const float_sw4 a1   = 0;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 i12  = 1.0/12;
   const float_sw4 i144 = 1.0/144;
   const float_sw4 tf   = 0.75;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = 3*base-1;
   const int nic  = 3*ni;
   const int nijc = 3*nij;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   //   const int kfirst0 = kfirst;

   int kb;
   int qb, mb;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4;//, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof, mucof, mu1zz, mu2zz, mu3zz;
   float_sw4 lap2mu, u3zip2, u3zip1, u3zim1, u3zim2, lau3zx, mu3xz, u3zjp2, u3zjp1, u3zjm1, u3zjm2;
   float_sw4 lau3zy, mu3yz, mu1zx, mu2zy, u1zip2, u1zip1, u1zim1, u1zim2;
   float_sw4 u2zjp2, u2zjp1, u2zjm1, u2zjm2, lau1xz, lau2yz;

   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if( k < nk-5 || k > nk )
      return;

   cof = 1.0/(h*h);
   if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
   mux1 = mu(i-1,j,k)*strx(i-1)-
		     tf*(mu(i,j,k)*strx(i)+mu(i-2,j,k)*strx(i-2));
   mux2 = mu(i-2,j,k)*strx(i-2)+mu(i+1,j,k)*strx(i+1)+
		     3*(mu(i,j,k)*strx(i)+mu(i-1,j,k)*strx(i-1));
   mux3 = mu(i-1,j,k)*strx(i-1)+mu(i+2,j,k)*strx(i+2)+
		     3*(mu(i+1,j,k)*strx(i+1)+mu(i,j,k)*strx(i));
   mux4 = mu(i+1,j,k)*strx(i+1)-
		     tf*(mu(i,j,k)*strx(i)+mu(i+2,j,k)*strx(i+2));

   muy1 = mu(i,j-1,k)*stry(j-1)-
		     tf*(mu(i,j,k)*stry(j)+mu(i,j-2,k)*stry(j-2));
   muy2 = mu(i,j-2,k)*stry(j-2)+mu(i,j+1,k)*stry(j+1)+
		     3*(mu(i,j,k)*stry(j)+mu(i,j-1,k)*stry(j-1));
   muy3 = mu(i,j-1,k)*stry(j-1)+mu(i,j+2,k)*stry(j+2)+
		     3*(mu(i,j+1,k)*stry(j+1)+mu(i,j,k)*stry(j));
   muy4 = mu(i,j+1,k)*stry(j+1)-
	       tf*(mu(i,j,k)*stry(j)+mu(i,j+2,k)*stry(j+2));

	    /* xx, yy, and zz derivatives: */
	    /* note that we could have introduced intermediate variables for the average of lambda  */
	    /* in the same way as we did for mu */
   r1 = i6*(strx(i)*((2*mux1+la(i-1,j,k)*strx(i-1)-
                       tf*(la(i,j,k)*strx(i)+la(i-2,j,k)*strx(i-2)))*
                              (u(1,i-2,j,k)-u(1,i,j,k))+
           (2*mux2+la(i-2,j,k)*strx(i-2)+la(i+1,j,k)*strx(i+1)+
                        3*(la(i,j,k)*strx(i)+la(i-1,j,k)*strx(i-1)))*
                              (u(1,i-1,j,k)-u(1,i,j,k))+ 
           (2*mux3+la(i-1,j,k)*strx(i-1)+la(i+2,j,k)*strx(i+2)+
                        3*(la(i+1,j,k)*strx(i+1)+la(i,j,k)*strx(i)))*
                              (u(1,i+1,j,k)-u(1,i,j,k))+
                (2*mux4+ la(i+1,j,k)*strx(i+1)-
                       tf*(la(i,j,k)*strx(i)+la(i+2,j,k)*strx(i+2)))*
                (u(1,i+2,j,k)-u(1,i,j,k)) ) + stry(j)*(
                   + muy1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                     muy2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                     muy3*(u(1,i,j+1,k)-u(1,i,j,k)) +
		   muy4*(u(1,i,j+2,k)-u(1,i,j,k)) ) );

    /* all indices ending with 'b' are indices relative to the boundary, going into the domain (1,2,3,...)*/
   kb = nk-k+1;
    /* all coefficient arrays (acof, bope, ghcof) should be indexed with these indices */
    /* all solution and material property arrays should be indexed with (i,j,k) */

	       /* (mu*uz)_z can not be centered */
	       /* second derivative (mu*u_z)_z at grid point z_k */
	       /* averaging the coefficient */
   mu1zz = 0;
   mu2zz = 0;
   mu3zz = 0;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      mucof = 0;
      lap2mu = 0;
      for(  mb=1; mb <= 8; mb++ )
      {
	 mucof  += acof(kb,qb,mb)*mu(i,j,nk-mb+1);
	 lap2mu += acof(kb,qb,mb)*(2*mu(i,j,nk-mb+1)+la(i,j,nk-mb+1));
      }
      mu1zz += mucof*u(1,i,j,nk-qb+1);
      mu2zz += mucof*u(2,i,j,nk-qb+1);
      mu3zz += lap2mu*u(3,i,j,nk-qb+1);
   }
  /* computing the second derivative */
  /* ghost point only influences the first point (k=1) because ghcof(k)=0 for k>=2*/
   r1 = r1 + (mu1zz + ghcof(kb)*mu(i,j,nk)*u(1,i,j,nk+1));

   r2 = i6*(strx(i)*(mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                      mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                      mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                      mux4*(u(2,i+2,j,k)-u(2,i,j,k)) )+ stry(j)*(
                  (2*muy1+la(i,j-1,k)*stry(j-1)-
                        tf*(la(i,j,k)*stry(j)+la(i,j-2,k)*stry(j-2)))*
                          (u(2,i,j-2,k)-u(2,i,j,k))+
           (2*muy2+la(i,j-2,k)*stry(j-2)+la(i,j+1,k)*stry(j+1)+
                        3*(la(i,j,k)*stry(j)+la(i,j-1,k)*stry(j-1)))*
                          (u(2,i,j-1,k)-u(2,i,j,k))+ 
           (2*muy3+la(i,j-1,k)*stry(j-1)+la(i,j+2,k)*stry(j+2)+
                        3*(la(i,j+1,k)*stry(j+1)+la(i,j,k)*stry(j)))*
                          (u(2,i,j+1,k)-u(2,i,j,k))+
                  (2*muy4+la(i,j+1,k)*stry(j+1)-
                       tf*(la(i,j,k)*stry(j)+la(i,j+2,k)*stry(j+2)))*
		  (u(2,i,j+2,k)-u(2,i,j,k)) ) );

		  /* (mu*vz)_z can not be centered */
		  /* second derivative (mu*v_z)_z at grid point z_k */
		  /* averaging the coefficient: already done above */
   r2 = r2 + (mu2zz + ghcof(kb)*mu(i,j,nk)*u(2,i,j,nk+1));

   r3 = i6*(strx(i)*(mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                      mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                      mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                      mux4*(u(3,i+2,j,k)-u(3,i,j,k))  ) + stry(j)*(
                     muy1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                     muy2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                     muy3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                     muy4*(u(3,i,j+2,k)-u(3,i,j,k)) ) );
   r3 = r3 + (mu3zz + ghcof(kb)*(la(i,j,nk)+2*mu(i,j,nk))*
			     u(3,i,j,nk+1));

		  /* cross-terms in first component of rhs */
		  /*   (la*v_y)_x */
   r1 = r1 + strx(i)*stry(j)*(
                 i144*( la(i-2,j,k)*(u(2,i-2,j-2,k)-u(2,i-2,j+2,k)+
                             8*(-u(2,i-2,j-1,k)+u(2,i-2,j+1,k))) - 8*(
                        la(i-1,j,k)*(u(2,i-1,j-2,k)-u(2,i-1,j+2,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i-1,j+1,k))) )+8*(
                        la(i+1,j,k)*(u(2,i+1,j-2,k)-u(2,i+1,j+2,k)+
                             8*(-u(2,i+1,j-1,k)+u(2,i+1,j+1,k))) ) - (
                        la(i+2,j,k)*(u(2,i+2,j-2,k)-u(2,i+2,j+2,k)+
                             8*(-u(2,i+2,j-1,k)+u(2,i+2,j+1,k))) )) 
		 /*   (mu*v_x)_y */
               + i144*( mu(i,j-2,k)*(u(2,i-2,j-2,k)-u(2,i+2,j-2,k)+
                             8*(-u(2,i-1,j-2,k)+u(2,i+1,j-2,k))) - 8*(
                        mu(i,j-1,k)*(u(2,i-2,j-1,k)-u(2,i+2,j-1,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i+1,j-1,k))) )+8*(
                        mu(i,j+1,k)*(u(2,i-2,j+1,k)-u(2,i+2,j+1,k)+
                             8*(-u(2,i-1,j+1,k)+u(2,i+1,j+1,k))) ) - (
                        mu(i,j+2,k)*(u(2,i-2,j+2,k)-u(2,i+2,j+2,k)+
				     8*(-u(2,i-1,j+2,k)+u(2,i+1,j+2,k))) )) );
    /*   (la*w_z)_x: NOT CENTERED */
   u3zip2=0.;
   u3zip1=0.;
   u3zim1=0.;
   u3zim2=0.;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      u3zip2 -= bope(kb,qb)*u(3,i+2,j,nk-qb+1);
      u3zip1 -= bope(kb,qb)*u(3,i+1,j,nk-qb+1);
      u3zim1 -= bope(kb,qb)*u(3,i-1,j,nk-qb+1);
      u3zim2 -= bope(kb,qb)*u(3,i-2,j,nk-qb+1);
   }
   lau3zx= i12*(-la(i+2,j,k)*u3zip2 + 8*la(i+1,j,k)*u3zip1
			 -8*la(i-1,j,k)*u3zim1 + la(i-2,j,k)*u3zim2);
   r1 = r1 + strx(i)*lau3zx;

    /*   (mu*w_x)_z: NOT CENTERED */
   mu3xz=0.;
   for(  qb=1; qb <= 8 ; qb++ )
      mu3xz -= bope(kb,qb)*( mu(i,j,nk-qb+1)*i12*
                  (-u(3,i+2,j,nk-qb+1) + 8*u(3,i+1,j,nk-qb+1)
		   -8*u(3,i-1,j,nk-qb+1) + u(3,i-2,j,nk-qb+1)) );

   r1 = r1 + strx(i)*mu3xz;

	    /* cross-terms in second component of rhs */
	    /*   (mu*u_y)_x */
   r2 = r2 + strx(i)*stry(j)*(
                 i144*( mu(i-2,j,k)*(u(1,i-2,j-2,k)-u(1,i-2,j+2,k)+
                             8*(-u(1,i-2,j-1,k)+u(1,i-2,j+1,k))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j-2,k)-u(1,i-1,j+2,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i-1,j+1,k))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j-2,k)-u(1,i+1,j+2,k)+
                             8*(-u(1,i+1,j-1,k)+u(1,i+1,j+1,k))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j-2,k)-u(1,i+2,j+2,k)+
                             8*(-u(1,i+2,j-1,k)+u(1,i+2,j+1,k))) )) 
		 /* (la*u_x)_y */
               + i144*( la(i,j-2,k)*(u(1,i-2,j-2,k)-u(1,i+2,j-2,k)+
                             8*(-u(1,i-1,j-2,k)+u(1,i+1,j-2,k))) - 8*(
                        la(i,j-1,k)*(u(1,i-2,j-1,k)-u(1,i+2,j-1,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i+1,j-1,k))) )+8*(
                        la(i,j+1,k)*(u(1,i-2,j+1,k)-u(1,i+2,j+1,k)+
                             8*(-u(1,i-1,j+1,k)+u(1,i+1,j+1,k))) ) - (
                        la(i,j+2,k)*(u(1,i-2,j+2,k)-u(1,i+2,j+2,k)+
				     8*(-u(1,i-1,j+2,k)+u(1,i+1,j+2,k))) )) );
	    /* (la*w_z)_y : NOT CENTERED */
   u3zjp2=0.;
   u3zjp1=0.;
   u3zjm1=0.;
   u3zjm2=0.;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      u3zjp2 -= bope(kb,qb)*u(3,i,j+2,nk-qb+1);
      u3zjp1 -= bope(kb,qb)*u(3,i,j+1,nk-qb+1);
      u3zjm1 -= bope(kb,qb)*u(3,i,j-1,nk-qb+1);
      u3zjm2 -= bope(kb,qb)*u(3,i,j-2,nk-qb+1);
   }
   lau3zy= i12*(-la(i,j+2,k)*u3zjp2 + 8*la(i,j+1,k)*u3zjp1
			 -8*la(i,j-1,k)*u3zjm1 + la(i,j-2,k)*u3zjm2);
   r2 = r2 + stry(j)*lau3zy;

	    /* (mu*w_y)_z: NOT CENTERED */
   mu3yz=0.;
   for(  qb=1; qb <= 8 ; qb++ )
      mu3yz -= bope(kb,qb)*( mu(i,j,nk-qb+1)*i12*
                  (-u(3,i,j+2,nk-qb+1) + 8*u(3,i,j+1,nk-qb+1)
                   -8*u(3,i,j-1,nk-qb+1) + u(3,i,j-2,nk-qb+1)) );
   r2 = r2 + stry(j)*mu3yz;

	    /* No centered cross terms in r3 */
	    /*  (mu*u_z)_x: NOT CENTERED */
   u1zip2=0.;
   u1zip1=0.;
   u1zim1=0.;
   u1zim2=0.;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      u1zip2 -= bope(kb,qb)*u(1,i+2,j,nk-qb+1);
      u1zip1 -= bope(kb,qb)*u(1,i+1,j,nk-qb+1);
      u1zim1 -= bope(kb,qb)*u(1,i-1,j,nk-qb+1);
      u1zim2 -= bope(kb,qb)*u(1,i-2,j,nk-qb+1);
   }
   mu1zx= i12*(-mu(i+2,j,k)*u1zip2 + 8*mu(i+1,j,k)*u1zip1
                        -8*mu(i-1,j,k)*u1zim1 + mu(i-2,j,k)*u1zim2);
   r3 = r3 + strx(i)*mu1zx;

	    /* (mu*v_z)_y: NOT CENTERED */
   u2zjp2=0.;
   u2zjp1=0.;
   u2zjm1=0.;
   u2zjm2=0.;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      u2zjp2 -= bope(kb,qb)*u(2,i,j+2,nk-qb+1);
      u2zjp1 -= bope(kb,qb)*u(2,i,j+1,nk-qb+1);
      u2zjm1 -= bope(kb,qb)*u(2,i,j-1,nk-qb+1);
      u2zjm2 -= bope(kb,qb)*u(2,i,j-2,nk-qb+1);
   }
   mu2zy= i12*(-mu(i,j+2,k)*u2zjp2 + 8*mu(i,j+1,k)*u2zjp1
                        -8*mu(i,j-1,k)*u2zjm1 + mu(i,j-2,k)*u2zjm2);
   r3 = r3 + stry(j)*mu2zy;

	    /*   (la*u_x)_z: NOT CENTERED */
   lau1xz=0.;
   for(  qb=1; qb <= 8 ; qb++ )
      lau1xz -= bope(kb,qb)*( la(i,j,nk-qb+1)*i12*
                 (-u(1,i+2,j,nk-qb+1) + 8*u(1,i+1,j,nk-qb+1)
	         -8*u(1,i-1,j,nk-qb+1) + u(1,i-2,j,nk-qb+1)) );
   r3 = r3 + strx(i)*lau1xz;

	    /* (la*v_y)_z: NOT CENTERED */
   lau2yz=0.;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      lau2yz -= bope(kb,qb)*( la(i,j,nk-qb+1)*i12*
                  (-u(2,i,j+2,nk-qb+1) + 8*u(2,i,j+1,nk-qb+1)
                   -8*u(2,i,j-1,nk-qb+1) + u(2,i,j-2,nk-qb+1)) );
   }
   r3 = r3 + stry(j)*lau2yz;

   lu(1,i,j,k) = a1*lu(1,i,j,k) + cof*r1;
   lu(2,i,j,k) = a1*lu(2,i,j,k) + cof*r2;
   lu(3,i,j,k) = a1*lu(3,i,j,k) + cof*r3;
#undef mu
#undef la
#undef u
#undef lu
#undef strx
#undef stry
#undef strz
#undef acof
#undef bope
#undef ghcof
}
}

//-----------------------------------------------------------------------
__global__ void pred_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			  float_sw4* up, float_sw4* u, float_sw4* um, float_sw4* lu, float_sw4* fo,
			  float_sw4* rho, float_sw4 dt2, int ghost_points )
{
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t nthreads = static_cast<size_t>(gridDim.x)*blockDim.x;
   const size_t npts = static_cast<size_t>((ilast-ifirst+1))*(jlast-jfirst+1)*(klast-kfirst+1);
   for (size_t i = myi; i < 3*npts; i += nthreads) 
      up[i] = 2*u[i] - um[i] + dt2/rho[i%npts]*(lu[i]+fo[i]);
}

//-----------------------------------------------------------------------
__global__ void corr_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			  float_sw4* up, float_sw4* lu, float_sw4* fo,
			  float_sw4* rho, float_sw4 dt4, int ghost_points )
{
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   const size_t npts = static_cast<size_t>((ilast-ifirst+1))*(jlast-jfirst+1)*(klast-kfirst+1);
   for (size_t i = myi; i < 3*npts; i += nthreads) 
      up[i  ] += dt4/(12*rho[i%npts])*(lu[i  ]+fo[i  ]);
}

//----------------------------------------------------------------------
__global__ void addsgd4_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
		      float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
			     float_sw4 beta, int ghost_points )
{
#define rho(i,j,k) a_rho[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define up(c,i,j,k) a_up[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define u(c,i,j,k)   a_u[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define um(c,i,j,k) a_um[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define strx(i) a_strx[(i-ifirst)]
#define dcx(i) a_dcx[(i-ifirst)]
#define cox(i) a_cox[(i-ifirst)]
#define stry(j) a_stry[(j-jfirst)]
#define dcy(j) a_dcy[(j-jfirst)]
#define coy(j) a_coy[(j-jfirst)]
#define strz(k) a_strz[(k-kfirst)]
#define dcz(k) a_dcz[(k-kfirst)]
#define coz(k) a_coz[(k-kfirst)]
   if( beta == 0. )
      return;

   const size_t ni = ilast-ifirst+1;
   const size_t nij = ni*(jlast-jfirst+1);
   const size_t nijk = nij*(klast-kfirst+1);
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
      float_sw4 birho=beta/rho(i,j,k);
      for( int c=0 ; c < 3 ; c++ )
      {
	 up(c,i,j,k) -= birho*( 
		  // x-differences
		   strx(i)*coy(j)*coz(k)*(
       rho(i+1,j,k)*dcx(i+1)*
                   ( u(c,i+2,j,k) -2*u(c,i+1,j,k)+ u(c,i,  j,k))
      -2*rho(i,j,k)*dcx(i)  *
                   ( u(c,i+1,j,k) -2*u(c,i,  j,k)+ u(c,i-1,j,k))
      +rho(i-1,j,k)*dcx(i-1)*
                   ( u(c,i,  j,k) -2*u(c,i-1,j,k)+ u(c,i-2,j,k)) 
      -rho(i+1,j,k)*dcx(i+1)*
                   (um(c,i+2,j,k)-2*um(c,i+1,j,k)+um(c,i,  j,k)) 
      +2*rho(i,j,k)*dcx(i)  *
                   (um(c,i+1,j,k)-2*um(c,i,  j,k)+um(c,i-1,j,k)) 
      -rho(i-1,j,k)*dcx(i-1)*
                   (um(c,i,  j,k)-2*um(c,i-1,j,k)+um(c,i-2,j,k)) ) +
// y-differences
      stry(j)*cox(i)*coz(k)*(
      +rho(i,j+1,k)*dcy(j+1)*
                   ( u(c,i,j+2,k) -2*u(c,i,j+1,k)+ u(c,i,j,  k)) 
      -2*rho(i,j,k)*dcy(j)  *
                   ( u(c,i,j+1,k) -2*u(c,i,j,  k)+ u(c,i,j-1,k))
      +rho(i,j-1,k)*dcy(j-1)*
                   ( u(c,i,j,  k) -2*u(c,i,j-1,k)+ u(c,i,j-2,k)) 
      -rho(i,j+1,k)*dcy(j+1)*
                   (um(c,i,j+2,k)-2*um(c,i,j+1,k)+um(c,i,j,  k)) 
      +2*rho(i,j,k)*dcy(j)  *
                   (um(c,i,j+1,k)-2*um(c,i,j,  k)+um(c,i,j-1,k)) 
      -rho(i,j-1,k)*dcy(j-1)*
                   (um(c,i,j,  k)-2*um(c,i,j-1,k)+um(c,i,j-2,k)) ) +
       strz(k)*cox(i)*coy(j)*(
// z-differences
      +rho(i,j,k+1)*dcz(k+1)* 
                 ( u(c,i,j,k+2) -2*u(c,i,j,k+1)+ u(c,i,j,k  )) 
      -2*rho(i,j,k)*dcz(k)  *
                 ( u(c,i,j,k+1) -2*u(c,i,j,k  )+ u(c,i,j,k-1))
      +rho(i,j,k-1)*dcz(k-1)*
                 ( u(c,i,j,k  ) -2*u(c,i,j,k-1)+ u(c,i,j,k-2)) 
      -rho(i,j,k+1)*dcz(k+1)*
                 (um(c,i,j,k+2)-2*um(c,i,j,k+1)+um(c,i,j,k  )) 
      +2*rho(i,j,k)*dcz(k)  *
                 (um(c,i,j,k+1)-2*um(c,i,j,k  )+um(c,i,j,k-1)) 
      -rho(i,j,k-1)*dcz(k-1)*
                 (um(c,i,j,k  )-2*um(c,i,j,k-1)+um(c,i,j,k-2)) ) 
				);
      }
   }
#undef rho
#undef up
#undef u
#undef um
#undef strx
#undef dcx
#undef cox
#undef stry
#undef dcy
#undef coy
#undef strz
#undef dcz
#undef coz
}
 
//----------------------------------------------------------------------
 
__global__ void 
__launch_bounds__(384, 1)
addsgd4_dev_v2( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                    float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
                    float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
                    float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
                    float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
                    float_sw4 beta, int ghost_points )
{   
#define rho(i,j,k) a_rho[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define up(c,i,j,k) a_up[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define u(c,i,j,k) a_u[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define um(c,i,j,k) a_um[c + 3*(i-ifirst)+3*ni*(j-jfirst)+3*nij*(k-kfirst)]
#define strx(i) a_strx[(i-ifirst)]
#define dcx(i) a_dcx[(i-ifirst)]
#define cox(i) a_cox[(i-ifirst)]
#define stry(j) a_stry[(j-jfirst)]
#define dcy(j) a_dcy[(j-jfirst)]
#define coy(j) a_coy[(j-jfirst)]
#define strz(k) a_strz[(k-kfirst)]
#define dcz(k) a_dcz[(k-kfirst)]
#define coz(k) a_coz[(k-kfirst)]
#define su(c,i,j,k) su[i][j][c]
#define sum(c,i,j,k) sum[i][j][c]
  if( beta == 0 )
    return;

  const size_t ni = ilast-ifirst+1;
  const size_t nij = ni*(jlast-jfirst+1);
  const size_t nijk = nij*(klast-kfirst+1);
  int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
  int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
  int ith = threadIdx.x + RADIUS;
  int jth = threadIdx.y + RADIUS;
  __shared__ float_sw4 su[ADDSGD4_BLOCKX+2*RADIUS][ADDSGD4_BLOCKY+2*RADIUS][3],
    sum[ADDSGD4_BLOCKX+2*RADIUS][ADDSGD4_BLOCKY+2*RADIUS][3];
  float_sw4 qu[DIAMETER][3], qum[DIAMETER][3];

#pragma unroll
  for (int q = 1; q < DIAMETER; q++)
  {
    for (int c = 0; c < 3; c++)
    {
      qu[q][c] = u(c,i,j,kfirst+q-1);
      qum[q][c] = um(c,i,j,kfirst+q-1);
    }
  }

  for (int k = kfirst+RADIUS; k <= klast-RADIUS; k++)
  {
    __syncthreads();
    for (int tj = threadIdx.y; tj < blockDim.y+2*RADIUS; tj += blockDim.y)
    {
      int gj = (j-threadIdx.y) + tj - RADIUS;
      if (gj <= jlast)
      {
        for (int ti = threadIdx.x; ti < blockDim.x+2*RADIUS; ti += blockDim.x)
        {
          int gi =  (i-threadIdx.x) + ti - RADIUS;
          if (gi <= ilast)
          {
            #pragma unroll
            for( int c=0 ; c < 3 ; c++ )
            {
              su[c][ti][tj] = u(c,gi,gj,k);
              sum[c][ti][tj] = um(c,gi,gj,k);
            }
          }
        }
      }
    }
    __syncthreads();

    if (i <= ilast-ghost_points && j <= jlast-ghost_points)
    {
#pragma unroll
      for (int q = 0; q < DIAMETER-1; q++)
      {
        for (int c = 0; c < 3; c++)
        {
          qu[q][c] = qu[q+1][c];
          qum[q][c] = qum[q+1][c];
        }
      }
#pragma unroll
      for (int c = 0; c < 3; c++)
      {
        qu[DIAMETER-1][c] = u(c,i,j,k+RADIUS);
        qum[DIAMETER-1][c] = um(c,i,j,k+RADIUS);
      }

      float_sw4 birho=beta/rho(i,j,k);
#pragma unroll
      for( int c=0 ; c < 3 ; c++ )
      {
        up(c,i,j,k) -= birho*(
          // x-differences
          strx(i)*coy(j)*coz(k)*(
            rho(i+1,j,k)*dcx(i+1)*
            ( su(c,ith+2,jth,k) -2*su(c,ith+1,jth,k)+ su(c,ith,  jth,k))
            -2*rho(i,j,k)*dcx(i)  *
            ( su(c,ith+1,jth,k) -2*su(c,ith,  jth,k)+ su(c,ith-1,jth,k))
            +rho(i-1,j,k)*dcx(i-1)*
            ( su(c,ith,  jth,k) -2*su(c,ith-1,jth,k)+ su(c,ith-2,jth,k))
            -rho(i+1,j,k)*dcx(i+1)*
            (sum(c,ith+2,jth,k)-2*sum(c,ith+1,jth,k)+sum(c,ith,  jth,k))
            +2*rho(i,j,k)*dcx(i)  *
            (sum(c,ith+1,jth,k)-2*sum(c,ith,  jth,k)+sum(c,ith-1,jth,k))
            -rho(i-1,j,k)*dcx(i-1)*
            (sum(c,ith,  jth,k)-2*sum(c,ith-1,jth,k)+sum(c,ith-2,jth,k)) ) +
// y-differences
          stry(j)*cox(i)*coz(k)*(
            +rho(i,j+1,k)*dcy(j+1)*
            ( su(c,ith,jth+2,k) -2*su(c,ith,jth+1,k)+ su(c,ith,jth,  k))
            -2*rho(i,j,k)*dcy(j)  *
            ( su(c,ith,jth+1,k) -2*su(c,ith,jth,  k)+ su(c,ith,jth-1,k))
            +rho(i,j-1,k)*dcy(j-1)*
            ( su(c,ith,jth,  k) -2*su(c,ith,jth-1,k)+ su(c,ith,jth-2,k))
            -rho(i,j+1,k)*dcy(j+1)*
            (sum(c,ith,jth+2,k)-2*sum(c,ith,jth+1,k)+sum(c,ith,jth,  k))
            +2*rho(i,j,k)*dcy(j)  *
            (sum(c,ith,jth+1,k)-2*sum(c,ith,jth,  k)+sum(c,ith,jth-1,k))
            -rho(i,j-1,k)*dcy(j-1)*
            (sum(c,ith,jth,  k)-2*sum(c,ith,jth-1,k)+sum(c,ith,jth-2,k)) ) +
          strz(k)*cox(i)*coy(j)*(
// z-differences
            +rho(i,j,k+1)*dcz(k+1)*
            ( qu[4][c] -2*qu[3][c] + su(c,ith,jth,k  ))
            -2*rho(i,j,k)*dcz(k)  *
            ( qu[3][c] -2*su(c,ith,jth,k  )+ qu[1][c])
            +rho(i,j,k-1)*dcz(k-1)*
            ( su(c,ith,jth,k  ) -2*qu[1][c]+ qu[0][c])
            -rho(i,j,k+1)*dcz(k+1)*
            (qum[4][c]-2*qum[3][c]+sum(c,ith,jth,k  ))
            +2*rho(i,j,k)*dcz(k)  *
            (qum[3][c]-2*sum(c,ith,jth,k  )+qum[1][c])
            -rho(i,j,k-1)*dcz(k-1)*
            (sum(c,ith,jth,k  )-2*qum[1][c]+qum[0][c]) )
          );
      }
    }
  }
#undef rho
#undef up
#undef u
#undef um
#undef strx
#undef dcx
#undef cox
#undef stry
#undef dcy
#undef coy
#undef strz
#undef dcz
#undef coz
#undef su
#undef sum
}

//----------------------------------------------------------------------

__global__ void 
__launch_bounds__(384, 1)
addsgd4_dev_rev_v2( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                    float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
                    float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
                    float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
                    float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
                    float_sw4 beta, int ghost_points )
{   
#define rho(i,j,k) a_rho[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define up(c,i,j,k) a_up[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define u(c,i,j,k)   a_u[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define um(c,i,j,k) a_um[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define strx(i) a_strx[(i-ifirst)]
#define dcx(i) a_dcx[(i-ifirst)]
#define cox(i) a_cox[(i-ifirst)]
#define stry(j) a_stry[(j-jfirst)]
#define dcy(j) a_dcy[(j-jfirst)]
#define coy(j) a_coy[(j-jfirst)]
#define strz(k) a_strz[(k-kfirst)]
#define dcz(k) a_dcz[(k-kfirst)]
#define coz(k) a_coz[(k-kfirst)]
#define su(c,i,j,k) su[c][i][j]
#define sum(c,i,j,k) sum[c][i][j]
  if( beta == 0 )
    return;

  const size_t ni = ilast-ifirst+1;
  const size_t nij = ni*(jlast-jfirst+1);
  const size_t nijk = nij*(klast-kfirst+1);
  int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
  int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
  int ith = threadIdx.x + RADIUS;
  int jth = threadIdx.y + RADIUS;
  __shared__ float_sw4 su[3][ADDSGD4_BLOCKX+2*RADIUS][ADDSGD4_BLOCKY+2*RADIUS],
    sum[3][ADDSGD4_BLOCKX+2*RADIUS][ADDSGD4_BLOCKY+2*RADIUS];
  float_sw4 qu[3][DIAMETER], qum[3][DIAMETER];

#pragma unroll
  for (int q = 1; q < DIAMETER; q++)
  {
    for (int c = 0; c < 3; c++)
    {
      qu[c][q] = u(c,i,j,kfirst+q-1);
      qum[c][q] = um(c,i,j,kfirst+q-1);
    }
  }

  for (int k = kfirst+RADIUS; k <= klast-RADIUS; k++)
  {
    __syncthreads();
    for (int tj = threadIdx.y; tj < blockDim.y+2*RADIUS; tj += blockDim.y)
    {
      int gj = (j-threadIdx.y) + tj - RADIUS;
      if (gj <= jlast)
      {
        for (int ti = threadIdx.x; ti < blockDim.x+2*RADIUS; ti += blockDim.x)
        {
          int gi =  (i-threadIdx.x) + ti - RADIUS;
          if (gi <= ilast)
          {
            #pragma unroll
            for( int c=0 ; c < 3 ; c++ )
            {
              su[c][ti][tj] = u(c,gi,gj,k);
              sum[c][ti][tj] = um(c,gi,gj,k);
            }
          }
        }
      }
    }
    __syncthreads();

    if (i <= ilast-ghost_points && j <= jlast-ghost_points)
    {
#pragma unroll
      for (int q = 0; q < DIAMETER-1; q++)
      {
        for (int c = 0; c < 3; c++)
        {
          qu[c][q] = qu[c][q+1];
          qum[c][q] = qum[c][q+1];
        }
      }
#pragma unroll
      for (int c = 0; c < 3; c++)
      {
        qu[c][DIAMETER-1] = u(c,i,j,k+RADIUS);
        qum[c][DIAMETER-1] = um(c,i,j,k+RADIUS);
      }

      float_sw4 birho=beta/rho(i,j,k);
#pragma unroll
      for( int c=0 ; c < 3 ; c++ )
      {
        up(c,i,j,k) -= birho*(
          // x-differences
          strx(i)*coy(j)*coz(k)*(
            rho(i+1,j,k)*dcx(i+1)*
            ( su(c,ith+2,jth,k) -2*su(c,ith+1,jth,k)+ su(c,ith,  jth,k))
            -2*rho(i,j,k)*dcx(i)  *
            ( su(c,ith+1,jth,k) -2*su(c,ith,  jth,k)+ su(c,ith-1,jth,k))
            +rho(i-1,j,k)*dcx(i-1)*
            ( su(c,ith,  jth,k) -2*su(c,ith-1,jth,k)+ su(c,ith-2,jth,k))
            -rho(i+1,j,k)*dcx(i+1)*
            (sum(c,ith+2,jth,k)-2*sum(c,ith+1,jth,k)+sum(c,ith,  jth,k))
            +2*rho(i,j,k)*dcx(i)  *
            (sum(c,ith+1,jth,k)-2*sum(c,ith,  jth,k)+sum(c,ith-1,jth,k))
            -rho(i-1,j,k)*dcx(i-1)*
            (sum(c,ith,  jth,k)-2*sum(c,ith-1,jth,k)+sum(c,ith-2,jth,k)) ) +
// y-differences
          stry(j)*cox(i)*coz(k)*(
            +rho(i,j+1,k)*dcy(j+1)*
            ( su(c,ith,jth+2,k) -2*su(c,ith,jth+1,k)+ su(c,ith,jth,  k))
            -2*rho(i,j,k)*dcy(j)  *
            ( su(c,ith,jth+1,k) -2*su(c,ith,jth,  k)+ su(c,ith,jth-1,k))
            +rho(i,j-1,k)*dcy(j-1)*
            ( su(c,ith,jth,  k) -2*su(c,ith,jth-1,k)+ su(c,ith,jth-2,k))
            -rho(i,j+1,k)*dcy(j+1)*
            (sum(c,ith,jth+2,k)-2*sum(c,ith,jth+1,k)+sum(c,ith,jth,  k))
            +2*rho(i,j,k)*dcy(j)  *
            (sum(c,ith,jth+1,k)-2*sum(c,ith,jth,  k)+sum(c,ith,jth-1,k))
            -rho(i,j-1,k)*dcy(j-1)*
            (sum(c,ith,jth,  k)-2*sum(c,ith,jth-1,k)+sum(c,ith,jth-2,k)) ) +
          strz(k)*cox(i)*coy(j)*(
// z-differences
            +rho(i,j,k+1)*dcz(k+1)*
            ( qu[c][4] -2*qu[c][3] + su(c,ith,jth,k  ))
            -2*rho(i,j,k)*dcz(k)  *
            ( qu[c][3] -2*su(c,ith,jth,k  )+ qu[c][1])
            +rho(i,j,k-1)*dcz(k-1)*
            ( su(c,ith,jth,k  ) -2*qu[c][1]+ qu[c][0])
            -rho(i,j,k+1)*dcz(k+1)*
            (qum[c][4]-2*qum[c][3]+sum(c,ith,jth,k  ))
            +2*rho(i,j,k)*dcz(k)  *
            (qum[c][3]-2*sum(c,ith,jth,k  )+qum[c][1])
            -rho(i,j,k-1)*dcz(k-1)*
            (sum(c,ith,jth,k  )-2*qum[c][1]+qum[c][0]) )
          );
      }
    }
  }
#undef rho
#undef up
#undef u
#undef um
#undef strx
#undef dcx
#undef cox
#undef stry
#undef dcy
#undef coy
#undef strz
#undef dcz
#undef coz
#undef su
#undef sum
}


//-----------------------------------------------------------------------
__global__ void addsgd6_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		      float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
		      float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
			     float_sw4 beta, int ghost_points )
{
#define rho(i,j,k) a_rho[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)]
#define up(c,i,j,k) a_up[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define u(c,i,j,k)   a_u[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define um(c,i,j,k) a_um[(i-ifirst)+ni*(j-jfirst)+nij*(k-kfirst)+nijk*(c)]
#define strx(i) a_strx[(i-ifirst)]
#define dcx(i) a_dcx[(i-ifirst)]
#define cox(i) a_cox[(i-ifirst)]
#define stry(j) a_stry[(j-jfirst)]
#define dcy(j) a_dcy[(j-jfirst)]
#define coy(j) a_coy[(j-jfirst)]
#define strz(k) a_strz[(k-kfirst)]
#define dcz(k) a_dcz[(k-kfirst)]
#define coz(k) a_coz[(k-kfirst)]
   const size_t ni = ilast-ifirst+1;
   const size_t nij = ni*(jlast-jfirst+1);
   const size_t nijk = nij*(klast-kfirst+1);
   int i = ifirst + ghost_points + 1 + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + 1 + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + 1 + threadIdx.z + blockIdx.z*blockDim.z;
   if( (i <= ilast-3) && (j <= jlast-3) && (k <= klast-3) )
   {
   float_sw4 birho=0.5*beta/rho(i,j,k);
   for( int c=0 ; c < 3 ; c++ )
   {
      up(c,i,j,k) += birho*( 
       strx(i)*coy(j)*coz(k)*(
// x-differences
         (rho(i+2,j,k)*dcx(i+2)+rho(i+1,j,k)*dcx(i+1))*(
         u(c,i+3,j,k) -3*u(c,i+2,j,k)+ 3*u(c,i+1,j,k)- u(c,i, j,k) 
      -(um(c,i+3,j,k)-3*um(c,i+2,j,k)+3*um(c,i+1,j,k)-um(c,i, j,k)) )
      -3*(rho(i+1,j,k)*dcx(i+1)+rho(i,j,k)*dcx(i))*(
         u(c,i+2,j,k)- 3*u(c,i+1,j,k)+ 3*u(c,i, j,k)- u(c,i-1,j,k)
      -(um(c,i+2,j,k)-3*um(c,i+1,j,k)+3*um(c,i, j,k)-um(c,i-1,j,k)) )
      +3*(rho(i,j,k)*dcx(i)+rho(i-1,j,k)*dcx(i-1))*(
         u(c,i+1,j,k)- 3*u(c,i,  j,k)+3*u(c,i-1,j,k)- u(c,i-2,j,k) 
      -(um(c,i+1,j,k)-3*um(c,i, j,k)+3*um(c,i-1,j,k)-um(c,i-2,j,k)) )
       - (rho(i-1,j,k)*dcx(i-1)+rho(i-2,j,k)*dcx(i-2))*(
         u(c,i, j,k)- 3*u(c,i-1,j,k)+ 3*u(c,i-2,j,k)- u(c,i-3,j,k) 
      -(um(c,i, j,k)-3*um(c,i-1,j,k)+3*um(c,i-2,j,k)-um(c,i-3,j,k)) )
                 ) +  stry(j)*cox(i)*coz(k)*(
// y-differences
         (rho(i,j+2,k)*dcy(j+2)+rho(i,j+1,k)*dcy(j+1))*(
         u(c,i,j+3,k) -3*u(c,i,j+2,k)+ 3*u(c,i,j+1,k)- u(c,i,  j,k)
      -(um(c,i,j+3,k)-3*um(c,i,j+2,k)+3*um(c,i,j+1,k)-um(c,i,  j,k)) )
      -3*(rho(i,j+1,k)*dcy(j+1)+rho(i,j,k)*dcy(j))*(
         u(c,i,j+2,k) -3*u(c,i,j+1,k)+ 3*u(c,i,  j,k)- u(c,i,j-1,k) 
      -(um(c,i,j+2,k)-3*um(c,i,j+1,k)+3*um(c,i,  j,k)-um(c,i,j-1,k)) )
      +3*(rho(i,j,k)*dcy(j)+rho(i,j-1,k)*dcy(j-1))*(
         u(c,i,j+1,k)- 3*u(c,i, j,k)+ 3*u(c,i,j-1,k)- u(c,i,j-2,k) 
      -(um(c,i,j+1,k)-3*um(c,i, j,k)+3*um(c,i,j-1,k)-um(c,i,j-2,k)) )
       - (rho(i,j-1,k)*dcy(j-1)+rho(i,j-2,k)*dcy(j-2))*(
         u(c,i, j,k)- 3*u(c,i,j-1,k)+  3*u(c,i,j-2,k)- u(c,i,j-3,k) 
      -(um(c,i, j,k)-3*um(c,i,j-1,k)+ 3*um(c,i,j-2,k)-um(c,i,j-3,k)) )
                 ) +  strz(k)*cox(i)*coy(j)*(
// z-differences
         ( rho(i,j,k+2)*dcz(k+2) + rho(i,j,k+1)*dcz(k+1) )*(
         u(c,i,j,k+3)- 3*u(c,i,j,k+2)+ 3*u(c,i,j,k+1)- u(c,i,  j,k) 
      -(um(c,i,j,k+3)-3*um(c,i,j,k+2)+3*um(c,i,j,k+1)-um(c,i,  j,k)) )
      -3*(rho(i,j,k+1)*dcz(k+1)+rho(i,j,k)*dcz(k))*(
         u(c,i,j,k+2) -3*u(c,i,j,k+1)+ 3*u(c,i,  j,k)- u(c,i,j,k-1) 
      -(um(c,i,j,k+2)-3*um(c,i,j,k+1)+3*um(c,i,  j,k)-um(c,i,j,k-1)) )
      +3*(rho(i,j,k)*dcz(k)+rho(i,j,k-1)*dcz(k-1))*(
         u(c,i,j,k+1)- 3*u(c,i,  j,k)+ 3*u(c,i,j,k-1)-u(c,i,j,k-2) 
      -(um(c,i,j,k+1)-3*um(c,i,  j,k)+3*um(c,i,j,k-1)-um(c,i,j,k-2)) )
       - (rho(i,j,k-1)*dcz(k-1)+rho(i,j,k-2)*dcz(k-2))*(
         u(c,i,  j,k) -3*u(c,i,j,k-1)+ 3*u(c,i,j,k-2)- u(c,i,j,k-3)
      -(um(c,i,  j,k)-3*um(c,i,j,k-1)+3*um(c,i,j,k-2)-um(c,i,j,k-3)) )
					     )  );
   }
   }
#undef rho
#undef up
#undef u
#undef um
#undef strx
#undef dcx
#undef cox
#undef stry
#undef dcy
#undef coy
#undef strz
#undef dcz
#undef coz
}

//-----------------------------------------------------------------------
__global__ void rhs4center_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
				float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
				int ghost_points )
{
 // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+(i)+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+(i)+ni*(j)+nij*(k)]
#define u(c,i,j,k)   a_u[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define lu(c,i,j,k) a_lu[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define strz(k) a_strz[k-kfirst0]
   const float_sw4 a1   = 0;
   const float_sw4 i6   = 1.0/6;
   //   const float_sw4 i12  = 1.0/12;
   const float_sw4 i144 = 1.0/144;
   const float_sw4 tf   = 0.75;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = nij*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = base-nijk;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof;
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;

   if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
   cof = 1.0/(h*h);
   mux1 = mu(i-1,j,k)*strx(i-1)-
      tf*(mu(i,j,k)*strx(i)+mu(i-2,j,k)*strx(i-2));
   mux2 = mu(i-2,j,k)*strx(i-2)+mu(i+1,j,k)*strx(i+1)+
      3*(mu(i,j,k)*strx(i)+mu(i-1,j,k)*strx(i-1));
   mux3 = mu(i-1,j,k)*strx(i-1)+mu(i+2,j,k)*strx(i+2)+
      3*(mu(i+1,j,k)*strx(i+1)+mu(i,j,k)*strx(i));
   mux4 = mu(i+1,j,k)*strx(i+1)-
      tf*(mu(i,j,k)*strx(i)+mu(i+2,j,k)*strx(i+2));

   muy1 = mu(i,j-1,k)*stry(j-1)-
      tf*(mu(i,j,k)*stry(j)+mu(i,j-2,k)*stry(j-2));
   muy2 = mu(i,j-2,k)*stry(j-2)+mu(i,j+1,k)*stry(j+1)+
      3*(mu(i,j,k)*stry(j)+mu(i,j-1,k)*stry(j-1));
   muy3 = mu(i,j-1,k)*stry(j-1)+mu(i,j+2,k)*stry(j+2)+
      3*(mu(i,j+1,k)*stry(j+1)+mu(i,j,k)*stry(j));
   muy4 = mu(i,j+1,k)*stry(j+1)-
      tf*(mu(i,j,k)*stry(j)+mu(i,j+2,k)*stry(j+2));

   muz1 = mu(i,j,k-1)*strz(k-1)-
      tf*(mu(i,j,k)*strz(k)+mu(i,j,k-2)*strz(k-2));
   muz2 = mu(i,j,k-2)*strz(k-2)+mu(i,j,k+1)*strz(k+1)+
      3*(mu(i,j,k)*strz(k)+mu(i,j,k-1)*strz(k-1));
   muz3 = mu(i,j,k-1)*strz(k-1)+mu(i,j,k+2)*strz(k+2)+
      3*(mu(i,j,k+1)*strz(k+1)+mu(i,j,k)*strz(k));
   muz4 = mu(i,j,k+1)*strz(k+1)-
      tf*(mu(i,j,k)*strz(k)+mu(i,j,k+2)*strz(k+2));
/* xx, yy, and zz derivatives:*/
/* 75 ops */
   r1 = i6*( strx(i)*( (2*mux1+la(i-1,j,k)*strx(i-1)-
               tf*(la(i,j,k)*strx(i)+la(i-2,j,k)*strx(i-2)))*
                              (u(1,i-2,j,k)-u(1,i,j,k))+
           (2*mux2+la(i-2,j,k)*strx(i-2)+la(i+1,j,k)*strx(i+1)+
                3*(la(i,j,k)*strx(i)+la(i-1,j,k)*strx(i-1)))*
                              (u(1,i-1,j,k)-u(1,i,j,k))+ 
           (2*mux3+la(i-1,j,k)*strx(i-1)+la(i+2,j,k)*strx(i+2)+
                3*(la(i+1,j,k)*strx(i+1)+la(i,j,k)*strx(i)))*
                              (u(1,i+1,j,k)-u(1,i,j,k))+
                (2*mux4+ la(i+1,j,k)*strx(i+1)-
               tf*(la(i,j,k)*strx(i)+la(i+2,j,k)*strx(i+2)))*
                (u(1,i+2,j,k)-u(1,i,j,k)) ) + stry(j)*(
                     muy1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                     muy2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                     muy3*(u(1,i,j+1,k)-u(1,i,j,k)) +
                     muy4*(u(1,i,j+2,k)-u(1,i,j,k)) ) + strz(k)*(
                     muz1*(u(1,i,j,k-2)-u(1,i,j,k)) + 
                     muz2*(u(1,i,j,k-1)-u(1,i,j,k)) + 
                     muz3*(u(1,i,j,k+1)-u(1,i,j,k)) +
                     muz4*(u(1,i,j,k+2)-u(1,i,j,k)) ) );

/* 75 ops */
   r2 = i6*( strx(i)*(mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                      mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                      mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                      mux4*(u(2,i+2,j,k)-u(2,i,j,k)) ) + stry(j)*(
                  (2*muy1+la(i,j-1,k)*stry(j-1)-
                      tf*(la(i,j,k)*stry(j)+la(i,j-2,k)*stry(j-2)))*
                          (u(2,i,j-2,k)-u(2,i,j,k))+
           (2*muy2+la(i,j-2,k)*stry(j-2)+la(i,j+1,k)*stry(j+1)+
                     3*(la(i,j,k)*stry(j)+la(i,j-1,k)*stry(j-1)))*
                          (u(2,i,j-1,k)-u(2,i,j,k))+ 
           (2*muy3+la(i,j-1,k)*stry(j-1)+la(i,j+2,k)*stry(j+2)+
                     3*(la(i,j+1,k)*stry(j+1)+la(i,j,k)*stry(j)))*
                          (u(2,i,j+1,k)-u(2,i,j,k))+
                  (2*muy4+la(i,j+1,k)*stry(j+1)-
                    tf*(la(i,j,k)*stry(j)+la(i,j+2,k)*stry(j+2)))*
                          (u(2,i,j+2,k)-u(2,i,j,k)) ) + strz(k)*(
                     muz1*(u(2,i,j,k-2)-u(2,i,j,k)) + 
                     muz2*(u(2,i,j,k-1)-u(2,i,j,k)) + 
                     muz3*(u(2,i,j,k+1)-u(2,i,j,k)) +
                     muz4*(u(2,i,j,k+2)-u(2,i,j,k)) ) );

/* 75 ops */
   r3 = i6*( strx(i)*(mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                      mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                      mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                      mux4*(u(3,i+2,j,k)-u(3,i,j,k))  ) + stry(j)*(
                     muy1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                     muy2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                     muy3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                     muy4*(u(3,i,j+2,k)-u(3,i,j,k)) ) + strz(k)*(
                  (2*muz1+la(i,j,k-1)*strz(k-1)-
                      tf*(la(i,j,k)*strz(k)+la(i,j,k-2)*strz(k-2)))*
                          (u(3,i,j,k-2)-u(3,i,j,k))+
           (2*muz2+la(i,j,k-2)*strz(k-2)+la(i,j,k+1)*strz(k+1)+
                      3*(la(i,j,k)*strz(k)+la(i,j,k-1)*strz(k-1)))*
                          (u(3,i,j,k-1)-u(3,i,j,k))+ 
           (2*muz3+la(i,j,k-1)*strz(k-1)+la(i,j,k+2)*strz(k+2)+
                      3*(la(i,j,k+1)*strz(k+1)+la(i,j,k)*strz(k)))*
                          (u(3,i,j,k+1)-u(3,i,j,k))+
                  (2*muz4+la(i,j,k+1)*strz(k+1)-
                    tf*(la(i,j,k)*strz(k)+la(i,j,k+2)*strz(k+2)))*
		  (u(3,i,j,k+2)-u(3,i,j,k)) ) );


/* Mixed derivatives: */
/* 29ops /mixed derivative */
/* 116 ops for r1 */
/*   (la*v_y)_x */
   r1 = r1 + strx(i)*stry(j)*
                 i144*( la(i-2,j,k)*(u(2,i-2,j-2,k)-u(2,i-2,j+2,k)+
                             8*(-u(2,i-2,j-1,k)+u(2,i-2,j+1,k))) - 8*(
                        la(i-1,j,k)*(u(2,i-1,j-2,k)-u(2,i-1,j+2,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i-1,j+1,k))) )+8*(
                        la(i+1,j,k)*(u(2,i+1,j-2,k)-u(2,i+1,j+2,k)+
                             8*(-u(2,i+1,j-1,k)+u(2,i+1,j+1,k))) ) - (
                        la(i+2,j,k)*(u(2,i+2,j-2,k)-u(2,i+2,j+2,k)+
                             8*(-u(2,i+2,j-1,k)+u(2,i+2,j+1,k))) )) 
/*   (la*w_z)_x */
               + strx(i)*strz(k)*       
                 i144*( la(i-2,j,k)*(u(3,i-2,j,k-2)-u(3,i-2,j,k+2)+
                             8*(-u(3,i-2,j,k-1)+u(3,i-2,j,k+1))) - 8*(
                        la(i-1,j,k)*(u(3,i-1,j,k-2)-u(3,i-1,j,k+2)+
                             8*(-u(3,i-1,j,k-1)+u(3,i-1,j,k+1))) )+8*(
                        la(i+1,j,k)*(u(3,i+1,j,k-2)-u(3,i+1,j,k+2)+
                             8*(-u(3,i+1,j,k-1)+u(3,i+1,j,k+1))) ) - (
                        la(i+2,j,k)*(u(3,i+2,j,k-2)-u(3,i+2,j,k+2)+
                             8*(-u(3,i+2,j,k-1)+u(3,i+2,j,k+1))) )) 
/*   (mu*v_x)_y */
               + strx(i)*stry(j)*       
                 i144*( mu(i,j-2,k)*(u(2,i-2,j-2,k)-u(2,i+2,j-2,k)+
                             8*(-u(2,i-1,j-2,k)+u(2,i+1,j-2,k))) - 8*(
                        mu(i,j-1,k)*(u(2,i-2,j-1,k)-u(2,i+2,j-1,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i+1,j-1,k))) )+8*(
                        mu(i,j+1,k)*(u(2,i-2,j+1,k)-u(2,i+2,j+1,k)+
                             8*(-u(2,i-1,j+1,k)+u(2,i+1,j+1,k))) ) - (
                        mu(i,j+2,k)*(u(2,i-2,j+2,k)-u(2,i+2,j+2,k)+
                             8*(-u(2,i-1,j+2,k)+u(2,i+1,j+2,k))) )) 
/*   (mu*w_x)_z */
               + strx(i)*strz(k)*       
                 i144*( mu(i,j,k-2)*(u(3,i-2,j,k-2)-u(3,i+2,j,k-2)+
                             8*(-u(3,i-1,j,k-2)+u(3,i+1,j,k-2))) - 8*(
                        mu(i,j,k-1)*(u(3,i-2,j,k-1)-u(3,i+2,j,k-1)+
                             8*(-u(3,i-1,j,k-1)+u(3,i+1,j,k-1))) )+8*(
                        mu(i,j,k+1)*(u(3,i-2,j,k+1)-u(3,i+2,j,k+1)+
                             8*(-u(3,i-1,j,k+1)+u(3,i+1,j,k+1))) ) - (
                        mu(i,j,k+2)*(u(3,i-2,j,k+2)-u(3,i+2,j,k+2)+
				     8*(-u(3,i-1,j,k+2)+u(3,i+1,j,k+2))) )) ;

/* 116 ops for r2 */
/*   (mu*u_y)_x */
   r2 = r2 + strx(i)*stry(j)*
                 i144*( mu(i-2,j,k)*(u(1,i-2,j-2,k)-u(1,i-2,j+2,k)+
                             8*(-u(1,i-2,j-1,k)+u(1,i-2,j+1,k))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j-2,k)-u(1,i-1,j+2,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i-1,j+1,k))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j-2,k)-u(1,i+1,j+2,k)+
                             8*(-u(1,i+1,j-1,k)+u(1,i+1,j+1,k))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j-2,k)-u(1,i+2,j+2,k)+
                             8*(-u(1,i+2,j-1,k)+u(1,i+2,j+1,k))) )) 
/* (la*u_x)_y */
              + strx(i)*stry(j)*
                 i144*( la(i,j-2,k)*(u(1,i-2,j-2,k)-u(1,i+2,j-2,k)+
                             8*(-u(1,i-1,j-2,k)+u(1,i+1,j-2,k))) - 8*(
                        la(i,j-1,k)*(u(1,i-2,j-1,k)-u(1,i+2,j-1,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i+1,j-1,k))) )+8*(
                        la(i,j+1,k)*(u(1,i-2,j+1,k)-u(1,i+2,j+1,k)+
                             8*(-u(1,i-1,j+1,k)+u(1,i+1,j+1,k))) ) - (
                        la(i,j+2,k)*(u(1,i-2,j+2,k)-u(1,i+2,j+2,k)+
                             8*(-u(1,i-1,j+2,k)+u(1,i+1,j+2,k))) )) 
/* (la*w_z)_y */
               + stry(j)*strz(k)*
                 i144*( la(i,j-2,k)*(u(3,i,j-2,k-2)-u(3,i,j-2,k+2)+
                             8*(-u(3,i,j-2,k-1)+u(3,i,j-2,k+1))) - 8*(
                        la(i,j-1,k)*(u(3,i,j-1,k-2)-u(3,i,j-1,k+2)+
                             8*(-u(3,i,j-1,k-1)+u(3,i,j-1,k+1))) )+8*(
                        la(i,j+1,k)*(u(3,i,j+1,k-2)-u(3,i,j+1,k+2)+
                             8*(-u(3,i,j+1,k-1)+u(3,i,j+1,k+1))) ) - (
                        la(i,j+2,k)*(u(3,i,j+2,k-2)-u(3,i,j+2,k+2)+
                             8*(-u(3,i,j+2,k-1)+u(3,i,j+2,k+1))) ))
/* (mu*w_y)_z */
               + stry(j)*strz(k)*
                 i144*( mu(i,j,k-2)*(u(3,i,j-2,k-2)-u(3,i,j+2,k-2)+
                             8*(-u(3,i,j-1,k-2)+u(3,i,j+1,k-2))) - 8*(
                        mu(i,j,k-1)*(u(3,i,j-2,k-1)-u(3,i,j+2,k-1)+
                             8*(-u(3,i,j-1,k-1)+u(3,i,j+1,k-1))) )+8*(
                        mu(i,j,k+1)*(u(3,i,j-2,k+1)-u(3,i,j+2,k+1)+
                             8*(-u(3,i,j-1,k+1)+u(3,i,j+1,k+1))) ) - (
                        mu(i,j,k+2)*(u(3,i,j-2,k+2)-u(3,i,j+2,k+2)+
				     8*(-u(3,i,j-1,k+2)+u(3,i,j+1,k+2))) )) ;
/* 116 ops for r3 */
/*  (mu*u_z)_x */
   r3 = r3 + strx(i)*strz(k)*
                 i144*( mu(i-2,j,k)*(u(1,i-2,j,k-2)-u(1,i-2,j,k+2)+
                             8*(-u(1,i-2,j,k-1)+u(1,i-2,j,k+1))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j,k-2)-u(1,i-1,j,k+2)+
                             8*(-u(1,i-1,j,k-1)+u(1,i-1,j,k+1))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j,k-2)-u(1,i+1,j,k+2)+
                             8*(-u(1,i+1,j,k-1)+u(1,i+1,j,k+1))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j,k-2)-u(1,i+2,j,k+2)+
                             8*(-u(1,i+2,j,k-1)+u(1,i+2,j,k+1))) )) 
/* (mu*v_z)_y */
              + stry(j)*strz(k)*
                 i144*( mu(i,j-2,k)*(u(2,i,j-2,k-2)-u(2,i,j-2,k+2)+
                             8*(-u(2,i,j-2,k-1)+u(2,i,j-2,k+1))) - 8*(
                        mu(i,j-1,k)*(u(2,i,j-1,k-2)-u(2,i,j-1,k+2)+
                             8*(-u(2,i,j-1,k-1)+u(2,i,j-1,k+1))) )+8*(
                        mu(i,j+1,k)*(u(2,i,j+1,k-2)-u(2,i,j+1,k+2)+
                             8*(-u(2,i,j+1,k-1)+u(2,i,j+1,k+1))) ) - (
                        mu(i,j+2,k)*(u(2,i,j+2,k-2)-u(2,i,j+2,k+2)+
                             8*(-u(2,i,j+2,k-1)+u(2,i,j+2,k+1))) ))
/*   (la*u_x)_z */
              + strx(i)*strz(k)*
                 i144*( la(i,j,k-2)*(u(1,i-2,j,k-2)-u(1,i+2,j,k-2)+
                             8*(-u(1,i-1,j,k-2)+u(1,i+1,j,k-2))) - 8*(
                        la(i,j,k-1)*(u(1,i-2,j,k-1)-u(1,i+2,j,k-1)+
                             8*(-u(1,i-1,j,k-1)+u(1,i+1,j,k-1))) )+8*(
                        la(i,j,k+1)*(u(1,i-2,j,k+1)-u(1,i+2,j,k+1)+
                             8*(-u(1,i-1,j,k+1)+u(1,i+1,j,k+1))) ) - (
                        la(i,j,k+2)*(u(1,i-2,j,k+2)-u(1,i+2,j,k+2)+
                             8*(-u(1,i-1,j,k+2)+u(1,i+1,j,k+2))) )) 
/* (la*v_y)_z */
              + stry(j)*strz(k)*
                 i144*( la(i,j,k-2)*(u(2,i,j-2,k-2)-u(2,i,j+2,k-2)+
                             8*(-u(2,i,j-1,k-2)+u(2,i,j+1,k-2))) - 8*(
                        la(i,j,k-1)*(u(2,i,j-2,k-1)-u(2,i,j+2,k-1)+
                             8*(-u(2,i,j-1,k-1)+u(2,i,j+1,k-1))) )+8*(
                        la(i,j,k+1)*(u(2,i,j-2,k+1)-u(2,i,j+2,k+1)+
                             8*(-u(2,i,j-1,k+1)+u(2,i,j+1,k+1))) ) - (
                        la(i,j,k+2)*(u(2,i,j-2,k+2)-u(2,i,j+2,k+2)+
				     8*(-u(2,i,j-1,k+2)+u(2,i,j+1,k+2))) )) ;

/* 9 ops */
   lu(1,i,j,k) = a1*lu(1,i,j,k) + cof*r1;
   lu(2,i,j,k) = a1*lu(2,i,j,k) + cof*r2;
   lu(3,i,j,k) = a1*lu(3,i,j,k) + cof*r3;
#undef mu
#undef la
#undef u
#undef lu
#undef strx
#undef stry
#undef strz
}
}

//-----------------------------------------------------------------------
__global__ void 
__launch_bounds__(256,1)
rhs4sgcurv_dev_v2( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_met, float_sw4* a_jac, float_sw4* a_lu, 
				int onesided4, float_sw4* a_strx, float_sw4* a_stry, int ghost_points )
{
 // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+i+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+i+ni*(j)+nij*(k)]
#define jac(i,j,k)   a_jac[base+i+ni*(j)+nij*(k)]
#define u(c,i,j,k)     a_u[base3+(c)+3*(i)+ni3*(j)+nij3*(k)]   
#define lu(c,i,j,k)   a_lu[base3+(c)+3*(i)+ni3*(j)+nij3*(k)]   
#define met(c,i,j,k) a_met[base4+(c)+4*(i)+ni4*(j)+nij4*(k)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define uSh(c,i,j,k) a_uSh[k][j][i][c-1]
   const float_sw4 a1   = 0;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 tf   = 0.75;
   const float_sw4 c1   = 2.0/3;
   const float_sw4 c2   = -1.0/12;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = nij*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = 3*base-1;
   const int base4 = 4*base-1;
   const int ni3  = 3*ni;
   const int nij3 = 3*nij;
   const int ni4  = 4*ni;
   const int nij4 = 4*nij;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof;
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int kbegin = kfirst;
//   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;

  __shared__ float_sw4 a_uSh[DIAMETER][RHS4_BLOCKY+2*RADIUS][RHS4_BLOCKX+2*RADIUS][3];
   const int ith = threadIdx.x + RADIUS;
   const int jth = threadIdx.y + RADIUS;
   int kthm0=3, kthm1=2, kthm2=1, kthm3=0, kthm4=4, kthtmp;

   if(onesided4 == 1)
       kbegin = 5;
//     k = 7 + threadIdx.z + blockIdx.z*blockDim.z;

   for (int tk = 0; tk < DIAMETER; tk++)
   {
     int gk =  kbegin + tk;
     if ( gk <= klast)
     {
       for (int tj = threadIdx.y; tj < blockDim.y+2*RADIUS; tj += blockDim.y)
       {
         int gj =  (j-threadIdx.y) + tj - RADIUS;
         if ( gj <= jlast)
         {
           for (int ti = threadIdx.x; ti < blockDim.x+2*RADIUS; ti += blockDim.x)
           {
             int gi =  (i-threadIdx.x) + ti - RADIUS;
             if ( gi <= ilast)
             {
               uSh(1, ti, tj, tk) = u(1, gi, gj, gk);
               uSh(2, ti, tj, tk) = u(2, gi, gj, gk);
               uSh(3, ti, tj, tk) = u(3, gi, gj, gk);
             }
           }
         }
       }
     }
   }
   __syncthreads();

   for (int k = kbegin+RADIUS; k <= klast-RADIUS; k++)
   {
     kthtmp = kthm4;
     kthm4  = kthm3;
     kthm3  = kthm2;
     kthm2  = kthm1;
     kthm1  = kthm0;
     kthm0  = kthtmp;
   
   //if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   if ( (i <= ilast-RADIUS) && (j <= jlast-RADIUS) )
	 {
// 5 ops
	    float_sw4 ijac = strx(i)*stry(j)/jac(i,j,k);
            float_sw4 istry = 1/(stry(j));
            float_sw4 istrx = 1/(strx(i));
            float_sw4 istrxy = istry*istrx;

            float_sw4 r1 = 0.;

	    // pp derivative (u)
// 53 ops, tot=58
	    float_sw4 cof1=(2*mu(i-2,j,k)+la(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)
	       *strx(i-2);
	    float_sw4 cof2=(2*mu(i-1,j,k)+la(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)
	       *strx(i-1);
	    float_sw4 cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)
		  *strx(i);
	    float_sw4 cof4=(2*mu(i+1,j,k)+la(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)
	     *strx(i+1);
	    float_sw4 cof5=(2*mu(i+2,j,k)+la(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)
	     *strx(i+2);
            float_sw4 mux1 = cof2 -tf*(cof3+cof1);
            float_sw4 mux2 = cof1 + cof4+3*(cof3+cof2);
            float_sw4 mux3 = cof2 + cof5+3*(cof4+cof3);
            float_sw4 mux4 = cof4-tf*(cof3+cof5);

            r1 +=  i6* (
                    mux1*(uSh(1,ith-2,jth,kthm2)-uSh(1,ith,jth,kthm2)) + 
                    mux2*(uSh(1,ith-1,jth,kthm2)-uSh(1,ith,jth,kthm2)) + 
                    mux3*(uSh(1,ith+1,jth,kthm2)-uSh(1,ith,jth,kthm2)) +
                    mux4*(uSh(1,ith+2,jth,kthm2)-uSh(1,ith,jth,kthm2))  )*istry;
	    // qq derivative (u)
// 43 ops, tot=101
	    cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	    cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	    cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	    cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(1,ith,jth-2,kthm2)-uSh(1,ith,jth,kthm2)) + 
                    mux2*(uSh(1,ith,jth-1,kthm2)-uSh(1,ith,jth,kthm2)) + 
                    mux3*(uSh(1,ith,jth+1,kthm2)-uSh(1,ith,jth,kthm2)) +
                    mux4*(uSh(1,ith,jth+2,kthm2)-uSh(1,ith,jth,kthm2))  )*istrx;
	    // rr derivative (u)
// 5*11+14+14=83 ops, tot=184
	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)
	       +   mu(i,j,k-2)*(met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j)+
				met(4,i,j,k-2)*met(4,i,j,k-2));
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)
	       +   mu(i,j,k-1)*(met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j)+
				met(4,i,j,k-1)*met(4,i,j,k-1) );
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i) +
	     mu(i,j,k)*(met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j)+
			met(4,i,j,k)*met(4,i,j,k));
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)
	       +   mu(i,j,k+1)*(met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j)+
				met(4,i,j,k+1)*met(4,i,j,k+1));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)
	       +   mu(i,j,k+2)*( met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j)+
				 met(4,i,j,k+2)*met(4,i,j,k+2));

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(1,ith,jth,kthm4)-uSh(1,ith,jth,kthm2)) + 
                    mux2*(uSh(1,ith,jth,kthm3)-uSh(1,ith,jth,kthm2)) + 
                    mux3*(uSh(1,ith,jth,kthm1)-uSh(1,ith,jth,kthm2)) +
                    mux4*(uSh(1,ith,jth,kthm0)-uSh(1,ith,jth,kthm2))  )*istrxy;

	    // rr derivative (v)
	    // 42 ops, tot=226
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(3,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(3,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(3,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(3,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(3,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(2,ith,jth,kthm4)-uSh(2,ith,jth,kthm2)) + 
                    mux2*(uSh(2,ith,jth,kthm3)-uSh(2,ith,jth,kthm2)) + 
                    mux3*(uSh(2,ith,jth,kthm1)-uSh(2,ith,jth,kthm2)) +
                    mux4*(uSh(2,ith,jth,kthm0)-uSh(2,ith,jth,kthm2))  );

	    // rr derivative (w)
// 43 ops, tot=269
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(4,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
		    mux1*(uSh(3,ith,jth,kthm4)-uSh(3,ith,jth,kthm2)) + 
                    mux2*(uSh(3,ith,jth,kthm3)-uSh(3,ith,jth,kthm2)) + 
                    mux3*(uSh(3,ith,jth,kthm1)-uSh(3,ith,jth,kthm2)) +
                    mux4*(uSh(3,ith,jth,kthm0)-uSh(3,ith,jth,kthm2))  )*istry;

	    // pq-derivatives
// 38 ops, tot=307
	    r1 += 
        c2*(  mu(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(2,ith+2,jth+2,kthm2)-uSh(2,ith-2,jth+2,kthm2)) +
             c1*(uSh(2,ith+1,jth+2,kthm2)-uSh(2,ith-1,jth+2,kthm2))    )
          - mu(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(2,ith+2,jth-2,kthm2)-uSh(2,ith-2,jth-2,kthm2))+
             c1*(uSh(2,ith+1,jth-2,kthm2)-uSh(2,ith-1,jth-2,kthm2))     )
          ) +
        c1*(  mu(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(uSh(2,ith+2,jth+1,kthm2)-uSh(2,ith-2,jth+1,kthm2)) +
               c1*(uSh(2,ith+1,jth+1,kthm2)-uSh(2,ith-1,jth+1,kthm2))  )
           - mu(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(uSh(2,ith+2,jth-1,kthm2)-uSh(2,ith-2,jth-1,kthm2)) + 
               c1*(uSh(2,ith+1,jth-1,kthm2)-uSh(2,ith-1,jth-1,kthm2))));

	    // qp-derivatives
// 38 ops, tot=345
	    r1 += 
        c2*(  la(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(2,ith+2,jth+2,kthm2)-uSh(2,ith+2,jth-2,kthm2)) +
             c1*(uSh(2,ith+2,jth+1,kthm2)-uSh(2,ith+2,jth-1,kthm2))    )
          - la(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(2,ith-2,jth+2,kthm2)-uSh(2,ith-2,jth-2,kthm2))+
             c1*(uSh(2,ith-2,jth+1,kthm2)-uSh(2,ith-2,jth-1,kthm2))     )
          ) +
        c1*(  la(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(uSh(2,ith+1,jth+2,kthm2)-uSh(2,ith+1,jth-2,kthm2)) +
               c1*(uSh(2,ith+1,jth+1,kthm2)-uSh(2,ith+1,jth-1,kthm2))  )
           - la(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(uSh(2,ith-1,jth+2,kthm2)-uSh(2,ith-1,jth-2,kthm2)) + 
               c1*(uSh(2,ith-1,jth+1,kthm2)-uSh(2,ith-1,jth-1,kthm2))));

      // pr-derivatives
// 130 ops., tot=475
	    r1 += c2*(
       (2*mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith-2,jth,kthm0)) +
             c1*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith-1,jth,kthm0))   )*strx(i)*istry 
        + mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(2,ith+2,jth,kthm0)-uSh(2,ith-2,jth,kthm0)) +
             c1*(uSh(2,ith+1,jth,kthm0)-uSh(2,ith-1,jth,kthm0))  ) 
        + mu(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(3,ith+2,jth,kthm0)-uSh(3,ith-2,jth,kthm0)) +
             c1*(uSh(3,ith+1,jth,kthm0)-uSh(3,ith-1,jth,kthm0))  )*istry
       - ((2*mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(1,ith+2,jth,kthm4)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm4)-uSh(1,ith-1,jth,kthm4))  )*strx(i)*istry  
          + mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(2,ith+2,jth,kthm4)-uSh(2,ith-2,jth,kthm4)) +
             c1*(uSh(2,ith+1,jth,kthm4)-uSh(2,ith-1,jth,kthm4))   ) 
          + mu(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(3,ith+2,jth,kthm4)-uSh(3,ith-2,jth,kthm4)) +
             c1*(uSh(3,ith+1,jth,kthm4)-uSh(3,ith-1,jth,kthm4))   )*istry )
                  ) + c1*(  
          (2*mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith-2,jth,kthm1)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith-1,jth,kthm1)) )*strx(i)*istry 
          + mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(2,ith+2,jth,kthm1)-uSh(2,ith-2,jth,kthm1)) +
             c1*(uSh(2,ith+1,jth,kthm1)-uSh(2,ith-1,jth,kthm1)) ) 
          + mu(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(3,ith+2,jth,kthm1)-uSh(3,ith-2,jth,kthm1)) +
             c1*(uSh(3,ith+1,jth,kthm1)-uSh(3,ith-1,jth,kthm1))  )*istry
       - ((2*mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(1,ith+2,jth,kthm3)-uSh(1,ith-2,jth,kthm3)) +
             c1*(uSh(1,ith+1,jth,kthm3)-uSh(1,ith-1,jth,kthm3)) )*strx(i)*istry  
          + mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(2,ith+2,jth,kthm3)-uSh(2,ith-2,jth,kthm3)) +
             c1*(uSh(2,ith+1,jth,kthm3)-uSh(2,ith-1,jth,kthm3)) ) 
          + mu(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(3,ith+2,jth,kthm3)-uSh(3,ith-2,jth,kthm3)) +
             c1*(uSh(3,ith+1,jth,kthm3)-uSh(3,ith-1,jth,kthm3))   )*istry  ) );

	    // rp derivatives
// 130 ops, tot=605
	    r1 += ( c2*(
       (2*mu(i+2,j,k)+la(i+2,j,k))*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith+2,jth,kthm4)) +
             c1*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith+2,jth,kthm3))   )*strx(i+2) 
        + la(i+2,j,k)*met(3,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(2,ith+2,jth,kthm0)-uSh(2,ith+2,jth,kthm4)) +
             c1*(uSh(2,ith+2,jth,kthm1)-uSh(2,ith+2,jth,kthm3))  )*stry(j) 
        + la(i+2,j,k)*met(4,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(3,ith+2,jth,kthm0)-uSh(3,ith+2,jth,kthm4)) +
             c1*(uSh(3,ith+2,jth,kthm1)-uSh(3,ith+2,jth,kthm3))  )
       - ((2*mu(i-2,j,k)+la(i-2,j,k))*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(1,ith-2,jth,kthm0)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith-2,jth,kthm1)-uSh(1,ith-2,jth,kthm3))  )*strx(i-2) 
          + la(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(2,ith-2,jth,kthm0)-uSh(2,ith-2,jth,kthm4)) +
             c1*(uSh(2,ith-2,jth,kthm1)-uSh(2,ith-2,jth,kthm3))   )*stry(j) 
          + la(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(3,ith-2,jth,kthm0)-uSh(3,ith-2,jth,kthm4)) +
             c1*(uSh(3,ith-2,jth,kthm1)-uSh(3,ith-2,jth,kthm3))   ) )
                  ) + c1*(  
          (2*mu(i+1,j,k)+la(i+1,j,k))*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith+1,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith+1,jth,kthm3)) )*strx(i+1) 
          + la(i+1,j,k)*met(3,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(2,ith+1,jth,kthm0)-uSh(2,ith+1,jth,kthm4)) +
             c1*(uSh(2,ith+1,jth,kthm1)-uSh(2,ith+1,jth,kthm3)) )*stry(j) 
          + la(i+1,j,k)*met(4,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(3,ith+1,jth,kthm0)-uSh(3,ith+1,jth,kthm4)) +
             c1*(uSh(3,ith+1,jth,kthm1)-uSh(3,ith+1,jth,kthm3))  )
       - ((2*mu(i-1,j,k)+la(i-1,j,k))*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(1,ith-1,jth,kthm0)-uSh(1,ith-1,jth,kthm4)) +
             c1*(uSh(1,ith-1,jth,kthm1)-uSh(1,ith-1,jth,kthm3)) )*strx(i-1) 
          + la(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(2,ith-1,jth,kthm0)-uSh(2,ith-1,jth,kthm4)) +
             c1*(uSh(2,ith-1,jth,kthm1)-uSh(2,ith-1,jth,kthm3)) )*stry(j) 
          + la(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(3,ith-1,jth,kthm0)-uSh(3,ith-1,jth,kthm4)) +
             c1*(uSh(3,ith-1,jth,kthm1)-uSh(3,ith-1,jth,kthm3))   )  ) ) )*istry;

	    // qr derivatives
// 82 ops, tot=687
	    r1 += c2*(
         mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(1,ith,jth+2,kthm0)-uSh(1,ith,jth-2,kthm0)) +
             c1*(uSh(1,ith,jth+1,kthm0)-uSh(1,ith,jth-1,kthm0))   )*stry(j)*istrx 
        + la(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth-2,kthm0)) +
             c1*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth-1,kthm0))  ) 
       - ( mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(1,ith,jth+2,kthm4)-uSh(1,ith,jth-2,kthm4)) +
             c1*(uSh(1,ith,jth+1,kthm4)-uSh(1,ith,jth-1,kthm4))  )*stry(j)*istrx  
          + la(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(2,ith,jth+2,kthm4)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm4)-uSh(2,ith,jth-1,kthm4))   ) ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(1,ith,jth+2,kthm1)-uSh(1,ith,jth-2,kthm1)) +
             c1*(uSh(1,ith,jth+1,kthm1)-uSh(1,ith,jth-1,kthm1)) )*stry(j)*istrx  
          + la(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth-2,kthm1)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth-1,kthm1)) )  
       - ( mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(1,ith,jth+2,kthm3)-uSh(1,ith,jth-2,kthm3)) +
             c1*(uSh(1,ith,jth+1,kthm3)-uSh(1,ith,jth-1,kthm3)) )*stry(j)*istrx  
          + la(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(2,ith,jth+2,kthm3)-uSh(2,ith,jth-2,kthm3)) +
             c1*(uSh(2,ith,jth+1,kthm3)-uSh(2,ith,jth-1,kthm3)) ) ) );

	    // rq derivatives
// 82 ops, tot=769
	    r1 += c2*(
         mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(1,ith,jth+2,kthm0)-uSh(1,ith,jth+2,kthm4)) +
             c1*(uSh(1,ith,jth+2,kthm1)-uSh(1,ith,jth+2,kthm3))   )*stry(j+2)*istrx 
        + mu(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth+2,kthm4)) +
             c1*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth+2,kthm3))  ) 
       - ( mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(1,ith,jth-2,kthm0)-uSh(1,ith,jth-2,kthm4)) +
             c1*(uSh(1,ith,jth-2,kthm1)-uSh(1,ith,jth-2,kthm3))  )*stry(j-2)*istrx  
          + mu(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(2,ith,jth-2,kthm0)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth-2,kthm1)-uSh(2,ith,jth-2,kthm3))   ) ) 
                  ) + c1*(  
           mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(1,ith,jth+1,kthm0)-uSh(1,ith,jth+1,kthm4)) +
             c1*(uSh(1,ith,jth+1,kthm1)-uSh(1,ith,jth+1,kthm3)) )*stry(j+1)*istrx
          + mu(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth+1,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth+1,kthm3)) )
       - ( mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(1,ith,jth-1,kthm0)-uSh(1,ith,jth-1,kthm4)) +
             c1*(uSh(1,ith,jth-1,kthm1)-uSh(1,ith,jth-1,kthm3)) )*stry(j-1)*istrx    
          + mu(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(2,ith,jth-1,kthm0)-uSh(2,ith,jth-1,kthm4)) +
             c1*(uSh(2,ith,jth-1,kthm1)-uSh(2,ith,jth-1,kthm3)) ) ) );

// 4 ops, tot=773
	    lu(1,i,j,k) = a1*lu(1,i,j,k) + r1*ijac;
// v-equation

	    r1 = 0.;
	    // pp derivative (v)
// 43 ops, tot=816
	    cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	    cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	    cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	    cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(2,ith-2,jth,kthm2)-uSh(2,ith,jth,kthm2)) + 
                    mux2*(uSh(2,ith-1,jth,kthm2)-uSh(2,ith,jth,kthm2)) + 
                    mux3*(uSh(2,ith+1,jth,kthm2)-uSh(2,ith,jth,kthm2)) +
                    mux4*(uSh(2,ith+2,jth,kthm2)-uSh(2,ith,jth,kthm2))  )*istry;
// qq derivative (v)
// 53 ops, tot=869
	    cof1=(2*mu(i,j-2,k)+la(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)
	     *stry(j-2);
	    cof2=(2*mu(i,j-1,k)+la(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)
	       *stry(j-1);
	    cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)
	       *stry(j);
	    cof4=(2*mu(i,j+1,k)+la(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)
	       *stry(j+1);
	    cof5=(2*mu(i,j+2,k)+la(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)
	       *stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(2,ith,jth-2,kthm2)-uSh(2,ith,jth,kthm2)) + 
                    mux2*(uSh(2,ith,jth-1,kthm2)-uSh(2,ith,jth,kthm2)) + 
                    mux3*(uSh(2,ith,jth+1,kthm2)-uSh(2,ith,jth,kthm2)) +
                    mux4*(uSh(2,ith,jth+2,kthm2)-uSh(2,ith,jth,kthm2))  )*istrx;

// rr derivative (u)
// 42 ops, tot=911
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(3,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(3,i,j,k-1);
	    cof3=(mu(i,j,k)+  la(i,j,k)  )*met(2,i,j,k)*  met(3,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(3,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(3,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(1,ith,jth,kthm4)-uSh(1,ith,jth,kthm2)) + 
                    mux2*(uSh(1,ith,jth,kthm3)-uSh(1,ith,jth,kthm2)) + 
                    mux3*(uSh(1,ith,jth,kthm1)-uSh(1,ith,jth,kthm2)) +
                    mux4*(uSh(1,ith,jth,kthm0)-uSh(1,ith,jth,kthm2))  );

// rr derivative (v)
// 83 ops, tot=994
 	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j)
	       +    mu(i,j,k-2)*(met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)+
				 met(4,i,j,k-2)*met(4,i,j,k-2));
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j)
	       +    mu(i,j,k-1)*(met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)+
				 met(4,i,j,k-1)*met(4,i,j,k-1));
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j) +
	       mu(i,j,k)*(met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i)+
			  met(4,i,j,k)*met(4,i,j,k));
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j)
	       +    mu(i,j,k+1)*(met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)+
				 met(4,i,j,k+1)*met(4,i,j,k+1));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j)
	       +    mu(i,j,k+2)*(met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)+
				 met(4,i,j,k+2)*met(4,i,j,k+2));

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(2,ith,jth,kthm4)-uSh(2,ith,jth,kthm2)) + 
                    mux2*(uSh(2,ith,jth,kthm3)-uSh(2,ith,jth,kthm2)) + 
                    mux3*(uSh(2,ith,jth,kthm1)-uSh(2,ith,jth,kthm2)) +
                    mux4*(uSh(2,ith,jth,kthm0)-uSh(2,ith,jth,kthm2))  )*istrxy;

// rr derivative (w)
// 43 ops, tot=1037
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)  +la(i,j,k)  )*met(3,i,j,k)*  met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(4,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(3,ith,jth,kthm4)-uSh(3,ith,jth,kthm2)) + 
                    mux2*(uSh(3,ith,jth,kthm3)-uSh(3,ith,jth,kthm2)) + 
                    mux3*(uSh(3,ith,jth,kthm1)-uSh(3,ith,jth,kthm2)) +
                    mux4*(uSh(3,ith,jth,kthm0)-uSh(3,ith,jth,kthm2))  )*istrx;

// pq-derivatives
// 38 ops, tot=1075
	    r1 += 
        c2*(  la(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(1,ith+2,jth+2,kthm2)-uSh(1,ith-2,jth+2,kthm2)) +
             c1*(uSh(1,ith+1,jth+2,kthm2)-uSh(1,ith-1,jth+2,kthm2))    )
          - la(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(1,ith+2,jth-2,kthm2)-uSh(1,ith-2,jth-2,kthm2))+
             c1*(uSh(1,ith+1,jth-2,kthm2)-uSh(1,ith-1,jth-2,kthm2))     )
          ) +
        c1*(  la(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(uSh(1,ith+2,jth+1,kthm2)-uSh(1,ith-2,jth+1,kthm2)) +
               c1*(uSh(1,ith+1,jth+1,kthm2)-uSh(1,ith-1,jth+1,kthm2))  )
           - la(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(uSh(1,ith+2,jth-1,kthm2)-uSh(1,ith-2,jth-1,kthm2)) + 
               c1*(uSh(1,ith+1,jth-1,kthm2)-uSh(1,ith-1,jth-1,kthm2))));

// qp-derivatives
// 38 ops, tot=1113
	    r1 += 
        c2*(  mu(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(1,ith+2,jth+2,kthm2)-uSh(1,ith+2,jth-2,kthm2)) +
             c1*(uSh(1,ith+2,jth+1,kthm2)-uSh(1,ith+2,jth-1,kthm2))    )
          - mu(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(1,ith-2,jth+2,kthm2)-uSh(1,ith-2,jth-2,kthm2))+
             c1*(uSh(1,ith-2,jth+1,kthm2)-uSh(1,ith-2,jth-1,kthm2))     )
          ) +
        c1*(  mu(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(uSh(1,ith+1,jth+2,kthm2)-uSh(1,ith+1,jth-2,kthm2)) +
               c1*(uSh(1,ith+1,jth+1,kthm2)-uSh(1,ith+1,jth-1,kthm2))  )
           - mu(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(uSh(1,ith-1,jth+2,kthm2)-uSh(1,ith-1,jth-2,kthm2)) + 
               c1*(uSh(1,ith-1,jth+1,kthm2)-uSh(1,ith-1,jth-1,kthm2))));

// pr-derivatives
// 82 ops, tot=1195
	    r1 += c2*(
       (la(i,j,k+2))*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith-2,jth,kthm0)) +
             c1*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith-1,jth,kthm0))   ) 
        + mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(2,ith+2,jth,kthm0)-uSh(2,ith-2,jth,kthm0)) +
             c1*(uSh(2,ith+1,jth,kthm0)-uSh(2,ith-1,jth,kthm0))  )*strx(i)*istry 
       - ((la(i,j,k-2))*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(1,ith+2,jth,kthm4)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm4)-uSh(1,ith-1,jth,kthm4))  ) 
          + mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(2,ith+2,jth,kthm4)-uSh(2,ith-2,jth,kthm4)) +
             c1*(uSh(2,ith+1,jth,kthm4)-uSh(2,ith-1,jth,kthm4)) )*strx(i)*istry ) 
                  ) + c1*(  
          (la(i,j,k+1))*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith-2,jth,kthm1)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith-1,jth,kthm1)) ) 
          + mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(2,ith+2,jth,kthm1)-uSh(2,ith-2,jth,kthm1)) +
             c1*(uSh(2,ith+1,jth,kthm1)-uSh(2,ith-1,jth,kthm1)) )*strx(i)*istry  
       - (la(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(1,ith+2,jth,kthm3)-uSh(1,ith-2,jth,kthm3)) +
             c1*(uSh(1,ith+1,jth,kthm3)-uSh(1,ith-1,jth,kthm3)) ) 
          + mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(2,ith+2,jth,kthm3)-uSh(2,ith-2,jth,kthm3)) +
             c1*(uSh(2,ith+1,jth,kthm3)-uSh(2,ith-1,jth,kthm3)) )*strx(i)*istry  ) );

// rp derivatives
// 82 ops, tot=1277
	    r1 += c2*(
       (mu(i+2,j,k))*met(3,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith+2,jth,kthm4)) +
             c1*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith+2,jth,kthm3))   ) 
        + mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(2,ith+2,jth,kthm0)-uSh(2,ith+2,jth,kthm4)) +
             c1*(uSh(2,ith+2,jth,kthm1)-uSh(2,ith+2,jth,kthm3))  )*strx(i+2)*istry 
       - (mu(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(1,ith-2,jth,kthm0)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith-2,jth,kthm1)-uSh(1,ith-2,jth,kthm3))  )
          + mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(2,ith-2,jth,kthm0)-uSh(2,ith-2,jth,kthm4)) +
             c1*(uSh(2,ith-2,jth,kthm1)-uSh(2,ith-2,jth,kthm3))   )*strx(i-2)*istry )
                  ) + c1*(  
          (mu(i+1,j,k))*met(3,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith+1,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith+1,jth,kthm3)) ) 
          + mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(2,ith+1,jth,kthm0)-uSh(2,ith+1,jth,kthm4)) +
             c1*(uSh(2,ith+1,jth,kthm1)-uSh(2,ith+1,jth,kthm3)) )*strx(i+1)*istry 
       - (mu(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(1,ith-1,jth,kthm0)-uSh(1,ith-1,jth,kthm4)) +
             c1*(uSh(1,ith-1,jth,kthm1)-uSh(1,ith-1,jth,kthm3)) ) 
          + mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(2,ith-1,jth,kthm0)-uSh(2,ith-1,jth,kthm4)) +
             c1*(uSh(2,ith-1,jth,kthm1)-uSh(2,ith-1,jth,kthm3)) )*strx(i-1)*istry  ) );
	 
// qr derivatives
// 130 ops, tot=1407
	    r1 += c2*(
         mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(1,ith,jth+2,kthm0)-uSh(1,ith,jth-2,kthm0)) +
             c1*(uSh(1,ith,jth+1,kthm0)-uSh(1,ith,jth-1,kthm0))   ) 
        + (2*mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth-2,kthm0)) +
             c1*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth-1,kthm0))  )*stry(j)*istrx 
        +mu(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(3,ith,jth+2,kthm0)-uSh(3,ith,jth-2,kthm0)) +
             c1*(uSh(3,ith,jth+1,kthm0)-uSh(3,ith,jth-1,kthm0))   )*istrx 
       - ( mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(1,ith,jth+2,kthm4)-uSh(1,ith,jth-2,kthm4)) +
             c1*(uSh(1,ith,jth+1,kthm4)-uSh(1,ith,jth-1,kthm4))  ) 
         +(2*mu(i,j,k-2)+ la(i,j,k-2))*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(2,ith,jth+2,kthm4)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm4)-uSh(2,ith,jth-1,kthm4))   )*stry(j)*istrx +
            mu(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(3,ith,jth+2,kthm4)-uSh(3,ith,jth-2,kthm4)) +
             c1*(uSh(3,ith,jth+1,kthm4)-uSh(3,ith,jth-1,kthm4))  )*istrx ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(1,ith,jth+2,kthm1)-uSh(1,ith,jth-2,kthm1)) +
             c1*(uSh(1,ith,jth+1,kthm1)-uSh(1,ith,jth-1,kthm1)) ) 
         + (2*mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth-2,kthm1)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth-1,kthm1)) )*stry(j)*istrx
         + mu(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(3,ith,jth+2,kthm1)-uSh(3,ith,jth-2,kthm1)) +
             c1*(uSh(3,ith,jth+1,kthm1)-uSh(3,ith,jth-1,kthm1)) )*istrx   
       - ( mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(1,ith,jth+2,kthm3)-uSh(1,ith,jth-2,kthm3)) +
             c1*(uSh(1,ith,jth+1,kthm3)-uSh(1,ith,jth-1,kthm3)) ) 
         + (2*mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(2,ith,jth+2,kthm3)-uSh(2,ith,jth-2,kthm3)) +
             c1*(uSh(2,ith,jth+1,kthm3)-uSh(2,ith,jth-1,kthm3)) )*stry(j)*istrx
         +  mu(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(3,ith,jth+2,kthm3)-uSh(3,ith,jth-2,kthm3)) +
             c1*(uSh(3,ith,jth+1,kthm3)-uSh(3,ith,jth-1,kthm3)) )*istrx  ) );
	 
// rq derivatives
// 130 ops, tot=1537
	    r1 += c2*(
         la(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(1,ith,jth+2,kthm0)-uSh(1,ith,jth+2,kthm4)) +
             c1*(uSh(1,ith,jth+2,kthm1)-uSh(1,ith,jth+2,kthm3))   ) 
        +(2*mu(i,j+2,k)+la(i,j+2,k))*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth+2,kthm4)) +
             c1*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth+2,kthm3))  )*stry(j+2)*istrx 
         + la(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(3,ith,jth+2,kthm0)-uSh(3,ith,jth+2,kthm4)) +
             c1*(uSh(3,ith,jth+2,kthm1)-uSh(3,ith,jth+2,kthm3))   )*istrx 
       - ( la(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(1,ith,jth-2,kthm0)-uSh(1,ith,jth-2,kthm4)) +
             c1*(uSh(1,ith,jth-2,kthm1)-uSh(1,ith,jth-2,kthm3))  ) 
          +(2*mu(i,j-2,k)+la(i,j-2,k))*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(2,ith,jth-2,kthm0)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth-2,kthm1)-uSh(2,ith,jth-2,kthm3))   )*stry(j-2)*istrx 
         + la(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(3,ith,jth-2,kthm0)-uSh(3,ith,jth-2,kthm4)) +
             c1*(uSh(3,ith,jth-2,kthm1)-uSh(3,ith,jth-2,kthm3))  )*istrx  ) 
		      ) + c1*(  
           la(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(1,ith,jth+1,kthm0)-uSh(1,ith,jth+1,kthm4)) +
             c1*(uSh(1,ith,jth+1,kthm1)-uSh(1,ith,jth+1,kthm3)) ) 
          + (2*mu(i,j+1,k)+la(i,j+1,k))*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth+1,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth+1,kthm3)) )*stry(j+1)*istrx 
          +la(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(3,ith,jth+1,kthm0)-uSh(3,ith,jth+1,kthm4)) +
             c1*(uSh(3,ith,jth+1,kthm1)-uSh(3,ith,jth+1,kthm3)) )*istrx   
       - ( la(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(1,ith,jth-1,kthm0)-uSh(1,ith,jth-1,kthm4)) +
             c1*(uSh(1,ith,jth-1,kthm1)-uSh(1,ith,jth-1,kthm3)) ) 
          + (2*mu(i,j-1,k)+la(i,j-1,k))*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(2,ith,jth-1,kthm0)-uSh(2,ith,jth-1,kthm4)) +
             c1*(uSh(2,ith,jth-1,kthm1)-uSh(2,ith,jth-1,kthm3)) )*stry(j-1)*istrx
          + la(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(3,ith,jth-1,kthm0)-uSh(3,ith,jth-1,kthm4)) +
             c1*(uSh(3,ith,jth-1,kthm1)-uSh(3,ith,jth-1,kthm3)) )*istrx   ) );

// 4 ops, tot=1541
	    lu(2,i,j,k) = a1*lu(2,i,j,k) + r1*ijac;
	 
// w-equation
	    r1 = 0.;
// pp derivative (w)
// 43 ops, tot=1580
	    cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	    cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	    cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	    cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(3,ith-2,jth,kthm2)-uSh(3,ith,jth,kthm2)) + 
                    mux2*(uSh(3,ith-1,jth,kthm2)-uSh(3,ith,jth,kthm2)) + 
                    mux3*(uSh(3,ith+1,jth,kthm2)-uSh(3,ith,jth,kthm2)) +
                    mux4*(uSh(3,ith+2,jth,kthm2)-uSh(3,ith,jth,kthm2))  )*istry;

// qq derivative (w)
// 43 ops, tot=1623
	    cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	    cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	    cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	    cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(3,ith,jth-2,kthm2)-uSh(3,ith,jth,kthm2)) + 
                    mux2*(uSh(3,ith,jth-1,kthm2)-uSh(3,ith,jth,kthm2)) + 
                    mux3*(uSh(3,ith,jth+1,kthm2)-uSh(3,ith,jth,kthm2)) +
                    mux4*(uSh(3,ith,jth+2,kthm2)-uSh(3,ith,jth,kthm2))  )*istrx;
// rr derivative (u)
// 43 ops, tot=1666
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(4,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(1,ith,jth,kthm4)-uSh(1,ith,jth,kthm2)) + 
                    mux2*(uSh(1,ith,jth,kthm3)-uSh(1,ith,jth,kthm2)) + 
                    mux3*(uSh(1,ith,jth,kthm1)-uSh(1,ith,jth,kthm2)) +
                    mux4*(uSh(1,ith,jth,kthm0)-uSh(1,ith,jth,kthm2))  )*istry;

// rr derivative (v)
// 43 ops, tot=1709
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(3,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(4,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(2,ith,jth,kthm4)-uSh(2,ith,jth,kthm2)) + 
                    mux2*(uSh(2,ith,jth,kthm3)-uSh(2,ith,jth,kthm2)) + 
                    mux3*(uSh(2,ith,jth,kthm1)-uSh(2,ith,jth,kthm2)) +
                    mux4*(uSh(2,ith,jth,kthm0)-uSh(2,ith,jth,kthm2))  )*istrx;

// rr derivative (w)
// 83 ops, tot=1792
	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(4,i,j,k-2)*met(4,i,j,k-2) +
              mu(i,j,k-2)*(met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)+
			   met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j) );
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(4,i,j,k-1)*met(4,i,j,k-1) +
              mu(i,j,k-1)*(met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)+
			   met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j) );
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(4,i,j,k)*met(4,i,j,k) +
              mu(i,j,k)*(met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i)+
			 met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j) );
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(4,i,j,k+1)*met(4,i,j,k+1) +
              mu(i,j,k+1)*(met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)+
			   met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(4,i,j,k+2)*met(4,i,j,k+2) +
              mu(i,j,k+2)*( met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)+
			    met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j) );
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(3,ith,jth,kthm4)-uSh(3,ith,jth,kthm2)) + 
                    mux2*(uSh(3,ith,jth,kthm3)-uSh(3,ith,jth,kthm2)) + 
                    mux3*(uSh(3,ith,jth,kthm1)-uSh(3,ith,jth,kthm2)) +
                    mux4*(uSh(3,ith,jth,kthm0)-uSh(3,ith,jth,kthm2))  )*istrxy
// pr-derivatives
// 86 ops, tot=1878
// r1 += 
          + c2*(
       (la(i,j,k+2))*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith-2,jth,kthm0)) +
             c1*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith-1,jth,kthm0))   )*istry 
        + mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(3,ith+2,jth,kthm0)-uSh(3,ith-2,jth,kthm0)) +
             c1*(uSh(3,ith+1,jth,kthm0)-uSh(3,ith-1,jth,kthm0))  )*strx(i)*istry 
       - ((la(i,j,k-2))*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(1,ith+2,jth,kthm4)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm4)-uSh(1,ith-1,jth,kthm4))  )*istry  
          + mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(3,ith+2,jth,kthm4)-uSh(3,ith-2,jth,kthm4)) +
             c1*(uSh(3,ith+1,jth,kthm4)-uSh(3,ith-1,jth,kthm4)) )*strx(i)*istry ) 
                  ) + c1*(  
          (la(i,j,k+1))*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith-2,jth,kthm1)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith-1,jth,kthm1)) )*istry  
          + mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(3,ith+2,jth,kthm1)-uSh(3,ith-2,jth,kthm1)) +
             c1*(uSh(3,ith+1,jth,kthm1)-uSh(3,ith-1,jth,kthm1)) )*strx(i)*istry  
       - (la(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(1,ith+2,jth,kthm3)-uSh(1,ith-2,jth,kthm3)) +
             c1*(uSh(1,ith+1,jth,kthm3)-uSh(1,ith-1,jth,kthm3)) )*istry  
          + mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(3,ith+2,jth,kthm3)-uSh(3,ith-2,jth,kthm3)) +
             c1*(uSh(3,ith+1,jth,kthm3)-uSh(3,ith-1,jth,kthm3)) )*strx(i)*istry  ) )
// rp derivatives
// 79 ops, tot=1957
//   r1 += 
         + istry*(c2*(
       (mu(i+2,j,k))*met(4,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith+2,jth,kthm4)) +
             c1*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith+2,jth,kthm3))   ) 
        + mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(3,ith+2,jth,kthm0)-uSh(3,ith+2,jth,kthm4)) +
             c1*(uSh(3,ith+2,jth,kthm1)-uSh(3,ith+2,jth,kthm3)) )*strx(i+2) 
       - (mu(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(1,ith-2,jth,kthm0)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith-2,jth,kthm1)-uSh(1,ith-2,jth,kthm3))  )
          + mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(3,ith-2,jth,kthm0)-uSh(3,ith-2,jth,kthm4)) +
             c1*(uSh(3,ith-2,jth,kthm1)-uSh(3,ith-2,jth,kthm3)) )*strx(i-2)  )
                  ) + c1*(  
          (mu(i+1,j,k))*met(4,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith+1,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith+1,jth,kthm3)) ) 
          + mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(3,ith+1,jth,kthm0)-uSh(3,ith+1,jth,kthm4)) +
             c1*(uSh(3,ith+1,jth,kthm1)-uSh(3,ith+1,jth,kthm3)) )*strx(i+1)  
       - (mu(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(1,ith-1,jth,kthm0)-uSh(1,ith-1,jth,kthm4)) +
             c1*(uSh(1,ith-1,jth,kthm1)-uSh(1,ith-1,jth,kthm3)) ) 
          + mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(3,ith-1,jth,kthm0)-uSh(3,ith-1,jth,kthm4)) +
             c1*(uSh(3,ith-1,jth,kthm1)-uSh(3,ith-1,jth,kthm3)) )*strx(i-1)  ) ) )
// qr derivatives
// 86 ops, tot=2043
//     r1 +=
         + c2*(
         mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(3,ith,jth+2,kthm0)-uSh(3,ith,jth-2,kthm0)) +
             c1*(uSh(3,ith,jth+1,kthm0)-uSh(3,ith,jth-1,kthm0))   )*stry(j)*istrx 
        + la(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth-2,kthm0)) +
             c1*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth-1,kthm0))  )*istrx 
       - ( mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(3,ith,jth+2,kthm4)-uSh(3,ith,jth-2,kthm4)) +
             c1*(uSh(3,ith,jth+1,kthm4)-uSh(3,ith,jth-1,kthm4))  )*stry(j)*istrx  
          + la(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(2,ith,jth+2,kthm4)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm4)-uSh(2,ith,jth-1,kthm4))   )*istrx  ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(3,ith,jth+2,kthm1)-uSh(3,ith,jth-2,kthm1)) +
             c1*(uSh(3,ith,jth+1,kthm1)-uSh(3,ith,jth-1,kthm1)) )*stry(j)*istrx  
          + la(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth-2,kthm1)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth-1,kthm1)) )*istrx   
       - ( mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(3,ith,jth+2,kthm3)-uSh(3,ith,jth-2,kthm3)) +
             c1*(uSh(3,ith,jth+1,kthm3)-uSh(3,ith,jth-1,kthm3)) )*stry(j)*istrx  
          + la(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(2,ith,jth+2,kthm3)-uSh(2,ith,jth-2,kthm3)) +
             c1*(uSh(2,ith,jth+1,kthm3)-uSh(2,ith,jth-1,kthm3)) )*istrx  ) )
// rq derivatives
//  79 ops, tot=2122
//  r1 += 
          + istrx*(c2*(
         mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(3,ith,jth+2,kthm0)-uSh(3,ith,jth+2,kthm4)) +
             c1*(uSh(3,ith,jth+2,kthm1)-uSh(3,ith,jth+2,kthm3))   )*stry(j+2) 
        + mu(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth+2,kthm4)) +
             c1*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth+2,kthm3))  ) 
       - ( mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(3,ith,jth-2,kthm0)-uSh(3,ith,jth-2,kthm4)) +
             c1*(uSh(3,ith,jth-2,kthm1)-uSh(3,ith,jth-2,kthm3))  )*stry(j-2) 
          + mu(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(2,ith,jth-2,kthm0)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth-2,kthm1)-uSh(2,ith,jth-2,kthm3))   ) ) 
                  ) + c1*(  
           mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(3,ith,jth+1,kthm0)-uSh(3,ith,jth+1,kthm4)) +
             c1*(uSh(3,ith,jth+1,kthm1)-uSh(3,ith,jth+1,kthm3)) )*stry(j+1) 
          + mu(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth+1,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth+1,kthm3)) )  
       - ( mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(3,ith,jth-1,kthm0)-uSh(3,ith,jth-1,kthm4)) +
             c1*(uSh(3,ith,jth-1,kthm1)-uSh(3,ith,jth-1,kthm3)) )*stry(j-1) 
          + mu(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(2,ith,jth-1,kthm0)-uSh(2,ith,jth-1,kthm4)) +
             c1*(uSh(2,ith,jth-1,kthm1)-uSh(2,ith,jth-1,kthm3)) ) ) ) );
// 4 ops, tot=2126
	    lu(3,i,j,k) = a1*lu(3,i,j,k) + r1*ijac;
	 }
      __syncthreads();

      if (k+RADIUS+1 <= klast)
      {
        for (int tj = threadIdx.y; tj < blockDim.y+2*RADIUS; tj += blockDim.y) {
          int gj =  (j-threadIdx.y) + tj - RADIUS;
          if ( gj <= jlast) {
            for (int ti = threadIdx.x; ti < blockDim.x+2*RADIUS; ti += blockDim.x) {
              int gi =  (i-threadIdx.x) + ti - RADIUS;
              if ( gi <= ilast) {
                uSh(1, ti, tj, kthm4) = u(1, gi, gj, k+RADIUS+1);
                uSh(2, ti, tj, kthm4) = u(2, gi, gj, k+RADIUS+1);
                uSh(3, ti, tj, kthm4) = u(3, gi, gj, k+RADIUS+1);
              }
            }
          }
        }
      }
      __syncthreads();
    }
#undef mu
#undef la
#undef jac
#undef u
#undef lu
#undef met
#undef strx
#undef stry
#undef uSh
}
//-----------------------------------------------------------------------
__global__ void 
__launch_bounds__(256,1)
rhs4sgcurv_dev_rev_v2( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_met, float_sw4* a_jac, float_sw4* a_lu, 
				int onesided4, float_sw4* a_strx, float_sw4* a_stry, int ghost_points )
{
 // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+(i)+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+(i)+ni*(j)+nij*(k)]
#define jac(i,j,k)   a_jac[base+i+ni*(j)+nij*(k)]
#define u(c,i,j,k)   a_u[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define lu(c,i,j,k) a_lu[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define met(c,i,j,k) a_met[base4+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define uSh(c,i,j,k) a_uSh[c-1][k][j][i]
   const float_sw4 a1   = 0;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 tf   = 0.75;
   const float_sw4 c1   = 2.0/3;
   const float_sw4 c2   = -1.0/12;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = nij*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = base-nijk;
   const int base4 = base-nijk;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof;
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int kbegin = kfirst;
//   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;

  __shared__ float_sw4 a_uSh[3][DIAMETER][RHS4_BLOCKY+2*RADIUS][RHS4_BLOCKX+2*RADIUS];
   const int ith = threadIdx.x + RADIUS;
   const int jth = threadIdx.y + RADIUS;
   int kthm0=3, kthm1=2, kthm2=1, kthm3=0, kthm4=4, kthtmp;

   if(onesided4 == 1)
       kbegin = 5;
//     k = 7 + threadIdx.z + blockIdx.z*blockDim.z;

   for (int tk = 0; tk < DIAMETER; tk++)
   {
     int gk =  kbegin + tk;
     if ( gk <= klast)
     {
       for (int tj = threadIdx.y; tj < blockDim.y+2*RADIUS; tj += blockDim.y)
       {
         int gj =  (j-threadIdx.y) + tj - RADIUS;
         if ( gj <= jlast)
         {
           for (int ti = threadIdx.x; ti < blockDim.x+2*RADIUS; ti += blockDim.x)
           {
             int gi =  (i-threadIdx.x) + ti - RADIUS;
             if ( gi <= ilast)
             {
               uSh(1, ti, tj, tk) = u(1, gi, gj, gk);
               uSh(2, ti, tj, tk) = u(2, gi, gj, gk);
               uSh(3, ti, tj, tk) = u(3, gi, gj, gk);
             }
           }
         }
       }
     }
   }
   __syncthreads();

   for (int k = kbegin+RADIUS; k <= klast-RADIUS; k++)
   {
     kthtmp = kthm4;
     kthm4  = kthm3;
     kthm3  = kthm2;
     kthm2  = kthm1;
     kthm1  = kthm0;
     kthm0  = kthtmp;
   
   //if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   if ( (i <= ilast-RADIUS) && (j <= jlast-RADIUS) )
	 {
// 5 ops
	    float_sw4 ijac = strx(i)*stry(j)/jac(i,j,k);
            float_sw4 istry = 1/(stry(j));
            float_sw4 istrx = 1/(strx(i));
            float_sw4 istrxy = istry*istrx;

            float_sw4 r1 = 0.;

	    // pp derivative (u)
// 53 ops, tot=58
	    float_sw4 cof1=(2*mu(i-2,j,k)+la(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)
	       *strx(i-2);
	    float_sw4 cof2=(2*mu(i-1,j,k)+la(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)
	       *strx(i-1);
	    float_sw4 cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)
		  *strx(i);
	    float_sw4 cof4=(2*mu(i+1,j,k)+la(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)
	     *strx(i+1);
	    float_sw4 cof5=(2*mu(i+2,j,k)+la(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)
	     *strx(i+2);
            float_sw4 mux1 = cof2 -tf*(cof3+cof1);
            float_sw4 mux2 = cof1 + cof4+3*(cof3+cof2);
            float_sw4 mux3 = cof2 + cof5+3*(cof4+cof3);
            float_sw4 mux4 = cof4-tf*(cof3+cof5);

            r1 +=  i6* (
                    mux1*(uSh(1,ith-2,jth,kthm2)-uSh(1,ith,jth,kthm2)) + 
                    mux2*(uSh(1,ith-1,jth,kthm2)-uSh(1,ith,jth,kthm2)) + 
                    mux3*(uSh(1,ith+1,jth,kthm2)-uSh(1,ith,jth,kthm2)) +
                    mux4*(uSh(1,ith+2,jth,kthm2)-uSh(1,ith,jth,kthm2))  )*istry;
	    // qq derivative (u)
// 43 ops, tot=101
	    cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	    cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	    cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	    cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(1,ith,jth-2,kthm2)-uSh(1,ith,jth,kthm2)) + 
                    mux2*(uSh(1,ith,jth-1,kthm2)-uSh(1,ith,jth,kthm2)) + 
                    mux3*(uSh(1,ith,jth+1,kthm2)-uSh(1,ith,jth,kthm2)) +
                    mux4*(uSh(1,ith,jth+2,kthm2)-uSh(1,ith,jth,kthm2))  )*istrx;
	    // rr derivative (u)
// 5*11+14+14=83 ops, tot=184
	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)
	       +   mu(i,j,k-2)*(met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j)+
				met(4,i,j,k-2)*met(4,i,j,k-2));
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)
	       +   mu(i,j,k-1)*(met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j)+
				met(4,i,j,k-1)*met(4,i,j,k-1) );
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i) +
	     mu(i,j,k)*(met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j)+
			met(4,i,j,k)*met(4,i,j,k));
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)
	       +   mu(i,j,k+1)*(met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j)+
				met(4,i,j,k+1)*met(4,i,j,k+1));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)
	       +   mu(i,j,k+2)*( met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j)+
				 met(4,i,j,k+2)*met(4,i,j,k+2));

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(1,ith,jth,kthm4)-uSh(1,ith,jth,kthm2)) + 
                    mux2*(uSh(1,ith,jth,kthm3)-uSh(1,ith,jth,kthm2)) + 
                    mux3*(uSh(1,ith,jth,kthm1)-uSh(1,ith,jth,kthm2)) +
                    mux4*(uSh(1,ith,jth,kthm0)-uSh(1,ith,jth,kthm2))  )*istrxy;

	    // rr derivative (v)
	    // 42 ops, tot=226
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(3,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(3,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(3,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(3,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(3,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(2,ith,jth,kthm4)-uSh(2,ith,jth,kthm2)) + 
                    mux2*(uSh(2,ith,jth,kthm3)-uSh(2,ith,jth,kthm2)) + 
                    mux3*(uSh(2,ith,jth,kthm1)-uSh(2,ith,jth,kthm2)) +
                    mux4*(uSh(2,ith,jth,kthm0)-uSh(2,ith,jth,kthm2))  );

	    // rr derivative (w)
// 43 ops, tot=269
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(4,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
		    mux1*(uSh(3,ith,jth,kthm4)-uSh(3,ith,jth,kthm2)) + 
                    mux2*(uSh(3,ith,jth,kthm3)-uSh(3,ith,jth,kthm2)) + 
                    mux3*(uSh(3,ith,jth,kthm1)-uSh(3,ith,jth,kthm2)) +
                    mux4*(uSh(3,ith,jth,kthm0)-uSh(3,ith,jth,kthm2))  )*istry;

	    // pq-derivatives
// 38 ops, tot=307
	    r1 += 
        c2*(  mu(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(2,ith+2,jth+2,kthm2)-uSh(2,ith-2,jth+2,kthm2)) +
             c1*(uSh(2,ith+1,jth+2,kthm2)-uSh(2,ith-1,jth+2,kthm2))    )
          - mu(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(2,ith+2,jth-2,kthm2)-uSh(2,ith-2,jth-2,kthm2))+
             c1*(uSh(2,ith+1,jth-2,kthm2)-uSh(2,ith-1,jth-2,kthm2))     )
          ) +
        c1*(  mu(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(uSh(2,ith+2,jth+1,kthm2)-uSh(2,ith-2,jth+1,kthm2)) +
               c1*(uSh(2,ith+1,jth+1,kthm2)-uSh(2,ith-1,jth+1,kthm2))  )
           - mu(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(uSh(2,ith+2,jth-1,kthm2)-uSh(2,ith-2,jth-1,kthm2)) + 
               c1*(uSh(2,ith+1,jth-1,kthm2)-uSh(2,ith-1,jth-1,kthm2))));

	    // qp-derivatives
// 38 ops, tot=345
	    r1 += 
        c2*(  la(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(2,ith+2,jth+2,kthm2)-uSh(2,ith+2,jth-2,kthm2)) +
             c1*(uSh(2,ith+2,jth+1,kthm2)-uSh(2,ith+2,jth-1,kthm2))    )
          - la(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(2,ith-2,jth+2,kthm2)-uSh(2,ith-2,jth-2,kthm2))+
             c1*(uSh(2,ith-2,jth+1,kthm2)-uSh(2,ith-2,jth-1,kthm2))     )
          ) +
        c1*(  la(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(uSh(2,ith+1,jth+2,kthm2)-uSh(2,ith+1,jth-2,kthm2)) +
               c1*(uSh(2,ith+1,jth+1,kthm2)-uSh(2,ith+1,jth-1,kthm2))  )
           - la(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(uSh(2,ith-1,jth+2,kthm2)-uSh(2,ith-1,jth-2,kthm2)) + 
               c1*(uSh(2,ith-1,jth+1,kthm2)-uSh(2,ith-1,jth-1,kthm2))));

      // pr-derivatives
// 130 ops., tot=475
	    r1 += c2*(
       (2*mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith-2,jth,kthm0)) +
             c1*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith-1,jth,kthm0))   )*strx(i)*istry 
        + mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(2,ith+2,jth,kthm0)-uSh(2,ith-2,jth,kthm0)) +
             c1*(uSh(2,ith+1,jth,kthm0)-uSh(2,ith-1,jth,kthm0))  ) 
        + mu(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(3,ith+2,jth,kthm0)-uSh(3,ith-2,jth,kthm0)) +
             c1*(uSh(3,ith+1,jth,kthm0)-uSh(3,ith-1,jth,kthm0))  )*istry
       - ((2*mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(1,ith+2,jth,kthm4)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm4)-uSh(1,ith-1,jth,kthm4))  )*strx(i)*istry  
          + mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(2,ith+2,jth,kthm4)-uSh(2,ith-2,jth,kthm4)) +
             c1*(uSh(2,ith+1,jth,kthm4)-uSh(2,ith-1,jth,kthm4))   ) 
          + mu(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(3,ith+2,jth,kthm4)-uSh(3,ith-2,jth,kthm4)) +
             c1*(uSh(3,ith+1,jth,kthm4)-uSh(3,ith-1,jth,kthm4))   )*istry )
                  ) + c1*(  
          (2*mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith-2,jth,kthm1)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith-1,jth,kthm1)) )*strx(i)*istry 
          + mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(2,ith+2,jth,kthm1)-uSh(2,ith-2,jth,kthm1)) +
             c1*(uSh(2,ith+1,jth,kthm1)-uSh(2,ith-1,jth,kthm1)) ) 
          + mu(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(3,ith+2,jth,kthm1)-uSh(3,ith-2,jth,kthm1)) +
             c1*(uSh(3,ith+1,jth,kthm1)-uSh(3,ith-1,jth,kthm1))  )*istry
       - ((2*mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(1,ith+2,jth,kthm3)-uSh(1,ith-2,jth,kthm3)) +
             c1*(uSh(1,ith+1,jth,kthm3)-uSh(1,ith-1,jth,kthm3)) )*strx(i)*istry  
          + mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(2,ith+2,jth,kthm3)-uSh(2,ith-2,jth,kthm3)) +
             c1*(uSh(2,ith+1,jth,kthm3)-uSh(2,ith-1,jth,kthm3)) ) 
          + mu(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(3,ith+2,jth,kthm3)-uSh(3,ith-2,jth,kthm3)) +
             c1*(uSh(3,ith+1,jth,kthm3)-uSh(3,ith-1,jth,kthm3))   )*istry  ) );

	    // rp derivatives
// 130 ops, tot=605
	    r1 += ( c2*(
       (2*mu(i+2,j,k)+la(i+2,j,k))*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith+2,jth,kthm4)) +
             c1*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith+2,jth,kthm3))   )*strx(i+2) 
        + la(i+2,j,k)*met(3,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(2,ith+2,jth,kthm0)-uSh(2,ith+2,jth,kthm4)) +
             c1*(uSh(2,ith+2,jth,kthm1)-uSh(2,ith+2,jth,kthm3))  )*stry(j) 
        + la(i+2,j,k)*met(4,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(3,ith+2,jth,kthm0)-uSh(3,ith+2,jth,kthm4)) +
             c1*(uSh(3,ith+2,jth,kthm1)-uSh(3,ith+2,jth,kthm3))  )
       - ((2*mu(i-2,j,k)+la(i-2,j,k))*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(1,ith-2,jth,kthm0)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith-2,jth,kthm1)-uSh(1,ith-2,jth,kthm3))  )*strx(i-2) 
          + la(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(2,ith-2,jth,kthm0)-uSh(2,ith-2,jth,kthm4)) +
             c1*(uSh(2,ith-2,jth,kthm1)-uSh(2,ith-2,jth,kthm3))   )*stry(j) 
          + la(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(3,ith-2,jth,kthm0)-uSh(3,ith-2,jth,kthm4)) +
             c1*(uSh(3,ith-2,jth,kthm1)-uSh(3,ith-2,jth,kthm3))   ) )
                  ) + c1*(  
          (2*mu(i+1,j,k)+la(i+1,j,k))*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith+1,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith+1,jth,kthm3)) )*strx(i+1) 
          + la(i+1,j,k)*met(3,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(2,ith+1,jth,kthm0)-uSh(2,ith+1,jth,kthm4)) +
             c1*(uSh(2,ith+1,jth,kthm1)-uSh(2,ith+1,jth,kthm3)) )*stry(j) 
          + la(i+1,j,k)*met(4,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(3,ith+1,jth,kthm0)-uSh(3,ith+1,jth,kthm4)) +
             c1*(uSh(3,ith+1,jth,kthm1)-uSh(3,ith+1,jth,kthm3))  )
       - ((2*mu(i-1,j,k)+la(i-1,j,k))*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(1,ith-1,jth,kthm0)-uSh(1,ith-1,jth,kthm4)) +
             c1*(uSh(1,ith-1,jth,kthm1)-uSh(1,ith-1,jth,kthm3)) )*strx(i-1) 
          + la(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(2,ith-1,jth,kthm0)-uSh(2,ith-1,jth,kthm4)) +
             c1*(uSh(2,ith-1,jth,kthm1)-uSh(2,ith-1,jth,kthm3)) )*stry(j) 
          + la(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(3,ith-1,jth,kthm0)-uSh(3,ith-1,jth,kthm4)) +
             c1*(uSh(3,ith-1,jth,kthm1)-uSh(3,ith-1,jth,kthm3))   )  ) ) )*istry;

	    // qr derivatives
// 82 ops, tot=687
	    r1 += c2*(
         mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(1,ith,jth+2,kthm0)-uSh(1,ith,jth-2,kthm0)) +
             c1*(uSh(1,ith,jth+1,kthm0)-uSh(1,ith,jth-1,kthm0))   )*stry(j)*istrx 
        + la(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth-2,kthm0)) +
             c1*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth-1,kthm0))  ) 
       - ( mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(1,ith,jth+2,kthm4)-uSh(1,ith,jth-2,kthm4)) +
             c1*(uSh(1,ith,jth+1,kthm4)-uSh(1,ith,jth-1,kthm4))  )*stry(j)*istrx  
          + la(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(2,ith,jth+2,kthm4)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm4)-uSh(2,ith,jth-1,kthm4))   ) ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(1,ith,jth+2,kthm1)-uSh(1,ith,jth-2,kthm1)) +
             c1*(uSh(1,ith,jth+1,kthm1)-uSh(1,ith,jth-1,kthm1)) )*stry(j)*istrx  
          + la(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth-2,kthm1)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth-1,kthm1)) )  
       - ( mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(1,ith,jth+2,kthm3)-uSh(1,ith,jth-2,kthm3)) +
             c1*(uSh(1,ith,jth+1,kthm3)-uSh(1,ith,jth-1,kthm3)) )*stry(j)*istrx  
          + la(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(2,ith,jth+2,kthm3)-uSh(2,ith,jth-2,kthm3)) +
             c1*(uSh(2,ith,jth+1,kthm3)-uSh(2,ith,jth-1,kthm3)) ) ) );

	    // rq derivatives
// 82 ops, tot=769
	    r1 += c2*(
         mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(1,ith,jth+2,kthm0)-uSh(1,ith,jth+2,kthm4)) +
             c1*(uSh(1,ith,jth+2,kthm1)-uSh(1,ith,jth+2,kthm3))   )*stry(j+2)*istrx 
        + mu(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth+2,kthm4)) +
             c1*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth+2,kthm3))  ) 
       - ( mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(1,ith,jth-2,kthm0)-uSh(1,ith,jth-2,kthm4)) +
             c1*(uSh(1,ith,jth-2,kthm1)-uSh(1,ith,jth-2,kthm3))  )*stry(j-2)*istrx  
          + mu(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(2,ith,jth-2,kthm0)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth-2,kthm1)-uSh(2,ith,jth-2,kthm3))   ) ) 
                  ) + c1*(  
           mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(1,ith,jth+1,kthm0)-uSh(1,ith,jth+1,kthm4)) +
             c1*(uSh(1,ith,jth+1,kthm1)-uSh(1,ith,jth+1,kthm3)) )*stry(j+1)*istrx
          + mu(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth+1,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth+1,kthm3)) )
       - ( mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(1,ith,jth-1,kthm0)-uSh(1,ith,jth-1,kthm4)) +
             c1*(uSh(1,ith,jth-1,kthm1)-uSh(1,ith,jth-1,kthm3)) )*stry(j-1)*istrx    
          + mu(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(2,ith,jth-1,kthm0)-uSh(2,ith,jth-1,kthm4)) +
             c1*(uSh(2,ith,jth-1,kthm1)-uSh(2,ith,jth-1,kthm3)) ) ) );

// 4 ops, tot=773
	    lu(1,i,j,k) = a1*lu(1,i,j,k) + r1*ijac;
// v-equation

	    r1 = 0.;
	    // pp derivative (v)
// 43 ops, tot=816
	    cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	    cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	    cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	    cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(2,ith-2,jth,kthm2)-uSh(2,ith,jth,kthm2)) + 
                    mux2*(uSh(2,ith-1,jth,kthm2)-uSh(2,ith,jth,kthm2)) + 
                    mux3*(uSh(2,ith+1,jth,kthm2)-uSh(2,ith,jth,kthm2)) +
                    mux4*(uSh(2,ith+2,jth,kthm2)-uSh(2,ith,jth,kthm2))  )*istry;
// qq derivative (v)
// 53 ops, tot=869
	    cof1=(2*mu(i,j-2,k)+la(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)
	     *stry(j-2);
	    cof2=(2*mu(i,j-1,k)+la(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)
	       *stry(j-1);
	    cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)
	       *stry(j);
	    cof4=(2*mu(i,j+1,k)+la(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)
	       *stry(j+1);
	    cof5=(2*mu(i,j+2,k)+la(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)
	       *stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(2,ith,jth-2,kthm2)-uSh(2,ith,jth,kthm2)) + 
                    mux2*(uSh(2,ith,jth-1,kthm2)-uSh(2,ith,jth,kthm2)) + 
                    mux3*(uSh(2,ith,jth+1,kthm2)-uSh(2,ith,jth,kthm2)) +
                    mux4*(uSh(2,ith,jth+2,kthm2)-uSh(2,ith,jth,kthm2))  )*istrx;

// rr derivative (u)
// 42 ops, tot=911
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(3,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(3,i,j,k-1);
	    cof3=(mu(i,j,k)+  la(i,j,k)  )*met(2,i,j,k)*  met(3,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(3,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(3,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(1,ith,jth,kthm4)-uSh(1,ith,jth,kthm2)) + 
                    mux2*(uSh(1,ith,jth,kthm3)-uSh(1,ith,jth,kthm2)) + 
                    mux3*(uSh(1,ith,jth,kthm1)-uSh(1,ith,jth,kthm2)) +
                    mux4*(uSh(1,ith,jth,kthm0)-uSh(1,ith,jth,kthm2))  );

// rr derivative (v)
// 83 ops, tot=994
 	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j)
	       +    mu(i,j,k-2)*(met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)+
				 met(4,i,j,k-2)*met(4,i,j,k-2));
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j)
	       +    mu(i,j,k-1)*(met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)+
				 met(4,i,j,k-1)*met(4,i,j,k-1));
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j) +
	       mu(i,j,k)*(met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i)+
			  met(4,i,j,k)*met(4,i,j,k));
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j)
	       +    mu(i,j,k+1)*(met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)+
				 met(4,i,j,k+1)*met(4,i,j,k+1));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j)
	       +    mu(i,j,k+2)*(met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)+
				 met(4,i,j,k+2)*met(4,i,j,k+2));

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(2,ith,jth,kthm4)-uSh(2,ith,jth,kthm2)) + 
                    mux2*(uSh(2,ith,jth,kthm3)-uSh(2,ith,jth,kthm2)) + 
                    mux3*(uSh(2,ith,jth,kthm1)-uSh(2,ith,jth,kthm2)) +
                    mux4*(uSh(2,ith,jth,kthm0)-uSh(2,ith,jth,kthm2))  )*istrxy;

// rr derivative (w)
// 43 ops, tot=1037
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)  +la(i,j,k)  )*met(3,i,j,k)*  met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(4,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(3,ith,jth,kthm4)-uSh(3,ith,jth,kthm2)) + 
                    mux2*(uSh(3,ith,jth,kthm3)-uSh(3,ith,jth,kthm2)) + 
                    mux3*(uSh(3,ith,jth,kthm1)-uSh(3,ith,jth,kthm2)) +
                    mux4*(uSh(3,ith,jth,kthm0)-uSh(3,ith,jth,kthm2))  )*istrx;

// pq-derivatives
// 38 ops, tot=1075
	    r1 += 
        c2*(  la(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(1,ith+2,jth+2,kthm2)-uSh(1,ith-2,jth+2,kthm2)) +
             c1*(uSh(1,ith+1,jth+2,kthm2)-uSh(1,ith-1,jth+2,kthm2))    )
          - la(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(1,ith+2,jth-2,kthm2)-uSh(1,ith-2,jth-2,kthm2))+
             c1*(uSh(1,ith+1,jth-2,kthm2)-uSh(1,ith-1,jth-2,kthm2))     )
          ) +
        c1*(  la(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(uSh(1,ith+2,jth+1,kthm2)-uSh(1,ith-2,jth+1,kthm2)) +
               c1*(uSh(1,ith+1,jth+1,kthm2)-uSh(1,ith-1,jth+1,kthm2))  )
           - la(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(uSh(1,ith+2,jth-1,kthm2)-uSh(1,ith-2,jth-1,kthm2)) + 
               c1*(uSh(1,ith+1,jth-1,kthm2)-uSh(1,ith-1,jth-1,kthm2))));

// qp-derivatives
// 38 ops, tot=1113
	    r1 += 
        c2*(  mu(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(1,ith+2,jth+2,kthm2)-uSh(1,ith+2,jth-2,kthm2)) +
             c1*(uSh(1,ith+2,jth+1,kthm2)-uSh(1,ith+2,jth-1,kthm2))    )
          - mu(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(1,ith-2,jth+2,kthm2)-uSh(1,ith-2,jth-2,kthm2))+
             c1*(uSh(1,ith-2,jth+1,kthm2)-uSh(1,ith-2,jth-1,kthm2))     )
          ) +
        c1*(  mu(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(uSh(1,ith+1,jth+2,kthm2)-uSh(1,ith+1,jth-2,kthm2)) +
               c1*(uSh(1,ith+1,jth+1,kthm2)-uSh(1,ith+1,jth-1,kthm2))  )
           - mu(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(uSh(1,ith-1,jth+2,kthm2)-uSh(1,ith-1,jth-2,kthm2)) + 
               c1*(uSh(1,ith-1,jth+1,kthm2)-uSh(1,ith-1,jth-1,kthm2))));

// pr-derivatives
// 82 ops, tot=1195
	    r1 += c2*(
       (la(i,j,k+2))*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith-2,jth,kthm0)) +
             c1*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith-1,jth,kthm0))   ) 
        + mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(2,ith+2,jth,kthm0)-uSh(2,ith-2,jth,kthm0)) +
             c1*(uSh(2,ith+1,jth,kthm0)-uSh(2,ith-1,jth,kthm0))  )*strx(i)*istry 
       - ((la(i,j,k-2))*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(1,ith+2,jth,kthm4)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm4)-uSh(1,ith-1,jth,kthm4))  ) 
          + mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(2,ith+2,jth,kthm4)-uSh(2,ith-2,jth,kthm4)) +
             c1*(uSh(2,ith+1,jth,kthm4)-uSh(2,ith-1,jth,kthm4)) )*strx(i)*istry ) 
                  ) + c1*(  
          (la(i,j,k+1))*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith-2,jth,kthm1)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith-1,jth,kthm1)) ) 
          + mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(2,ith+2,jth,kthm1)-uSh(2,ith-2,jth,kthm1)) +
             c1*(uSh(2,ith+1,jth,kthm1)-uSh(2,ith-1,jth,kthm1)) )*strx(i)*istry  
       - (la(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(1,ith+2,jth,kthm3)-uSh(1,ith-2,jth,kthm3)) +
             c1*(uSh(1,ith+1,jth,kthm3)-uSh(1,ith-1,jth,kthm3)) ) 
          + mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(2,ith+2,jth,kthm3)-uSh(2,ith-2,jth,kthm3)) +
             c1*(uSh(2,ith+1,jth,kthm3)-uSh(2,ith-1,jth,kthm3)) )*strx(i)*istry  ) );

// rp derivatives
// 82 ops, tot=1277
	    r1 += c2*(
       (mu(i+2,j,k))*met(3,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith+2,jth,kthm4)) +
             c1*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith+2,jth,kthm3))   ) 
        + mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(2,ith+2,jth,kthm0)-uSh(2,ith+2,jth,kthm4)) +
             c1*(uSh(2,ith+2,jth,kthm1)-uSh(2,ith+2,jth,kthm3))  )*strx(i+2)*istry 
       - (mu(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(1,ith-2,jth,kthm0)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith-2,jth,kthm1)-uSh(1,ith-2,jth,kthm3))  )
          + mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(2,ith-2,jth,kthm0)-uSh(2,ith-2,jth,kthm4)) +
             c1*(uSh(2,ith-2,jth,kthm1)-uSh(2,ith-2,jth,kthm3))   )*strx(i-2)*istry )
                  ) + c1*(  
          (mu(i+1,j,k))*met(3,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith+1,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith+1,jth,kthm3)) ) 
          + mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(2,ith+1,jth,kthm0)-uSh(2,ith+1,jth,kthm4)) +
             c1*(uSh(2,ith+1,jth,kthm1)-uSh(2,ith+1,jth,kthm3)) )*strx(i+1)*istry 
       - (mu(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(1,ith-1,jth,kthm0)-uSh(1,ith-1,jth,kthm4)) +
             c1*(uSh(1,ith-1,jth,kthm1)-uSh(1,ith-1,jth,kthm3)) ) 
          + mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(2,ith-1,jth,kthm0)-uSh(2,ith-1,jth,kthm4)) +
             c1*(uSh(2,ith-1,jth,kthm1)-uSh(2,ith-1,jth,kthm3)) )*strx(i-1)*istry  ) );
	 
// qr derivatives
// 130 ops, tot=1407
	    r1 += c2*(
         mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(1,ith,jth+2,kthm0)-uSh(1,ith,jth-2,kthm0)) +
             c1*(uSh(1,ith,jth+1,kthm0)-uSh(1,ith,jth-1,kthm0))   ) 
        + (2*mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth-2,kthm0)) +
             c1*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth-1,kthm0))  )*stry(j)*istrx 
        +mu(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(3,ith,jth+2,kthm0)-uSh(3,ith,jth-2,kthm0)) +
             c1*(uSh(3,ith,jth+1,kthm0)-uSh(3,ith,jth-1,kthm0))   )*istrx 
       - ( mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(1,ith,jth+2,kthm4)-uSh(1,ith,jth-2,kthm4)) +
             c1*(uSh(1,ith,jth+1,kthm4)-uSh(1,ith,jth-1,kthm4))  ) 
         +(2*mu(i,j,k-2)+ la(i,j,k-2))*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(2,ith,jth+2,kthm4)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm4)-uSh(2,ith,jth-1,kthm4))   )*stry(j)*istrx +
            mu(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(3,ith,jth+2,kthm4)-uSh(3,ith,jth-2,kthm4)) +
             c1*(uSh(3,ith,jth+1,kthm4)-uSh(3,ith,jth-1,kthm4))  )*istrx ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(1,ith,jth+2,kthm1)-uSh(1,ith,jth-2,kthm1)) +
             c1*(uSh(1,ith,jth+1,kthm1)-uSh(1,ith,jth-1,kthm1)) ) 
         + (2*mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth-2,kthm1)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth-1,kthm1)) )*stry(j)*istrx
         + mu(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(3,ith,jth+2,kthm1)-uSh(3,ith,jth-2,kthm1)) +
             c1*(uSh(3,ith,jth+1,kthm1)-uSh(3,ith,jth-1,kthm1)) )*istrx   
       - ( mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(1,ith,jth+2,kthm3)-uSh(1,ith,jth-2,kthm3)) +
             c1*(uSh(1,ith,jth+1,kthm3)-uSh(1,ith,jth-1,kthm3)) ) 
         + (2*mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(2,ith,jth+2,kthm3)-uSh(2,ith,jth-2,kthm3)) +
             c1*(uSh(2,ith,jth+1,kthm3)-uSh(2,ith,jth-1,kthm3)) )*stry(j)*istrx
         +  mu(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(3,ith,jth+2,kthm3)-uSh(3,ith,jth-2,kthm3)) +
             c1*(uSh(3,ith,jth+1,kthm3)-uSh(3,ith,jth-1,kthm3)) )*istrx  ) );
	 
// rq derivatives
// 130 ops, tot=1537
	    r1 += c2*(
         la(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(1,ith,jth+2,kthm0)-uSh(1,ith,jth+2,kthm4)) +
             c1*(uSh(1,ith,jth+2,kthm1)-uSh(1,ith,jth+2,kthm3))   ) 
        +(2*mu(i,j+2,k)+la(i,j+2,k))*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth+2,kthm4)) +
             c1*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth+2,kthm3))  )*stry(j+2)*istrx 
         + la(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(3,ith,jth+2,kthm0)-uSh(3,ith,jth+2,kthm4)) +
             c1*(uSh(3,ith,jth+2,kthm1)-uSh(3,ith,jth+2,kthm3))   )*istrx 
       - ( la(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(1,ith,jth-2,kthm0)-uSh(1,ith,jth-2,kthm4)) +
             c1*(uSh(1,ith,jth-2,kthm1)-uSh(1,ith,jth-2,kthm3))  ) 
          +(2*mu(i,j-2,k)+la(i,j-2,k))*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(2,ith,jth-2,kthm0)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth-2,kthm1)-uSh(2,ith,jth-2,kthm3))   )*stry(j-2)*istrx 
         + la(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(3,ith,jth-2,kthm0)-uSh(3,ith,jth-2,kthm4)) +
             c1*(uSh(3,ith,jth-2,kthm1)-uSh(3,ith,jth-2,kthm3))  )*istrx  ) 
		      ) + c1*(  
           la(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(1,ith,jth+1,kthm0)-uSh(1,ith,jth+1,kthm4)) +
             c1*(uSh(1,ith,jth+1,kthm1)-uSh(1,ith,jth+1,kthm3)) ) 
          + (2*mu(i,j+1,k)+la(i,j+1,k))*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth+1,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth+1,kthm3)) )*stry(j+1)*istrx 
          +la(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(3,ith,jth+1,kthm0)-uSh(3,ith,jth+1,kthm4)) +
             c1*(uSh(3,ith,jth+1,kthm1)-uSh(3,ith,jth+1,kthm3)) )*istrx   
       - ( la(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(1,ith,jth-1,kthm0)-uSh(1,ith,jth-1,kthm4)) +
             c1*(uSh(1,ith,jth-1,kthm1)-uSh(1,ith,jth-1,kthm3)) ) 
          + (2*mu(i,j-1,k)+la(i,j-1,k))*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(2,ith,jth-1,kthm0)-uSh(2,ith,jth-1,kthm4)) +
             c1*(uSh(2,ith,jth-1,kthm1)-uSh(2,ith,jth-1,kthm3)) )*stry(j-1)*istrx
          + la(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(3,ith,jth-1,kthm0)-uSh(3,ith,jth-1,kthm4)) +
             c1*(uSh(3,ith,jth-1,kthm1)-uSh(3,ith,jth-1,kthm3)) )*istrx   ) );

// 4 ops, tot=1541
	    lu(2,i,j,k) = a1*lu(2,i,j,k) + r1*ijac;
	 
// w-equation
	    r1 = 0.;
// pp derivative (w)
// 43 ops, tot=1580
	    cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	    cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	    cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	    cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(3,ith-2,jth,kthm2)-uSh(3,ith,jth,kthm2)) + 
                    mux2*(uSh(3,ith-1,jth,kthm2)-uSh(3,ith,jth,kthm2)) + 
                    mux3*(uSh(3,ith+1,jth,kthm2)-uSh(3,ith,jth,kthm2)) +
                    mux4*(uSh(3,ith+2,jth,kthm2)-uSh(3,ith,jth,kthm2))  )*istry;

// qq derivative (w)
// 43 ops, tot=1623
	    cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	    cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	    cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	    cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(3,ith,jth-2,kthm2)-uSh(3,ith,jth,kthm2)) + 
                    mux2*(uSh(3,ith,jth-1,kthm2)-uSh(3,ith,jth,kthm2)) + 
                    mux3*(uSh(3,ith,jth+1,kthm2)-uSh(3,ith,jth,kthm2)) +
                    mux4*(uSh(3,ith,jth+2,kthm2)-uSh(3,ith,jth,kthm2))  )*istrx;
// rr derivative (u)
// 43 ops, tot=1666
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(4,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(1,ith,jth,kthm4)-uSh(1,ith,jth,kthm2)) + 
                    mux2*(uSh(1,ith,jth,kthm3)-uSh(1,ith,jth,kthm2)) + 
                    mux3*(uSh(1,ith,jth,kthm1)-uSh(1,ith,jth,kthm2)) +
                    mux4*(uSh(1,ith,jth,kthm0)-uSh(1,ith,jth,kthm2))  )*istry;

// rr derivative (v)
// 43 ops, tot=1709
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(3,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(4,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(2,ith,jth,kthm4)-uSh(2,ith,jth,kthm2)) + 
                    mux2*(uSh(2,ith,jth,kthm3)-uSh(2,ith,jth,kthm2)) + 
                    mux3*(uSh(2,ith,jth,kthm1)-uSh(2,ith,jth,kthm2)) +
                    mux4*(uSh(2,ith,jth,kthm0)-uSh(2,ith,jth,kthm2))  )*istrx;

// rr derivative (w)
// 83 ops, tot=1792
	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(4,i,j,k-2)*met(4,i,j,k-2) +
              mu(i,j,k-2)*(met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)+
			   met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j) );
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(4,i,j,k-1)*met(4,i,j,k-1) +
              mu(i,j,k-1)*(met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)+
			   met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j) );
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(4,i,j,k)*met(4,i,j,k) +
              mu(i,j,k)*(met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i)+
			 met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j) );
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(4,i,j,k+1)*met(4,i,j,k+1) +
              mu(i,j,k+1)*(met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)+
			   met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(4,i,j,k+2)*met(4,i,j,k+2) +
              mu(i,j,k+2)*( met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)+
			    met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j) );
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(uSh(3,ith,jth,kthm4)-uSh(3,ith,jth,kthm2)) + 
                    mux2*(uSh(3,ith,jth,kthm3)-uSh(3,ith,jth,kthm2)) + 
                    mux3*(uSh(3,ith,jth,kthm1)-uSh(3,ith,jth,kthm2)) +
                    mux4*(uSh(3,ith,jth,kthm0)-uSh(3,ith,jth,kthm2))  )*istrxy
// pr-derivatives
// 86 ops, tot=1878
// r1 += 
          + c2*(
       (la(i,j,k+2))*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith-2,jth,kthm0)) +
             c1*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith-1,jth,kthm0))   )*istry 
        + mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(3,ith+2,jth,kthm0)-uSh(3,ith-2,jth,kthm0)) +
             c1*(uSh(3,ith+1,jth,kthm0)-uSh(3,ith-1,jth,kthm0))  )*strx(i)*istry 
       - ((la(i,j,k-2))*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(1,ith+2,jth,kthm4)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm4)-uSh(1,ith-1,jth,kthm4))  )*istry  
          + mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(3,ith+2,jth,kthm4)-uSh(3,ith-2,jth,kthm4)) +
             c1*(uSh(3,ith+1,jth,kthm4)-uSh(3,ith-1,jth,kthm4)) )*strx(i)*istry ) 
                  ) + c1*(  
          (la(i,j,k+1))*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith-2,jth,kthm1)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith-1,jth,kthm1)) )*istry  
          + mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(3,ith+2,jth,kthm1)-uSh(3,ith-2,jth,kthm1)) +
             c1*(uSh(3,ith+1,jth,kthm1)-uSh(3,ith-1,jth,kthm1)) )*strx(i)*istry  
       - (la(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(1,ith+2,jth,kthm3)-uSh(1,ith-2,jth,kthm3)) +
             c1*(uSh(1,ith+1,jth,kthm3)-uSh(1,ith-1,jth,kthm3)) )*istry  
          + mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(3,ith+2,jth,kthm3)-uSh(3,ith-2,jth,kthm3)) +
             c1*(uSh(3,ith+1,jth,kthm3)-uSh(3,ith-1,jth,kthm3)) )*strx(i)*istry  ) )
// rp derivatives
// 79 ops, tot=1957
//   r1 += 
         + istry*(c2*(
       (mu(i+2,j,k))*met(4,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(1,ith+2,jth,kthm0)-uSh(1,ith+2,jth,kthm4)) +
             c1*(uSh(1,ith+2,jth,kthm1)-uSh(1,ith+2,jth,kthm3))   ) 
        + mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(uSh(3,ith+2,jth,kthm0)-uSh(3,ith+2,jth,kthm4)) +
             c1*(uSh(3,ith+2,jth,kthm1)-uSh(3,ith+2,jth,kthm3)) )*strx(i+2) 
       - (mu(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(1,ith-2,jth,kthm0)-uSh(1,ith-2,jth,kthm4)) +
             c1*(uSh(1,ith-2,jth,kthm1)-uSh(1,ith-2,jth,kthm3))  )
          + mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(uSh(3,ith-2,jth,kthm0)-uSh(3,ith-2,jth,kthm4)) +
             c1*(uSh(3,ith-2,jth,kthm1)-uSh(3,ith-2,jth,kthm3)) )*strx(i-2)  )
                  ) + c1*(  
          (mu(i+1,j,k))*met(4,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(1,ith+1,jth,kthm0)-uSh(1,ith+1,jth,kthm4)) +
             c1*(uSh(1,ith+1,jth,kthm1)-uSh(1,ith+1,jth,kthm3)) ) 
          + mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(uSh(3,ith+1,jth,kthm0)-uSh(3,ith+1,jth,kthm4)) +
             c1*(uSh(3,ith+1,jth,kthm1)-uSh(3,ith+1,jth,kthm3)) )*strx(i+1)  
       - (mu(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(1,ith-1,jth,kthm0)-uSh(1,ith-1,jth,kthm4)) +
             c1*(uSh(1,ith-1,jth,kthm1)-uSh(1,ith-1,jth,kthm3)) ) 
          + mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(uSh(3,ith-1,jth,kthm0)-uSh(3,ith-1,jth,kthm4)) +
             c1*(uSh(3,ith-1,jth,kthm1)-uSh(3,ith-1,jth,kthm3)) )*strx(i-1)  ) ) )
// qr derivatives
// 86 ops, tot=2043
//     r1 +=
         + c2*(
         mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(3,ith,jth+2,kthm0)-uSh(3,ith,jth-2,kthm0)) +
             c1*(uSh(3,ith,jth+1,kthm0)-uSh(3,ith,jth-1,kthm0))   )*stry(j)*istrx 
        + la(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth-2,kthm0)) +
             c1*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth-1,kthm0))  )*istrx 
       - ( mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(3,ith,jth+2,kthm4)-uSh(3,ith,jth-2,kthm4)) +
             c1*(uSh(3,ith,jth+1,kthm4)-uSh(3,ith,jth-1,kthm4))  )*stry(j)*istrx  
          + la(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(uSh(2,ith,jth+2,kthm4)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm4)-uSh(2,ith,jth-1,kthm4))   )*istrx  ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(3,ith,jth+2,kthm1)-uSh(3,ith,jth-2,kthm1)) +
             c1*(uSh(3,ith,jth+1,kthm1)-uSh(3,ith,jth-1,kthm1)) )*stry(j)*istrx  
          + la(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth-2,kthm1)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth-1,kthm1)) )*istrx   
       - ( mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(3,ith,jth+2,kthm3)-uSh(3,ith,jth-2,kthm3)) +
             c1*(uSh(3,ith,jth+1,kthm3)-uSh(3,ith,jth-1,kthm3)) )*stry(j)*istrx  
          + la(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(uSh(2,ith,jth+2,kthm3)-uSh(2,ith,jth-2,kthm3)) +
             c1*(uSh(2,ith,jth+1,kthm3)-uSh(2,ith,jth-1,kthm3)) )*istrx  ) )
// rq derivatives
//  79 ops, tot=2122
//  r1 += 
          + istrx*(c2*(
         mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(3,ith,jth+2,kthm0)-uSh(3,ith,jth+2,kthm4)) +
             c1*(uSh(3,ith,jth+2,kthm1)-uSh(3,ith,jth+2,kthm3))   )*stry(j+2) 
        + mu(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(uSh(2,ith,jth+2,kthm0)-uSh(2,ith,jth+2,kthm4)) +
             c1*(uSh(2,ith,jth+2,kthm1)-uSh(2,ith,jth+2,kthm3))  ) 
       - ( mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(3,ith,jth-2,kthm0)-uSh(3,ith,jth-2,kthm4)) +
             c1*(uSh(3,ith,jth-2,kthm1)-uSh(3,ith,jth-2,kthm3))  )*stry(j-2) 
          + mu(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(uSh(2,ith,jth-2,kthm0)-uSh(2,ith,jth-2,kthm4)) +
             c1*(uSh(2,ith,jth-2,kthm1)-uSh(2,ith,jth-2,kthm3))   ) ) 
                  ) + c1*(  
           mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(3,ith,jth+1,kthm0)-uSh(3,ith,jth+1,kthm4)) +
             c1*(uSh(3,ith,jth+1,kthm1)-uSh(3,ith,jth+1,kthm3)) )*stry(j+1) 
          + mu(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(uSh(2,ith,jth+1,kthm0)-uSh(2,ith,jth+1,kthm4)) +
             c1*(uSh(2,ith,jth+1,kthm1)-uSh(2,ith,jth+1,kthm3)) )  
       - ( mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(3,ith,jth-1,kthm0)-uSh(3,ith,jth-1,kthm4)) +
             c1*(uSh(3,ith,jth-1,kthm1)-uSh(3,ith,jth-1,kthm3)) )*stry(j-1) 
          + mu(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(uSh(2,ith,jth-1,kthm0)-uSh(2,ith,jth-1,kthm4)) +
             c1*(uSh(2,ith,jth-1,kthm1)-uSh(2,ith,jth-1,kthm3)) ) ) ) );
// 4 ops, tot=2126
	    lu(3,i,j,k) = a1*lu(3,i,j,k) + r1*ijac;
	 }
      __syncthreads();

      if (k+RADIUS+1 <= klast)
      {
        for (int tj = threadIdx.y; tj < blockDim.y+2*RADIUS; tj += blockDim.y) {
          int gj =  (j-threadIdx.y) + tj - RADIUS;
          if ( gj <= jlast) {
            for (int ti = threadIdx.x; ti < blockDim.x+2*RADIUS; ti += blockDim.x) {
              int gi =  (i-threadIdx.x) + ti - RADIUS;
              if ( gi <= ilast) {
                uSh(1, ti, tj, kthm4) = u(1, gi, gj, k+RADIUS+1);
                uSh(2, ti, tj, kthm4) = u(2, gi, gj, k+RADIUS+1);
                uSh(3, ti, tj, kthm4) = u(3, gi, gj, k+RADIUS+1);
              }
            }
          }
        }
      }
      __syncthreads();
    }
#undef mu
#undef la
#undef jac
#undef u
#undef lu
#undef met
#undef strx
#undef stry
#undef uSh
}
//-----------------------------------------------------------------------
__global__ void rhs4sgcurv_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_met, float_sw4* a_jac, float_sw4* a_lu, 
				int onesided4, float_sw4* a_strx, float_sw4* a_stry, int ghost_points )
{
 // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+(i)+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+(i)+ni*(j)+nij*(k)]
#define jac(i,j,k)   a_jac[base+i+ni*(j)+nij*(k)]
#define u(c,i,j,k)   a_u[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define lu(c,i,j,k) a_lu[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define met(c,i,j,k) a_met[base4+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
   const float_sw4 a1   = 0;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 tf   = 0.75;
   const float_sw4 c1   = 2.0/3;
   const float_sw4 c2   = -1.0/12;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = nij*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = base-nijk;
   const int base4 = base-nijk;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof;
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if(onesided4 == 1)
     k = 7 + threadIdx.z + blockIdx.z*blockDim.z;

   if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
	 {
// 5 ops
	    float_sw4 ijac = strx(i)*stry(j)/jac(i,j,k);
            float_sw4 istry = 1/(stry(j));
            float_sw4 istrx = 1/(strx(i));
            float_sw4 istrxy = istry*istrx;

            float_sw4 r1 = 0.;

	    // pp derivative (u)
// 53 ops, tot=58
	    float_sw4 cof1=(2*mu(i-2,j,k)+la(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)
	       *strx(i-2);
	    float_sw4 cof2=(2*mu(i-1,j,k)+la(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)
	       *strx(i-1);
	    float_sw4 cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)
		  *strx(i);
	    float_sw4 cof4=(2*mu(i+1,j,k)+la(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)
	     *strx(i+1);
	    float_sw4 cof5=(2*mu(i+2,j,k)+la(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)
	     *strx(i+2);
            float_sw4 mux1 = cof2 -tf*(cof3+cof1);
            float_sw4 mux2 = cof1 + cof4+3*(cof3+cof2);
            float_sw4 mux3 = cof2 + cof5+3*(cof4+cof3);
            float_sw4 mux4 = cof4-tf*(cof3+cof5);

            r1 +=  i6* (
                    mux1*(u(1,i-2,j,k)-u(1,i,j,k)) + 
                    mux2*(u(1,i-1,j,k)-u(1,i,j,k)) + 
                    mux3*(u(1,i+1,j,k)-u(1,i,j,k)) +
                    mux4*(u(1,i+2,j,k)-u(1,i,j,k))  )*istry;
	    // qq derivative (u)
// 43 ops, tot=101
	    cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	    cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	    cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	    cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                    mux2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                    mux3*(u(1,i,j+1,k)-u(1,i,j,k)) +
                    mux4*(u(1,i,j+2,k)-u(1,i,j,k))  )*istrx;
	    // rr derivative (u)
// 5*11+14+14=83 ops, tot=184
	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)
	       +   mu(i,j,k-2)*(met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j)+
				met(4,i,j,k-2)*met(4,i,j,k-2));
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)
	       +   mu(i,j,k-1)*(met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j)+
				met(4,i,j,k-1)*met(4,i,j,k-1) );
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i) +
	     mu(i,j,k)*(met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j)+
			met(4,i,j,k)*met(4,i,j,k));
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)
	       +   mu(i,j,k+1)*(met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j)+
				met(4,i,j,k+1)*met(4,i,j,k+1));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)
	       +   mu(i,j,k+2)*( met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j)+
				 met(4,i,j,k+2)*met(4,i,j,k+2));

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(1,i,j,k-2)-u(1,i,j,k)) + 
                    mux2*(u(1,i,j,k-1)-u(1,i,j,k)) + 
                    mux3*(u(1,i,j,k+1)-u(1,i,j,k)) +
                    mux4*(u(1,i,j,k+2)-u(1,i,j,k))  )*istrxy;

	    // rr derivative (v)
	    // 42 ops, tot=226
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(3,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(3,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(3,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(3,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(3,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(2,i,j,k-2)-u(2,i,j,k)) + 
                    mux2*(u(2,i,j,k-1)-u(2,i,j,k)) + 
                    mux3*(u(2,i,j,k+1)-u(2,i,j,k)) +
                    mux4*(u(2,i,j,k+2)-u(2,i,j,k))  );

	    // rr derivative (w)
// 43 ops, tot=269
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(4,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
		    mux1*(u(3,i,j,k-2)-u(3,i,j,k)) + 
                    mux2*(u(3,i,j,k-1)-u(3,i,j,k)) + 
                    mux3*(u(3,i,j,k+1)-u(3,i,j,k)) +
                    mux4*(u(3,i,j,k+2)-u(3,i,j,k))  )*istry;

	    // pq-derivatives
// 38 ops, tot=307
	    r1 += 
        c2*(  mu(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(2,i+2,j+2,k)-u(2,i-2,j+2,k)) +
             c1*(u(2,i+1,j+2,k)-u(2,i-1,j+2,k))    )
          - mu(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(2,i+2,j-2,k)-u(2,i-2,j-2,k))+
             c1*(u(2,i+1,j-2,k)-u(2,i-1,j-2,k))     )
          ) +
        c1*(  mu(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(u(2,i+2,j+1,k)-u(2,i-2,j+1,k)) +
               c1*(u(2,i+1,j+1,k)-u(2,i-1,j+1,k))  )
           - mu(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(u(2,i+2,j-1,k)-u(2,i-2,j-1,k)) + 
               c1*(u(2,i+1,j-1,k)-u(2,i-1,j-1,k))));

	    // qp-derivatives
// 38 ops, tot=345
	    r1 += 
        c2*(  la(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(2,i+2,j+2,k)-u(2,i+2,j-2,k)) +
             c1*(u(2,i+2,j+1,k)-u(2,i+2,j-1,k))    )
          - la(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(2,i-2,j+2,k)-u(2,i-2,j-2,k))+
             c1*(u(2,i-2,j+1,k)-u(2,i-2,j-1,k))     )
          ) +
        c1*(  la(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(u(2,i+1,j+2,k)-u(2,i+1,j-2,k)) +
               c1*(u(2,i+1,j+1,k)-u(2,i+1,j-1,k))  )
           - la(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(u(2,i-1,j+2,k)-u(2,i-1,j-2,k)) + 
               c1*(u(2,i-1,j+1,k)-u(2,i-1,j-1,k))));

      // pr-derivatives
// 130 ops., tot=475
	    r1 += c2*(
       (2*mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(1,i+2,j,k+2)-u(1,i-2,j,k+2)) +
             c1*(u(1,i+1,j,k+2)-u(1,i-1,j,k+2))   )*strx(i)*istry 
        + mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(2,i+2,j,k+2)-u(2,i-2,j,k+2)) +
             c1*(u(2,i+1,j,k+2)-u(2,i-1,j,k+2))  ) 
        + mu(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(3,i+2,j,k+2)-u(3,i-2,j,k+2)) +
             c1*(u(3,i+1,j,k+2)-u(3,i-1,j,k+2))  )*istry
       - ((2*mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(1,i+2,j,k-2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i+1,j,k-2)-u(1,i-1,j,k-2))  )*strx(i)*istry  
          + mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(2,i+2,j,k-2)-u(2,i-2,j,k-2)) +
             c1*(u(2,i+1,j,k-2)-u(2,i-1,j,k-2))   ) 
          + mu(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(3,i+2,j,k-2)-u(3,i-2,j,k-2)) +
             c1*(u(3,i+1,j,k-2)-u(3,i-1,j,k-2))   )*istry )
                  ) + c1*(  
          (2*mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(1,i+2,j,k+1)-u(1,i-2,j,k+1)) +
             c1*(u(1,i+1,j,k+1)-u(1,i-1,j,k+1)) )*strx(i)*istry 
          + mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(2,i+2,j,k+1)-u(2,i-2,j,k+1)) +
             c1*(u(2,i+1,j,k+1)-u(2,i-1,j,k+1)) ) 
          + mu(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(3,i+2,j,k+1)-u(3,i-2,j,k+1)) +
             c1*(u(3,i+1,j,k+1)-u(3,i-1,j,k+1))  )*istry
       - ((2*mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(1,i+2,j,k-1)-u(1,i-2,j,k-1)) +
             c1*(u(1,i+1,j,k-1)-u(1,i-1,j,k-1)) )*strx(i)*istry  
          + mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(2,i+2,j,k-1)-u(2,i-2,j,k-1)) +
             c1*(u(2,i+1,j,k-1)-u(2,i-1,j,k-1)) ) 
          + mu(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(3,i+2,j,k-1)-u(3,i-2,j,k-1)) +
             c1*(u(3,i+1,j,k-1)-u(3,i-1,j,k-1))   )*istry  ) );

	    // rp derivatives
// 130 ops, tot=605
	    r1 += ( c2*(
       (2*mu(i+2,j,k)+la(i+2,j,k))*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(1,i+2,j,k+2)-u(1,i+2,j,k-2)) +
             c1*(u(1,i+2,j,k+1)-u(1,i+2,j,k-1))   )*strx(i+2) 
        + la(i+2,j,k)*met(3,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(2,i+2,j,k+2)-u(2,i+2,j,k-2)) +
             c1*(u(2,i+2,j,k+1)-u(2,i+2,j,k-1))  )*stry(j) 
        + la(i+2,j,k)*met(4,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(3,i+2,j,k+2)-u(3,i+2,j,k-2)) +
             c1*(u(3,i+2,j,k+1)-u(3,i+2,j,k-1))  )
       - ((2*mu(i-2,j,k)+la(i-2,j,k))*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(1,i-2,j,k+2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i-2,j,k+1)-u(1,i-2,j,k-1))  )*strx(i-2) 
          + la(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(2,i-2,j,k+2)-u(2,i-2,j,k-2)) +
             c1*(u(2,i-2,j,k+1)-u(2,i-2,j,k-1))   )*stry(j) 
          + la(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(3,i-2,j,k+2)-u(3,i-2,j,k-2)) +
             c1*(u(3,i-2,j,k+1)-u(3,i-2,j,k-1))   ) )
                  ) + c1*(  
          (2*mu(i+1,j,k)+la(i+1,j,k))*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(1,i+1,j,k+2)-u(1,i+1,j,k-2)) +
             c1*(u(1,i+1,j,k+1)-u(1,i+1,j,k-1)) )*strx(i+1) 
          + la(i+1,j,k)*met(3,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(2,i+1,j,k+2)-u(2,i+1,j,k-2)) +
             c1*(u(2,i+1,j,k+1)-u(2,i+1,j,k-1)) )*stry(j) 
          + la(i+1,j,k)*met(4,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(3,i+1,j,k+2)-u(3,i+1,j,k-2)) +
             c1*(u(3,i+1,j,k+1)-u(3,i+1,j,k-1))  )
       - ((2*mu(i-1,j,k)+la(i-1,j,k))*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(1,i-1,j,k+2)-u(1,i-1,j,k-2)) +
             c1*(u(1,i-1,j,k+1)-u(1,i-1,j,k-1)) )*strx(i-1) 
          + la(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(2,i-1,j,k+2)-u(2,i-1,j,k-2)) +
             c1*(u(2,i-1,j,k+1)-u(2,i-1,j,k-1)) )*stry(j) 
          + la(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(3,i-1,j,k+2)-u(3,i-1,j,k-2)) +
             c1*(u(3,i-1,j,k+1)-u(3,i-1,j,k-1))   )  ) ) )*istry;

	    // qr derivatives
// 82 ops, tot=687
	    r1 += c2*(
         mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(1,i,j+2,k+2)-u(1,i,j-2,k+2)) +
             c1*(u(1,i,j+1,k+2)-u(1,i,j-1,k+2))   )*stry(j)*istrx 
        + la(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j-2,k+2)) +
             c1*(u(2,i,j+1,k+2)-u(2,i,j-1,k+2))  ) 
       - ( mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(1,i,j+2,k-2)-u(1,i,j-2,k-2)) +
             c1*(u(1,i,j+1,k-2)-u(1,i,j-1,k-2))  )*stry(j)*istrx  
          + la(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(2,i,j+2,k-2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j+1,k-2)-u(2,i,j-1,k-2))   ) ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(1,i,j+2,k+1)-u(1,i,j-2,k+1)) +
             c1*(u(1,i,j+1,k+1)-u(1,i,j-1,k+1)) )*stry(j)*istrx  
          + la(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(2,i,j+2,k+1)-u(2,i,j-2,k+1)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j-1,k+1)) )  
       - ( mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(1,i,j+2,k-1)-u(1,i,j-2,k-1)) +
             c1*(u(1,i,j+1,k-1)-u(1,i,j-1,k-1)) )*stry(j)*istrx  
          + la(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(2,i,j+2,k-1)-u(2,i,j-2,k-1)) +
             c1*(u(2,i,j+1,k-1)-u(2,i,j-1,k-1)) ) ) );

	    // rq derivatives
// 82 ops, tot=769
	    r1 += c2*(
         mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(1,i,j+2,k+2)-u(1,i,j+2,k-2)) +
             c1*(u(1,i,j+2,k+1)-u(1,i,j+2,k-1))   )*stry(j+2)*istrx 
        + mu(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j+2,k-2)) +
             c1*(u(2,i,j+2,k+1)-u(2,i,j+2,k-1))  ) 
       - ( mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(1,i,j-2,k+2)-u(1,i,j-2,k-2)) +
             c1*(u(1,i,j-2,k+1)-u(1,i,j-2,k-1))  )*stry(j-2)*istrx  
          + mu(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(2,i,j-2,k+2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j-2,k+1)-u(2,i,j-2,k-1))   ) ) 
                  ) + c1*(  
           mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(1,i,j+1,k+2)-u(1,i,j+1,k-2)) +
             c1*(u(1,i,j+1,k+1)-u(1,i,j+1,k-1)) )*stry(j+1)*istrx
          + mu(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(2,i,j+1,k+2)-u(2,i,j+1,k-2)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j+1,k-1)) )
       - ( mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(1,i,j-1,k+2)-u(1,i,j-1,k-2)) +
             c1*(u(1,i,j-1,k+1)-u(1,i,j-1,k-1)) )*stry(j-1)*istrx    
          + mu(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(2,i,j-1,k+2)-u(2,i,j-1,k-2)) +
             c1*(u(2,i,j-1,k+1)-u(2,i,j-1,k-1)) ) ) );

// 4 ops, tot=773
	    lu(1,i,j,k) = a1*lu(1,i,j,k) + r1*ijac;
// v-equation

	    r1 = 0.;
	    // pp derivative (v)
// 43 ops, tot=816
	    cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	    cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	    cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	    cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                    mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                    mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                    mux4*(u(2,i+2,j,k)-u(2,i,j,k))  )*istry;
// qq derivative (v)
// 53 ops, tot=869
	    cof1=(2*mu(i,j-2,k)+la(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)
	     *stry(j-2);
	    cof2=(2*mu(i,j-1,k)+la(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)
	       *stry(j-1);
	    cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)
	       *stry(j);
	    cof4=(2*mu(i,j+1,k)+la(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)
	       *stry(j+1);
	    cof5=(2*mu(i,j+2,k)+la(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)
	       *stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(2,i,j-2,k)-u(2,i,j,k)) + 
                    mux2*(u(2,i,j-1,k)-u(2,i,j,k)) + 
                    mux3*(u(2,i,j+1,k)-u(2,i,j,k)) +
                    mux4*(u(2,i,j+2,k)-u(2,i,j,k))  )*istrx;

// rr derivative (u)
// 42 ops, tot=911
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(3,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(3,i,j,k-1);
	    cof3=(mu(i,j,k)+  la(i,j,k)  )*met(2,i,j,k)*  met(3,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(3,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(3,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(1,i,j,k-2)-u(1,i,j,k)) + 
                    mux2*(u(1,i,j,k-1)-u(1,i,j,k)) + 
                    mux3*(u(1,i,j,k+1)-u(1,i,j,k)) +
                    mux4*(u(1,i,j,k+2)-u(1,i,j,k))  );

// rr derivative (v)
// 83 ops, tot=994
 	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j)
	       +    mu(i,j,k-2)*(met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)+
				 met(4,i,j,k-2)*met(4,i,j,k-2));
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j)
	       +    mu(i,j,k-1)*(met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)+
				 met(4,i,j,k-1)*met(4,i,j,k-1));
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j) +
	       mu(i,j,k)*(met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i)+
			  met(4,i,j,k)*met(4,i,j,k));
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j)
	       +    mu(i,j,k+1)*(met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)+
				 met(4,i,j,k+1)*met(4,i,j,k+1));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j)
	       +    mu(i,j,k+2)*(met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)+
				 met(4,i,j,k+2)*met(4,i,j,k+2));

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(2,i,j,k-2)-u(2,i,j,k)) + 
                    mux2*(u(2,i,j,k-1)-u(2,i,j,k)) + 
                    mux3*(u(2,i,j,k+1)-u(2,i,j,k)) +
                    mux4*(u(2,i,j,k+2)-u(2,i,j,k))  )*istrxy;

// rr derivative (w)
// 43 ops, tot=1037
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)  +la(i,j,k)  )*met(3,i,j,k)*  met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(4,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(3,i,j,k-2)-u(3,i,j,k)) + 
                    mux2*(u(3,i,j,k-1)-u(3,i,j,k)) + 
                    mux3*(u(3,i,j,k+1)-u(3,i,j,k)) +
                    mux4*(u(3,i,j,k+2)-u(3,i,j,k))  )*istrx;

// pq-derivatives
// 38 ops, tot=1075
	    r1 += 
        c2*(  la(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(1,i+2,j+2,k)-u(1,i-2,j+2,k)) +
             c1*(u(1,i+1,j+2,k)-u(1,i-1,j+2,k))    )
          - la(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(1,i+2,j-2,k)-u(1,i-2,j-2,k))+
             c1*(u(1,i+1,j-2,k)-u(1,i-1,j-2,k))     )
          ) +
        c1*(  la(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(u(1,i+2,j+1,k)-u(1,i-2,j+1,k)) +
               c1*(u(1,i+1,j+1,k)-u(1,i-1,j+1,k))  )
           - la(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(u(1,i+2,j-1,k)-u(1,i-2,j-1,k)) + 
               c1*(u(1,i+1,j-1,k)-u(1,i-1,j-1,k))));

// qp-derivatives
// 38 ops, tot=1113
	    r1 += 
        c2*(  mu(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(1,i+2,j+2,k)-u(1,i+2,j-2,k)) +
             c1*(u(1,i+2,j+1,k)-u(1,i+2,j-1,k))    )
          - mu(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(1,i-2,j+2,k)-u(1,i-2,j-2,k))+
             c1*(u(1,i-2,j+1,k)-u(1,i-2,j-1,k))     )
          ) +
        c1*(  mu(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(u(1,i+1,j+2,k)-u(1,i+1,j-2,k)) +
               c1*(u(1,i+1,j+1,k)-u(1,i+1,j-1,k))  )
           - mu(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(u(1,i-1,j+2,k)-u(1,i-1,j-2,k)) + 
               c1*(u(1,i-1,j+1,k)-u(1,i-1,j-1,k))));

// pr-derivatives
// 82 ops, tot=1195
	    r1 += c2*(
       (la(i,j,k+2))*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(1,i+2,j,k+2)-u(1,i-2,j,k+2)) +
             c1*(u(1,i+1,j,k+2)-u(1,i-1,j,k+2))   ) 
        + mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(2,i+2,j,k+2)-u(2,i-2,j,k+2)) +
             c1*(u(2,i+1,j,k+2)-u(2,i-1,j,k+2))  )*strx(i)*istry 
       - ((la(i,j,k-2))*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(1,i+2,j,k-2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i+1,j,k-2)-u(1,i-1,j,k-2))  ) 
          + mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(2,i+2,j,k-2)-u(2,i-2,j,k-2)) +
             c1*(u(2,i+1,j,k-2)-u(2,i-1,j,k-2)) )*strx(i)*istry ) 
                  ) + c1*(  
          (la(i,j,k+1))*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(1,i+2,j,k+1)-u(1,i-2,j,k+1)) +
             c1*(u(1,i+1,j,k+1)-u(1,i-1,j,k+1)) ) 
          + mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(2,i+2,j,k+1)-u(2,i-2,j,k+1)) +
             c1*(u(2,i+1,j,k+1)-u(2,i-1,j,k+1)) )*strx(i)*istry  
       - (la(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(1,i+2,j,k-1)-u(1,i-2,j,k-1)) +
             c1*(u(1,i+1,j,k-1)-u(1,i-1,j,k-1)) ) 
          + mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(2,i+2,j,k-1)-u(2,i-2,j,k-1)) +
             c1*(u(2,i+1,j,k-1)-u(2,i-1,j,k-1)) )*strx(i)*istry  ) );

// rp derivatives
// 82 ops, tot=1277
	    r1 += c2*(
       (mu(i+2,j,k))*met(3,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(1,i+2,j,k+2)-u(1,i+2,j,k-2)) +
             c1*(u(1,i+2,j,k+1)-u(1,i+2,j,k-1))   ) 
        + mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(2,i+2,j,k+2)-u(2,i+2,j,k-2)) +
             c1*(u(2,i+2,j,k+1)-u(2,i+2,j,k-1))  )*strx(i+2)*istry 
       - (mu(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(1,i-2,j,k+2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i-2,j,k+1)-u(1,i-2,j,k-1))  )
          + mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(2,i-2,j,k+2)-u(2,i-2,j,k-2)) +
             c1*(u(2,i-2,j,k+1)-u(2,i-2,j,k-1))   )*strx(i-2)*istry )
                  ) + c1*(  
          (mu(i+1,j,k))*met(3,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(1,i+1,j,k+2)-u(1,i+1,j,k-2)) +
             c1*(u(1,i+1,j,k+1)-u(1,i+1,j,k-1)) ) 
          + mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(2,i+1,j,k+2)-u(2,i+1,j,k-2)) +
             c1*(u(2,i+1,j,k+1)-u(2,i+1,j,k-1)) )*strx(i+1)*istry 
       - (mu(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(1,i-1,j,k+2)-u(1,i-1,j,k-2)) +
             c1*(u(1,i-1,j,k+1)-u(1,i-1,j,k-1)) ) 
          + mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(2,i-1,j,k+2)-u(2,i-1,j,k-2)) +
             c1*(u(2,i-1,j,k+1)-u(2,i-1,j,k-1)) )*strx(i-1)*istry  ) );
	 
// qr derivatives
// 130 ops, tot=1407
	    r1 += c2*(
         mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(1,i,j+2,k+2)-u(1,i,j-2,k+2)) +
             c1*(u(1,i,j+1,k+2)-u(1,i,j-1,k+2))   ) 
        + (2*mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j-2,k+2)) +
             c1*(u(2,i,j+1,k+2)-u(2,i,j-1,k+2))  )*stry(j)*istrx 
        +mu(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(3,i,j+2,k+2)-u(3,i,j-2,k+2)) +
             c1*(u(3,i,j+1,k+2)-u(3,i,j-1,k+2))   )*istrx 
       - ( mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(1,i,j+2,k-2)-u(1,i,j-2,k-2)) +
             c1*(u(1,i,j+1,k-2)-u(1,i,j-1,k-2))  ) 
         +(2*mu(i,j,k-2)+ la(i,j,k-2))*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(2,i,j+2,k-2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j+1,k-2)-u(2,i,j-1,k-2))   )*stry(j)*istrx +
            mu(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(3,i,j+2,k-2)-u(3,i,j-2,k-2)) +
             c1*(u(3,i,j+1,k-2)-u(3,i,j-1,k-2))  )*istrx ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(1,i,j+2,k+1)-u(1,i,j-2,k+1)) +
             c1*(u(1,i,j+1,k+1)-u(1,i,j-1,k+1)) ) 
         + (2*mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(2,i,j+2,k+1)-u(2,i,j-2,k+1)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j-1,k+1)) )*stry(j)*istrx
         + mu(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(3,i,j+2,k+1)-u(3,i,j-2,k+1)) +
             c1*(u(3,i,j+1,k+1)-u(3,i,j-1,k+1)) )*istrx   
       - ( mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(1,i,j+2,k-1)-u(1,i,j-2,k-1)) +
             c1*(u(1,i,j+1,k-1)-u(1,i,j-1,k-1)) ) 
         + (2*mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(2,i,j+2,k-1)-u(2,i,j-2,k-1)) +
             c1*(u(2,i,j+1,k-1)-u(2,i,j-1,k-1)) )*stry(j)*istrx
         +  mu(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(3,i,j+2,k-1)-u(3,i,j-2,k-1)) +
             c1*(u(3,i,j+1,k-1)-u(3,i,j-1,k-1)) )*istrx  ) );
	 
// rq derivatives
// 130 ops, tot=1537
	    r1 += c2*(
         la(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(1,i,j+2,k+2)-u(1,i,j+2,k-2)) +
             c1*(u(1,i,j+2,k+1)-u(1,i,j+2,k-1))   ) 
        +(2*mu(i,j+2,k)+la(i,j+2,k))*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j+2,k-2)) +
             c1*(u(2,i,j+2,k+1)-u(2,i,j+2,k-1))  )*stry(j+2)*istrx 
         + la(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(3,i,j+2,k+2)-u(3,i,j+2,k-2)) +
             c1*(u(3,i,j+2,k+1)-u(3,i,j+2,k-1))   )*istrx 
       - ( la(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(1,i,j-2,k+2)-u(1,i,j-2,k-2)) +
             c1*(u(1,i,j-2,k+1)-u(1,i,j-2,k-1))  ) 
          +(2*mu(i,j-2,k)+la(i,j-2,k))*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(2,i,j-2,k+2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j-2,k+1)-u(2,i,j-2,k-1))   )*stry(j-2)*istrx 
         + la(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(3,i,j-2,k+2)-u(3,i,j-2,k-2)) +
             c1*(u(3,i,j-2,k+1)-u(3,i,j-2,k-1))  )*istrx  ) 
		      ) + c1*(  
           la(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(1,i,j+1,k+2)-u(1,i,j+1,k-2)) +
             c1*(u(1,i,j+1,k+1)-u(1,i,j+1,k-1)) ) 
          + (2*mu(i,j+1,k)+la(i,j+1,k))*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(2,i,j+1,k+2)-u(2,i,j+1,k-2)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j+1,k-1)) )*stry(j+1)*istrx 
          +la(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(3,i,j+1,k+2)-u(3,i,j+1,k-2)) +
             c1*(u(3,i,j+1,k+1)-u(3,i,j+1,k-1)) )*istrx   
       - ( la(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(1,i,j-1,k+2)-u(1,i,j-1,k-2)) +
             c1*(u(1,i,j-1,k+1)-u(1,i,j-1,k-1)) ) 
          + (2*mu(i,j-1,k)+la(i,j-1,k))*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(2,i,j-1,k+2)-u(2,i,j-1,k-2)) +
             c1*(u(2,i,j-1,k+1)-u(2,i,j-1,k-1)) )*stry(j-1)*istrx
          + la(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(3,i,j-1,k+2)-u(3,i,j-1,k-2)) +
             c1*(u(3,i,j-1,k+1)-u(3,i,j-1,k-1)) )*istrx   ) );

// 4 ops, tot=1541
	    lu(2,i,j,k) = a1*lu(2,i,j,k) + r1*ijac;
	 
// w-equation
	    r1 = 0.;
// pp derivative (w)
// 43 ops, tot=1580
	    cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	    cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	    cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	    cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                    mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                    mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                    mux4*(u(3,i+2,j,k)-u(3,i,j,k))  )*istry;

// qq derivative (w)
// 43 ops, tot=1623
	    cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	    cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	    cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	    cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                    mux2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                    mux3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                    mux4*(u(3,i,j+2,k)-u(3,i,j,k))  )*istrx;
// rr derivative (u)
// 43 ops, tot=1666
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(4,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(1,i,j,k-2)-u(1,i,j,k)) + 
                    mux2*(u(1,i,j,k-1)-u(1,i,j,k)) + 
                    mux3*(u(1,i,j,k+1)-u(1,i,j,k)) +
                    mux4*(u(1,i,j,k+2)-u(1,i,j,k))  )*istry;

// rr derivative (v)
// 43 ops, tot=1709
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(3,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(4,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(2,i,j,k-2)-u(2,i,j,k)) + 
                    mux2*(u(2,i,j,k-1)-u(2,i,j,k)) + 
                    mux3*(u(2,i,j,k+1)-u(2,i,j,k)) +
                    mux4*(u(2,i,j,k+2)-u(2,i,j,k))  )*istrx;

// rr derivative (w)
// 83 ops, tot=1792
	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(4,i,j,k-2)*met(4,i,j,k-2) +
              mu(i,j,k-2)*(met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)+
			   met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j) );
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(4,i,j,k-1)*met(4,i,j,k-1) +
              mu(i,j,k-1)*(met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)+
			   met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j) );
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(4,i,j,k)*met(4,i,j,k) +
              mu(i,j,k)*(met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i)+
			 met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j) );
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(4,i,j,k+1)*met(4,i,j,k+1) +
              mu(i,j,k+1)*(met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)+
			   met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(4,i,j,k+2)*met(4,i,j,k+2) +
              mu(i,j,k+2)*( met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)+
			    met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j) );
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(3,i,j,k-2)-u(3,i,j,k)) + 
                    mux2*(u(3,i,j,k-1)-u(3,i,j,k)) + 
                    mux3*(u(3,i,j,k+1)-u(3,i,j,k)) +
                    mux4*(u(3,i,j,k+2)-u(3,i,j,k))  )*istrxy
// pr-derivatives
// 86 ops, tot=1878
// r1 += 
          + c2*(
       (la(i,j,k+2))*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(1,i+2,j,k+2)-u(1,i-2,j,k+2)) +
             c1*(u(1,i+1,j,k+2)-u(1,i-1,j,k+2))   )*istry 
        + mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(3,i+2,j,k+2)-u(3,i-2,j,k+2)) +
             c1*(u(3,i+1,j,k+2)-u(3,i-1,j,k+2))  )*strx(i)*istry 
       - ((la(i,j,k-2))*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(1,i+2,j,k-2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i+1,j,k-2)-u(1,i-1,j,k-2))  )*istry  
          + mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(3,i+2,j,k-2)-u(3,i-2,j,k-2)) +
             c1*(u(3,i+1,j,k-2)-u(3,i-1,j,k-2)) )*strx(i)*istry ) 
                  ) + c1*(  
          (la(i,j,k+1))*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(1,i+2,j,k+1)-u(1,i-2,j,k+1)) +
             c1*(u(1,i+1,j,k+1)-u(1,i-1,j,k+1)) )*istry  
          + mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(3,i+2,j,k+1)-u(3,i-2,j,k+1)) +
             c1*(u(3,i+1,j,k+1)-u(3,i-1,j,k+1)) )*strx(i)*istry  
       - (la(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(1,i+2,j,k-1)-u(1,i-2,j,k-1)) +
             c1*(u(1,i+1,j,k-1)-u(1,i-1,j,k-1)) )*istry  
          + mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(3,i+2,j,k-1)-u(3,i-2,j,k-1)) +
             c1*(u(3,i+1,j,k-1)-u(3,i-1,j,k-1)) )*strx(i)*istry  ) )
// rp derivatives
// 79 ops, tot=1957
//   r1 += 
         + istry*(c2*(
       (mu(i+2,j,k))*met(4,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(1,i+2,j,k+2)-u(1,i+2,j,k-2)) +
             c1*(u(1,i+2,j,k+1)-u(1,i+2,j,k-1))   ) 
        + mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(3,i+2,j,k+2)-u(3,i+2,j,k-2)) +
             c1*(u(3,i+2,j,k+1)-u(3,i+2,j,k-1)) )*strx(i+2) 
       - (mu(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(1,i-2,j,k+2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i-2,j,k+1)-u(1,i-2,j,k-1))  )
          + mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(3,i-2,j,k+2)-u(3,i-2,j,k-2)) +
             c1*(u(3,i-2,j,k+1)-u(3,i-2,j,k-1)) )*strx(i-2)  )
                  ) + c1*(  
          (mu(i+1,j,k))*met(4,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(1,i+1,j,k+2)-u(1,i+1,j,k-2)) +
             c1*(u(1,i+1,j,k+1)-u(1,i+1,j,k-1)) ) 
          + mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(3,i+1,j,k+2)-u(3,i+1,j,k-2)) +
             c1*(u(3,i+1,j,k+1)-u(3,i+1,j,k-1)) )*strx(i+1)  
       - (mu(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(1,i-1,j,k+2)-u(1,i-1,j,k-2)) +
             c1*(u(1,i-1,j,k+1)-u(1,i-1,j,k-1)) ) 
          + mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(3,i-1,j,k+2)-u(3,i-1,j,k-2)) +
             c1*(u(3,i-1,j,k+1)-u(3,i-1,j,k-1)) )*strx(i-1)  ) ) )
// qr derivatives
// 86 ops, tot=2043
//     r1 +=
         + c2*(
         mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(3,i,j+2,k+2)-u(3,i,j-2,k+2)) +
             c1*(u(3,i,j+1,k+2)-u(3,i,j-1,k+2))   )*stry(j)*istrx 
        + la(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j-2,k+2)) +
             c1*(u(2,i,j+1,k+2)-u(2,i,j-1,k+2))  )*istrx 
       - ( mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(3,i,j+2,k-2)-u(3,i,j-2,k-2)) +
             c1*(u(3,i,j+1,k-2)-u(3,i,j-1,k-2))  )*stry(j)*istrx  
          + la(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(2,i,j+2,k-2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j+1,k-2)-u(2,i,j-1,k-2))   )*istrx  ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(3,i,j+2,k+1)-u(3,i,j-2,k+1)) +
             c1*(u(3,i,j+1,k+1)-u(3,i,j-1,k+1)) )*stry(j)*istrx  
          + la(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(2,i,j+2,k+1)-u(2,i,j-2,k+1)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j-1,k+1)) )*istrx   
       - ( mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(3,i,j+2,k-1)-u(3,i,j-2,k-1)) +
             c1*(u(3,i,j+1,k-1)-u(3,i,j-1,k-1)) )*stry(j)*istrx  
          + la(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(2,i,j+2,k-1)-u(2,i,j-2,k-1)) +
             c1*(u(2,i,j+1,k-1)-u(2,i,j-1,k-1)) )*istrx  ) )
// rq derivatives
//  79 ops, tot=2122
//  r1 += 
          + istrx*(c2*(
         mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(3,i,j+2,k+2)-u(3,i,j+2,k-2)) +
             c1*(u(3,i,j+2,k+1)-u(3,i,j+2,k-1))   )*stry(j+2) 
        + mu(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j+2,k-2)) +
             c1*(u(2,i,j+2,k+1)-u(2,i,j+2,k-1))  ) 
       - ( mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(3,i,j-2,k+2)-u(3,i,j-2,k-2)) +
             c1*(u(3,i,j-2,k+1)-u(3,i,j-2,k-1))  )*stry(j-2) 
          + mu(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(2,i,j-2,k+2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j-2,k+1)-u(2,i,j-2,k-1))   ) ) 
                  ) + c1*(  
           mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(3,i,j+1,k+2)-u(3,i,j+1,k-2)) +
             c1*(u(3,i,j+1,k+1)-u(3,i,j+1,k-1)) )*stry(j+1) 
          + mu(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(2,i,j+1,k+2)-u(2,i,j+1,k-2)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j+1,k-1)) )  
       - ( mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(3,i,j-1,k+2)-u(3,i,j-1,k-2)) +
             c1*(u(3,i,j-1,k+1)-u(3,i,j-1,k-1)) )*stry(j-1) 
          + mu(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(2,i,j-1,k+2)-u(2,i,j-1,k-2)) +
             c1*(u(2,i,j-1,k+1)-u(2,i,j-1,k-1)) ) ) ) );
// 4 ops, tot=2126
	    lu(3,i,j,k) = a1*lu(3,i,j,k) + r1*ijac;
	 }
#undef mu
#undef la
#undef jac
#undef u
#undef lu
#undef met
#undef strx
#undef stry
}
//-----------------------------------------------------------------------
__global__ void rhs4sgcurv_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_met, float_sw4* a_jac, float_sw4* a_lu, 
				int onesided4, float_sw4* a_strx, float_sw4* a_stry, int ghost_points )
{
 // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+i+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+i+ni*(j)+nij*(k)]
#define jac(i,j,k)   a_jac[base+i+ni*(j)+nij*(k)]
#define u(c,i,j,k)     a_u[base3+(c)+3*(i)+ni3*(j)+nij3*(k)]   
#define lu(c,i,j,k)   a_lu[base3+(c)+3*(i)+ni3*(j)+nij3*(k)]   
#define met(c,i,j,k) a_met[base4+(c)+4*(i)+ni4*(j)+nij4*(k)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
   const float_sw4 a1   = 0;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 tf   = 0.75;
   const float_sw4 c1   = 2.0/3;
   const float_sw4 c2   = -1.0/12;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = nij*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = 3*base-1;
   const int base4 = 4*base-1;
   const int ni3  = 3*ni;
   const int nij3 = 3*nij;
   const int ni4  = 4*ni;
   const int nij4 = 4*nij;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof;
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if(onesided4 == 1)
     k = 7 + threadIdx.z + blockIdx.z*blockDim.z;

   if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
	 {
// 5 ops
	    float_sw4 ijac = strx(i)*stry(j)/jac(i,j,k);
            float_sw4 istry = 1./(stry(j));
            float_sw4 istrx = 1./(strx(i));
            float_sw4 istrxy = istry*istrx;

            float_sw4 r1 = 0.;

	    // pp derivative (u)
// 53 ops, tot=58
	    float_sw4 cof1=(2*mu(i-2,j,k)+la(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)
	       *strx(i-2);
	    float_sw4 cof2=(2*mu(i-1,j,k)+la(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)
	       *strx(i-1);
	    float_sw4 cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)
		  *strx(i);
	    float_sw4 cof4=(2*mu(i+1,j,k)+la(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)
	     *strx(i+1);
	    float_sw4 cof5=(2*mu(i+2,j,k)+la(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)
	     *strx(i+2);
            float_sw4 mux1 = cof2 -tf*(cof3+cof1);
            float_sw4 mux2 = cof1 + cof4+3*(cof3+cof2);
            float_sw4 mux3 = cof2 + cof5+3*(cof4+cof3);
            float_sw4 mux4 = cof4-tf*(cof3+cof5);

            r1 +=  i6* (
                    mux1*(u(1,i-2,j,k)-u(1,i,j,k)) + 
                    mux2*(u(1,i-1,j,k)-u(1,i,j,k)) + 
                    mux3*(u(1,i+1,j,k)-u(1,i,j,k)) +
                    mux4*(u(1,i+2,j,k)-u(1,i,j,k))  )*istry;
	    // qq derivative (u)
// 43 ops, tot=101
	    cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	    cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	    cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	    cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                    mux2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                    mux3*(u(1,i,j+1,k)-u(1,i,j,k)) +
                    mux4*(u(1,i,j+2,k)-u(1,i,j,k))  )*istrx;
	    // rr derivative (u)
// 5*11+14+14=83 ops, tot=184
	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)
	       +   mu(i,j,k-2)*(met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j)+
				met(4,i,j,k-2)*met(4,i,j,k-2));
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)
	       +   mu(i,j,k-1)*(met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j)+
				met(4,i,j,k-1)*met(4,i,j,k-1) );
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i) +
	     mu(i,j,k)*(met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j)+
			met(4,i,j,k)*met(4,i,j,k));
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)
	       +   mu(i,j,k+1)*(met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j)+
				met(4,i,j,k+1)*met(4,i,j,k+1));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)
	       +   mu(i,j,k+2)*( met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j)+
				 met(4,i,j,k+2)*met(4,i,j,k+2));

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(1,i,j,k-2)-u(1,i,j,k)) + 
                    mux2*(u(1,i,j,k-1)-u(1,i,j,k)) + 
                    mux3*(u(1,i,j,k+1)-u(1,i,j,k)) +
                    mux4*(u(1,i,j,k+2)-u(1,i,j,k))  )*istrxy;

	    // rr derivative (v)
	    // 42 ops, tot=226
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(3,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(3,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(3,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(3,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(3,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(2,i,j,k-2)-u(2,i,j,k)) + 
                    mux2*(u(2,i,j,k-1)-u(2,i,j,k)) + 
                    mux3*(u(2,i,j,k+1)-u(2,i,j,k)) +
                    mux4*(u(2,i,j,k+2)-u(2,i,j,k))  );

	    // rr derivative (w)
// 43 ops, tot=269
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(4,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
		    mux1*(u(3,i,j,k-2)-u(3,i,j,k)) + 
                    mux2*(u(3,i,j,k-1)-u(3,i,j,k)) + 
                    mux3*(u(3,i,j,k+1)-u(3,i,j,k)) +
                    mux4*(u(3,i,j,k+2)-u(3,i,j,k))  )*istry;

	    // pq-derivatives
// 38 ops, tot=307
	    r1 += 
        c2*(  mu(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(2,i+2,j+2,k)-u(2,i-2,j+2,k)) +
             c1*(u(2,i+1,j+2,k)-u(2,i-1,j+2,k))    )
          - mu(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(2,i+2,j-2,k)-u(2,i-2,j-2,k))+
             c1*(u(2,i+1,j-2,k)-u(2,i-1,j-2,k))     )
          ) +
        c1*(  mu(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(u(2,i+2,j+1,k)-u(2,i-2,j+1,k)) +
               c1*(u(2,i+1,j+1,k)-u(2,i-1,j+1,k))  )
           - mu(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(u(2,i+2,j-1,k)-u(2,i-2,j-1,k)) + 
               c1*(u(2,i+1,j-1,k)-u(2,i-1,j-1,k))));

	    // qp-derivatives
// 38 ops, tot=345
	    r1 += 
        c2*(  la(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(2,i+2,j+2,k)-u(2,i+2,j-2,k)) +
             c1*(u(2,i+2,j+1,k)-u(2,i+2,j-1,k))    )
          - la(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(2,i-2,j+2,k)-u(2,i-2,j-2,k))+
             c1*(u(2,i-2,j+1,k)-u(2,i-2,j-1,k))     )
          ) +
        c1*(  la(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(u(2,i+1,j+2,k)-u(2,i+1,j-2,k)) +
               c1*(u(2,i+1,j+1,k)-u(2,i+1,j-1,k))  )
           - la(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(u(2,i-1,j+2,k)-u(2,i-1,j-2,k)) + 
               c1*(u(2,i-1,j+1,k)-u(2,i-1,j-1,k))));

      // pr-derivatives
// 130 ops., tot=475
	    r1 += c2*(
       (2*mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(1,i+2,j,k+2)-u(1,i-2,j,k+2)) +
             c1*(u(1,i+1,j,k+2)-u(1,i-1,j,k+2))   )*strx(i)*istry 
        + mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(2,i+2,j,k+2)-u(2,i-2,j,k+2)) +
             c1*(u(2,i+1,j,k+2)-u(2,i-1,j,k+2))  ) 
        + mu(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(3,i+2,j,k+2)-u(3,i-2,j,k+2)) +
             c1*(u(3,i+1,j,k+2)-u(3,i-1,j,k+2))  )*istry
       - ((2*mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(1,i+2,j,k-2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i+1,j,k-2)-u(1,i-1,j,k-2))  )*strx(i)*istry  
          + mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(2,i+2,j,k-2)-u(2,i-2,j,k-2)) +
             c1*(u(2,i+1,j,k-2)-u(2,i-1,j,k-2))   ) 
          + mu(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(3,i+2,j,k-2)-u(3,i-2,j,k-2)) +
             c1*(u(3,i+1,j,k-2)-u(3,i-1,j,k-2))   )*istry )
                  ) + c1*(  
          (2*mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(1,i+2,j,k+1)-u(1,i-2,j,k+1)) +
             c1*(u(1,i+1,j,k+1)-u(1,i-1,j,k+1)) )*strx(i)*istry 
          + mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(2,i+2,j,k+1)-u(2,i-2,j,k+1)) +
             c1*(u(2,i+1,j,k+1)-u(2,i-1,j,k+1)) ) 
          + mu(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(3,i+2,j,k+1)-u(3,i-2,j,k+1)) +
             c1*(u(3,i+1,j,k+1)-u(3,i-1,j,k+1))  )*istry
       - ((2*mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(1,i+2,j,k-1)-u(1,i-2,j,k-1)) +
             c1*(u(1,i+1,j,k-1)-u(1,i-1,j,k-1)) )*strx(i)*istry  
          + mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(2,i+2,j,k-1)-u(2,i-2,j,k-1)) +
             c1*(u(2,i+1,j,k-1)-u(2,i-1,j,k-1)) ) 
          + mu(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(3,i+2,j,k-1)-u(3,i-2,j,k-1)) +
             c1*(u(3,i+1,j,k-1)-u(3,i-1,j,k-1))   )*istry  ) );

	    // rp derivatives
// 130 ops, tot=605
	    r1 += ( c2*(
       (2*mu(i+2,j,k)+la(i+2,j,k))*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(1,i+2,j,k+2)-u(1,i+2,j,k-2)) +
             c1*(u(1,i+2,j,k+1)-u(1,i+2,j,k-1))   )*strx(i+2) 
        + la(i+2,j,k)*met(3,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(2,i+2,j,k+2)-u(2,i+2,j,k-2)) +
             c1*(u(2,i+2,j,k+1)-u(2,i+2,j,k-1))  )*stry(j) 
        + la(i+2,j,k)*met(4,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(3,i+2,j,k+2)-u(3,i+2,j,k-2)) +
             c1*(u(3,i+2,j,k+1)-u(3,i+2,j,k-1))  )
       - ((2*mu(i-2,j,k)+la(i-2,j,k))*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(1,i-2,j,k+2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i-2,j,k+1)-u(1,i-2,j,k-1))  )*strx(i-2) 
          + la(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(2,i-2,j,k+2)-u(2,i-2,j,k-2)) +
             c1*(u(2,i-2,j,k+1)-u(2,i-2,j,k-1))   )*stry(j) 
          + la(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(3,i-2,j,k+2)-u(3,i-2,j,k-2)) +
             c1*(u(3,i-2,j,k+1)-u(3,i-2,j,k-1))   ) )
                  ) + c1*(  
          (2*mu(i+1,j,k)+la(i+1,j,k))*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(1,i+1,j,k+2)-u(1,i+1,j,k-2)) +
             c1*(u(1,i+1,j,k+1)-u(1,i+1,j,k-1)) )*strx(i+1) 
          + la(i+1,j,k)*met(3,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(2,i+1,j,k+2)-u(2,i+1,j,k-2)) +
             c1*(u(2,i+1,j,k+1)-u(2,i+1,j,k-1)) )*stry(j) 
          + la(i+1,j,k)*met(4,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(3,i+1,j,k+2)-u(3,i+1,j,k-2)) +
             c1*(u(3,i+1,j,k+1)-u(3,i+1,j,k-1))  )
       - ((2*mu(i-1,j,k)+la(i-1,j,k))*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(1,i-1,j,k+2)-u(1,i-1,j,k-2)) +
             c1*(u(1,i-1,j,k+1)-u(1,i-1,j,k-1)) )*strx(i-1) 
          + la(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(2,i-1,j,k+2)-u(2,i-1,j,k-2)) +
             c1*(u(2,i-1,j,k+1)-u(2,i-1,j,k-1)) )*stry(j) 
          + la(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(3,i-1,j,k+2)-u(3,i-1,j,k-2)) +
             c1*(u(3,i-1,j,k+1)-u(3,i-1,j,k-1))   )  ) ) )*istry;

	    // qr derivatives
// 82 ops, tot=687
	    r1 += c2*(
         mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(1,i,j+2,k+2)-u(1,i,j-2,k+2)) +
             c1*(u(1,i,j+1,k+2)-u(1,i,j-1,k+2))   )*stry(j)*istrx 
        + la(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j-2,k+2)) +
             c1*(u(2,i,j+1,k+2)-u(2,i,j-1,k+2))  ) 
       - ( mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(1,i,j+2,k-2)-u(1,i,j-2,k-2)) +
             c1*(u(1,i,j+1,k-2)-u(1,i,j-1,k-2))  )*stry(j)*istrx  
          + la(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(2,i,j+2,k-2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j+1,k-2)-u(2,i,j-1,k-2))   ) ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(1,i,j+2,k+1)-u(1,i,j-2,k+1)) +
             c1*(u(1,i,j+1,k+1)-u(1,i,j-1,k+1)) )*stry(j)*istrx  
          + la(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(2,i,j+2,k+1)-u(2,i,j-2,k+1)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j-1,k+1)) )  
       - ( mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(1,i,j+2,k-1)-u(1,i,j-2,k-1)) +
             c1*(u(1,i,j+1,k-1)-u(1,i,j-1,k-1)) )*stry(j)*istrx  
          + la(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(2,i,j+2,k-1)-u(2,i,j-2,k-1)) +
             c1*(u(2,i,j+1,k-1)-u(2,i,j-1,k-1)) ) ) );

	    // rq derivatives
// 82 ops, tot=769
	    r1 += c2*(
         mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(1,i,j+2,k+2)-u(1,i,j+2,k-2)) +
             c1*(u(1,i,j+2,k+1)-u(1,i,j+2,k-1))   )*stry(j+2)*istrx 
        + mu(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j+2,k-2)) +
             c1*(u(2,i,j+2,k+1)-u(2,i,j+2,k-1))  ) 
       - ( mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(1,i,j-2,k+2)-u(1,i,j-2,k-2)) +
             c1*(u(1,i,j-2,k+1)-u(1,i,j-2,k-1))  )*stry(j-2)*istrx  
          + mu(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(2,i,j-2,k+2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j-2,k+1)-u(2,i,j-2,k-1))   ) ) 
                  ) + c1*(  
           mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(1,i,j+1,k+2)-u(1,i,j+1,k-2)) +
             c1*(u(1,i,j+1,k+1)-u(1,i,j+1,k-1)) )*stry(j+1)*istrx
          + mu(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(2,i,j+1,k+2)-u(2,i,j+1,k-2)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j+1,k-1)) )
       - ( mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(1,i,j-1,k+2)-u(1,i,j-1,k-2)) +
             c1*(u(1,i,j-1,k+1)-u(1,i,j-1,k-1)) )*stry(j-1)*istrx    
          + mu(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(2,i,j-1,k+2)-u(2,i,j-1,k-2)) +
             c1*(u(2,i,j-1,k+1)-u(2,i,j-1,k-1)) ) ) );

// 4 ops, tot=773
	    lu(1,i,j,k) = a1*lu(1,i,j,k) + r1*ijac;
// v-equation

	    r1 = 0.;
	    // pp derivative (v)
// 43 ops, tot=816
	    cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	    cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	    cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	    cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                    mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                    mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                    mux4*(u(2,i+2,j,k)-u(2,i,j,k))  )*istry;
// qq derivative (v)
// 53 ops, tot=869
	    cof1=(2*mu(i,j-2,k)+la(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)
	     *stry(j-2);
	    cof2=(2*mu(i,j-1,k)+la(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)
	       *stry(j-1);
	    cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)
	       *stry(j);
	    cof4=(2*mu(i,j+1,k)+la(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)
	       *stry(j+1);
	    cof5=(2*mu(i,j+2,k)+la(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)
	       *stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(2,i,j-2,k)-u(2,i,j,k)) + 
                    mux2*(u(2,i,j-1,k)-u(2,i,j,k)) + 
                    mux3*(u(2,i,j+1,k)-u(2,i,j,k)) +
                    mux4*(u(2,i,j+2,k)-u(2,i,j,k))  )*istrx;

// rr derivative (u)
// 42 ops, tot=911
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(3,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(3,i,j,k-1);
	    cof3=(mu(i,j,k)+  la(i,j,k)  )*met(2,i,j,k)*  met(3,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(3,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(3,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(1,i,j,k-2)-u(1,i,j,k)) + 
                    mux2*(u(1,i,j,k-1)-u(1,i,j,k)) + 
                    mux3*(u(1,i,j,k+1)-u(1,i,j,k)) +
                    mux4*(u(1,i,j,k+2)-u(1,i,j,k))  );

// rr derivative (v)
// 83 ops, tot=994
 	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j)
	       +    mu(i,j,k-2)*(met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)+
				 met(4,i,j,k-2)*met(4,i,j,k-2));
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j)
	       +    mu(i,j,k-1)*(met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)+
				 met(4,i,j,k-1)*met(4,i,j,k-1));
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j) +
	       mu(i,j,k)*(met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i)+
			  met(4,i,j,k)*met(4,i,j,k));
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j)
	       +    mu(i,j,k+1)*(met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)+
				 met(4,i,j,k+1)*met(4,i,j,k+1));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j)
	       +    mu(i,j,k+2)*(met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)+
				 met(4,i,j,k+2)*met(4,i,j,k+2));

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(2,i,j,k-2)-u(2,i,j,k)) + 
                    mux2*(u(2,i,j,k-1)-u(2,i,j,k)) + 
                    mux3*(u(2,i,j,k+1)-u(2,i,j,k)) +
                    mux4*(u(2,i,j,k+2)-u(2,i,j,k))  )*istrxy;

// rr derivative (w)
// 43 ops, tot=1037
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)  +la(i,j,k)  )*met(3,i,j,k)*  met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(4,i,j,k+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(3,i,j,k-2)-u(3,i,j,k)) + 
                    mux2*(u(3,i,j,k-1)-u(3,i,j,k)) + 
                    mux3*(u(3,i,j,k+1)-u(3,i,j,k)) +
                    mux4*(u(3,i,j,k+2)-u(3,i,j,k))  )*istrx;

// pq-derivatives
// 38 ops, tot=1075
	    r1 += 
        c2*(  la(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(1,i+2,j+2,k)-u(1,i-2,j+2,k)) +
             c1*(u(1,i+1,j+2,k)-u(1,i-1,j+2,k))    )
          - la(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(1,i+2,j-2,k)-u(1,i-2,j-2,k))+
             c1*(u(1,i+1,j-2,k)-u(1,i-1,j-2,k))     )
          ) +
        c1*(  la(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(u(1,i+2,j+1,k)-u(1,i-2,j+1,k)) +
               c1*(u(1,i+1,j+1,k)-u(1,i-1,j+1,k))  )
           - la(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(u(1,i+2,j-1,k)-u(1,i-2,j-1,k)) + 
               c1*(u(1,i+1,j-1,k)-u(1,i-1,j-1,k))));

// qp-derivatives
// 38 ops, tot=1113
	    r1 += 
        c2*(  mu(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(1,i+2,j+2,k)-u(1,i+2,j-2,k)) +
             c1*(u(1,i+2,j+1,k)-u(1,i+2,j-1,k))    )
          - mu(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(1,i-2,j+2,k)-u(1,i-2,j-2,k))+
             c1*(u(1,i-2,j+1,k)-u(1,i-2,j-1,k))     )
          ) +
        c1*(  mu(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(u(1,i+1,j+2,k)-u(1,i+1,j-2,k)) +
               c1*(u(1,i+1,j+1,k)-u(1,i+1,j-1,k))  )
           - mu(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(u(1,i-1,j+2,k)-u(1,i-1,j-2,k)) + 
               c1*(u(1,i-1,j+1,k)-u(1,i-1,j-1,k))));

// pr-derivatives
// 82 ops, tot=1195
	    r1 += c2*(
       (la(i,j,k+2))*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(1,i+2,j,k+2)-u(1,i-2,j,k+2)) +
             c1*(u(1,i+1,j,k+2)-u(1,i-1,j,k+2))   ) 
        + mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(2,i+2,j,k+2)-u(2,i-2,j,k+2)) +
             c1*(u(2,i+1,j,k+2)-u(2,i-1,j,k+2))  )*strx(i)*istry 
       - ((la(i,j,k-2))*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(1,i+2,j,k-2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i+1,j,k-2)-u(1,i-1,j,k-2))  ) 
          + mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(2,i+2,j,k-2)-u(2,i-2,j,k-2)) +
             c1*(u(2,i+1,j,k-2)-u(2,i-1,j,k-2)) )*strx(i)*istry ) 
                  ) + c1*(  
          (la(i,j,k+1))*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(1,i+2,j,k+1)-u(1,i-2,j,k+1)) +
             c1*(u(1,i+1,j,k+1)-u(1,i-1,j,k+1)) ) 
          + mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(2,i+2,j,k+1)-u(2,i-2,j,k+1)) +
             c1*(u(2,i+1,j,k+1)-u(2,i-1,j,k+1)) )*strx(i)*istry  
       - (la(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(1,i+2,j,k-1)-u(1,i-2,j,k-1)) +
             c1*(u(1,i+1,j,k-1)-u(1,i-1,j,k-1)) ) 
          + mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(2,i+2,j,k-1)-u(2,i-2,j,k-1)) +
             c1*(u(2,i+1,j,k-1)-u(2,i-1,j,k-1)) )*strx(i)*istry  ) );

// rp derivatives
// 82 ops, tot=1277
	    r1 += c2*(
       (mu(i+2,j,k))*met(3,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(1,i+2,j,k+2)-u(1,i+2,j,k-2)) +
             c1*(u(1,i+2,j,k+1)-u(1,i+2,j,k-1))   ) 
        + mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(2,i+2,j,k+2)-u(2,i+2,j,k-2)) +
             c1*(u(2,i+2,j,k+1)-u(2,i+2,j,k-1))  )*strx(i+2)*istry 
       - (mu(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(1,i-2,j,k+2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i-2,j,k+1)-u(1,i-2,j,k-1))  )
          + mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(2,i-2,j,k+2)-u(2,i-2,j,k-2)) +
             c1*(u(2,i-2,j,k+1)-u(2,i-2,j,k-1))   )*strx(i-2)*istry )
                  ) + c1*(  
          (mu(i+1,j,k))*met(3,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(1,i+1,j,k+2)-u(1,i+1,j,k-2)) +
             c1*(u(1,i+1,j,k+1)-u(1,i+1,j,k-1)) ) 
          + mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(2,i+1,j,k+2)-u(2,i+1,j,k-2)) +
             c1*(u(2,i+1,j,k+1)-u(2,i+1,j,k-1)) )*strx(i+1)*istry 
       - (mu(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(1,i-1,j,k+2)-u(1,i-1,j,k-2)) +
             c1*(u(1,i-1,j,k+1)-u(1,i-1,j,k-1)) ) 
          + mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(2,i-1,j,k+2)-u(2,i-1,j,k-2)) +
             c1*(u(2,i-1,j,k+1)-u(2,i-1,j,k-1)) )*strx(i-1)*istry  ) );
	 
// qr derivatives
// 130 ops, tot=1407
	    r1 += c2*(
         mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(1,i,j+2,k+2)-u(1,i,j-2,k+2)) +
             c1*(u(1,i,j+1,k+2)-u(1,i,j-1,k+2))   ) 
        + (2*mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j-2,k+2)) +
             c1*(u(2,i,j+1,k+2)-u(2,i,j-1,k+2))  )*stry(j)*istrx 
        +mu(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(3,i,j+2,k+2)-u(3,i,j-2,k+2)) +
             c1*(u(3,i,j+1,k+2)-u(3,i,j-1,k+2))   )*istrx 
       - ( mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(1,i,j+2,k-2)-u(1,i,j-2,k-2)) +
             c1*(u(1,i,j+1,k-2)-u(1,i,j-1,k-2))  ) 
         +(2*mu(i,j,k-2)+ la(i,j,k-2))*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(2,i,j+2,k-2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j+1,k-2)-u(2,i,j-1,k-2))   )*stry(j)*istrx +
            mu(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(3,i,j+2,k-2)-u(3,i,j-2,k-2)) +
             c1*(u(3,i,j+1,k-2)-u(3,i,j-1,k-2))  )*istrx ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(1,i,j+2,k+1)-u(1,i,j-2,k+1)) +
             c1*(u(1,i,j+1,k+1)-u(1,i,j-1,k+1)) ) 
         + (2*mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(2,i,j+2,k+1)-u(2,i,j-2,k+1)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j-1,k+1)) )*stry(j)*istrx
         + mu(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(3,i,j+2,k+1)-u(3,i,j-2,k+1)) +
             c1*(u(3,i,j+1,k+1)-u(3,i,j-1,k+1)) )*istrx   
       - ( mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(1,i,j+2,k-1)-u(1,i,j-2,k-1)) +
             c1*(u(1,i,j+1,k-1)-u(1,i,j-1,k-1)) ) 
         + (2*mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(2,i,j+2,k-1)-u(2,i,j-2,k-1)) +
             c1*(u(2,i,j+1,k-1)-u(2,i,j-1,k-1)) )*stry(j)*istrx
         +  mu(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(3,i,j+2,k-1)-u(3,i,j-2,k-1)) +
             c1*(u(3,i,j+1,k-1)-u(3,i,j-1,k-1)) )*istrx  ) );
	 
// rq derivatives
// 130 ops, tot=1537
	    r1 += c2*(
         la(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(1,i,j+2,k+2)-u(1,i,j+2,k-2)) +
             c1*(u(1,i,j+2,k+1)-u(1,i,j+2,k-1))   ) 
        +(2*mu(i,j+2,k)+la(i,j+2,k))*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j+2,k-2)) +
             c1*(u(2,i,j+2,k+1)-u(2,i,j+2,k-1))  )*stry(j+2)*istrx 
         + la(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(3,i,j+2,k+2)-u(3,i,j+2,k-2)) +
             c1*(u(3,i,j+2,k+1)-u(3,i,j+2,k-1))   )*istrx 
       - ( la(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(1,i,j-2,k+2)-u(1,i,j-2,k-2)) +
             c1*(u(1,i,j-2,k+1)-u(1,i,j-2,k-1))  ) 
          +(2*mu(i,j-2,k)+la(i,j-2,k))*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(2,i,j-2,k+2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j-2,k+1)-u(2,i,j-2,k-1))   )*stry(j-2)*istrx 
         + la(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(3,i,j-2,k+2)-u(3,i,j-2,k-2)) +
             c1*(u(3,i,j-2,k+1)-u(3,i,j-2,k-1))  )*istrx  ) 
		      ) + c1*(  
           la(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(1,i,j+1,k+2)-u(1,i,j+1,k-2)) +
             c1*(u(1,i,j+1,k+1)-u(1,i,j+1,k-1)) ) 
          + (2*mu(i,j+1,k)+la(i,j+1,k))*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(2,i,j+1,k+2)-u(2,i,j+1,k-2)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j+1,k-1)) )*stry(j+1)*istrx 
          +la(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(3,i,j+1,k+2)-u(3,i,j+1,k-2)) +
             c1*(u(3,i,j+1,k+1)-u(3,i,j+1,k-1)) )*istrx   
       - ( la(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(1,i,j-1,k+2)-u(1,i,j-1,k-2)) +
             c1*(u(1,i,j-1,k+1)-u(1,i,j-1,k-1)) ) 
          + (2*mu(i,j-1,k)+la(i,j-1,k))*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(2,i,j-1,k+2)-u(2,i,j-1,k-2)) +
             c1*(u(2,i,j-1,k+1)-u(2,i,j-1,k-1)) )*stry(j-1)*istrx
          + la(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(3,i,j-1,k+2)-u(3,i,j-1,k-2)) +
             c1*(u(3,i,j-1,k+1)-u(3,i,j-1,k-1)) )*istrx   ) );

// 4 ops, tot=1541
	    lu(2,i,j,k) = a1*lu(2,i,j,k) + r1*ijac;
	 
// w-equation
	    r1 = 0.;
// pp derivative (w)
// 43 ops, tot=1580
	    cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	    cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	    cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	    cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                    mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                    mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                    mux4*(u(3,i+2,j,k)-u(3,i,j,k))  )*istry;

// qq derivative (w)
// 43 ops, tot=1623
	    cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	    cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	    cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	    cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	    cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                    mux2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                    mux3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                    mux4*(u(3,i,j+2,k)-u(3,i,j,k))  )*istrx;
// rr derivative (u)
// 43 ops, tot=1666
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(2,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(2,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(2,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(2,i,j,k+2)*met(4,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(1,i,j,k-2)-u(1,i,j,k)) + 
                    mux2*(u(1,i,j,k-1)-u(1,i,j,k)) + 
                    mux3*(u(1,i,j,k+1)-u(1,i,j,k)) +
                    mux4*(u(1,i,j,k+2)-u(1,i,j,k))  )*istry;

// rr derivative (v)
// 43 ops, tot=1709
	    cof1=(mu(i,j,k-2)+la(i,j,k-2))*met(3,i,j,k-2)*met(4,i,j,k-2);
	    cof2=(mu(i,j,k-1)+la(i,j,k-1))*met(3,i,j,k-1)*met(4,i,j,k-1);
	    cof3=(mu(i,j,k)+la(i,j,k))*met(3,i,j,k)*met(4,i,j,k);
	    cof4=(mu(i,j,k+1)+la(i,j,k+1))*met(3,i,j,k+1)*met(4,i,j,k+1);
	    cof5=(mu(i,j,k+2)+la(i,j,k+2))*met(3,i,j,k+2)*met(4,i,j,k+2);

            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(2,i,j,k-2)-u(2,i,j,k)) + 
                    mux2*(u(2,i,j,k-1)-u(2,i,j,k)) + 
                    mux3*(u(2,i,j,k+1)-u(2,i,j,k)) +
                    mux4*(u(2,i,j,k+2)-u(2,i,j,k))  )*istrx;

// rr derivative (w)
// 83 ops, tot=1792
	    cof1 = (2*mu(i,j,k-2)+la(i,j,k-2))*met(4,i,j,k-2)*met(4,i,j,k-2) +
              mu(i,j,k-2)*(met(2,i,j,k-2)*strx(i)*met(2,i,j,k-2)*strx(i)+
			   met(3,i,j,k-2)*stry(j)*met(3,i,j,k-2)*stry(j) );
	    cof2 = (2*mu(i,j,k-1)+la(i,j,k-1))*met(4,i,j,k-1)*met(4,i,j,k-1) +
              mu(i,j,k-1)*(met(2,i,j,k-1)*strx(i)*met(2,i,j,k-1)*strx(i)+
			   met(3,i,j,k-1)*stry(j)*met(3,i,j,k-1)*stry(j) );
	    cof3 = (2*mu(i,j,k)+la(i,j,k))*met(4,i,j,k)*met(4,i,j,k) +
              mu(i,j,k)*(met(2,i,j,k)*strx(i)*met(2,i,j,k)*strx(i)+
			 met(3,i,j,k)*stry(j)*met(3,i,j,k)*stry(j) );
	    cof4 = (2*mu(i,j,k+1)+la(i,j,k+1))*met(4,i,j,k+1)*met(4,i,j,k+1) +
              mu(i,j,k+1)*(met(2,i,j,k+1)*strx(i)*met(2,i,j,k+1)*strx(i)+
			   met(3,i,j,k+1)*stry(j)*met(3,i,j,k+1)*stry(j));
	    cof5 = (2*mu(i,j,k+2)+la(i,j,k+2))*met(4,i,j,k+2)*met(4,i,j,k+2) +
              mu(i,j,k+2)*( met(2,i,j,k+2)*strx(i)*met(2,i,j,k+2)*strx(i)+
			    met(3,i,j,k+2)*stry(j)*met(3,i,j,k+2)*stry(j) );
            mux1 = cof2 -tf*(cof3+cof1);
            mux2 = cof1 + cof4+3*(cof3+cof2);
            mux3 = cof2 + cof5+3*(cof4+cof3);
            mux4 = cof4-tf*(cof3+cof5);

            r1 += i6* (
                    mux1*(u(3,i,j,k-2)-u(3,i,j,k)) + 
                    mux2*(u(3,i,j,k-1)-u(3,i,j,k)) + 
                    mux3*(u(3,i,j,k+1)-u(3,i,j,k)) +
                    mux4*(u(3,i,j,k+2)-u(3,i,j,k))  )*istrxy
// pr-derivatives
// 86 ops, tot=1878
// r1 += 
          + c2*(
       (la(i,j,k+2))*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(1,i+2,j,k+2)-u(1,i-2,j,k+2)) +
             c1*(u(1,i+1,j,k+2)-u(1,i-1,j,k+2))   )*istry 
        + mu(i,j,k+2)*met(2,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(3,i+2,j,k+2)-u(3,i-2,j,k+2)) +
             c1*(u(3,i+1,j,k+2)-u(3,i-1,j,k+2))  )*strx(i)*istry 
       - ((la(i,j,k-2))*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(1,i+2,j,k-2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i+1,j,k-2)-u(1,i-1,j,k-2))  )*istry  
          + mu(i,j,k-2)*met(2,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(3,i+2,j,k-2)-u(3,i-2,j,k-2)) +
             c1*(u(3,i+1,j,k-2)-u(3,i-1,j,k-2)) )*strx(i)*istry ) 
                  ) + c1*(  
          (la(i,j,k+1))*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(1,i+2,j,k+1)-u(1,i-2,j,k+1)) +
             c1*(u(1,i+1,j,k+1)-u(1,i-1,j,k+1)) )*istry  
          + mu(i,j,k+1)*met(2,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(3,i+2,j,k+1)-u(3,i-2,j,k+1)) +
             c1*(u(3,i+1,j,k+1)-u(3,i-1,j,k+1)) )*strx(i)*istry  
       - (la(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(1,i+2,j,k-1)-u(1,i-2,j,k-1)) +
             c1*(u(1,i+1,j,k-1)-u(1,i-1,j,k-1)) )*istry  
          + mu(i,j,k-1)*met(2,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(3,i+2,j,k-1)-u(3,i-2,j,k-1)) +
             c1*(u(3,i+1,j,k-1)-u(3,i-1,j,k-1)) )*strx(i)*istry  ) )
// rp derivatives
// 79 ops, tot=1957
//   r1 += 
         + istry*(c2*(
       (mu(i+2,j,k))*met(4,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(1,i+2,j,k+2)-u(1,i+2,j,k-2)) +
             c1*(u(1,i+2,j,k+1)-u(1,i+2,j,k-1))   ) 
        + mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(3,i+2,j,k+2)-u(3,i+2,j,k-2)) +
             c1*(u(3,i+2,j,k+1)-u(3,i+2,j,k-1)) )*strx(i+2) 
       - (mu(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(1,i-2,j,k+2)-u(1,i-2,j,k-2)) +
             c1*(u(1,i-2,j,k+1)-u(1,i-2,j,k-1))  )
          + mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(3,i-2,j,k+2)-u(3,i-2,j,k-2)) +
             c1*(u(3,i-2,j,k+1)-u(3,i-2,j,k-1)) )*strx(i-2)  )
                  ) + c1*(  
          (mu(i+1,j,k))*met(4,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(1,i+1,j,k+2)-u(1,i+1,j,k-2)) +
             c1*(u(1,i+1,j,k+1)-u(1,i+1,j,k-1)) ) 
          + mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*(
             c2*(u(3,i+1,j,k+2)-u(3,i+1,j,k-2)) +
             c1*(u(3,i+1,j,k+1)-u(3,i+1,j,k-1)) )*strx(i+1)  
       - (mu(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(1,i-1,j,k+2)-u(1,i-1,j,k-2)) +
             c1*(u(1,i-1,j,k+1)-u(1,i-1,j,k-1)) ) 
          + mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*(
             c2*(u(3,i-1,j,k+2)-u(3,i-1,j,k-2)) +
             c1*(u(3,i-1,j,k+1)-u(3,i-1,j,k-1)) )*strx(i-1)  ) ) )
// qr derivatives
// 86 ops, tot=2043
//     r1 +=
         + c2*(
         mu(i,j,k+2)*met(3,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(3,i,j+2,k+2)-u(3,i,j-2,k+2)) +
             c1*(u(3,i,j+1,k+2)-u(3,i,j-1,k+2))   )*stry(j)*istrx 
        + la(i,j,k+2)*met(4,i,j,k+2)*met(1,i,j,k+2)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j-2,k+2)) +
             c1*(u(2,i,j+1,k+2)-u(2,i,j-1,k+2))  )*istrx 
       - ( mu(i,j,k-2)*met(3,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(3,i,j+2,k-2)-u(3,i,j-2,k-2)) +
             c1*(u(3,i,j+1,k-2)-u(3,i,j-1,k-2))  )*stry(j)*istrx  
          + la(i,j,k-2)*met(4,i,j,k-2)*met(1,i,j,k-2)*(
             c2*(u(2,i,j+2,k-2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j+1,k-2)-u(2,i,j-1,k-2))   )*istrx  ) 
                  ) + c1*(  
           mu(i,j,k+1)*met(3,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(3,i,j+2,k+1)-u(3,i,j-2,k+1)) +
             c1*(u(3,i,j+1,k+1)-u(3,i,j-1,k+1)) )*stry(j)*istrx  
          + la(i,j,k+1)*met(4,i,j,k+1)*met(1,i,j,k+1)*(
             c2*(u(2,i,j+2,k+1)-u(2,i,j-2,k+1)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j-1,k+1)) )*istrx   
       - ( mu(i,j,k-1)*met(3,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(3,i,j+2,k-1)-u(3,i,j-2,k-1)) +
             c1*(u(3,i,j+1,k-1)-u(3,i,j-1,k-1)) )*stry(j)*istrx  
          + la(i,j,k-1)*met(4,i,j,k-1)*met(1,i,j,k-1)*(
             c2*(u(2,i,j+2,k-1)-u(2,i,j-2,k-1)) +
             c1*(u(2,i,j+1,k-1)-u(2,i,j-1,k-1)) )*istrx  ) )
// rq derivatives
//  79 ops, tot=2122
//  r1 += 
          + istrx*(c2*(
         mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(3,i,j+2,k+2)-u(3,i,j+2,k-2)) +
             c1*(u(3,i,j+2,k+1)-u(3,i,j+2,k-1))   )*stry(j+2) 
        + mu(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(2,i,j+2,k+2)-u(2,i,j+2,k-2)) +
             c1*(u(2,i,j+2,k+1)-u(2,i,j+2,k-1))  ) 
       - ( mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(3,i,j-2,k+2)-u(3,i,j-2,k-2)) +
             c1*(u(3,i,j-2,k+1)-u(3,i,j-2,k-1))  )*stry(j-2) 
          + mu(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(2,i,j-2,k+2)-u(2,i,j-2,k-2)) +
             c1*(u(2,i,j-2,k+1)-u(2,i,j-2,k-1))   ) ) 
                  ) + c1*(  
           mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(3,i,j+1,k+2)-u(3,i,j+1,k-2)) +
             c1*(u(3,i,j+1,k+1)-u(3,i,j+1,k-1)) )*stry(j+1) 
          + mu(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*(
             c2*(u(2,i,j+1,k+2)-u(2,i,j+1,k-2)) +
             c1*(u(2,i,j+1,k+1)-u(2,i,j+1,k-1)) )  
       - ( mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(3,i,j-1,k+2)-u(3,i,j-1,k-2)) +
             c1*(u(3,i,j-1,k+1)-u(3,i,j-1,k-1)) )*stry(j-1) 
          + mu(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*(
             c2*(u(2,i,j-1,k+2)-u(2,i,j-1,k-2)) +
             c1*(u(2,i,j-1,k+1)-u(2,i,j-1,k-1)) ) ) ) );
// 4 ops, tot=2126
	    lu(3,i,j,k) = a1*lu(3,i,j,k) + r1*ijac;
	 }
#undef mu
#undef la
#undef jac
#undef u
#undef lu
#undef met
#undef strx
#undef stry
}

//-----------------------------------------------------------------------
__global__ void rhs4sgcurvupper_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_met, float_sw4* a_jac, float_sw4* a_lu, 
				float_sw4* a_strx, float_sw4* a_stry, int ghost_points )
{
 // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+(i)+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+(i)+ni*(j)+nij*(k)]
#define jac(i,j,k)   a_jac[base+(i)+ni*(j)+nij*(k)]
#define u(c,i,j,k)     a_u[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define lu(c,i,j,k)   a_lu[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define met(c,i,j,k) a_met[base4+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define acof(i,j,k) dev_acof[(i-1)+6*(j-1)+48*(k-1)]
#define bope(i,j) dev_bope[(i-1)+6*(j-1)]
#define ghcof(i) dev_ghcof[(i-1)]
   const float_sw4 a1   = 0;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 tf   = 0.75;
   const float_sw4 c1   = 2.0/3;
   const float_sw4 c2   = -1.0/12;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = nij*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = base-nijk;
   const int base4 = base-nijk;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof;
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if( k < 1 || k > 6 )
      return;
   if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
// 5 ops                  
               float_sw4 ijac   = strx(i)*stry(j)/jac(i,j,k);
               float_sw4 istry  = 1./(stry(j));
               float_sw4 istrx  = 1./(strx(i));
	       float_sw4 istrxy = istry*istrx;

               float_sw4 r1 = 0.,r2 = 0.,r3 = 0.;

       // pp derivative (u) (u-eq)
// 53 ops, tot=58
	       float_sw4 cof1=(2*mu(i-2,j,k)+la(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)
		  *strx(i-2);
	       float_sw4 cof2=(2*mu(i-1,j,k)+la(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)
		  *strx(i-1);
	       float_sw4 cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	       float_sw4 cof4=(2*mu(i+1,j,k)+la(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)
		  *strx(i+1);
	       float_sw4 cof5=(2*mu(i+2,j,k)+la(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)
		  *strx(i+2);

	       float_sw4 mux1 = cof2 -tf*(cof3+cof1);
	       float_sw4 mux2 = cof1 + cof4+3*(cof3+cof2);
	       float_sw4 mux3 = cof2 + cof5+3*(cof4+cof3);
	       float_sw4 mux4 = cof4-tf*(cof3+cof5);

	       r1 = r1 + i6* (
			      mux1*(u(1,i-2,j,k)-u(1,i,j,k)) + 
			      mux2*(u(1,i-1,j,k)-u(1,i,j,k)) + 
			      mux3*(u(1,i+1,j,k)-u(1,i,j,k)) +
			      mux4*(u(1,i+2,j,k)-u(1,i,j,k))  )*istry;

	       // qq derivative (u) (u-eq)
// 43 ops, tot=101
	       cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	       cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	       cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	       cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	       cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);

	       mux1 = cof2 -tf*(cof3+cof1);
	       mux2 = cof1 + cof4+3*(cof3+cof2);
	       mux3 = cof2 + cof5+3*(cof4+cof3);
	       mux4 = cof4-tf*(cof3+cof5);

	       r1 = r1 + i6* (
                    mux1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                    mux2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                    mux3*(u(1,i,j+1,k)-u(1,i,j,k)) +
                    mux4*(u(1,i,j+2,k)-u(1,i,j,k))  )*istrx;

	       // pp derivative (v) (v-eq)
// 43 ops, tot=144
	       cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	       cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	       cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	       cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	       cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

	       mux1 = cof2 -tf*(cof3+cof1);
	       mux2 = cof1 + cof4+3*(cof3+cof2);
	       mux3 = cof2 + cof5+3*(cof4+cof3);
	       mux4 = cof4-tf*(cof3+cof5);

	       r2 = r2 + i6* (
                    mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                    mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                    mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                    mux4*(u(2,i+2,j,k)-u(2,i,j,k))  )*istry;

	       // qq derivative (v) (v-eq)
// 53 ops, tot=197
	       cof1=(2*mu(i,j-2,k)+la(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	       cof2=(2*mu(i,j-1,k)+la(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	       cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	       cof4=(2*mu(i,j+1,k)+la(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	       cof5=(2*mu(i,j+2,k)+la(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
	       mux1 = cof2 -tf*(cof3+cof1);
	       mux2 = cof1 + cof4+3*(cof3+cof2);
	       mux3 = cof2 + cof5+3*(cof4+cof3);
	       mux4 = cof4-tf*(cof3+cof5);

	       r2 = r2 + i6* (
                    mux1*(u(2,i,j-2,k)-u(2,i,j,k)) + 
                    mux2*(u(2,i,j-1,k)-u(2,i,j,k)) + 
                    mux3*(u(2,i,j+1,k)-u(2,i,j,k)) +
                    mux4*(u(2,i,j+2,k)-u(2,i,j,k))  )*istrx;

	       // pp derivative (w) (w-eq)
// 43 ops, tot=240
	       cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	       cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	       cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	       cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	       cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

	       mux1 = cof2 -tf*(cof3+cof1);
	       mux2 = cof1 + cof4+3*(cof3+cof2);
	       mux3 = cof2 + cof5+3*(cof4+cof3);
	       mux4 = cof4-tf*(cof3+cof5);

	       r3 = r3 + i6* (
                    mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                    mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                    mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                    mux4*(u(3,i+2,j,k)-u(3,i,j,k))  )*istry;

	       // qq derivative (w) (w-eq)
// 43 ops, tot=283
	       cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	       cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	       cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	       cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	       cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
	       mux1 = cof2 -tf*(cof3+cof1);
	       mux2 = cof1 + cof4+3*(cof3+cof2);
	       mux3 = cof2 + cof5+3*(cof4+cof3);
	       mux4 = cof4-tf*(cof3+cof5);

	       r3 = r3 + i6* (
                    mux1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                    mux2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                    mux3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                    mux4*(u(3,i,j+2,k)-u(3,i,j,k))  )*istrx;


	       // All rr-derivatives at once
	       // averaging the coefficient
// 54*8*8+25*8 = 3656 ops, tot=3939
	       float_sw4 mucofu2, mucofuv, mucofuw, mucofvw, mucofv2, mucofw2;
	       for( int q=1 ; q <= 8 ; q++ )
	       {
		  mucofu2=0.;
		  mucofuv=0.;
		  mucofuw=0.;
		  mucofvw=0.;
		  mucofv2=0.;
		  mucofw2=0.;
		  for( int m=1 ; m <= 8 ; m++ )
		  {
		     mucofu2 += 
			acof(k,q,m)*((2*mu(i,j,m)+la(i,j,m))*(met(2,i,j,m)*strx(i))*(met(2,i,j,m)*strx(i))
			       + mu(i,j,m)*(met(3,i,j,m)*stry(j)*met(3,i,j,m)*stry(j)+
					    met(4,i,j,m)*met(4,i,j,m)));
		     mucofv2 += 
                  acof(k,q,m)*((2*mu(i,j,m)+la(i,j,m))*(met(3,i,j,m)*stry(j))*met(3,i,j,m)*stry(j)
			       + mu(i,j,m)*((met(2,i,j,m)*strx(i))*met(2,i,j,m)*strx(i)+
					    met(4,i,j,m)*met(4,i,j,m)));
		     mucofw2 += 
			acof(k,q,m)*((2*mu(i,j,m)+la(i,j,m))*met(4,i,j,m)*met(4,i,j,m)
                            + mu(i,j,m)*( met(2,i,j,m)*strx(i)*met(2,i,j,m)*strx(i)+
					  met(3,i,j,m)*stry(j)*met(3,i,j,m)*stry(j) ) );
		     mucofuv += acof(k,q,m)*(mu(i,j,m)+la(i,j,m))*met(2,i,j,m)*met(3,i,j,m);
		     mucofuw += acof(k,q,m)*(mu(i,j,m)+la(i,j,m))*met(2,i,j,m)*met(4,i,j,m);
		     mucofvw += acof(k,q,m)*(mu(i,j,m)+la(i,j,m))*met(3,i,j,m)*met(4,i,j,m);
		  }

	  // Computing the second derivative,
		  r1 += istrxy*mucofu2*u(1,i,j,q) + mucofuv*u(2,i,j,q) + istry*mucofuw*u(3,i,j,q);
		  r2 += mucofuv*u(1,i,j,q) + istrxy*mucofv2*u(2,i,j,q) + istrx*mucofvw*u(3,i,j,q);
		  r3 += istry*mucofuw*u(1,i,j,q) + istrx*mucofvw*u(2,i,j,q) + istrxy*mucofw2*u(3,i,j,q);
	       }

	       // Ghost point values, only nonzero for k=1.
// 72 ops., tot=4011
	       mucofu2 = ghcof(k)*((2*mu(i,j,1)+la(i,j,1))*
				   met(2,i,j,1)*strx(i)*met(2,i,j,1)*strx(i)
				   + mu(i,j,1)*(met(3,i,j,1)*stry(j)*met(3,i,j,1)*stry(j)+
						met(4,i,j,1)*met(4,i,j,1) ));
	       mucofv2 = ghcof(k)*((2*mu(i,j,1)+la(i,j,1))*
                                met(3,i,j,1)*stry(j)*met(3,i,j,1)*stry(j)
				   + mu(i,j,1)*( met(2,i,j,1)*strx(i)*met(2,i,j,1)*strx(i)+
						 met(4,i,j,1)*met(4,i,j,1) ) );
	       mucofw2 = ghcof(k)*((2*mu(i,j,1)+la(i,j,1))*met(4,i,j,1)*met(4,i,j,1)
                                  + mu(i,j,1)*
				( met(2,i,j,1)*strx(i)*met(2,i,j,1)*strx(i)+
				  met(3,i,j,1)*stry(j)*met(3,i,j,1)*stry(j) ) );
	       mucofuv = ghcof(k)*(mu(i,j,1)+la(i,j,1))*met(2,i,j,1)*met(3,i,j,1);
	       mucofuw = ghcof(k)*(mu(i,j,1)+la(i,j,1))*met(2,i,j,1)*met(4,i,j,1);
	       mucofvw = ghcof(k)*(mu(i,j,1)+la(i,j,1))*met(3,i,j,1)*met(4,i,j,1);
	       r1 += istrxy*mucofu2*u(1,i,j,0) + mucofuv*u(2,i,j,0) + istry*mucofuw*u(3,i,j,0);
	       r2 += mucofuv*u(1,i,j,0) + istrxy*mucofv2*u(2,i,j,0) + istrx*mucofvw*u(3,i,j,0);
	       r3 += istry*mucofuw*u(1,i,j,0) + istrx*mucofvw*u(2,i,j,0) + istrxy*mucofw2*u(3,i,j,0);

	       // pq-derivatives (u-eq)
// 38 ops., tot=4049
	       r1 += 
        c2*(  mu(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(2,i+2,j+2,k)-u(2,i-2,j+2,k)) +
             c1*(u(2,i+1,j+2,k)-u(2,i-1,j+2,k))    )
          - mu(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(2,i+2,j-2,k)-u(2,i-2,j-2,k))+
             c1*(u(2,i+1,j-2,k)-u(2,i-1,j-2,k))     )
          ) +
        c1*(  mu(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(u(2,i+2,j+1,k)-u(2,i-2,j+1,k)) +
               c1*(u(2,i+1,j+1,k)-u(2,i-1,j+1,k))  )
           - mu(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(u(2,i+2,j-1,k)-u(2,i-2,j-1,k)) + 
               c1*(u(2,i+1,j-1,k)-u(2,i-1,j-1,k))));

	       // qp-derivatives (u-eq)
// 38 ops. tot=4087
	       r1 += 
        c2*(  la(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(2,i+2,j+2,k)-u(2,i+2,j-2,k)) +
             c1*(u(2,i+2,j+1,k)-u(2,i+2,j-1,k))    )
          - la(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(2,i-2,j+2,k)-u(2,i-2,j-2,k))+
             c1*(u(2,i-2,j+1,k)-u(2,i-2,j-1,k))     )
          ) +
        c1*(  la(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(u(2,i+1,j+2,k)-u(2,i+1,j-2,k)) +
               c1*(u(2,i+1,j+1,k)-u(2,i+1,j-1,k))  )
           - la(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(u(2,i-1,j+2,k)-u(2,i-1,j-2,k)) + 
               c1*(u(2,i-1,j+1,k)-u(2,i-1,j-1,k))));

      // pq-derivatives (v-eq)
// 38 ops. , tot=4125
	       r2 += 
        c2*(  la(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(1,i+2,j+2,k)-u(1,i-2,j+2,k)) +
             c1*(u(1,i+1,j+2,k)-u(1,i-1,j+2,k))    )
          - la(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(1,i+2,j-2,k)-u(1,i-2,j-2,k))+
             c1*(u(1,i+1,j-2,k)-u(1,i-1,j-2,k))     )
          ) +
        c1*(  la(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(u(1,i+2,j+1,k)-u(1,i-2,j+1,k)) +
               c1*(u(1,i+1,j+1,k)-u(1,i-1,j+1,k))  )
           - la(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(u(1,i+2,j-1,k)-u(1,i-2,j-1,k)) + 
	       c1*(u(1,i+1,j-1,k)-u(1,i-1,j-1,k))));

      //* qp-derivatives (v-eq)
// 38 ops., tot=4163
	       r2 += 
        c2*(  mu(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(1,i+2,j+2,k)-u(1,i+2,j-2,k)) +
             c1*(u(1,i+2,j+1,k)-u(1,i+2,j-1,k))    )
          - mu(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(1,i-2,j+2,k)-u(1,i-2,j-2,k))+
             c1*(u(1,i-2,j+1,k)-u(1,i-2,j-1,k))     )
          ) +
        c1*(  mu(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(u(1,i+1,j+2,k)-u(1,i+1,j-2,k)) +
               c1*(u(1,i+1,j+1,k)-u(1,i+1,j-1,k))  )
           - mu(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(u(1,i-1,j+2,k)-u(1,i-1,j-2,k)) + 
               c1*(u(1,i-1,j+1,k)-u(1,i-1,j-1,k))));

      // rp - derivatives
// 24*8 = 192 ops, tot=4355
	       float_sw4 dudrm2 = 0., dudrm1=0., dudrp1=0., dudrp2=0.;
	       float_sw4 dvdrm2 = 0., dvdrm1=0., dvdrp1=0., dvdrp2=0.;
	       float_sw4 dwdrm2 = 0., dwdrm1=0., dwdrp1=0., dwdrp2=0.;
	       for( int q=1 ; q <= 8 ; q++ )
	       {
		  dudrm2 += bope(k,q)*u(1,i-2,j,q);
		  dvdrm2 += bope(k,q)*u(2,i-2,j,q);
		  dwdrm2 += bope(k,q)*u(3,i-2,j,q);
		  dudrm1 += bope(k,q)*u(1,i-1,j,q);
		  dvdrm1 += bope(k,q)*u(2,i-1,j,q);
		  dwdrm1 += bope(k,q)*u(3,i-1,j,q);
		  dudrp2 += bope(k,q)*u(1,i+2,j,q);
		  dvdrp2 += bope(k,q)*u(2,i+2,j,q);
		  dwdrp2 += bope(k,q)*u(3,i+2,j,q);
		  dudrp1 += bope(k,q)*u(1,i+1,j,q);
		  dvdrp1 += bope(k,q)*u(2,i+1,j,q);
		  dwdrp1 += bope(k,q)*u(3,i+1,j,q);
	       }

	       // rp derivatives (u-eq)
// 67 ops, tot=4422
	       r1 +=  ( c2*(
       (2*mu(i+2,j,k)+la(i+2,j,k))*met(2,i+2,j,k)*met(1,i+2,j,k)*
                                                      strx(i+2)*dudrp2
        + la(i+2,j,k)*met(3,i+2,j,k)*met(1,i+2,j,k)*dvdrp2*stry(j)
        + la(i+2,j,k)*met(4,i+2,j,k)*met(1,i+2,j,k)*dwdrp2
     -((2*mu(i-2,j,k)+la(i-2,j,k))*met(2,i-2,j,k)*met(1,i-2,j,k)*
                                                      strx(i-2)*dudrm2
        + la(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*dvdrm2*stry(j)
        + la(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*dwdrm2 )
                   ) + c1*(  
       (2*mu(i+1,j,k)+la(i+1,j,k))*met(2,i+1,j,k)*met(1,i+1,j,k)*
                                                      strx(i+1)*dudrp1
        + la(i+1,j,k)*met(3,i+1,j,k)*met(1,i+1,j,k)*dvdrp1*stry(j)
        + la(i+1,j,k)*met(4,i+1,j,k)*met(1,i+1,j,k)*dwdrp1 
     -((2*mu(i-1,j,k)+la(i-1,j,k))*met(2,i-1,j,k)*met(1,i-1,j,k)*
                                                      strx(i-1)*dudrm1
        + la(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*dvdrm1*stry(j)
       + la(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*dwdrm1 ) ) )*istry;

	       // rp derivatives (v-eq)
// 42 ops, tot=4464
	       r2 += c2*(
          mu(i+2,j,k)*met(3,i+2,j,k)*met(1,i+2,j,k)*dudrp2
       +  mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*dvdrp2*
                                                strx(i+2)*istry
       - (mu(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*dudrm2
       +  mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*dvdrm2*
                                                strx(i-2)*istry )
                  ) + c1*(  
          mu(i+1,j,k)*met(3,i+1,j,k)*met(1,i+1,j,k)*dudrp1
       +  mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*dvdrp1*
                                                strx(i+1)*istry
       - (mu(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*dudrm1
       +  mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*dvdrm1*
                                                strx(i-1)*istry )
			    );

      // rp derivatives (w-eq)
// 38 ops, tot=4502
	       r3 += istry*(c2*(
          mu(i+2,j,k)*met(4,i+2,j,k)*met(1,i+2,j,k)*dudrp2
       +  mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*dwdrp2*strx(i+2)
       - (mu(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*dudrm2
       +  mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*dwdrm2*strx(i-2))
                  ) + c1*(  
          mu(i+1,j,k)*met(4,i+1,j,k)*met(1,i+1,j,k)*dudrp1
       +  mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*dwdrp1*strx(i+1)
       - (mu(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*dudrm1
       +  mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*dwdrm1*strx(i-1))
			    ) );

      // rq - derivatives
// 24*8 = 192 ops , tot=4694

	       dudrm2 = 0.;
	       dudrm1 = 0.;
	       dudrp1 = 0.;
	       dudrp2 = 0.;
	       dvdrm2 = 0.;
	       dvdrm1 = 0.;
	       dvdrp1 = 0.;
	       dvdrp2 = 0.;
	       dwdrm2 = 0.;
	       dwdrm1 = 0.;
	       dwdrp1 = 0.;
	       dwdrp2 = 0.;
	       for( int q=1 ; q <= 8 ; q++ )
	       {
		  dudrm2 += bope(k,q)*u(1,i,j-2,q);
		  dvdrm2 += bope(k,q)*u(2,i,j-2,q);
		  dwdrm2 += bope(k,q)*u(3,i,j-2,q);
		  dudrm1 += bope(k,q)*u(1,i,j-1,q);
		  dvdrm1 += bope(k,q)*u(2,i,j-1,q);
		  dwdrm1 += bope(k,q)*u(3,i,j-1,q);
		  dudrp2 += bope(k,q)*u(1,i,j+2,q);
		  dvdrp2 += bope(k,q)*u(2,i,j+2,q);
		  dwdrp2 += bope(k,q)*u(3,i,j+2,q);
		  dudrp1 += bope(k,q)*u(1,i,j+1,q);
		  dvdrp1 += bope(k,q)*u(2,i,j+1,q);
		  dwdrp1 += bope(k,q)*u(3,i,j+1,q);
	       }

	       // rq derivatives (u-eq)
// 42 ops, tot=4736
	       r1 += c2*(
           mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*dudrp2*
                                                stry(j+2)*istrx
        +  mu(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*dvdrp2
        - (mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*dudrm2*
                                                stry(j-2)*istrx
        +  mu(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*dvdrm2)
                  ) + c1*(  
           mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*dudrp1*
                                                stry(j+1)*istrx
        +  mu(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*dvdrp1
        - (mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*dudrm1*
                                                stry(j-1)*istrx
        +  mu(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*dvdrm1)
			    );

	       // rq derivatives (v-eq)
// 70 ops, tot=4806
	       r2 += c2*(
           la(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*dudrp2
      +(2*mu(i,j+2,k)+la(i,j+2,k))*met(3,i,j+2,k)*met(1,i,j+2,k)*dvdrp2
                                                   *stry(j+2)*istrx
         + la(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*dwdrp2*istrx
       - ( la(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*dudrm2
      +(2*mu(i,j-2,k)+la(i,j-2,k))*met(3,i,j-2,k)*met(1,i,j-2,k)*dvdrm2
                                                   *stry(j-2)*istrx
         + la(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*dwdrm2*istrx )
                  ) + c1*(  
           la(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*dudrp1
      +(2*mu(i,j+1,k)+la(i,j+1,k))*met(3,i,j+1,k)*met(1,i,j+1,k)*dvdrp1
                                                   *stry(j+1)*istrx
         + la(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*dwdrp1*istrx
       - ( la(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*dudrm1
      +(2*mu(i,j-1,k)+la(i,j-1,k))*met(3,i,j-1,k)*met(1,i,j-1,k)*dvdrm1
                                                   *stry(j-1)*istrx
         + la(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*dwdrm1*istrx )
			    );

	       // rq derivatives (w-eq)
// 39 ops, tot=4845
	       r3 += ( c2*(
          mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*dwdrp2*stry(j+2)
       +  mu(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*dvdrp2
       - (mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*dwdrm2*stry(j-2)
       +  mu(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*dvdrm2)
                  ) + c1*(  
          mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*dwdrp1*stry(j+1)
       +  mu(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*dvdrp1
       - (mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*dwdrm1*stry(j-1)
       +  mu(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*dvdrm1)
			    ) )*istrx;

	       // pr and qr derivatives at once
// in loop: 8*(53+53+43) = 1192 ops, tot=6037
	       for( int q=1 ; q <= 8 ; q++ )
	       {
		  // (u-eq)
// 53 ops
		  r1 += bope(k,q)*( 
				   // pr
        (2*mu(i,j,q)+la(i,j,q))*met(2,i,j,q)*met(1,i,j,q)*(
               c2*(u(1,i+2,j,q)-u(1,i-2,j,q)) +
               c1*(u(1,i+1,j,q)-u(1,i-1,j,q))   )*strx(i)*istry
       + mu(i,j,q)*met(3,i,j,q)*met(1,i,j,q)*(
             c2*(u(2,i+2,j,q)-u(2,i-2,j,q)) +
             c1*(u(2,i+1,j,q)-u(2,i-1,j,q))  ) 
       + mu(i,j,q)*met(4,i,j,q)*met(1,i,j,q)*(
             c2*(u(3,i+2,j,q)-u(3,i-2,j,q)) +
             c1*(u(3,i+1,j,q)-u(3,i-1,j,q))  )*istry 
	// qr
       + mu(i,j,q)*met(3,i,j,q)*met(1,i,j,q)*(
             c2*(u(1,i,j+2,q)-u(1,i,j-2,q)) +
             c1*(u(1,i,j+1,q)-u(1,i,j-1,q))   )*stry(j)*istrx
       + la(i,j,q)*met(2,i,j,q)*met(1,i,j,q)*(
             c2*(u(2,i,j+2,q)-u(2,i,j-2,q)) +
             c1*(u(2,i,j+1,q)-u(2,i,j-1,q))  ) );

	       // (v-eq)
// 53 ops
		  r2 += bope(k,q)*(
			 // pr
         la(i,j,q)*met(3,i,j,q)*met(1,i,j,q)*(
             c2*(u(1,i+2,j,q)-u(1,i-2,j,q)) +
             c1*(u(1,i+1,j,q)-u(1,i-1,j,q))   ) 
       + mu(i,j,q)*met(2,i,j,q)*met(1,i,j,q)*(
             c2*(u(2,i+2,j,q)-u(2,i-2,j,q)) +
             c1*(u(2,i+1,j,q)-u(2,i-1,j,q))  )*strx(i)*istry 
     // qr
       + mu(i,j,q)*met(2,i,j,q)*met(1,i,j,q)*(
             c2*(u(1,i,j+2,q)-u(1,i,j-2,q)) +
             c1*(u(1,i,j+1,q)-u(1,i,j-1,q))   ) 
      + (2*mu(i,j,q)+la(i,j,q))*met(3,i,j,q)*met(1,i,j,q)*(
             c2*(u(2,i,j+2,q)-u(2,i,j-2,q)) +
             c1*(u(2,i,j+1,q)-u(2,i,j-1,q))  )*stry(j)*istrx 
       + mu(i,j,q)*met(4,i,j,q)*met(1,i,j,q)*(
             c2*(u(3,i,j+2,q)-u(3,i,j-2,q)) +
             c1*(u(3,i,j+1,q)-u(3,i,j-1,q))   )*istrx  );

	// (w-eq)
// 43 ops
		  r3 += bope(k,q)*(
			     // pr
         la(i,j,q)*met(4,i,j,q)*met(1,i,j,q)*(
             c2*(u(1,i+2,j,q)-u(1,i-2,j,q)) +
             c1*(u(1,i+1,j,q)-u(1,i-1,j,q))   )*istry 
       + mu(i,j,q)*met(2,i,j,q)*met(1,i,j,q)*(
             c2*(u(3,i+2,j,q)-u(3,i-2,j,q)) +
             c1*(u(3,i+1,j,q)-u(3,i-1,j,q))  )*strx(i)*istry
	// qr 
       + mu(i,j,q)*met(3,i,j,q)*met(1,i,j,q)*(
             c2*(u(3,i,j+2,q)-u(3,i,j-2,q)) +
             c1*(u(3,i,j+1,q)-u(3,i,j-1,q))   )*stry(j)*istrx 
       + la(i,j,q)*met(4,i,j,q)*met(1,i,j,q)*(
             c2*(u(2,i,j+2,q)-u(2,i,j-2,q)) +
             c1*(u(2,i,j+1,q)-u(2,i,j-1,q))  )*istrx );
	       }

// 12 ops, tot=6049
	       lu(1,i,j,k) = a1*lu(1,i,j,k) + r1*ijac;
	       lu(2,i,j,k) = a1*lu(2,i,j,k) + r2*ijac;
	       lu(3,i,j,k) = a1*lu(3,i,j,k) + r3*ijac;
   }

#undef mu
#undef la
#undef jac
#undef u
#undef lu
#undef met
#undef strx
#undef stry
#undef acof
#undef bope
#undef ghcof
}
//-----------------------------------------------------------------------
__global__ void rhs4sgcurvupper_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_met, float_sw4* a_jac, float_sw4* a_lu, 
				float_sw4* a_strx, float_sw4* a_stry, int ghost_points )
{
 // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+i+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+i+ni*(j)+nij*(k)]
#define jac(i,j,k)   a_jac[base+i+ni*(j)+nij*(k)]
#define u(c,i,j,k)     a_u[base3+(c)+3*(i)+ni3*(j)+nij3*(k)]   
#define lu(c,i,j,k)   a_lu[base3+(c)+3*(i)+ni3*(j)+nij3*(k)]   
#define met(c,i,j,k) a_met[base4+(c)+4*(i)+ni4*(j)+nij4*(k)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define acof(i,j,k) dev_acof[(i-1)+6*(j-1)+48*(k-1)]
#define bope(i,j) dev_bope[(i-1)+6*(j-1)]
#define ghcof(i) dev_ghcof[(i-1)]
   const float_sw4 a1   = 0.;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 tf   = 0.75;
   const float_sw4 c1   = 2.0/3;
   const float_sw4 c2   = -1.0/12;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = nij*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = 3*base-1;
   const int base4 = 4*base-1;
   const int ni3  = 3*ni;
   const int nij3 = 3*nij;
   const int ni4  = 4*ni;
   const int nij4 = 4*nij;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof;
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if( k < 1 || k > 6 )
      return;
   if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
// 5 ops                  
               float_sw4 ijac   = strx(i)*stry(j)/jac(i,j,k);
               float_sw4 istry  = 1./(stry(j));
               float_sw4 istrx  = 1./(strx(i));
	       float_sw4 istrxy = istry*istrx;

               float_sw4 r1 = 0.,r2 = 0.,r3 = 0.;

       // pp derivative (u) (u-eq)
// 53 ops, tot=58
	       float_sw4 cof1=(2*mu(i-2,j,k)+la(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)
		  *strx(i-2);
	       float_sw4 cof2=(2*mu(i-1,j,k)+la(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)
		  *strx(i-1);
	       float_sw4 cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	       float_sw4 cof4=(2*mu(i+1,j,k)+la(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)
		  *strx(i+1);
	       float_sw4 cof5=(2*mu(i+2,j,k)+la(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)
		  *strx(i+2);

	       float_sw4 mux1 = cof2 -tf*(cof3+cof1);
	       float_sw4 mux2 = cof1 + cof4+3*(cof3+cof2);
	       float_sw4 mux3 = cof2 + cof5+3*(cof4+cof3);
	       float_sw4 mux4 = cof4-tf*(cof3+cof5);

	       r1 = r1 + i6* (
			      mux1*(u(1,i-2,j,k)-u(1,i,j,k)) + 
			      mux2*(u(1,i-1,j,k)-u(1,i,j,k)) + 
			      mux3*(u(1,i+1,j,k)-u(1,i,j,k)) +
			      mux4*(u(1,i+2,j,k)-u(1,i,j,k))  )*istry;

	       // qq derivative (u) (u-eq)
// 43 ops, tot=101
	       cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	       cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	       cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	       cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	       cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);

	       mux1 = cof2 -tf*(cof3+cof1);
	       mux2 = cof1 + cof4+3*(cof3+cof2);
	       mux3 = cof2 + cof5+3*(cof4+cof3);
	       mux4 = cof4-tf*(cof3+cof5);

	       r1 = r1 + i6* (
                    mux1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                    mux2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                    mux3*(u(1,i,j+1,k)-u(1,i,j,k)) +
                    mux4*(u(1,i,j+2,k)-u(1,i,j,k))  )*istrx;

	       // pp derivative (v) (v-eq)
// 43 ops, tot=144
	       cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	       cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	       cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	       cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	       cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

	       mux1 = cof2 -tf*(cof3+cof1);
	       mux2 = cof1 + cof4+3*(cof3+cof2);
	       mux3 = cof2 + cof5+3*(cof4+cof3);
	       mux4 = cof4-tf*(cof3+cof5);

	       r2 = r2 + i6* (
                    mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                    mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                    mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                    mux4*(u(2,i+2,j,k)-u(2,i,j,k))  )*istry;

	       // qq derivative (v) (v-eq)
// 53 ops, tot=197
	       cof1=(2*mu(i,j-2,k)+la(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	       cof2=(2*mu(i,j-1,k)+la(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	       cof3=(2*mu(i,j,k)+la(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	       cof4=(2*mu(i,j+1,k)+la(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	       cof5=(2*mu(i,j+2,k)+la(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
	       mux1 = cof2 -tf*(cof3+cof1);
	       mux2 = cof1 + cof4+3*(cof3+cof2);
	       mux3 = cof2 + cof5+3*(cof4+cof3);
	       mux4 = cof4-tf*(cof3+cof5);

	       r2 = r2 + i6* (
                    mux1*(u(2,i,j-2,k)-u(2,i,j,k)) + 
                    mux2*(u(2,i,j-1,k)-u(2,i,j,k)) + 
                    mux3*(u(2,i,j+1,k)-u(2,i,j,k)) +
                    mux4*(u(2,i,j+2,k)-u(2,i,j,k))  )*istrx;

	       // pp derivative (w) (w-eq)
// 43 ops, tot=240
	       cof1=(mu(i-2,j,k))*met(1,i-2,j,k)*met(1,i-2,j,k)*strx(i-2);
	       cof2=(mu(i-1,j,k))*met(1,i-1,j,k)*met(1,i-1,j,k)*strx(i-1);
	       cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*strx(i);
	       cof4=(mu(i+1,j,k))*met(1,i+1,j,k)*met(1,i+1,j,k)*strx(i+1);
	       cof5=(mu(i+2,j,k))*met(1,i+2,j,k)*met(1,i+2,j,k)*strx(i+2);

	       mux1 = cof2 -tf*(cof3+cof1);
	       mux2 = cof1 + cof4+3*(cof3+cof2);
	       mux3 = cof2 + cof5+3*(cof4+cof3);
	       mux4 = cof4-tf*(cof3+cof5);

	       r3 = r3 + i6* (
                    mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                    mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                    mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                    mux4*(u(3,i+2,j,k)-u(3,i,j,k))  )*istry;

	       // qq derivative (w) (w-eq)
// 43 ops, tot=283
	       cof1=(mu(i,j-2,k))*met(1,i,j-2,k)*met(1,i,j-2,k)*stry(j-2);
	       cof2=(mu(i,j-1,k))*met(1,i,j-1,k)*met(1,i,j-1,k)*stry(j-1);
	       cof3=(mu(i,j,k))*met(1,i,j,k)*met(1,i,j,k)*stry(j);
	       cof4=(mu(i,j+1,k))*met(1,i,j+1,k)*met(1,i,j+1,k)*stry(j+1);
	       cof5=(mu(i,j+2,k))*met(1,i,j+2,k)*met(1,i,j+2,k)*stry(j+2);
	       mux1 = cof2 -tf*(cof3+cof1);
	       mux2 = cof1 + cof4+3*(cof3+cof2);
	       mux3 = cof2 + cof5+3*(cof4+cof3);
	       mux4 = cof4-tf*(cof3+cof5);

	       r3 = r3 + i6* (
                    mux1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                    mux2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                    mux3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                    mux4*(u(3,i,j+2,k)-u(3,i,j,k))  )*istrx;


	       // All rr-derivatives at once
	       // averaging the coefficient
// 54*8*8+25*8 = 3656 ops, tot=3939
	       float_sw4 mucofu2, mucofuv, mucofuw, mucofvw, mucofv2, mucofw2;
	       for( int q=1 ; q <= 8 ; q++ )
	       {
		  mucofu2=0.;
		  mucofuv=0.;
		  mucofuw=0.;
		  mucofvw=0.;
		  mucofv2=0.;
		  mucofw2=0.;
		  for( int m=1 ; m <= 8 ; m++ )
		  {
		     mucofu2 += 
			acof(k,q,m)*((2*mu(i,j,m)+la(i,j,m))*(met(2,i,j,m)*strx(i))*(met(2,i,j,m)*strx(i))
			       + mu(i,j,m)*(met(3,i,j,m)*stry(j)*met(3,i,j,m)*stry(j)+
					    met(4,i,j,m)*met(4,i,j,m)));
		     mucofv2 += 
                  acof(k,q,m)*((2*mu(i,j,m)+la(i,j,m))*(met(3,i,j,m)*stry(j))*met(3,i,j,m)*stry(j)
			       + mu(i,j,m)*((met(2,i,j,m)*strx(i))*met(2,i,j,m)*strx(i)+
					    met(4,i,j,m)*met(4,i,j,m)));
		     mucofw2 += 
			acof(k,q,m)*((2*mu(i,j,m)+la(i,j,m))*met(4,i,j,m)*met(4,i,j,m)
                            + mu(i,j,m)*( met(2,i,j,m)*strx(i)*met(2,i,j,m)*strx(i)+
					  met(3,i,j,m)*stry(j)*met(3,i,j,m)*stry(j) ) );
		     mucofuv += acof(k,q,m)*(mu(i,j,m)+la(i,j,m))*met(2,i,j,m)*met(3,i,j,m);
		     mucofuw += acof(k,q,m)*(mu(i,j,m)+la(i,j,m))*met(2,i,j,m)*met(4,i,j,m);
		     mucofvw += acof(k,q,m)*(mu(i,j,m)+la(i,j,m))*met(3,i,j,m)*met(4,i,j,m);
		  }

	  // Computing the second derivative,
		  r1 += istrxy*mucofu2*u(1,i,j,q) + mucofuv*u(2,i,j,q) + istry*mucofuw*u(3,i,j,q);
		  r2 += mucofuv*u(1,i,j,q) + istrxy*mucofv2*u(2,i,j,q) + istrx*mucofvw*u(3,i,j,q);
		  r3 += istry*mucofuw*u(1,i,j,q) + istrx*mucofvw*u(2,i,j,q) + istrxy*mucofw2*u(3,i,j,q);
	       }

	       // Ghost point values, only nonzero for k=1.
// 72 ops., tot=4011
	       mucofu2 = ghcof(k)*((2*mu(i,j,1)+la(i,j,1))*
				   met(2,i,j,1)*strx(i)*met(2,i,j,1)*strx(i)
				   + mu(i,j,1)*(met(3,i,j,1)*stry(j)*met(3,i,j,1)*stry(j)+
						met(4,i,j,1)*met(4,i,j,1) ));
	       mucofv2 = ghcof(k)*((2*mu(i,j,1)+la(i,j,1))*
                                met(3,i,j,1)*stry(j)*met(3,i,j,1)*stry(j)
				   + mu(i,j,1)*( met(2,i,j,1)*strx(i)*met(2,i,j,1)*strx(i)+
						 met(4,i,j,1)*met(4,i,j,1) ) );
	       mucofw2 = ghcof(k)*((2*mu(i,j,1)+la(i,j,1))*met(4,i,j,1)*met(4,i,j,1)
                                  + mu(i,j,1)*
				( met(2,i,j,1)*strx(i)*met(2,i,j,1)*strx(i)+
				  met(3,i,j,1)*stry(j)*met(3,i,j,1)*stry(j) ) );
	       mucofuv = ghcof(k)*(mu(i,j,1)+la(i,j,1))*met(2,i,j,1)*met(3,i,j,1);
	       mucofuw = ghcof(k)*(mu(i,j,1)+la(i,j,1))*met(2,i,j,1)*met(4,i,j,1);
	       mucofvw = ghcof(k)*(mu(i,j,1)+la(i,j,1))*met(3,i,j,1)*met(4,i,j,1);
	       r1 += istrxy*mucofu2*u(1,i,j,0) + mucofuv*u(2,i,j,0) + istry*mucofuw*u(3,i,j,0);
	       r2 += mucofuv*u(1,i,j,0) + istrxy*mucofv2*u(2,i,j,0) + istrx*mucofvw*u(3,i,j,0);
	       r3 += istry*mucofuw*u(1,i,j,0) + istrx*mucofvw*u(2,i,j,0) + istrxy*mucofw2*u(3,i,j,0);

	       // pq-derivatives (u-eq)
// 38 ops., tot=4049
	       r1 += 
        c2*(  mu(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(2,i+2,j+2,k)-u(2,i-2,j+2,k)) +
             c1*(u(2,i+1,j+2,k)-u(2,i-1,j+2,k))    )
          - mu(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(2,i+2,j-2,k)-u(2,i-2,j-2,k))+
             c1*(u(2,i+1,j-2,k)-u(2,i-1,j-2,k))     )
          ) +
        c1*(  mu(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(u(2,i+2,j+1,k)-u(2,i-2,j+1,k)) +
               c1*(u(2,i+1,j+1,k)-u(2,i-1,j+1,k))  )
           - mu(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(u(2,i+2,j-1,k)-u(2,i-2,j-1,k)) + 
               c1*(u(2,i+1,j-1,k)-u(2,i-1,j-1,k))));

	       // qp-derivatives (u-eq)
// 38 ops. tot=4087
	       r1 += 
        c2*(  la(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(2,i+2,j+2,k)-u(2,i+2,j-2,k)) +
             c1*(u(2,i+2,j+1,k)-u(2,i+2,j-1,k))    )
          - la(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(2,i-2,j+2,k)-u(2,i-2,j-2,k))+
             c1*(u(2,i-2,j+1,k)-u(2,i-2,j-1,k))     )
          ) +
        c1*(  la(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(u(2,i+1,j+2,k)-u(2,i+1,j-2,k)) +
               c1*(u(2,i+1,j+1,k)-u(2,i+1,j-1,k))  )
           - la(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(u(2,i-1,j+2,k)-u(2,i-1,j-2,k)) + 
               c1*(u(2,i-1,j+1,k)-u(2,i-1,j-1,k))));

      // pq-derivatives (v-eq)
// 38 ops. , tot=4125
	       r2 += 
        c2*(  la(i,j+2,k)*met(1,i,j+2,k)*met(1,i,j+2,k)*(
             c2*(u(1,i+2,j+2,k)-u(1,i-2,j+2,k)) +
             c1*(u(1,i+1,j+2,k)-u(1,i-1,j+2,k))    )
          - la(i,j-2,k)*met(1,i,j-2,k)*met(1,i,j-2,k)*(
             c2*(u(1,i+2,j-2,k)-u(1,i-2,j-2,k))+
             c1*(u(1,i+1,j-2,k)-u(1,i-1,j-2,k))     )
          ) +
        c1*(  la(i,j+1,k)*met(1,i,j+1,k)*met(1,i,j+1,k)*(
               c2*(u(1,i+2,j+1,k)-u(1,i-2,j+1,k)) +
               c1*(u(1,i+1,j+1,k)-u(1,i-1,j+1,k))  )
           - la(i,j-1,k)*met(1,i,j-1,k)*met(1,i,j-1,k)*(
               c2*(u(1,i+2,j-1,k)-u(1,i-2,j-1,k)) + 
	       c1*(u(1,i+1,j-1,k)-u(1,i-1,j-1,k))));

      //* qp-derivatives (v-eq)
// 38 ops., tot=4163
	       r2 += 
        c2*(  mu(i+2,j,k)*met(1,i+2,j,k)*met(1,i+2,j,k)*(
             c2*(u(1,i+2,j+2,k)-u(1,i+2,j-2,k)) +
             c1*(u(1,i+2,j+1,k)-u(1,i+2,j-1,k))    )
          - mu(i-2,j,k)*met(1,i-2,j,k)*met(1,i-2,j,k)*(
             c2*(u(1,i-2,j+2,k)-u(1,i-2,j-2,k))+
             c1*(u(1,i-2,j+1,k)-u(1,i-2,j-1,k))     )
          ) +
        c1*(  mu(i+1,j,k)*met(1,i+1,j,k)*met(1,i+1,j,k)*(
               c2*(u(1,i+1,j+2,k)-u(1,i+1,j-2,k)) +
               c1*(u(1,i+1,j+1,k)-u(1,i+1,j-1,k))  )
           - mu(i-1,j,k)*met(1,i-1,j,k)*met(1,i-1,j,k)*(
               c2*(u(1,i-1,j+2,k)-u(1,i-1,j-2,k)) + 
               c1*(u(1,i-1,j+1,k)-u(1,i-1,j-1,k))));

      // rp - derivatives
// 24*8 = 192 ops, tot=4355
	       float_sw4 dudrm2 = 0., dudrm1=0., dudrp1=0., dudrp2=0.;
	       float_sw4 dvdrm2 = 0., dvdrm1=0., dvdrp1=0., dvdrp2=0.;
	       float_sw4 dwdrm2 = 0., dwdrm1=0., dwdrp1=0., dwdrp2=0.;
	       for( int q=1 ; q <= 8 ; q++ )
	       {
		  dudrm2 += bope(k,q)*u(1,i-2,j,q);
		  dvdrm2 += bope(k,q)*u(2,i-2,j,q);
		  dwdrm2 += bope(k,q)*u(3,i-2,j,q);
		  dudrm1 += bope(k,q)*u(1,i-1,j,q);
		  dvdrm1 += bope(k,q)*u(2,i-1,j,q);
		  dwdrm1 += bope(k,q)*u(3,i-1,j,q);
		  dudrp2 += bope(k,q)*u(1,i+2,j,q);
		  dvdrp2 += bope(k,q)*u(2,i+2,j,q);
		  dwdrp2 += bope(k,q)*u(3,i+2,j,q);
		  dudrp1 += bope(k,q)*u(1,i+1,j,q);
		  dvdrp1 += bope(k,q)*u(2,i+1,j,q);
		  dwdrp1 += bope(k,q)*u(3,i+1,j,q);
	       }

	       // rp derivatives (u-eq)
// 67 ops, tot=4422
	       r1 +=  ( c2*(
       (2*mu(i+2,j,k)+la(i+2,j,k))*met(2,i+2,j,k)*met(1,i+2,j,k)*
                                                      strx(i+2)*dudrp2
        + la(i+2,j,k)*met(3,i+2,j,k)*met(1,i+2,j,k)*dvdrp2*stry(j)
        + la(i+2,j,k)*met(4,i+2,j,k)*met(1,i+2,j,k)*dwdrp2
     -((2*mu(i-2,j,k)+la(i-2,j,k))*met(2,i-2,j,k)*met(1,i-2,j,k)*
                                                      strx(i-2)*dudrm2
        + la(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*dvdrm2*stry(j)
        + la(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*dwdrm2 )
                   ) + c1*(  
       (2*mu(i+1,j,k)+la(i+1,j,k))*met(2,i+1,j,k)*met(1,i+1,j,k)*
                                                      strx(i+1)*dudrp1
        + la(i+1,j,k)*met(3,i+1,j,k)*met(1,i+1,j,k)*dvdrp1*stry(j)
        + la(i+1,j,k)*met(4,i+1,j,k)*met(1,i+1,j,k)*dwdrp1 
     -((2*mu(i-1,j,k)+la(i-1,j,k))*met(2,i-1,j,k)*met(1,i-1,j,k)*
                                                      strx(i-1)*dudrm1
        + la(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*dvdrm1*stry(j)
       + la(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*dwdrm1 ) ) )*istry;

	       // rp derivatives (v-eq)
// 42 ops, tot=4464
	       r2 += c2*(
          mu(i+2,j,k)*met(3,i+2,j,k)*met(1,i+2,j,k)*dudrp2
       +  mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*dvdrp2*
                                                strx(i+2)*istry
       - (mu(i-2,j,k)*met(3,i-2,j,k)*met(1,i-2,j,k)*dudrm2
       +  mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*dvdrm2*
                                                strx(i-2)*istry )
                  ) + c1*(  
          mu(i+1,j,k)*met(3,i+1,j,k)*met(1,i+1,j,k)*dudrp1
       +  mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*dvdrp1*
                                                strx(i+1)*istry
       - (mu(i-1,j,k)*met(3,i-1,j,k)*met(1,i-1,j,k)*dudrm1
       +  mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*dvdrm1*
                                                strx(i-1)*istry )
			    );

      // rp derivatives (w-eq)
// 38 ops, tot=4502
	       r3 += istry*(c2*(
          mu(i+2,j,k)*met(4,i+2,j,k)*met(1,i+2,j,k)*dudrp2
       +  mu(i+2,j,k)*met(2,i+2,j,k)*met(1,i+2,j,k)*dwdrp2*strx(i+2)
       - (mu(i-2,j,k)*met(4,i-2,j,k)*met(1,i-2,j,k)*dudrm2
       +  mu(i-2,j,k)*met(2,i-2,j,k)*met(1,i-2,j,k)*dwdrm2*strx(i-2))
                  ) + c1*(  
          mu(i+1,j,k)*met(4,i+1,j,k)*met(1,i+1,j,k)*dudrp1
       +  mu(i+1,j,k)*met(2,i+1,j,k)*met(1,i+1,j,k)*dwdrp1*strx(i+1)
       - (mu(i-1,j,k)*met(4,i-1,j,k)*met(1,i-1,j,k)*dudrm1
       +  mu(i-1,j,k)*met(2,i-1,j,k)*met(1,i-1,j,k)*dwdrm1*strx(i-1))
			    ) );

      // rq - derivatives
// 24*8 = 192 ops , tot=4694

	       dudrm2 = 0.;
	       dudrm1 = 0.;
	       dudrp1 = 0.;
	       dudrp2 = 0.;
	       dvdrm2 = 0.;
	       dvdrm1 = 0.;
	       dvdrp1 = 0.;
	       dvdrp2 = 0.;
	       dwdrm2 = 0.;
	       dwdrm1 = 0.;
	       dwdrp1 = 0.;
	       dwdrp2 = 0.;
	       for( int q=1 ; q <= 8 ; q++ )
	       {
		  dudrm2 += bope(k,q)*u(1,i,j-2,q);
		  dvdrm2 += bope(k,q)*u(2,i,j-2,q);
		  dwdrm2 += bope(k,q)*u(3,i,j-2,q);
		  dudrm1 += bope(k,q)*u(1,i,j-1,q);
		  dvdrm1 += bope(k,q)*u(2,i,j-1,q);
		  dwdrm1 += bope(k,q)*u(3,i,j-1,q);
		  dudrp2 += bope(k,q)*u(1,i,j+2,q);
		  dvdrp2 += bope(k,q)*u(2,i,j+2,q);
		  dwdrp2 += bope(k,q)*u(3,i,j+2,q);
		  dudrp1 += bope(k,q)*u(1,i,j+1,q);
		  dvdrp1 += bope(k,q)*u(2,i,j+1,q);
		  dwdrp1 += bope(k,q)*u(3,i,j+1,q);
	       }

	       // rq derivatives (u-eq)
// 42 ops, tot=4736
	       r1 += c2*(
           mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*dudrp2*
                                                stry(j+2)*istrx
        +  mu(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*dvdrp2
        - (mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*dudrm2*
                                                stry(j-2)*istrx
        +  mu(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*dvdrm2)
                  ) + c1*(  
           mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*dudrp1*
                                                stry(j+1)*istrx
        +  mu(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*dvdrp1
        - (mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*dudrm1*
                                                stry(j-1)*istrx
        +  mu(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*dvdrm1)
			    );

	       // rq derivatives (v-eq)
// 70 ops, tot=4806
	       r2 += c2*(
           la(i,j+2,k)*met(2,i,j+2,k)*met(1,i,j+2,k)*dudrp2
      +(2*mu(i,j+2,k)+la(i,j+2,k))*met(3,i,j+2,k)*met(1,i,j+2,k)*dvdrp2
                                                   *stry(j+2)*istrx
         + la(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*dwdrp2*istrx
       - ( la(i,j-2,k)*met(2,i,j-2,k)*met(1,i,j-2,k)*dudrm2
      +(2*mu(i,j-2,k)+la(i,j-2,k))*met(3,i,j-2,k)*met(1,i,j-2,k)*dvdrm2
                                                   *stry(j-2)*istrx
         + la(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*dwdrm2*istrx )
                  ) + c1*(  
           la(i,j+1,k)*met(2,i,j+1,k)*met(1,i,j+1,k)*dudrp1
      +(2*mu(i,j+1,k)+la(i,j+1,k))*met(3,i,j+1,k)*met(1,i,j+1,k)*dvdrp1
                                                   *stry(j+1)*istrx
         + la(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*dwdrp1*istrx
       - ( la(i,j-1,k)*met(2,i,j-1,k)*met(1,i,j-1,k)*dudrm1
      +(2*mu(i,j-1,k)+la(i,j-1,k))*met(3,i,j-1,k)*met(1,i,j-1,k)*dvdrm1
                                                   *stry(j-1)*istrx
         + la(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*dwdrm1*istrx )
			    );

	       // rq derivatives (w-eq)
// 39 ops, tot=4845
	       r3 += ( c2*(
          mu(i,j+2,k)*met(3,i,j+2,k)*met(1,i,j+2,k)*dwdrp2*stry(j+2)
       +  mu(i,j+2,k)*met(4,i,j+2,k)*met(1,i,j+2,k)*dvdrp2
       - (mu(i,j-2,k)*met(3,i,j-2,k)*met(1,i,j-2,k)*dwdrm2*stry(j-2)
       +  mu(i,j-2,k)*met(4,i,j-2,k)*met(1,i,j-2,k)*dvdrm2)
                  ) + c1*(  
          mu(i,j+1,k)*met(3,i,j+1,k)*met(1,i,j+1,k)*dwdrp1*stry(j+1)
       +  mu(i,j+1,k)*met(4,i,j+1,k)*met(1,i,j+1,k)*dvdrp1
       - (mu(i,j-1,k)*met(3,i,j-1,k)*met(1,i,j-1,k)*dwdrm1*stry(j-1)
       +  mu(i,j-1,k)*met(4,i,j-1,k)*met(1,i,j-1,k)*dvdrm1)
			    ) )*istrx;

	       // pr and qr derivatives at once
// in loop: 8*(53+53+43) = 1192 ops, tot=6037
	       for( int q=1 ; q <= 8 ; q++ )
	       {
		  // (u-eq)
// 53 ops
		  r1 += bope(k,q)*( 
				   // pr
        (2*mu(i,j,q)+la(i,j,q))*met(2,i,j,q)*met(1,i,j,q)*(
               c2*(u(1,i+2,j,q)-u(1,i-2,j,q)) +
               c1*(u(1,i+1,j,q)-u(1,i-1,j,q))   )*strx(i)*istry
       + mu(i,j,q)*met(3,i,j,q)*met(1,i,j,q)*(
             c2*(u(2,i+2,j,q)-u(2,i-2,j,q)) +
             c1*(u(2,i+1,j,q)-u(2,i-1,j,q))  ) 
       + mu(i,j,q)*met(4,i,j,q)*met(1,i,j,q)*(
             c2*(u(3,i+2,j,q)-u(3,i-2,j,q)) +
             c1*(u(3,i+1,j,q)-u(3,i-1,j,q))  )*istry 
	// qr
       + mu(i,j,q)*met(3,i,j,q)*met(1,i,j,q)*(
             c2*(u(1,i,j+2,q)-u(1,i,j-2,q)) +
             c1*(u(1,i,j+1,q)-u(1,i,j-1,q))   )*stry(j)*istrx
       + la(i,j,q)*met(2,i,j,q)*met(1,i,j,q)*(
             c2*(u(2,i,j+2,q)-u(2,i,j-2,q)) +
             c1*(u(2,i,j+1,q)-u(2,i,j-1,q))  ) );

	       // (v-eq)
// 53 ops
		  r2 += bope(k,q)*(
			 // pr
         la(i,j,q)*met(3,i,j,q)*met(1,i,j,q)*(
             c2*(u(1,i+2,j,q)-u(1,i-2,j,q)) +
             c1*(u(1,i+1,j,q)-u(1,i-1,j,q))   ) 
       + mu(i,j,q)*met(2,i,j,q)*met(1,i,j,q)*(
             c2*(u(2,i+2,j,q)-u(2,i-2,j,q)) +
             c1*(u(2,i+1,j,q)-u(2,i-1,j,q))  )*strx(i)*istry 
     // qr
       + mu(i,j,q)*met(2,i,j,q)*met(1,i,j,q)*(
             c2*(u(1,i,j+2,q)-u(1,i,j-2,q)) +
             c1*(u(1,i,j+1,q)-u(1,i,j-1,q))   ) 
      + (2*mu(i,j,q)+la(i,j,q))*met(3,i,j,q)*met(1,i,j,q)*(
             c2*(u(2,i,j+2,q)-u(2,i,j-2,q)) +
             c1*(u(2,i,j+1,q)-u(2,i,j-1,q))  )*stry(j)*istrx 
       + mu(i,j,q)*met(4,i,j,q)*met(1,i,j,q)*(
             c2*(u(3,i,j+2,q)-u(3,i,j-2,q)) +
             c1*(u(3,i,j+1,q)-u(3,i,j-1,q))   )*istrx  );

	// (w-eq)
// 43 ops
		  r3 += bope(k,q)*(
			     // pr
         la(i,j,q)*met(4,i,j,q)*met(1,i,j,q)*(
             c2*(u(1,i+2,j,q)-u(1,i-2,j,q)) +
             c1*(u(1,i+1,j,q)-u(1,i-1,j,q))   )*istry 
       + mu(i,j,q)*met(2,i,j,q)*met(1,i,j,q)*(
             c2*(u(3,i+2,j,q)-u(3,i-2,j,q)) +
             c1*(u(3,i+1,j,q)-u(3,i-1,j,q))  )*strx(i)*istry
	// qr 
       + mu(i,j,q)*met(3,i,j,q)*met(1,i,j,q)*(
             c2*(u(3,i,j+2,q)-u(3,i,j-2,q)) +
             c1*(u(3,i,j+1,q)-u(3,i,j-1,q))   )*stry(j)*istrx 
       + la(i,j,q)*met(4,i,j,q)*met(1,i,j,q)*(
             c2*(u(2,i,j+2,q)-u(2,i,j-2,q)) +
             c1*(u(2,i,j+1,q)-u(2,i,j-1,q))  )*istrx );
	       }

// 12 ops, tot=6049
	       lu(1,i,j,k) = a1*lu(1,i,j,k) + r1*ijac;
	       lu(2,i,j,k) = a1*lu(2,i,j,k) + r2*ijac;
	       lu(3,i,j,k) = a1*lu(3,i,j,k) + r3*ijac;
   }
#undef mu
#undef la
#undef jac
#undef u
#undef lu
#undef met
#undef strx
#undef stry
#undef acof
#undef bope
#undef ghcof
}

//-----------------------------------------------------------------------
__global__ void 
__launch_bounds__(256,1)
rhs4center_dev_v2( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                       float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda,
                       float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
                       int ghost_points )
{
  // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+(i)+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+(i)+ni*(j)+nij*(k)]
#define u(c,i,j,k)   a_u[base3+c+3*(i)+nic*(j)+nijc*(k)]   
#define lu(c,i,j,k) a_lu[base3+c+3*(i)+nic*(j)+nijc*(k)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define strz(k) a_strz[k-kfirst0]
#define uSh(c,i,j,k) a_uSh[k][j][i][c-1]

  const float_sw4 a1   = 0.;
  const float_sw4 i6   = 1.0/6;
  //   const float_sw4 i12  = 1.0/12;
  const float_sw4 i144 = 1.0/144;
  const float_sw4 tf   = 0.75;

  const int ni    = ilast-ifirst+1;
  const int nij   = ni*(jlast-jfirst+1);
  const int nijk  = nij*(klast-kfirst+1);
  const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = 3*base-1;
  const int nic  = 3*ni;
  const int nijc = 3*nij;
  const int ifirst0 = ifirst;
  const int jfirst0 = jfirst;
  const int kfirst0 = kfirst;
  float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
  float_sw4 r1, r2, r3, cof;
  const int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
  const int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;

  __shared__ float_sw4 a_uSh[DIAMETER][RHS4_BLOCKY+2*RADIUS][RHS4_BLOCKX+2*RADIUS][3];

  const int ith = threadIdx.x + RADIUS;
  const int jth = threadIdx.y + RADIUS;
  //const int kth = threadIdx.z + RADIUS;

  int kthm0=3, kthm1=2, kthm2=1, kthm3=0, kthm4=4, kthtmp;

  for (int tk = 0; tk < DIAMETER; tk++)
  {
    int gk =  kfirst + tk;
    if ( gk <= klast)
    {
      for (int tj = threadIdx.y; tj < blockDim.y+2*RADIUS; tj += blockDim.y)
      {
        int gj =  (j-threadIdx.y) + tj - RADIUS;
        if ( gj <= jlast)
        {
          for (int ti = threadIdx.x; ti < blockDim.x+2*RADIUS; ti += blockDim.x)
          {
            int gi =  (i-threadIdx.x) + ti - RADIUS;
            if ( gi <= ilast)
            {
              uSh(1, ti, tj, tk) = u(1, gi, gj, gk);
              uSh(2, ti, tj, tk) = u(2, gi, gj, gk);
              uSh(3, ti, tj, tk) = u(3, gi, gj, gk);
            }
          }
        }
      }
    }
  }
  __syncthreads();

  cof = 1.0/(h*h);

  for (int k = kfirst+RADIUS; k <= klast-RADIUS; k++)
  {
    kthtmp = kthm4;
    kthm4  = kthm3;
    kthm3  = kthm2;
    kthm2  = kthm1;
    kthm1  = kthm0;
    kthm0  = kthtmp;
    if ( (i <= ilast-RADIUS) && (j <= jlast-RADIUS) )
    {
      mux1 = mu(i-1,j,k)*strx(i-1)-
        tf*(mu(i,j,k)*strx(i)+mu(i-2,j,k)*strx(i-2));
      mux2 = mu(i-2,j,k)*strx(i-2)+mu(i+1,j,k)*strx(i+1)+
        3*(mu(i,j,k)*strx(i)+mu(i-1,j,k)*strx(i-1));
      mux3 = mu(i-1,j,k)*strx(i-1)+mu(i+2,j,k)*strx(i+2)+
        3*(mu(i+1,j,k)*strx(i+1)+mu(i,j,k)*strx(i));
      mux4 = mu(i+1,j,k)*strx(i+1)-
        tf*(mu(i,j,k)*strx(i)+mu(i+2,j,k)*strx(i+2));

      muy1 = mu(i,j-1,k)*stry(j-1)-
        tf*(mu(i,j,k)*stry(j)+mu(i,j-2,k)*stry(j-2));
      muy2 = mu(i,j-2,k)*stry(j-2)+mu(i,j+1,k)*stry(j+1)+
        3*(mu(i,j,k)*stry(j)+mu(i,j-1,k)*stry(j-1));
      muy3 = mu(i,j-1,k)*stry(j-1)+mu(i,j+2,k)*stry(j+2)+
        3*(mu(i,j+1,k)*stry(j+1)+mu(i,j,k)*stry(j));
      muy4 = mu(i,j+1,k)*stry(j+1)-
        tf*(mu(i,j,k)*stry(j)+mu(i,j+2,k)*stry(j+2));

      muz1 = mu(i,j,k-1)*strz(k-1)-
        tf*(mu(i,j,k)*strz(k)+mu(i,j,k-2)*strz(k-2));
      muz2 = mu(i,j,k-2)*strz(k-2)+mu(i,j,k+1)*strz(k+1)+
        3*(mu(i,j,k)*strz(k)+mu(i,j,k-1)*strz(k-1));
      muz3 = mu(i,j,k-1)*strz(k-1)+mu(i,j,k+2)*strz(k+2)+
        3*(mu(i,j,k+1)*strz(k+1)+mu(i,j,k)*strz(k));
      muz4 = mu(i,j,k+1)*strz(k+1)-
        tf*(mu(i,j,k)*strz(k)+mu(i,j,k+2)*strz(k+2));
/* xx, yy, and zz derivatives:*/
/* 75 ops */
      r1 = i6*( strx(i)*( (2*mux1+la(i-1,j,k)*strx(i-1)-
                           tf*(la(i,j,k)*strx(i)+la(i-2,j,k)*strx(i-2)))*
                          (uSh(1,ith-2,jth,kthm2)-uSh(1,ith,jth,kthm2))+
                          (2*mux2+la(i-2,j,k)*strx(i-2)+la(i+1,j,k)*strx(i+1)+
                           3*(la(i,j,k)*strx(i)+la(i-1,j,k)*strx(i-1)))*
                          (uSh(1,ith-1,jth,kthm2)-uSh(1,ith,jth,kthm2))+
                          (2*mux3+la(i-1,j,k)*strx(i-1)+la(i+2,j,k)*strx(i+2)+
                           3*(la(i+1,j,k)*strx(i+1)+la(i,j,k)*strx(i)))*
                          (uSh(1,ith+1,jth,kthm2)-uSh(1,ith,jth,kthm2))+
                          (2*mux4+ la(i+1,j,k)*strx(i+1)-
                           tf*(la(i,j,k)*strx(i)+la(i+2,j,k)*strx(i+2)))*
                          (uSh(1,ith+2,jth,kthm2)-uSh(1,ith,jth,kthm2)) ) + stry(j)*(
                            muy1*(uSh(1,ith,jth-2,kthm2)-uSh(1,ith,jth,kthm2)) +
                            muy2*(uSh(1,ith,jth-1,kthm2)-uSh(1,ith,jth,kthm2)) +
                            muy3*(uSh(1,ith,jth+1,kthm2)-uSh(1,ith,jth,kthm2)) +
                            muy4*(uSh(1,ith,jth+2,kthm2)-uSh(1,ith,jth,kthm2)) ) + strz(k)*(
                              muz1*(uSh(1,ith,jth,kthm4)-uSh(1,ith,jth,kthm2)) +
                              muz2*(uSh(1,ith,jth,kthm3)-uSh(1,ith,jth,kthm2)) +
                              muz3*(uSh(1,ith,jth,kthm1)-uSh(1,ith,jth,kthm2)) +
                              muz4*(uSh(1,ith,jth,kthm0)-uSh(1,ith,jth,kthm2)) ) );

/* 75 ops */
      r2 = i6*( strx(i)*(mux1*(uSh(2,ith-2,jth,kthm2)-uSh(2,ith,jth,kthm2)) +
                         mux2*(uSh(2,ith-1,jth,kthm2)-uSh(2,ith,jth,kthm2)) +
                         mux3*(uSh(2,ith+1,jth,kthm2)-uSh(2,ith,jth,kthm2)) +
                         mux4*(uSh(2,ith+2,jth,kthm2)-uSh(2,ith,jth,kthm2)) ) + stry(j)*(
                           (2*muy1+la(i,j-1,k)*stry(j-1)-
                            tf*(la(i,j,k)*stry(j)+la(i,j-2,k)*stry(j-2)))*
                           (uSh(2,ith,jth-2,kthm2)-uSh(2,ith,jth,kthm2))+
                           (2*muy2+la(i,j-2,k)*stry(j-2)+la(i,j+1,k)*stry(j+1)+
                            3*(la(i,j,k)*stry(j)+la(i,j-1,k)*stry(j-1)))*
                           (uSh(2,ith,jth-1,kthm2)-uSh(2,ith,jth,kthm2))+
                           (2*muy3+la(i,j-1,k)*stry(j-1)+la(i,j+2,k)*stry(j+2)+
                            3*(la(i,j+1,k)*stry(j+1)+la(i,j,k)*stry(j)))*
                           (uSh(2,ith,jth+1,kthm2)-uSh(2,ith,jth,kthm2))+
                           (2*muy4+la(i,j+1,k)*stry(j+1)-
                            tf*(la(i,j,k)*stry(j)+la(i,j+2,k)*stry(j+2)))*
                           (uSh(2,ith,jth+2,kthm2)-uSh(2,ith,jth,kthm2)) ) + strz(k)*(
                             muz1*(uSh(2,ith,jth,kthm4)-uSh(2,ith,jth,kthm2)) +
                             muz2*(uSh(2,ith,jth,kthm3)-uSh(2,ith,jth,kthm2)) +
                             muz3*(uSh(2,ith,jth,kthm1)-uSh(2,ith,jth,kthm2)) +
                             muz4*(uSh(2,ith,jth,kthm0)-uSh(2,ith,jth,kthm2)) ) );

/* 75 ops */
      r3 = i6*( strx(i)*(mux1*(uSh(3,ith-2,jth,kthm2)-uSh(3,ith,jth,kthm2)) +
                         mux2*(uSh(3,ith-1,jth,kthm2)-uSh(3,ith,jth,kthm2)) +
                         mux3*(uSh(3,ith+1,jth,kthm2)-uSh(3,ith,jth,kthm2)) +
                         mux4*(uSh(3,ith+2,jth,kthm2)-uSh(3,ith,jth,kthm2))  ) + stry(j)*(
                           muy1*(uSh(3,ith,jth-2,kthm2)-uSh(3,ith,jth,kthm2)) +
                           muy2*(uSh(3,ith,jth-1,kthm2)-uSh(3,ith,jth,kthm2)) +
                           muy3*(uSh(3,ith,jth+1,kthm2)-uSh(3,ith,jth,kthm2)) +
                           muy4*(uSh(3,ith,jth+2,kthm2)-uSh(3,ith,jth,kthm2)) ) + strz(k)*(
                             (2*muz1+la(i,j,k-1)*strz(k-1)-
                              tf*(la(i,j,k)*strz(k)+la(i,j,k-2)*strz(k-2)))*
                             (uSh(3,ith,jth,kthm4)-uSh(3,ith,jth,kthm2))+
                             (2*muz2+la(i,j,k-2)*strz(k-2)+la(i,j,k+1)*strz(k+1)+
                              3*(la(i,j,k)*strz(k)+la(i,j,k-1)*strz(k-1)))*
                             (uSh(3,ith,jth,kthm3)-uSh(3,ith,jth,kthm2))+
                             (2*muz3+la(i,j,k-1)*strz(k-1)+la(i,j,k+2)*strz(k+2)+
                              3*(la(i,j,k+1)*strz(k+1)+la(i,j,k)*strz(k)))*
                             (uSh(3,ith,jth,kthm1)-uSh(3,ith,jth,kthm2))+
                             (2*muz4+la(i,j,k+1)*strz(k+1)-
                              tf*(la(i,j,k)*strz(k)+la(i,j,k+2)*strz(k+2)))*
                             (uSh(3,ith,jth,kthm0)-uSh(3,ith,jth,kthm2)) ) );


/* Mixed derivatives: */
/* 29ops /mixed derivative */
/* 116 ops for r1 */
/*   (la*v_y)_x */
      r1 = r1 + strx(i)*stry(j)*
        i144*( la(i-2,j,k)*(uSh(2,ith-2,jth-2,kthm2)-uSh(2,ith-2,jth+2,kthm2)+
                            8*(-uSh(2,ith-2,jth-1,kthm2)+uSh(2,ith-2,jth+1,kthm2))) - 8*(
                              la(i-1,j,k)*(uSh(2,ith-1,jth-2,kthm2)-uSh(2,ith-1,jth+2,kthm2)+
                                           8*(-uSh(2,ith-1,jth-1,kthm2)+uSh(2,ith-1,jth+1,kthm2))) )+8*(
                                             la(i+1,j,k)*(uSh(2,ith+1,jth-2,kthm2)-uSh(2,ith+1,jth+2,kthm2)+
                                                          8*(-uSh(2,ith+1,jth-1,kthm2)+uSh(2,ith+1,jth+1,kthm2))) ) - (
                                                            la(i+2,j,k)*(uSh(2,ith+2,jth-2,kthm2)-uSh(2,ith+2,jth+2,kthm2)+
                                                                         8*(-uSh(2,ith+2,jth-1,kthm2)+uSh(2,ith+2,jth+1,kthm2))) ))
/*   (la*w_z)_x */
        + strx(i)*strz(k)*
        i144*( la(i-2,j,k)*(uSh(3,ith-2,jth,kthm4)-uSh(3,ith-2,jth,kthm0)+
                            8*(-uSh(3,ith-2,jth,kthm3)+uSh(3,ith-2,jth,kthm1))) - 8*(
                              la(i-1,j,k)*(uSh(3,ith-1,jth,kthm4)-uSh(3,ith-1,jth,kthm0)+
                                           8*(-uSh(3,ith-1,jth,kthm3)+uSh(3,ith-1,jth,kthm1))) )+8*(
                                             la(i+1,j,k)*(uSh(3,ith+1,jth,kthm4)-uSh(3,ith+1,jth,kthm0)+
                                                          8*(-uSh(3,ith+1,jth,kthm3)+uSh(3,ith+1,jth,kthm1))) ) - (
                                                            la(i+2,j,k)*(uSh(3,ith+2,jth,kthm4)-uSh(3,ith+2,jth,kthm0)+
                                                                         8*(-uSh(3,ith+2,jth,kthm3)+uSh(3,ith+2,jth,kthm1))) ))
/*   (mu*v_x)_y */
        + strx(i)*stry(j)*
        i144*( mu(i,j-2,k)*(uSh(2,ith-2,jth-2,kthm2)-uSh(2,ith+2,jth-2,kthm2)+
                            8*(-uSh(2,ith-1,jth-2,kthm2)+uSh(2,ith+1,jth-2,kthm2))) - 8*(
                              mu(i,j-1,k)*(uSh(2,ith-2,jth-1,kthm2)-uSh(2,ith+2,jth-1,kthm2)+
                                           8*(-uSh(2,ith-1,jth-1,kthm2)+uSh(2,ith+1,jth-1,kthm2))) )+8*(
                                             mu(i,j+1,k)*(uSh(2,ith-2,jth+1,kthm2)-uSh(2,ith+2,jth+1,kthm2)+
                                                          8*(-uSh(2,ith-1,jth+1,kthm2)+uSh(2,ith+1,jth+1,kthm2))) ) - (
                                                            mu(i,j+2,k)*(uSh(2,ith-2,jth+2,kthm2)-uSh(2,ith+2,jth+2,kthm2)+
                                                                         8*(-uSh(2,ith-1,jth+2,kthm2)+uSh(2,ith+1,jth+2,kthm2))) ))
/*   (mu*w_x)_z */
        + strx(i)*strz(k)*
        i144*( mu(i,j,k-2)*(uSh(3,ith-2,jth,kthm4)-uSh(3,ith+2,jth,kthm4)+
                            8*(-uSh(3,ith-1,jth,kthm4)+uSh(3,ith+1,jth,kthm4))) - 8*(
                              mu(i,j,k-1)*(uSh(3,ith-2,jth,kthm3)-uSh(3,ith+2,jth,kthm3)+
                                           8*(-uSh(3,ith-1,jth,kthm3)+uSh(3,ith+1,jth,kthm3))) )+8*(
                                             mu(i,j,k+1)*(uSh(3,ith-2,jth,kthm1)-uSh(3,ith+2,jth,kthm1)+
                                                          8*(-uSh(3,ith-1,jth,kthm1)+uSh(3,ith+1,jth,kthm1))) ) - (
                                                            mu(i,j,k+2)*(uSh(3,ith-2,jth,kthm0)-uSh(3,ith+2,jth,kthm0)+
                                                                         8*(-uSh(3,ith-1,jth,kthm0)+uSh(3,ith+1,jth,kthm0))) )) ;

/* 116 ops for r2 */
/*   (mu*u_y)_x */
      r2 = r2 + strx(i)*stry(j)*
        i144*( mu(i-2,j,k)*(uSh(1,ith-2,jth-2,kthm2)-uSh(1,ith-2,jth+2,kthm2)+
                            8*(-uSh(1,ith-2,jth-1,kthm2)+uSh(1,ith-2,jth+1,kthm2))) - 8*(
                              mu(i-1,j,k)*(uSh(1,ith-1,jth-2,kthm2)-uSh(1,ith-1,jth+2,kthm2)+
                                           8*(-uSh(1,ith-1,jth-1,kthm2)+uSh(1,ith-1,jth+1,kthm2))) )+8*(
                                             mu(i+1,j,k)*(uSh(1,ith+1,jth-2,kthm2)-uSh(1,ith+1,jth+2,kthm2)+
                                                          8*(-uSh(1,ith+1,jth-1,kthm2)+uSh(1,ith+1,jth+1,kthm2))) ) - (
                                                            mu(i+2,j,k)*(uSh(1,ith+2,jth-2,kthm2)-uSh(1,ith+2,jth+2,kthm2)+
                                                                         8*(-uSh(1,ith+2,jth-1,kthm2)+uSh(1,ith+2,jth+1,kthm2))) ))
/* (la*u_x)_y */
        + strx(i)*stry(j)*
        i144*( la(i,j-2,k)*(uSh(1,ith-2,jth-2,kthm2)-uSh(1,ith+2,jth-2,kthm2)+
                            8*(-uSh(1,ith-1,jth-2,kthm2)+uSh(1,ith+1,jth-2,kthm2))) - 8*(
                              la(i,j-1,k)*(uSh(1,ith-2,jth-1,kthm2)-uSh(1,ith+2,jth-1,kthm2)+
                                           8*(-uSh(1,ith-1,jth-1,kthm2)+uSh(1,ith+1,jth-1,kthm2))) )+8*(
                                             la(i,j+1,k)*(uSh(1,ith-2,jth+1,kthm2)-uSh(1,ith+2,jth+1,kthm2)+
                                                          8*(-uSh(1,ith-1,jth+1,kthm2)+uSh(1,ith+1,jth+1,kthm2))) ) - (
                                                            la(i,j+2,k)*(uSh(1,ith-2,jth+2,kthm2)-uSh(1,ith+2,jth+2,kthm2)+
                                                                         8*(-uSh(1,ith-1,jth+2,kthm2)+uSh(1,ith+1,jth+2,kthm2))) ))
/* (la*w_z)_y */
        + stry(j)*strz(k)*
        i144*( la(i,j-2,k)*(uSh(3,ith,jth-2,kthm4)-uSh(3,ith,jth-2,kthm0)+
                            8*(-uSh(3,ith,jth-2,kthm3)+uSh(3,ith,jth-2,kthm1))) - 8*(
                              la(i,j-1,k)*(uSh(3,ith,jth-1,kthm4)-uSh(3,ith,jth-1,kthm0)+
                                           8*(-uSh(3,ith,jth-1,kthm3)+uSh(3,ith,jth-1,kthm1))) )+8*(
                                             la(i,j+1,k)*(uSh(3,ith,jth+1,kthm4)-uSh(3,ith,jth+1,kthm0)+
                                                          8*(-uSh(3,ith,jth+1,kthm3)+uSh(3,ith,jth+1,kthm1))) ) - (
                                                            la(i,j+2,k)*(uSh(3,ith,jth+2,kthm4)-uSh(3,ith,jth+2,kthm0)+
                                                                         8*(-uSh(3,ith,jth+2,kthm3)+uSh(3,ith,jth+2,kthm1))) ))
/* (mu*w_y)_z */
        + stry(j)*strz(k)*
        i144*( mu(i,j,k-2)*(uSh(3,ith,jth-2,kthm4)-uSh(3,ith,jth+2,kthm4)+
                            8*(-uSh(3,ith,jth-1,kthm4)+uSh(3,ith,jth+1,kthm4))) - 8*(
                              mu(i,j,k-1)*(uSh(3,ith,jth-2,kthm3)-uSh(3,ith,jth+2,kthm3)+
                                           8*(-uSh(3,ith,jth-1,kthm3)+uSh(3,ith,jth+1,kthm3))) )+8*(
                                             mu(i,j,k+1)*(uSh(3,ith,jth-2,kthm1)-uSh(3,ith,jth+2,kthm1)+
                                                          8*(-uSh(3,ith,jth-1,kthm1)+uSh(3,ith,jth+1,kthm1))) ) - (
                                                            mu(i,j,k+2)*(uSh(3,ith,jth-2,kthm0)-uSh(3,ith,jth+2,kthm0)+
                                                                         8*(-uSh(3,ith,jth-1,kthm0)+uSh(3,ith,jth+1,kthm0))) )) ;
/* 116 ops for r3 */
/*  (mu*u_z)_x */
      r3 = r3 + strx(i)*strz(k)*
        i144*( mu(i-2,j,k)*(uSh(1,ith-2,jth,kthm4)-uSh(1,ith-2,jth,kthm0)+
                            8*(-uSh(1,ith-2,jth,kthm3)+uSh(1,ith-2,jth,kthm1))) - 8*(
                              mu(i-1,j,k)*(uSh(1,ith-1,jth,kthm4)-uSh(1,ith-1,jth,kthm0)+
                                           8*(-uSh(1,ith-1,jth,kthm3)+uSh(1,ith-1,jth,kthm1))) )+8*(
                                             mu(i+1,j,k)*(uSh(1,ith+1,jth,kthm4)-uSh(1,ith+1,jth,kthm0)+
                                                          8*(-uSh(1,ith+1,jth,kthm3)+uSh(1,ith+1,jth,kthm1))) ) - (
                                                            mu(i+2,j,k)*(uSh(1,ith+2,jth,kthm4)-uSh(1,ith+2,jth,kthm0)+
                                                                         8*(-uSh(1,ith+2,jth,kthm3)+uSh(1,ith+2,jth,kthm1))) ))
/* (mu*v_z)_y */
        + stry(j)*strz(k)*
        i144*( mu(i,j-2,k)*(uSh(2,ith,jth-2,kthm4)-uSh(2,ith,jth-2,kthm0)+
                            8*(-uSh(2,ith,jth-2,kthm3)+uSh(2,ith,jth-2,kthm1))) - 8*(
                              mu(i,j-1,k)*(uSh(2,ith,jth-1,kthm4)-uSh(2,ith,jth-1,kthm0)+
                                           8*(-uSh(2,ith,jth-1,kthm3)+uSh(2,ith,jth-1,kthm1))) )+8*(
                                             mu(i,j+1,k)*(uSh(2,ith,jth+1,kthm4)-uSh(2,ith,jth+1,kthm0)+
                                                          8*(-uSh(2,ith,jth+1,kthm3)+uSh(2,ith,jth+1,kthm1))) ) - (
                                                            mu(i,j+2,k)*(uSh(2,ith,jth+2,kthm4)-uSh(2,ith,jth+2,kthm0)+
                                                                         8*(-uSh(2,ith,jth+2,kthm3)+uSh(2,ith,jth+2,kthm1))) ))
/*   (la*u_x)_z */
        + strx(i)*strz(k)*
        i144*( la(i,j,k-2)*(uSh(1,ith-2,jth,kthm4)-uSh(1,ith+2,jth,kthm4)+
                            8*(-uSh(1,ith-1,jth,kthm4)+uSh(1,ith+1,jth,kthm4))) - 8*(
                              la(i,j,k-1)*(uSh(1,ith-2,jth,kthm3)-uSh(1,ith+2,jth,kthm3)+
                                           8*(-uSh(1,ith-1,jth,kthm3)+uSh(1,ith+1,jth,kthm3))) )+8*(
                                             la(i,j,k+1)*(uSh(1,ith-2,jth,kthm1)-uSh(1,ith+2,jth,kthm1)+
                                                          8*(-uSh(1,ith-1,jth,kthm1)+uSh(1,ith+1,jth,kthm1))) ) - (
                                                            la(i,j,k+2)*(uSh(1,ith-2,jth,kthm0)-uSh(1,ith+2,jth,kthm0)+
                                                                         8*(-uSh(1,ith-1,jth,kthm0)+uSh(1,ith+1,jth,kthm0))) ))
/* (la*v_y)_z */
        + stry(j)*strz(k)*
        i144*( la(i,j,k-2)*(uSh(2,ith,jth-2,kthm4)-uSh(2,ith,jth+2,kthm4)+
                            8*(-uSh(2,ith,jth-1,kthm4)+uSh(2,ith,jth+1,kthm4))) - 8*(
                              la(i,j,k-1)*(uSh(2,ith,jth-2,kthm3)-uSh(2,ith,jth+2,kthm3)+
                                           8*(-uSh(2,ith,jth-1,kthm3)+uSh(2,ith,jth+1,kthm3))) )+8*(
                                             la(i,j,k+1)*(uSh(2,ith,jth-2,kthm1)-uSh(2,ith,jth+2,kthm1)+
                                                          8*(-uSh(2,ith,jth-1,kthm1)+uSh(2,ith,jth+1,kthm1))) ) - (
                                                            la(i,j,k+2)*(uSh(2,ith,jth-2,kthm0)-uSh(2,ith,jth+2,kthm0)+
                                                                         8*(-uSh(2,ith,jth-1,kthm0)+uSh(2,ith,jth+1,kthm0))) )) ;

/* 9 ops */
      lu(1,i,j,k) = a1*lu(1,i,j,k) + cof*r1;
      lu(2,i,j,k) = a1*lu(2,i,j,k) + cof*r2;
      lu(3,i,j,k) = a1*lu(3,i,j,k) + cof*r3;
    }

    __syncthreads();
    if (k+RADIUS+1 <= klast)
    {
      for (int tj = threadIdx.y; tj < blockDim.y+2*RADIUS; tj += blockDim.y) {
        int gj =  (j-threadIdx.y) + tj - RADIUS;
        if ( gj <= jlast) {
          for (int ti = threadIdx.x; ti < blockDim.x+2*RADIUS; ti += blockDim.x) {
            int gi =  (i-threadIdx.x) + ti - RADIUS;
            if ( gi <= ilast) {
              uSh(1, ti, tj, kthm4) = u(1, gi, gj, k+RADIUS+1);
              uSh(2, ti, tj, kthm4) = u(2, gi, gj, k+RADIUS+1);
              uSh(3, ti, tj, kthm4) = u(3, gi, gj, k+RADIUS+1);
            }
          }
        }
      }
    }
    __syncthreads();
  }
#undef mu
#undef la
#undef u
#undef lu
#undef strx
#undef stry
#undef strz
#undef uSh
}


//-----------------------------------------------------------------------

__global__ void 
__launch_bounds__(256,1)
rhs4center_dev_rev_v2( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                       float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda,
                       float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
                       int ghost_points )
{
  // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+(i)+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+(i)+ni*(j)+nij*(k)]
#define u(c,i,j,k)   a_u[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define lu(c,i,j,k) a_lu[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define strz(k) a_strz[k-kfirst0]
#define uSh(c,i,j,k) a_uSh[c-1][k][j][i]

  const float_sw4 a1   = 0.;
  const float_sw4 i6   = 1.0/6;
  //   const float_sw4 i12  = 1.0/12;
  const float_sw4 i144 = 1.0/144;
  const float_sw4 tf   = 0.75;

  const int ni    = ilast-ifirst+1;
  const int nij   = ni*(jlast-jfirst+1);
  const int nijk  = nij*(klast-kfirst+1);
  const int base  = -(ifirst+ni*jfirst+nij*kfirst);
  const int base3 = base-nijk;
  const int nic  = 3*ni;
  const int nijc = 3*nij;
  const int ifirst0 = ifirst;
  const int jfirst0 = jfirst;
  const int kfirst0 = kfirst;
  float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4, muz1, muz2, muz3, muz4;
  float_sw4 r1, r2, r3, cof;
  const int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
  const int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;

  __shared__ float_sw4 a_uSh[3][DIAMETER][RHS4_BLOCKY+2*RADIUS][RHS4_BLOCKX+2*RADIUS];

  const int ith = threadIdx.x + RADIUS;
  const int jth = threadIdx.y + RADIUS;
//  const int kth = threadIdx.z + RADIUS;

  int kthm0=3, kthm1=2, kthm2=1, kthm3=0, kthm4=4, kthtmp;

  for (int tk = 0; tk < DIAMETER; tk++)
  {
    int gk =  kfirst + tk;
    if ( gk <= klast)
    {
      for (int tj = threadIdx.y; tj < blockDim.y+2*RADIUS; tj += blockDim.y)
      {
        int gj =  (j-threadIdx.y) + tj - RADIUS;
        if ( gj <= jlast)
        {
          for (int ti = threadIdx.x; ti < blockDim.x+2*RADIUS; ti += blockDim.x)
          {
            int gi =  (i-threadIdx.x) + ti - RADIUS;
            if ( gi <= ilast)
            {
              uSh(1, ti, tj, tk) = u(1, gi, gj, gk);
              uSh(2, ti, tj, tk) = u(2, gi, gj, gk);
              uSh(3, ti, tj, tk) = u(3, gi, gj, gk);
            }
          }
        }
      }
    }
  }
  __syncthreads();

  cof = 1.0/(h*h);

  for (int k = kfirst+RADIUS; k <= klast-RADIUS; k++)
  {
    kthtmp = kthm4;
    kthm4  = kthm3;
    kthm3  = kthm2;
    kthm2  = kthm1;
    kthm1  = kthm0;
    kthm0  = kthtmp;
    if ( (i <= ilast-RADIUS) && (j <= jlast-RADIUS) )
    {
      mux1 = mu(i-1,j,k)*strx(i-1)-
        tf*(mu(i,j,k)*strx(i)+mu(i-2,j,k)*strx(i-2));
      mux2 = mu(i-2,j,k)*strx(i-2)+mu(i+1,j,k)*strx(i+1)+
        3*(mu(i,j,k)*strx(i)+mu(i-1,j,k)*strx(i-1));
      mux3 = mu(i-1,j,k)*strx(i-1)+mu(i+2,j,k)*strx(i+2)+
        3*(mu(i+1,j,k)*strx(i+1)+mu(i,j,k)*strx(i));
      mux4 = mu(i+1,j,k)*strx(i+1)-
        tf*(mu(i,j,k)*strx(i)+mu(i+2,j,k)*strx(i+2));

      muy1 = mu(i,j-1,k)*stry(j-1)-
        tf*(mu(i,j,k)*stry(j)+mu(i,j-2,k)*stry(j-2));
      muy2 = mu(i,j-2,k)*stry(j-2)+mu(i,j+1,k)*stry(j+1)+
        3*(mu(i,j,k)*stry(j)+mu(i,j-1,k)*stry(j-1));
      muy3 = mu(i,j-1,k)*stry(j-1)+mu(i,j+2,k)*stry(j+2)+
        3*(mu(i,j+1,k)*stry(j+1)+mu(i,j,k)*stry(j));
      muy4 = mu(i,j+1,k)*stry(j+1)-
        tf*(mu(i,j,k)*stry(j)+mu(i,j+2,k)*stry(j+2));

      muz1 = mu(i,j,k-1)*strz(k-1)-
        tf*(mu(i,j,k)*strz(k)+mu(i,j,k-2)*strz(k-2));
      muz2 = mu(i,j,k-2)*strz(k-2)+mu(i,j,k+1)*strz(k+1)+
        3*(mu(i,j,k)*strz(k)+mu(i,j,k-1)*strz(k-1));
      muz3 = mu(i,j,k-1)*strz(k-1)+mu(i,j,k+2)*strz(k+2)+
        3*(mu(i,j,k+1)*strz(k+1)+mu(i,j,k)*strz(k));
      muz4 = mu(i,j,k+1)*strz(k+1)-
        tf*(mu(i,j,k)*strz(k)+mu(i,j,k+2)*strz(k+2));
/* xx, yy, and zz derivatives:*/
/* 75 ops */
      r1 = i6*( strx(i)*( (2*mux1+la(i-1,j,k)*strx(i-1)-
                           tf*(la(i,j,k)*strx(i)+la(i-2,j,k)*strx(i-2)))*
                          (uSh(1,ith-2,jth,kthm2)-uSh(1,ith,jth,kthm2))+
                          (2*mux2+la(i-2,j,k)*strx(i-2)+la(i+1,j,k)*strx(i+1)+
                           3*(la(i,j,k)*strx(i)+la(i-1,j,k)*strx(i-1)))*
                          (uSh(1,ith-1,jth,kthm2)-uSh(1,ith,jth,kthm2))+
                          (2*mux3+la(i-1,j,k)*strx(i-1)+la(i+2,j,k)*strx(i+2)+
                           3*(la(i+1,j,k)*strx(i+1)+la(i,j,k)*strx(i)))*
                          (uSh(1,ith+1,jth,kthm2)-uSh(1,ith,jth,kthm2))+
                          (2*mux4+ la(i+1,j,k)*strx(i+1)-
                           tf*(la(i,j,k)*strx(i)+la(i+2,j,k)*strx(i+2)))*
                          (uSh(1,ith+2,jth,kthm2)-uSh(1,ith,jth,kthm2)) ) + stry(j)*(
                            muy1*(uSh(1,ith,jth-2,kthm2)-uSh(1,ith,jth,kthm2)) +
                            muy2*(uSh(1,ith,jth-1,kthm2)-uSh(1,ith,jth,kthm2)) +
                            muy3*(uSh(1,ith,jth+1,kthm2)-uSh(1,ith,jth,kthm2)) +
                            muy4*(uSh(1,ith,jth+2,kthm2)-uSh(1,ith,jth,kthm2)) ) + strz(k)*(
                              muz1*(uSh(1,ith,jth,kthm4)-uSh(1,ith,jth,kthm2)) +
                              muz2*(uSh(1,ith,jth,kthm3)-uSh(1,ith,jth,kthm2)) +
                              muz3*(uSh(1,ith,jth,kthm1)-uSh(1,ith,jth,kthm2)) +
                              muz4*(uSh(1,ith,jth,kthm0)-uSh(1,ith,jth,kthm2)) ) );

/* 75 ops */
      r2 = i6*( strx(i)*(mux1*(uSh(2,ith-2,jth,kthm2)-uSh(2,ith,jth,kthm2)) +
                         mux2*(uSh(2,ith-1,jth,kthm2)-uSh(2,ith,jth,kthm2)) +
                         mux3*(uSh(2,ith+1,jth,kthm2)-uSh(2,ith,jth,kthm2)) +
                         mux4*(uSh(2,ith+2,jth,kthm2)-uSh(2,ith,jth,kthm2)) ) + stry(j)*(
                           (2*muy1+la(i,j-1,k)*stry(j-1)-
                            tf*(la(i,j,k)*stry(j)+la(i,j-2,k)*stry(j-2)))*
                           (uSh(2,ith,jth-2,kthm2)-uSh(2,ith,jth,kthm2))+
                           (2*muy2+la(i,j-2,k)*stry(j-2)+la(i,j+1,k)*stry(j+1)+
                            3*(la(i,j,k)*stry(j)+la(i,j-1,k)*stry(j-1)))*
                           (uSh(2,ith,jth-1,kthm2)-uSh(2,ith,jth,kthm2))+
                           (2*muy3+la(i,j-1,k)*stry(j-1)+la(i,j+2,k)*stry(j+2)+
                            3*(la(i,j+1,k)*stry(j+1)+la(i,j,k)*stry(j)))*
                           (uSh(2,ith,jth+1,kthm2)-uSh(2,ith,jth,kthm2))+
                           (2*muy4+la(i,j+1,k)*stry(j+1)-
                            tf*(la(i,j,k)*stry(j)+la(i,j+2,k)*stry(j+2)))*
                           (uSh(2,ith,jth+2,kthm2)-uSh(2,ith,jth,kthm2)) ) + strz(k)*(
                             muz1*(uSh(2,ith,jth,kthm4)-uSh(2,ith,jth,kthm2)) +
                             muz2*(uSh(2,ith,jth,kthm3)-uSh(2,ith,jth,kthm2)) +
                             muz3*(uSh(2,ith,jth,kthm1)-uSh(2,ith,jth,kthm2)) +
                             muz4*(uSh(2,ith,jth,kthm0)-uSh(2,ith,jth,kthm2)) ) );

/* 75 ops */
      r3 = i6*( strx(i)*(mux1*(uSh(3,ith-2,jth,kthm2)-uSh(3,ith,jth,kthm2)) +
                         mux2*(uSh(3,ith-1,jth,kthm2)-uSh(3,ith,jth,kthm2)) +
                         mux3*(uSh(3,ith+1,jth,kthm2)-uSh(3,ith,jth,kthm2)) +
                         mux4*(uSh(3,ith+2,jth,kthm2)-uSh(3,ith,jth,kthm2))  ) + stry(j)*(
                           muy1*(uSh(3,ith,jth-2,kthm2)-uSh(3,ith,jth,kthm2)) +
                           muy2*(uSh(3,ith,jth-1,kthm2)-uSh(3,ith,jth,kthm2)) +
                           muy3*(uSh(3,ith,jth+1,kthm2)-uSh(3,ith,jth,kthm2)) +
                           muy4*(uSh(3,ith,jth+2,kthm2)-uSh(3,ith,jth,kthm2)) ) + strz(k)*(
                             (2*muz1+la(i,j,k-1)*strz(k-1)-
                              tf*(la(i,j,k)*strz(k)+la(i,j,k-2)*strz(k-2)))*
                             (uSh(3,ith,jth,kthm4)-uSh(3,ith,jth,kthm2))+
                             (2*muz2+la(i,j,k-2)*strz(k-2)+la(i,j,k+1)*strz(k+1)+
                              3*(la(i,j,k)*strz(k)+la(i,j,k-1)*strz(k-1)))*
                             (uSh(3,ith,jth,kthm3)-uSh(3,ith,jth,kthm2))+
                             (2*muz3+la(i,j,k-1)*strz(k-1)+la(i,j,k+2)*strz(k+2)+
                              3*(la(i,j,k+1)*strz(k+1)+la(i,j,k)*strz(k)))*
                             (uSh(3,ith,jth,kthm1)-uSh(3,ith,jth,kthm2))+
                             (2*muz4+la(i,j,k+1)*strz(k+1)-
                              tf*(la(i,j,k)*strz(k)+la(i,j,k+2)*strz(k+2)))*
                             (uSh(3,ith,jth,kthm0)-uSh(3,ith,jth,kthm2)) ) );


/* Mixed derivatives: */
/* 29ops /mixed derivative */
/* 116 ops for r1 */
/*   (la*v_y)_x */
      r1 = r1 + strx(i)*stry(j)*
        i144*( la(i-2,j,k)*(uSh(2,ith-2,jth-2,kthm2)-uSh(2,ith-2,jth+2,kthm2)+
                            8*(-uSh(2,ith-2,jth-1,kthm2)+uSh(2,ith-2,jth+1,kthm2))) - 8*(
                              la(i-1,j,k)*(uSh(2,ith-1,jth-2,kthm2)-uSh(2,ith-1,jth+2,kthm2)+
                                           8*(-uSh(2,ith-1,jth-1,kthm2)+uSh(2,ith-1,jth+1,kthm2))) )+8*(
                                             la(i+1,j,k)*(uSh(2,ith+1,jth-2,kthm2)-uSh(2,ith+1,jth+2,kthm2)+
                                                          8*(-uSh(2,ith+1,jth-1,kthm2)+uSh(2,ith+1,jth+1,kthm2))) ) - (
                                                            la(i+2,j,k)*(uSh(2,ith+2,jth-2,kthm2)-uSh(2,ith+2,jth+2,kthm2)+
                                                                         8*(-uSh(2,ith+2,jth-1,kthm2)+uSh(2,ith+2,jth+1,kthm2))) ))
/*   (la*w_z)_x */
        + strx(i)*strz(k)*
        i144*( la(i-2,j,k)*(uSh(3,ith-2,jth,kthm4)-uSh(3,ith-2,jth,kthm0)+
                            8*(-uSh(3,ith-2,jth,kthm3)+uSh(3,ith-2,jth,kthm1))) - 8*(
                              la(i-1,j,k)*(uSh(3,ith-1,jth,kthm4)-uSh(3,ith-1,jth,kthm0)+
                                           8*(-uSh(3,ith-1,jth,kthm3)+uSh(3,ith-1,jth,kthm1))) )+8*(
                                             la(i+1,j,k)*(uSh(3,ith+1,jth,kthm4)-uSh(3,ith+1,jth,kthm0)+
                                                          8*(-uSh(3,ith+1,jth,kthm3)+uSh(3,ith+1,jth,kthm1))) ) - (
                                                            la(i+2,j,k)*(uSh(3,ith+2,jth,kthm4)-uSh(3,ith+2,jth,kthm0)+
                                                                         8*(-uSh(3,ith+2,jth,kthm3)+uSh(3,ith+2,jth,kthm1))) ))
/*   (mu*v_x)_y */
        + strx(i)*stry(j)*
        i144*( mu(i,j-2,k)*(uSh(2,ith-2,jth-2,kthm2)-uSh(2,ith+2,jth-2,kthm2)+
                            8*(-uSh(2,ith-1,jth-2,kthm2)+uSh(2,ith+1,jth-2,kthm2))) - 8*(
                              mu(i,j-1,k)*(uSh(2,ith-2,jth-1,kthm2)-uSh(2,ith+2,jth-1,kthm2)+
                                           8*(-uSh(2,ith-1,jth-1,kthm2)+uSh(2,ith+1,jth-1,kthm2))) )+8*(
                                             mu(i,j+1,k)*(uSh(2,ith-2,jth+1,kthm2)-uSh(2,ith+2,jth+1,kthm2)+
                                                          8*(-uSh(2,ith-1,jth+1,kthm2)+uSh(2,ith+1,jth+1,kthm2))) ) - (
                                                            mu(i,j+2,k)*(uSh(2,ith-2,jth+2,kthm2)-uSh(2,ith+2,jth+2,kthm2)+
                                                                         8*(-uSh(2,ith-1,jth+2,kthm2)+uSh(2,ith+1,jth+2,kthm2))) ))
/*   (mu*w_x)_z */
        + strx(i)*strz(k)*
        i144*( mu(i,j,k-2)*(uSh(3,ith-2,jth,kthm4)-uSh(3,ith+2,jth,kthm4)+
                            8*(-uSh(3,ith-1,jth,kthm4)+uSh(3,ith+1,jth,kthm4))) - 8*(
                              mu(i,j,k-1)*(uSh(3,ith-2,jth,kthm3)-uSh(3,ith+2,jth,kthm3)+
                                           8*(-uSh(3,ith-1,jth,kthm3)+uSh(3,ith+1,jth,kthm3))) )+8*(
                                             mu(i,j,k+1)*(uSh(3,ith-2,jth,kthm1)-uSh(3,ith+2,jth,kthm1)+
                                                          8*(-uSh(3,ith-1,jth,kthm1)+uSh(3,ith+1,jth,kthm1))) ) - (
                                                            mu(i,j,k+2)*(uSh(3,ith-2,jth,kthm0)-uSh(3,ith+2,jth,kthm0)+
                                                                         8*(-uSh(3,ith-1,jth,kthm0)+uSh(3,ith+1,jth,kthm0))) )) ;

/* 116 ops for r2 */
/*   (mu*u_y)_x */
      r2 = r2 + strx(i)*stry(j)*
        i144*( mu(i-2,j,k)*(uSh(1,ith-2,jth-2,kthm2)-uSh(1,ith-2,jth+2,kthm2)+
                            8*(-uSh(1,ith-2,jth-1,kthm2)+uSh(1,ith-2,jth+1,kthm2))) - 8*(
                              mu(i-1,j,k)*(uSh(1,ith-1,jth-2,kthm2)-uSh(1,ith-1,jth+2,kthm2)+
                                           8*(-uSh(1,ith-1,jth-1,kthm2)+uSh(1,ith-1,jth+1,kthm2))) )+8*(
                                             mu(i+1,j,k)*(uSh(1,ith+1,jth-2,kthm2)-uSh(1,ith+1,jth+2,kthm2)+
                                                          8*(-uSh(1,ith+1,jth-1,kthm2)+uSh(1,ith+1,jth+1,kthm2))) ) - (
                                                            mu(i+2,j,k)*(uSh(1,ith+2,jth-2,kthm2)-uSh(1,ith+2,jth+2,kthm2)+
                                                                         8*(-uSh(1,ith+2,jth-1,kthm2)+uSh(1,ith+2,jth+1,kthm2))) ))
/* (la*u_x)_y */
        + strx(i)*stry(j)*
        i144*( la(i,j-2,k)*(uSh(1,ith-2,jth-2,kthm2)-uSh(1,ith+2,jth-2,kthm2)+
                            8*(-uSh(1,ith-1,jth-2,kthm2)+uSh(1,ith+1,jth-2,kthm2))) - 8*(
                              la(i,j-1,k)*(uSh(1,ith-2,jth-1,kthm2)-uSh(1,ith+2,jth-1,kthm2)+
                                           8*(-uSh(1,ith-1,jth-1,kthm2)+uSh(1,ith+1,jth-1,kthm2))) )+8*(
                                             la(i,j+1,k)*(uSh(1,ith-2,jth+1,kthm2)-uSh(1,ith+2,jth+1,kthm2)+
                                                          8*(-uSh(1,ith-1,jth+1,kthm2)+uSh(1,ith+1,jth+1,kthm2))) ) - (
                                                            la(i,j+2,k)*(uSh(1,ith-2,jth+2,kthm2)-uSh(1,ith+2,jth+2,kthm2)+
                                                                         8*(-uSh(1,ith-1,jth+2,kthm2)+uSh(1,ith+1,jth+2,kthm2))) ))
/* (la*w_z)_y */
        + stry(j)*strz(k)*
        i144*( la(i,j-2,k)*(uSh(3,ith,jth-2,kthm4)-uSh(3,ith,jth-2,kthm0)+
                            8*(-uSh(3,ith,jth-2,kthm3)+uSh(3,ith,jth-2,kthm1))) - 8*(
                              la(i,j-1,k)*(uSh(3,ith,jth-1,kthm4)-uSh(3,ith,jth-1,kthm0)+
                                           8*(-uSh(3,ith,jth-1,kthm3)+uSh(3,ith,jth-1,kthm1))) )+8*(
                                             la(i,j+1,k)*(uSh(3,ith,jth+1,kthm4)-uSh(3,ith,jth+1,kthm0)+
                                                          8*(-uSh(3,ith,jth+1,kthm3)+uSh(3,ith,jth+1,kthm1))) ) - (
                                                            la(i,j+2,k)*(uSh(3,ith,jth+2,kthm4)-uSh(3,ith,jth+2,kthm0)+
                                                                         8*(-uSh(3,ith,jth+2,kthm3)+uSh(3,ith,jth+2,kthm1))) ))
/* (mu*w_y)_z */
        + stry(j)*strz(k)*
        i144*( mu(i,j,k-2)*(uSh(3,ith,jth-2,kthm4)-uSh(3,ith,jth+2,kthm4)+
                            8*(-uSh(3,ith,jth-1,kthm4)+uSh(3,ith,jth+1,kthm4))) - 8*(
                              mu(i,j,k-1)*(uSh(3,ith,jth-2,kthm3)-uSh(3,ith,jth+2,kthm3)+
                                           8*(-uSh(3,ith,jth-1,kthm3)+uSh(3,ith,jth+1,kthm3))) )+8*(
                                             mu(i,j,k+1)*(uSh(3,ith,jth-2,kthm1)-uSh(3,ith,jth+2,kthm1)+
                                                          8*(-uSh(3,ith,jth-1,kthm1)+uSh(3,ith,jth+1,kthm1))) ) - (
                                                            mu(i,j,k+2)*(uSh(3,ith,jth-2,kthm0)-uSh(3,ith,jth+2,kthm0)+
                                                                         8*(-uSh(3,ith,jth-1,kthm0)+uSh(3,ith,jth+1,kthm0))) )) ;
/* 116 ops for r3 */
/*  (mu*u_z)_x */
      r3 = r3 + strx(i)*strz(k)*
        i144*( mu(i-2,j,k)*(uSh(1,ith-2,jth,kthm4)-uSh(1,ith-2,jth,kthm0)+
                            8*(-uSh(1,ith-2,jth,kthm3)+uSh(1,ith-2,jth,kthm1))) - 8*(
                              mu(i-1,j,k)*(uSh(1,ith-1,jth,kthm4)-uSh(1,ith-1,jth,kthm0)+
                                           8*(-uSh(1,ith-1,jth,kthm3)+uSh(1,ith-1,jth,kthm1))) )+8*(
                                             mu(i+1,j,k)*(uSh(1,ith+1,jth,kthm4)-uSh(1,ith+1,jth,kthm0)+
                                                          8*(-uSh(1,ith+1,jth,kthm3)+uSh(1,ith+1,jth,kthm1))) ) - (
                                                            mu(i+2,j,k)*(uSh(1,ith+2,jth,kthm4)-uSh(1,ith+2,jth,kthm0)+
                                                                         8*(-uSh(1,ith+2,jth,kthm3)+uSh(1,ith+2,jth,kthm1))) ))
/* (mu*v_z)_y */
        + stry(j)*strz(k)*
        i144*( mu(i,j-2,k)*(uSh(2,ith,jth-2,kthm4)-uSh(2,ith,jth-2,kthm0)+
                            8*(-uSh(2,ith,jth-2,kthm3)+uSh(2,ith,jth-2,kthm1))) - 8*(
                              mu(i,j-1,k)*(uSh(2,ith,jth-1,kthm4)-uSh(2,ith,jth-1,kthm0)+
                                           8*(-uSh(2,ith,jth-1,kthm3)+uSh(2,ith,jth-1,kthm1))) )+8*(
                                             mu(i,j+1,k)*(uSh(2,ith,jth+1,kthm4)-uSh(2,ith,jth+1,kthm0)+
                                                          8*(-uSh(2,ith,jth+1,kthm3)+uSh(2,ith,jth+1,kthm1))) ) - (
                                                            mu(i,j+2,k)*(uSh(2,ith,jth+2,kthm4)-uSh(2,ith,jth+2,kthm0)+
                                                                         8*(-uSh(2,ith,jth+2,kthm3)+uSh(2,ith,jth+2,kthm1))) ))
/*   (la*u_x)_z */
        + strx(i)*strz(k)*
        i144*( la(i,j,k-2)*(uSh(1,ith-2,jth,kthm4)-uSh(1,ith+2,jth,kthm4)+
                            8*(-uSh(1,ith-1,jth,kthm4)+uSh(1,ith+1,jth,kthm4))) - 8*(
                              la(i,j,k-1)*(uSh(1,ith-2,jth,kthm3)-uSh(1,ith+2,jth,kthm3)+
                                           8*(-uSh(1,ith-1,jth,kthm3)+uSh(1,ith+1,jth,kthm3))) )+8*(
                                             la(i,j,k+1)*(uSh(1,ith-2,jth,kthm1)-uSh(1,ith+2,jth,kthm1)+
                                                          8*(-uSh(1,ith-1,jth,kthm1)+uSh(1,ith+1,jth,kthm1))) ) - (
                                                            la(i,j,k+2)*(uSh(1,ith-2,jth,kthm0)-uSh(1,ith+2,jth,kthm0)+
                                                                         8*(-uSh(1,ith-1,jth,kthm0)+uSh(1,ith+1,jth,kthm0))) ))
/* (la*v_y)_z */
        + stry(j)*strz(k)*
        i144*( la(i,j,k-2)*(uSh(2,ith,jth-2,kthm4)-uSh(2,ith,jth+2,kthm4)+
                            8*(-uSh(2,ith,jth-1,kthm4)+uSh(2,ith,jth+1,kthm4))) - 8*(
                              la(i,j,k-1)*(uSh(2,ith,jth-2,kthm3)-uSh(2,ith,jth+2,kthm3)+
                                           8*(-uSh(2,ith,jth-1,kthm3)+uSh(2,ith,jth+1,kthm3))) )+8*(
                                             la(i,j,k+1)*(uSh(2,ith,jth-2,kthm1)-uSh(2,ith,jth+2,kthm1)+
                                                          8*(-uSh(2,ith,jth-1,kthm1)+uSh(2,ith,jth+1,kthm1))) ) - (
                                                            la(i,j,k+2)*(uSh(2,ith,jth-2,kthm0)-uSh(2,ith,jth+2,kthm0)+
                                                                         8*(-uSh(2,ith,jth-1,kthm0)+uSh(2,ith,jth+1,kthm0))) )) ;

/* 9 ops */
      lu(1,i,j,k) = a1*lu(1,i,j,k) + cof*r1;
      lu(2,i,j,k) = a1*lu(2,i,j,k) + cof*r2;
      lu(3,i,j,k) = a1*lu(3,i,j,k) + cof*r3;
    }

    __syncthreads();
    if (k+RADIUS+1 <= klast)
    {
      for (int tj = threadIdx.y; tj < blockDim.y+2*RADIUS; tj += blockDim.y) {
        int gj =  (j-threadIdx.y) + tj - RADIUS;
        if ( gj <= jlast) {
          for (int ti = threadIdx.x; ti < blockDim.x+2*RADIUS; ti += blockDim.x) {
            int gi =  (i-threadIdx.x) + ti - RADIUS;
            if ( gi <= ilast) {
              uSh(1, ti, tj, kthm4) = u(1, gi, gj, k+RADIUS+1);
              uSh(2, ti, tj, kthm4) = u(2, gi, gj, k+RADIUS+1);
              uSh(3, ti, tj, kthm4) = u(3, gi, gj, k+RADIUS+1);
            }
          }
        }
      }
    }
    __syncthreads();
  }
#undef mu
#undef la
#undef u
#undef lu
#undef strx
#undef stry
#undef strz
#undef uSh
}


//-----------------------------------------------------------------------
__global__ void rhs4upper_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				   // float_sw4* dev_acof, float_sw4* dev_bope, float_sw4* dev_ghcof,
			       float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
			       float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			       int ghost_points )
{
   // For 1 <= k <= 6 if free surface boundary.
#define mu(i,j,k)     a_mu[base+(i)+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+(i)+ni*(j)+nij*(k)]
#define u(c,i,j,k)   a_u[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define lu(c,i,j,k) a_lu[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
   //#define strz(k) a_strz[k-kfirst0]
#define acof(i,j,k) dev_acof[(i-1)+6*(j-1)+48*(k-1)]
#define bope(i,j) dev_bope[(i-1)+6*(j-1)]
#define ghcof(i) dev_ghcof[(i-1)]
   const float_sw4 a1   = 0.;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 i12  = 1.0/12;
   const float_sw4 i144 = 1.0/144;
   const float_sw4 tf   = 0.75;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = nij*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = base-nijk;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   //   const int kfirst0 = kfirst;

   int q, m;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4;//, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof, mucof, mu1zz, mu2zz, mu3zz;
   float_sw4 lap2mu, u3zip2, u3zip1, u3zim1, u3zim2, lau3zx, mu3xz, u3zjp2, u3zjp1, u3zjm1, u3zjm2;
   float_sw4 lau3zy, mu3yz, mu1zx, mu2zy, u1zip2, u1zip1, u1zim1, u1zim2;
   float_sw4 u2zjp2, u2zjp1, u2zjm1, u2zjm2, lau1xz, lau2yz;

   //   return;

   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if( k < 1 || k > 6 )
      return;

   if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
   cof = 1.0/(h*h);
   mux1 = mu(i-1,j,k)*strx(i-1)-
		     tf*(mu(i,j,k)*strx(i)+mu(i-2,j,k)*strx(i-2));
   mux2 = mu(i-2,j,k)*strx(i-2)+mu(i+1,j,k)*strx(i+1)+
		     3*(mu(i,j,k)*strx(i)+mu(i-1,j,k)*strx(i-1));
   mux3 = mu(i-1,j,k)*strx(i-1)+mu(i+2,j,k)*strx(i+2)+
		     3*(mu(i+1,j,k)*strx(i+1)+mu(i,j,k)*strx(i));
   mux4 = mu(i+1,j,k)*strx(i+1)-
		     tf*(mu(i,j,k)*strx(i)+mu(i+2,j,k)*strx(i+2));

   muy1 = mu(i,j-1,k)*stry(j-1)-
		     tf*(mu(i,j,k)*stry(j)+mu(i,j-2,k)*stry(j-2));
   muy2 = mu(i,j-2,k)*stry(j-2)+mu(i,j+1,k)*stry(j+1)+
		     3*(mu(i,j,k)*stry(j)+mu(i,j-1,k)*stry(j-1));
   muy3 = mu(i,j-1,k)*stry(j-1)+mu(i,j+2,k)*stry(j+2)+
		     3*(mu(i,j+1,k)*stry(j+1)+mu(i,j,k)*stry(j));
   muy4 = mu(i,j+1,k)*stry(j+1)-
		     tf*(mu(i,j,k)*stry(j)+mu(i,j+2,k)*stry(j+2));

   r1 = i6*(strx(i)*((2*mux1+la(i-1,j,k)*strx(i-1)-
                       tf*(la(i,j,k)*strx(i)+la(i-2,j,k)*strx(i-2)))*
                              (u(1,i-2,j,k)-u(1,i,j,k))+
           (2*mux2+la(i-2,j,k)*strx(i-2)+la(i+1,j,k)*strx(i+1)+
                        3*(la(i,j,k)*strx(i)+la(i-1,j,k)*strx(i-1)))*
                              (u(1,i-1,j,k)-u(1,i,j,k))+ 
           (2*mux3+la(i-1,j,k)*strx(i-1)+la(i+2,j,k)*strx(i+2)+
                        3*(la(i+1,j,k)*strx(i+1)+la(i,j,k)*strx(i)))*
                              (u(1,i+1,j,k)-u(1,i,j,k))+
                (2*mux4+ la(i+1,j,k)*strx(i+1)-
                       tf*(la(i,j,k)*strx(i)+la(i+2,j,k)*strx(i+2)))*
                (u(1,i+2,j,k)-u(1,i,j,k)) ) + stry(j)*(
                   + muy1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                     muy2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                     muy3*(u(1,i,j+1,k)-u(1,i,j,k)) +
                     muy4*(u(1,i,j+2,k)-u(1,i,j,k)) ) );

		  /* (mu*uz)_z can not be centered */
		  /* second derivative (mu*u_z)_z at grid point z_k */
		  /* averaging the coefficient, */
		  /* leave out the z-supergrid stretching strz, since it will */
		  /* never be used together with the sbp-boundary operator */
   mu1zz = 0.;
   mu2zz = 0.;
   mu3zz = 0.;
   for( q=1; q <= 8; q ++ )
   {
      lap2mu= 0.;
      mucof = 0.;
      for( m=1 ; m<=8; m++ )
      {
	 mucof  += acof(k,q,m)*mu(i,j,m);
	 lap2mu += acof(k,q,m)*(la(i,j,m)+2*mu(i,j,m));
      }
      mu1zz += mucof*u(1,i,j,q);
      mu2zz += mucof*u(2,i,j,q);
      mu3zz += lap2mu*u(3,i,j,q);
   }
		  /* ghost point only influences the first point (k=1) because ghcof(k)=0 for k>=2*/
   r1 = r1 + (mu1zz + ghcof(k)*mu(i,j,1)*u(1,i,j,0));

   r2 = i6*(strx(i)*(mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                      mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                      mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                      mux4*(u(2,i+2,j,k)-u(2,i,j,k)) )+ stry(j)*(
                  (2*muy1+la(i,j-1,k)*stry(j-1)-
                        tf*(la(i,j,k)*stry(j)+la(i,j-2,k)*stry(j-2)))*
                          (u(2,i,j-2,k)-u(2,i,j,k))+
           (2*muy2+la(i,j-2,k)*stry(j-2)+la(i,j+1,k)*stry(j+1)+
                        3*(la(i,j,k)*stry(j)+la(i,j-1,k)*stry(j-1)))*
                          (u(2,i,j-1,k)-u(2,i,j,k))+ 
           (2*muy3+la(i,j-1,k)*stry(j-1)+la(i,j+2,k)*stry(j+2)+
                        3*(la(i,j+1,k)*stry(j+1)+la(i,j,k)*stry(j)))*
                          (u(2,i,j+1,k)-u(2,i,j,k))+
                  (2*muy4+la(i,j+1,k)*stry(j+1)-
                       tf*(la(i,j,k)*stry(j)+la(i,j+2,k)*stry(j+2)))*
		  (u(2,i,j+2,k)-u(2,i,j,k)) ) );

 /* ghost point only influences the first point (k=1) because ghcof(k)=0 for k>=2 */
   r2 = r2 + (mu2zz + ghcof(k)*mu(i,j,1)*u(2,i,j,0));

   r3 = i6*(strx(i)*(mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                      mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                      mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                      mux4*(u(3,i+2,j,k)-u(3,i,j,k))  ) + stry(j)*(
                     muy1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                     muy2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                     muy3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                     muy4*(u(3,i,j+2,k)-u(3,i,j,k)) ) );
/* ghost point only influences the first point (k=1) because ghcof(k)=0 for k>=2 */
   r3 = r3 + (mu3zz + ghcof(k)*(la(i,j,1)+2*mu(i,j,1))*
			     u(3,i,j,0));

  /* cross-terms in first component of rhs */
/*   (la*v_y)_x */
   r1 = r1 + strx(i)*stry(j)*(
                 i144*( la(i-2,j,k)*(u(2,i-2,j-2,k)-u(2,i-2,j+2,k)+
                             8*(-u(2,i-2,j-1,k)+u(2,i-2,j+1,k))) - 8*(
                        la(i-1,j,k)*(u(2,i-1,j-2,k)-u(2,i-1,j+2,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i-1,j+1,k))) )+8*(
                        la(i+1,j,k)*(u(2,i+1,j-2,k)-u(2,i+1,j+2,k)+
                             8*(-u(2,i+1,j-1,k)+u(2,i+1,j+1,k))) ) - (
                        la(i+2,j,k)*(u(2,i+2,j-2,k)-u(2,i+2,j+2,k)+
				     8*(-u(2,i+2,j-1,k)+u(2,i+2,j+1,k))) ))
/*   (mu*v_x)_y */
               + i144*( mu(i,j-2,k)*(u(2,i-2,j-2,k)-u(2,i+2,j-2,k)+
                             8*(-u(2,i-1,j-2,k)+u(2,i+1,j-2,k))) - 8*(
                        mu(i,j-1,k)*(u(2,i-2,j-1,k)-u(2,i+2,j-1,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i+1,j-1,k))) )+8*(
                        mu(i,j+1,k)*(u(2,i-2,j+1,k)-u(2,i+2,j+1,k)+
                             8*(-u(2,i-1,j+1,k)+u(2,i+1,j+1,k))) ) - (
                        mu(i,j+2,k)*(u(2,i-2,j+2,k)-u(2,i+2,j+2,k)+
				     8*(-u(2,i-1,j+2,k)+u(2,i+1,j+2,k))) )) );
/*   (la*w_z)_x: NOT CENTERED */
   u3zip2=0.;
   u3zip1=0.;
   u3zim1=0.;
   u3zim2=0.;
   for( q=1 ; q <=8 ; q++ )
   {
      u3zip2 += bope(k,q)*u(3,i+2,j,q);
      u3zip1 += bope(k,q)*u(3,i+1,j,q);
      u3zim1 += bope(k,q)*u(3,i-1,j,q);
      u3zim2 += bope(k,q)*u(3,i-2,j,q);
   }
   lau3zx= i12*(-la(i+2,j,k)*u3zip2 + 8*la(i+1,j,k)*u3zip1
	               -8*la(i-1,j,k)*u3zim1 +   la(i-2,j,k)*u3zim2);
   r1 = r1 + strx(i)*lau3zx;
	    /*   (mu*w_x)_z: NOT CENTERED */
   mu3xz=0.;
   for( q=1 ; q<=8 ; q++ )
      mu3xz += bope(k,q)*( mu(i,j,q)*i12*
                  (-u(3,i+2,j,q) + 8*u(3,i+1,j,q)
                   -8*u(3,i-1,j,q) + u(3,i-2,j,q)) );
   r1 = r1 + strx(i)*mu3xz;

/* cross-terms in second component of rhs */
/*   (mu*u_y)_x */
   r2 = r2 + strx(i)*stry(j)*(
                 i144*( mu(i-2,j,k)*(u(1,i-2,j-2,k)-u(1,i-2,j+2,k)+
                             8*(-u(1,i-2,j-1,k)+u(1,i-2,j+1,k))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j-2,k)-u(1,i-1,j+2,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i-1,j+1,k))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j-2,k)-u(1,i+1,j+2,k)+
                             8*(-u(1,i+1,j-1,k)+u(1,i+1,j+1,k))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j-2,k)-u(1,i+2,j+2,k)+
				     8*(-u(1,i+2,j-1,k)+u(1,i+2,j+1,k))) )) 
/* (la*u_x)_y  */
               + i144*( la(i,j-2,k)*(u(1,i-2,j-2,k)-u(1,i+2,j-2,k)+
                             8*(-u(1,i-1,j-2,k)+u(1,i+1,j-2,k))) - 8*(
                        la(i,j-1,k)*(u(1,i-2,j-1,k)-u(1,i+2,j-1,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i+1,j-1,k))) )+8*(
                        la(i,j+1,k)*(u(1,i-2,j+1,k)-u(1,i+2,j+1,k)+
                             8*(-u(1,i-1,j+1,k)+u(1,i+1,j+1,k))) ) - (
                        la(i,j+2,k)*(u(1,i-2,j+2,k)-u(1,i+2,j+2,k)+
				     8*(-u(1,i-1,j+2,k)+u(1,i+1,j+2,k))) )) );
/* (la*w_z)_y : NOT CENTERED */
   u3zjp2=0.;
   u3zjp1=0.;
   u3zjm1=0.;
   u3zjm2=0.;
   for( q=1 ; q <=8 ; q++ )
   {
      u3zjp2 += bope(k,q)*u(3,i,j+2,q);
      u3zjp1 += bope(k,q)*u(3,i,j+1,q);
      u3zjm1 += bope(k,q)*u(3,i,j-1,q);
      u3zjm2 += bope(k,q)*u(3,i,j-2,q);
   }
   lau3zy= i12*(-la(i,j+2,k)*u3zjp2 + 8*la(i,j+1,k)*u3zjp1
			 -8*la(i,j-1,k)*u3zjm1 + la(i,j-2,k)*u3zjm2);

   r2 = r2 + stry(j)*lau3zy;

/* (mu*w_y)_z: NOT CENTERED */
   mu3yz=0.;
   for(  q=1 ; q <=8 ; q++ )
      mu3yz += bope(k,q)*( mu(i,j,q)*i12*
                  (-u(3,i,j+2,q) + 8*u(3,i,j+1,q)
                   -8*u(3,i,j-1,q) + u(3,i,j-2,q)) );

   r2 = r2 + stry(j)*mu3yz;

	    /* No centered cross terms in r3 */
	    /*  (mu*u_z)_x: NOT CENTERED */
   u1zip2=0.;
   u1zip1=0.;
   u1zim1=0.;
   u1zim2=0.;
   for(  q=1 ; q <=8 ; q++ )
   {
      u1zip2 += bope(k,q)*u(1,i+2,j,q);
      u1zip1 += bope(k,q)*u(1,i+1,j,q);
      u1zim1 += bope(k,q)*u(1,i-1,j,q);
      u1zim2 += bope(k,q)*u(1,i-2,j,q);
   }
   mu1zx= i12*(-mu(i+2,j,k)*u1zip2 + 8*mu(i+1,j,k)*u1zip1
                   -8*mu(i-1,j,k)*u1zim1 + mu(i-2,j,k)*u1zim2);
   r3 = r3 + strx(i)*mu1zx;

	    /* (mu*v_z)_y: NOT CENTERED */
   u2zjp2=0.;
   u2zjp1=0.;
   u2zjm1=0.;
   u2zjm2=0.;
   for(  q=1 ; q <=8 ; q++ )
   {
      u2zjp2 += bope(k,q)*u(2,i,j+2,q);
      u2zjp1 += bope(k,q)*u(2,i,j+1,q);
      u2zjm1 += bope(k,q)*u(2,i,j-1,q);
      u2zjm2 += bope(k,q)*u(2,i,j-2,q);
   }
   mu2zy= i12*(-mu(i,j+2,k)*u2zjp2 + 8*mu(i,j+1,k)*u2zjp1
                        -8*mu(i,j-1,k)*u2zjm1 + mu(i,j-2,k)*u2zjm2);
   r3 = r3 + stry(j)*mu2zy;

/*   (la*u_x)_z: NOT CENTERED */
   lau1xz=0.;
   for(  q=1 ; q <=8 ; q++ )
      lau1xz += bope(k,q)*( la(i,j,q)*i12*
                  (-u(1,i+2,j,q) + 8*u(1,i+1,j,q)
		   -8*u(1,i-1,j,q) + u(1,i-2,j,q)) );
   r3 = r3 + strx(i)*lau1xz;

/* (la*v_y)_z: NOT CENTERED */
   lau2yz=0.;
   for(  q=1 ; q <=8 ; q++ )
      lau2yz += bope(k,q)*( la(i,j,q)*i12*
                  (-u(2,i,j+2,q) + 8*u(2,i,j+1,q)
                   -8*u(2,i,j-1,q) + u(2,i,j-2,q)) );
   r3 = r3 + stry(j)*lau2yz;

   lu(1,i,j,k) = a1*lu(1,i,j,k) + cof*r1;
   lu(2,i,j,k) = a1*lu(2,i,j,k) + cof*r2;
   lu(3,i,j,k) = a1*lu(3,i,j,k) + cof*r3;
#undef mu
#undef la
#undef u
#undef lu
#undef strx
#undef stry
#undef strz
#undef acof
#undef bope
#undef ghcof
}
}

//-----------------------------------------------------------------------
__device__ void rhs4lower_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
				   int nk,
				   float_sw4* a_lu, float_sw4* a_u, float_sw4* a_mu, float_sw4* a_lambda, 
				   float_sw4 h, float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
				   int ghost_points )
{
   // Lower boundary nk-5 <= k <= nk
#define mu(i,j,k)     a_mu[base+(i)+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+(i)+ni*(j)+nij*(k)]
#define u(c,i,j,k)    a_u[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]
#define lu(c,i,j,k)  a_lu[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
   //#define strz(k) a_strz[k-kfirst0]
#define acof(i,j,k) dev_acof[(i-1)+6*(j-1)+48*(k-1)]
#define bope(i,j) dev_bope[i-1+6*(j-1)]
#define ghcof(i) dev_ghcof[i-1]
   
   const float_sw4 a1   = 0.;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 i12  = 1.0/12;
   const float_sw4 i144 = 1.0/144;
   const float_sw4 tf   = 0.75;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = ni*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = base-nijk;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   //   const int kfirst0 = kfirst;

   int kb;
   int qb, mb;
   float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4;//, muz1, muz2, muz3, muz4;
   float_sw4 r1, r2, r3, cof, mucof, mu1zz, mu2zz, mu3zz;
   float_sw4 lap2mu, u3zip2, u3zip1, u3zim1, u3zim2, lau3zx, mu3xz, u3zjp2, u3zjp1, u3zjm1, u3zjm2;
   float_sw4 lau3zy, mu3yz, mu1zx, mu2zy, u1zip2, u1zip1, u1zim1, u1zim2;
   float_sw4 u2zjp2, u2zjp1, u2zjm1, u2zjm2, lau1xz, lau2yz;

   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;
   int k = kfirst + ghost_points + threadIdx.z + blockIdx.z*blockDim.z;
   if( k < nk-5 || k > nk )
      return;

   if ( (i <= ilast-2) && (j <= jlast-2) && (k <= klast-2) )
   {
   cof = 1.0/(h*h);

   mux1 = mu(i-1,j,k)*strx(i-1)-
		     tf*(mu(i,j,k)*strx(i)+mu(i-2,j,k)*strx(i-2));
   mux2 = mu(i-2,j,k)*strx(i-2)+mu(i+1,j,k)*strx(i+1)+
		     3*(mu(i,j,k)*strx(i)+mu(i-1,j,k)*strx(i-1));
   mux3 = mu(i-1,j,k)*strx(i-1)+mu(i+2,j,k)*strx(i+2)+
		     3*(mu(i+1,j,k)*strx(i+1)+mu(i,j,k)*strx(i));
   mux4 = mu(i+1,j,k)*strx(i+1)-
		     tf*(mu(i,j,k)*strx(i)+mu(i+2,j,k)*strx(i+2));

   muy1 = mu(i,j-1,k)*stry(j-1)-
		     tf*(mu(i,j,k)*stry(j)+mu(i,j-2,k)*stry(j-2));
   muy2 = mu(i,j-2,k)*stry(j-2)+mu(i,j+1,k)*stry(j+1)+
		     3*(mu(i,j,k)*stry(j)+mu(i,j-1,k)*stry(j-1));
   muy3 = mu(i,j-1,k)*stry(j-1)+mu(i,j+2,k)*stry(j+2)+
		     3*(mu(i,j+1,k)*stry(j+1)+mu(i,j,k)*stry(j));
   muy4 = mu(i,j+1,k)*stry(j+1)-
	       tf*(mu(i,j,k)*stry(j)+mu(i,j+2,k)*stry(j+2));

	    /* xx, yy, and zz derivatives: */
	    /* note that we could have introduced intermediate variables for the average of lambda  */
	    /* in the same way as we did for mu */
   r1 = i6*(strx(i)*((2*mux1+la(i-1,j,k)*strx(i-1)-
                       tf*(la(i,j,k)*strx(i)+la(i-2,j,k)*strx(i-2)))*
                              (u(1,i-2,j,k)-u(1,i,j,k))+
           (2*mux2+la(i-2,j,k)*strx(i-2)+la(i+1,j,k)*strx(i+1)+
                        3*(la(i,j,k)*strx(i)+la(i-1,j,k)*strx(i-1)))*
                              (u(1,i-1,j,k)-u(1,i,j,k))+ 
           (2*mux3+la(i-1,j,k)*strx(i-1)+la(i+2,j,k)*strx(i+2)+
                        3*(la(i+1,j,k)*strx(i+1)+la(i,j,k)*strx(i)))*
                              (u(1,i+1,j,k)-u(1,i,j,k))+
                (2*mux4+ la(i+1,j,k)*strx(i+1)-
                       tf*(la(i,j,k)*strx(i)+la(i+2,j,k)*strx(i+2)))*
                (u(1,i+2,j,k)-u(1,i,j,k)) ) + stry(j)*(
                   + muy1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                     muy2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                     muy3*(u(1,i,j+1,k)-u(1,i,j,k)) +
		   muy4*(u(1,i,j+2,k)-u(1,i,j,k)) ) );

    /* all indices ending with 'b' are indices relative to the boundary, going into the domain (1,2,3,...)*/
   kb = nk-k+1;
    /* all coefficient arrays (acof, bope, ghcof) should be indexed with these indices */
    /* all solution and material property arrays should be indexed with (i,j,k) */

	       /* (mu*uz)_z can not be centered */
	       /* second derivative (mu*u_z)_z at grid point z_k */
	       /* averaging the coefficient */
   mu1zz = 0.;
   mu2zz = 0.;
   mu3zz = 0.;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      mucof = 0.;
      lap2mu = 0.;
      for(  mb=1; mb <= 8; mb++ )
      {
	 mucof  += acof(kb,qb,mb)*mu(i,j,nk-mb+1);
	 lap2mu += acof(kb,qb,mb)*(2*mu(i,j,nk-mb+1)+la(i,j,nk-mb+1));
      }
      mu1zz += mucof*u(1,i,j,nk-qb+1);
      mu2zz += mucof*u(2,i,j,nk-qb+1);
      mu3zz += lap2mu*u(3,i,j,nk-qb+1);
   }
  /* computing the second derivative */
  /* ghost point only influences the first point (k=1) because ghcof(k)=0 for k>=2*/
   r1 = r1 + (mu1zz + ghcof(kb)*mu(i,j,nk)*u(1,i,j,nk+1));

   r2 = i6*(strx(i)*(mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                      mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                      mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                      mux4*(u(2,i+2,j,k)-u(2,i,j,k)) )+ stry(j)*(
                  (2*muy1+la(i,j-1,k)*stry(j-1)-
                        tf*(la(i,j,k)*stry(j)+la(i,j-2,k)*stry(j-2)))*
                          (u(2,i,j-2,k)-u(2,i,j,k))+
           (2*muy2+la(i,j-2,k)*stry(j-2)+la(i,j+1,k)*stry(j+1)+
                        3*(la(i,j,k)*stry(j)+la(i,j-1,k)*stry(j-1)))*
                          (u(2,i,j-1,k)-u(2,i,j,k))+ 
           (2*muy3+la(i,j-1,k)*stry(j-1)+la(i,j+2,k)*stry(j+2)+
                        3*(la(i,j+1,k)*stry(j+1)+la(i,j,k)*stry(j)))*
                          (u(2,i,j+1,k)-u(2,i,j,k))+
                  (2*muy4+la(i,j+1,k)*stry(j+1)-
                       tf*(la(i,j,k)*stry(j)+la(i,j+2,k)*stry(j+2)))*
		  (u(2,i,j+2,k)-u(2,i,j,k)) ) );

		  /* (mu*vz)_z can not be centered */
		  /* second derivative (mu*v_z)_z at grid point z_k */
		  /* averaging the coefficient: already done above */
   r2 = r2 + (mu2zz + ghcof(kb)*mu(i,j,nk)*u(2,i,j,nk+1));

   r3 = i6*(strx(i)*(mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                      mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                      mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                      mux4*(u(3,i+2,j,k)-u(3,i,j,k))  ) + stry(j)*(
                     muy1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                     muy2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                     muy3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                     muy4*(u(3,i,j+2,k)-u(3,i,j,k)) ) );
   r3 = r3 + (mu3zz + ghcof(kb)*(la(i,j,nk)+2*mu(i,j,nk))*
			     u(3,i,j,nk+1));

		  /* cross-terms in first component of rhs */
		  /*   (la*v_y)_x */
   r1 = r1 + strx(i)*stry(j)*(
                 i144*( la(i-2,j,k)*(u(2,i-2,j-2,k)-u(2,i-2,j+2,k)+
                             8*(-u(2,i-2,j-1,k)+u(2,i-2,j+1,k))) - 8*(
                        la(i-1,j,k)*(u(2,i-1,j-2,k)-u(2,i-1,j+2,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i-1,j+1,k))) )+8*(
                        la(i+1,j,k)*(u(2,i+1,j-2,k)-u(2,i+1,j+2,k)+
                             8*(-u(2,i+1,j-1,k)+u(2,i+1,j+1,k))) ) - (
                        la(i+2,j,k)*(u(2,i+2,j-2,k)-u(2,i+2,j+2,k)+
                             8*(-u(2,i+2,j-1,k)+u(2,i+2,j+1,k))) )) 
		 /*   (mu*v_x)_y */
               + i144*( mu(i,j-2,k)*(u(2,i-2,j-2,k)-u(2,i+2,j-2,k)+
                             8*(-u(2,i-1,j-2,k)+u(2,i+1,j-2,k))) - 8*(
                        mu(i,j-1,k)*(u(2,i-2,j-1,k)-u(2,i+2,j-1,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i+1,j-1,k))) )+8*(
                        mu(i,j+1,k)*(u(2,i-2,j+1,k)-u(2,i+2,j+1,k)+
                             8*(-u(2,i-1,j+1,k)+u(2,i+1,j+1,k))) ) - (
                        mu(i,j+2,k)*(u(2,i-2,j+2,k)-u(2,i+2,j+2,k)+
				     8*(-u(2,i-1,j+2,k)+u(2,i+1,j+2,k))) )) );
    /*   (la*w_z)_x: NOT CENTERED */
   u3zip2=0.;
   u3zip1=0.;
   u3zim1=0.;
   u3zim2=0.;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      u3zip2 -= bope(kb,qb)*u(3,i+2,j,nk-qb+1);
      u3zip1 -= bope(kb,qb)*u(3,i+1,j,nk-qb+1);
      u3zim1 -= bope(kb,qb)*u(3,i-1,j,nk-qb+1);
      u3zim2 -= bope(kb,qb)*u(3,i-2,j,nk-qb+1);
   }
   lau3zx= i12*(-la(i+2,j,k)*u3zip2 + 8*la(i+1,j,k)*u3zip1
			 -8*la(i-1,j,k)*u3zim1 + la(i-2,j,k)*u3zim2);
   r1 = r1 + strx(i)*lau3zx;

    /*   (mu*w_x)_z: NOT CENTERED */
   mu3xz=0.;
   for(  qb=1; qb <= 8 ; qb++ )
      mu3xz -= bope(kb,qb)*( mu(i,j,nk-qb+1)*i12*
                  (-u(3,i+2,j,nk-qb+1) + 8*u(3,i+1,j,nk-qb+1)
		   -8*u(3,i-1,j,nk-qb+1) + u(3,i-2,j,nk-qb+1)) );

   r1 = r1 + strx(i)*mu3xz;

	    /* cross-terms in second component of rhs */
	    /*   (mu*u_y)_x */
   r2 = r2 + strx(i)*stry(j)*(
                 i144*( mu(i-2,j,k)*(u(1,i-2,j-2,k)-u(1,i-2,j+2,k)+
                             8*(-u(1,i-2,j-1,k)+u(1,i-2,j+1,k))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j-2,k)-u(1,i-1,j+2,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i-1,j+1,k))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j-2,k)-u(1,i+1,j+2,k)+
                             8*(-u(1,i+1,j-1,k)+u(1,i+1,j+1,k))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j-2,k)-u(1,i+2,j+2,k)+
                             8*(-u(1,i+2,j-1,k)+u(1,i+2,j+1,k))) )) 
		 /* (la*u_x)_y */
               + i144*( la(i,j-2,k)*(u(1,i-2,j-2,k)-u(1,i+2,j-2,k)+
                             8*(-u(1,i-1,j-2,k)+u(1,i+1,j-2,k))) - 8*(
                        la(i,j-1,k)*(u(1,i-2,j-1,k)-u(1,i+2,j-1,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i+1,j-1,k))) )+8*(
                        la(i,j+1,k)*(u(1,i-2,j+1,k)-u(1,i+2,j+1,k)+
                             8*(-u(1,i-1,j+1,k)+u(1,i+1,j+1,k))) ) - (
                        la(i,j+2,k)*(u(1,i-2,j+2,k)-u(1,i+2,j+2,k)+
				     8*(-u(1,i-1,j+2,k)+u(1,i+1,j+2,k))) )) );
	    /* (la*w_z)_y : NOT CENTERED */
   u3zjp2=0.;
   u3zjp1=0.;
   u3zjm1=0.;
   u3zjm2=0.;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      u3zjp2 -= bope(kb,qb)*u(3,i,j+2,nk-qb+1);
      u3zjp1 -= bope(kb,qb)*u(3,i,j+1,nk-qb+1);
      u3zjm1 -= bope(kb,qb)*u(3,i,j-1,nk-qb+1);
      u3zjm2 -= bope(kb,qb)*u(3,i,j-2,nk-qb+1);
   }
   lau3zy= i12*(-la(i,j+2,k)*u3zjp2 + 8*la(i,j+1,k)*u3zjp1
			 -8*la(i,j-1,k)*u3zjm1 + la(i,j-2,k)*u3zjm2);
   r2 = r2 + stry(j)*lau3zy;

	    /* (mu*w_y)_z: NOT CENTERED */
   mu3yz=0.;
   for(  qb=1; qb <= 8 ; qb++ )
      mu3yz -= bope(kb,qb)*( mu(i,j,nk-qb+1)*i12*
                  (-u(3,i,j+2,nk-qb+1) + 8*u(3,i,j+1,nk-qb+1)
                   -8*u(3,i,j-1,nk-qb+1) + u(3,i,j-2,nk-qb+1)) );
   r2 = r2 + stry(j)*mu3yz;

	    /* No centered cross terms in r3 */
	    /*  (mu*u_z)_x: NOT CENTERED */
   u1zip2=0.;
   u1zip1=0.;
   u1zim1=0.;
   u1zim2=0.;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      u1zip2 -= bope(kb,qb)*u(1,i+2,j,nk-qb+1);
      u1zip1 -= bope(kb,qb)*u(1,i+1,j,nk-qb+1);
      u1zim1 -= bope(kb,qb)*u(1,i-1,j,nk-qb+1);
      u1zim2 -= bope(kb,qb)*u(1,i-2,j,nk-qb+1);
   }
   mu1zx= i12*(-mu(i+2,j,k)*u1zip2 + 8*mu(i+1,j,k)*u1zip1
                        -8*mu(i-1,j,k)*u1zim1 + mu(i-2,j,k)*u1zim2);
   r3 = r3 + strx(i)*mu1zx;

	    /* (mu*v_z)_y: NOT CENTERED */
   u2zjp2=0.;
   u2zjp1=0.;
   u2zjm1=0.;
   u2zjm2=0.;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      u2zjp2 -= bope(kb,qb)*u(2,i,j+2,nk-qb+1);
      u2zjp1 -= bope(kb,qb)*u(2,i,j+1,nk-qb+1);
      u2zjm1 -= bope(kb,qb)*u(2,i,j-1,nk-qb+1);
      u2zjm2 -= bope(kb,qb)*u(2,i,j-2,nk-qb+1);
   }
   mu2zy= i12*(-mu(i,j+2,k)*u2zjp2 + 8*mu(i,j+1,k)*u2zjp1
                        -8*mu(i,j-1,k)*u2zjm1 + mu(i,j-2,k)*u2zjm2);
   r3 = r3 + stry(j)*mu2zy;

	    /*   (la*u_x)_z: NOT CENTERED */
   lau1xz=0.;
   for(  qb=1; qb <= 8 ; qb++ )
      lau1xz -= bope(kb,qb)*( la(i,j,nk-qb+1)*i12*
                 (-u(1,i+2,j,nk-qb+1) + 8*u(1,i+1,j,nk-qb+1)
	         -8*u(1,i-1,j,nk-qb+1) + u(1,i-2,j,nk-qb+1)) );
   r3 = r3 + strx(i)*lau1xz;

	    /* (la*v_y)_z: NOT CENTERED */
   lau2yz=0.;
   for(  qb=1; qb <= 8 ; qb++ )
   {
      lau2yz -= bope(kb,qb)*( la(i,j,nk-qb+1)*i12*
                  (-u(2,i,j+2,nk-qb+1) + 8*u(2,i,j+1,nk-qb+1)
                   -8*u(2,i,j-1,nk-qb+1) + u(2,i,j-2,nk-qb+1)) );
   }
   r3 = r3 + stry(j)*lau2yz;

   lu(1,i,j,k) = a1*lu(1,i,j,k) + cof*r1;
   lu(2,i,j,k) = a1*lu(2,i,j,k) + cof*r2;
   lu(3,i,j,k) = a1*lu(3,i,j,k) + cof*r3;
#undef mu
#undef la
#undef u
#undef lu
#undef strx
#undef stry
#undef strz
#undef acof
#undef bope
#undef ghcof
}
}
//-----------------------------------------------------------------------
__global__ void check_nan_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                               float_sw4* u, int *retval_global, int *idx_global )
{
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;
   const size_t npts = static_cast<size_t>((ilast-ifirst+1))*(jlast-jfirst+1)*(klast-kfirst+1);
   int  retval = 0;
   int idx = 0;
   //const size_t ni = ilast-ifirst+1;
   //const size_t nij = ni*(jlast-jfirst+1)
   for ( size_t i = myi; i < 3*npts; i += nthreads )
   {
      if (isnan(u[i]))
      {
         if (retval==0)
            idx = i; 
         retval += 1;
      }
   }

   __syncthreads( );

  atomicAdd(retval_global, retval);
  atomicMin(idx_global, idx);
}

//-----------------------------------------------------------------------
__global__ void forcing_dev( float_sw4 a_t, Sarray* a_F, int ng,
			     GridPointSource* point_sources, int nsrc,
			     int* identsources, int nident, bool a_tt )
{

   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t r = threadIdx.x + static_cast<size_t>(blockIdx.x)* blockDim.x;

   while( r < nident-1 )
   {
      int s0 = identsources[r];
      int g = point_sources[s0].m_grid;
      int i = point_sources[s0].m_i0;
      int j = point_sources[s0].m_j0;
      int k = point_sources[s0].m_k0;
      size_t ind1 = a_F[g].index(1,i,j,k);
      //      size_t ind2 = a_F[g].index(2,i,j,k);
      //      size_t ind3 = a_F[g].index(3,i,j,k);
      size_t oc = a_F[g].m_offc;
      float_sw4* fptr =a_F[g].dev_ptr();
      fptr[ind1] = fptr[ind1+oc] = fptr[ind1+2*oc] = 0;
      for( int s=identsources[r]; s < identsources[r+1] ; s++ )
      {
	float_sw4 fxyz[3];
	if( a_tt )
	   point_sources[s].getFxyztt(a_t,fxyz);
	else
	   point_sources[s].getFxyz(a_t,fxyz);
	fptr[ind1]      += fxyz[0];
	fptr[ind1+oc]   += fxyz[1];
	fptr[ind1+2*oc] += fxyz[2];
      }
      r += nthreads;
   }
}

//-----------------------------------------------------------------------
__global__ void init_forcing_dev( GridPointSource* point_sources, int nsrc )
{
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t r = threadIdx.x + static_cast<size_t>(blockIdx.x)* blockDim.x;
   while( r < nsrc )
   {
      point_sources[r].init_dev();
      r += nthreads;
   }
}

//-----------------------------------------------------------------------
__global__ void HaloToBufferKernel_dev(float_sw4* block_left, float_sw4* block_right, float_sw4* block_up, float_sw4* block_down,
                        float_sw4 * leftSideEdge, float_sw4 * rightSideEdge, float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                        int ni, int nj, int nk, int m_padding, const int m_neighbor0 ,const int  m_neighbor1, const int m_neighbor2,
                        const int m_neighbor3, const int mpi_process_null)
{
   int njk = nj*nk;
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = 3*m_padding;
   size_t npts = static_cast<size_t>(nj*nk);

   if( m_neighbor1 !=  mpi_process_null)
      for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
      {
         idx = i/(n_m_padding);
         j  = i - idx*n_m_padding;
         istart = idx*(3*ni);
         block_up[istart+j] = upSideEdge[i];
      }

   if( m_neighbor0 !=  mpi_process_null)
      for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
      {
         idx = i/(n_m_padding);
         j  = i - idx*n_m_padding;
         istart = idx*(3*ni);
         block_down[istart+j] = downSideEdge[i];
      }

   n_m_padding = 3*m_padding*ni;
   npts = static_cast<size_t>(nk);

   if( m_neighbor2 !=  mpi_process_null)
      for (i = myi; i < n_m_padding*npts; i += nthreads)
      {
         idx = i/n_m_padding;
         j  = i - idx*n_m_padding;
         istart = idx*(3*ni*nj);
         block_left[istart+j] = leftSideEdge[i];
      }


   if( m_neighbor3 !=  mpi_process_null)
      for (i = myi; i < n_m_padding*npts; i += nthreads)
      {
         idx = i/n_m_padding;
         j  = i - idx*n_m_padding;
         istart = idx*(3*ni*nj);
         block_right[istart+j] = rightSideEdge[i];
      }

}

//-----------------------------------------------------------------------
__global__ void HaloToBufferKernel_dev_rev_v2(float_sw4* block_left, float_sw4* block_right, 
                        float_sw4 * leftSideEdge, float_sw4 * rightSideEdge, 
                        int ni, int nj, int nk, int m_padding, int size, int nstep, const int m_neighbor_left ,const int  m_neighbor_right, const int mpi_process_null)
{
   int njk = nj*nk;
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = m_padding;
   size_t npts = static_cast<size_t>(size);

   if( m_neighbor_left !=  mpi_process_null)
      for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
      {
         idx = i/(n_m_padding);
         j  = i - idx*n_m_padding;
         istart = idx*nstep;
         block_left[istart+j] = leftSideEdge[i];
      }

   if( m_neighbor_right !=  mpi_process_null)
      for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
      {
         idx = i/(n_m_padding);
         j  = i - idx*n_m_padding;
         istart = idx*nstep;
         block_right[istart+j] = rightSideEdge[i];
      }

}
//-----------------------------------------------------------------------
__global__ void HaloToBufferKernel_dev_rev(float_sw4* block_left, float_sw4* block_right, float_sw4* block_up, float_sw4* block_down,
                        float_sw4 * leftSideEdge, float_sw4 * rightSideEdge, float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                        int ni, int nj, int nk, int m_padding, const int m_neighbor0 ,const int  m_neighbor1, const int m_neighbor2,
                        const int m_neighbor3, const int mpi_process_null)
{
   int njk = nj*nk;
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = m_padding;
   size_t npts = static_cast<size_t>(3*nj*nk);

   if( m_neighbor1 !=  mpi_process_null)
      for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
      {
         idx = i/(n_m_padding);
         j  = i - idx*n_m_padding;
         istart = idx*ni;
         block_up[istart+j] = upSideEdge[i];
      }

   if( m_neighbor0 !=  mpi_process_null)
      for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
      {
         idx = i/(n_m_padding);
         j  = i - idx*n_m_padding;
         istart = idx*ni;
         block_down[istart+j] = downSideEdge[i];
      }

   n_m_padding = m_padding*ni;
   npts = static_cast<size_t>(3*nk);

   if( m_neighbor2 !=  mpi_process_null)
      for (i = myi; i < n_m_padding*npts; i += nthreads)
      {
         idx = i/n_m_padding;
         j  = i - idx*n_m_padding;
         istart = idx*(ni*nj);
         block_left[istart+j] = leftSideEdge[i];
      }


   if( m_neighbor3 !=  mpi_process_null)
      for (i = myi; i < n_m_padding*npts; i += nthreads)
      {
         idx = i/n_m_padding;
         j  = i - idx*n_m_padding;
         istart = idx*(ni*nj);
         block_right[istart+j] = rightSideEdge[i];
      }

}

//-----------------------------------------------------------------------
__global__ void BufferToHaloKernel_dev(float_sw4* block_left, float_sw4* block_right, float_sw4* block_up, float_sw4* block_down,
                float_sw4 * leftSideEdge, float_sw4 * rightSideEdge, float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                int ni, int nj, int nk, int m_padding, const int m_neighbor0 ,const int  m_neighbor1, const int m_neighbor2,
                        const int m_neighbor3, const int mpi_process_null )
{
   int njk = nj*nk;
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = 3*m_padding;
   size_t npts = static_cast<size_t>(nj*nk);

   if( m_neighbor1 !=  mpi_process_null)
   for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
   {
        idx = i/(n_m_padding);
        j  = i - idx*n_m_padding;
        istart = idx*(3*ni);
        upSideEdge[i] = block_up[istart+j];
   }

   if( m_neighbor0 !=  mpi_process_null)
   for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
   {
        idx = i/(n_m_padding);
        j  = i - idx*n_m_padding;
        istart = idx*(3*ni);
        downSideEdge[i] = block_down[istart+j];
   }

   n_m_padding = 3*m_padding*ni;
   npts = static_cast<size_t>(nk);

   if( m_neighbor2 !=  mpi_process_null)
   for (i = myi; i < n_m_padding*npts; i += nthreads)
   {
        idx = i/n_m_padding;
        j  = i - idx*n_m_padding;
        istart = idx*(3*ni*nj);
        leftSideEdge[i] = block_left[istart+j];
   }

   if( m_neighbor3 !=  mpi_process_null)
   for (i = myi; i < n_m_padding*npts; i += nthreads)
   {
        idx = i/n_m_padding;
        j  = i - idx*n_m_padding;
        istart = idx*(3*ni*nj);
        rightSideEdge[i] = block_right[istart+j];
   }

}

//-----------------------------------------------------------------------
__global__ void BufferToHaloKernel_dev_rev_v2(float_sw4* block_left, float_sw4* block_right, 
                float_sw4 * leftSideEdge, float_sw4 * rightSideEdge,
                int ni, int nj, int nk, int m_padding, int size, int nstep, const int m_neighbor_left ,const int  m_neighbor_right, const int mpi_process_null )
{
   int njk = nj*nk;
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = m_padding;
   size_t npts = static_cast<size_t>(size);

   if( m_neighbor_left !=  mpi_process_null)
   for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
   {
        idx = i/(n_m_padding);
        j  = i - idx*n_m_padding;
        istart = idx*nstep;
        leftSideEdge[i] = block_left[istart+j];
   }

   if( m_neighbor_right !=  mpi_process_null)
   for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
   {
        idx = i/(n_m_padding);
        j  = i - idx*n_m_padding;
        istart = idx*nstep;
        rightSideEdge[i] = block_right[istart+j];
   }

}
//-----------------------------------------------------------------------
__global__ void BufferToHaloKernel_dev_rev(float_sw4* block_left, float_sw4* block_right, float_sw4* block_up, float_sw4* block_down,
                float_sw4 * leftSideEdge, float_sw4 * rightSideEdge, float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                int ni, int nj, int nk, int m_padding, const int m_neighbor0 ,const int  m_neighbor1, const int m_neighbor2,
                        const int m_neighbor3, const int mpi_process_null )
{
   int njk = nj*nk;
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = m_padding;
   size_t npts = static_cast<size_t>(3*nj*nk);

   if( m_neighbor1 !=  mpi_process_null)
   for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
   {
        idx = i/(n_m_padding);
        j  = i - idx*n_m_padding;
        istart = idx*ni;
        upSideEdge[i] = block_up[istart+j];
   }

   if( m_neighbor0 !=  mpi_process_null)
   for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
   {
        idx = i/(n_m_padding);
        j  = i - idx*n_m_padding;
        istart = idx*ni;
        downSideEdge[i] = block_down[istart+j];
   }

   n_m_padding = m_padding*ni;
   npts = static_cast<size_t>(3*nk);

   if( m_neighbor2 !=  mpi_process_null)
   for (i = myi; i < n_m_padding*npts; i += nthreads)
   {
        idx = i/n_m_padding;
        j  = i - idx*n_m_padding;
        istart = idx*(ni*nj);
        leftSideEdge[i] = block_left[istart+j];
   }

   if( m_neighbor3 !=  mpi_process_null)
   for (i = myi; i < n_m_padding*npts; i += nthreads)
   {
        idx = i/n_m_padding;
        j  = i - idx*n_m_padding;
        istart = idx*(ni*nj);
        rightSideEdge[i] = block_right[istart+j];
   }

}
//-----------------------------------------------------------------------
__global__ void freesurfcurvisg_dev_rev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                                     int nz, int side, float_sw4* a_u, float_sw4* a_mu, 
                                     float_sw4* a_lambda, float_sw4* a_met,
                                     float_sw4* a_forcing,float_sw4* a_strx, float_sw4* a_stry, int ghost_points )
{
#define mu(i,j,k)     a_mu[base+(i)+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+(i)+ni*(j)+nij*(k)]
#define u(c,i,j,k)   a_u[base3+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define met(c,i,j,k) a_met[base4+(i)+ni*(j)+nij*(k)+nijk*(c)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define forcing(c,i,j)   a_forcing[3*basef-1+(c)+3*(i)+ni3*(j)]

   const float_sw4 c1=2.0/3, c2=-1.0/12;
   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = nij*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int basef = -(ifirst+ni*jfirst);
   const int base3 = base-nijk;
   const int base4 = base-nijk;
   const int ni3  = 3*ni;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;

   int k, kl;
   if( side == 5 )
   {
      k = 1;
      kl= 1;
   }
   else if( side == 6 )
   {
      k = nz;
      kl= -1;
   }

   if ( (i <= ilast-2) && (j <= jlast-2) )
   {
      float_sw4 s0i = 1/dev_sbop[0];
      float_sw4 istry = 1/stry(j);
      float_sw4 istrx = 1/strx(i);

    // First tangential derivatives
            float_sw4 rhs1 = 
// pr
        (2*mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(1,i,j,k)*(
               c2*(u(1,i+2,j,k)-u(1,i-2,j,k)) +
               c1*(u(1,i+1,j,k)-u(1,i-1,j,k))  )*strx(i)*istry 
       + mu(i,j,k)*met(3,i,j,k)*met(1,i,j,k)*(
             c2*(u(2,i+2,j,k)-u(2,i-2,j,k)) +
             c1*(u(2,i+1,j,k)-u(2,i-1,j,k))  ) 
       + mu(i,j,k)*met(4,i,j,k)*met(1,i,j,k)*(
             c2*(u(3,i+2,j,k)-u(3,i-2,j,k)) +
             c1*(u(3,i+1,j,k)-u(3,i-1,j,k))  )*istry   
// qr
       + mu(i,j,k)*met(3,i,j,k)*met(1,i,j,k)*(
             c2*(u(1,i,j+2,k)-u(1,i,j-2,k)) +
             c1*(u(1,i,j+1,k)-u(1,i,j-1,k))   )*istrx*stry(j) 
       + la(i,j,k)*met(2,i,j,k)*met(1,i,j,k)*(
             c2*(u(2,i,j+2,k)-u(2,i,j-2,k)) +
             c1*(u(2,i,j+1,k)-u(2,i,j-1,k))  )  -
	       forcing(1,i,j);

// (v-eq)
            float_sw4 rhs2 = 
// pr
         la(i,j,k)*met(3,i,j,k)*met(1,i,j,k)*(
             c2*(u(1,i+2,j,k)-u(1,i-2,j,k)) +
             c1*(u(1,i+1,j,k)-u(1,i-1,j,k))   ) 
       + mu(i,j,k)*met(2,i,j,k)*met(1,i,j,k)*(
             c2*(u(2,i+2,j,k)-u(2,i-2,j,k)) +
             c1*(u(2,i+1,j,k)-u(2,i-1,j,k))  )*strx(i)*istry 
// qr
       + mu(i,j,k)*met(2,i,j,k)*met(1,i,j,k)*(
             c2*(u(1,i,j+2,k)-u(1,i,j-2,k)) +
             c1*(u(1,i,j+1,k)-u(1,i,j-1,k))   ) 
      + (2*mu(i,j,k)+la(i,j,k))*met(3,i,j,k)*met(1,i,j,k)*(
             c2*(u(2,i,j+2,k)-u(2,i,j-2,k)) +
             c1*(u(2,i,j+1,k)-u(2,i,j-1,k))  )*stry(j)*istrx 
       + mu(i,j,k)*met(4,i,j,k)*met(1,i,j,k)*(
             c2*(u(3,i,j+2,k)-u(3,i,j-2,k)) +
             c1*(u(3,i,j+1,k)-u(3,i,j-1,k))   )*istrx -
	       forcing(2,i,j);

// (w-eq)
            float_sw4 rhs3 = 
// pr
         la(i,j,k)*met(4,i,j,k)*met(1,i,j,k)*(
             c2*(u(1,i+2,j,k)-u(1,i-2,j,k)) +
             c1*(u(1,i+1,j,k)-u(1,i-1,j,k))   )*istry 
       + mu(i,j,k)*met(2,i,j,k)*met(1,i,j,k)*(
             c2*(u(3,i+2,j,k)-u(3,i-2,j,k)) +
             c1*(u(3,i+1,j,k)-u(3,i-1,j,k))  )*strx(i)*istry
// qr 
       + mu(i,j,k)*met(3,i,j,k)*met(1,i,j,k)*(
             c2*(u(3,i,j+2,k)-u(3,i,j-2,k)) +
             c1*(u(3,i,j+1,k)-u(3,i,j-1,k))   )*stry(j)*istrx
       + la(i,j,k)*met(4,i,j,k)*met(1,i,j,k)*(
             c2*(u(2,i,j+2,k)-u(2,i,j-2,k)) +
             c1*(u(2,i,j+1,k)-u(2,i,j-1,k))  )*istrx -
	       forcing(3,i,j);

// Normal derivatives
            float_sw4 ac = strx(i)*istry*met(2,i,j,k)*met(2,i,j,k)+
	       stry(j)*istrx*met(3,i,j,k)*met(3,i,j,k)+
	       met(4,i,j,k)*met(4,i,j,k)*istry*istrx;
            float_sw4 bc = 1/(mu(i,j,k)*ac);
            float_sw4 cc = (mu(i,j,k)+la(i,j,k))/(2*mu(i,j,k)+la(i,j,k))*bc/ac;

            float_sw4 xoysqrt = sqrt(strx(i)*istry);
            float_sw4 yoxsqrt = 1/xoysqrt;
            float_sw4 isqrtxy = istrx*xoysqrt;
            float_sw4 dc = cc*( xoysqrt*met(2,i,j,k)*rhs1 + 
		      yoxsqrt*met(3,i,j,k)*rhs2 + isqrtxy*met(4,i,j,k)*rhs3);

            u(1,i,j,k-kl) = -s0i*(  dev_sbop[1]*u(1,i,j,k)+dev_sbop[2]*u(1,i,j,k+kl)+
                dev_sbop[3]*u(1,i,j,k+2*kl)+dev_sbop[4]*u(1,i,j,k+3*kl) + bc*rhs1 - 
				    dc*met(2,i,j,k)*xoysqrt );
            u(2,i,j,k-kl) = -s0i*(  dev_sbop[1]*u(2,i,j,k)+dev_sbop[2]*u(2,i,j,k+kl)+
                dev_sbop[3]*u(2,i,j,k+2*kl)+dev_sbop[4]*u(2,i,j,k+3*kl) + bc*rhs2 - 
				    dc*met(3,i,j,k)*yoxsqrt );
            u(3,i,j,k-kl) = -s0i*(  dev_sbop[1]*u(3,i,j,k)+dev_sbop[2]*u(3,i,j,k+kl)+
                dev_sbop[3]*u(3,i,j,k+2*kl)+dev_sbop[4]*u(3,i,j,k+3*kl) + bc*rhs3 - 
				    dc*met(4,i,j,k)*isqrtxy );
   }

#undef mu
#undef la
#undef u
#undef met
#undef strx
#undef stry
#undef forcing
}

//-----------------------------------------------------------------------
__global__ void freesurfcurvisg_dev( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                                     int nz, int side, float_sw4* a_u, float_sw4* a_mu, 
                                     float_sw4* a_lambda, float_sw4* a_met,
                                     float_sw4* a_forcing,float_sw4* a_strx, float_sw4* a_stry, int ghost_points )
{
#define mu(i,j,k)     a_mu[base+i+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+i+ni*(j)+nij*(k)]
#define u(c,i,j,k)     a_u[base3+(c)+3*(i)+ni3*(j)+nij3*(k)]   
#define met(c,i,j,k) a_met[base4+(c)+4*(i)+ni4*(j)+nij4*(k)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define forcing(c,i,j)   a_forcing[basef+(c)+3*(i)+ni3*(j)]

   const float_sw4 c1=2.0/3, c2=-1.0/12;
   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = 3*base-1;
   const int base4 = 4*base-1;
   const int basef = -3*(ifirst+ni*jfirst)-1;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;
   const int ni3  = 3*ni;
   const int ni4  = 4*ni;
   const int nij3 = 3*nij;
   const int nij4 = 4*nij;
   int i = ifirst + ghost_points + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirst + ghost_points + threadIdx.y + blockIdx.y*blockDim.y;

   int k, kl;
   if( side == 5 )
   {
      k = 1;
      kl= 1;
   }
   else if( side == 6 )
   {
      k = nz;
      kl= -1;
   }

   if ( (i <= ilast-2) && (j <= jlast-2) )
   {
      float_sw4 s0i = 1/dev_sbop[0];
      float_sw4 istry = 1/stry(j);
      float_sw4 istrx = 1/strx(i);

    // First tangential derivatives
            float_sw4 rhs1 = 
// pr
        (2*mu(i,j,k)+la(i,j,k))*met(2,i,j,k)*met(1,i,j,k)*(
               c2*(u(1,i+2,j,k)-u(1,i-2,j,k)) +
               c1*(u(1,i+1,j,k)-u(1,i-1,j,k))  )*strx(i)*istry 
       + mu(i,j,k)*met(3,i,j,k)*met(1,i,j,k)*(
             c2*(u(2,i+2,j,k)-u(2,i-2,j,k)) +
             c1*(u(2,i+1,j,k)-u(2,i-1,j,k))  ) 
       + mu(i,j,k)*met(4,i,j,k)*met(1,i,j,k)*(
             c2*(u(3,i+2,j,k)-u(3,i-2,j,k)) +
             c1*(u(3,i+1,j,k)-u(3,i-1,j,k))  )*istry   
// qr
       + mu(i,j,k)*met(3,i,j,k)*met(1,i,j,k)*(
             c2*(u(1,i,j+2,k)-u(1,i,j-2,k)) +
             c1*(u(1,i,j+1,k)-u(1,i,j-1,k))   )*istrx*stry(j) 
       + la(i,j,k)*met(2,i,j,k)*met(1,i,j,k)*(
             c2*(u(2,i,j+2,k)-u(2,i,j-2,k)) +
             c1*(u(2,i,j+1,k)-u(2,i,j-1,k))  )  -
	       forcing(1,i,j);

// (v-eq)
            float_sw4 rhs2 = 
// pr
         la(i,j,k)*met(3,i,j,k)*met(1,i,j,k)*(
             c2*(u(1,i+2,j,k)-u(1,i-2,j,k)) +
             c1*(u(1,i+1,j,k)-u(1,i-1,j,k))   ) 
       + mu(i,j,k)*met(2,i,j,k)*met(1,i,j,k)*(
             c2*(u(2,i+2,j,k)-u(2,i-2,j,k)) +
             c1*(u(2,i+1,j,k)-u(2,i-1,j,k))  )*strx(i)*istry 
// qr
       + mu(i,j,k)*met(2,i,j,k)*met(1,i,j,k)*(
             c2*(u(1,i,j+2,k)-u(1,i,j-2,k)) +
             c1*(u(1,i,j+1,k)-u(1,i,j-1,k))   ) 
      + (2*mu(i,j,k)+la(i,j,k))*met(3,i,j,k)*met(1,i,j,k)*(
             c2*(u(2,i,j+2,k)-u(2,i,j-2,k)) +
             c1*(u(2,i,j+1,k)-u(2,i,j-1,k))  )*stry(j)*istrx 
       + mu(i,j,k)*met(4,i,j,k)*met(1,i,j,k)*(
             c2*(u(3,i,j+2,k)-u(3,i,j-2,k)) +
             c1*(u(3,i,j+1,k)-u(3,i,j-1,k))   )*istrx -
	       forcing(2,i,j);

// (w-eq)
            float_sw4 rhs3 = 
// pr
         la(i,j,k)*met(4,i,j,k)*met(1,i,j,k)*(
             c2*(u(1,i+2,j,k)-u(1,i-2,j,k)) +
             c1*(u(1,i+1,j,k)-u(1,i-1,j,k))   )*istry 
       + mu(i,j,k)*met(2,i,j,k)*met(1,i,j,k)*(
             c2*(u(3,i+2,j,k)-u(3,i-2,j,k)) +
             c1*(u(3,i+1,j,k)-u(3,i-1,j,k))  )*strx(i)*istry
// qr 
       + mu(i,j,k)*met(3,i,j,k)*met(1,i,j,k)*(
             c2*(u(3,i,j+2,k)-u(3,i,j-2,k)) +
             c1*(u(3,i,j+1,k)-u(3,i,j-1,k))   )*stry(j)*istrx
       + la(i,j,k)*met(4,i,j,k)*met(1,i,j,k)*(
             c2*(u(2,i,j+2,k)-u(2,i,j-2,k)) +
             c1*(u(2,i,j+1,k)-u(2,i,j-1,k))  )*istrx -
	       forcing(3,i,j);

// Normal derivatives
            float_sw4 ac = strx(i)*istry*met(2,i,j,k)*met(2,i,j,k)+
	       stry(j)*istrx*met(3,i,j,k)*met(3,i,j,k)+
	       met(4,i,j,k)*met(4,i,j,k)*istry*istrx;
            float_sw4 bc = 1/(mu(i,j,k)*ac);
            float_sw4 cc = (mu(i,j,k)+la(i,j,k))/(2*mu(i,j,k)+la(i,j,k))*bc/ac;

            float_sw4 xoysqrt = sqrt(strx(i)*istry);
            float_sw4 yoxsqrt = 1/xoysqrt;
            float_sw4 isqrtxy = istrx*xoysqrt;
            float_sw4 dc = cc*( xoysqrt*met(2,i,j,k)*rhs1 + 
		      yoxsqrt*met(3,i,j,k)*rhs2 + isqrtxy*met(4,i,j,k)*rhs3);

            u(1,i,j,k-kl) = -s0i*(  dev_sbop[1]*u(1,i,j,k)+dev_sbop[2]*u(1,i,j,k+kl)+
                dev_sbop[3]*u(1,i,j,k+2*kl)+dev_sbop[4]*u(1,i,j,k+3*kl) + bc*rhs1 - 
				    dc*met(2,i,j,k)*xoysqrt );
            u(2,i,j,k-kl) = -s0i*(  dev_sbop[1]*u(2,i,j,k)+dev_sbop[2]*u(2,i,j,k+kl)+
                dev_sbop[3]*u(2,i,j,k+2*kl)+dev_sbop[4]*u(2,i,j,k+3*kl) + bc*rhs2 - 
				    dc*met(3,i,j,k)*yoxsqrt );
            u(3,i,j,k-kl) = -s0i*(  dev_sbop[1]*u(3,i,j,k)+dev_sbop[2]*u(3,i,j,k+kl)+
                dev_sbop[3]*u(3,i,j,k+2*kl)+dev_sbop[4]*u(3,i,j,k+3*kl) + bc*rhs3 - 
				    dc*met(4,i,j,k)*isqrtxy );
   }

#undef mu
#undef la
#undef u
#undef met
#undef strx
#undef stry
#undef forcing
}

//-----------------------------------------------------------------------
__global__ void bcfortsg_dev_indrev( int ib, int ie, int jb, int je, int kb, int ke, int* wind,
                                     int nx, int ny, int nz, float_sw4* u, float_sw4 h, boundaryConditionType *bccnd,
                                     float_sw4* mu, float_sw4* la, float_sw4 t,
                                     float_sw4* bforce1, float_sw4* bforce2, float_sw4* bforce3,
                                     float_sw4* bforce4, float_sw4* bforce5, float_sw4* bforce6,
                                     float_sw4 om, float_sw4 ph, float_sw4 cv,
                                     float_sw4* strx, float_sw4* stry )
{
  const float_sw4 d4a = 2.0/3.0;
  const float_sw4 d4b = -1.0/12.0;
  const size_t ni  = ie-ib+1;
  const size_t nij = ni*(je-jb+1);
  const size_t npts = static_cast<size_t>((ie-ib+1))*(je-jb+1)*(ke-kb+1);
  for( int s=0 ; s < 6 ; s++ )
  {
    if( bccnd[s]==1 || bccnd[s]==2 )
    {
      size_t idel = 1+wind[1+6*s]-wind[6*s];
      size_t ijdel = idel * (1+wind[3+6*s]-wind[2+6*s]);
      int i = wind[6*s] + threadIdx.x + blockIdx.x*blockDim.x;
      int j = wind[6*s+2] + threadIdx.y + blockIdx.y*blockDim.y;
      int k = wind[6*s+4] +  threadIdx.z + blockIdx.z*blockDim.z;
      if( i > wind[6*s+1] || j > wind[6*s+3] || k > wind[6*s+5])
        continue;
      int qq = i-wind[6*s] + idel*(j-wind[6*s+2]) + (k-wind[4+6*s])*ijdel;
      size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
      if( s== 0 )
      {
        u[ind  ]      = bforce1[  3*qq];
        u[ind+npts]   = bforce1[1+3*qq];
        u[ind+2*npts] = bforce1[2+3*qq];
      }
      else if( s== 1 )
      {
        u[ind]        = bforce2[  3*qq];
        u[ind+npts]   = bforce2[1+3*qq];
        u[ind+2*npts] = bforce2[2+3*qq];
      }
      else if( s==2 )
      {
        u[ind  ] = bforce3[  3*qq];
        u[ind+npts] = bforce3[1+3*qq];
        u[ind+2*npts] = bforce3[2+3*qq];
      }
      else if( s==3 )
      {
        u[ind  ] = bforce4[  3*qq];
        u[ind+npts] = bforce4[1+3*qq];
        u[ind+2*npts] = bforce4[2+3*qq];
      }
      else if( s==4 )
      {
        u[ind  ] = bforce5[  3*qq];
        u[ind+npts] = bforce5[1+3*qq];
        u[ind+2*npts] = bforce5[2+3*qq];
      }
      else if( s==5 )
      {
        u[ind  ] = bforce6[  3*qq];
        u[ind+npts] = bforce6[1+3*qq];
        u[ind+2*npts] = bforce6[2+3*qq];
      }
    }
    else if( bccnd[s]==3 )
    {
      int i = wind[6*s] + threadIdx.x + blockIdx.x*blockDim.x;
      int j = wind[6*s+2] + threadIdx.y + blockIdx.y*blockDim.y;
      int k = wind[6*s+4] +  threadIdx.z + blockIdx.z*blockDim.z;
      if( i > wind[6*s+1] || j > wind[6*s+3] || k > wind[6*s+5])
        continue;
      size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
      if( s==0 )
      {
        size_t indp= ind+nx;
        u[ind  ] = u[indp];
        u[ind+npts] = u[indp+npts];
        u[ind+2*npts] = u[indp+2*npts];
      }
      else if( s==1 )
      {
        size_t indp= ind-nx;
        u[ind  ] = u[indp];
        u[ind+npts] = u[indp+npts];
        u[ind+2*npts] = u[indp+2*npts];
      }
      else if( s==2 )
      {
        size_t indp= ind+ni*ny;
        u[ind  ] = u[indp];
        u[ind+npts] = u[indp+npts];
        u[ind+2*npts] = u[indp+2*npts];
      }
      else if( s==3 )
      {
        size_t indp= ind-ni*ny;
        u[ind ] = u[indp];
        u[ind+npts] = u[indp+npts];
        u[ind+2*npts] = u[indp+2*npts];
      }
      else if( s==4 )
      {
        size_t indp= ind+nij*nz;
        u[ind  ] = u[indp];
        u[ind+npts] = u[indp+npts];
        u[ind+2*npts] = u[indp+2*npts];
      }
      else if( s==5 )
      {
        size_t indp= ind-nij*nz;
        u[ind  ] = u[indp];
        u[ind+npts] = u[indp+npts];
        u[ind+2*npts] = u[indp+2*npts];
      }
    }
    else if( bccnd[s]==0 )
    {
      if( s != 4 && s != 5)
      {
        printf("EW::bcfortsg_dev_indrev, ERROR: Free surface conditio not implemented for side\n");
        return ;
      }

      int i = ib+2 + threadIdx.x + blockIdx.x*blockDim.x;
      int j = jb+2 + threadIdx.y + blockIdx.y*blockDim.y;
      int kk = kb+2 + threadIdx.z + blockIdx.z*blockDim.z;
      if( i > ie-2 || j > je-2 || kk != kb+2)
        continue;

      if( s==4 )
      {
        int k=1, kl=1;
        size_t qq = i-ib+ni*(j-jb);
        size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
        float_sw4 wx = strx[i-ib]*(d4a*(u[2*npts+ind+1]-u[2*npts+ind-1])+d4b*(u[2*npts+ind+2]-u[2*npts+ind-2]));
        float_sw4 ux = strx[i-ib]*(d4a*(u[ind+1]-u[ind-1])+d4b*(u[ind+2]-u[ind-2]));
        float_sw4 wy = stry[j-jb]*(d4a*(u[2*npts+ind+ni  ]-u[2*npts+ind-ni  ])+
                                   d4b*(u[2*npts+ind+2*ni]-u[2*npts+ind-2*ni]));
        float_sw4 vy = stry[j-jb]*(d4a*(u[npts+ind+ni]  -u[npts+ind-ni])+
                                   d4b*(u[npts+ind+2*ni]-u[npts+ind-2*ni]));
        float_sw4 uz=0., vz=0., wz=0.;
        for( int w=1 ; w <= 4 ; w++ )
        {
          uz += dev_sbop[w]*u[       ind+nij*kl*(w-1)];
          vz += dev_sbop[w]*u[npts  +ind+nij*kl*(w-1)];
          wz += dev_sbop[w]*u[2*npts+ind+nij*kl*(w-1)];
        }
        u[       ind-nij*kl] = (-uz-kl*wx+kl*h*bforce5[  3*qq]/mu[ind])/dev_sbop[0];
        u[npts  +ind-nij*kl] = (-vz-kl*wy+kl*h*bforce5[1+3*qq]/mu[ind])/dev_sbop[0];
        u[2*npts+ind-nij*kl] = (-wz + (-kl*la[ind]*(ux+vy)+kl*h*bforce5[2+3*qq])/
                                (2*mu[ind]+la[ind]))/dev_sbop[0];
      }
      else
      {
        int k=nz, kl=-1;
        size_t qq = i-ib+ni*(j-jb);
        size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
        float_sw4 wx = strx[i-ib]*(d4a*(u[2*npts+ind+1]-u[2*npts+ind-1])+d4b*(u[2*npts+ind+2]-u[2*npts+ind-2]));
        float_sw4 ux = strx[i-ib]*(d4a*(u[ind+1]-u[ind-1])+d4b*(u[ind+2]-u[ind-2]));
        float_sw4 wy = stry[j-jb]*(d4a*(u[2*npts+ind+ni  ]-u[2*npts+ind-ni  ])+
                                   d4b*(u[2*npts+ind+2*ni]-u[2*npts+ind-2*ni]));
        float_sw4 vy = stry[j-jb]*(d4a*(u[npts+ind+ni]  -u[npts+ind-ni])+
                                   d4b*(u[npts+ind+2*ni]-u[npts+ind-2*ni]));
        float_sw4 uz=0., vz=0., wz=0.;
        for( int w=1 ; w <= 4 ; w++ )
        {
          uz += dev_sbop[w]*u[       ind+nij*kl*(w-1)];
          vz += dev_sbop[w]*u[npts  +ind+nij*kl*(w-1)];
          wz += dev_sbop[w]*u[2*npts+ind+nij*kl*(w-1)];
        }
        u[       ind-nij*kl] = (-uz-kl*wx+kl*h*bforce6[  3*qq]/mu[ind])/dev_sbop[0];
        u[npts  +ind-nij*kl] = (-vz-kl*wy+kl*h*bforce6[1+3*qq]/mu[ind])/dev_sbop[0];
        u[2*npts+ind-nij*kl] = (-wz+(-kl*la[ind]*(ux+vy)+kl*h*bforce6[2+3*qq])/
                                (2*mu[ind]+la[ind]))/dev_sbop[0];

      }
    }
  }
}
                                                           

//-----------------------------------------------------------------------
__global__ void bcfortsg_dev( int ib, int ie, int jb, int je, int kb, int ke, int* wind,
                              int nx, int ny, int nz, float_sw4* u, float_sw4 h, boundaryConditionType *bccnd,
                              float_sw4* mu, float_sw4* la, float_sw4 t,
                              float_sw4* bforce1, float_sw4* bforce2, float_sw4* bforce3,
                              float_sw4* bforce4, float_sw4* bforce5, float_sw4* bforce6,
                              float_sw4 om, float_sw4 ph, float_sw4 cv,
                              float_sw4* strx, float_sw4* stry )
{
  const float_sw4 d4a = 2.0/3.0;
  const float_sw4 d4b = -1.0/12.0;
  const size_t ni  = ie-ib+1;
  const size_t nij = ni*(je-jb+1);

  for( int s=0 ; s < 6 ; s++ )
  {

    if( bccnd[s]==1 || bccnd[s]==2 )
    {
      size_t idel = 1+wind[1+6*s]-wind[6*s];
      size_t ijdel = idel * (1+wind[3+6*s]-wind[2+6*s]);
      int i = wind[6*s] + threadIdx.x + blockIdx.x*blockDim.x;
      int j = wind[6*s+2] + threadIdx.y + blockIdx.y*blockDim.y;
      int k = wind[6*s+4] +  threadIdx.z + blockIdx.z*blockDim.z;
      if( i > wind[6*s+1] || j > wind[6*s+3] || k > wind[6*s+5])
        continue;
      int qq = i-wind[6*s] + idel*(j-wind[6*s+2]) + (k-wind[4+6*s])*ijdel;
      size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
      if( s== 0 )
      {
        u[3*ind  ] = bforce1[  3*qq];
        u[3*ind+1] = bforce1[1+3*qq];
        u[3*ind+2] = bforce1[2+3*qq];

      }
      else if( s== 1 )
      {
        u[3*ind  ] = bforce2[  3*qq];
        u[3*ind+1] = bforce2[1+3*qq];
        u[3*ind+2] = bforce2[2+3*qq];
      }
      else if( s==2 )
      {
        u[3*ind  ] = bforce3[  3*qq];
        u[3*ind+1] = bforce3[1+3*qq];
        u[3*ind+2] = bforce3[2+3*qq];
      }
      else if( s==3 )
      {
        u[3*ind  ] = bforce4[  3*qq];
        u[3*ind+1] = bforce4[1+3*qq];
        u[3*ind+2] = bforce4[2+3*qq];
      }
      else if( s==4 )
      {
        u[3*ind  ] = bforce5[  3*qq];
        u[3*ind+1] = bforce5[1+3*qq];
        u[3*ind+2] = bforce5[2+3*qq];
      }
      else if( s==5 )
      {
        u[3*ind  ] = bforce6[  3*qq];
        u[3*ind+1] = bforce6[1+3*qq];
        u[3*ind+2] = bforce6[2+3*qq];
      }
    }
    else if( bccnd[s]==3 )
    {
      int i = wind[6*s] + threadIdx.x + blockIdx.x*blockDim.x;
      int j = wind[6*s+2] + threadIdx.y + blockIdx.y*blockDim.y;
      int k = wind[6*s+4] +  threadIdx.z + blockIdx.z*blockDim.z;
      if( i > wind[6*s+1] || j > wind[6*s+3] || k > wind[6*s+5])
        continue;
      size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
      if( s==0 )
      {
        size_t indp= ind+nx;
        u[3*ind  ] = u[3*indp];
        u[3*ind+1] = u[3*indp+1];
        u[3*ind+2] = u[3*indp+2];
      }
      else if( s==1 )
      {
        size_t indp= ind-nx;
        u[3*ind  ] = u[3*indp];
        u[3*ind+1] = u[3*indp+1];
        u[3*ind+2] = u[3*indp+2];
      }
      else if( s==2 )
      {
        size_t indp= ind+ni*ny;
        u[3*ind  ] = u[3*indp];
        u[3*ind+1] = u[3*indp+1];
        u[3*ind+2] = u[3*indp+2];
      }
      else if( s==3 )
      {
        size_t indp= ind-ni*ny;
        u[3*ind  ] = u[3*indp];
        u[3*ind+1] = u[3*indp+1];
        u[3*ind+2] = u[3*indp+2];
      }
      else if( s==4 )
      {
        size_t indp= ind+nij*nz;
        u[3*ind  ] = u[3*indp];
        u[3*ind+1] = u[3*indp+1];
        u[3*ind+2] = u[3*indp+2];
      }
      else if( s==5 )
      {
        size_t indp= ind-nij*nz;
        u[3*ind  ] = u[3*indp];
        u[3*ind+1] = u[3*indp+1];
        u[3*ind+2] = u[3*indp+2];
      }
    }
    else if( bccnd[s]==0 )
    {
      if( s != 4 && s != 5)
      {
        printf("EW::bcfortsg_dev, ERROR: Free surface conditio not implemented for side\n");
        return ;
      }

      int i = ib+2 + threadIdx.x + blockIdx.x*blockDim.x;
      int j = jb+2 + threadIdx.y + blockIdx.y*blockDim.y;
      int kk = kb+2 + threadIdx.z + blockIdx.z*blockDim.z;
      if( i > ie-2 || j > je-2 || kk != (kb+2) )
        continue;
      if( s==4 )
      {
        int k=1, kl=1;
        size_t qq = i-ib+ni*(j-jb);
        size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
        float_sw4 wx = strx[i-ib]*(d4a*(u[2+3*ind+3]-u[2+3*ind-3])+d4b*(u[2+3*ind+6]-u[2+3*ind-6]));
        float_sw4 ux = strx[i-ib]*(d4a*(u[  3*ind+3]-u[  3*ind-3])+d4b*(u[  3*ind+6]-u[  3*ind-6]));
        float_sw4 wy = stry[j-jb]*(d4a*(u[2+3*ind+3*ni]-u[2+3*ind-3*ni])+
                                   d4b*(u[2+3*ind+6*ni]-u[2+3*ind-6*ni]));
        float_sw4 vy = stry[j-jb]*(d4a*(u[1+3*ind+3*ni]-u[1+3*ind-3*ni])+
                                   d4b*(u[1+3*ind+6*ni]-u[1+3*ind-6*ni]));
        float_sw4 uz=0., vz=0., wz=0.;
        for( int w=1 ; w <= 4 ; w++ )
        {
          uz += dev_sbop[w]*u[  3*ind+3*nij*kl*(w-1)];
          vz += dev_sbop[w]*u[1+3*ind+3*nij*kl*(w-1)];
          wz += dev_sbop[w]*u[2+3*ind+3*nij*kl*(w-1)];
        }
        u[  3*ind-3*nij*kl] = (-uz-kl*wx+kl*h*bforce5[  3*qq]/mu[ind])/dev_sbop[0];
        u[1+3*ind-3*nij*kl] = (-vz-kl*wy+kl*h*bforce5[1+3*qq]/mu[ind])/dev_sbop[0];
        u[2+3*ind-3*nij*kl] = (-wz + (-kl*la[ind]*(ux+vy)+kl*h*bforce5[2+3*qq])/
                               (2*mu[ind]+la[ind]))/dev_sbop[0];

      }
      else
      {
        int k=nz, kl=-1;
        size_t qq = i-ib+ni*(j-jb);
        size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
        float_sw4 wx = strx[i-ib]*(d4a*(u[2+3*ind+3]-u[2+3*ind-3])+d4b*(u[2+3*ind+6]-u[2+3*ind-6]));
        float_sw4 ux = strx[i-ib]*(d4a*(u[  3*ind+3]-u[  3*ind-3])+d4b*(u[  3*ind+6]-u[  3*ind-6]));
        float_sw4 wy = stry[j-jb]*(d4a*(u[2+3*ind+3*ni]-u[2+3*ind-3*ni])+
                                   d4b*(u[2+3*ind+6*ni]-u[2+3*ind-6*ni]));
        float_sw4 vy = stry[j-jb]*(d4a*(u[1+3*ind+3*ni]-u[1+3*ind-3*ni])+
                                   d4b*(u[1+3*ind+6*ni]-u[1+3*ind-6*ni]));
        float_sw4 uz=0., vz=0., wz=0.;
        for( int w=1 ; w <= 4 ; w++ )
        {
          uz += dev_sbop[w]*u[  3*ind+3*nij*kl*(w-1)];
          vz += dev_sbop[w]*u[1+3*ind+3*nij*kl*(w-1)];
          wz += dev_sbop[w]*u[2+3*ind+3*nij*kl*(w-1)];
        }
        u[  3*ind-3*nij*kl] = (-uz-kl*wx+kl*h*bforce6[  3*qq]/mu[ind])/dev_sbop[0];
        u[1+3*ind-3*nij*kl] = (-vz-kl*wy+kl*h*bforce6[1+3*qq]/mu[ind])/dev_sbop[0];
        u[2+3*ind-3*nij*kl] = (-wz+(-kl*la[ind]*(ux+vy)+kl*h*bforce6[2+3*qq])/
                               (2*mu[ind]+la[ind]))/dev_sbop[0];

      }
    }
  }
}

//-----------------------------------------------------------------------

__global__ void enforceCartTopo_dev_rev( int ifirstCart, int ilastCart, int jfirstCart, int jlastCart, int kfirstCart, int klastCart,
                                         int ifirstCurv, int ilastCurv, int jfirstCurv, int jlastCurv, int kfirstCurv, int klastCurv,
                                         float_sw4* a_uCart, float_sw4* a_uCurv, int ghost_points )
{
#define uCart(c,i,j,k)   a_uCart[base3Cart+(i)+niCart*(j)+nijCart*(k)+nijkCart*(c)]   
#define uCurv(c,i,j,k)   a_uCurv[base3Curv+(i)+niCurv*(j)+nijCurv*(k)+nijkCurv*(c)]   
   const int niCart    = ilastCart-ifirstCart+1;
   const int nijCart   = niCart*(jlastCart-jfirstCart+1);
   const int nijkCart  = nijCart*(klastCart-kfirstCart+1);
   const int baseCart  = -(ifirstCart+niCart*jfirstCart+nijCart*kfirstCart);
   const int base3Cart = baseCart-nijkCart;
   const int niCurv    = ilastCurv-ifirstCurv+1;
   const int nijCurv   = niCurv*(jlastCurv-jfirstCurv+1);
   const int nijkCurv  = nijCurv*(klastCurv-kfirstCurv+1);
   const int baseCurv  = -(ifirstCurv+niCurv*jfirstCurv+nijCurv*kfirstCurv);
   const int base3Curv = baseCurv-nijkCurv;
   const int nc = 3;
   int i = ifirstCart + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirstCart + threadIdx.y + blockIdx.y*blockDim.y;
   int q;  
   if ( (i <= ilastCart) && (j <= jlastCart) )
   {
// assign ghost points in the Cartesian grid
     for (q = 0; q < ghost_points; q++) // only once when m_ghost_points==1
     {
       for( int c = 1; c <= nc ; c++ )
	  uCart(c,i,j,kfirstCart + q) = uCurv(c,i,j,klastCurv-2*ghost_points + q);
     }
// assign ghost points in the Curvilinear grid
     for (q = 0; q <= ghost_points; q++) // twice when m_ghost_points==1 (overwrites solution on the common grid line)
     {
         for( int c = 1; c <= nc ; c++ )
           uCurv(c,i,j,klastCurv-q) = uCart(c,i,j,kfirstCart+2*ghost_points - q);
     }
   }
#undef uCart
#undef uCurv
}

//-----------------------------------------------------------------------

__global__ void enforceCartTopo_dev( int ifirstCart, int ilastCart, int jfirstCart, int jlastCart, int kfirstCart, int klastCart,
                                         int ifirstCurv, int ilastCurv, int jfirstCurv, int jlastCurv, int kfirstCurv, int klastCurv,
                                         float_sw4* a_uCart, float_sw4* a_uCurv, int ghost_points )
{
#define uCart(c,i,j,k)   a_uCart[base3Cart+(c)+3*(i)+ni3Cart*(j)+nij3Cart*(k)]   
#define uCurv(c,i,j,k)   a_uCurv[base3Curv+(c)+3*(i)+ni3Curv*(j)+nij3Curv*(k)]   
   const int niCart    = ilastCart-ifirstCart+1;
   const int nijCart   = niCart*(jlastCart-jfirstCart+1);
   const int ni3Cart  = 3*niCart;
   const int nij3Cart = 3*nijCart;
   const int baseCart  = -(ifirstCart+niCart*jfirstCart+nijCart*kfirstCart);
   const int base3Cart = 3*baseCart-1;
   const int niCurv    = ilastCurv-ifirstCurv+1;
   const int nijCurv   = niCurv*(jlastCurv-jfirstCurv+1);
   const int ni3Curv  = 3*niCurv;
   const int nij3Curv = 3*nijCurv;
   const int baseCurv  = -(ifirstCurv+niCurv*jfirstCurv+nijCurv*kfirstCurv);
   const int base3Curv = 3*baseCurv-1;
   const int nc = 3;
   int i = ifirstCart + threadIdx.x + blockIdx.x*blockDim.x;
   int j = jfirstCart + threadIdx.y + blockIdx.y*blockDim.y;
   int q;  
   if ( (i <= ilastCart) && (j <= jlastCart) )
   {
// assign ghost points in the Cartesian grid
     for (q = 0; q < ghost_points; q++) // only once when m_ghost_points==1
     {
       for( int c = 1; c <= nc ; c++ )
	  uCart(c,i,j,kfirstCart + q) = uCurv(c,i,j,klastCurv-2*ghost_points + q);
     }
// assign ghost points in the Curvilinear grid
     for (q = 0; q <= ghost_points; q++) // twice when m_ghost_points==1 (overwrites solution on the common grid line)
     {
         for( int c = 1; c <= nc ; c++ )
           uCurv(c,i,j,klastCurv-q) = uCart(c,i,j,kfirstCart+2*ghost_points - q);
     }
   }
#undef uCart
#undef uCurv
}
// *****************************************************************************
// *****************************************************************************
// *****************************************************************************
// New GPU kernels from G. Thomas-Collignon


// Macros to access the triple-arrays in C order (structure of arrays) or not (array of structure)
// This relies on nijk (cube size) to be defined.
#define u(c,idx) (c_order ? a_u[idx+(c)*nijk] : a_u[3*(idx)+c])
#define lu(c,idx) (c_order ? a_lu[idx+(c)*nijk] : a_lu[3*(idx)+c])
#define up(c,idx) (c_order ? a_up[idx+(c)*nijk] : a_up[3*(idx)+c])
#define um(c,idx) (c_order ? a_um[idx+(c)*nijk] : a_um[3*(idx)+c])
#define fo(c,idx) (c_order ? a_fo[idx+(c)*nijk] : a_fo[3*(idx)+c])

// Block size for most kernels.
// Note : BX should be defined in sw4.h, based on the type
#ifndef BX
#define BX 16
#endif
#define BY 16

// Default block size for the upper kernel
#define UX 16

// *****************************************************************************
// *****************************************************************************
// *****************************************************************************
// RHS4 :
// -----
// Launch enough threads to cover the full NI dimension
// This will allow aligned memory accesses if NI is a multiple of the cache line size for
// all the arrays that are don't have halos (the majority of memory accesses).
// In the J dimension, launch enough threads to cover [jfirst:jlast].
//
// We load U in shared memory, and keep most of lambda / mu in registers.
// The halos for lambda and mu are loaded directly (L1/L2 caches should help).
//
// Due to the very low occupancy, the code is very sensitive to latency.
// We try to hide the memory latency even more by prefetching data for the next iteration
// or trigger loads as early as possible (at the expense of register usage).
//
// [ifirst:ilast][jfirst:jlast][kfirst:klast] defines the volume computed.
// There's a 2-point halo being read on top of that.
// ni, nj, nk are the volume dimensions.
//
// The RHS kernel has been fused with the predictor / corrector kernels.

template <bool c_order, bool pred>
  __launch_bounds__(BX*BY)
  __global__ void rhs4 (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                        int ni, int nj, int nk,
                        float_sw4* __restrict__ a_up,
                        const float_sw4* __restrict__ a_u,
                        const float_sw4* __restrict__ a_um,
                        const float_sw4* __restrict__ a_mu,
                        const float_sw4* __restrict__ a_lambda,
                        const float_sw4* __restrict__ a_rho,
                        const float_sw4* __restrict__ a_fo, 
                        const float_sw4* __restrict__ a_strx,
                        const float_sw4* __restrict__ a_stry,
                        const float_sw4* __restrict__ a_strz,
                        const float_sw4 h,
                        const float_sw4 dt) {
  int index;

  const float_sw4 i6   = 1.0/6;
  const float_sw4 i144 = 1.0/144;
  const float_sw4 tf   = 0.75;
  
  float_sw4 strx_im2, strx_im1, strx_i, strx_ip1, strx_ip2;
  float_sw4 stry_jm2, stry_jm1, stry_j, stry_jp1, stry_jp2;
  float_sw4 strz_km2, strz_km1, strz_k, strz_kp1, strz_kp2;

  float_sw4 la_km2, la_km1, la_ijk, la_kp1, la_kp2;
  float_sw4 la_im2, la_im1, la_ip1, la_ip2;
  float_sw4 la_jm2, la_jm1, la_jp1, la_jp2;

  float_sw4 mu_km2, mu_km1, mu_ijk, mu_kp1, mu_kp2;
  float_sw4 mu_im2, mu_im1, mu_ip1, mu_ip2;
  float_sw4 mu_jm2, mu_jm1, mu_jp1, mu_jp2;

  float_sw4 fo0, fo1, fo2, rho;
  float_sw4 up0, up1, up2;
  float_sw4 um0, um1, um2;

  // Prefetch variables
  float_sw4 a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2;
  
  float_sw4 r1, r2, r3, cof;
  int active=0, loader=0, loady2=0, loadx2=0;

  __shared__ float_sw4 shu[3][5][BY+4][BX+4];

  // i starts at 0, while j starts at jfirst
  const int i = threadIdx.x + blockIdx.x * BX;
  const int j = jfirst + threadIdx.y + blockIdx.y * BY;

  const int ti = threadIdx.x + 2;
  const int tj = threadIdx.y + 2;
  const int tk = 2;

  const int nij = ni * nj;
  const int nijk = nij * nk;

  // Active threads doing the computation at (i,j,k)
  if (i >= ifirst && i <= ilast && j <= jlast)
    active = 1;

  // Threads loading data with halos (4 regions)
  if (j-2 <= jlast + 2) {
    if (i-2 >= 0 && i-2 < ni)
      loader = 1;
    if (threadIdx.x < 4 && i+BX-2 < ni)
      loadx2 = 1;
    if (threadIdx.y < 4 && j+BY-2 <= jlast+2)
      loady2 = 1;
  }

  // Loading index for data with halos, starting at (i-2,j-2,kfirst-2)
  index = (kfirst - 2) * nij + (j - 2) * ni + i - 2;

  // Load the first 4 plans of U in the shared memory
  // and prefetch the 5th plan in a,b,c,d
  if (loader) {
    int idx = index;
    shu[0][1][threadIdx.y][threadIdx.x] = u(0,idx);
    shu[1][1][threadIdx.y][threadIdx.x] = u(1,idx);
    shu[2][1][threadIdx.y][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][2][threadIdx.y][threadIdx.x] = u(0,idx);
    shu[1][2][threadIdx.y][threadIdx.x] = u(1,idx);
    shu[2][2][threadIdx.y][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][3][threadIdx.y][threadIdx.x] = u(0,idx);
    shu[1][3][threadIdx.y][threadIdx.x] = u(1,idx);
    shu[2][3][threadIdx.y][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][4][threadIdx.y][threadIdx.x] = u(0,idx);
    shu[1][4][threadIdx.y][threadIdx.x] = u(1,idx);
    shu[2][4][threadIdx.y][threadIdx.x] = u(2,idx);
    idx += nij;
    a0 = u(0,idx);
    a1 = u(1,idx);
    a2 = u(2,idx);
  }
  if (loadx2) {
    int idx = index + BX;
    shu[0][1][threadIdx.y][threadIdx.x+BX] = u(0,idx);
    shu[1][1][threadIdx.y][threadIdx.x+BX] = u(1,idx);
    shu[2][1][threadIdx.y][threadIdx.x+BX] = u(2,idx);
    idx += nij;		         
    shu[0][2][threadIdx.y][threadIdx.x+BX] = u(0,idx);
    shu[1][2][threadIdx.y][threadIdx.x+BX] = u(1,idx);
    shu[2][2][threadIdx.y][threadIdx.x+BX] = u(2,idx);
    idx += nij;		         
    shu[0][3][threadIdx.y][threadIdx.x+BX] = u(0,idx);
    shu[1][3][threadIdx.y][threadIdx.x+BX] = u(1,idx);
    shu[2][3][threadIdx.y][threadIdx.x+BX] = u(2,idx);
    idx += nij;		         
    shu[0][4][threadIdx.y][threadIdx.x+BX] = u(0,idx);
    shu[1][4][threadIdx.y][threadIdx.x+BX] = u(1,idx);
    shu[2][4][threadIdx.y][threadIdx.x+BX] = u(2,idx);
    idx += nij;
    b0 = u(0,idx);
    b1 = u(1,idx);
    b2 = u(2,idx);
  }
  if (loader && loady2) {
    int idx = index + BY * ni;
    shu[0][1][threadIdx.y+BY][threadIdx.x] = u(0,idx);
    shu[1][1][threadIdx.y+BY][threadIdx.x] = u(1,idx);
    shu[2][1][threadIdx.y+BY][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][2][threadIdx.y+BY][threadIdx.x] = u(0,idx);
    shu[1][2][threadIdx.y+BY][threadIdx.x] = u(1,idx);
    shu[2][2][threadIdx.y+BY][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][3][threadIdx.y+BY][threadIdx.x] = u(0,idx);
    shu[1][3][threadIdx.y+BY][threadIdx.x] = u(1,idx);
    shu[2][3][threadIdx.y+BY][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][4][threadIdx.y+BY][threadIdx.x] = u(0,idx);
    shu[1][4][threadIdx.y+BY][threadIdx.x] = u(1,idx);
    shu[2][4][threadIdx.y+BY][threadIdx.x] = u(2,idx);
    idx += nij;
    c0 = u(0,idx);
    c1 = u(1,idx);
    c2 = u(2,idx);
  }
  if (loadx2 && loady2) {
    int idx = index + BY * ni + BX;
    shu[0][1][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
    shu[1][1][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
    shu[2][1][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
    idx += nij;
    shu[0][2][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
    shu[1][2][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
    shu[2][2][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
    idx += nij;
    shu[0][3][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
    shu[1][3][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
    shu[2][3][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
    idx += nij;
    shu[0][4][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
    shu[1][4][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
    shu[2][4][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
    idx += nij;
    d0 = u(0,idx);
    d1 = u(1,idx);
    d2 = u(2,idx);
  }

  if (active) {
    // Load strx & stry, centered at i and j
    strx_im2 = a_strx[i-2];
    strx_im1 = a_strx[i-1];
    strx_i   = a_strx[i];
    strx_ip1 = a_strx[i+1];
    strx_ip2 = a_strx[i+2];
    
    stry_jm2 = a_stry[j-2];
    stry_jm1 = a_stry[j-1];
    stry_j   = a_stry[j];
    stry_jp1 = a_stry[j+1];
    stry_jp2 = a_stry[j+2];

    // Load first 4 lambda and mu values 
    int idx = index + 2 * ni + 2;
    la_km2 = a_lambda[idx];
    mu_km2 = a_mu[idx];
    idx += nij;
    la_km1 = a_lambda[idx];
    mu_km1 = a_mu[idx];
    idx += nij;
    la_ijk = a_lambda[idx];
    mu_ijk = a_mu[idx];
    idx += nij;
    la_kp1 = a_lambda[idx];
    mu_kp1 = a_mu[idx];
  }

  // Move the loading index at kfirst + 2
  index += 4 * nij;

  // Load the first 4 values of strz
  strz_km2 = a_strz[kfirst-2];
  strz_km1 = a_strz[kfirst-1];
  strz_k   = a_strz[kfirst];
  strz_kp1 = a_strz[kfirst+1];

  cof = 1.0/(h*h);

  // Main loop on K dimension
  for (int k = kfirst; k <= klast; k++) {

    // Rotate the shared memory and store the prefetched variables
    __syncthreads();
    for (int kk=0; kk<4; kk++) {
      shu[0][kk][threadIdx.y][threadIdx.x] = shu[0][kk+1][threadIdx.y][threadIdx.x];
      shu[1][kk][threadIdx.y][threadIdx.x] = shu[1][kk+1][threadIdx.y][threadIdx.x];
      shu[2][kk][threadIdx.y][threadIdx.x] = shu[2][kk+1][threadIdx.y][threadIdx.x];
    }    
    shu[0][4][threadIdx.y][threadIdx.x] = a0;
    shu[1][4][threadIdx.y][threadIdx.x] = a1;
    shu[2][4][threadIdx.y][threadIdx.x] = a2;
    if (threadIdx.x < 4) {
      for (int kk=0; kk<4; kk++) {
	shu[0][kk][threadIdx.y][threadIdx.x+BX] = shu[0][kk+1][threadIdx.y][threadIdx.x+BX];
	shu[1][kk][threadIdx.y][threadIdx.x+BX] = shu[1][kk+1][threadIdx.y][threadIdx.x+BX];
	shu[2][kk][threadIdx.y][threadIdx.x+BX] = shu[2][kk+1][threadIdx.y][threadIdx.x+BX];
      }    
      shu[0][4][threadIdx.y][threadIdx.x+BX] = b0;
      shu[1][4][threadIdx.y][threadIdx.x+BX] = b1;
      shu[2][4][threadIdx.y][threadIdx.x+BX] = b2;
    }
    if (threadIdx.y < 4) {
      for (int kk=0; kk<4; kk++) {
	shu[0][kk][threadIdx.y+BY][threadIdx.x] = shu[0][kk+1][threadIdx.y+BY][threadIdx.x];
	shu[1][kk][threadIdx.y+BY][threadIdx.x] = shu[1][kk+1][threadIdx.y+BY][threadIdx.x];
	shu[2][kk][threadIdx.y+BY][threadIdx.x] = shu[2][kk+1][threadIdx.y+BY][threadIdx.x];
      }    
      shu[0][4][threadIdx.y+BY][threadIdx.x] = c0;
      shu[1][4][threadIdx.y+BY][threadIdx.x] = c1;
      shu[2][4][threadIdx.y+BY][threadIdx.x] = c2;
      if (threadIdx.x < 4) {
	for (int kk=0; kk<4; kk++) {
	  shu[0][kk][threadIdx.y+BY][threadIdx.x+BX] = shu[0][kk+1][threadIdx.y+BY][threadIdx.x+BX];
	  shu[1][kk][threadIdx.y+BY][threadIdx.x+BX] = shu[1][kk+1][threadIdx.y+BY][threadIdx.x+BX];
	  shu[2][kk][threadIdx.y+BY][threadIdx.x+BX] = shu[2][kk+1][threadIdx.y+BY][threadIdx.x+BX];
	}
	shu[0][4][threadIdx.y+BY][threadIdx.x+BX] = d0;
	shu[1][4][threadIdx.y+BY][threadIdx.x+BX] = d1;
	shu[2][4][threadIdx.y+BY][threadIdx.x+BX] = d2;
      }
    }
    __syncthreads();
    
    // Prefetch the next iteration variables for U to better hide the latency
    if (k < klast) {
      int idx = index + nij;
      if (loader) {
	a0 = u(0,idx);
	a1 = u(1,idx);
	a2 = u(2,idx);
      }
      idx += BX;
      if (loadx2) {
	b0 = u(0,idx);
	b1 = u(1,idx);
	b2 = u(2,idx);
      }
      idx = index + nij + BY * ni;
      if (loader & loady2) {
	c0 = u(0,idx);
	c1 = u(1,idx);
	c2 = u(2,idx);
      }
      idx += BX;
      if (loadx2 && loady2) {
	d0 = u(0,idx);
	d1 = u(1,idx);
	d2 = u(2,idx);
      }
    }

    // Active threads load center values and do the computation
    if (active) {
      // Load a new value of strz
      strz_kp2 = a_strz[k+2];

      // Load new values of lambda and mu, and the (i,j) halos
      {
	int idx = index + 2 * ni + 2;
	la_kp2 = a_lambda[idx];
	mu_kp2 = a_mu[idx];
	// Halos at k
	idx -= 2 * nij;
	la_im2 = a_lambda[idx-2];
	la_im1 = a_lambda[idx-1];
	la_ip1 = a_lambda[idx+1];
	la_ip2 = a_lambda[idx+2];
	la_jm2 = a_lambda[idx-2*ni];
	la_jm1 = a_lambda[idx-1*ni];
	la_jp1 = a_lambda[idx+ni];
	la_jp2 = a_lambda[idx+2*ni];
	mu_im2 = a_mu[idx-2];
	mu_im1 = a_mu[idx-1];
	mu_ip1 = a_mu[idx+1];
	mu_ip2 = a_mu[idx+2];
	mu_jm2 = a_mu[idx-2*ni];
	mu_jm1 = a_mu[idx-ni];
	mu_jp1 = a_mu[idx+ni];
	mu_jp2 = a_mu[idx+2*ni];
	// Read rho, fo, um, up in advance
	fo0 = fo(0,idx);
	fo1 = fo(1,idx);
	fo2 = fo(2,idx);
	rho = a_rho[idx];
	if (pred) {
	  um0 = um(0,idx);
	  um1 = um(1,idx);
	  um2 = um(2,idx);
	}
	else {
	  up0 = up(0,idx);
	  up1 = up(1,idx);
	  up2 = up(2,idx);
	}
      }

      // X contribution
      {
	float_sw4 mux1, mux2, mux3, mux4;
	mux1 = mu_im1 * strx_im1 - tf * (mu_ijk * strx_i + mu_im2 * strx_im2);
	mux2 = mu_im2 * strx_im2 + mu_ip1 * strx_ip1 + 3 * (mu_ijk * strx_i   + mu_im1 * strx_im1);
	mux3 = mu_im1 * strx_im1 + mu_ip2 * strx_ip2 + 3 * (mu_ip1 * strx_ip1 + mu_ijk * strx_i  );
	mux4 = mu_ip1 * strx_ip1 - tf * (mu_ijk * strx_i + mu_ip2 * strx_ip2);

	r1 = strx_i * ((2 * mux1 + la_im1 * strx_im1 - tf * (la_ijk * strx_i   + la_im2 * strx_im2)) *
		       (shu[0][tk+0][tj+0][ti-2] - shu[0][tk+0][tj+0][ti+0]) +
		       (2 * mux2 + la_im2 * strx_im2 + la_ip1 * strx_ip1 +
			3 * (la_ijk * strx_i   + la_im1 * strx_im1)) *
		       (shu[0][tk+0][tj+0][ti-1] - shu[0][tk+0][tj+0][ti+0]) +
		       (2 * mux3 + la_im1 * strx_im1 + la_ip2 * strx_ip2 +
			3 * (la_ip1 * strx_ip1 + la_ijk * strx_i  )) *
		       (shu[0][tk+0][tj+0][ti+1] - shu[0][tk+0][tj+0][ti+0]) +
		       (2 * mux4 + la_ip1 * strx_ip1 - tf * (la_ijk * strx_i   + la_ip2 * strx_ip2)) *
		       (shu[0][tk+0][tj+0][ti+2] - shu[0][tk+0][tj+0][ti+0]));
	r2 = strx_i * (mux1 * (shu[1][tk+0][tj+0][ti-2] - shu[1][tk+0][tj+0][ti+0]) +
		       mux2 * (shu[1][tk+0][tj+0][ti-1] - shu[1][tk+0][tj+0][ti+0]) +
		       mux3 * (shu[1][tk+0][tj+0][ti+1] - shu[1][tk+0][tj+0][ti+0]) +
		       mux4 * (shu[1][tk+0][tj+0][ti+2] - shu[1][tk+0][tj+0][ti+0]));
	r3 = strx_i * (mux1 * (shu[2][tk+0][tj+0][ti-2] - shu[2][tk+0][tj+0][ti+0]) +
		       mux2 * (shu[2][tk+0][tj+0][ti-1] - shu[2][tk+0][tj+0][ti+0]) +
		       mux3 * (shu[2][tk+0][tj+0][ti+1] - shu[2][tk+0][tj+0][ti+0]) +
		       mux4 * (shu[2][tk+0][tj+0][ti+2] - shu[2][tk+0][tj+0][ti+0]));
      }
      // Y contribution
      {
	float_sw4 muy1, muy2, muy3, muy4;
	muy1 = mu_jm1 * stry_jm1 - tf * (mu_ijk * stry_j + mu_jm2 * stry_jm2);
	muy2 = mu_jm2 * stry_jm2 + mu_jp1 * stry_jp1 + 3 * (mu_ijk * stry_j   + mu_jm1 * stry_jm1);
	muy3 = mu_jm1 * stry_jm1 + mu_jp2 * stry_jp2 + 3 * (mu_jp1 * stry_jp1 + mu_ijk * stry_j  );
	muy4 = mu_jp1 * stry_jp1 - tf * (mu_ijk * stry_j + mu_jp2 * stry_jp2);

	r1 += stry_j * (muy1 * (shu[0][tk+0][tj-2][ti+0] - shu[0][tk+0][tj+0][ti+0]) +
			muy2 * (shu[0][tk+0][tj-1][ti+0] - shu[0][tk+0][tj+0][ti+0]) +
			muy3 * (shu[0][tk+0][tj+1][ti+0] - shu[0][tk+0][tj+0][ti+0]) +
			muy4 * (shu[0][tk+0][tj+2][ti+0] - shu[0][tk+0][tj+0][ti+0]));
	r2 += stry_j * ((2 * muy1 + la_jm1 * stry_jm1 - tf * (la_ijk * stry_j + la_jm2 * stry_jm2)) *
			(shu[1][tk+0][tj-2][ti+0] - shu[1][tk+0][tj+0][ti+0]) +
			(2 * muy2 + la_jm2 * stry_jm2 + la_jp1 * stry_jp1 +
			 3 * (la_ijk * stry_j   + la_jm1 * stry_jm1)) *
			(shu[1][tk+0][tj-1][ti+0] - shu[1][tk+0][tj+0][ti+0]) +
			(2 * muy3 + la_jm1 * stry_jm1 + la_jp2 * stry_jp2 +
			 3 * (la_jp1 * stry_jp1 + la_ijk * stry_j  )) *
			(shu[1][tk+0][tj+1][ti+0] - shu[1][tk+0][tj+0][ti+0]) +
			(2 * muy4 + la_jp1 * stry_jp1 - tf * (la_ijk * stry_j  + la_jp2 * stry_jp2)) *
			(shu[1][tk+0][tj+2][ti+0] - shu[1][tk+0][tj+0][ti+0]));
	r3 += stry_j  *(muy1 * (shu[2][tk+0][tj-2][ti+0] - shu[2][tk+0][tj+0][ti+0]) +
			muy2 * (shu[2][tk+0][tj-1][ti+0] - shu[2][tk+0][tj+0][ti+0]) +
			muy3 * (shu[2][tk+0][tj+1][ti+0] - shu[2][tk+0][tj+0][ti+0]) +
			muy4 * (shu[2][tk+0][tj+2][ti+0] - shu[2][tk+0][tj+0][ti+0]));
      }
      // Z contribution
      {
	float_sw4 muz1, muz2, muz3, muz4;
	muz1 = mu_km1 * strz_km1 - tf * (mu_ijk * strz_k + mu_km2 * strz_km2);
	muz2 = mu_km2 * strz_km2 + mu_kp1 * strz_kp1 + 3 * (mu_ijk * strz_k   + mu_km1 * strz_km1);
	muz3 = mu_km1 * strz_km1 + mu_kp2 * strz_kp2 + 3 * (mu_kp1 * strz_kp1 + mu_ijk * strz_k);
	muz4 = mu_kp1 * strz_kp1 - tf * (mu_ijk * strz_k + mu_kp2 * strz_kp2);
	r1 += strz_k * (muz1 * (shu[0][tk-2][tj+0][ti+0] - shu[0][tk+0][tj+0][ti+0]) +
			muz2 * (shu[0][tk-1][tj+0][ti+0] - shu[0][tk+0][tj+0][ti+0]) +
			muz3 * (shu[0][tk+1][tj+0][ti+0] - shu[0][tk+0][tj+0][ti+0]) +
			muz4 * (shu[0][tk+2][tj+0][ti+0] - shu[0][tk+0][tj+0][ti+0]));
	r2 += strz_k * (muz1 * (shu[1][tk-2][tj+0][ti+0] - shu[1][tk+0][tj+0][ti+0]) +
			muz2 * (shu[1][tk-1][tj+0][ti+0] - shu[1][tk+0][tj+0][ti+0]) +
			muz3 * (shu[1][tk+1][tj+0][ti+0] - shu[1][tk+0][tj+0][ti+0]) +
			muz4 * (shu[1][tk+2][tj+0][ti+0] - shu[1][tk+0][tj+0][ti+0]));
	r3 += strz_k * ((2 * muz1 + la_km1 * strz_km1 - tf * (la_ijk * strz_k + la_km2 * strz_km2)) *
			(shu[2][tk-2][tj+0][ti+0] - shu[2][tk+0][tj+0][ti+0]) +
			(2 * muz2 + la_km2 * strz_km2 + la_kp1 * strz_kp1 +
			 3 * (la_ijk * strz_k   + la_km1 * strz_km1)) *
			(shu[2][tk-1][tj+0][ti+0] - shu[2][tk+0][tj+0][ti+0]) +
			(2 * muz3 + la_km1 * strz_km1 + la_kp2 * strz_kp2 +
			 3 * (la_kp1 * strz_kp1 + la_ijk * strz_k  )) *
			(shu[2][tk+1][tj+0][ti+0] - shu[2][tk+0][tj+0][ti+0]) +
			(2 * muz4 + la_kp1 * strz_kp1 - tf * (la_ijk * strz_k + la_kp2 * strz_kp2)) *
			(shu[2][tk+2][tj+0][ti+0] - shu[2][tk+0][tj+0][ti+0]));
      }
      r1 *= i6;
      r2 *= i6;
      r3 *= i6;


      /* Mixed derivatives: */
      /*   (la*v_y)_x */
      r1 += strx_i  * stry_j * i144 * 
	(la_im2 * (shu[1][tk+0][tj-2][ti-2] - shu[1][tk+0][tj+2][ti-2] + 8 *
		   (-shu[1][tk+0][tj-1][ti-2] + shu[1][tk+0][tj+1][ti-2])) - 8 *
	 (la_im1 * (shu[1][tk+0][tj-2][ti-1] - shu[1][tk+0][tj+2][ti-1] + 8 *
		    (-shu[1][tk+0][tj-1][ti-1] + shu[1][tk+0][tj+1][ti-1]))) + 8 *
	 (la_ip1 * (shu[1][tk+0][tj-2][ti+1] - shu[1][tk+0][tj+2][ti+1] + 8 *
		    (-shu[1][tk+0][tj-1][ti+1] + shu[1][tk+0][tj+1][ti+1]))) - 
	 (la_ip2 * (shu[1][tk+0][tj-2][ti+2] - shu[1][tk+0][tj+2][ti+2] + 8 *
		    (-shu[1][tk+0][tj-1][ti+2] + shu[1][tk+0][tj+1][ti+2]))))
	/*   (la*w_z)_x */
        + strx_i * strz_k * i144 *
	(la_im2 * (shu[2][tk-2][tj+0][ti-2] - shu[2][tk+2][tj+0][ti-2] + 8 *
		   (-shu[2][tk-1][tj+0][ti-2] + shu[2][tk+1][tj+0][ti-2])) - 8 *
	 (la_im1 * (shu[2][tk-2][tj+0][ti-1] - shu[2][tk+2][tj+0][ti-1] + 8 *
		    (-shu[2][tk-1][tj+0][ti-1] + shu[2][tk+1][tj+0][ti-1]))) + 8 *
	 (la_ip1 * (shu[2][tk-2][tj+0][ti+1] - shu[2][tk+2][tj+0][ti+1] + 8 *
		    (-shu[2][tk-1][tj+0][ti+1] + shu[2][tk+1][tj+0][ti+1]))) - 
	 (la_ip2 * (shu[2][tk-2][tj+0][ti+2] - shu[2][tk+2][tj+0][ti+2] + 8 *
		    (-shu[2][tk-1][tj+0][ti+2] + shu[2][tk+1][tj+0][ti+2]))))
	/*   (mu*v_x)_y */
        + strx_i * stry_j * i144 *
	(mu_jm2 * (shu[1][tk+0][tj-2][ti-2] - shu[1][tk+0][tj-2][ti+2] + 8 *
		   (-shu[1][tk+0][tj-2][ti-1] + shu[1][tk+0][tj-2][ti+1])) - 8 *
	 (mu_jm1 * (shu[1][tk+0][tj-1][ti-2] - shu[1][tk+0][tj-1][ti+2] + 8 *
		    (-shu[1][tk+0][tj-1][ti-1] + shu[1][tk+0][tj-1][ti+1]))) + 8 *
	 (mu_jp1 * (shu[1][tk+0][tj+1][ti-2] - shu[1][tk+0][tj+1][ti+2] + 8 *
		    (-shu[1][tk+0][tj+1][ti-1] + shu[1][tk+0][tj+1][ti+1]))) - 
	 (mu_jp2 * (shu[1][tk+0][tj+2][ti-2] - shu[1][tk+0][tj+2][ti+2] + 8 *
		    (-shu[1][tk+0][tj+2][ti-1] + shu[1][tk+0][tj+2][ti+1]))))
	/*   (mu*w_x)_z */
        + strx_i * strz_k * i144 *
	(mu_km2 * (shu[2][tk-2][tj+0][ti-2] - shu[2][tk-2][tj+0][ti+2] + 8 *
		   (-shu[2][tk-2][tj+0][ti-1] + shu[2][tk-2][tj+0][ti+1])) - 8 *
 	 (mu_km1 * (shu[2][tk-1][tj+0][ti-2] - shu[2][tk-1][tj+0][ti+2] + 8 *
		    (-shu[2][tk-1][tj+0][ti-1] + shu[2][tk-1][tj+0][ti+1]))) + 8 *
 	 (mu_kp1 * (shu[2][tk+1][tj+0][ti-2] - shu[2][tk+1][tj+0][ti+2] + 8 *
		    (-shu[2][tk+1][tj+0][ti-1] + shu[2][tk+1][tj+0][ti+1]))) -
	 (mu_kp2 * (shu[2][tk+2][tj+0][ti-2] - shu[2][tk+2][tj+0][ti+2] + 8 *
		    (-shu[2][tk+2][tj+0][ti-1] + shu[2][tk+2][tj+0][ti+1]))));

      /*   (mu*u_y)_x */
      r2 += strx_i  *stry_j  *i144*
	(mu_im2 * (shu[0][tk+0][tj-2][ti-2] - shu[0][tk+0][tj+2][ti-2] + 8 *
		   (-shu[0][tk+0][tj-1][ti-2] + shu[0][tk+0][tj+1][ti-2])) - 8 *
	 (mu_im1 * (shu[0][tk+0][tj-2][ti-1] - shu[0][tk+0][tj+2][ti-1] + 8 *
		    (-shu[0][tk+0][tj-1][ti-1] + shu[0][tk+0][tj+1][ti-1]))) + 8 *
	 (mu_ip1 * (shu[0][tk+0][tj-2][ti+1] - shu[0][tk+0][tj+2][ti+1] + 8 *
		    (-shu[0][tk+0][tj-1][ti+1] + shu[0][tk+0][tj+1][ti+1]))) -
	 (mu_ip2 * (shu[0][tk+0][tj-2][ti+2] - shu[0][tk+0][tj+2][ti+2] + 8 *
		    (-shu[0][tk+0][tj-1][ti+2] + shu[0][tk+0][tj+1][ti+2]))))
	/* (la*u_x)_y */
        + strx_i * stry_j * i144 *
	(la_jm2 * (shu[0][tk+0][tj-2][ti-2] - shu[0][tk+0][tj-2][ti+2] + 8 *
		   (-shu[0][tk+0][tj-2][ti-1] + shu[0][tk+0][tj-2][ti+1])) - 8 *
	 (la_jm1 * (shu[0][tk+0][tj-1][ti-2] - shu[0][tk+0][tj-1][ti+2] + 8 *
		    (-shu[0][tk+0][tj-1][ti-1] + shu[0][tk+0][tj-1][ti+1]))) + 8 *
	 (la_jp1 * (shu[0][tk+0][tj+1][ti-2] - shu[0][tk+0][tj+1][ti+2] + 8 *
		    (-shu[0][tk+0][tj+1][ti-1] + shu[0][tk+0][tj+1][ti+1]))) -
	 (la_jp2 * (shu[0][tk+0][tj+2][ti-2] - shu[0][tk+0][tj+2][ti+2] + 8 *
		    (-shu[0][tk+0][tj+2][ti-1] + shu[0][tk+0][tj+2][ti+1]))))
	/* (la*w_z)_y */
        + stry_j * strz_k * i144 *
	(la_jm2 * (shu[2][tk-2][tj-2][ti+0] - shu[2][tk+2][tj-2][ti+0] + 8 *
		   (-shu[2][tk-1][tj-2][ti+0] + shu[2][tk+1][tj-2][ti+0])) - 8 *
	 (la_jm1 * (shu[2][tk-2][tj-1][ti+0] - shu[2][tk+2][tj-1][ti+0] + 8 *
		    (-shu[2][tk-1][tj-1][ti+0] + shu[2][tk+1][tj-1][ti+0]))) + 8 *
	 (la_jp1 * (shu[2][tk-2][tj+1][ti+0] - shu[2][tk+2][tj+1][ti+0] + 8 *
		    (-shu[2][tk-1][tj+1][ti+0] + shu[2][tk+1][tj+1][ti+0]))) -
	 (la_jp2 * (shu[2][tk-2][tj+2][ti+0] - shu[2][tk+2][tj+2][ti+0] + 8 *
		    (-shu[2][tk-1][tj+2][ti+0] + shu[2][tk+1][tj+2][ti+0]))))
	/* (mu*w_y)_z */
        + stry_j * strz_k * i144 *
	(mu_km2 * (shu[2][tk-2][tj-2][ti+0] - shu[2][tk-2][tj+2][ti+0] + 8 *
		   (-shu[2][tk-2][tj-1][ti+0] + shu[2][tk-2][tj+1][ti+0])) - 8 *
	 (mu_km1 * (shu[2][tk-1][tj-2][ti+0] - shu[2][tk-1][tj+2][ti+0] + 8 *
		    (-shu[2][tk-1][tj-1][ti+0] + shu[2][tk-1][tj+1][ti+0]))) + 8 *
	 (mu_kp1 * (shu[2][tk+1][tj-2][ti+0] - shu[2][tk+1][tj+2][ti+0] + 8 *
		    (-shu[2][tk+1][tj-1][ti+0] + shu[2][tk+1][tj+1][ti+0]))) -
	 (mu_kp2 * (shu[2][tk+2][tj-2][ti+0] - shu[2][tk+2][tj+2][ti+0] + 8 *
		    (-shu[2][tk+2][tj-1][ti+0] + shu[2][tk+2][tj+1][ti+0]))));

      /*  (mu*u_z)_x */
      r3 += strx_i * strz_k * i144 *
	(mu_im2 * (shu[0][tk-2][tj+0][ti-2] - shu[0][tk+2][tj+0][ti-2] + 8 *
		   (-shu[0][tk-1][tj+0][ti-2] + shu[0][tk+1][tj+0][ti-2])) - 8 *
	 (mu_im1 * (shu[0][tk-2][tj+0][ti-1] - shu[0][tk+2][tj+0][ti-1] + 8 *
		    (-shu[0][tk-1][tj+0][ti-1] + shu[0][tk+1][tj+0][ti-1]))) + 8 *
	 (mu_ip1 * (shu[0][tk-2][tj+0][ti+1] - shu[0][tk+2][tj+0][ti+1] + 8 *
		    (-shu[0][tk-1][tj+0][ti+1] + shu[0][tk+1][tj+0][ti+1]))) -
	 (mu_ip2 * (shu[0][tk-2][tj+0][ti+2] - shu[0][tk+2][tj+0][ti+2] + 8 *
		    (-shu[0][tk-1][tj+0][ti+2] + shu[0][tk+1][tj+0][ti+2]))))
	/* (mu*v_z)_y */
        + stry_j  *strz_k*i144*
	(mu_jm2 * (shu[1][tk-2][tj-2][ti+0] - shu[1][tk+2][tj-2][ti+0] + 8 *
		   (- shu[1][tk-1][tj-2][ti+0]+shu[1][tk+1][tj-2][ti+0])) - 8 * 
	 (mu_jm1 * (shu[1][tk-2][tj-1][ti+0] - shu[1][tk+2][tj-1][ti+0] + 8 *
		    (- shu[1][tk-1][tj-1][ti+0]+shu[1][tk+1][tj-1][ti+0]))) + 8 * 
	 (mu_jp1 * (shu[1][tk-2][tj+1][ti+0] - shu[1][tk+2][tj+1][ti+0] + 8 *
		    (- shu[1][tk-1][tj+1][ti+0]+shu[1][tk+1][tj+1][ti+0]))) -
	 (mu_jp2 * (shu[1][tk-2][tj+2][ti+0] - shu[1][tk+2][tj+2][ti+0] + 8 *
		    (- shu[1][tk-1][tj+2][ti+0]+shu[1][tk+1][tj+2][ti+0]))))
	/*   (la*u_x)_z */
        + strx_i  *strz_k*i144*
	(la_km2*(shu[0][tk-2][tj+0][ti-2] - shu[0][tk-2][tj+0][ti+2] + 8 *
		 (- shu[0][tk-2][tj+0][ti-1]+shu[0][tk-2][tj+0][ti+1])) - 8*
	 (la_km1*(shu[0][tk-1][tj+0][ti-2] - shu[0][tk-1][tj+0][ti+2] + 8 *
		  (- shu[0][tk-1][tj+0][ti-1]+shu[0][tk-1][tj+0][ti+1]))) + 8*
	 (la_kp1*(shu[0][tk+1][tj+0][ti-2] - shu[0][tk+1][tj+0][ti+2] + 8 *
		  (- shu[0][tk+1][tj+0][ti-1]+shu[0][tk+1][tj+0][ti+1]))) -
	 (la_kp2*(shu[0][tk+2][tj+0][ti-2] - shu[0][tk+2][tj+0][ti+2] + 8 *
		  (- shu[0][tk+2][tj+0][ti-1]+shu[0][tk+2][tj+0][ti+1]))))
	/* (la*v_y)_z */
        + stry_j  *strz_k*i144*
	(la_km2*(shu[1][tk-2][tj-2][ti+0] - shu[1][tk-2][tj+2][ti+0] + 8 *
		 (- shu[1][tk-2][tj-1][ti+0]+shu[1][tk-2][tj+1][ti+0])) - 8*
	 (la_km1*(shu[1][tk-1][tj-2][ti+0] - shu[1][tk-1][tj+2][ti+0] + 8 *
		  (- shu[1][tk-1][tj-1][ti+0]+shu[1][tk-1][tj+1][ti+0]))) + 8*
	 (la_kp1*(shu[1][tk+1][tj-2][ti+0] - shu[1][tk+1][tj+2][ti+0] + 8 *
		  (- shu[1][tk+1][tj-1][ti+0]+shu[1][tk+1][tj+1][ti+0]))) -
	 (la_kp2*(shu[1][tk+2][tj-2][ti+0] - shu[1][tk+2][tj+2][ti+0] + 8 *
		  (- shu[1][tk+2][tj-1][ti+0]+shu[1][tk+2][tj+1][ti+0]))));

      
      // Use the result at point (i,j,k)
      if (pred) {
	// Apply predictor
	int idx = k * nij + j * ni + i;
	float_sw4 fact = dt*dt / rho;
	up(0,idx) = 2 * shu[0][tk][tj][ti] - um0 + fact * (cof * r1 + fo0);
	up(1,idx) = 2 * shu[1][tk][tj][ti] - um1 + fact * (cof * r2 + fo1);
	up(2,idx) = 2 * shu[2][tk][tj][ti] - um2 + fact * (cof * r3 + fo2);
      }
      else {
	// Apply 4th order correction
	int idx = k * nij + j * ni + i;
	float_sw4 fact = dt*dt*dt*dt / (12 * rho);
	up(0,idx) = up0 + fact * (cof * r1 + fo0);
	up(1,idx) = up1 + fact * (cof * r2 + fo1);
	up(2,idx) = up2 + fact * (cof * r3 + fo2);
      }
      
      // Rotate the register queues
      strz_km2 = strz_km1;
      strz_km1 = strz_k;
      strz_k = strz_kp1;
      strz_kp1 = strz_kp2;

      la_km2 = la_km1;
      la_km1 = la_ijk;
      la_ijk = la_kp1;
      la_kp1 = la_kp2;

      mu_km2 = mu_km1;
      mu_km1 = mu_ijk;
      mu_ijk = mu_kp1;
      mu_kp1 = mu_kp2;

    }  // End active

    // Move the index to the next K
    index += nij;

  } // End K loop
}

// *****************************************************************************
template <bool c_order, bool pred>
  __launch_bounds__(BX*BY)
  __global__ void rhs4_v2 (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                        int ni, int nj, int nk,
                        float_sw4* __restrict__ a_up,
                        const float_sw4* __restrict__ a_u,
                        const float_sw4* __restrict__ a_um,
                        const float_sw4* __restrict__ a_mu,
                        const float_sw4* __restrict__ a_lambda,
                        const float_sw4* __restrict__ a_rho,
                        const float_sw4* __restrict__ a_fo, 
                        const float_sw4* __restrict__ a_strx,
                        const float_sw4* __restrict__ a_stry,
                        const float_sw4* __restrict__ a_strz,
                        const float_sw4 h,
                        const float_sw4 dt) {
  int index;

  const float_sw4 i6   = 1.0/6;
  const float_sw4 i144 = 1.0/144;
  const float_sw4 tf   = 0.75;
  
  float_sw4 strx_im2, strx_im1, strx_i, strx_ip1, strx_ip2;
  float_sw4 stry_jm2, stry_jm1, stry_j, stry_jp1, stry_jp2;
  float_sw4 strz_km2, strz_km1, strz_k, strz_kp1, strz_kp2;

  float_sw4 la_km2, la_km1, la_ijk, la_kp1, la_kp2;
  float_sw4 la_im2, la_im1, la_ip1, la_ip2;
  float_sw4 la_jm2, la_jm1, la_jp1, la_jp2;

  float_sw4 mu_km2, mu_km1, mu_ijk, mu_kp1, mu_kp2;
  float_sw4 mu_im2, mu_im1, mu_ip1, mu_ip2;
  float_sw4 mu_jm2, mu_jm1, mu_jp1, mu_jp2;

  float_sw4 fo0, fo1, fo2, rho;
  float_sw4 up0, up1, up2;
  float_sw4 um0, um1, um2;

  // Prefetch variables
  float_sw4 a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2;
  
  float_sw4 r1, r2, r3, cof;
  int active=0, loader=0, loady2=0, loadx2=0;

  __shared__ float_sw4 shu[3][5][BY+4][BX+4];

  // i starts at 0, while j starts at jfirst
  const int i = threadIdx.x + blockIdx.x * BX;
  const int j = jfirst + threadIdx.y + blockIdx.y * BY;

  const int ti = threadIdx.x + 2;
  const int tj = threadIdx.y + 2;
  const int tk = 2;

  const int nij = ni * nj;
  const int nijk = nij * nk;

  int kthm0=3, kthm1=2, kthm2=1, kthm3=0, kthm4=4, kthtmp;

  // Active threads doing the computation at (i,j,k)
  if (i >= ifirst && i <= ilast && j <= jlast)
    active = 1;

  // Threads loading data with halos (4 regions)
  if (j-2 <= jlast + 2) {
    if (i-2 >= 0 && i-2 < ni)
      loader = 1;
    if (threadIdx.x < 4 && i+BX-2 < ni)
      loadx2 = 1;
    if (threadIdx.y < 4 && j+BY-2 <= jlast+2)
      loady2 = 1;
  }

  // Loading index for data with halos, starting at (i-2,j-2,kfirst-2)
  index = (kfirst - 2) * nij + (j - 2) * ni + i - 2;

  // Load the first 4 plans of U in the shared memory
  // and prefetch the 5th plan in a,b,c,d
  if (loader) {
    int idx = index;
    shu[0][0][threadIdx.y][threadIdx.x] = u(0,idx);
    shu[1][0][threadIdx.y][threadIdx.x] = u(1,idx);
    shu[2][0][threadIdx.y][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][1][threadIdx.y][threadIdx.x] = u(0,idx);
    shu[1][1][threadIdx.y][threadIdx.x] = u(1,idx);
    shu[2][1][threadIdx.y][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][2][threadIdx.y][threadIdx.x] = u(0,idx);
    shu[1][2][threadIdx.y][threadIdx.x] = u(1,idx);
    shu[2][2][threadIdx.y][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][3][threadIdx.y][threadIdx.x] = u(0,idx);
    shu[1][3][threadIdx.y][threadIdx.x] = u(1,idx);
    shu[2][3][threadIdx.y][threadIdx.x] = u(2,idx);
    idx += nij;
    a0 = u(0,idx);
    a1 = u(1,idx);
    a2 = u(2,idx);
  }
  if (loadx2) {
    int idx = index + BX;
    shu[0][0][threadIdx.y][threadIdx.x+BX] = u(0,idx);
    shu[1][0][threadIdx.y][threadIdx.x+BX] = u(1,idx);
    shu[2][0][threadIdx.y][threadIdx.x+BX] = u(2,idx);
    idx += nij;		         
    shu[0][1][threadIdx.y][threadIdx.x+BX] = u(0,idx);
    shu[1][1][threadIdx.y][threadIdx.x+BX] = u(1,idx);
    shu[2][1][threadIdx.y][threadIdx.x+BX] = u(2,idx);
    idx += nij;		         
    shu[0][2][threadIdx.y][threadIdx.x+BX] = u(0,idx);
    shu[1][2][threadIdx.y][threadIdx.x+BX] = u(1,idx);
    shu[2][2][threadIdx.y][threadIdx.x+BX] = u(2,idx);
    idx += nij;		         
    shu[0][3][threadIdx.y][threadIdx.x+BX] = u(0,idx);
    shu[1][3][threadIdx.y][threadIdx.x+BX] = u(1,idx);
    shu[2][3][threadIdx.y][threadIdx.x+BX] = u(2,idx);
    idx += nij;
    b0 = u(0,idx);
    b1 = u(1,idx);
    b2 = u(2,idx);
  }
  if (loader && loady2) {
    int idx = index + BY * ni;
    shu[0][0][threadIdx.y+BY][threadIdx.x] = u(0,idx);
    shu[1][0][threadIdx.y+BY][threadIdx.x] = u(1,idx);
    shu[2][0][threadIdx.y+BY][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][1][threadIdx.y+BY][threadIdx.x] = u(0,idx);
    shu[1][1][threadIdx.y+BY][threadIdx.x] = u(1,idx);
    shu[2][1][threadIdx.y+BY][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][2][threadIdx.y+BY][threadIdx.x] = u(0,idx);
    shu[1][2][threadIdx.y+BY][threadIdx.x] = u(1,idx);
    shu[2][2][threadIdx.y+BY][threadIdx.x] = u(2,idx);
    idx += nij;
    shu[0][3][threadIdx.y+BY][threadIdx.x] = u(0,idx);
    shu[1][3][threadIdx.y+BY][threadIdx.x] = u(1,idx);
    shu[2][3][threadIdx.y+BY][threadIdx.x] = u(2,idx);
    idx += nij;
    c0 = u(0,idx);
    c1 = u(1,idx);
    c2 = u(2,idx);
  }
  if (loadx2 && loady2) {
    int idx = index + BY * ni + BX;
    shu[0][0][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
    shu[1][0][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
    shu[2][0][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
    idx += nij;
    shu[0][1][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
    shu[1][1][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
    shu[2][1][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
    idx += nij;
    shu[0][2][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
    shu[1][2][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
    shu[2][2][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
    idx += nij;
    shu[0][3][threadIdx.y+BY][threadIdx.x+BX] = u(0,idx);
    shu[1][3][threadIdx.y+BY][threadIdx.x+BX] = u(1,idx);
    shu[2][3][threadIdx.y+BY][threadIdx.x+BX] = u(2,idx);
    idx += nij;
    d0 = u(0,idx);
    d1 = u(1,idx);
    d2 = u(2,idx);
  }

  if (active) {
    // Load strx & stry, centered at i and j
    strx_im2 = a_strx[i-2];
    strx_im1 = a_strx[i-1];
    strx_i   = a_strx[i];
    strx_ip1 = a_strx[i+1];
    strx_ip2 = a_strx[i+2];
    
    stry_jm2 = a_stry[j-2];
    stry_jm1 = a_stry[j-1];
    stry_j   = a_stry[j];
    stry_jp1 = a_stry[j+1];
    stry_jp2 = a_stry[j+2];

    // Load first 4 lambda and mu values 
    int idx = index + 2 * ni + 2;
    la_km2 = a_lambda[idx];
    mu_km2 = a_mu[idx];
    idx += nij;
    la_km1 = a_lambda[idx];
    mu_km1 = a_mu[idx];
    idx += nij;
    la_ijk = a_lambda[idx];
    mu_ijk = a_mu[idx];
    idx += nij;
    la_kp1 = a_lambda[idx];
    mu_kp1 = a_mu[idx];
  }

  // Move the loading index at kfirst + 2
  index += 4 * nij;

  // Load the first 4 values of strz
  strz_km2 = a_strz[kfirst-2];
  strz_km1 = a_strz[kfirst-1];
  strz_k   = a_strz[kfirst];
  strz_kp1 = a_strz[kfirst+1];

  cof = 1.0/(h*h);

  // Main loop on K dimension
  for (int k = kfirst; k <= klast; k++) {

    // Rotate the shared memory indices 
    kthtmp = kthm4;
    kthm4  = kthm3;
    kthm3  = kthm2;
    kthm2  = kthm1;
    kthm1  = kthm0;
    kthm0  = kthtmp;

    // Rotate the shared memory and store the prefetched variables
    __syncthreads();

    shu[0][kthm0][threadIdx.y][threadIdx.x] = a0;
    shu[1][kthm0][threadIdx.y][threadIdx.x] = a1;
    shu[2][kthm0][threadIdx.y][threadIdx.x] = a2;

    if (threadIdx.x < 4) {
      shu[0][kthm0][threadIdx.y][threadIdx.x+BX] = b0;
      shu[1][kthm0][threadIdx.y][threadIdx.x+BX] = b1;
      shu[2][kthm0][threadIdx.y][threadIdx.x+BX] = b2;
    }
    if (threadIdx.y < 4) {
      shu[0][kthm0][threadIdx.y+BY][threadIdx.x] = c0;
      shu[1][kthm0][threadIdx.y+BY][threadIdx.x] = c1;
      shu[2][kthm0][threadIdx.y+BY][threadIdx.x] = c2;
      if (threadIdx.x < 4) {
        shu[0][kthm0][threadIdx.y+BY][threadIdx.x+BX] = d0;
        shu[1][kthm0][threadIdx.y+BY][threadIdx.x+BX] = d1;
        shu[2][kthm0][threadIdx.y+BY][threadIdx.x+BX] = d2;
      }
    }
    __syncthreads();
    
    // Prefetch the next iteration variables for U to better hide the latency
    if (k < klast) {
      int idx = index + nij;
      if (loader) {
	a0 = u(0,idx);
	a1 = u(1,idx);
	a2 = u(2,idx);
      }
      idx += BX;
      if (loadx2) {
	b0 = u(0,idx);
	b1 = u(1,idx);
	b2 = u(2,idx);
      }
      idx = index + nij + BY * ni;
      if (loader & loady2) {
	c0 = u(0,idx);
	c1 = u(1,idx);
	c2 = u(2,idx);
      }
      idx += BX;
      if (loadx2 && loady2) {
	d0 = u(0,idx);
	d1 = u(1,idx);
	d2 = u(2,idx);
      }
    }

    // Active threads load center values and do the computation
    if (active) {
      // Load a new value of strz
      strz_kp2 = a_strz[k+2];

      // Load new values of lambda and mu, and the (i,j) halos
      {
	int idx = index + 2 * ni + 2;
	la_kp2 = a_lambda[idx];
	mu_kp2 = a_mu[idx];
	// Halos at k
	idx -= 2 * nij;
	la_im2 = a_lambda[idx-2];
	la_im1 = a_lambda[idx-1];
	la_ip1 = a_lambda[idx+1];
	la_ip2 = a_lambda[idx+2];
	la_jm2 = a_lambda[idx-2*ni];
	la_jm1 = a_lambda[idx-1*ni];
	la_jp1 = a_lambda[idx+ni];
	la_jp2 = a_lambda[idx+2*ni];
	mu_im2 = a_mu[idx-2];
	mu_im1 = a_mu[idx-1];
	mu_ip1 = a_mu[idx+1];
	mu_ip2 = a_mu[idx+2];
	mu_jm2 = a_mu[idx-2*ni];
	mu_jm1 = a_mu[idx-ni];
	mu_jp1 = a_mu[idx+ni];
	mu_jp2 = a_mu[idx+2*ni];
	// Read rho, fo, um, up in advance
	fo0 = fo(0,idx);
	fo1 = fo(1,idx);
	fo2 = fo(2,idx);
	rho = a_rho[idx];
	if (pred) {
	  um0 = um(0,idx);
	  um1 = um(1,idx);
	  um2 = um(2,idx);
	}
	else {
	  up0 = up(0,idx);
	  up1 = up(1,idx);
	  up2 = up(2,idx);
	}
      }

      // X contribution
      {
	float_sw4 mux1, mux2, mux3, mux4;
	mux1 = mu_im1 * strx_im1 - tf * (mu_ijk * strx_i + mu_im2 * strx_im2);
	mux2 = mu_im2 * strx_im2 + mu_ip1 * strx_ip1 + 3 * (mu_ijk * strx_i   + mu_im1 * strx_im1);
	mux3 = mu_im1 * strx_im1 + mu_ip2 * strx_ip2 + 3 * (mu_ip1 * strx_ip1 + mu_ijk * strx_i  );
	mux4 = mu_ip1 * strx_ip1 - tf * (mu_ijk * strx_i + mu_ip2 * strx_ip2);

	r1 = strx_i * ((2 * mux1 + la_im1 * strx_im1 - tf * (la_ijk * strx_i   + la_im2 * strx_im2)) *
		       (shu[0][kthm2][tj+0][ti-2] - shu[0][kthm2][tj+0][ti+0]) +
		       (2 * mux2 + la_im2 * strx_im2 + la_ip1 * strx_ip1 +
			3 * (la_ijk * strx_i   + la_im1 * strx_im1)) *
		       (shu[0][kthm2][tj+0][ti-1] - shu[0][kthm2][tj+0][ti+0]) +
		       (2 * mux3 + la_im1 * strx_im1 + la_ip2 * strx_ip2 +
			3 * (la_ip1 * strx_ip1 + la_ijk * strx_i  )) *
		       (shu[0][kthm2][tj+0][ti+1] - shu[0][kthm2][tj+0][ti+0]) +
		       (2 * mux4 + la_ip1 * strx_ip1 - tf * (la_ijk * strx_i   + la_ip2 * strx_ip2)) *
		       (shu[0][kthm2][tj+0][ti+2] - shu[0][kthm2][tj+0][ti+0]));
	r2 = strx_i * (mux1 * (shu[1][kthm2][tj+0][ti-2] - shu[1][kthm2][tj+0][ti+0]) +
		       mux2 * (shu[1][kthm2][tj+0][ti-1] - shu[1][kthm2][tj+0][ti+0]) +
		       mux3 * (shu[1][kthm2][tj+0][ti+1] - shu[1][kthm2][tj+0][ti+0]) +
		       mux4 * (shu[1][kthm2][tj+0][ti+2] - shu[1][kthm2][tj+0][ti+0]));
	r3 = strx_i * (mux1 * (shu[2][kthm2][tj+0][ti-2] - shu[2][kthm2][tj+0][ti+0]) +
		       mux2 * (shu[2][kthm2][tj+0][ti-1] - shu[2][kthm2][tj+0][ti+0]) +
		       mux3 * (shu[2][kthm2][tj+0][ti+1] - shu[2][kthm2][tj+0][ti+0]) +
		       mux4 * (shu[2][kthm2][tj+0][ti+2] - shu[2][kthm2][tj+0][ti+0]));
      }
      // Y contribution
      {
	float_sw4 muy1, muy2, muy3, muy4;
	muy1 = mu_jm1 * stry_jm1 - tf * (mu_ijk * stry_j + mu_jm2 * stry_jm2);
	muy2 = mu_jm2 * stry_jm2 + mu_jp1 * stry_jp1 + 3 * (mu_ijk * stry_j   + mu_jm1 * stry_jm1);
	muy3 = mu_jm1 * stry_jm1 + mu_jp2 * stry_jp2 + 3 * (mu_jp1 * stry_jp1 + mu_ijk * stry_j  );
	muy4 = mu_jp1 * stry_jp1 - tf * (mu_ijk * stry_j + mu_jp2 * stry_jp2);

	r1 += stry_j * (muy1 * (shu[0][kthm2][tj-2][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
			muy2 * (shu[0][kthm2][tj-1][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
			muy3 * (shu[0][kthm2][tj+1][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
			muy4 * (shu[0][kthm2][tj+2][ti+0] - shu[0][kthm2][tj+0][ti+0]));
	r2 += stry_j * ((2 * muy1 + la_jm1 * stry_jm1 - tf * (la_ijk * stry_j + la_jm2 * stry_jm2)) *
			(shu[1][kthm2][tj-2][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
			(2 * muy2 + la_jm2 * stry_jm2 + la_jp1 * stry_jp1 +
			 3 * (la_ijk * stry_j   + la_jm1 * stry_jm1)) *
			(shu[1][kthm2][tj-1][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
			(2 * muy3 + la_jm1 * stry_jm1 + la_jp2 * stry_jp2 +
			 3 * (la_jp1 * stry_jp1 + la_ijk * stry_j  )) *
			(shu[1][kthm2][tj+1][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
			(2 * muy4 + la_jp1 * stry_jp1 - tf * (la_ijk * stry_j  + la_jp2 * stry_jp2)) *
			(shu[1][kthm2][tj+2][ti+0] - shu[1][kthm2][tj+0][ti+0]));
	r3 += stry_j  *(muy1 * (shu[2][kthm2][tj-2][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
			muy2 * (shu[2][kthm2][tj-1][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
			muy3 * (shu[2][kthm2][tj+1][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
			muy4 * (shu[2][kthm2][tj+2][ti+0] - shu[2][kthm2][tj+0][ti+0]));
      }
      // Z contribution
      {
	float_sw4 muz1, muz2, muz3, muz4;
	muz1 = mu_km1 * strz_km1 - tf * (mu_ijk * strz_k + mu_km2 * strz_km2);
	muz2 = mu_km2 * strz_km2 + mu_kp1 * strz_kp1 + 3 * (mu_ijk * strz_k   + mu_km1 * strz_km1);
	muz3 = mu_km1 * strz_km1 + mu_kp2 * strz_kp2 + 3 * (mu_kp1 * strz_kp1 + mu_ijk * strz_k);
	muz4 = mu_kp1 * strz_kp1 - tf * (mu_ijk * strz_k + mu_kp2 * strz_kp2);
	r1 += strz_k * (muz1 * (shu[0][kthm4][tj+0][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
			muz2 * (shu[0][kthm3][tj+0][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
			muz3 * (shu[0][kthm1][tj+0][ti+0] - shu[0][kthm2][tj+0][ti+0]) +
			muz4 * (shu[0][kthm0][tj+0][ti+0] - shu[0][kthm2][tj+0][ti+0]));
	r2 += strz_k * (muz1 * (shu[1][kthm4][tj+0][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
			muz2 * (shu[1][kthm3][tj+0][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
			muz3 * (shu[1][kthm1][tj+0][ti+0] - shu[1][kthm2][tj+0][ti+0]) +
			muz4 * (shu[1][kthm0][tj+0][ti+0] - shu[1][kthm2][tj+0][ti+0]));
	r3 += strz_k * ((2 * muz1 + la_km1 * strz_km1 - tf * (la_ijk * strz_k + la_km2 * strz_km2)) *
			(shu[2][kthm4][tj+0][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
			(2 * muz2 + la_km2 * strz_km2 + la_kp1 * strz_kp1 +
			 3 * (la_ijk * strz_k   + la_km1 * strz_km1)) *
			(shu[2][kthm3][tj+0][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
			(2 * muz3 + la_km1 * strz_km1 + la_kp2 * strz_kp2 +
			 3 * (la_kp1 * strz_kp1 + la_ijk * strz_k  )) *
			(shu[2][kthm1][tj+0][ti+0] - shu[2][kthm2][tj+0][ti+0]) +
			(2 * muz4 + la_kp1 * strz_kp1 - tf * (la_ijk * strz_k + la_kp2 * strz_kp2)) *
			(shu[2][kthm0][tj+0][ti+0] - shu[2][kthm2][tj+0][ti+0]));
      }
      r1 *= i6;
      r2 *= i6;
      r3 *= i6;


      /* Mixed derivatives: */
      /*   (la*v_y)_x */
      r1 += strx_i  * stry_j * i144 * 
	(la_im2 * (shu[1][kthm2][tj-2][ti-2] - shu[1][kthm2][tj+2][ti-2] + 8 *
		   (-shu[1][kthm2][tj-1][ti-2] + shu[1][kthm2][tj+1][ti-2])) - 8 *
	 (la_im1 * (shu[1][kthm2][tj-2][ti-1] - shu[1][kthm2][tj+2][ti-1] + 8 *
		    (-shu[1][kthm2][tj-1][ti-1] + shu[1][kthm2][tj+1][ti-1]))) + 8 *
	 (la_ip1 * (shu[1][kthm2][tj-2][ti+1] - shu[1][kthm2][tj+2][ti+1] + 8 *
		    (-shu[1][kthm2][tj-1][ti+1] + shu[1][kthm2][tj+1][ti+1]))) - 
	 (la_ip2 * (shu[1][kthm2][tj-2][ti+2] - shu[1][kthm2][tj+2][ti+2] + 8 *
		    (-shu[1][kthm2][tj-1][ti+2] + shu[1][kthm2][tj+1][ti+2]))))
	/*   (la*w_z)_x */
        + strx_i * strz_k * i144 *
	(la_im2 * (shu[2][kthm4][tj+0][ti-2] - shu[2][kthm0][tj+0][ti-2] + 8 *
		   (-shu[2][kthm3][tj+0][ti-2] + shu[2][kthm1][tj+0][ti-2])) - 8 *
	 (la_im1 * (shu[2][kthm4][tj+0][ti-1] - shu[2][kthm0][tj+0][ti-1] + 8 *
		    (-shu[2][kthm3][tj+0][ti-1] + shu[2][kthm1][tj+0][ti-1]))) + 8 *
	 (la_ip1 * (shu[2][kthm4][tj+0][ti+1] - shu[2][kthm0][tj+0][ti+1] + 8 *
		    (-shu[2][kthm3][tj+0][ti+1] + shu[2][kthm1][tj+0][ti+1]))) - 
	 (la_ip2 * (shu[2][kthm4][tj+0][ti+2] - shu[2][kthm0][tj+0][ti+2] + 8 *
		    (-shu[2][kthm3][tj+0][ti+2] + shu[2][kthm1][tj+0][ti+2]))))
	/*   (mu*v_x)_y */
        + strx_i * stry_j * i144 *
	(mu_jm2 * (shu[1][kthm2][tj-2][ti-2] - shu[1][kthm2][tj-2][ti+2] + 8 *
		   (-shu[1][kthm2][tj-2][ti-1] + shu[1][kthm2][tj-2][ti+1])) - 8 *
	 (mu_jm1 * (shu[1][kthm2][tj-1][ti-2] - shu[1][kthm2][tj-1][ti+2] + 8 *
		    (-shu[1][kthm2][tj-1][ti-1] + shu[1][kthm2][tj-1][ti+1]))) + 8 *
	 (mu_jp1 * (shu[1][kthm2][tj+1][ti-2] - shu[1][kthm2][tj+1][ti+2] + 8 *
		    (-shu[1][kthm2][tj+1][ti-1] + shu[1][kthm2][tj+1][ti+1]))) - 
	 (mu_jp2 * (shu[1][kthm2][tj+2][ti-2] - shu[1][kthm2][tj+2][ti+2] + 8 *
		    (-shu[1][kthm2][tj+2][ti-1] + shu[1][kthm2][tj+2][ti+1]))))
	/*   (mu*w_x)_z */
        + strx_i * strz_k * i144 *
	(mu_km2 * (shu[2][kthm4][tj+0][ti-2] - shu[2][kthm4][tj+0][ti+2] + 8 *
		   (-shu[2][kthm4][tj+0][ti-1] + shu[2][kthm4][tj+0][ti+1])) - 8 *
 	 (mu_km1 * (shu[2][kthm3][tj+0][ti-2] - shu[2][kthm3][tj+0][ti+2] + 8 *
		    (-shu[2][kthm3][tj+0][ti-1] + shu[2][kthm3][tj+0][ti+1]))) + 8 *
 	 (mu_kp1 * (shu[2][kthm1][tj+0][ti-2] - shu[2][kthm1][tj+0][ti+2] + 8 *
		    (-shu[2][kthm1][tj+0][ti-1] + shu[2][kthm1][tj+0][ti+1]))) -
	 (mu_kp2 * (shu[2][kthm0][tj+0][ti-2] - shu[2][kthm0][tj+0][ti+2] + 8 *
		    (-shu[2][kthm0][tj+0][ti-1] + shu[2][kthm0][tj+0][ti+1]))));

      /*   (mu*u_y)_x */
      r2 += strx_i  *stry_j  *i144*
	(mu_im2 * (shu[0][kthm2][tj-2][ti-2] - shu[0][kthm2][tj+2][ti-2] + 8 *
		   (-shu[0][kthm2][tj-1][ti-2] + shu[0][kthm2][tj+1][ti-2])) - 8 *
	 (mu_im1 * (shu[0][kthm2][tj-2][ti-1] - shu[0][kthm2][tj+2][ti-1] + 8 *
		    (-shu[0][kthm2][tj-1][ti-1] + shu[0][kthm2][tj+1][ti-1]))) + 8 *
	 (mu_ip1 * (shu[0][kthm2][tj-2][ti+1] - shu[0][kthm2][tj+2][ti+1] + 8 *
		    (-shu[0][kthm2][tj-1][ti+1] + shu[0][kthm2][tj+1][ti+1]))) -
	 (mu_ip2 * (shu[0][kthm2][tj-2][ti+2] - shu[0][kthm2][tj+2][ti+2] + 8 *
		    (-shu[0][kthm2][tj-1][ti+2] + shu[0][kthm2][tj+1][ti+2]))))
	/* (la*u_x)_y */
        + strx_i * stry_j * i144 *
	(la_jm2 * (shu[0][kthm2][tj-2][ti-2] - shu[0][kthm2][tj-2][ti+2] + 8 *
		   (-shu[0][kthm2][tj-2][ti-1] + shu[0][kthm2][tj-2][ti+1])) - 8 *
	 (la_jm1 * (shu[0][kthm2][tj-1][ti-2] - shu[0][kthm2][tj-1][ti+2] + 8 *
		    (-shu[0][kthm2][tj-1][ti-1] + shu[0][kthm2][tj-1][ti+1]))) + 8 *
	 (la_jp1 * (shu[0][kthm2][tj+1][ti-2] - shu[0][kthm2][tj+1][ti+2] + 8 *
		    (-shu[0][kthm2][tj+1][ti-1] + shu[0][kthm2][tj+1][ti+1]))) -
	 (la_jp2 * (shu[0][kthm2][tj+2][ti-2] - shu[0][kthm2][tj+2][ti+2] + 8 *
		    (-shu[0][kthm2][tj+2][ti-1] + shu[0][kthm2][tj+2][ti+1]))))
	/* (la*w_z)_y */
        + stry_j * strz_k * i144 *
	(la_jm2 * (shu[2][kthm4][tj-2][ti+0] - shu[2][kthm0][tj-2][ti+0] + 8 *
		   (-shu[2][kthm3][tj-2][ti+0] + shu[2][kthm1][tj-2][ti+0])) - 8 *
	 (la_jm1 * (shu[2][kthm4][tj-1][ti+0] - shu[2][kthm0][tj-1][ti+0] + 8 *
		    (-shu[2][kthm3][tj-1][ti+0] + shu[2][kthm1][tj-1][ti+0]))) + 8 *
	 (la_jp1 * (shu[2][kthm4][tj+1][ti+0] - shu[2][kthm0][tj+1][ti+0] + 8 *
		    (-shu[2][kthm3][tj+1][ti+0] + shu[2][kthm1][tj+1][ti+0]))) -
	 (la_jp2 * (shu[2][kthm4][tj+2][ti+0] - shu[2][kthm0][tj+2][ti+0] + 8 *
		    (-shu[2][kthm3][tj+2][ti+0] + shu[2][kthm1][tj+2][ti+0]))))
	/* (mu*w_y)_z */
        + stry_j * strz_k * i144 *
	(mu_km2 * (shu[2][kthm4][tj-2][ti+0] - shu[2][kthm4][tj+2][ti+0] + 8 *
		   (-shu[2][kthm4][tj-1][ti+0] + shu[2][kthm4][tj+1][ti+0])) - 8 *
	 (mu_km1 * (shu[2][kthm3][tj-2][ti+0] - shu[2][kthm3][tj+2][ti+0] + 8 *
		    (-shu[2][kthm3][tj-1][ti+0] + shu[2][kthm3][tj+1][ti+0]))) + 8 *
	 (mu_kp1 * (shu[2][kthm1][tj-2][ti+0] - shu[2][kthm1][tj+2][ti+0] + 8 *
		    (-shu[2][kthm1][tj-1][ti+0] + shu[2][kthm1][tj+1][ti+0]))) -
	 (mu_kp2 * (shu[2][kthm0][tj-2][ti+0] - shu[2][kthm0][tj+2][ti+0] + 8 *
		    (-shu[2][kthm0][tj-1][ti+0] + shu[2][kthm0][tj+1][ti+0]))));

      /*  (mu*u_z)_x */
      r3 += strx_i * strz_k * i144 *
	(mu_im2 * (shu[0][kthm4][tj+0][ti-2] - shu[0][kthm0][tj+0][ti-2] + 8 *
		   (-shu[0][kthm3][tj+0][ti-2] + shu[0][kthm1][tj+0][ti-2])) - 8 *
	 (mu_im1 * (shu[0][kthm4][tj+0][ti-1] - shu[0][kthm0][tj+0][ti-1] + 8 *
		    (-shu[0][kthm3][tj+0][ti-1] + shu[0][kthm1][tj+0][ti-1]))) + 8 *
	 (mu_ip1 * (shu[0][kthm4][tj+0][ti+1] - shu[0][kthm0][tj+0][ti+1] + 8 *
		    (-shu[0][kthm3][tj+0][ti+1] + shu[0][kthm1][tj+0][ti+1]))) -
	 (mu_ip2 * (shu[0][kthm4][tj+0][ti+2] - shu[0][kthm0][tj+0][ti+2] + 8 *
		    (-shu[0][kthm3][tj+0][ti+2] + shu[0][kthm1][tj+0][ti+2]))))
	/* (mu*v_z)_y */
        + stry_j  *strz_k*i144*
	(mu_jm2 * (shu[1][kthm4][tj-2][ti+0] - shu[1][kthm0][tj-2][ti+0] + 8 *
		   (- shu[1][kthm3][tj-2][ti+0]+shu[1][kthm1][tj-2][ti+0])) - 8 * 
	 (mu_jm1 * (shu[1][kthm4][tj-1][ti+0] - shu[1][kthm0][tj-1][ti+0] + 8 *
		    (- shu[1][kthm3][tj-1][ti+0]+shu[1][kthm1][tj-1][ti+0]))) + 8 * 
	 (mu_jp1 * (shu[1][kthm4][tj+1][ti+0] - shu[1][kthm0][tj+1][ti+0] + 8 *
		    (- shu[1][kthm3][tj+1][ti+0]+shu[1][kthm1][tj+1][ti+0]))) -
	 (mu_jp2 * (shu[1][kthm4][tj+2][ti+0] - shu[1][kthm0][tj+2][ti+0] + 8 *
		    (- shu[1][kthm3][tj+2][ti+0]+shu[1][kthm1][tj+2][ti+0]))))
	/*   (la*u_x)_z */
        + strx_i  *strz_k*i144*
	(la_km2*(shu[0][kthm4][tj+0][ti-2] - shu[0][kthm4][tj+0][ti+2] + 8 *
		 (- shu[0][kthm4][tj+0][ti-1]+shu[0][kthm4][tj+0][ti+1])) - 8*
	 (la_km1*(shu[0][kthm3][tj+0][ti-2] - shu[0][kthm3][tj+0][ti+2] + 8 *
		  (- shu[0][kthm3][tj+0][ti-1]+shu[0][kthm3][tj+0][ti+1]))) + 8*
	 (la_kp1*(shu[0][kthm1][tj+0][ti-2] - shu[0][kthm1][tj+0][ti+2] + 8 *
		  (- shu[0][kthm1][tj+0][ti-1]+shu[0][kthm1][tj+0][ti+1]))) -
	 (la_kp2*(shu[0][kthm0][tj+0][ti-2] - shu[0][kthm0][tj+0][ti+2] + 8 *
		  (- shu[0][kthm0][tj+0][ti-1]+shu[0][kthm0][tj+0][ti+1]))))
	/* (la*v_y)_z */
        + stry_j  *strz_k*i144*
	(la_km2*(shu[1][kthm4][tj-2][ti+0] - shu[1][kthm4][tj+2][ti+0] + 8 *
		 (- shu[1][kthm4][tj-1][ti+0]+shu[1][kthm4][tj+1][ti+0])) - 8*
	 (la_km1*(shu[1][kthm3][tj-2][ti+0] - shu[1][kthm3][tj+2][ti+0] + 8 *
		  (- shu[1][kthm3][tj-1][ti+0]+shu[1][kthm3][tj+1][ti+0]))) + 8*
	 (la_kp1*(shu[1][kthm1][tj-2][ti+0] - shu[1][kthm1][tj+2][ti+0] + 8 *
		  (- shu[1][kthm1][tj-1][ti+0]+shu[1][kthm1][tj+1][ti+0]))) -
	 (la_kp2*(shu[1][kthm0][tj-2][ti+0] - shu[1][kthm0][tj+2][ti+0] + 8 *
		  (- shu[1][kthm0][tj-1][ti+0]+shu[1][kthm0][tj+1][ti+0]))));

      
      // Use the result at point (i,j,k)
      if (pred) {
	// Apply predictor
	int idx = k * nij + j * ni + i;
	float_sw4 fact = dt*dt / rho;
	up(0,idx) = 2 * shu[0][kthm2][tj][ti] - um0 + fact * (cof * r1 + fo0);
	up(1,idx) = 2 * shu[1][kthm2][tj][ti] - um1 + fact * (cof * r2 + fo1);
	up(2,idx) = 2 * shu[2][kthm2][tj][ti] - um2 + fact * (cof * r3 + fo2);
      }
      else {
	// Apply 4th order correction
	int idx = k * nij + j * ni + i;
	float_sw4 fact = dt*dt*dt*dt / (12 * rho);
	up(0,idx) = up0 + fact * (cof * r1 + fo0);
	up(1,idx) = up1 + fact * (cof * r2 + fo1);
	up(2,idx) = up2 + fact * (cof * r3 + fo2);
      }
      
      // Rotate the register queues
      strz_km2 = strz_km1;
      strz_km1 = strz_k;
      strz_k = strz_kp1;
      strz_kp1 = strz_kp2;

      la_km2 = la_km1;
      la_km1 = la_ijk;
      la_ijk = la_kp1;
      la_kp1 = la_kp2;

      mu_km2 = mu_km1;
      mu_km1 = mu_ijk;
      mu_ijk = mu_kp1;
      mu_kp1 = mu_kp2;

    }  // End active

    // Move the index to the next K
    index += nij;

  } // End K loop
}
// *************************************************************************************
// Kernel launchers

void rhs4_pred_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                    int ni, int nj, int nk,
                    float_sw4* a_up, float_sw4* a_u, float_sw4* a_um,
                    float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
                    float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
                    float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream) {
  // Cover [0:ni-1][jfirst:jlast]
  int njcomp = jlast - jfirst + 1;
  dim3 blocks = dim3((ni+BX-1)/BX, (njcomp+BY-1)/BY, 1);
  dim3 threads = dim3(BX, BY, 1);
  if (c_order)
    rhs4_v2<1,1><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, a_um,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
  else
    rhs4_v2<0,1><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, a_um,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
}

void rhs4_corr_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                    int ni, int nj, int nk,
                    float_sw4* a_up, float_sw4* a_u,
                    float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
                    float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
                    float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream) {
  // Cover [0:ni-1][jfirst:jlast]
  int njcomp = jlast - jfirst + 1;
  dim3 blocks = dim3((ni+BX-1)/BX, (njcomp+BY-1)/BY, 1);
  dim3 threads = dim3(BX, BY, 1);
  if (c_order)
    rhs4_v2<1,0><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, NULL,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
  else
    rhs4_v2<0,0><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, NULL,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
}

// *************************************************************************************
// *************************************************************************************
// *************************************************************************************
// RHS4 X halos
// ------------
// This kernel computes the halos on each side in the X dimension :
//   [ifirst:ifirst+1, jfirst:jlast, kfirst:klast]
//   [ilast-1:ilast  , jfirst:jlast, kfirst:klast]
// We're using 384 threads : 2x16x12 to compute the 2 elements for a (16x12) area.
// The 384 threads are re-mapped as 6x16x4 threads to load the input data with halos,
// so that the 6 coalesced elements that we need can be loaded efficiently in one operation.

template <bool c_order, bool pred>
  __launch_bounds__(384)
  __global__ void rhs4_X (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			  int ni, int nj, int nk,
			  float_sw4* __restrict__ a_up,
			  const float_sw4* __restrict__ a_u,
			  const float_sw4* __restrict__ a_um,
			  const float_sw4* __restrict__ a_mu,
			  const float_sw4* __restrict__ a_lambda,
			  const float_sw4* __restrict__ a_rho,
			  const float_sw4* __restrict__ a_fo,
			  const float_sw4* __restrict__ a_strx,
			  const float_sw4* __restrict__ a_stry,
			  const float_sw4* __restrict__ a_strz,
			  const float_sw4 h,
			  const float_sw4 dt) {
  int index, i;

  const float_sw4 i6   = 1.0/6;
  const float_sw4 i144 = 1.0/144;
  const float_sw4 tf   = 0.75;
  
  float_sw4 strx_im2, strx_im1, strx_i, strx_ip1, strx_ip2;
  float_sw4 stry_jm2, stry_jm1, stry_j, stry_jp1, stry_jp2;
  float_sw4 strz_km2, strz_km1, strz_k, strz_kp1, strz_kp2;

  float_sw4 la_km2, la_km1, la_ijk, la_kp1, la_kp2;
  float_sw4 la_im2, la_im1, la_ip1, la_ip2;
  float_sw4 la_jm2, la_jm1, la_jp1, la_jp2;

  float_sw4 mu_km2, mu_km1, mu_ijk, mu_kp1, mu_kp2;
  float_sw4 mu_im2, mu_im1, mu_ip1, mu_ip2;
  float_sw4 mu_jm2, mu_jm1, mu_jp1, mu_jp2;

  float_sw4 r1, r2, r3;
  int active=0, loadj1=0, loadj2=0, loadk1=0, loadk2=0, loadk3=0, loadk4=0;

  // Shared memory for 3 wavefields, and for lambda / mu
  __shared__ float_sw4 shm[3][12+4][16+4][2+4];

  float_sw4 cof = 1.0 / (h * h);

  // J and K start at (jfirst, kfirst)
  const int j = jfirst + threadIdx.y + blockIdx.y * 16;
  const int k = kfirst + threadIdx.z + blockIdx.z * 12;

  // Natural thread mapping for the computation
  const int ti = threadIdx.x + 2;
  const int tj = threadIdx.y + 2;
  const int tk = threadIdx.z + 2;

  // Use the (6,16,4) mapping to load the data into shared memory
  int tid = (threadIdx.z * 16 + threadIdx.y) * 2 + threadIdx.x;
  int txload = (tid % 6);
  int tyload = (tid / 6);
  int tzload = tyload / 16;
  tyload = tyload % 16;
  
  const int nij = ni * nj;
  const int nijk = nij * nk;

  // Active threads doing the computation
  if (j <= jlast && k <= klast)
    active = 1;

  if (active) {
    // Load stry centered at j
    stry_jm2 = a_stry[j-2];
    stry_jm1 = a_stry[j-1];
    stry_j   = a_stry[j];
    stry_jp1 = a_stry[j+1];
    stry_jp2 = a_stry[j+2];
    // Load strz centered at k
    strz_km2 = a_strz[k-2];
    strz_km1 = a_strz[k-1];
    strz_k   = a_strz[k];
    strz_kp1 = a_strz[k+1];
    strz_kp2 = a_strz[k+2];
  }

  // Loading index, computed with the alternate thread mapping, starting at (jfirst-2, kfirst-2)
  int jload, kload;
  jload = blockIdx.y * 16 + tyload + jfirst - 2;
  kload = blockIdx.z * 12 + tzload + kfirst - 2;

  // We're loading 6x20x16 elements with 6x16x4 threads => 2 x 4 = 8 sections.
  // Find out Which threads can load the data for the different sections
  if (jload <= jlast+2) loadj1 = 1;
  if (tyload < 4 && jload+16 <= jlast+2) loadj2 = 1;
  if (kload <= klast+2) loadk1 = 1;
  if (kload+4 <= klast+2) loadk2 = 1;
  if (kload+8 <= klast+2) loadk3 = 1;
  if (kload+12 <= klast+2) loadk4 = 1;

  // Loop on the 2 sides
#pragma unroll
  for (int iside=0; iside < 2; iside++) {
    
    if (iside == 0) {
      // ifirst side
      index = (kload * nj + jload) * ni + ifirst - 2 + txload;
      i = ifirst + threadIdx.x;
    }
    else {
      // ilast side
      index = (kload * nj + jload) * ni + ilast - 3 + txload;
      i = ilast - 1 + threadIdx.x;
    }
    
    // Load lambda and mu using the shared memory
    if (loadj1) {
      int idx = index;
      if (loadk1) {
	shm[0][tzload][tyload][txload] = a_lambda[idx];
	shm[1][tzload][tyload][txload] = a_mu[idx];
      }
      idx += 4 * nij;
      if (loadk2) {
	shm[0][tzload+4][tyload][txload] = a_lambda[idx];
	shm[1][tzload+4][tyload][txload] = a_mu[idx];
      }
      idx += 4 * nij;
      if (loadk3) {
	shm[0][tzload+8][tyload][txload] = a_lambda[idx];
	shm[1][tzload+8][tyload][txload] = a_mu[idx];
      }
      idx += 4 * nij;
      if (loadk4) {
	shm[0][tzload+12][tyload][txload] = a_lambda[idx];
	shm[1][tzload+12][tyload][txload] = a_mu[idx];
      }
    }
    if (loadj2) {
      int idx = index + 16 * ni;
      if (loadk1) {
	shm[0][tzload][tyload+16][txload] = a_lambda[idx];
	shm[1][tzload][tyload+16][txload] = a_mu[idx];
      }
      idx += 4 * nij;
      if (loadk2) {
	shm[0][tzload+4][tyload+16][txload] = a_lambda[idx];
	shm[1][tzload+4][tyload+16][txload] = a_mu[idx];
      }
      idx += 4 * nij;
      if (loadk3) {
	shm[0][tzload+8][tyload+16][txload] = a_lambda[idx];
	shm[1][tzload+8][tyload+16][txload] = a_mu[idx];
      }
      idx += 4 * nij;
      if (loadk4) {
	shm[0][tzload+12][tyload+16][txload] = a_lambda[idx];
	shm[1][tzload+12][tyload+16][txload] = a_mu[idx];
      }
    }
    __syncthreads();
    // Put lambda and mu in registers, using the natural thread mapping
    la_km2 = shm[0][tk-2][tj][ti];
    la_km1 = shm[0][tk-1][tj][ti];
    la_ijk = shm[0][tk  ][tj][ti];
    la_kp1 = shm[0][tk+1][tj][ti];
    la_kp2 = shm[0][tk+2][tj][ti];

    la_jm2 = shm[0][tk][tj-2][ti];
    la_jm1 = shm[0][tk][tj-1][ti];
    la_jp1 = shm[0][tk][tj+1][ti];
    la_jp2 = shm[0][tk][tj+2][ti];

    la_im2 = shm[0][tk][tj][ti-2];
    la_im1 = shm[0][tk][tj][ti-1];
    la_ip1 = shm[0][tk][tj][ti+1];
    la_ip2 = shm[0][tk][tj][ti+2];

    mu_km2 = shm[1][tk-2][tj][ti];
    mu_km1 = shm[1][tk-1][tj][ti];
    mu_ijk = shm[1][tk  ][tj][ti];
    mu_kp1 = shm[1][tk+1][tj][ti];
    mu_kp2 = shm[1][tk+2][tj][ti];

    mu_jm2 = shm[1][tk][tj-2][ti];
    mu_jm1 = shm[1][tk][tj-1][ti];
    mu_jp1 = shm[1][tk][tj+1][ti];
    mu_jp2 = shm[1][tk][tj+2][ti];

    mu_im2 = shm[1][tk][tj][ti-2];
    mu_im1 = shm[1][tk][tj][ti-1];
    mu_ip1 = shm[1][tk][tj][ti+1];
    mu_ip2 = shm[1][tk][tj][ti+2];
    __syncthreads();

    // Load the 3 wavefields to shared memory
    if (loadj1) {
      int idx = index;
      if (loadk1) {
	shm[0][tzload][tyload][txload] = u(0,idx);
	shm[1][tzload][tyload][txload] = u(1,idx);
	shm[2][tzload][tyload][txload] = u(2,idx);
      }
      idx += 4 * nij;
      if (loadk2) {
	shm[0][tzload+4][tyload][txload] = u(0,idx);
	shm[1][tzload+4][tyload][txload] = u(1,idx);
	shm[2][tzload+4][tyload][txload] = u(2,idx);
      }
      idx += 4 * nij;
      if (loadk3) {
	shm[0][tzload+8][tyload][txload] = u(0,idx);
	shm[1][tzload+8][tyload][txload] = u(1,idx);
	shm[2][tzload+8][tyload][txload] = u(2,idx);
      }
      idx += 4 * nij;
      if (loadk4) {
	shm[0][tzload+12][tyload][txload] = u(0,idx);
	shm[1][tzload+12][tyload][txload] = u(1,idx);
	shm[2][tzload+12][tyload][txload] = u(2,idx);
      }
    }
    if (loadj2) {
      int idx = index + 16 * ni;
      if (loadk1) {
	shm[0][tzload][tyload+16][txload] = u(0,idx);
	shm[1][tzload][tyload+16][txload] = u(1,idx);
	shm[2][tzload][tyload+16][txload] = u(2,idx);
      }
      idx += 4 * nij;
      if (loadk2) {
	shm[0][tzload+4][tyload+16][txload] = u(0,idx);
	shm[1][tzload+4][tyload+16][txload] = u(1,idx);
	shm[2][tzload+4][tyload+16][txload] = u(2,idx);
      }
      idx += 4 * nij;
      if (loadk3) {
	shm[0][tzload+8][tyload+16][txload] = u(0,idx);
	shm[1][tzload+8][tyload+16][txload] = u(1,idx);
	shm[2][tzload+8][tyload+16][txload] = u(2,idx);
      }
      idx += 4 * nij;
      if (loadk4) {
	shm[0][tzload+12][tyload+16][txload] = u(0,idx);
	shm[1][tzload+12][tyload+16][txload] = u(1,idx);
	shm[2][tzload+12][tyload+16][txload] = u(2,idx);
      }
    }
    __syncthreads();

    if (active) {

      // Load strx centered at i
      strx_im2 = a_strx[i-2];
      strx_im1 = a_strx[i-1];
      strx_i   = a_strx[i];
      strx_ip1 = a_strx[i+1];
      strx_ip2 = a_strx[i+2];
    
      // X contribution
      {
	float_sw4 mux1, mux2, mux3, mux4;
	mux1 = mu_im1 * strx_im1 - tf * (mu_ijk * strx_i + mu_im2 * strx_im2);
	mux2 = mu_im2 * strx_im2 + mu_ip1 * strx_ip1 + 3 * (mu_ijk * strx_i   + mu_im1 * strx_im1);
	mux3 = mu_im1 * strx_im1 + mu_ip2 * strx_ip2 + 3 * (mu_ip1 * strx_ip1 + mu_ijk * strx_i  );
	mux4 = mu_ip1 * strx_ip1 - tf * (mu_ijk * strx_i + mu_ip2 * strx_ip2);

	r1 = strx_i * ((2 * mux1 + la_im1 * strx_im1 - tf * (la_ijk * strx_i   + la_im2 * strx_im2)) *
		       (shm[0][tk+0][tj+0][ti-2] - shm[0][tk+0][tj+0][ti+0])+
		       (2 * mux2 + la_im2 * strx_im2 + la_ip1 * strx_ip1 +
			3 * (la_ijk * strx_i   + la_im1 * strx_im1)) *
		       (shm[0][tk+0][tj+0][ti-1] - shm[0][tk+0][tj+0][ti+0]) +
		       (2 * mux3 + la_im1 * strx_im1 + la_ip2 * strx_ip2 +
			3 * (la_ip1 * strx_ip1 + la_ijk * strx_i  )) *
		       (shm[0][tk+0][tj+0][ti+1] - shm[0][tk+0][tj+0][ti+0]) +
		       (2 * mux4 + la_ip1 * strx_ip1 - tf * (la_ijk * strx_i   + la_ip2 * strx_ip2)) *
		       (shm[0][tk+0][tj+0][ti+2] - shm[0][tk+0][tj+0][ti+0]));
	r2 = strx_i * (mux1 * (shm[1][tk+0][tj+0][ti-2] - shm[1][tk+0][tj+0][ti+0]) +
		       mux2 * (shm[1][tk+0][tj+0][ti-1] - shm[1][tk+0][tj+0][ti+0]) +
		       mux3 * (shm[1][tk+0][tj+0][ti+1] - shm[1][tk+0][tj+0][ti+0]) +
		       mux4 * (shm[1][tk+0][tj+0][ti+2] - shm[1][tk+0][tj+0][ti+0]));
	r3 = strx_i * (mux1 * (shm[2][tk+0][tj+0][ti-2] - shm[2][tk+0][tj+0][ti+0]) +
		       mux2 * (shm[2][tk+0][tj+0][ti-1] - shm[2][tk+0][tj+0][ti+0]) +
		       mux3 * (shm[2][tk+0][tj+0][ti+1] - shm[2][tk+0][tj+0][ti+0]) +
		       mux4 * (shm[2][tk+0][tj+0][ti+2] - shm[2][tk+0][tj+0][ti+0]));
      }
      // Y contribution
      {
	float_sw4 muy1, muy2, muy3, muy4;
	muy1 = mu_jm1 * stry_jm1 - tf * (mu_ijk * stry_j + mu_jm2 * stry_jm2);
	muy2 = mu_jm2 * stry_jm2 + mu_jp1 * stry_jp1 + 3 * (mu_ijk * stry_j   + mu_jm1 * stry_jm1);
	muy3 = mu_jm1 * stry_jm1 + mu_jp2 * stry_jp2 + 3 * (mu_jp1 * stry_jp1 + mu_ijk * stry_j  );
	muy4 = mu_jp1 * stry_jp1 - tf * (mu_ijk * stry_j + mu_jp2 * stry_jp2);

	r1 += stry_j * (muy1 * (shm[0][tk+0][tj-2][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muy2 * (shm[0][tk+0][tj-1][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muy3 * (shm[0][tk+0][tj+1][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muy4 * (shm[0][tk+0][tj+2][ti+0] - shm[0][tk+0][tj+0][ti+0]));
	r2 += stry_j * ((2 * muy1 + la_jm1 * stry_jm1 - tf * (la_ijk * stry_j + la_jm2 * stry_jm2)) *
			(shm[1][tk+0][tj-2][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			(2 * muy2 + la_jm2 * stry_jm2 + la_jp1 * stry_jp1 +
			 3 * (la_ijk * stry_j   + la_jm1 * stry_jm1)) *
			(shm[1][tk+0][tj-1][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			(2 * muy3 + la_jm1 * stry_jm1 + la_jp2 * stry_jp2 +
			 3 * (la_jp1 * stry_jp1 + la_ijk * stry_j  )) *
			(shm[1][tk+0][tj+1][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			(2 * muy4 + la_jp1 * stry_jp1 - tf * (la_ijk * stry_j  + la_jp2 * stry_jp2)) *
			(shm[1][tk+0][tj+2][ti+0] - shm[1][tk+0][tj+0][ti+0]));
	r3 += stry_j  *(muy1 * (shm[2][tk+0][tj-2][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			muy2 * (shm[2][tk+0][tj-1][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			muy3 * (shm[2][tk+0][tj+1][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			muy4 * (shm[2][tk+0][tj+2][ti+0] - shm[2][tk+0][tj+0][ti+0]));
      }
      // Z contribution
      {
	float_sw4 muz1, muz2, muz3, muz4;
	muz1 = mu_km1 * strz_km1 - tf * (mu_ijk * strz_k + mu_km2 * strz_km2);
	muz2 = mu_km2 * strz_km2 + mu_kp1 * strz_kp1 + 3 * (mu_ijk * strz_k   + mu_km1 * strz_km1);
	muz3 = mu_km1 * strz_km1 + mu_kp2 * strz_kp2 + 3 * (mu_kp1 * strz_kp1 + mu_ijk * strz_k);
	muz4 = mu_kp1 * strz_kp1 - tf * (mu_ijk * strz_k + mu_kp2 * strz_kp2);
	r1 += strz_k * (muz1 * (shm[0][tk-2][tj+0][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muz2 * (shm[0][tk-1][tj+0][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muz3 * (shm[0][tk+1][tj+0][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muz4 * (shm[0][tk+2][tj+0][ti+0] - shm[0][tk+0][tj+0][ti+0]));
	r2 += strz_k * (muz1 * (shm[1][tk-2][tj+0][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			muz2 * (shm[1][tk-1][tj+0][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			muz3 * (shm[1][tk+1][tj+0][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			muz4 * (shm[1][tk+2][tj+0][ti+0] - shm[1][tk+0][tj+0][ti+0]));
	r3 += strz_k * ((2 * muz1 + la_km1 * strz_km1 - tf * (la_ijk * strz_k + la_km2 * strz_km2)) *
			(shm[2][tk-2][tj+0][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			(2 * muz2 + la_km2 * strz_km2 + la_kp1 * strz_kp1 +
			 3 * (la_ijk * strz_k   + la_km1 * strz_km1)) *
			(shm[2][tk-1][tj+0][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			(2 * muz3 + la_km1 * strz_km1 + la_kp2 * strz_kp2 +
			 3 * (la_kp1 * strz_kp1 + la_ijk * strz_k  )) *
			(shm[2][tk+1][tj+0][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			(2 * muz4 + la_kp1 * strz_kp1 - tf * (la_ijk * strz_k + la_kp2 * strz_kp2)) *
			(shm[2][tk+2][tj+0][ti+0] - shm[2][tk+0][tj+0][ti+0]));
      }
      r1 *= i6;
      r2 *= i6;
      r3 *= i6;


      /* Mixed derivatives: */
      /*   (la*v_y)_x */
      r1 += strx_i  * stry_j * i144 * 
	(la_im2 * (shm[1][tk+0][tj-2][ti-2] - shm[1][tk+0][tj+2][ti-2] + 8 *
		   (-shm[1][tk+0][tj-1][ti-2] + shm[1][tk+0][tj+1][ti-2])) - 8 *
	 (la_im1 * (shm[1][tk+0][tj-2][ti-1] - shm[1][tk+0][tj+2][ti-1] + 8 *
		   (-shm[1][tk+0][tj-1][ti-1] + shm[1][tk+0][tj+1][ti-1]))) + 8 *
	 (la_ip1 * (shm[1][tk+0][tj-2][ti+1] - shm[1][tk+0][tj+2][ti+1] + 8 *
		   (-shm[1][tk+0][tj-1][ti+1] + shm[1][tk+0][tj+1][ti+1]))) - 
	 (la_ip2 * (shm[1][tk+0][tj-2][ti+2] - shm[1][tk+0][tj+2][ti+2] + 8 *
		   (-shm[1][tk+0][tj-1][ti+2] + shm[1][tk+0][tj+1][ti+2]))))
	/*   (la*w_z)_x */
        + strx_i * strz_k * i144 *
	(la_im2 * (shm[2][tk-2][tj+0][ti-2] - shm[2][tk+2][tj+0][ti-2] + 8 *
		   (-shm[2][tk-1][tj+0][ti-2] + shm[2][tk+1][tj+0][ti-2])) - 8 *
	 (la_im1 * (shm[2][tk-2][tj+0][ti-1] - shm[2][tk+2][tj+0][ti-1] + 8 *
		   (-shm[2][tk-1][tj+0][ti-1] + shm[2][tk+1][tj+0][ti-1]))) + 8 *
	 (la_ip1 * (shm[2][tk-2][tj+0][ti+1] - shm[2][tk+2][tj+0][ti+1] + 8 *
		   (-shm[2][tk-1][tj+0][ti+1] + shm[2][tk+1][tj+0][ti+1]))) - 
	 (la_ip2 * (shm[2][tk-2][tj+0][ti+2] - shm[2][tk+2][tj+0][ti+2] + 8 *
		   (-shm[2][tk-1][tj+0][ti+2] + shm[2][tk+1][tj+0][ti+2]))))
	/*   (mu*v_x)_y */
        + strx_i * stry_j * i144 *
	(mu_jm2 * (shm[1][tk+0][tj-2][ti-2] - shm[1][tk+0][tj-2][ti+2] + 8 *
		   (-shm[1][tk+0][tj-2][ti-1] + shm[1][tk+0][tj-2][ti+1])) - 8 *
	 (mu_jm1 * (shm[1][tk+0][tj-1][ti-2] - shm[1][tk+0][tj-1][ti+2] + 8 *
		   (-shm[1][tk+0][tj-1][ti-1] + shm[1][tk+0][tj-1][ti+1]))) + 8 *
	 (mu_jp1 * (shm[1][tk+0][tj+1][ti-2] - shm[1][tk+0][tj+1][ti+2] + 8 *
		   (-shm[1][tk+0][tj+1][ti-1] + shm[1][tk+0][tj+1][ti+1]))) - 
	 (mu_jp2 * (shm[1][tk+0][tj+2][ti-2] - shm[1][tk+0][tj+2][ti+2] + 8 *
		   (-shm[1][tk+0][tj+2][ti-1] + shm[1][tk+0][tj+2][ti+1]))))
	/*   (mu*w_x)_z */
        + strx_i * strz_k * i144 *
	(mu_km2 * (shm[2][tk-2][tj+0][ti-2] - shm[2][tk-2][tj+0][ti+2] + 8 *
		   (-shm[2][tk-2][tj+0][ti-1] + shm[2][tk-2][tj+0][ti+1])) - 8 *
 	 (mu_km1 * (shm[2][tk-1][tj+0][ti-2] - shm[2][tk-1][tj+0][ti+2] + 8 *
		   (-shm[2][tk-1][tj+0][ti-1] + shm[2][tk-1][tj+0][ti+1]))) + 8 *
 	 (mu_kp1 * (shm[2][tk+1][tj+0][ti-2] - shm[2][tk+1][tj+0][ti+2] + 8 *
		   (-shm[2][tk+1][tj+0][ti-1] + shm[2][tk+1][tj+0][ti+1]))) -
	 (mu_kp2 * (shm[2][tk+2][tj+0][ti-2] - shm[2][tk+2][tj+0][ti+2] + 8 *
		   (-shm[2][tk+2][tj+0][ti-1] + shm[2][tk+2][tj+0][ti+1]))));

      /*   (mu*u_y)_x */
      r2 += strx_i  *stry_j  *i144*
	(mu_im2 * (shm[0][tk+0][tj-2][ti-2] - shm[0][tk+0][tj+2][ti-2] + 8 *
		   (-shm[0][tk+0][tj-1][ti-2] + shm[0][tk+0][tj+1][ti-2])) - 8 *
	 (mu_im1 * (shm[0][tk+0][tj-2][ti-1] - shm[0][tk+0][tj+2][ti-1] + 8 *
		   (-shm[0][tk+0][tj-1][ti-1] + shm[0][tk+0][tj+1][ti-1]))) + 8 *
	 (mu_ip1 * (shm[0][tk+0][tj-2][ti+1] - shm[0][tk+0][tj+2][ti+1] + 8 *
		   (-shm[0][tk+0][tj-1][ti+1] + shm[0][tk+0][tj+1][ti+1]))) -
	 (mu_ip2 * (shm[0][tk+0][tj-2][ti+2] - shm[0][tk+0][tj+2][ti+2] + 8 *
		   (-shm[0][tk+0][tj-1][ti+2] + shm[0][tk+0][tj+1][ti+2]))))
	/* (la*u_x)_y */
        + strx_i * stry_j * i144 *
	(la_jm2 * (shm[0][tk+0][tj-2][ti-2] - shm[0][tk+0][tj-2][ti+2] + 8 *
		   (-shm[0][tk+0][tj-2][ti-1] + shm[0][tk+0][tj-2][ti+1])) - 8 *
	 (la_jm1 * (shm[0][tk+0][tj-1][ti-2] - shm[0][tk+0][tj-1][ti+2] + 8 *
		   (-shm[0][tk+0][tj-1][ti-1] + shm[0][tk+0][tj-1][ti+1]))) + 8 *
	 (la_jp1 * (shm[0][tk+0][tj+1][ti-2] - shm[0][tk+0][tj+1][ti+2] + 8 *
		   (-shm[0][tk+0][tj+1][ti-1] + shm[0][tk+0][tj+1][ti+1]))) -
	 (la_jp2 * (shm[0][tk+0][tj+2][ti-2] - shm[0][tk+0][tj+2][ti+2] + 8 *
		   (-shm[0][tk+0][tj+2][ti-1] + shm[0][tk+0][tj+2][ti+1]))))
	/* (la*w_z)_y */
        + stry_j * strz_k * i144 *
	(la_jm2 * (shm[2][tk-2][tj-2][ti+0] - shm[2][tk+2][tj-2][ti+0] + 8 *
		   (-shm[2][tk-1][tj-2][ti+0] + shm[2][tk+1][tj-2][ti+0])) - 8 *
	 (la_jm1 * (shm[2][tk-2][tj-1][ti+0] - shm[2][tk+2][tj-1][ti+0] + 8 *
		   (-shm[2][tk-1][tj-1][ti+0] + shm[2][tk+1][tj-1][ti+0]))) + 8 *
	 (la_jp1 * (shm[2][tk-2][tj+1][ti+0] - shm[2][tk+2][tj+1][ti+0] + 8 *
		   (-shm[2][tk-1][tj+1][ti+0] + shm[2][tk+1][tj+1][ti+0]))) -
	 (la_jp2 * (shm[2][tk-2][tj+2][ti+0] - shm[2][tk+2][tj+2][ti+0] + 8 *
		   (-shm[2][tk-1][tj+2][ti+0] + shm[2][tk+1][tj+2][ti+0]))))
	/* (mu*w_y)_z */
        + stry_j * strz_k * i144 *
	(mu_km2 * (shm[2][tk-2][tj-2][ti+0] - shm[2][tk-2][tj+2][ti+0] + 8 *
		   (-shm[2][tk-2][tj-1][ti+0] + shm[2][tk-2][tj+1][ti+0])) - 8 *
	 (mu_km1 * (shm[2][tk-1][tj-2][ti+0] - shm[2][tk-1][tj+2][ti+0] + 8 *
		   (-shm[2][tk-1][tj-1][ti+0] + shm[2][tk-1][tj+1][ti+0]))) + 8 *
	 (mu_kp1 * (shm[2][tk+1][tj-2][ti+0] - shm[2][tk+1][tj+2][ti+0] + 8 *
		   (-shm[2][tk+1][tj-1][ti+0] + shm[2][tk+1][tj+1][ti+0]))) -
	 (mu_kp2 * (shm[2][tk+2][tj-2][ti+0] - shm[2][tk+2][tj+2][ti+0] + 8 *
		   (-shm[2][tk+2][tj-1][ti+0] + shm[2][tk+2][tj+1][ti+0]))));

      /*  (mu*u_z)_x */
      r3 += strx_i * strz_k * i144 *
	(mu_im2 * (shm[0][tk-2][tj+0][ti-2] - shm[0][tk+2][tj+0][ti-2] + 8 *
		   (-shm[0][tk-1][tj+0][ti-2] + shm[0][tk+1][tj+0][ti-2])) - 8 *
	 (mu_im1 * (shm[0][tk-2][tj+0][ti-1] - shm[0][tk+2][tj+0][ti-1] + 8 *
		   (-shm[0][tk-1][tj+0][ti-1] + shm[0][tk+1][tj+0][ti-1]))) + 8 *
	 (mu_ip1 * (shm[0][tk-2][tj+0][ti+1] - shm[0][tk+2][tj+0][ti+1] + 8 *
		   (-shm[0][tk-1][tj+0][ti+1] + shm[0][tk+1][tj+0][ti+1]))) -
	 (mu_ip2 * (shm[0][tk-2][tj+0][ti+2] - shm[0][tk+2][tj+0][ti+2] + 8 *
		   (-shm[0][tk-1][tj+0][ti+2] + shm[0][tk+1][tj+0][ti+2]))))
	/* (mu*v_z)_y */
        + stry_j  *strz_k*i144*
	(mu_jm2 * (shm[1][tk-2][tj-2][ti+0] - shm[1][tk+2][tj-2][ti+0] + 8 *
		   (- shm[1][tk-1][tj-2][ti+0]+shm[1][tk+1][tj-2][ti+0])) - 8 * 
	 (mu_jm1 * (shm[1][tk-2][tj-1][ti+0] - shm[1][tk+2][tj-1][ti+0] + 8 *
		   (- shm[1][tk-1][tj-1][ti+0]+shm[1][tk+1][tj-1][ti+0]))) + 8 * 
	 (mu_jp1 * (shm[1][tk-2][tj+1][ti+0] - shm[1][tk+2][tj+1][ti+0] + 8 *
		   (- shm[1][tk-1][tj+1][ti+0]+shm[1][tk+1][tj+1][ti+0]))) -
	 (mu_jp2 * (shm[1][tk-2][tj+2][ti+0] - shm[1][tk+2][tj+2][ti+0] + 8 *
		   (- shm[1][tk-1][tj+2][ti+0]+shm[1][tk+1][tj+2][ti+0]))))
	/*   (la*u_x)_z */
        + strx_i  *strz_k*i144*
	(la_km2*(shm[0][tk-2][tj+0][ti-2] - shm[0][tk-2][tj+0][ti+2] + 8 *
		   (- shm[0][tk-2][tj+0][ti-1]+shm[0][tk-2][tj+0][ti+1])) - 8*
	 (la_km1*(shm[0][tk-1][tj+0][ti-2] - shm[0][tk-1][tj+0][ti+2] + 8 *
		   (- shm[0][tk-1][tj+0][ti-1]+shm[0][tk-1][tj+0][ti+1]))) + 8*
	 (la_kp1*(shm[0][tk+1][tj+0][ti-2] - shm[0][tk+1][tj+0][ti+2] + 8 *
		   (- shm[0][tk+1][tj+0][ti-1]+shm[0][tk+1][tj+0][ti+1]))) -
	 (la_kp2*(shm[0][tk+2][tj+0][ti-2] - shm[0][tk+2][tj+0][ti+2] + 8 *
		   (- shm[0][tk+2][tj+0][ti-1]+shm[0][tk+2][tj+0][ti+1]))))
	/* (la*v_y)_z */
        + stry_j  *strz_k*i144*
	(la_km2*(shm[1][tk-2][tj-2][ti+0] - shm[1][tk-2][tj+2][ti+0] + 8 *
		   (- shm[1][tk-2][tj-1][ti+0]+shm[1][tk-2][tj+1][ti+0])) - 8*
	 (la_km1*(shm[1][tk-1][tj-2][ti+0] - shm[1][tk-1][tj+2][ti+0] + 8 *
		   (- shm[1][tk-1][tj-1][ti+0]+shm[1][tk-1][tj+1][ti+0]))) + 8*
	 (la_kp1*(shm[1][tk+1][tj-2][ti+0] - shm[1][tk+1][tj+2][ti+0] + 8 *
		   (- shm[1][tk+1][tj-1][ti+0]+shm[1][tk+1][tj+1][ti+0]))) -
	 (la_kp2*(shm[1][tk+2][tj-2][ti+0] - shm[1][tk+2][tj+2][ti+0] + 8 *
		   (- shm[1][tk+2][tj-1][ti+0]+shm[1][tk+2][tj+1][ti+0]))));

      // Use the result at point (i,j,k)
      if (pred) {
	// Apply predictor
	int idx = k * nij + j * ni + i;
	float_sw4 fact = dt*dt / a_rho[idx];
	up(0,idx) = 2 * shm[0][tk][tj][ti] - um(0,idx) + fact * (cof * r1 + fo(0,idx));
	up(1,idx) = 2 * shm[1][tk][tj][ti] - um(1,idx) + fact * (cof * r2 + fo(1,idx));
	up(2,idx) = 2 * shm[2][tk][tj][ti] - um(2,idx) + fact * (cof * r3 + fo(2,idx));
      }
      else {
	// Apply 4th order correction
	int idx = k * nij + j * ni + i;
	float_sw4 fact = dt*dt*dt*dt / (12 * a_rho[idx]);
	up(0,idx) += fact * (cof * r1 + fo(0,idx));
	up(1,idx) += fact * (cof * r2 + fo(1,idx));
	up(2,idx) += fact * (cof * r3 + fo(2,idx));
      }

    }  // End active
    
    if (iside == 0)
      __syncthreads();
    
  } // End side loop
}

// *************************************************************************************
// Kernel launcher

void rhs4_X_pred_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      int ni, int nj, int nk,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um,
		      float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo, 
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
		      float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream) {

  // Launch enough threads to cover [jfirst:jlast][kfirst:klast]
  int njcomp = jlast - jfirst + 1;
  int nkcomp = klast - kfirst + 1;

  // 384 threads as 2x16x12, they will also be used as 6x16x4 inside the kernel
  dim3 blocks = dim3(1, (njcomp+15)/16, (nkcomp+11)/12);
  dim3 threads = dim3(2, 16, 12);
  if (c_order)
    rhs4_X<1,1><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, a_um,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
  else
    rhs4_X<0,1><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, a_um,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
}

void rhs4_X_corr_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      int ni, int nj, int nk,
		      float_sw4* a_up, float_sw4* a_u,
		      float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo, 
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
		      float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream) {

  // Launch enough threads to cover [jfirst:jlast][kfirst:klast]
  int njcomp = jlast - jfirst + 1;
  int nkcomp = klast - kfirst + 1;

  // 384 threads as 2x16x12, they will also be used as 6x16x4 inside the kernel
  dim3 blocks = dim3(1, (njcomp+15)/16, (nkcomp+11)/12);
  dim3 threads = dim3(2, 16, 12);
  if (c_order)
    rhs4_X<1,0><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, NULL,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
  else
    rhs4_X<0,0><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, NULL,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
}

// *************************************************************************************
// *************************************************************************************
// *************************************************************************************
// RHS4 Y halos :
// --------------
// Compute RHS4 in the halos, for each side in the Y dimension.
// Use blocks of (BX,2,4) threads, and launch enough threads to cover the whole
// volume (0:ni-1, jfirst:jlast, kfirst:klast).
// The block of (BX,2,4) threads allows to load (BX+4,6,8) elements in shared memory
// relatively easily, while keeping enough blocks to load the GPU.

template <bool c_order, bool pred>
__launch_bounds__(BX*8)
  __global__ void rhs4_Y (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
			  int ni, int nj, int nk,
			  float_sw4* __restrict__ a_up,
			  const float_sw4* __restrict__ a_u,
			  const float_sw4* __restrict__ a_um,
			  const float_sw4* __restrict__ a_mu,
			  const float_sw4* __restrict__ a_lambda,
			  const float_sw4* __restrict__ a_rho,
			  const float_sw4* __restrict__ a_fo,
			  const float_sw4* __restrict__ a_strx,
			  const float_sw4* __restrict__ a_stry,
			  const float_sw4* __restrict__ a_strz,
			  const float_sw4 h,
			  const float_sw4 dt) {
  int index, j;

  const float_sw4 i6   = 1.0/6;
  const float_sw4 i144 = 1.0/144;
  const float_sw4 tf   = 0.75;
  
  float_sw4 strx_im2, strx_im1, strx_i, strx_ip1, strx_ip2;
  float_sw4 stry_jm2, stry_jm1, stry_j, stry_jp1, stry_jp2;
  float_sw4 strz_km2, strz_km1, strz_k, strz_kp1, strz_kp2;

  float_sw4 la_km2, la_km1, la_ijk, la_kp1, la_kp2;
  float_sw4 la_im2, la_im1, la_ip1, la_ip2;
  float_sw4 la_jm2, la_jm1, la_jp1, la_jp2;

  float_sw4 mu_km2, mu_km1, mu_ijk, mu_kp1, mu_kp2;
  float_sw4 mu_im2, mu_im1, mu_ip1, mu_ip2;
  float_sw4 mu_jm2, mu_jm1, mu_jp1, mu_jp2;

  float_sw4 r1, r2, r3;
  int active=0, loader=0, loadx2=0, loadk2=0;

  __shared__ float_sw4 shm[3][8][6][BX+4];

  float_sw4 cof = 1.0 / (h * h);

  // I starts at 0 and covers the whole dimension
  const int i = threadIdx.x + blockIdx.x * BX;
  // K starts at kfirst
  const int k = kfirst + threadIdx.z + blockIdx.z * 4;

  const int ti = threadIdx.x + 2;
  const int tj = threadIdx.y + 2;
  const int tk = threadIdx.z + 2;
  
  const int nij = ni * nj;
  const int nijk = nij * nk;

  // Active threads doing the computation at (i+2,j,k)
  if (i+2 >= ifirst && i+2 <= ilast && k <= klast)
    active = 1;

  if (i < ni && k-2 <= klast+2) {
    loader = 1;
    if (threadIdx.x < 4 && i+BX < ni)
      loadx2 = 1;
    if (k <= klast) // (k+4-2 <= klast+2)
      loadk2 = 1;
  }

  if (active) {
    // Load strx & strz, centered at i+2 and k
    strx_im2 = a_strx[i];
    strx_im1 = a_strx[i+1];
    strx_i   = a_strx[i+2];
    strx_ip1 = a_strx[i+3];
    strx_ip2 = a_strx[i+4];

    strz_km2 = a_strz[k-2];
    strz_km1 = a_strz[k-1];
    strz_k   = a_strz[k];
    strz_kp1 = a_strz[k+1];
    strz_kp2 = a_strz[k+2];
  }

  // Loop on the 2 sides
#pragma unroll
  for (int iside=0; iside < 2; iside++) {

    if (iside == 0)
      j = jfirst + threadIdx.y;
    else
      j = jlast - 1 + threadIdx.y;

    index = (k - 2) * nij + (j - 2) * ni + i;

    // Load lambda and mu in the shared memory
    if (loader) {
      int idx = index;
      shm[0][threadIdx.z][threadIdx.y][threadIdx.x] = a_lambda[idx];
      shm[1][threadIdx.z][threadIdx.y][threadIdx.x] = a_mu[idx];
      idx += 2 * ni;
      shm[0][threadIdx.z][threadIdx.y+2][threadIdx.x] = a_lambda[idx];
      shm[1][threadIdx.z][threadIdx.y+2][threadIdx.x] = a_mu[idx];
      idx += 2 * ni;
      shm[0][threadIdx.z][threadIdx.y+4][threadIdx.x] = a_lambda[idx];
      shm[1][threadIdx.z][threadIdx.y+4][threadIdx.x] = a_mu[idx];
    }
    if (loadx2) {
      int idx = index + BX;
      shm[0][threadIdx.z][threadIdx.y][threadIdx.x+BX] = a_lambda[idx];
      shm[1][threadIdx.z][threadIdx.y][threadIdx.x+BX] = a_mu[idx];
      idx += 2 * ni;
      shm[0][threadIdx.z][threadIdx.y+2][threadIdx.x+BX] = a_lambda[idx];
      shm[1][threadIdx.z][threadIdx.y+2][threadIdx.x+BX] = a_mu[idx];
      idx += 2 * ni;
      shm[0][threadIdx.z][threadIdx.y+4][threadIdx.x+BX] = a_lambda[idx];
      shm[1][threadIdx.z][threadIdx.y+4][threadIdx.x+BX] = a_mu[idx];
    }
    if (loadk2) {
      int idx = index + 4 * nij;
      shm[0][threadIdx.z+4][threadIdx.y][threadIdx.x] = a_lambda[idx];
      shm[1][threadIdx.z+4][threadIdx.y][threadIdx.x] = a_mu[idx];
      idx += 2 * ni;
      shm[0][threadIdx.z+4][threadIdx.y+2][threadIdx.x] = a_lambda[idx];
      shm[1][threadIdx.z+4][threadIdx.y+2][threadIdx.x] = a_mu[idx];
      idx += 2 * ni;
      shm[0][threadIdx.z+4][threadIdx.y+4][threadIdx.x] = a_lambda[idx];
      shm[1][threadIdx.z+4][threadIdx.y+4][threadIdx.x] = a_mu[idx];
    }
    if (loadk2 && loadx2) {
      int idx = index + 4 * nij + BX;
      shm[0][threadIdx.z+4][threadIdx.y][threadIdx.x+BX] = a_lambda[idx];
      shm[1][threadIdx.z+4][threadIdx.y][threadIdx.x+BX] = a_mu[idx];
      idx += 2 * ni;
      shm[0][threadIdx.z+4][threadIdx.y+2][threadIdx.x+BX] = a_lambda[idx];
      shm[1][threadIdx.z+4][threadIdx.y+2][threadIdx.x+BX] = a_mu[idx];
      idx += 2 * ni;
      shm[0][threadIdx.z+4][threadIdx.y+4][threadIdx.x+BX] = a_lambda[idx];
      shm[1][threadIdx.z+4][threadIdx.y+4][threadIdx.x+BX] = a_mu[idx];
    }
    __syncthreads();
    // Put lambda and mu in registers, using the natural thread mapping
    la_km2 = shm[0][tk-2][tj][ti];
    la_km1 = shm[0][tk-1][tj][ti];
    la_ijk = shm[0][tk  ][tj][ti];
    la_kp1 = shm[0][tk+1][tj][ti];
    la_kp2 = shm[0][tk+2][tj][ti];

    la_jm2 = shm[0][tk][tj-2][ti];
    la_jm1 = shm[0][tk][tj-1][ti];
    la_jp1 = shm[0][tk][tj+1][ti];
    la_jp2 = shm[0][tk][tj+2][ti];

    la_im2 = shm[0][tk][tj][ti-2];
    la_im1 = shm[0][tk][tj][ti-1];
    la_ip1 = shm[0][tk][tj][ti+1];
    la_ip2 = shm[0][tk][tj][ti+2];

    mu_km2 = shm[1][tk-2][tj][ti];
    mu_km1 = shm[1][tk-1][tj][ti];
    mu_ijk = shm[1][tk  ][tj][ti];
    mu_kp1 = shm[1][tk+1][tj][ti];
    mu_kp2 = shm[1][tk+2][tj][ti];

    mu_jm2 = shm[1][tk][tj-2][ti];
    mu_jm1 = shm[1][tk][tj-1][ti];
    mu_jp1 = shm[1][tk][tj+1][ti];
    mu_jp2 = shm[1][tk][tj+2][ti];

    mu_im2 = shm[1][tk][tj][ti-2];
    mu_im1 = shm[1][tk][tj][ti-1];
    mu_ip1 = shm[1][tk][tj][ti+1];
    mu_ip2 = shm[1][tk][tj][ti+2];
    __syncthreads();
    
    // Load U in the shared memory
    if (loader) {
      int idx = index;
      shm[0][threadIdx.z][threadIdx.y][threadIdx.x] = u(0,idx);
      shm[1][threadIdx.z][threadIdx.y][threadIdx.x] = u(1,idx);
      shm[2][threadIdx.z][threadIdx.y][threadIdx.x] = u(2,idx);
      idx += 2 * ni;
      shm[0][threadIdx.z][threadIdx.y+2][threadIdx.x] = u(0,idx);
      shm[1][threadIdx.z][threadIdx.y+2][threadIdx.x] = u(1,idx);
      shm[2][threadIdx.z][threadIdx.y+2][threadIdx.x] = u(2,idx);
      idx += 2 * ni;
      shm[0][threadIdx.z][threadIdx.y+4][threadIdx.x] = u(0,idx);
      shm[1][threadIdx.z][threadIdx.y+4][threadIdx.x] = u(1,idx);
      shm[2][threadIdx.z][threadIdx.y+4][threadIdx.x] = u(2,idx);
    }
    if (loadx2) {
      int idx = index + BX;
      shm[0][threadIdx.z][threadIdx.y][threadIdx.x+BX] = u(0,idx);
      shm[1][threadIdx.z][threadIdx.y][threadIdx.x+BX] = u(1,idx);
      shm[2][threadIdx.z][threadIdx.y][threadIdx.x+BX] = u(2,idx);
      idx += 2 * ni;
      shm[0][threadIdx.z][threadIdx.y+2][threadIdx.x+BX] = u(0,idx);
      shm[1][threadIdx.z][threadIdx.y+2][threadIdx.x+BX] = u(1,idx);
      shm[2][threadIdx.z][threadIdx.y+2][threadIdx.x+BX] = u(2,idx);
      idx += 2 * ni;
      shm[0][threadIdx.z][threadIdx.y+4][threadIdx.x+BX] = u(0,idx);
      shm[1][threadIdx.z][threadIdx.y+4][threadIdx.x+BX] = u(1,idx);
      shm[2][threadIdx.z][threadIdx.y+4][threadIdx.x+BX] = u(2,idx);
    }
    if (loadk2) {
      int idx = index + 4 * nij;
      shm[0][threadIdx.z+4][threadIdx.y][threadIdx.x] = u(0,idx);
      shm[1][threadIdx.z+4][threadIdx.y][threadIdx.x] = u(1,idx);
      shm[2][threadIdx.z+4][threadIdx.y][threadIdx.x] = u(2,idx);
      idx += 2 * ni;
      shm[0][threadIdx.z+4][threadIdx.y+2][threadIdx.x] = u(0,idx);
      shm[1][threadIdx.z+4][threadIdx.y+2][threadIdx.x] = u(1,idx);
      shm[2][threadIdx.z+4][threadIdx.y+2][threadIdx.x] = u(2,idx);
      idx += 2 * ni;
      shm[0][threadIdx.z+4][threadIdx.y+4][threadIdx.x] = u(0,idx);
      shm[1][threadIdx.z+4][threadIdx.y+4][threadIdx.x] = u(1,idx);
      shm[2][threadIdx.z+4][threadIdx.y+4][threadIdx.x] = u(2,idx);
    }
    if (loadk2 && loadx2) {
      int idx = index + 4 * nij + BX;
      shm[0][threadIdx.z+4][threadIdx.y][threadIdx.x+BX] = u(0,idx);
      shm[1][threadIdx.z+4][threadIdx.y][threadIdx.x+BX] = u(1,idx);
      shm[2][threadIdx.z+4][threadIdx.y][threadIdx.x+BX] = u(2,idx);
      idx += 2 * ni;
      shm[0][threadIdx.z+4][threadIdx.y+2][threadIdx.x+BX] = u(0,idx);
      shm[1][threadIdx.z+4][threadIdx.y+2][threadIdx.x+BX] = u(1,idx);
      shm[2][threadIdx.z+4][threadIdx.y+2][threadIdx.x+BX] = u(2,idx);
      idx += 2 * ni;
      shm[0][threadIdx.z+4][threadIdx.y+4][threadIdx.x+BX] = u(0,idx);
      shm[1][threadIdx.z+4][threadIdx.y+4][threadIdx.x+BX] = u(1,idx);
      shm[2][threadIdx.z+4][threadIdx.y+4][threadIdx.x+BX] = u(2,idx);
    }
    __syncthreads();

    if (active) {

      // Load stry, centered at j
      stry_jm2 = a_stry[j-2];
      stry_jm1 = a_stry[j-1];
      stry_j   = a_stry[j];
      stry_jp1 = a_stry[j+1];
      stry_jp2 = a_stry[j+2];

      // X contribution
      {
	float_sw4 mux1, mux2, mux3, mux4;
	mux1 = mu_im1 * strx_im1 - tf * (mu_ijk * strx_i + mu_im2 * strx_im2);
	mux2 = mu_im2 * strx_im2 + mu_ip1 * strx_ip1 + 3 * (mu_ijk * strx_i   + mu_im1 * strx_im1);
	mux3 = mu_im1 * strx_im1 + mu_ip2 * strx_ip2 + 3 * (mu_ip1 * strx_ip1 + mu_ijk * strx_i  );
	mux4 = mu_ip1 * strx_ip1 - tf * (mu_ijk * strx_i + mu_ip2 * strx_ip2);

	r1 = strx_i * ((2 * mux1 + la_im1 * strx_im1 - tf * (la_ijk * strx_i + la_im2 * strx_im2)) *
		       (shm[0][tk+0][tj+0][ti-2] - shm[0][tk+0][tj+0][ti+0])+
		       (2 * mux2 + la_im2 * strx_im2 + la_ip1 * strx_ip1 +
			3 * (la_ijk * strx_i + la_im1 * strx_im1)) *
		       (shm[0][tk+0][tj+0][ti-1] - shm[0][tk+0][tj+0][ti+0]) +
		       (2 * mux3 + la_im1 * strx_im1 + la_ip2 * strx_ip2 +
			3 * (la_ip1 * strx_ip1 + la_ijk * strx_i)) *
		       (shm[0][tk+0][tj+0][ti+1] - shm[0][tk+0][tj+0][ti+0]) +
		       (2 * mux4 + la_ip1 * strx_ip1 - tf * (la_ijk * strx_i + la_ip2 * strx_ip2)) *
		       (shm[0][tk+0][tj+0][ti+2] - shm[0][tk+0][tj+0][ti+0]));
	r2 = strx_i * (mux1 * (shm[1][tk+0][tj+0][ti-2] - shm[1][tk+0][tj+0][ti+0]) +
		       mux2 * (shm[1][tk+0][tj+0][ti-1] - shm[1][tk+0][tj+0][ti+0]) +
		       mux3 * (shm[1][tk+0][tj+0][ti+1] - shm[1][tk+0][tj+0][ti+0]) +
		       mux4 * (shm[1][tk+0][tj+0][ti+2] - shm[1][tk+0][tj+0][ti+0]));
	r3 = strx_i * (mux1 * (shm[2][tk+0][tj+0][ti-2] - shm[2][tk+0][tj+0][ti+0]) +
		       mux2 * (shm[2][tk+0][tj+0][ti-1] - shm[2][tk+0][tj+0][ti+0]) +
		       mux3 * (shm[2][tk+0][tj+0][ti+1] - shm[2][tk+0][tj+0][ti+0]) +
		       mux4 * (shm[2][tk+0][tj+0][ti+2] - shm[2][tk+0][tj+0][ti+0]));
      }
      // Y contribution
      {
	float_sw4 muy1, muy2, muy3, muy4;
	muy1 = mu_jm1 * stry_jm1 - tf * (mu_ijk * stry_j + mu_jm2 * stry_jm2);
	muy2 = mu_jm2 * stry_jm2 + mu_jp1 * stry_jp1 + 3 * (mu_ijk * stry_j   + mu_jm1 * stry_jm1);
	muy3 = mu_jm1 * stry_jm1 + mu_jp2 * stry_jp2 + 3 * (mu_jp1 * stry_jp1 + mu_ijk * stry_j  );
	muy4 = mu_jp1 * stry_jp1 - tf * (mu_ijk * stry_j + mu_jp2 * stry_jp2);

	r1 += stry_j * (muy1 * (shm[0][tk+0][tj-2][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muy2 * (shm[0][tk+0][tj-1][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muy3 * (shm[0][tk+0][tj+1][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muy4 * (shm[0][tk+0][tj+2][ti+0] - shm[0][tk+0][tj+0][ti+0]));
	r2 += stry_j * ((2 * muy1 + la_jm1 * stry_jm1 - tf * (la_ijk * stry_j + la_jm2 * stry_jm2)) *
			(shm[1][tk+0][tj-2][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			(2 * muy2 + la_jm2 * stry_jm2 + la_jp1 * stry_jp1 +
			 3 * (la_ijk * stry_j   + la_jm1 * stry_jm1)) *
			(shm[1][tk+0][tj-1][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			(2 * muy3 + la_jm1 * stry_jm1 + la_jp2 * stry_jp2 +
			 3 * (la_jp1 * stry_jp1 + la_ijk * stry_j  )) *
			(shm[1][tk+0][tj+1][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			(2 * muy4 + la_jp1 * stry_jp1 - tf * (la_ijk * stry_j  + la_jp2 * stry_jp2)) *
			(shm[1][tk+0][tj+2][ti+0] - shm[1][tk+0][tj+0][ti+0]));
	r3 += stry_j  *(muy1 * (shm[2][tk+0][tj-2][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			muy2 * (shm[2][tk+0][tj-1][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			muy3 * (shm[2][tk+0][tj+1][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			muy4 * (shm[2][tk+0][tj+2][ti+0] - shm[2][tk+0][tj+0][ti+0]));
      }
      // Z contribution
      {
	float_sw4 muz1, muz2, muz3, muz4;
	muz1 = mu_km1 * strz_km1 - tf * (mu_ijk * strz_k + mu_km2 * strz_km2);
	muz2 = mu_km2 * strz_km2 + mu_kp1 * strz_kp1 + 3 * (mu_ijk * strz_k   + mu_km1 * strz_km1);
	muz3 = mu_km1 * strz_km1 + mu_kp2 * strz_kp2 + 3 * (mu_kp1 * strz_kp1 + mu_ijk * strz_k);
	muz4 = mu_kp1 * strz_kp1 - tf * (mu_ijk * strz_k + mu_kp2 * strz_kp2);
	r1 += strz_k * (muz1 * (shm[0][tk-2][tj+0][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muz2 * (shm[0][tk-1][tj+0][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muz3 * (shm[0][tk+1][tj+0][ti+0] - shm[0][tk+0][tj+0][ti+0]) +
			muz4 * (shm[0][tk+2][tj+0][ti+0] - shm[0][tk+0][tj+0][ti+0]));
	r2 += strz_k * (muz1 * (shm[1][tk-2][tj+0][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			muz2 * (shm[1][tk-1][tj+0][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			muz3 * (shm[1][tk+1][tj+0][ti+0] - shm[1][tk+0][tj+0][ti+0]) +
			muz4 * (shm[1][tk+2][tj+0][ti+0] - shm[1][tk+0][tj+0][ti+0]));
	r3 += strz_k * ((2 * muz1 + la_km1 * strz_km1 - tf * (la_ijk * strz_k + la_km2 * strz_km2)) *
			(shm[2][tk-2][tj+0][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			(2 * muz2 + la_km2 * strz_km2 + la_kp1 * strz_kp1 +
			 3 * (la_ijk * strz_k   + la_km1 * strz_km1)) *
			(shm[2][tk-1][tj+0][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			(2 * muz3 + la_km1 * strz_km1 + la_kp2 * strz_kp2 +
			 3 * (la_kp1 * strz_kp1 + la_ijk * strz_k  )) *
			(shm[2][tk+1][tj+0][ti+0] - shm[2][tk+0][tj+0][ti+0]) +
			(2 * muz4 + la_kp1 * strz_kp1 - tf * (la_ijk * strz_k + la_kp2 * strz_kp2)) *
			(shm[2][tk+2][tj+0][ti+0] - shm[2][tk+0][tj+0][ti+0]));
      }
      r1 *= i6;
      r2 *= i6;
      r3 *= i6;


      /* Mixed derivatives: */
      /*   (la*v_y)_x */
      r1 += strx_i  * stry_j * i144 * 
	(la_im2 * (shm[1][tk+0][tj-2][ti-2] - shm[1][tk+0][tj+2][ti-2] + 8 *
		   (-shm[1][tk+0][tj-1][ti-2] + shm[1][tk+0][tj+1][ti-2])) - 8 *
	 (la_im1 * (shm[1][tk+0][tj-2][ti-1] - shm[1][tk+0][tj+2][ti-1] + 8 *
		   (-shm[1][tk+0][tj-1][ti-1] + shm[1][tk+0][tj+1][ti-1]))) + 8 *
	 (la_ip1 * (shm[1][tk+0][tj-2][ti+1] - shm[1][tk+0][tj+2][ti+1] + 8 *
		   (-shm[1][tk+0][tj-1][ti+1] + shm[1][tk+0][tj+1][ti+1]))) - 
	 (la_ip2 * (shm[1][tk+0][tj-2][ti+2] - shm[1][tk+0][tj+2][ti+2] + 8 *
		   (-shm[1][tk+0][tj-1][ti+2] + shm[1][tk+0][tj+1][ti+2]))))
	/*   (la*w_z)_x */
	+ strx_i * strz_k * i144 *
	(la_im2 * (shm[2][tk-2][tj+0][ti-2] - shm[2][tk+2][tj+0][ti-2] + 8 *
		   (-shm[2][tk-1][tj+0][ti-2] + shm[2][tk+1][tj+0][ti-2])) - 8 *
	 (la_im1 * (shm[2][tk-2][tj+0][ti-1] - shm[2][tk+2][tj+0][ti-1] + 8 *
		   (-shm[2][tk-1][tj+0][ti-1] + shm[2][tk+1][tj+0][ti-1]))) + 8 *
	 (la_ip1 * (shm[2][tk-2][tj+0][ti+1] - shm[2][tk+2][tj+0][ti+1] + 8 *
		   (-shm[2][tk-1][tj+0][ti+1] + shm[2][tk+1][tj+0][ti+1]))) - 
	 (la_ip2 * (shm[2][tk-2][tj+0][ti+2] - shm[2][tk+2][tj+0][ti+2] + 8 *
		   (-shm[2][tk-1][tj+0][ti+2] + shm[2][tk+1][tj+0][ti+2]))))
	/*   (mu*v_x)_y */
	+ strx_i * stry_j * i144 *
	(mu_jm2 * (shm[1][tk+0][tj-2][ti-2] - shm[1][tk+0][tj-2][ti+2] + 8 *
		   (-shm[1][tk+0][tj-2][ti-1] + shm[1][tk+0][tj-2][ti+1])) - 8 *
	 (mu_jm1 * (shm[1][tk+0][tj-1][ti-2] - shm[1][tk+0][tj-1][ti+2] + 8 *
		   (-shm[1][tk+0][tj-1][ti-1] + shm[1][tk+0][tj-1][ti+1]))) + 8 *
	 (mu_jp1 * (shm[1][tk+0][tj+1][ti-2] - shm[1][tk+0][tj+1][ti+2] + 8 *
		   (-shm[1][tk+0][tj+1][ti-1] + shm[1][tk+0][tj+1][ti+1]))) - 
	 (mu_jp2 * (shm[1][tk+0][tj+2][ti-2] - shm[1][tk+0][tj+2][ti+2] + 8 *
		   (-shm[1][tk+0][tj+2][ti-1] + shm[1][tk+0][tj+2][ti+1]))))
	/*   (mu*w_x)_z */
	+ strx_i * strz_k * i144 *
	(mu_km2 * (shm[2][tk-2][tj+0][ti-2] - shm[2][tk-2][tj+0][ti+2] + 8 *
		   (-shm[2][tk-2][tj+0][ti-1] + shm[2][tk-2][tj+0][ti+1])) - 8 *
	 (mu_km1 * (shm[2][tk-1][tj+0][ti-2] - shm[2][tk-1][tj+0][ti+2] + 8 *
		   (-shm[2][tk-1][tj+0][ti-1] + shm[2][tk-1][tj+0][ti+1]))) + 8 *
	 (mu_kp1 * (shm[2][tk+1][tj+0][ti-2] - shm[2][tk+1][tj+0][ti+2] + 8 *
		   (-shm[2][tk+1][tj+0][ti-1] + shm[2][tk+1][tj+0][ti+1]))) -
	 (mu_kp2 * (shm[2][tk+2][tj+0][ti-2] - shm[2][tk+2][tj+0][ti+2] + 8 *
		   (-shm[2][tk+2][tj+0][ti-1] + shm[2][tk+2][tj+0][ti+1]))));

      /*   (mu*u_y)_x */
      r2 += strx_i  *stry_j  *i144*
	(mu_im2 * (shm[0][tk+0][tj-2][ti-2] - shm[0][tk+0][tj+2][ti-2] + 8 *
		   (-shm[0][tk+0][tj-1][ti-2] + shm[0][tk+0][tj+1][ti-2])) - 8 *
	 (mu_im1 * (shm[0][tk+0][tj-2][ti-1] - shm[0][tk+0][tj+2][ti-1] + 8 *
		   (-shm[0][tk+0][tj-1][ti-1] + shm[0][tk+0][tj+1][ti-1]))) + 8 *
	 (mu_ip1 * (shm[0][tk+0][tj-2][ti+1] - shm[0][tk+0][tj+2][ti+1] + 8 *
		   (-shm[0][tk+0][tj-1][ti+1] + shm[0][tk+0][tj+1][ti+1]))) -
	 (mu_ip2 * (shm[0][tk+0][tj-2][ti+2] - shm[0][tk+0][tj+2][ti+2] + 8 *
		   (-shm[0][tk+0][tj-1][ti+2] + shm[0][tk+0][tj+1][ti+2]))))
	/* (la*u_x)_y */
	+ strx_i * stry_j * i144 *
	(la_jm2 * (shm[0][tk+0][tj-2][ti-2] - shm[0][tk+0][tj-2][ti+2] + 8 *
		   (-shm[0][tk+0][tj-2][ti-1] + shm[0][tk+0][tj-2][ti+1])) - 8 *
	 (la_jm1 * (shm[0][tk+0][tj-1][ti-2] - shm[0][tk+0][tj-1][ti+2] + 8 *
		   (-shm[0][tk+0][tj-1][ti-1] + shm[0][tk+0][tj-1][ti+1]))) + 8 *
	 (la_jp1 * (shm[0][tk+0][tj+1][ti-2] - shm[0][tk+0][tj+1][ti+2] + 8 *
		   (-shm[0][tk+0][tj+1][ti-1] + shm[0][tk+0][tj+1][ti+1]))) -
	 (la_jp2 * (shm[0][tk+0][tj+2][ti-2] - shm[0][tk+0][tj+2][ti+2] + 8 *
		   (-shm[0][tk+0][tj+2][ti-1] + shm[0][tk+0][tj+2][ti+1]))))
	/* (la*w_z)_y */
	+ stry_j * strz_k * i144 *
	(la_jm2 * (shm[2][tk-2][tj-2][ti+0] - shm[2][tk+2][tj-2][ti+0] + 8 *
		   (-shm[2][tk-1][tj-2][ti+0] + shm[2][tk+1][tj-2][ti+0])) - 8 *
	 (la_jm1 * (shm[2][tk-2][tj-1][ti+0] - shm[2][tk+2][tj-1][ti+0] + 8 *
		   (-shm[2][tk-1][tj-1][ti+0] + shm[2][tk+1][tj-1][ti+0]))) + 8 *
	 (la_jp1 * (shm[2][tk-2][tj+1][ti+0] - shm[2][tk+2][tj+1][ti+0] + 8 *
		   (-shm[2][tk-1][tj+1][ti+0] + shm[2][tk+1][tj+1][ti+0]))) -
	 (la_jp2 * (shm[2][tk-2][tj+2][ti+0] - shm[2][tk+2][tj+2][ti+0] + 8 *
		   (-shm[2][tk-1][tj+2][ti+0] + shm[2][tk+1][tj+2][ti+0]))))
	/* (mu*w_y)_z */
	+ stry_j * strz_k * i144 *
	(mu_km2 * (shm[2][tk-2][tj-2][ti+0] - shm[2][tk-2][tj+2][ti+0] + 8 *
		   (-shm[2][tk-2][tj-1][ti+0] + shm[2][tk-2][tj+1][ti+0])) - 8 *
	 (mu_km1 * (shm[2][tk-1][tj-2][ti+0] - shm[2][tk-1][tj+2][ti+0] + 8 *
		   (-shm[2][tk-1][tj-1][ti+0] + shm[2][tk-1][tj+1][ti+0]))) + 8 *
	 (mu_kp1 * (shm[2][tk+1][tj-2][ti+0] - shm[2][tk+1][tj+2][ti+0] + 8 *
		   (-shm[2][tk+1][tj-1][ti+0] + shm[2][tk+1][tj+1][ti+0]))) -
	 (mu_kp2 * (shm[2][tk+2][tj-2][ti+0] - shm[2][tk+2][tj+2][ti+0] + 8 *
		   (-shm[2][tk+2][tj-1][ti+0] + shm[2][tk+2][tj+1][ti+0]))));

      /*  (mu*u_z)_x */
      r3 += strx_i * strz_k * i144 *
	(mu_im2 * (shm[0][tk-2][tj+0][ti-2] - shm[0][tk+2][tj+0][ti-2] + 8 *
		   (-shm[0][tk-1][tj+0][ti-2] + shm[0][tk+1][tj+0][ti-2])) - 8 *
	 (mu_im1 * (shm[0][tk-2][tj+0][ti-1] - shm[0][tk+2][tj+0][ti-1] + 8 *
		   (-shm[0][tk-1][tj+0][ti-1] + shm[0][tk+1][tj+0][ti-1]))) + 8 *
	 (mu_ip1 * (shm[0][tk-2][tj+0][ti+1] - shm[0][tk+2][tj+0][ti+1] + 8 *
		   (-shm[0][tk-1][tj+0][ti+1] + shm[0][tk+1][tj+0][ti+1]))) -
	 (mu_ip2 * (shm[0][tk-2][tj+0][ti+2] - shm[0][tk+2][tj+0][ti+2] + 8 *
		   (-shm[0][tk-1][tj+0][ti+2] + shm[0][tk+1][tj+0][ti+2]))))
	/* (mu*v_z)_y */
	+ stry_j  *strz_k*i144*
	(mu_jm2 * (shm[1][tk-2][tj-2][ti+0] - shm[1][tk+2][tj-2][ti+0] + 8 *
		   (- shm[1][tk-1][tj-2][ti+0]+shm[1][tk+1][tj-2][ti+0])) - 8 * 
	 (mu_jm1 * (shm[1][tk-2][tj-1][ti+0] - shm[1][tk+2][tj-1][ti+0] + 8 *
		   (- shm[1][tk-1][tj-1][ti+0]+shm[1][tk+1][tj-1][ti+0]))) + 8 * 
	 (mu_jp1 * (shm[1][tk-2][tj+1][ti+0] - shm[1][tk+2][tj+1][ti+0] + 8 *
		   (- shm[1][tk-1][tj+1][ti+0]+shm[1][tk+1][tj+1][ti+0]))) -
	 (mu_jp2 * (shm[1][tk-2][tj+2][ti+0] - shm[1][tk+2][tj+2][ti+0] + 8 *
		   (- shm[1][tk-1][tj+2][ti+0]+shm[1][tk+1][tj+2][ti+0]))))
	/*   (la*u_x)_z */
	+ strx_i  *strz_k*i144*
	(la_km2 * (shm[0][tk-2][tj+0][ti-2] - shm[0][tk-2][tj+0][ti+2] + 8 *
		   (- shm[0][tk-2][tj+0][ti-1]+shm[0][tk-2][tj+0][ti+1])) - 8*
	 (la_km1 * (shm[0][tk-1][tj+0][ti-2] - shm[0][tk-1][tj+0][ti+2] + 8 *
		   (- shm[0][tk-1][tj+0][ti-1]+shm[0][tk-1][tj+0][ti+1]))) + 8*
	 (la_kp1 * (shm[0][tk+1][tj+0][ti-2] - shm[0][tk+1][tj+0][ti+2] + 8 *
		   (- shm[0][tk+1][tj+0][ti-1]+shm[0][tk+1][tj+0][ti+1]))) -
	 (la_kp2 * (shm[0][tk+2][tj+0][ti-2] - shm[0][tk+2][tj+0][ti+2] + 8 *
		   (- shm[0][tk+2][tj+0][ti-1]+shm[0][tk+2][tj+0][ti+1]))))
	/* (la*v_y)_z */
	+ stry_j  *strz_k*i144*
	(la_km2 * (shm[1][tk-2][tj-2][ti+0] - shm[1][tk-2][tj+2][ti+0] + 8 *
		   (- shm[1][tk-2][tj-1][ti+0]+shm[1][tk-2][tj+1][ti+0])) - 8*
	 (la_km1 * (shm[1][tk-1][tj-2][ti+0] - shm[1][tk-1][tj+2][ti+0] + 8 *
		   (- shm[1][tk-1][tj-1][ti+0]+shm[1][tk-1][tj+1][ti+0]))) + 8*
	 (la_kp1 * (shm[1][tk+1][tj-2][ti+0] - shm[1][tk+1][tj+2][ti+0] + 8 *
		   (- shm[1][tk+1][tj-1][ti+0]+shm[1][tk+1][tj+1][ti+0]))) -
	 (la_kp2 * (shm[1][tk+2][tj-2][ti+0] - shm[1][tk+2][tj+2][ti+0] + 8 *
		   (- shm[1][tk+2][tj-1][ti+0]+shm[1][tk+2][tj+1][ti+0]))));

      // Use the result at point (i+2,j,k)
      if (pred) {
	// Apply predictor
	int idx = k * nij + j * ni + i + 2;
	float_sw4 fact = dt*dt / a_rho[idx];
	up(0,idx) = 2 * shm[0][tk][tj][ti] - um(0,idx) + fact * (cof * r1 + fo(0,idx));
	up(1,idx) = 2 * shm[1][tk][tj][ti] - um(1,idx) + fact * (cof * r2 + fo(1,idx));
	up(2,idx) = 2 * shm[2][tk][tj][ti] - um(2,idx) + fact * (cof * r3 + fo(2,idx));
      }
      else {
	// Apply 4th order correction
	int idx = k * nij + j * ni + i + 2;
	float_sw4 fact = dt*dt*dt*dt / (12 * a_rho[idx]);
	up(0,idx) += fact * (cof * r1 + fo(0,idx));
	up(1,idx) += fact * (cof * r2 + fo(1,idx));
	up(2,idx) += fact * (cof * r3 + fo(2,idx));
      }

    }  // End active

    if (iside == 0)
      __syncthreads();

  } // End side loop
}

// *************************************************************************************
// Kernel launcher

void rhs4_Y_pred_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      int ni, int nj, int nk,
		      float_sw4* a_up, float_sw4* a_u, float_sw4* a_um,
		      float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
		      float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream) {

  // Launch enough threads to cover [kfirst:klast]
  int nkcomp = klast - kfirst + 1;
  dim3 blocks = dim3((ni+BX-1)/BX, 1, (nkcomp+3)/4);
  dim3 threads = dim3(BX, 2, 4);
  if (c_order)
    rhs4_Y<1,1><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, a_um,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
  else
    rhs4_Y<0,1><<<blocks,threads,0,stream>>> 
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, a_um,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
}

void rhs4_Y_corr_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		      int ni, int nj, int nk,
		      float_sw4* a_up, float_sw4* a_u,
		      float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
		      float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz, 
		      float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream) {

  // Launch enough threads to cover [kfirst:klast]
  int nkcomp = klast - kfirst + 1;
  dim3 blocks = dim3((ni+BX-1)/BX, 1, (nkcomp+3)/4);
  dim3 threads = dim3(BX, 2, 4);
  if (c_order)
    rhs4_Y<1,0><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, NULL,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
  else
    rhs4_Y<0,0><<<blocks,threads,0,stream>>> 
      (ifirst, ilast, jfirst, jlast, kfirst, klast,
       ni, nj, nk,
       a_up, a_u, NULL,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, a_strz, h, dt);
}

// *************************************************************************************
// *************************************************************************************
// *************************************************************************************
// RHS at the surface (free surface)
// Launch UXx2x6 = 192 threads to compute the first 6 plans in the Z dimension.
// Strategy : Load the first 10 depth plans of U, lambda, mu in shared memory,
// and compute 6 output plans.
// low specifies low-K boundary, otherwise for high-K boundary
 
template <bool c_order, bool pred, bool low>
__launch_bounds__ (UX*12)
__global__ void rhs4_Kboundary (int ifirst, int ilast, int jfirst, int jlast,
			   int ni, int nj, int nk, int nz,
			   float_sw4* __restrict__ a_up,
			   const float_sw4* __restrict__ a_u,
			   const float_sw4* __restrict__ a_um,
			   const float_sw4* __restrict__ a_mu,
			   const float_sw4* __restrict__ a_lambda, 
			   const float_sw4* __restrict__ a_rho,
			   const float_sw4* __restrict__ a_fo,
			   const float_sw4* __restrict__ a_strx,
			   const float_sw4* __restrict__ a_stry,
			   const float_sw4 h,
			   const float_sw4 dt) {
  int index;
  const float_sw4 i6   = 1.0 / 6;
  const float_sw4 i12  = 1.0 / 12;
  const float_sw4 i144 = 1.0 / 144;
  const float_sw4 tf   = 0.75;

  float_sw4 mux1, mux2, mux3, mux4, muy1, muy2, muy3, muy4;
  float_sw4 r1, r2, r3, cof, mu1zz, mu2zz, mu3zz;
  float_sw4 u3zip2, u3zip1, u3zim1, u3zim2, lau3zx, mu3xz, u3zjp2, u3zjp1, u3zjm1, u3zjm2;
  float_sw4 lau3zy, mu3yz, mu1zx, mu2zy, u1zip2, u1zip1, u1zim1, u1zim2;
  float_sw4 u2zjp2, u2zjp1, u2zjm1, u2zjm2, lau1xz, lau2yz;

  float_sw4 strx_im2, strx_im1, strx_i, strx_ip1, strx_ip2;
  float_sw4 stry_jm2, stry_jm1, stry_j, stry_jp1, stry_jp2;
  
  int active=0, loadx1=0, loadx2=0, loady1=0, loady2=0, loady3=0;
  
  __shared__ float_sw4 shu[3][8][2+4][UX+4];
  __shared__ float_sw4 shl[8][2+4][UX+4];
  __shared__ float_sw4 shm[8][2+4][UX+4];

  // i starts at 0, j starts at jfirst, k starts at 0.
  const int i = threadIdx.x + blockIdx.x * UX;
  const int j = jfirst + threadIdx.y + blockIdx.y * 2;
  int k;

  if(low)
    k = threadIdx.z;
  else
    k = nz-6; // (nz-5) - (-1) - 2
  
  const int ti = threadIdx.x + 2;
  const int tj = threadIdx.y + 2;
  const int tk = threadIdx.z;

  const int nij = ni * nj;
  const int nijk = nij * nk;
   
  // Active threads doing the computation at (i,j,k+2)
  if (i >= ifirst && i <= ilast && j <= jlast)
    active = 1;

  // Define which threads can load the data in shared memory
  if (i-2 >= 0 && i-2 < ni)
    loadx1 = 1;
  if (threadIdx.x < 4 && i-2+UX < ni)
    loadx2 = 1;
  if (j-2 < nj)
    loady1 = 1;
  if (j < nj)
    loady2 = 1;
  if (j+2 < nj)
    loady3 = 1;

  float_sw4 u0, u1, u2;
  if(low)
  {
    index = 1 * nij + j * ni + i;
    u0 = u(0,index);
    u1 = u(1,index);
    u2 = u(2,index);
  }
  else
  {
    index = (nz+1) * nij + j * ni + i;
    u0 = u(0,index);
    u1 = u(1,index);
    u2 = u(2,index);
  }
  // Loading index starts at (i-2,j-2,k)
  index = (k + 2) * nij + (j - 2) * ni + i - 2;

  // Load u, lambda and mu in the shared memory. 2 zones(X) x 3 zones(Y) x 2 zones(Z) = 12 zones
  if (loady1) {
    int idx = index;
    if (loadx1) {
      shu[0][threadIdx.z][threadIdx.y][threadIdx.x] = u(0,idx); 
      shu[1][threadIdx.z][threadIdx.y][threadIdx.x] = u(1,idx); 
      shu[2][threadIdx.z][threadIdx.y][threadIdx.x] = u(2,idx); 
      shl[threadIdx.z][threadIdx.y][threadIdx.x] = a_lambda[idx];
      shm[threadIdx.z][threadIdx.y][threadIdx.x] = a_mu[idx];
    }
    if (loadx2) {
      shu[0][threadIdx.z][threadIdx.y][threadIdx.x+UX] = u(0,idx+UX); 
      shu[1][threadIdx.z][threadIdx.y][threadIdx.x+UX] = u(1,idx+UX); 
      shu[2][threadIdx.z][threadIdx.y][threadIdx.x+UX] = u(2,idx+UX); 
      shl[threadIdx.z][threadIdx.y][threadIdx.x+UX] = a_lambda[idx+UX];
      shm[threadIdx.z][threadIdx.y][threadIdx.x+UX] = a_mu[idx+UX]; 
    }
    if (threadIdx.z < 2) {
      idx += 6 * nij;
      if (loadx1) {
	shu[0][threadIdx.z+6][threadIdx.y][threadIdx.x] = u(0,idx); 
	shu[1][threadIdx.z+6][threadIdx.y][threadIdx.x] = u(1,idx); 
	shu[2][threadIdx.z+6][threadIdx.y][threadIdx.x] = u(2,idx); 
	shl[threadIdx.z+6][threadIdx.y][threadIdx.x] = a_lambda[idx];
	shm[threadIdx.z+6][threadIdx.y][threadIdx.x] = a_mu[idx];
      }
      if (loadx2) {
	shu[0][threadIdx.z+6][threadIdx.y][threadIdx.x+UX] = u(0,idx+UX); 
	shu[1][threadIdx.z+6][threadIdx.y][threadIdx.x+UX] = u(1,idx+UX); 
	shu[2][threadIdx.z+6][threadIdx.y][threadIdx.x+UX] = u(2,idx+UX); 
	shl[threadIdx.z+6][threadIdx.y][threadIdx.x+UX] = a_lambda[idx+UX];
	shm[threadIdx.z+6][threadIdx.y][threadIdx.x+UX] = a_mu[idx+UX]; 
      }
    }
  }

  if (loady2) {
    int idx = index + 2 * ni;
    if (loadx1) {
      shu[0][threadIdx.z][threadIdx.y+2][threadIdx.x] = u(0,idx); 
      shu[1][threadIdx.z][threadIdx.y+2][threadIdx.x] = u(1,idx); 
      shu[2][threadIdx.z][threadIdx.y+2][threadIdx.x] = u(2,idx); 
      shl[threadIdx.z][threadIdx.y+2][threadIdx.x] = a_lambda[idx];
      shm[threadIdx.z][threadIdx.y+2][threadIdx.x] = a_mu[idx];
    }
    if (loadx2) {
      shu[0][threadIdx.z][threadIdx.y+2][threadIdx.x+UX] = u(0,idx+UX); 
      shu[1][threadIdx.z][threadIdx.y+2][threadIdx.x+UX] = u(1,idx+UX); 
      shu[2][threadIdx.z][threadIdx.y+2][threadIdx.x+UX] = u(2,idx+UX); 
      shl[threadIdx.z][threadIdx.y+2][threadIdx.x+UX] = a_lambda[idx+UX];
      shm[threadIdx.z][threadIdx.y+2][threadIdx.x+UX] = a_mu[idx+UX]; 
    }
    if (threadIdx.z < 2) {
      idx += 6 * nij;
      if (loadx1) {
	shu[0][threadIdx.z+6][threadIdx.y+2][threadIdx.x] = u(0,idx); 
	shu[1][threadIdx.z+6][threadIdx.y+2][threadIdx.x] = u(1,idx); 
	shu[2][threadIdx.z+6][threadIdx.y+2][threadIdx.x] = u(2,idx); 
	shl[threadIdx.z+6][threadIdx.y+2][threadIdx.x] = a_lambda[idx];
	shm[threadIdx.z+6][threadIdx.y+2][threadIdx.x] = a_mu[idx];
      }
      if (loadx2) {
	shu[0][threadIdx.z+6][threadIdx.y+2][threadIdx.x+UX] = u(0,idx+UX); 
	shu[1][threadIdx.z+6][threadIdx.y+2][threadIdx.x+UX] = u(1,idx+UX); 
	shu[2][threadIdx.z+6][threadIdx.y+2][threadIdx.x+UX] = u(2,idx+UX); 
	shl[threadIdx.z+6][threadIdx.y+2][threadIdx.x+UX] = a_lambda[idx+UX];
	shm[threadIdx.z+6][threadIdx.y+2][threadIdx.x+UX] = a_mu[idx+UX]; 
      }
    }
  }
      
  if (loady3) {
    int idx = index + 4 * ni;
    if (loadx1) {
      shu[0][threadIdx.z][threadIdx.y+4][threadIdx.x] = u(0,idx); 
      shu[1][threadIdx.z][threadIdx.y+4][threadIdx.x] = u(1,idx); 
      shu[2][threadIdx.z][threadIdx.y+4][threadIdx.x] = u(2,idx); 
      shl[threadIdx.z][threadIdx.y+4][threadIdx.x] = a_lambda[idx];
      shm[threadIdx.z][threadIdx.y+4][threadIdx.x] = a_mu[idx];
    }
    if (loadx2) {
      shu[0][threadIdx.z][threadIdx.y+4][threadIdx.x+UX] = u(0,idx+UX); 
      shu[1][threadIdx.z][threadIdx.y+4][threadIdx.x+UX] = u(1,idx+UX); 
      shu[2][threadIdx.z][threadIdx.y+4][threadIdx.x+UX] = u(2,idx+UX); 
      shl[threadIdx.z][threadIdx.y+4][threadIdx.x+UX] = a_lambda[idx+UX];
      shm[threadIdx.z][threadIdx.y+4][threadIdx.x+UX] = a_mu[idx+UX]; 
    }
    if (threadIdx.z < 2) {
      idx += 6 * nij;
      if (loadx1) {
	shu[0][threadIdx.z+6][threadIdx.y+4][threadIdx.x] = u(0,idx); 
	shu[1][threadIdx.z+6][threadIdx.y+4][threadIdx.x] = u(1,idx); 
	shu[2][threadIdx.z+6][threadIdx.y+4][threadIdx.x] = u(2,idx); 
	shl[threadIdx.z+6][threadIdx.y+4][threadIdx.x] = a_lambda[idx];
	shm[threadIdx.z+6][threadIdx.y+4][threadIdx.x] = a_mu[idx];
      }
      if (loadx2) {
	shu[0][threadIdx.z+6][threadIdx.y+4][threadIdx.x+UX] = u(0,idx+UX); 
	shu[1][threadIdx.z+6][threadIdx.y+4][threadIdx.x+UX] = u(1,idx+UX); 
	shu[2][threadIdx.z+6][threadIdx.y+4][threadIdx.x+UX] = u(2,idx+UX); 
	shl[threadIdx.z+6][threadIdx.y+4][threadIdx.x+UX] = a_lambda[idx+UX];
	shm[threadIdx.z+6][threadIdx.y+4][threadIdx.x+UX] = a_mu[idx+UX]; 
      }
    }
  }
  __syncthreads();

  if (active) {

   // Load strx & stry, centered at (i,j)
    strx_im2 = a_strx[i-2];
    strx_im1 = a_strx[i-1];
    strx_i   = a_strx[i];
    strx_ip1 = a_strx[i+1];
    strx_ip2 = a_strx[i+2];
    
    stry_jm2 = a_stry[j-2];
    stry_jm1 = a_stry[j-1];
    stry_j   = a_stry[j];
    stry_jp1 = a_stry[j+1];
    stry_jp2 = a_stry[j+2];

    cof = 1.0 / (h * h);
    mux1 = shm[tk][tj][ti-1] * strx_im1 - tf *
      (shm[tk][tj][ti] * strx_i + shm[tk][tj][ti-2] * strx_im2);
    mux2 = shm[tk][tj][ti-2] * strx_im2 + shm[tk][tj][ti+1] * strx_ip1 + 3 *
      (shm[tk][tj][ti] * strx_i + shm[tk][tj][ti-1] * strx_im1);
    mux3 = shm[tk][tj][ti-1] * strx_im1 + shm[tk][tj][ti+2] * strx_ip2 + 3 *
      (shm[tk][tj][ti+1] * strx_ip1 + shm[tk][tj][ti] * strx_i);
    mux4 = shm[tk][tj][ti+1] * strx_ip1 - tf *
      (shm[tk][tj][ti]* strx_i + shm[tk][tj][ti+2] * strx_ip2);
    
    muy1 = shm[tk][tj-1][ti] * stry_jm1 - tf *
      (shm[tk][tj][ti]* stry_j + shm[tk][tj-2][ti] * stry_jm2);
    muy2 = shm[tk][tj-2][ti] * stry_jm2 + shm[tk][tj+1][ti] * stry_jp1 + 3 *
      (shm[tk][tj][ti] * stry_j + shm[tk][tj-1][ti] * stry_jm1);
    muy3 = shm[tk][tj-1][ti] * stry_jm1 + shm[tk][tj+2][ti] * stry_jp2 + 3 *
      (shm[tk][tj+1][ti] * stry_jp1 + shm[tk][tj][ti] * stry_j);
    muy4 = shm[tk][tj+1][ti] * stry_jp1 - tf *
      (shm[tk][tj][ti]* stry_j + shm[tk][tj+2][ti] * stry_jp2);

    r1 = i6 * (strx_i *
	       ((2 * mux1 + shl[tk][tj][ti-1] * strx_im1 - tf *
		 (shl[tk][tj][ti] * strx_i + shl[tk][tj][ti-2] * strx_im2)) *
		(shu[0][tk][tj][ti-2] - shu[0][tk][tj][ti]) +
		(2 * mux2 + shl[tk][tj][ti-2] * strx_im2 + shl[tk][tj][ti+1] * strx_ip1 + 3 *
		 (shl[tk][tj][ti] * strx_i + shl[tk][tj][ti-1] * strx_im1)) *
		(shu[0][tk][tj][ti-1] - shu[0][tk][tj][ti]) + 
		(2 * mux3 + shl[tk][tj][ti-1] * strx_im1 + shl[tk][tj][ti+2] * strx_ip2 + 3 *
		 (shl[tk][tj][ti+1] * strx_ip1 + shl[tk][tj][ti] * strx_i)) *
		(shu[0][tk][tj][ti+1] - shu[0][tk][tj][ti]) +
                (2 * mux4 + shl[tk][tj][ti+1] * strx_ip1 - tf *
		 (shl[tk][tj][ti] * strx_i + shl[tk][tj][ti+2] * strx_ip2)) *
		(shu[0][tk][tj][ti+2] - shu[0][tk][tj][ti]) ) +
	       stry_j * (muy1 * (shu[0][tk][tj-2][ti] - shu[0][tk][tj][ti]) + 
			    muy2 * (shu[0][tk][tj-1][ti] - shu[0][tk][tj][ti]) + 
			    muy3 * (shu[0][tk][tj+1][ti] - shu[0][tk][tj][ti]) +
			    muy4 * (shu[0][tk][tj+2][ti] - shu[0][tk][tj][ti]) ) );
    
    /* (mu*uz)_z can not be centered */
    /* second derivative (mu*u_z)_z at grid point z_k */
    /* averaging the coefficient, */
    /* leave out the z-supergrid stretching strz, since it will */
    /* never be used together with the sbp-boundary operator */
    mu1zz = 0;
    mu2zz = 0;
    mu3zz = 0;
    for (int q=0; q<8; q++) {
      int qi = (low)? q: (7-q);
      float_sw4 lap2mu = 0;
      float_sw4 mucof  = 0;
      int idx = q * 6 + k;
      for (int m=0 ; m<8; m++) {
        int mi = (low)? m: (7-m);
	mucof  += dev_acof[idx] * shm[mi][tj][ti];
	lap2mu += dev_acof[idx] * (shl[mi][tj][ti] + 2 * shm[mi][tj][ti]);
	idx += 48;
      }
      mu1zz += mucof * shu[0][qi][tj][ti];
      mu2zz += mucof * shu[1][qi][tj][ti];
      mu3zz += lap2mu * shu[2][qi][tj][ti];
    }

    /* ghost point only influences the first point (k=1) because ghcof(k)=0 for k>=2*/
    //r1 += mu1zz + dev_ghcof[k] * shm[2][tj][ti] * shu[0][1][tj][ti];
    if(low)
      r1 += mu1zz + dev_ghcof[k] * shm[0][tj][ti] * u0; 
    else
      r1 += mu1zz + dev_ghcof[k] * shm[7][tj][ti] * u0; 
    
    r2 = i6 * (strx_i *
	       (mux1 * (shu[1][tk][tj][ti-2] - shu[1][tk][tj][ti]) + 
		mux2 * (shu[1][tk][tj][ti-1] - shu[1][tk][tj][ti]) + 
		mux3 * (shu[1][tk][tj][ti+1] - shu[1][tk][tj][ti]) +
		mux4 * (shu[1][tk][tj][ti+2] - shu[1][tk][tj][ti]) ) +
	       stry_j *
	       ((2 * muy1 + shl[tk][tj-1][ti] * stry_jm1 - tf *
		 (shl[tk][tj][ti] * stry_j + shl[tk][tj-2][ti] * stry_jm2)) *
		(shu[1][tk][tj-2][ti] - shu[1][tk][tj][ti]) +
		(2 * muy2 + shl[tk][tj-2][ti] * stry_jm2 + shl[tk][tj+1][ti] * stry_jp1 + 3 *
		 (shl[tk][tj][ti] * stry_j + shl[tk][tj-1][ti] * stry_jm1)) *
		(shu[1][tk][tj-1][ti] - shu[1][tk][tj][ti]) + 
		(2 * muy3 + shl[tk][tj-1][ti] * stry_jm1 + shl[tk][tj+2][ti] * stry_jp2 + 3 *
		 (shl[tk][tj+1][ti] * stry_jp1 + shl[tk][tj][ti] * stry_j)) *
		(shu[1][tk][tj+1][ti] - shu[1][tk][tj][ti]) +
		(2 * muy4 + shl[tk][tj+1][ti] * stry_jp1 - tf *
		 (shl[tk][tj][ti] * stry_j + shl[tk][tj+2][ti] * stry_jp2)) *
		(shu[1][tk][tj+2][ti] - shu[1][tk][tj][ti]) ) );

    /* ghost point only influences the first point (k=1) because ghcof(k)=0 for k>=2 */
    //r2 += mu2zz + dev_ghcof[k] * shm[2][tj][ti] * shu[1][1][tj][ti];
    if(low)
      r2 += mu2zz + dev_ghcof[k] * shm[0][tj][ti] * u1; 
    else
      r2 += mu2zz + dev_ghcof[k] * shm[7][tj][ti] * u1; 


    r3 = i6 * (strx_i * (mux1 * (shu[2][tk][tj][ti-2] - shu[2][tk][tj][ti]) + 
			    mux2 * (shu[2][tk][tj][ti-1] - shu[2][tk][tj][ti]) + 
			    mux3 * (shu[2][tk][tj][ti+1] - shu[2][tk][tj][ti]) +
			    mux4 * (shu[2][tk][tj][ti+2] - shu[2][tk][tj][ti])  ) +
	       stry_j * (muy1 * (shu[2][tk][tj-2][ti] - shu[2][tk][tj][ti]) + 
			    muy2 * (shu[2][tk][tj-1][ti] - shu[2][tk][tj][ti]) + 
			    muy3 * (shu[2][tk][tj+1][ti] - shu[2][tk][tj][ti]) +
			    muy4 * (shu[2][tk][tj+2][ti] - shu[2][tk][tj][ti]) ) );
    /* ghost point only influences the first point (k=1) because ghcof(k)=0 for k>=2 */
    //r3 += mu3zz + dev_ghcof[k] * (shl[2][tj][ti] + 2 * shm[2][tj][ti]) *  shu[2][1][tj][ti];
    if(low)
      r3 += mu3zz + dev_ghcof[k] * (shl[0][tj][ti] + 2 * shm[0][tj][ti]) * u2; 
    else
      r3 += mu3zz + dev_ghcof[k] * (shl[7][tj][ti] + 2 * shm[7][tj][ti]) * u2; 
    
    /* cross-terms in first component of rhs */
    /*   (la*v_y)_x */
    r1 = r1 + strx_i * stry_j *
      (i144 * (shl[tk][tj][ti-2] * (shu[1][tk][tj-2][ti-2] - shu[1][tk][tj+2][ti-2] + 8 *
					(-shu[1][tk][tj-1][ti-2] + shu[1][tk][tj+1][ti-2])) - 8 *
	       (shl[tk][tj][ti-1] * (shu[1][tk][tj-2][ti-1] - shu[1][tk][tj+2][ti-1] + 8 *
					 (-shu[1][tk][tj-1][ti-1] + shu[1][tk][tj+1][ti-1])) )+ 8 *
	       (shl[tk][tj][ti+1] * (shu[1][tk][tj-2][ti+1] - shu[1][tk][tj+2][ti+1] + 8 *
					 (-shu[1][tk][tj-1][ti+1] + shu[1][tk][tj+1][ti+1])) ) -
	       (shl[tk][tj][ti+2] * (shu[1][tk][tj-2][ti+2] - shu[1][tk][tj+2][ti+2] + 8 *
					 (-shu[1][tk][tj-1][ti+2] + shu[1][tk][tj+1][ti+2])) ))
       /*   (mu*v_x)_y */
       + i144 * (shm[tk][tj-2][ti] * (shu[1][tk][tj-2][ti-2] - shu[1][tk][tj-2][ti+2] + 8 *
					  (-shu[1][tk][tj-2][ti-1] + shu[1][tk][tj-2][ti+1])) - 8 *
		 (shm[tk][tj-1][ti] * (shu[1][tk][tj-1][ti-2] - shu[1][tk][tj-1][ti+2] + 8 *
					   (-shu[1][tk][tj-1][ti-1] + shu[1][tk][tj-1][ti+1])) ) + 8 *
		 (shm[tk][tj+1][ti] * (shu[1][tk][tj+1][ti-2] - shu[1][tk][tj+1][ti+2] + 8 *
					   (-shu[1][tk][tj+1][ti-1] + shu[1][tk][tj+1][ti+1])) ) -
		 (shm[tk][tj+2][ti] * (shu[1][tk][tj+2][ti-2] - shu[1][tk][tj+2][ti+2] + 8 *
					   (-shu[1][tk][tj+2][ti-1] + shu[1][tk][tj+2][ti+1])) )) );
    /*   (la*w_z)_x: NOT CENTERED */
    u3zip2 = 0;
    u3zip1 = 0;
    u3zim1 = 0;
    u3zim2 = 0;
    for (int q=0; q<8; q++) {
      int qi = (low) ? q : (7-q);
      float_sw4 bope = dev_bope[q*6+k];
      u3zip2 += bope * shu[2][qi][tj][ti+2];
      u3zip1 += bope * shu[2][qi][tj][ti+1];
      u3zim1 += bope * shu[2][qi][tj][ti-1];
      u3zim2 += bope * shu[2][qi][tj][ti-2];
    }
    lau3zx = i12 * (-shl[tk][tj][ti+2] * u3zip2 + 8 *
		    shl[tk][tj][ti+1] * u3zip1 -8 *
		    shl[tk][tj][ti-1] * u3zim1 +
		    shl[tk][tj][ti-2] * u3zim2);
    r1 += strx_i * lau3zx;
    /*   (mu*w_x)_z: NOT CENTERED */
    mu3xz = 0;
    for (int q=0; q<8; q++) {
      int qi = (low) ? q : (7-q);
      mu3xz += dev_bope[q*6+k] * (shm[qi][tj][ti] * i12 *
				      (-shu[2][qi][tj][ti+2] + 8 * shu[2][qi][tj][ti+1]
				       -8 * shu[2][qi][tj][ti-1] + shu[2][qi][tj][ti-2]));
    }
    r1 += strx_i * mu3xz;
   
    /* cross-terms in second component of rhs */
    /*   (mu*u_y)_x */
    r2 = r2 + strx_i * stry_j *
      (i144 * (shm[tk][tj][ti-2] * (shu[0][tk][tj-2][ti-2]-shu[0][tk][tj+2][ti-2] + 8*
				      (-shu[0][tk][tj-1][ti-2]+shu[0][tk][tj+1][ti-2])) - 8*
	       (shm[tk][tj][ti-1] * (shu[0][tk][tj-2][ti-1]-shu[0][tk][tj+2][ti-1] + 8 *
				       (-shu[0][tk][tj-1][ti-1]+shu[0][tk][tj+1][ti-1])) ) + 8 *
	       (shm[tk][tj][ti+1] * (shu[0][tk][tj-2][ti+1]-shu[0][tk][tj+2][ti+1] + 8*
				       (-shu[0][tk][tj-1][ti+1]+shu[0][tk][tj+1][ti+1])) ) -
	       (shm[tk][tj][ti+2] * (shu[0][tk][tj-2][ti+2]-shu[0][tk][tj+2][ti+2] + 8 *
				       (-shu[0][tk][tj-1][ti+2]+shu[0][tk][tj+1][ti+2])) )) 
       /* (la*u_x)_y  */
       + i144*( shl[tk][tj-2][ti] * (shu[0][tk][tj-2][ti-2] - shu[0][tk][tj-2][ti+2] + 8 *
					 (-shu[0][tk][tj-2][ti-1] + shu[0][tk][tj-2][ti+1])) - 8*
		(shl[tk][tj-1][ti] * (shu[0][tk][tj-1][ti-2] - shu[0][tk][tj-1][ti+2] + 8 *
					  (-shu[0][tk][tj-1][ti-1] + shu[0][tk][tj-1][ti+1])) ) + 8 *
		(shl[tk][tj+1][ti] * (shu[0][tk][tj+1][ti-2] - shu[0][tk][tj+1][ti+2] + 8 *
					  (-shu[0][tk][tj+1][ti-1] + shu[0][tk][tj+1][ti+1])) ) -
		(shl[tk][tj+2][ti] * (shu[0][tk][tj+2][ti-2] - shu[0][tk][tj+2][ti+2] + 8 *
					  (-shu[0][tk][tj+2][ti-1] + shu[0][tk][tj+2][ti+1])) )) );

    /* (la*w_z)_y : NOT CENTERED */
    u3zjp2 = 0;
    u3zjp1 = 0;
    u3zjm1 = 0;
    u3zjm2 = 0;
    for (int q=0; q<8; q++) {
      int qi = (low) ? q : (7-q);
      float_sw4 bope = dev_bope[q*6+k];
      u3zjp2 += bope * shu[2][qi][tj+2][ti];
      u3zjp1 += bope * shu[2][qi][tj+1][ti];
      u3zjm1 += bope * shu[2][qi][tj-1][ti];
      u3zjm2 += bope * shu[2][qi][tj-2][ti];
    }
    lau3zy= i12 * (-shl[tk][tj+2][ti] * u3zjp2 + 8 *
		   shl[tk][tj+1][ti] * u3zjp1 - 8 *
		   shl[tk][tj-1][ti] * u3zjm1 +
		   shl[tk][tj-2][ti] * u3zjm2);

    r2 += stry_j * lau3zy;

    /* (mu*w_y)_z: NOT CENTERED */
    mu3yz = 0;
    for (int q=0; q<8; q++) {
      int qi = (low) ? q : (7-q);
      mu3yz += dev_bope[q*6+k] *
	(shm[qi][tj][ti] * i12 *
	 (-shu[2][qi][tj+2][ti] + 8 * shu[2][qi][tj+1][ti] - 8 *
	  shu[2][qi][tj-1][ti] + shu[2][qi][tj-2][ti]) );
    }
    r2 += stry_j * mu3yz;

    /* No centered cross terms in r3 */
    /*  (mu*u_z)_x: NOT CENTERED */
    u1zip2 = 0;
    u1zip1 = 0;
    u1zim1 = 0;
    u1zim2 = 0;
    for (int q=0; q<8; q++) {
      int qi = (low) ? q : (7-q);
      float_sw4 bope = dev_bope[q*6+k];
      u1zip2 += bope * shu[0][qi][tj][ti+2];
      u1zip1 += bope * shu[0][qi][tj][ti+1];
      u1zim1 += bope * shu[0][qi][tj][ti-1];
      u1zim2 += bope * shu[0][qi][tj][ti-2];
    }
    mu1zx= i12 * (-shm[tk][tj][ti+2] * u1zip2 + 8 *
		  shm[tk][tj][ti+1] * u1zip1 - 8 *
		  shm[tk][tj][ti-1] * u1zim1 +
		  shm[tk][tj][ti-2] * u1zim2);
    r3 += strx_i * mu1zx;

    /* (mu*v_z)_y: NOT CENTERED */
    u2zjp2 = 0;
    u2zjp1 = 0;
    u2zjm1 = 0;
    u2zjm2 = 0;
    for (int q=0; q<8; q++) {
      int qi = (low) ? q : (7-q);
      float_sw4 bope = dev_bope[q*6+k];
      u2zjp2 += bope * shu[1][qi][tj+2][ti];
      u2zjp1 += bope * shu[1][qi][tj+1][ti];
      u2zjm1 += bope * shu[1][qi][tj-1][ti];
      u2zjm2 += bope * shu[1][qi][tj-2][ti];
    }
    mu2zy= i12 * (-shm[tk][tj+2][ti] * u2zjp2 + 8 *
		  shm[tk][tj+1][ti] * u2zjp1 - 8 *
		  shm[tk][tj-1][ti] * u2zjm1 +
		  shm[tk][tj-2][ti] * u2zjm2);
    r3 += stry_j * mu2zy;

    /*   (la*u_x)_z: NOT CENTERED */
    lau1xz = 0;
    for (int q=0; q<8; q++)
    {
      int qi = (low) ? q : (7-q);
      lau1xz += dev_bope[q*6+k] *
	(shl[qi][tj][ti] * i12 *
	 (-shu[0][qi][tj][ti+2] + 8 * shu[0][qi][tj][ti+1] - 8 *
	  shu[0][qi][tj][ti-1] + shu[0][qi][tj][ti-2]) );
    }
    r3 += strx_i * lau1xz;

    /* (la*v_y)_z: NOT CENTERED */
    lau2yz = 0;
    for (int q=0; q<8; q++)
    {
      int qi = (low) ? q : (7-q);
      lau2yz += dev_bope[q*6+k] *
	(shl[qi][tj][ti] * i12 *
	 (-shu[1][qi][tj+2][ti] + 8 * shu[1][qi][tj+1][ti] - 8 *
	  shu[1][qi][tj-1][ti] + shu[1][qi][tj-2][ti]) );
    }
    r3 += stry_j * lau2yz;

    // Store the result at (i,j,k+2)
    index = (k + 2) * nij + j * ni + i;
    if (pred) {
      // Apply predictor
      float_sw4 fact = dt*dt / a_rho[index];
      up(0,index) = 2 * shu[0][tk][tj][ti] - um(0,index) + fact * (cof * r1 + fo(0,index));
      up(1,index) = 2 * shu[1][tk][tj][ti] - um(1,index) + fact * (cof * r2 + fo(1,index));
      up(2,index) = 2 * shu[2][tk][tj][ti] - um(2,index) + fact * (cof * r3 + fo(2,index));
    }
    else {
      // Apply 4th order correction
      float_sw4 fact = dt*dt*dt*dt / (12 * a_rho[index]);
      up(0,index) += fact * (cof * r1 + fo(0,index));
      up(1,index) += fact * (cof * r2 + fo(1,index));
      up(2,index) += fact * (cof * r3 + fo(2,index));
    }
  } // End active
}

// *************************************************************************************
// Kernel launcher

void rhs4_lowk_pred_gpu (int ifirst, int ilast, int jfirst, int jlast,
			 int ni, int nj, int nk, int nz,
			 float_sw4* a_up, float_sw4* a_u, float_sw4* a_um,
			 float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
			 float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			 float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream) {

  dim3 blocks = dim3((ni+UX-1)/UX, (nj+1)/2, 1);
  dim3 threads = dim3(UX, 2, 6);
  if (c_order)
    rhs4_Kboundary<1,1,1><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast,
       ni, nj, nk, nz,
       a_up, a_u, a_um,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, h, dt);
  else
    rhs4_Kboundary<0,1,1><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast,
       ni, nj, nk, nz,
       a_up, a_u, a_um,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, h, dt);
}

void rhs4_highk_pred_gpu (int ifirst, int ilast, int jfirst, int jlast,
			 int ni, int nj, int nk, int nz,
			 float_sw4* a_up, float_sw4* a_u, float_sw4* a_um,
			 float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
			 float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			 float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream) {

  dim3 blocks = dim3((ni+UX-1)/UX, (nj+1)/2, 1);
  dim3 threads = dim3(UX, 2, 6);
  if (c_order)
    rhs4_Kboundary<1,1,0><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast,
       ni, nj, nk, nz,
       a_up, a_u, a_um,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, h, dt);
  else
    rhs4_Kboundary<0,1,0><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast,
       ni, nj, nk, nz,
       a_up, a_u, a_um,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, h, dt);
}

void rhs4_lowk_corr_gpu (int ifirst, int ilast, int jfirst, int jlast,
			 int ni, int nj, int nk, int nz, 
			 float_sw4* a_up, float_sw4* a_u,
			 float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
			 float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			 float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream) {

  dim3 blocks = dim3((ni+UX-1)/UX, (nj+1)/2, 1);
  dim3 threads = dim3(UX, 2, 6);
  if (c_order)
    rhs4_Kboundary<1,0,1><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast,
       ni, nj, nk, nz,
       a_up, a_u, NULL,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, h, dt);
  else
    rhs4_Kboundary<0,0,1><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast,
       ni, nj, nk, nz,
       a_up, a_u, NULL,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, h, dt);
}

void rhs4_highk_corr_gpu (int ifirst, int ilast, int jfirst, int jlast,
			 int ni, int nj, int nk, int nz, 
			 float_sw4* a_up, float_sw4* a_u,
			 float_sw4* a_mu, float_sw4* a_lambda, float_sw4* a_rho, float_sw4* a_fo,
			 float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
			 float_sw4 h, float_sw4 dt, bool c_order, cudaStream_t stream) {

  dim3 blocks = dim3((ni+UX-1)/UX, (nj+1)/2, 1);
  dim3 threads = dim3(UX, 2, 6);
  if (c_order)
    rhs4_Kboundary<1,0,0><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast,
       ni, nj, nk, nz,
       a_up, a_u, NULL,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, h, dt);
  else
    rhs4_Kboundary<0,0,0><<<blocks,threads,0,stream>>>
      (ifirst, ilast, jfirst, jlast,
       ni, nj, nk, nz,
       a_up, a_u, NULL,
       a_mu, a_lambda, a_rho, a_fo,
       a_strx, a_stry, h, dt);
}

// *************************************************************************************
// *************************************************************************************
// *************************************************************************************
// Super grid damping terms :
// --------------------------
// Launch enough threads to cover the full NI dimension, and [jfirst:jlast]
// The kernel loops on all the elements in the K dimension, and uses register queues.
// The "forward" half of the stencil for the wavefields U is kept in shared memory.
// Rho is also in shared memory.

template <bool c_order>
// No launch_bounds here, it decreases performance.
__launch_bounds__(BX*BY,1)
__global__ void addsgd4 (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast, 
			 int ni, int nj, int nk,
			 float_sw4* __restrict__ a_up,
			 const float_sw4* __restrict__ a_u,
			 const float_sw4* __restrict__ a_um,
			 const float_sw4* __restrict__ a_rho,
			 const float_sw4* __restrict__ a_dcx,
			 const float_sw4* __restrict__ a_dcy,
			 const float_sw4* __restrict__ a_dcz,
			 const float_sw4* __restrict__ a_strx,
			 const float_sw4* __restrict__ a_stry,
			 const float_sw4* __restrict__ a_strz,
			 const float_sw4* __restrict__ a_cox,
			 const float_sw4* __restrict__ a_coy,
			 const float_sw4* __restrict__ a_coz,
			 const float_sw4 beta) {
  int index;
  float_sw4 rho_km1, rho_k, rho_kp1;
  float_sw4 rho_im1, rho_ip1;
  float_sw4 rho_jm1, rho_jp1;

  float_sw4 u_km2[3], u_km1[3], old_up[3];

  float_sw4 strx_i, stry_j, strz_k;

  float_sw4 dcx_im1, dcx_i, dcx_ip1;
  float_sw4 dcy_jm1, dcy_j, dcy_jp1;
  float_sw4 dcz_km1, dcz_k, dcz_kp1;
  float_sw4 cox_i, coy_j, coz_k;

  __shared__ float_sw4 shu[3][3][BY+4][BX+4];
  __shared__ float_sw4 shr[BY+4][BX+4];

  // I starts at 0 and covers the whole dimension
  const int i = threadIdx.x + blockIdx.x * BX;
  // J starts at jfirst
  const int j = jfirst + threadIdx.y + blockIdx.y * BY;

  const int ti = threadIdx.x + 2;
  const int tj = threadIdx.y + 2;

  const int nij = ni * nj;
  const int nijk = nij * nk;

  int active=0, loader=0, loady2=0, loaderx2=0;

  // Active threads doing the computation at (i+2,j)
  if (i+2 >= ifirst && i+2 <= ilast && j <= jlast)
    active = 1;

  if (i < ni && j-2 <= jlast+2) {
    loader = 1;
    if (threadIdx.x < 4 && i+BX < ni)
      loaderx2 = 1;
    if (threadIdx.y < 4 && j-2+BY <= jlast+2)
      loady2 = 1;
  }

  if (active) {
    // Load the first values of U-UM at (i+2, j, kfirst-2:kfirst-1)
    int idx = (kfirst - 2) * nij + j * ni + i + 2;
    for (int c=0; c<3; c++)
      u_km2[c] = u(c,idx) - um(c,idx);
    idx += nij;
    for (int c=0; c<3; c++)
      u_km1[c] = u(c,idx) - um(c,idx);
    // Load the first values of rho at kfirst-1
    rho_km1 = a_rho[idx];

    // Load strx, stry, cox, coy, , dcx, dcy centered at (i+2, j)
    strx_i   = a_strx[i+2];
    stry_j   = a_stry[j];
    cox_i   = a_cox[i+2];
    coy_j   = a_coy[j];

    dcx_im1 = a_dcx[i+1];
    dcx_i   = a_dcx[i+2];
    dcx_ip1 = a_dcx[i+3];

    dcy_jm1 = a_dcy[j-1];
    dcy_j   = a_dcy[j];
    dcy_jp1 = a_dcy[j+1];

    // Load the first values of dcz, strz
    dcz_km1 = a_dcz[kfirst-1];
    dcz_k   = a_dcz[kfirst];
  }

  // Loading index for data with halos at (i,j-2,kfirst)
  index = kfirst * nij + (j - 2) * ni + i;

  // Load the 2 plans of U-UM in the shared memory, at kfirst and kfirst+1
  // Also load Rho at kfirst in shared memory
  for (int c=0; c<3; c++) {
    if (loader) {
      int idx = index;
      shu[c][1][threadIdx.y][threadIdx.x] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y][threadIdx.x] = u(c,idx+nij) - um(c,idx+nij);
    }
    if (loaderx2) {
      int idx = index + BX;
      shu[c][1][threadIdx.y][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y][threadIdx.x+BX] = u(c,idx+nij) - um(c,idx+nij);
    }
    if (loady2) {
      int idx = index + BY * ni;
      shu[c][1][threadIdx.y+BY][threadIdx.x] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y+BY][threadIdx.x] = u(c,idx+nij) - um(c,idx+nij);
    }
    if (loady2 && loaderx2) {
      int idx = index + BY * ni + BX;
      shu[c][1][threadIdx.y+BY][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y+BY][threadIdx.x+BX] = u(c,idx+nij) - um(c,idx+nij);
    }
  }
  // Load rho
  if (loader)
    shr[threadIdx.y][threadIdx.x] = a_rho[index];
  if (loaderx2)
    shr[threadIdx.y][threadIdx.x+BX] = a_rho[index+BX];
  if (loady2)
    shr[threadIdx.y+BY][threadIdx.x] = a_rho[index+BY*ni];
  if (loady2 && loaderx2)
    shr[threadIdx.y+BY][threadIdx.x+BX] = a_rho[index+BY*ni+BX];

  // Extract the Rho data from the SHR memory, to the register queue
  __syncthreads();
  rho_im1 = shr[tj][ti-1];
  rho_k = shr[tj][ti];
  rho_ip1 = shr[tj][ti+1];
  rho_jm1 = shr[tj-1][ti];
  rho_jp1 = shr[tj+1][ti];

  // Move the index to kfirst+2
  index += 2 * nij;

  // Main loop on the K dimension
  for (int k=kfirst; k <= klast; k++) {

    __syncthreads();
    // Rotate the shared memory and load (U-UM) at k+2
    for (int c=0; c<3; c++) {
      if (loader) { 
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y][threadIdx.x] = shu[c][s+1][threadIdx.y][threadIdx.x];
        int idx = index;
        shu[c][2][threadIdx.y][threadIdx.x] = u(c,idx) - um(c,idx);
      }
      if (loaderx2) {
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y][threadIdx.x+BX] = shu[c][s+1][threadIdx.y][threadIdx.x+BX];
        int idx = index + BX;
        shu[c][2][threadIdx.y][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      }
      if (loady2) {
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y+BY][threadIdx.x] = shu[c][s+1][threadIdx.y+BY][threadIdx.x];
        int idx = index + BY * ni;
        shu[c][2][threadIdx.y+BY][threadIdx.x] = u(c,idx) - um(c,idx);
      }
      if (loady2 && loaderx2) {
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y+BY][threadIdx.x+BX] = shu[c][s+1][threadIdx.y+BY][threadIdx.x+BX];
        int idx = index + BY * ni + BX;
        shu[c][2][threadIdx.y+BY][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      }
    }
    // Load Rho at k+1
    if (loader)
      shr[threadIdx.y][threadIdx.x] = a_rho[index-nij];
    if (loaderx2)
      shr[threadIdx.y][threadIdx.x+BX] = a_rho[index+BX-nij];
    if (loady2)
      shr[threadIdx.y+BY][threadIdx.x] = a_rho[index+BY*ni-nij];
    if (loady2 && loaderx2)
      shr[threadIdx.y+BY][threadIdx.x+BX] = a_rho[index+BY*ni+BX-nij];

    // Move the loading index to the next K
    index += nij;
    __syncthreads();

    if (active) {

      // Load the new values for strz, coz, dcz
      strz_k = a_strz[k];
      dcz_kp1 = a_dcz[k+1];
      coz_k = a_coz[k];

      float_sw4 birho = beta / rho_k;
      
      // Extract the values from shared memory into registers
      rho_kp1 = shr[tj][ti];

      // Update up(i+2,j,k)
      int idx = k * nij + j * ni + i + 2;
      old_up[0] = up(0,idx);
      old_up[1] = up(1,idx);
      old_up[2] = up(2,idx);
      
      for (int c=0; c<3; c++) {
        up(c,idx) = old_up[c] - birho * 
          (// x-differences
           strx_i * coy_j * coz_k * (rho_ip1 * dcx_ip1 * (shu[c][0][tj][ti+2] - 2 * shu[c][0][tj][ti+1] + shu[c][0][tj][ti  ]) -
                                     2 * rho_k * dcx_i * (shu[c][0][tj][ti+1] - 2 * shu[c][0][tj][ti  ] + shu[c][0][tj][ti-1]) +
                                     rho_im1 * dcx_im1 * (shu[c][0][tj][ti  ] - 2 * shu[c][0][tj][ti-1] + shu[c][0][tj][ti-2])) +
           // y-differences
           stry_j * cox_i * coz_k * (rho_jp1 * dcy_jp1 * (shu[c][0][tj+2][ti] - 2 * shu[c][0][tj+1][ti] + shu[c][0][tj  ][ti]) -
                                     2 * rho_k * dcy_j * (shu[c][0][tj+1][ti] - 2 * shu[c][0][tj  ][ti] + shu[c][0][tj-1][ti]) +
                                     rho_jm1 * dcy_jm1 * (shu[c][0][tj  ][ti] - 2 * shu[c][0][tj-1][ti] + shu[c][0][tj-2][ti])) +
           // z-differences
           strz_k * cox_i * coy_j * (rho_kp1 * dcz_kp1 * (shu[c][2][tj][ti] - 2 * shu[c][1][tj][ti] + shu[c][0][tj][ti]) -
                                     2 * rho_k * dcz_k * (shu[c][1][tj][ti] - 2 * shu[c][0][tj][ti] + u_km1[c]         ) +
                                     rho_km1 * dcz_km1 * (shu[c][0][tj][ti] - 2 * u_km1[c]          + u_km2[c])));
        // Rotate the registers
        u_km2[c] = u_km1[c];
        u_km1[c] = shu[c][0][tj][ti];
      }

      dcz_km1 = dcz_k;
      dcz_k = dcz_kp1;

      rho_km1 = rho_k;
      rho_k = rho_kp1;
      rho_im1 = shr[tj][ti-1];
      rho_ip1 = shr[tj][ti+1];
      rho_jm1 = shr[tj-1][ti];
      rho_jp1 = shr[tj+1][ti];

    } // End active

  } // End K loop
}

template <bool c_order>
// No launch_bounds here, it decreases performance.
__launch_bounds__(BX*BY,1)
__global__ void addsgd4_no_SM (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast, 
			 int ni, int nj, int nk,
			 float_sw4* __restrict__ a_up,
			 const float_sw4* __restrict__ a_u,
			 const float_sw4* __restrict__ a_um,
			 const float_sw4* __restrict__ a_rho,
			 const float_sw4* __restrict__ a_dcx,
			 const float_sw4* __restrict__ a_dcy,
			 const float_sw4* __restrict__ a_dcz,
			 const float_sw4* __restrict__ a_strx,
			 const float_sw4* __restrict__ a_stry,
			 const float_sw4* __restrict__ a_strz,
			 const float_sw4* __restrict__ a_cox,
			 const float_sw4* __restrict__ a_coy,
			 const float_sw4* __restrict__ a_coz,
			 const float_sw4 beta) {
  int index;
  // I starts at 0 and covers the whole dimension
  const int i = threadIdx.x + blockIdx.x * BX;
  // J starts at jfirst
  const int j = jfirst + threadIdx.y + blockIdx.y * BY;

  const int nij = ni * nj;
  const int nijk = nij * nk;

  int active=0;

  // Active threads doing the computation at (i+2,j)
  if (i >= ifirst && i <= ilast && j >= jfirst && j <= jlast)
    active = 1;


  // Main loop on the K dimension
  for (int k=kfirst; k <= klast; k++) {
    index = k * nij + j * ni + i;
    if (active) {
      float_sw4 birho=beta/a_rho[index];
      for( int c=0 ; c < 3 ; c++ )
      {
	 up(c,index) -= birho*(
		  // x-differences
		   a_strx[i]*a_coy[j]*a_coz[k]*(
       a_rho[index+1]*a_dcx[i+1]*
                   ( u(c,index+2) -2*u(c,index+1)+ u(c,index))
      -2*a_rho[index]*a_dcx[i]  *
                   ( u(c,index+1) -2*u(c,index)+ u(c,index-1))
      +a_rho[index-1]*a_dcx[i-1]*
                   ( u(c,index) -2*u(c,index-1)+ u(c,index-2)) 
      -a_rho[index+1]*a_dcx[i+1]*
                   (um(c,index+2)-2*um(c,index+1)+um(c,index)) 
      +2*a_rho[index]*a_dcx[i]  *
                   (um(c,index+1)-2*um(c,index)+um(c,index-1)) 
      -a_rho[index-1]*a_dcx[i-1]*
                   (um(c,index)-2*um(c,index-1)+um(c,index-2)) ) +
// y-differences
      a_stry[j]*a_cox[i]*a_coz[k]*(
      +a_rho[index+ni]*a_dcy[j+1]*
                   ( u(c,index+2*ni) -2*u(c,index+ni)+ u(c,index)) 
      -2*a_rho[index]*a_dcy[j]  *
                   ( u(c,index+ni) -2*u(c,index)+ u(c,index-ni))
      +a_rho[index-ni]*a_dcy[j-1]*
                   ( u(c,index) -2*u(c,index-ni)+ u(c,index-2*ni)) 
      -a_rho[index+ni]*a_dcy[j+1]*
                   (um(c,index+2*ni)-2*um(c,index+ni)+um(c,index)) 
      +2*a_rho[index]*a_dcy[j]  *
                   (um(c,index+ni)-2*um(c,index)+um(c,index-ni)) 
      -a_rho[index-ni]*a_dcy[j-1]*
                   (um(c,index)-2*um(c,index-ni)+um(c,index-2*ni)) ) +
       a_strz[k]*a_cox[i]*a_coy[j]*(
// z-differences
      +a_rho[index+nij]*a_dcz[k+1]* 
                 ( u(c,index+2*nij) -2*u(c,index+nij)+ u(c,index)) 
      -2*a_rho[index]*a_dcz[k]  *
                 ( u(c,index+nij) -2*u(c,index)+ u(c,index-nij))
      +a_rho[index-nij]*a_dcz[k-1]*
                 ( u(c,index) -2*u(c,index-nij)+ u(c,index-2*nij)) 
      -a_rho[index+nij]*a_dcz[k+1]*
                 (um(c,index+2*nij)-2*um(c,index+nij)+um(c,index)) 
      +2*a_rho[index]*a_dcz[k]  *
                 (um(c,index+nij)-2*um(c,index)+um(c,index-nij)) 
      -a_rho[index-nij]*a_dcz[k-1]*
                 (um(c,index)-2*um(c,index-nij)+um(c,index-2*nij)) ) 
				);
      }
    } // End active

  } // End K loop
}

template <bool c_order>
// No launch_bounds here, it decreases performance.
__launch_bounds__(BX*BY,1)
__global__ void addsgd4_SM (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast, 
			 int ni, int nj, int nk,
			 float_sw4* __restrict__ a_up,
			 const float_sw4* __restrict__ a_u,
			 const float_sw4* __restrict__ a_um,
			 const float_sw4* __restrict__ a_rho,
			 const float_sw4* __restrict__ a_dcx,
			 const float_sw4* __restrict__ a_dcy,
			 const float_sw4* __restrict__ a_dcz,
			 const float_sw4* __restrict__ a_strx,
			 const float_sw4* __restrict__ a_stry,
			 const float_sw4* __restrict__ a_strz,
			 const float_sw4* __restrict__ a_cox,
			 const float_sw4* __restrict__ a_coy,
			 const float_sw4* __restrict__ a_coz,
			 const float_sw4 beta) {
  int index;
  float_sw4 u_km2[3], u_km1[3], old_up[3];

  __shared__ float_sw4 shu[3][3][BY+4][BX+4];

  // I starts at 0 and covers the whole dimension
  const int i = threadIdx.x + blockIdx.x * BX;
  // J starts at jfirst
  const int j = jfirst + threadIdx.y + blockIdx.y * BY;

  const int ti = threadIdx.x + 2;
  const int tj = threadIdx.y + 2;

  const int nij = ni * nj;
  const int nijk = nij * nk;

  int active=0, loader=0, loady2=0, loaderx2=0;

  // Active threads doing the computation at (i+2,j)
  if (i+2 >= ifirst && i+2 <= ilast && j <= jlast)
    active = 1;

  if (i < ni && j-2 <= jlast+2) {
    loader = 1;
    if (threadIdx.x < 4 && i+BX < ni)
      loaderx2 = 1;
    if (threadIdx.y < 4 && j-2+BY <= jlast+2)
      loady2 = 1;
  }

  if (active) {
    // Load the first values of U-UM at (i+2, j, kfirst-2:kfirst-1)
    int idx = (kfirst - 2) * nij + j * ni + i + 2;
    for (int c=0; c<3; c++)
      u_km2[c] = u(c,idx) - um(c,idx);
    idx += nij;
    for (int c=0; c<3; c++)
      u_km1[c] = u(c,idx) - um(c,idx);
  }

  // Loading index for data with halos at (i,j-2,kfirst)
  index = kfirst * nij + (j - 2) * ni + i;

  // Load the 2 plans of U-UM in the shared memory, at kfirst and kfirst+1
  // Also load Rho at kfirst in shared memory
  for (int c=0; c<3; c++) {
    if (loader) {
      int idx = index;
      shu[c][1][threadIdx.y][threadIdx.x] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y][threadIdx.x] = u(c,idx+nij) - um(c,idx+nij);
    }
    if (loaderx2) {
      int idx = index + BX;
      shu[c][1][threadIdx.y][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y][threadIdx.x+BX] = u(c,idx+nij) - um(c,idx+nij);
    }
    if (loady2) {
      int idx = index + BY * ni;
      shu[c][1][threadIdx.y+BY][threadIdx.x] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y+BY][threadIdx.x] = u(c,idx+nij) - um(c,idx+nij);
    }
    if (loady2 && loaderx2) {
      int idx = index + BY * ni + BX;
      shu[c][1][threadIdx.y+BY][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      shu[c][2][threadIdx.y+BY][threadIdx.x+BX] = u(c,idx+nij) - um(c,idx+nij);
    }
  }

  // Move the index to kfirst+2
  index += 2 * nij;

  // Main loop on the K dimension
  for (int k=kfirst; k <= klast; k++) {
    __syncthreads();
    // Rotate the shared memory and load (U-UM) at k+2
    for (int c=0; c<3; c++) {
      if (loader) { 
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y][threadIdx.x] = shu[c][s+1][threadIdx.y][threadIdx.x];
        int idx = index;
        shu[c][2][threadIdx.y][threadIdx.x] = u(c,idx) - um(c,idx);
      }
      if (loaderx2) {
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y][threadIdx.x+BX] = shu[c][s+1][threadIdx.y][threadIdx.x+BX];
        int idx = index + BX;
        shu[c][2][threadIdx.y][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      }
      if (loady2) {
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y+BY][threadIdx.x] = shu[c][s+1][threadIdx.y+BY][threadIdx.x];
        int idx = index + BY * ni;
        shu[c][2][threadIdx.y+BY][threadIdx.x] = u(c,idx) - um(c,idx);
      }
      if (loady2 && loaderx2) {
        for (int s=0; s<2; s++)
          shu[c][s][threadIdx.y+BY][threadIdx.x+BX] = shu[c][s+1][threadIdx.y+BY][threadIdx.x+BX];
        int idx = index + BY * ni + BX;
        shu[c][2][threadIdx.y+BY][threadIdx.x+BX] = u(c,idx) - um(c,idx);
      }
    }

    // Move the loading index to the next K
    index += nij;
    __syncthreads();

    if (active) {
      int idx = k * nij + j * ni + i + 2;
      float_sw4 birho=beta/a_rho[idx];

      for (int c=0; c<3; c++) {
        up(c,idx) -= birho * 
          (// x-differences
           a_strx[i+2] * a_coy[j] * a_coz[k] * (a_rho[idx+1] * a_dcx[i+3] * (shu[c][0][tj][ti+2] - 2 * shu[c][0][tj][ti+1] + shu[c][0][tj][ti  ]) -
                                     2 * a_rho[idx] * a_dcx[i+2] * (shu[c][0][tj][ti+1] - 2 * shu[c][0][tj][ti  ] + shu[c][0][tj][ti-1]) +
                                     a_rho[idx-1] * a_dcx[i+1] * (shu[c][0][tj][ti  ] - 2 * shu[c][0][tj][ti-1] + shu[c][0][tj][ti-2])) +
           // y-differences
           a_stry[j] * a_cox[i+2] * a_coz[k] * (a_rho[idx+ni] * a_dcy[j+1] * (shu[c][0][tj+2][ti] - 2 * shu[c][0][tj+1][ti] + shu[c][0][tj  ][ti]) -
                                     2 * a_rho[idx] * a_dcy[j] * (shu[c][0][tj+1][ti] - 2 * shu[c][0][tj  ][ti] + shu[c][0][tj-1][ti]) +
                                     a_rho[idx-ni] * a_dcy[j-1] * (shu[c][0][tj  ][ti] - 2 * shu[c][0][tj-1][ti] + shu[c][0][tj-2][ti])) +
           // z-differences
           a_strz[k] * a_cox[i+2] * a_coy[j] * (a_rho[idx+nij] * a_dcz[k+1] * (shu[c][2][tj][ti] - 2 * shu[c][1][tj][ti] + shu[c][0][tj][ti]) -
                                     2 * a_rho[idx] * a_dcz[k] * (shu[c][1][tj][ti] - 2 * shu[c][0][tj][ti] + u_km1[c]         ) +
                                     a_rho[idx-nij] * a_dcz[k-1] * (shu[c][0][tj][ti] - 2 * u_km1[c]          + u_km2[c])));
        // Rotate the registers
        u_km2[c] = u_km1[c];
        u_km1[c] = shu[c][0][tj][ti];
      }
    } // End active

  } // End K loop
}


// *************************************************************************************
// Kernel launcher

void addsgd4_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		  int ni, int nj, int nk,
		  float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		  float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
		  float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
		  float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
		  float_sw4 beta, int c_order, cudaStream_t stream) {

  if (beta == 0.0)
    return;
  // Cover [0:ni-1][jfirst:jlast]
  int njcomp = jlast - jfirst + 1;
  dim3 blocks = dim3((ni+BX-1)/BX, (njcomp+BY-1)/BY, 1);
  dim3 threads = dim3(BX, BY, 1);
  if (c_order)
    addsgd4_SM<true><<<blocks,threads,0,stream>>> (ifirst, ilast, jfirst, jlast, kfirst, klast,
						ni, nj, nk,
						a_up, a_u, a_um, a_rho,
						a_dcx,  a_dcy,  a_dcz,
						a_strx, a_stry, a_strz,
						a_cox,  a_coy,  a_coz, beta);
  else
    addsgd4_SM<false><<<blocks,threads,0,stream>>> (ifirst, ilast, jfirst, jlast, kfirst, klast,
						 ni, nj, nk,
						 a_up, a_u, a_um, a_rho,
						 a_dcx,  a_dcy,  a_dcz,
						 a_strx, a_stry, a_strz,
						 a_cox,  a_coy,  a_coz, beta);
}


// *************************************************************************************
// *************************************************************************************
// *************************************************************************************
// SuperGrid X halos.
// ------------------
// This kernel computes the volumes :
//   [ifirst:ifirst+1, jfirst:jlast, kfirst:klast]
//   [ilast-1:ilast  , jfirst:jlast, kfirst:klast]
// We're using 384 threads : 2x16x12 to compute the 2 elements for a (16x12) area.
// The 384 threads are re-mapped as 6x16x4 threads to load the input data with halos,
// so that the 6 coalesced elements that we need can be loaded efficiently in one operation.

template <bool c_order>
__launch_bounds__(384)
__global__ void addsgd4_X (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast, 
                           int ni, int nj, int nk,
                           float_sw4* __restrict__ a_up,
                           const float_sw4* __restrict__ a_u,
                           const float_sw4* __restrict__ a_um,
                           const float_sw4* __restrict__ a_rho,
                           const float_sw4* __restrict__ a_dcx,
                           const float_sw4* __restrict__ a_dcy,
                           const float_sw4* __restrict__ a_dcz,
                           const float_sw4* __restrict__ a_strx,
                           const float_sw4* __restrict__ a_stry,
                           const float_sw4* __restrict__ a_strz,
                           const float_sw4* __restrict__ a_cox,
                           const float_sw4* __restrict__ a_coy,
                           const float_sw4* __restrict__ a_coz,
                           const float_sw4 beta) {
  int index, i;

  float_sw4 rho_km1, rho_k, rho_kp1;
  float_sw4 rho_im1, rho_ip1;
  float_sw4 rho_jm1, rho_jp1;

  float_sw4 strx_i, stry_j, strz_k;

  float_sw4 dcx_im1, dcx_i, dcx_ip1;
  float_sw4 dcy_jm1, dcy_j, dcy_jp1;
  float_sw4 dcz_km1, dcz_k, dcz_kp1;
  float_sw4 cox_i, coy_j, coz_k;

  int active=0, loadj1=0, loadj2=0, loadk1=0, loadk2=0, loadk3=0, loadk4=0;

  // Shared memory for 3 wavefields
  __shared__ float_sw4 shu[3][12+4][16+4][2+4];

  // J and K start at (jfirst, kfirst)
  const int j = jfirst + threadIdx.y + blockIdx.y * 16;
  const int k = kfirst + threadIdx.z + blockIdx.z * 12;

  // Natural thread mapping for the computation
  const int ti = threadIdx.x + 2;
  const int tj = threadIdx.y + 2;
  const int tk = threadIdx.z + 2;

  // Use the (6,16,4) mapping to load the data into shared memory
  int tid = (threadIdx.z * 16 + threadIdx.y) * 2 + threadIdx.x;
  int txload = (tid % 6);
  int tyload = (tid / 6);
  int tzload = tyload / 16;
  tyload = tyload % 16;
  
  const int nij = ni * nj;
  const int nijk = nij * nk;

  // Active threads doing the computation
  if (j <= jlast && k <= klast)
    active = 1;

  if (active) {
    // Load stry, stry, coy, coz, dcy, dcz centered at (j,k)
    stry_j = a_stry[j];
    strz_k = a_strz[k];
    coy_j = a_coy[j];
    coz_k = a_coz[k];
    dcy_jm1 = a_dcy[j-1];
    dcy_j = a_dcy[j];
    dcy_jp1 = a_dcy[j+1];
    dcz_km1 = a_dcz[k-1];
    dcz_k = a_dcz[k];
    dcz_kp1 = a_dcz[k+1];
  }

  // Loading index, computed with the alternate thread mapping, starting at (jfirst-2, kfirst-2)
  int jload, kload;
  jload = blockIdx.y * 16 + tyload + jfirst - 2;
  kload = blockIdx.z * 12 + tzload + kfirst - 2;

  // We're loading 6x20x16 elements with 6x16x4 threads => 2 x 4 = 8 sections.
  // Find out Which threads can load the data for the different sections
  if (jload <= jlast+2) loadj1 = 1;
  if (tyload < 4 && jload+16 <= jlast+2) loadj2 = 1;
  if (kload <= klast+2) loadk1 = 1;
  if (kload+4 <= klast+2) loadk2 = 1;
  if (kload+8 <= klast+2) loadk3 = 1;
  if (kload+12 <= klast+2) loadk4 = 1;

  // Loop on the 2 sides
#pragma unroll
  for (int iside=0; iside < 2; iside++) {
    
    if (iside == 0) {
      index = (kload * nj + jload) * ni + ifirst - 2 + txload;
      i = ifirst + threadIdx.x;
    }
    else {
      index = (kload * nj + jload) * ni + ilast - 3 + txload;
      i = ilast - 1 + threadIdx.x;
    }
    
    // Load the 3 (U-UM) wavefields in shared memory
    for (int c=0; c<3; c++) {
      int idx = index;
      if (loadk1) {
        if (loadj1)
          shu[c][tzload][tyload][txload] = u(c,idx) - um(c,idx);
        if (loadj2)
          shu[c][tzload][tyload+16][txload] = u(c,idx+16*ni) - um(c,idx+16*ni);
      }
      idx += 4 * nij;
      if (loadk2) {
        if (loadj1)
          shu[c][tzload+4][tyload][txload] = u(c,idx) - um(c,idx);
        if (loadj2)
          shu[c][tzload+4][tyload+16][txload] = u(c,idx+16*ni) - um(c,idx+16*ni);
      }
      idx += 4 * nij;
      if (loadk3) {
        if (loadj1)
          shu[c][tzload+8][tyload][txload] = u(c,idx) - um(c,idx);
        if (loadj2)
          shu[c][tzload+8][tyload+16][txload] = u(c,idx+16*ni) - um(c,idx+16*ni);
      }
      idx += 4 * nij;
      if (loadk4) {
        if (loadj1)
          shu[c][tzload+12][tyload][txload] = u(c,idx) - um(c,idx);
        if (loadj2)
          shu[c][tzload+12][tyload+16][txload] = u(c,idx+16*ni) - um(c,idx+16*ni);
      }
    }
    __syncthreads();

    if (active) {

      // Load strx, cox, dcx centered at i
      strx_i   = a_strx[i];
      cox_i = a_cox[i];
      dcx_im1 = a_dcx[i-1];
      dcx_i = a_dcx[i];
      dcx_ip1 = a_dcx[i+1];

      // Natural thread index at (i,j,k)
      int idx = k * nij + j * ni + i;
      
      // Load rho directly 
      rho_k   = a_rho[idx];
      rho_im1 = a_rho[idx-1];
      rho_ip1 = a_rho[idx+1];
      rho_jm1 = a_rho[idx-ni];
      rho_jp1 = a_rho[idx+ni];      
      rho_km1 = a_rho[idx-nij];
      rho_kp1 = a_rho[idx+nij];

      float_sw4 birho = beta / rho_k;

      for (int c=0; c<3; c++) 
        up(c, idx) -= birho * 
          (// x-differences
           strx_i * coy_j * coz_k *
           (rho_ip1 * dcx_ip1 * (shu[c][tk][tj][ti+2] - 2 * shu[c][tk][tj][ti+1] + shu[c][tk][tj][ti  ]) -
            2 * rho_k * dcx_i * (shu[c][tk][tj][ti+1] - 2 * shu[c][tk][tj][ti  ] + shu[c][tk][tj][ti-1]) +
            rho_im1 * dcx_im1 * (shu[c][tk][tj][ti  ] - 2 * shu[c][tk][tj][ti-1] + shu[c][tk][tj][ti-2])) +
           // y-differences
           stry_j * cox_i * coz_k *
           (rho_jp1 * dcy_jp1 * (shu[c][tk][tj+2][ti] - 2 * shu[c][tk][tj+1][ti] + shu[c][tk][tj  ][ti]) -
            2 * rho_k * dcy_j * (shu[c][tk][tj+1][ti] - 2 * shu[c][tk][tj  ][ti] + shu[c][tk][tj-1][ti]) +
            rho_jm1 * dcy_jm1 * (shu[c][tk][tj  ][ti] - 2 * shu[c][tk][tj-1][ti] + shu[c][tk][tj-2][ti])) +
           // z-differences
           strz_k * cox_i * coy_j *
           (rho_kp1 * dcz_kp1 * (shu[c][tk+2][tj][ti] - 2 * shu[c][tk+1][tj][ti] + shu[c][tk  ][tj][ti]) -
            2 * rho_k * dcz_k * (shu[c][tk+1][tj][ti] - 2 * shu[c][tk  ][tj][ti] + shu[c][tk-1][tj][ti]) +
            rho_km1 * dcz_km1 * (shu[c][tk  ][tj][ti] - 2 * shu[c][tk-1][tj][ti] + shu[c][tk-2][tj][ti])));


    }  // End active
    
    if (iside == 0)
      __syncthreads();
    
  } // End side loop
}


// *************************************************************************************
// Kernel launcher

void addsgd4_X_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		    int ni, int nj, int nk,
		    float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		    float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
		    float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
		    float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
		    float_sw4 beta, int c_order, cudaStream_t stream) {

  if (beta == 0.0)
    return;

  // Launch enough threads to cover [jfirst:jlast][kfirst:klast]
  int njcomp = jlast - jfirst + 1;
  int nkcomp = klast - kfirst + 1;

  // 384 threads as 2x16x12, they will also be used as 6x16x4 inside the kernel
  dim3 blocks = dim3(1, (njcomp+15)/16, (nkcomp+11)/12);
  dim3 threads = dim3(2, 16, 12);
  if (c_order)
    addsgd4_X<1><<<blocks,threads,0,stream>>> (ifirst, ilast, jfirst, jlast, kfirst, klast,
					       ni, nj, nk,
					       a_up, a_u, a_um, a_rho,
					       a_dcx,  a_dcy,  a_dcz,
					       a_strx, a_stry, a_strz,
					       a_cox,  a_coy,  a_coz, beta);
  else
    addsgd4_X<0><<<blocks,threads,0,stream>>> (ifirst, ilast, jfirst, jlast, kfirst, klast,
					       ni, nj, nk,
					       a_up, a_u, a_um, a_rho,
					       a_dcx,  a_dcy,  a_dcz,
					       a_strx, a_stry, a_strz,
					       a_cox,  a_coy,  a_coz, beta);
}

// *************************************************************************************
// *************************************************************************************
// *************************************************************************************
// RHS4 Y halos :
// --------------
// Compute RHS4 in the halos, for each side in the Y dimension.
// Use blocks of (BX,2,4) threads, and launch enough threads to cover the whole
// volume (0:ni-1, jfirst:jlast, kfirst:klast).
// The block of (BX,2,4) threads allows to load (BX+4,6,8) elements in shared memory
// relatively easily, while keeping enough blocks to load the GPU.



template <bool c_order>
__launch_bounds__(BX*8)
__global__ void addsgd4_Y (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast, 
                           int ni, int nj, int nk,
                           float_sw4* __restrict__ a_up,
                           const float_sw4* __restrict__ a_u,
                           const float_sw4* __restrict__ a_um,
                           const float_sw4* __restrict__ a_rho,
                           const float_sw4* __restrict__ a_dcx,
                           const float_sw4* __restrict__ a_dcy,
                           const float_sw4* __restrict__ a_dcz,
                           const float_sw4* __restrict__ a_strx,
                           const float_sw4* __restrict__ a_stry,
                           const float_sw4* __restrict__ a_strz,
                           const float_sw4* __restrict__ a_cox,
                           const float_sw4* __restrict__ a_coy,
                           const float_sw4* __restrict__ a_coz,
                           const float_sw4 beta) {
  int index, j;

  float_sw4 rho_km1, rho_k, rho_kp1;
  float_sw4 rho_im1, rho_ip1;
  float_sw4 rho_jm1, rho_jp1;

  float_sw4 strx_i, stry_j, strz_k;

  float_sw4 dcx_im1, dcx_i, dcx_ip1;
  float_sw4 dcy_jm1, dcy_j, dcy_jp1;
  float_sw4 dcz_km1, dcz_k, dcz_kp1;
  float_sw4 cox_i, coy_j, coz_k;

  int active=0, loader=0, loadx2=0, loadk2=0;

  __shared__ float_sw4 shu[3][8][6][BX+4];

  // I starts at 0 and covers the whole dimension
  const int i = threadIdx.x + blockIdx.x * BX;
  // K starts at kfirst
  const int k = kfirst + threadIdx.z + blockIdx.z * 4;

  const int ti = threadIdx.x + 2;
  const int tj = threadIdx.y + 2;
  const int tk = threadIdx.z + 2;
  
  const int nij = ni * nj;
  const int nijk = nij * nk;

  // Active threads doing the computation at (i,j,k)
  if (i >= ifirst && i <= ilast && k <= klast)
    active = 1;

  if (k-2 <= klast+2) {
    if (i-2 >= 0 && i-2 < ni)
      loader = 1;
    if (threadIdx.x < 4 && i+BX-2 < ni)
      loadx2 = 1;
    if (k <= klast) // (k+4-2 <= klast+2)
      loadk2 = 1;
  }

  if (active) {
    // Load strx, strz, cox, coz, dcx, dcz centered at i and k
    strx_i = a_strx[i];
    strz_k = a_strz[k];
    cox_i = a_cox[i];
    coz_k = a_coz[k];
    dcx_im1 = a_dcx[i-1];
    dcx_i   = a_dcx[i];
    dcx_ip1 = a_dcx[i+1];
    dcz_km1 = a_dcz[k-1];
    dcz_k   = a_dcz[k];
    dcz_kp1 = a_dcz[k+1];
  }

  // Loop on the 2 sides
#pragma unroll
  for (int iside=0; iside < 2; iside++) {

    if (iside == 0)
      j = jfirst + threadIdx.y;
    else
      j = jlast - 1 + threadIdx.y;

    index = (k - 2) * nij + (j - 2) * ni + i - 2;
    
    // Load U-Um in the shared memory
    if (loader) {
      int idx = index;
      shu[0][threadIdx.z][threadIdx.y][threadIdx.x] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z][threadIdx.y][threadIdx.x] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z][threadIdx.y][threadIdx.x] = u(2,idx) - um(2,idx);
      idx += 2 * ni;
      shu[0][threadIdx.z][threadIdx.y+2][threadIdx.x] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z][threadIdx.y+2][threadIdx.x] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z][threadIdx.y+2][threadIdx.x] = u(2,idx) - um(2,idx);
      idx += 2 * ni;
      shu[0][threadIdx.z][threadIdx.y+4][threadIdx.x] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z][threadIdx.y+4][threadIdx.x] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z][threadIdx.y+4][threadIdx.x] = u(2,idx) - um(2,idx);
    }
    if (loadx2) {
      int idx = index + BX;
      shu[0][threadIdx.z][threadIdx.y][threadIdx.x+BX] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z][threadIdx.y][threadIdx.x+BX] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z][threadIdx.y][threadIdx.x+BX] = u(2,idx) - um(2,idx);
      idx += 2 * ni;
      shu[0][threadIdx.z][threadIdx.y+2][threadIdx.x+BX] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z][threadIdx.y+2][threadIdx.x+BX] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z][threadIdx.y+2][threadIdx.x+BX] = u(2,idx) - um(2,idx);
      idx += 2 * ni;
      shu[0][threadIdx.z][threadIdx.y+4][threadIdx.x+BX] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z][threadIdx.y+4][threadIdx.x+BX] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z][threadIdx.y+4][threadIdx.x+BX] = u(2,idx) - um(2,idx);
    }
    if (loader && loadk2) {
      int idx = index + 4 * nij;
      shu[0][threadIdx.z+4][threadIdx.y][threadIdx.x] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z+4][threadIdx.y][threadIdx.x] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z+4][threadIdx.y][threadIdx.x] = u(2,idx) - um(2,idx);
      idx += 2 * ni;
      shu[0][threadIdx.z+4][threadIdx.y+2][threadIdx.x] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z+4][threadIdx.y+2][threadIdx.x] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z+4][threadIdx.y+2][threadIdx.x] = u(2,idx) - um(2,idx);
      idx += 2 * ni;
      shu[0][threadIdx.z+4][threadIdx.y+4][threadIdx.x] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z+4][threadIdx.y+4][threadIdx.x] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z+4][threadIdx.y+4][threadIdx.x] = u(2,idx) - um(2,idx);
    }
    if (loadk2 && loadx2) {
      int idx = index + 4 * nij + BX;
      shu[0][threadIdx.z+4][threadIdx.y][threadIdx.x+BX] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z+4][threadIdx.y][threadIdx.x+BX] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z+4][threadIdx.y][threadIdx.x+BX] = u(2,idx) - um(2,idx);
      idx += 2 * ni;
      shu[0][threadIdx.z+4][threadIdx.y+2][threadIdx.x+BX] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z+4][threadIdx.y+2][threadIdx.x+BX] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z+4][threadIdx.y+2][threadIdx.x+BX] = u(2,idx) - um(2,idx);
      idx += 2 * ni;
      shu[0][threadIdx.z+4][threadIdx.y+4][threadIdx.x+BX] = u(0,idx) - um(0,idx);
      shu[1][threadIdx.z+4][threadIdx.y+4][threadIdx.x+BX] = u(1,idx) - um(1,idx);
      shu[2][threadIdx.z+4][threadIdx.y+4][threadIdx.x+BX] = u(2,idx) - um(2,idx);
    }
    __syncthreads();

    if (active) {

      // Load stry, coy, dcy, centered at j
      stry_j = a_stry[j];
      coy_j = a_coy[j];
      dcy_jm1 = a_dcy[j-1];
      dcy_j   = a_dcy[j];
      dcy_jp1 = a_dcy[j+1];

      // Natural indexing for (i,j,k)
      int idx = (k * nj + j) * ni + i;
      
      // Load rho directly
      rho_k   = a_rho[idx];
      rho_im1 = a_rho[idx-1];
      rho_ip1 = a_rho[idx+1];
      rho_jm1 = a_rho[idx-ni];
      rho_jp1 = a_rho[idx+ni];      
      rho_km1 = a_rho[idx-nij];
      rho_kp1 = a_rho[idx+nij];

      float_sw4 birho = beta / rho_k;
      
      // Update up(i,j,k)
      for (int c=0; c<3; c++) {
        up(c, idx) -= birho * 
          (// x-differences
           strx_i * coy_j * coz_k *
           (rho_ip1 * dcx_ip1 * (shu[c][tk][tj][ti+2] - 2 * shu[c][tk][tj][ti+1] + shu[c][tk][tj][ti  ]) -
            2 * rho_k * dcx_i * (shu[c][tk][tj][ti+1] - 2 * shu[c][tk][tj][ti  ] + shu[c][tk][tj][ti-1]) +
            rho_im1 * dcx_im1 * (shu[c][tk][tj][ti  ] - 2 * shu[c][tk][tj][ti-1] + shu[c][tk][tj][ti-2])) +
           // y-differences
           stry_j * cox_i * coz_k *
           (rho_jp1 * dcy_jp1 * (shu[c][tk][tj+2][ti] - 2 * shu[c][tk][tj+1][ti] + shu[c][tk][tj  ][ti]) -
            2 * rho_k * dcy_j * (shu[c][tk][tj+1][ti] - 2 * shu[c][tk][tj  ][ti] + shu[c][tk][tj-1][ti]) +
            rho_jm1 * dcy_jm1 * (shu[c][tk][tj  ][ti] - 2 * shu[c][tk][tj-1][ti] + shu[c][tk][tj-2][ti])) +
           // z-differences
           strz_k * cox_i * coy_j *
           (rho_kp1 * dcz_kp1 * (shu[c][tk+2][tj][ti] - 2 * shu[c][tk+1][tj][ti] + shu[c][tk  ][tj][ti]) -
            2 * rho_k * dcz_k * (shu[c][tk+1][tj][ti] - 2 * shu[c][tk  ][tj][ti] + shu[c][tk-1][tj][ti]) +
            rho_km1 * dcz_km1 * (shu[c][tk  ][tj][ti] - 2 * shu[c][tk-1][tj][ti] + shu[c][tk-2][tj][ti])));
      }
    }  // End active

    if (iside == 0)
      __syncthreads();

  } // End side loop
}


// *************************************************************************************
// Kernel launcher

void addsgd4_Y_gpu (int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		    int ni, int nj, int nk,
		    float_sw4* a_up, float_sw4* a_u, float_sw4* a_um, float_sw4* a_rho,
		    float_sw4* a_dcx,  float_sw4* a_dcy,  float_sw4* a_dcz,
		    float_sw4* a_strx, float_sw4* a_stry, float_sw4* a_strz,
		    float_sw4* a_cox,  float_sw4* a_coy,  float_sw4* a_coz,
		    float_sw4 beta, int c_order, cudaStream_t stream) {

  if (beta == 0.0)
    return;

  // Launch enough threads to cover [kfirst:klast]
  int nkcomp = klast - kfirst + 1;
  dim3 blocks = dim3((ni+BX-1)/BX, 1, (nkcomp+3)/4);
  dim3 threads = dim3(BX, 2, 4);

  if (c_order)
    addsgd4_Y<1><<<blocks,threads,0,stream>>> (ifirst, ilast, jfirst, jlast, kfirst, klast,
					       ni, nj, nk,
					       a_up, a_u, a_um, a_rho,
					       a_dcx,  a_dcy,  a_dcz,
					       a_strx, a_stry, a_strz,
					       a_cox,  a_coy,  a_coz, beta);
  else
    addsgd4_Y<0><<<blocks,threads,0,stream>>> (ifirst, ilast, jfirst, jlast, kfirst, klast,
					       ni, nj, nk,
					       a_up, a_u, a_um, a_rho,
					       a_dcx,  a_dcy,  a_dcz,
					       a_strx, a_stry, a_strz,
					       a_cox,  a_coy,  a_coz, beta);
}


// *************************************************************************************
// *************************************************************************************
// *************************************************************************************
template <bool c_order>
__launch_bounds__(BX*BY)
__global__ void bcfortsg (int ib, int ie, int jb, int je, int kb, int ke, int* wind,
			  int nx, int ny, int nz, float_sw4* a_u, float_sw4 h, boundaryConditionType *bccnd,
			  float_sw4* mu, float_sw4* la, float_sw4 t,
			  float_sw4* bforce1, float_sw4* bforce2, float_sw4* bforce3,
			  float_sw4* bforce4, float_sw4* bforce5, float_sw4* bforce6,
			  float_sw4 om, float_sw4 ph, float_sw4 cv,
			  float_sw4* strx, float_sw4* stry )
{
  const float_sw4 d4a = 2.0/3.0;
  const float_sw4 d4b = -1.0/12.0;
  const size_t ni  = ie-ib+1;
  const size_t nij = ni*(je-jb+1);
  const size_t nijk = nij*(ke-kb+1);
  for( int s=0 ; s < 6 ; s++ )
  {
    if( bccnd[s]==1 || bccnd[s]==2 )
    {
      size_t idel = 1+wind[1+6*s]-wind[6*s];
      size_t ijdel = idel * (1+wind[3+6*s]-wind[2+6*s]);
      int i = wind[6*s] + threadIdx.x + blockIdx.x*blockDim.x;
      int j = wind[6*s+2] + threadIdx.y + blockIdx.y*blockDim.y;
      int k = wind[6*s+4] +  threadIdx.z + blockIdx.z*blockDim.z;
      if( i > wind[6*s+1] || j > wind[6*s+3] || k > wind[6*s+5])
        continue;
      int qq = i-wind[6*s] + idel*(j-wind[6*s+2]) + (k-wind[4+6*s])*ijdel;
      size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
      if( s== 0 )
      {
        u(0,ind) = bforce1[  3*qq];
        u(1,ind) = bforce1[1+3*qq];
        u(2,ind) = bforce1[2+3*qq];
      }
      else if( s== 1 )
      {
        u(0,ind) = bforce2[  3*qq];
        u(1,ind) = bforce2[1+3*qq];
        u(2,ind) = bforce2[2+3*qq];
      }
      else if( s==2 )
      {
        u(0,ind) = bforce3[  3*qq];
        u(1,ind) = bforce3[1+3*qq];
        u(2,ind) = bforce3[2+3*qq];
      }
      else if( s==3 )
      {
        u(0,ind) = bforce4[  3*qq];
        u(1,ind) = bforce4[1+3*qq];
        u(2,ind) = bforce4[2+3*qq];
      }
      else if( s==4 )
      {
        u(0,ind) = bforce5[  3*qq];
        u(1,ind) = bforce5[1+3*qq];
        u(2,ind) = bforce5[2+3*qq];
      }
      else if( s==5 )
      {
        u(0,ind) = bforce6[  3*qq];
        u(1,ind) = bforce6[1+3*qq];
        u(2,ind) = bforce6[2+3*qq];
      }
    }
    else if( bccnd[s]==3 )
    {
      int i = wind[6*s] + threadIdx.x + blockIdx.x*blockDim.x;
      int j = wind[6*s+2] + threadIdx.y + blockIdx.y*blockDim.y;
      int k = wind[6*s+4] +  threadIdx.z + blockIdx.z*blockDim.z;
      if( i > wind[6*s+1] || j > wind[6*s+3] || k > wind[6*s+5])
        continue;
      size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
      if( s==0 )
      {
        size_t indp = ind+nx;
        u(0,ind) = u(0,indp);
        u(1,ind) = u(1,indp);
        u(2,ind) = u(2,indp);
      }
      else if( s==1 )
      {
        size_t indp= ind-nx;
        u(0,ind) = u(0,indp);
        u(1,ind) = u(1,indp);
        u(2,ind) = u(2,indp);
      }
      else if( s==2 )
      {
        size_t indp= ind+ni*ny;
        u(0,ind) = u(0,indp);
        u(1,ind) = u(1,indp);
        u(2,ind) = u(2,indp);
      }
      else if( s==3 )
      {
        size_t indp= ind-ni*ny;
        u(0,ind) = u(0,indp);
        u(1,ind) = u(1,indp);
        u(2,ind) = u(2,indp);
      }
      else if( s==4 )
      {
        size_t indp= ind+nij*nz;
        u(0,ind) = u(0,indp);
        u(1,ind) = u(1,indp);
        u(2,ind) = u(2,indp);
      }
      else if( s==5 )
      {
        size_t indp= ind-nij*nz;
        u(0,ind) = u(0,indp);
        u(1,ind) = u(1,indp);
        u(2,ind) = u(2,indp);
      }
    }
    else if( bccnd[s]==0 )
    {
      if( s != 4 && s != 5)
      {
        printf("EW::bcfortsg_dev_indrev, ERROR: Free surface conditio not implemented for side\n");
        return ;
      }

      int i = ib+2 + threadIdx.x + blockIdx.x*blockDim.x;
      int j = jb+2 + threadIdx.y + blockIdx.y*blockDim.y;
      int kk = kb+2 + threadIdx.z + blockIdx.z*blockDim.z;
      if( i > ie-2 || j > je-2 || kk != kb+2)
        continue;

      if( s==4 )
      {
        int k=1, kl=1;
        size_t qq = i-ib+ni*(j-jb);
        size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
        float_sw4 wx = strx[i-ib]*(d4a*(u(2,ind+1   )-u(2,ind-1   ))+
				   d4b*(u(2,ind+2   )-u(2,ind-2   )));
        float_sw4 ux = strx[i-ib]*(d4a*(u(0,ind+1   )-u(0,ind-1   ))+
				   d4b*(u(0,ind+2   )-u(0,ind-2   )));
        float_sw4 wy = stry[j-jb]*(d4a*(u(2,ind+ni  )-u(2,ind-ni  ))+
                                   d4b*(u(2,ind+2*ni)-u(2,ind-2*ni)));
        float_sw4 vy = stry[j-jb]*(d4a*(u(1,ind+ni  )-u(1,ind-ni  ))+
                                   d4b*(u(1,ind+2*ni)-u(1,ind-2*ni)));
        float_sw4 uz=0, vz=0, wz=0;
        for( int w=1 ; w <= 4 ; w++ )
        {
          uz += dev_sbop[w]*u(0,ind+nij*kl*(w-1));
          vz += dev_sbop[w]*u(1,ind+nij*kl*(w-1));
          wz += dev_sbop[w]*u(2,ind+nij*kl*(w-1));
        }
        u(0,ind-nij*kl) = (-uz-kl*wx+kl*h*bforce5[  3*qq]/mu[ind])/dev_sbop[0];
        u(1,ind-nij*kl) = (-vz-kl*wy+kl*h*bforce5[1+3*qq]/mu[ind])/dev_sbop[0];
	u(2,ind-nij*kl) = (-wz + (-kl*la[ind]*(ux+vy)+kl*h*bforce5[2+3*qq])/
                                (2*mu[ind]+la[ind]))/dev_sbop[0];
      }
      else
      {
        int k=nz, kl=-1;
        size_t qq = i-ib+ni*(j-jb);
        size_t ind = i-ib+ni*(j-jb)+nij*(k-kb);
        float_sw4 wx = strx[i-ib]*(d4a*(u(2,ind+1   )-u(2,ind-1   ))+
				   d4b*(u(2,ind+2   )-u(2,ind-2   )));
        float_sw4 ux = strx[i-ib]*(d4a*(u(0,ind+1   )-u(0,ind-1   ))+
				   d4b*(u(0,ind+2   )-u(0,ind-2   )));
        float_sw4 wy = stry[j-jb]*(d4a*(u(2,ind+ni  )-u(2,ind-ni  ))+
                                   d4b*(u(2,ind+2*ni)-u(2,ind-2*ni)));
        float_sw4 vy = stry[j-jb]*(d4a*(u(1,ind+ni)  -u(1,ind-ni  ))+
                                   d4b*(u(1,ind+2*ni)-u(1,ind-2*ni)));
        float_sw4 uz=0, vz=0, wz=0;
        for( int w=1 ; w <= 4 ; w++ )
        {
          uz += dev_sbop[w]*u(0,ind+nij*kl*(w-1));
          vz += dev_sbop[w]*u(1,ind+nij*kl*(w-1));
          wz += dev_sbop[w]*u(2,ind+nij*kl*(w-1));
        }
        u(0,ind-nij*kl) = (-uz-kl*wx+kl*h*bforce6[  3*qq]/mu[ind])/dev_sbop[0];
        u(1,ind-nij*kl) = (-vz-kl*wy+kl*h*bforce6[1+3*qq]/mu[ind])/dev_sbop[0];
        u(2,ind-nij*kl) = (-wz+(-kl*la[ind]*(ux+vy)+kl*h*bforce6[2+3*qq])/
                                (2*mu[ind]+la[ind]))/dev_sbop[0];

      }
    }
  }
}

// *************************************************************************************
// Launcher
void bcfortsg_gpu (int ib, int ie, int jb, int je, int kb, int ke, int* wind,
		   int nx, int ny, int nz, float_sw4* a_u, float_sw4 h, boundaryConditionType *bccnd,
		   float_sw4* mu, float_sw4* la, float_sw4 t,
		   float_sw4* bforce1, float_sw4* bforce2, float_sw4* bforce3,
		   float_sw4* bforce4, float_sw4* bforce5, float_sw4* bforce6,
		   float_sw4 om, float_sw4 ph, float_sw4 cv,
		   float_sw4* strx, float_sw4* stry, int c_order, cudaStream_t stream) {
  int nicomp = ie - ib + 1;
  int njcomp = je - jb + 1;
  dim3 blocks = dim3((nicomp+BX-1)/BX, (njcomp+BY-1)/BY, 1);
  dim3 threads = dim3(BX, BY, 1);
  if (c_order)
    bcfortsg<true><<<blocks,threads,0,stream>>>
      (ib, ie, jb, je, kb, ke, wind,
       nx, ny, nz, a_u, h, bccnd,
       mu, la, t,
       bforce1, bforce2, bforce3,
       bforce4, bforce5, bforce6,
       om, ph, cv,
       strx, stry);
  else
    bcfortsg<false><<<blocks,threads,0,stream>>>
      (ib, ie, jb, je, kb, ke, wind,
       nx, ny, nz, a_u, h, bccnd,
       mu, la, t,
       bforce1, bforce2, bforce3,
       bforce4, bforce5, bforce6,
       om, ph, cv,
       strx, stry);
}

//-----------------------------------------------------------------------
__global__ void HaloToBufferKernelX_dev(float_sw4* block_left, float_sw4* block_right,
                                        float_sw4 * leftSideEdge, float_sw4 * rightSideEdge,
                                        int ni, int nj, int nk, int m_padding, const int m_neighbor2,
                                        const int m_neighbor3, const int mpi_process_null)
{
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = 3*m_padding*ni;
   size_t npts = static_cast<size_t>(nk);

   if( m_neighbor2 !=  mpi_process_null)
      for (i = myi; i < n_m_padding*npts; i += nthreads)
      {
         idx = i/n_m_padding;
         j  = i - idx*n_m_padding;
         istart = idx*(3*ni*nj);
         block_left[istart+j] = leftSideEdge[i];
      }


   if( m_neighbor3 !=  mpi_process_null)
      for (i = myi; i < n_m_padding*npts; i += nthreads)
      {
         idx = i/n_m_padding;
         j  = i - idx*n_m_padding;
         istart = idx*(3*ni*nj);
         block_right[istart+j] = rightSideEdge[i];
      }
}

//-----------------------------------------------------------------------
__global__ void HaloToBufferKernelX_dev_rev(float_sw4* block_left, float_sw4* block_right,
                                            float_sw4 * leftSideEdge, float_sw4 * rightSideEdge,
                                            int ni, int nj, int nk, int m_padding, const int m_neighbor2,
                                            const int m_neighbor3, const int mpi_process_null)
{
   int njk = nj*nk;
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = m_padding*ni;
   size_t npts = static_cast<size_t>(3*nk);

   if( m_neighbor2 !=  mpi_process_null)
      for (i = myi; i < n_m_padding*npts; i += nthreads)
      {
         idx = i/n_m_padding;
         j  = i - idx*n_m_padding;
         istart = idx*(ni*nj);
         block_left[istart+j] = leftSideEdge[i];
      }


   if( m_neighbor3 !=  mpi_process_null)
      for (i = myi; i < n_m_padding*npts; i += nthreads)
      {
         idx = i/n_m_padding;
         j  = i - idx*n_m_padding;
         istart = idx*(ni*nj);
         block_right[istart+j] = rightSideEdge[i];
      }

}

//-----------------------------------------------------------------------
__global__ void HaloToBufferKernelY_dev(float_sw4* block_up, float_sw4* block_down,
                                        float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                                        int ni, int nj, int nk, int m_padding, const int m_neighbor0,
                                        const int  m_neighbor1, const int mpi_process_null)
{
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = 3*m_padding;
   size_t npts = static_cast<size_t>(nj*nk);

   if( m_neighbor1 !=  mpi_process_null)
      for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
      {
         idx = i/(n_m_padding);
         j  = i - idx*n_m_padding;
         istart = idx*(3*ni);
         block_up[istart+j] = upSideEdge[i];
      }

   if( m_neighbor0 !=  mpi_process_null)
      for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
      {
         idx = i/(n_m_padding);
         j  = i - idx*n_m_padding;
         istart = idx*(3*ni);
         block_down[istart+j] = downSideEdge[i];
      }
}

//-----------------------------------------------------------------------
__global__ void HaloToBufferKernelY_dev_rev(float_sw4* block_up, float_sw4* block_down,
                                            float_sw4 * upSideEdge, float_sw4 * downSideEdge,
                                            int ni, int nj, int nk, int m_padding, const int m_neighbor0,
                                            const int  m_neighbor1, const int mpi_process_null)
{
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = m_padding;
   size_t npts = static_cast<size_t>(3*nj*nk);

   if( m_neighbor1 !=  mpi_process_null)
      for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
      {
         idx = i/(n_m_padding);
         j  = i - idx*n_m_padding;
         istart = idx*ni;
         block_up[istart+j] = upSideEdge[i];
      }

   if( m_neighbor0 !=  mpi_process_null)
      for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
      {
         idx = i/(n_m_padding);
         j  = i - idx*n_m_padding;
         istart = idx*ni;
         block_down[istart+j] = downSideEdge[i];
      }
}

//-----------------------------------------------------------------------
__global__ void BufferToHaloKernelX_dev(float_sw4* block_left, float_sw4* block_right, float_sw4 * leftSideEdge, 
                                        float_sw4 * rightSideEdge, int ni, int nj, int nk, int m_padding, 
                                        const int m_neighbor2, const int m_neighbor3, const int mpi_process_null )
{
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = 3*m_padding*ni;
   size_t npts = static_cast<size_t>(nk);

   if( m_neighbor2 !=  mpi_process_null)
   for (i = myi; i < n_m_padding*npts; i += nthreads)
   {
        idx = i/n_m_padding;
        j  = i - idx*n_m_padding;
        istart = idx*(3*ni*nj);
        leftSideEdge[i] = block_left[istart+j];
   }

   if( m_neighbor3 !=  mpi_process_null)
   for (i = myi; i < n_m_padding*npts; i += nthreads)
   {
        idx = i/n_m_padding;
        j  = i - idx*n_m_padding;
        istart = idx*(3*ni*nj);
        rightSideEdge[i] = block_right[istart+j];
   }

}

//-----------------------------------------------------------------------
__global__ void BufferToHaloKernelX_dev_rev(float_sw4* block_left, float_sw4* block_right,
                                            float_sw4 * leftSideEdge, float_sw4 * rightSideEdge, 
                                            int ni, int nj, int nk, int m_padding, const int m_neighbor2,
                                            const int m_neighbor3, const int mpi_process_null)
{
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = m_padding*ni;
   size_t npts = static_cast<size_t>(3*nk);

   if( m_neighbor2 !=  mpi_process_null)
   for (i = myi; i < n_m_padding*npts; i += nthreads)
   {
        idx = i/n_m_padding;
        j  = i - idx*n_m_padding;
        istart = idx*(ni*nj);
        leftSideEdge[i] = block_left[istart+j];
   }

   if( m_neighbor3 !=  mpi_process_null)
   for (i = myi; i < n_m_padding*npts; i += nthreads)
   {
        idx = i/n_m_padding;
        j  = i - idx*n_m_padding;
        istart = idx*(ni*nj);
        rightSideEdge[i] = block_right[istart+j];
   }
}

//-----------------------------------------------------------------------
__global__ void BufferToHaloKernelY_dev(float_sw4* block_up, float_sw4* block_down,
                                        float_sw4* upSideEdge, float_sw4* downSideEdge,
                                        int ni, int nj, int nk, int m_padding, const int m_neighbor0 ,
                                        const int  m_neighbor1, const int mpi_process_null )
{
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = 3*m_padding;
   size_t npts = static_cast<size_t>(nj*nk);

   if( m_neighbor1 !=  mpi_process_null)
   for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
   {
        idx = i/(n_m_padding);
        j  = i - idx*n_m_padding;
        istart = idx*(3*ni);
        upSideEdge[i] = block_up[istart+j];
   }

   if( m_neighbor0 !=  mpi_process_null)
   for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
   {
        idx = i/(n_m_padding);
        j  = i - idx*n_m_padding;
        istart = idx*(3*ni);
        downSideEdge[i] = block_down[istart+j];
   }
}


//-----------------------------------------------------------------------
__global__ void BufferToHaloKernelY_dev_rev(float_sw4* block_up, float_sw4* block_down,
                                            float_sw4* upSideEdge, float_sw4* downSideEdge,
                                            int ni, int nj, int nk, int m_padding, const int m_neighbor0,
                                            const int m_neighbor1, const int mpi_process_null )
{
   int n_m_padding, istart, idx_halo,  i, idx, j;
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myi = threadIdx.x + blockIdx.x * blockDim.x;

   n_m_padding = m_padding;
   size_t npts = static_cast<size_t>(3*nj*nk);

   if( m_neighbor1 !=  mpi_process_null)
   for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
   {
        idx = i/(n_m_padding);
        j  = i - idx*n_m_padding;
        istart = idx*ni;
        upSideEdge[i] = block_up[istart+j];
   }

   if( m_neighbor0 !=  mpi_process_null)
   for ( size_t i = myi; i < n_m_padding*npts; i += nthreads )
   {
        idx = i/(n_m_padding);
        j  = i - idx*n_m_padding;
        istart = idx*ni;
        downSideEdge[i] = block_down[istart+j];
   }

}

#include "TimeSeries.h"

// Code to extract time series data on GPU
//-----------------------------------------------------------------------
__global__ void extractRecordData_dev( int nt, int* mode, int* i0v, int* j0v, int* k0v,
				       int* g0v, float_sw4** uRec, Sarray* Um2, Sarray* U,
				       float_sw4 dt, float_sw4* h, int numberOfCartesianGrids,
				       Sarray* Metric, Sarray* J )
{
   size_t nthreads = static_cast<size_t> (gridDim.x) * (blockDim.x);
   size_t myid = threadIdx.x + blockIdx.x * blockDim.x;
   Sarray& mMetric = *Metric;
   Sarray& mJ = *J;
   for( size_t ts=myid ; ts < nt ; ts += nthreads )
   {
      int i0 = i0v[ts];
      int j0 = j0v[ts];
      int k0 = k0v[ts];
      int g0 = g0v[ts];
      if (mode[ts] == TimeSeries::Displacement)
      {
	 uRec[ts][0] = U[g0](1, i0, j0, k0,true);
	 uRec[ts][1] = U[g0](2, i0, j0, k0,true);
	 uRec[ts][2] = U[g0](3, i0, j0, k0,true);
      }
      else if (mode[ts] ==  TimeSeries::Velocity)
      {
	 uRec[ts][0] = (U[g0](1, i0, j0, k0,true) - Um2[g0](1, i0, j0, k0,true))/(2*dt);
	 uRec[ts][1] = (U[g0](2, i0, j0, k0,true) - Um2[g0](2, i0, j0, k0,true))/(2*dt);
	 uRec[ts][2] = (U[g0](3, i0, j0, k0,true) - Um2[g0](3, i0, j0, k0,true))/(2*dt);
      }
      else if(mode[ts] ==  TimeSeries::Div)
      {
	 if (g0 < numberOfCartesianGrids) // must be a Cartesian grid
	 {
	    float_sw4 factor = 1.0/(2*h[g0]);
	    uRec[ts][0] = ((U[g0](1,i0+1, j0, k0,true) - U[g0](1,i0-1, j0, k0,true)+
			    U[g0](2,i0, j0+1, k0,true) - U[g0](2,i0, j0-1, k0,true)+
			    U[g0](3,i0, j0, k0+1,true) - U[g0](3,i0, j0, k0-1,true))*factor);
	 }
	 else // must be curvilinear
	 {
	    float_sw4 factor = 0.5/sqrt(mJ(i0,j0,k0,true));
	    uRec[ts][0] = ( ( mMetric(1,i0,j0,k0,true)*(U[g0](1,i0+1,j0,k0,true) - U[g0](1,i0-1,j0,k0,true))+
			      mMetric(1,i0,j0,k0,true)*(U[g0](2,i0,j0+1,k0,true) - U[g0](2,i0,j0-1,k0,true))+
			      mMetric(2,i0,j0,k0,true)*(U[g0](1,i0,j0,k0+1,true) - U[g0](1,i0,j0,k0-1,true))+
			      mMetric(3,i0,j0,k0,true)*(U[g0](2,i0,j0,k0+1,true) - U[g0](2,i0,j0,k0-1,true))+
			      mMetric(4,i0,j0,k0,true)*(U[g0](3,i0,j0,k0+1,true) - U[g0](3,i0,j0,k0-1,true))  )*factor);
	 }
      } // end div
      else if(mode[ts] ==  TimeSeries::Curl)
      {
	 if (g0 < numberOfCartesianGrids) // must be a Cartesian grid
	 {
	    float_sw4 factor = 1.0/(2*h[g0]);
	    float_sw4 duydx = (U[g0](2,i0+1,j0,k0,true) - U[g0](2,i0-1,j0,k0,true))*factor;
	    float_sw4 duzdx = (U[g0](3,i0+1,j0,k0,true) - U[g0](3,i0-1,j0,k0,true))*factor;
	    float_sw4 duxdy = (U[g0](1,i0,j0+1,k0,true) - U[g0](1,i0,j0-1,k0,true))*factor;
	    float_sw4 duzdy = (U[g0](3,i0,j0+1,k0,true) - U[g0](3,i0,j0-1,k0,true))*factor;
	    float_sw4 duxdz = (U[g0](1,i0,j0,k0+1,true) - U[g0](1,i0,j0,k0-1,true))*factor;
	    float_sw4 duydz = (U[g0](2,i0,j0,k0+1,true) - U[g0](2,i0,j0,k0-1,true))*factor;
	    uRec[ts][0] = ( duzdy-duydz );
	    uRec[ts][1] = ( duxdz-duzdx );
	    uRec[ts][2] = ( duydx-duxdy );
	 }
	 else // must be curvilinear
	 {
	    float_sw4 factor = 0.5/sqrt(mJ(i0,j0,k0,true));
	    float_sw4 duxdq = (U[g0](1,i0+1,j0,k0,true) - U[g0](1,i0-1,j0,k0,true));
	    float_sw4 duydq = (U[g0](2,i0+1,j0,k0,true) - U[g0](2,i0-1,j0,k0,true));
	    float_sw4 duzdq = (U[g0](3,i0+1,j0,k0,true) - U[g0](3,i0-1,j0,k0,true));
	    float_sw4 duxdr = (U[g0](1,i0,j0+1,k0,true) - U[g0](1,i0,j0-1,k0,true));
	    float_sw4 duydr = (U[g0](2,i0,j0+1,k0,true) - U[g0](2,i0,j0-1,k0,true));
	    float_sw4 duzdr = (U[g0](3,i0,j0+1,k0,true) - U[g0](3,i0,j0-1,k0,true));
	    float_sw4 duxds = (U[g0](1,i0,j0,k0+1,true) - U[g0](1,i0,j0,k0-1,true));
	    float_sw4 duyds = (U[g0](2,i0,j0,k0+1,true) - U[g0](2,i0,j0,k0-1,true));
	    float_sw4 duzds = (U[g0](3,i0,j0,k0+1,true) - U[g0](3,i0,j0,k0-1,true));

	    float_sw4 duydx = mMetric(1,i0,j0,k0,true)*duydq+mMetric(2,i0,j0,k0,true)*duyds;
	    float_sw4 duzdx = mMetric(1,i0,j0,k0,true)*duzdq+mMetric(2,i0,j0,k0,true)*duzds;
	    float_sw4 duxdy = mMetric(1,i0,j0,k0,true)*duxdr+mMetric(3,i0,j0,k0,true)*duxds;
	    float_sw4 duzdy = mMetric(1,i0,j0,k0,true)*duzdr+mMetric(3,i0,j0,k0,true)*duzds;
	    float_sw4 duydz = mMetric(4,i0,j0,k0,true)*duyds;
	    float_sw4 duxdz = mMetric(4,i0,j0,k0,true)*duxds;

	    uRec[ts][0] = (duzdy-duydz)*factor;
	    uRec[ts][1] = (duxdz-duzdx)*factor;
	    uRec[ts][2] = (duydx-duxdy)*factor;
	 }
      } // end Curl
      else if(mode[ts] ==  TimeSeries::Strains )
      {
	 if (g0 < numberOfCartesianGrids) // must be a Cartesian grid
	 {
	    float_sw4 factor = 1.0/(2*h[g0]);
	    float_sw4 duydx = (U[g0](2,i0+1,j0,k0,true) - U[g0](2,i0-1,j0,k0,true))*factor;
	    float_sw4 duzdx = (U[g0](3,i0+1,j0,k0,true) - U[g0](3,i0-1,j0,k0,true))*factor;
	    float_sw4 duxdy = (U[g0](1,i0,j0+1,k0,true) - U[g0](1,i0,j0-1,k0,true))*factor;
	    float_sw4 duzdy = (U[g0](3,i0,j0+1,k0,true) - U[g0](3,i0,j0-1,k0,true))*factor;
	    float_sw4 duxdz = (U[g0](1,i0,j0,k0+1,true) - U[g0](1,i0,j0,k0-1,true))*factor;
	    float_sw4 duydz = (U[g0](2,i0,j0,k0+1,true) - U[g0](2,i0,j0,k0-1,true))*factor;
	    float_sw4 duxdx = (U[g0](1,i0+1,j0,k0,true) - U[g0](1,i0-1,j0,k0,true))*factor;
	    float_sw4 duydy = (U[g0](2,i0,j0+1,k0,true) - U[g0](2,i0,j0-1,k0,true))*factor;
	    float_sw4 duzdz = (U[g0](3,i0,j0,k0+1,true) - U[g0](3,i0,j0,k0-1,true))*factor;
	    uRec[ts][0] = ( duxdx );
	    uRec[ts][1] = ( duydy );
	    uRec[ts][2] = ( duzdz );
	    uRec[ts][3] = ( 0.5*(duydx+duxdy) );
	    uRec[ts][4] = ( 0.5*(duzdx+duxdz) );
	    uRec[ts][5] = ( 0.5*(duydz+duzdy) );
	 }
	 else // must be curvilinear
	 {
	    float_sw4 factor = 0.5/sqrt(mJ(i0,j0,k0,true));
	    float_sw4 duxdq = (U[g0](1,i0+1,j0,k0,true) - U[g0](1,i0-1,j0,k0,true));
	    float_sw4 duydq = (U[g0](2,i0+1,j0,k0,true) - U[g0](2,i0-1,j0,k0,true));
	    float_sw4 duzdq = (U[g0](3,i0+1,j0,k0,true) - U[g0](3,i0-1,j0,k0,true));
	    float_sw4 duxdr = (U[g0](1,i0,j0+1,k0,true) - U[g0](1,i0,j0-1,k0,true));
	    float_sw4 duydr = (U[g0](2,i0,j0+1,k0,true) - U[g0](2,i0,j0-1,k0,true));
	    float_sw4 duzdr = (U[g0](3,i0,j0+1,k0,true) - U[g0](3,i0,j0-1,k0,true));
	    float_sw4 duxds = (U[g0](1,i0,j0,k0+1,true) - U[g0](1,i0,j0,k0-1,true));
	    float_sw4 duyds = (U[g0](2,i0,j0,k0+1,true) - U[g0](2,i0,j0,k0-1,true));
	    float_sw4 duzds = (U[g0](3,i0,j0,k0+1,true) - U[g0](3,i0,j0,k0-1,true));
	    float_sw4 duzdy = (mMetric(1,i0,j0,k0,true)*duzdr+mMetric(3,i0,j0,k0,true)*duzds)*factor;
	    float_sw4 duydz = (mMetric(4,i0,j0,k0,true)*duyds)*factor;
	    float_sw4 duxdz = (mMetric(4,i0,j0,k0,true)*duxds)*factor;
	    float_sw4 duzdx = (mMetric(1,i0,j0,k0,true)*duzdq+mMetric(2,i0,j0,k0,true)*duzds)*factor;
	    float_sw4 duydx = (mMetric(1,i0,j0,k0,true)*duydq+mMetric(2,i0,j0,k0,true)*duyds)*factor;
	    float_sw4 duxdy = (mMetric(1,i0,j0,k0,true)*duxdr+mMetric(3,i0,j0,k0,true)*duxds)*factor;
	    float_sw4 duxdx = ( mMetric(1,i0,j0,k0,true)*(U[g0](1,i0+1,j0,k0,true) - U[g0](1,i0-1,j0,k0,true))+
				mMetric(2,i0,j0,k0,true)*(U[g0](1,i0,j0,k0+1,true) - U[g0](1,i0,j0,k0-1,true)) )*factor;
	    float_sw4 duydy = ( mMetric(1,i0,j0,k0,true)*(U[g0](2,i0,j0+1,k0,true) - U[g0](2,i0,j0-1,k0,true))+
				mMetric(3,i0,j0,k0,true)*(U[g0](2,i0,j0,k0+1,true) - U[g0](2,i0,j0,k0-1,true)) )*factor;
	    float_sw4 duzdz = ( mMetric(4,i0,j0,k0,true)*(U[g0](3,i0,j0,k0+1,true) - U[g0](3,i0,j0,k0-1,true)) )*factor;
	    uRec[ts][0] = ( duxdx );
	    uRec[ts][1] = ( duydy );
	    uRec[ts][2] = ( duzdz );
	    uRec[ts][3] = ( 0.5*(duydx+duxdy) );
	    uRec[ts][4] = ( 0.5*(duzdx+duxdz) );
	    uRec[ts][5] = ( 0.5*(duydz+duzdy) );
	 }
      } // end Strains
      else if(mode[ts] ==  TimeSeries::DisplacementGradient )
      {
	 if (g0 < numberOfCartesianGrids) // must be a Cartesian grid
	 {
	    float_sw4 factor = 1.0/(2*h[g0]);
	    float_sw4 duydx = (U[g0](2,i0+1,j0,k0,true) - U[g0](2,i0-1,j0,k0,true))*factor;
	    float_sw4 duzdx = (U[g0](3,i0+1,j0,k0,true) - U[g0](3,i0-1,j0,k0,true))*factor;
	    float_sw4 duxdy = (U[g0](1,i0,j0+1,k0,true) - U[g0](1,i0,j0-1,k0,true))*factor;
	    float_sw4 duzdy = (U[g0](3,i0,j0+1,k0,true) - U[g0](3,i0,j0-1,k0,true))*factor;
	    float_sw4 duxdz = (U[g0](1,i0,j0,k0+1,true) - U[g0](1,i0,j0,k0-1,true))*factor;
	    float_sw4 duydz = (U[g0](2,i0,j0,k0+1,true) - U[g0](2,i0,j0,k0-1,true))*factor;
	    float_sw4 duxdx = (U[g0](1,i0+1,j0,k0,true) - U[g0](1,i0-1,j0,k0,true))*factor;
	    float_sw4 duydy = (U[g0](2,i0,j0+1,k0,true) - U[g0](2,i0,j0-1,k0,true))*factor;
	    float_sw4 duzdz = (U[g0](3,i0,j0,k0+1,true) - U[g0](3,i0,j0,k0-1,true))*factor;
	    uRec[ts][0] =  duxdx;
	    uRec[ts][1] =  duxdy;
	    uRec[ts][2] =  duxdz;
	    uRec[ts][3] =  duydx;
	    uRec[ts][4] =  duydy;
	    uRec[ts][5] =  duydz;
	    uRec[ts][6] =  duzdx;
	    uRec[ts][7] =  duzdy;
	    uRec[ts][8] =  duzdz;
	 }
	 else // must be curvilinear
	 {
	    float_sw4 factor = 0.5/sqrt(mJ(i0,j0,k0,true));
	    float_sw4 duxdq = (U[g0](1,i0+1,j0,k0,true) - U[g0](1,i0-1,j0,k0,true));
	    float_sw4 duydq = (U[g0](2,i0+1,j0,k0,true) - U[g0](2,i0-1,j0,k0,true));
	    float_sw4 duzdq = (U[g0](3,i0+1,j0,k0,true) - U[g0](3,i0-1,j0,k0,true));
	    float_sw4 duxdr = (U[g0](1,i0,j0+1,k0,true) - U[g0](1,i0,j0-1,k0,true));
	    float_sw4 duydr = (U[g0](2,i0,j0+1,k0,true) - U[g0](2,i0,j0-1,k0,true));
	    float_sw4 duzdr = (U[g0](3,i0,j0+1,k0,true) - U[g0](3,i0,j0-1,k0,true));
	    float_sw4 duxds = (U[g0](1,i0,j0,k0+1,true) - U[g0](1,i0,j0,k0-1,true));
	    float_sw4 duyds = (U[g0](2,i0,j0,k0+1,true) - U[g0](2,i0,j0,k0-1,true));
	    float_sw4 duzds = (U[g0](3,i0,j0,k0+1,true) - U[g0](3,i0,j0,k0-1,true));
	    float_sw4 duzdy = (mMetric(1,i0,j0,k0,true)*duzdr+mMetric(3,i0,j0,k0,true)*duzds)*factor;
	    float_sw4 duydz = (mMetric(4,i0,j0,k0,true)*duyds)*factor;
	    float_sw4 duxdz = (mMetric(4,i0,j0,k0,true)*duxds)*factor;
	    float_sw4 duzdx = (mMetric(1,i0,j0,k0,true)*duzdq+mMetric(2,i0,j0,k0,true)*duzds)*factor;
	    float_sw4 duydx = (mMetric(1,i0,j0,k0,true)*duydq+mMetric(2,i0,j0,k0,true)*duyds)*factor;
	    float_sw4 duxdy = (mMetric(1,i0,j0,k0,true)*duxdr+mMetric(3,i0,j0,k0,true)*duxds)*factor;
	    float_sw4 duxdx = ( mMetric(1,i0,j0,k0,true)*(U[g0](1,i0+1,j0,k0,true) - U[g0](1,i0-1,j0,k0,true))+
				mMetric(2,i0,j0,k0,true)*(U[g0](1,i0,j0,k0+1,true) - U[g0](1,i0,j0,k0-1,true)) )*factor;
	    float_sw4 duydy = ( mMetric(1,i0,j0,k0,true)*(U[g0](2,i0,j0+1,k0,true) - U[g0](2,i0,j0-1,k0,true))+
				mMetric(3,i0,j0,k0,true)*(U[g0](2,i0,j0,k0+1,true) - U[g0](2,i0,j0,k0-1,true)) )*factor;
	    float_sw4 duzdz = ( mMetric(4,i0,j0,k0,true)*(U[g0](3,i0,j0,k0+1,true) - U[g0](3,i0,j0,k0-1,true)) )*factor;
	    uRec[ts][0] =  duxdx;
	    uRec[ts][1] =  duxdy;
	    uRec[ts][2] =  duxdz;
	    uRec[ts][3] =  duydx;
	    uRec[ts][4] =  duydy;
	    uRec[ts][5] =  duydz;
	    uRec[ts][6] =  duzdx;
	    uRec[ts][7] =  duzdy;
	    uRec[ts][8] =  duzdz;
	 }
      }// end DisplacementGradient
   } 
}

#endif
