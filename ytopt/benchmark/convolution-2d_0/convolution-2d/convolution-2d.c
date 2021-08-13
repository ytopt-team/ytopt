/* POLYBENCH/GPU-OPENMP
 *
 * This file is a part of the Polybench/GPU-OpenMP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 * 
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4096x4096. */
#include "convolution-2d.h"

/* Array initialization. */
static
void init_array (int ni, int nj,
		 DATA_TYPE POLYBENCH_2D_CUDA(A,NI,NJ,ni,nj))
{

  //	printf("Initializing Array\n");
      	int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      {
	POLYBENCH_2D_REF(A,NI,NJ,i,j) = ((DATA_TYPE) (i + j) / nj);
      }

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D_CUDA(B,NI,NJ,ni,nj))

{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, POLYBENCH_2D_REF(B,NI,NJ,i,j));
      if ((i * NJ + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_conv2d(int ni,
		   int nj,
		   DATA_TYPE POLYBENCH_2D_CUDA(A,NI,NJ,ni,nj),
		   DATA_TYPE POLYBENCH_2D_CUDA(B,NI,NJ,ni,nj))
{
  #P1
  for (int i = 1; i < NI - 1; ++i)
  {
    for (int j = 1; j < NJ - 1; ++j)
    {
      POLYBENCH_2D_REF(B,NI,NJ,i,j) = 
	               0.2 * POLYBENCH_2D_REF(A,NI,NJ,i-1,j-1) + 
                       0.5 * POLYBENCH_2D_REF(A,NI,NJ,i-1,j) + 
                      -0.8 * POLYBENCH_2D_REF(A,NI,NJ,i-1,j+1) +
                      -0.3 * POLYBENCH_2D_REF(A,NI,NJ,i,j-1) +
                       0.6 * POLYBENCH_2D_REF(A,NI,NJ,i,j) +
                      -0.9 * POLYBENCH_2D_REF(A,NI,NJ,i,j+1) +
                       0.4 * POLYBENCH_2D_REF(A,NI,NJ,i+1,j-1) +
                       0.7 * POLYBENCH_2D_REF(A,NI,NJ,i+1,j) +
                       0.1 * POLYBENCH_2D_REF(A,NI,NJ,i+1,j+1);
    } 
  }
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL_CUDA(A, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL_CUDA(B, DATA_TYPE, NI, NJ, ni, nj);

  /* Initialize array(s). */
  init_array (ni, nj, POLYBENCH_ARRAY_CUDA(A));
  
  //print_array(ni, nj, POLYBENCH_ARRAY_CUDA(A));
  /* Start timer. */
  //polybench_start_instruments;
polybench_timer_start();

  /* Run kernel. */
  kernel_conv2d (ni, nj, POLYBENCH_ARRAY_CUDA(A), POLYBENCH_ARRAY_CUDA(B));

  /* Stop and print timer. */
polybench_timer_stop();
polybench_timer_print();
  //polybench_stop_instruments;
  //polybench_print_instruments;
  
  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY_CUDA(B)));

  /* Be clean. */
  //  POLYBENCH_FREE_ARRAY(A);
  // POLYBENCH_FREE_ARRAY(B);
  
  return 0;
}
