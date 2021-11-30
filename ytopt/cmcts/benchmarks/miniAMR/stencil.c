// ************************************************************************
//
// miniAMR: stencil computations with boundary exchange and AMR.
//
// Copyright (2014) Sandia Corporation. Under the terms of Contract
// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government
// retains certain rights in this software.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
// Questions? Contact Courtenay T. Vaughan (ctvaugh@sandia.gov)
//                    Richard F. Barrett (rfbarre@sandia.gov)
//
// ************************************************************************

#include <mpi.h>
#include <omp.h>
#include <math.h>

#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "block.h"
#include "comm.h"
#include "proto.h"

#include "polybench.h"

void stencil_calc(int, int);
void stencil_0(int);
void stencil_x(int);
void stencil_y(int);
void stencil_z(int);
void stencil_7(int);
void stencil_27(int);
void stencil_check(int);
int num_cells;

// This routine does the stencil calculations.
void stencil_driver(int var, int cacl_stage)
{
   num_cells = x_block_size*y_block_size*z_block_size;

   if (stencil)
      stencil_calc(var, stencil);
   else {
      assert(!"Supporting stencil=7 for now");
      //FIXME: extend refactoring to other stencils
      //if (!var)
      //   stencil_calc(var, 7);
      //else if (var < 4*mat) {
      //   switch (cacl_stage%6) {
      //   case 0:
      //      stencil_0(var);
      //      break;
      //   case 1:
      //      stencil_x(var);
      //      break;
      //   case 2:
      //      stencil_y(var);
      //      break;
      //   case 3:
      //      stencil_z(var);
      //      break;
      //   case 4:
      //      stencil_7(var);
      //      break;
      //   case 5:
      //      stencil_27(var);
      //      break;
      //   }
      //   stencil_check(var);
      //} else
      //    stencil_calc(var, 7);
   }
}



static __attribute__((noinline))
void stencil7_calc_all(int num_indices, int x_size, int y_size, int z_size, double allarray[static const restrict num_indices][x_size][y_size][z_size], double allwork[static const restrict num_indices][x_size][y_size][z_size]) {
  // #pragma omp parallel for default(shared) private(i, j, k, bp)



//#pragma clang loop(in) parallelize_thread
//#pragma clang loop(i,j,k) tile sizes(8,16,16)


//#pragma clang loop id(in)
  for (int in = 0; in < num_indices; in++)
//#pragma clang loop id(i)
    for (int i = 1; i <= x_size-2; i++)
//#pragma clang loop id(j)
      for (int j = 1; j <= y_size-2; j++)
//#pragma clang loop id(k)
        for (int k = 1; k <= z_size-2; k++)
          allarray[in][i][j][k] = (allwork[in][i-1][j  ][k  ] +
            allwork[in][i  ][j-1][k  ] +
            allwork[in][i  ][j  ][k-1] +
            allwork[in][i  ][j  ][k  ] +
            allwork[in][i  ][j  ][k+1] +
            allwork[in][i  ][j+1][k  ] +
            allwork[in][i+1][j  ][k  ])/7.0;
}


static void stencil7_calc(int var, int num_indices, int x_size, int y_size, int z_size) {
  char *mem = malloc(2ll*num_indices*x_size*y_size*z_size*sizeof(double));
  typedef double (block3D_t)[y_size][z_size];
  double (*allwork)[x_size][y_size][z_size] = (void*)mem;
  double (*allarray)[x_size][y_size][z_size] = (void*)(mem+num_indices*x_size*y_size*z_size*sizeof(double));
  double data[x_size][y_size][z_size];

  // copy-in
  for (int in = 0; in < num_indices; in++) {
    block* bp = &blocks[sorted_list[in].n];
    void* array = &bp->array[var * x_size * y_size * z_size];
    void* work = &allwork[in];

    memcpy(work, array, sizeof(double) * x_size * y_size * z_size);
  }

#ifdef POLYBENCH_TIME
  printf("var=%d num_indices=%d z_size=%d z_size=%d z_size=%d\n", var, num_indices, x_size, y_size, z_size);
  static int warmup = 0;
  if (warmup >= 3) {
        polybench_start_instruments
  }
#endif

  stencil7_calc_all(num_indices, x_size, y_size, z_size, allarray, allwork);

#ifdef POLYBENCH_TIME
  if (warmup >= 3) {
        polybench_stop_instruments
        polybench_print_instruments
        exit(0);
  }
  warmup+=1;
#endif

  // copy back
  for (int in = 0; in < num_indices; in++) {
    block* bp = &blocks[sorted_list[in].n];
    void* array = &bp->array[var * x_size * y_size * z_size];
    void* result = allarray[in];

    memcpy(array, result, sizeof(double) * x_size * y_size * z_size);
  }

  free(mem);
}


#if 1
void stencil_calc(int var, int stencil_in)
{
   int i, j, k, in;
   double sb, sm, sf;
   block *bp;

   typedef double (*block3D_t)[y_block_size+2][z_block_size+2];

   if (stencil_in == 7) {
     stencil7_calc(var,sorted_index[num_refine + 1],x_block_size+2, y_block_size+2,z_block_size+2);
     total_fp_divs += (double) num_active*num_cells;
     total_fp_adds += (double) 6*num_active*num_cells;
   } else {
     assert(0);
//#pragma omp parallel for default(shared) private (i, j, k, bp, sb, sm, sf)
      for (in = 0; in < sorted_index[num_refine+1]; in++) {
         bp = &blocks[sorted_list[in].n];
         block3D_t array = (block3D_t)&bp->array[var*block3D_size];
         double work[x_block_size+2][y_block_size+2][z_block_size+2];
         memcpy(work, array, sizeof(work));
         for (i = 1; i <= x_block_size; i++)
            for (j = 1; j <= y_block_size; j++)
               for (k = 1; k <= z_block_size; k++) {
                  sb = work[i-1][j-1][k-1] +
                       work[i-1][j-1][k  ] +
                       work[i-1][j-1][k+1] +
                       work[i-1][j  ][k-1] +
                       work[i-1][j  ][k  ] +
                       work[i-1][j  ][k+1] +
                       work[i-1][j+1][k-1] +
                       work[i-1][j+1][k  ] +
                       work[i-1][j+1][k+1];
                  sm = work[i  ][j-1][k-1] +
                       work[i  ][j-1][k  ] +
                       work[i  ][j-1][k+1] +
                       work[i  ][j  ][k-1] +
                       work[i  ][j  ][k  ] +
                       work[i  ][j  ][k+1] +
                       work[i  ][j+1][k-1] +
                       work[i  ][j+1][k  ] +
                       work[i  ][j+1][k+1];
                  sf = work[i+1][j-1][k-1] +
                       work[i+1][j-1][k  ] +
                       work[i+1][j-1][k+1] +
                       work[i+1][j  ][k-1] +
                       work[i+1][j  ][k  ] +
                       work[i+1][j  ][k+1] +
                       work[i+1][j+1][k-1] +
                       work[i+1][j+1][k  ] +
                       work[i+1][j+1][k+1];
                  array[i][j][k] = (sb + sm + sf)/27.0;
               }
      }

      total_fp_divs += (double) num_active*num_cells;
      total_fp_adds += (double) 26*num_active*num_cells;
   }
}
#else
void stencil_calc(int var, int stencil_in)
{
  int i, j, k, in;
  double sb, sm, sf;
  block *bp;

  typedef double (*block3D_t)[y_block_size+2][z_block_size+2];

  if (stencil_in == 7) {
#pragma omp parallel for default(shared) private(i, j, k, bp)
    for (in = 0; in < sorted_index[num_refine+1]; in++) {
      bp = &blocks[sorted_list[in].n];
      block3D_t array = (block3D_t)&bp->array[var*block3D_size];
      double work[x_block_size+2][y_block_size+2][z_block_size+2];
      memcpy(work, array, sizeof(work));
      for (i = 1; i <= x_block_size; i++)
        for (j = 1; j <= y_block_size; j++)
          for (k = 1; k <= z_block_size; k++)
            array[i][j][k] = (work[i-1][j  ][k  ] +
              work[i  ][j-1][k  ] +
              work[i  ][j  ][k-1] +
              work[i  ][j  ][k  ] +
              work[i  ][j  ][k+1] +
              work[i  ][j+1][k  ] +
              work[i+1][j  ][k  ])/7.0;
    }

    total_fp_divs += (double) num_active*num_cells;
    total_fp_adds += (double) 6*num_active*num_cells;
  } else {
#pragma omp parallel for default(shared) private (i, j, k, bp, sb, sm, sf)
    for (in = 0; in < sorted_index[num_refine+1]; in++) {
      bp = &blocks[sorted_list[in].n];
      block3D_t array = (block3D_t)&bp->array[var*block3D_size];
      double work[x_block_size+2][y_block_size+2][z_block_size+2];
      memcpy(work, array, sizeof(work));
      for (i = 1; i <= x_block_size; i++)
        for (j = 1; j <= y_block_size; j++)
          for (k = 1; k <= z_block_size; k++) {
            sb = work[i-1][j-1][k-1] +
              work[i-1][j-1][k  ] +
              work[i-1][j-1][k+1] +
              work[i-1][j  ][k-1] +
              work[i-1][j  ][k  ] +
              work[i-1][j  ][k+1] +
              work[i-1][j+1][k-1] +
              work[i-1][j+1][k  ] +
              work[i-1][j+1][k+1];
            sm = work[i  ][j-1][k-1] +
              work[i  ][j-1][k  ] +
              work[i  ][j-1][k+1] +
              work[i  ][j  ][k-1] +
              work[i  ][j  ][k  ] +
              work[i  ][j  ][k+1] +
              work[i  ][j+1][k-1] +
              work[i  ][j+1][k  ] +
              work[i  ][j+1][k+1];
            sf = work[i+1][j-1][k-1] +
              work[i+1][j-1][k  ] +
              work[i+1][j-1][k+1] +
              work[i+1][j  ][k-1] +
              work[i+1][j  ][k  ] +
              work[i+1][j  ][k+1] +
              work[i+1][j+1][k-1] +
              work[i+1][j+1][k  ] +
              work[i+1][j+1][k+1];
            array[i][j][k] = (sb + sm + sf)/27.0;
          }
    }

    total_fp_divs += (double) num_active*num_cells;
    total_fp_adds += (double) 26*num_active*num_cells;
  }
}
#endif

//void stencil_0(int var)
//{
//   int in, i, j, k, v;
//   block *bp;
//
//   if (var == 1) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  for (v = mat; v < 2*mat; v++)
//                     bp->array[var][i][j][k] += bp->array[v][i][j][k]*
//                                                bp->array[0][i][j][k];
//      }
//}
//
//      total_fp_adds += (double) mat*num_active*num_cells;
//      total_fp_muls += (double) mat*num_active*num_cells;
//    } else if (var < mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] += bp->array[var][i][j][k]*
//                                             (bp->array[0][i][j][k] +
//                                              bp->array[1][i][j][k] -
//                                              beta*bp->array[var][i][j][k]);
//      }
//}
//
//      total_fp_adds += (double) 3*num_active*num_cells;
//      total_fp_muls += (double) 2*num_active*num_cells;
//   } else if (var < 2*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] = bp->array[var][i][j][k]*
//                                            (bp->array[0][i][j][k] +
//                                             bp->array[var][i][j][k] +
//                                             beta*bp->array[var+mat][i][j][k] +
//                                     (1.0-beta)*bp->array[var+2*mat][i][j][k])/
//                                            bp->array[1][i][j][k];
//      }
//}
//
//      total_fp_adds += (double) 3*num_active*num_cells;
//      total_fp_muls += (double) 3*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < 3*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] += bp->array[var-mat][i][j][k]*
//                                             (beta*bp->array[0][i][j][k] +
//                                     alpha[var-2*mat]*bp->array[var][i][j][k] +
//                                       (1.0-beta)*bp->array[var+mat][i][j][k])/
//                                             bp->array[1][i][j][k];
//         }
//}
//
//      total_fp_adds += (double) 4*num_active*num_cells;
//      total_fp_muls += (double) 3*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] += bp->array[var-2*mat][i][j][k]*
//                                             (beta*bp->array[0][i][j][k] +
//                                     alpha[var-3*mat]*bp->array[var][i][j][k] +
//                           (1.0-alpha[var-3*mat])*bp->array[var-mat][i][j][k] +
//                                     (1.0-beta)*bp->array[var-2*mat][i][j][k])/
//                                             (bp->array[1][i][j][k]*
//                                              bp->array[1][i][j][k]);
//         }
//}
//      total_fp_adds += (double) 6*num_active*num_cells;
//      total_fp_muls += (double) 6*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   }
//}
//
//void stencil_x(int var)
//{
//   int in, i, j, k, v;
//   double tmp1, tmp2;
//   block *bp;
//
//   if (var == 1) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++) {
//                  for (v = 2; v < mat+2; v++)
//                     bp->array[1][i][j][k] += bp->array[v][i][j][k]*
//                                              bp->array[0][i][j][k];
//                  bp->array[1][i][j][k] /= (beta + bp->array[1][i][j][k]);
//               }
//      }
//}
//      total_fp_adds += (double) (mat+1)*num_active*num_cells;
//      total_fp_muls += (double) mat*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] += bp->array[var][i][j][k]*
//                                             (bp->array[0][i][j][k] +
//                                              bp->array[1][i][j][k] -
//                                              beta*bp->array[var][i][j][k])/
//                                          (alpha[var] + bp->array[1][i][j][k]);
//      }
//}
//      total_fp_adds += (double) 4*num_active*num_cells;
//      total_fp_muls += (double) 2*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < 2*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  if ((tmp1 = fabs(bp->array[var][i][j][k] -
//                                   bp->array[var][i-1][j][k])) >
//                      (tmp2 = fabs(bp->array[var][i][j][k] -
//                                   bp->array[var][i+1][j][k])))
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i-1][j][k] +
//                                         (tmp1-tmp2)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i+1][j][k])/
//                                              (beta + alpha[var-mat] +
//                                               bp->array[var][i-1][j][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i+1][j][k] +
//                                               bp->array[0][i][j][k] +
//                                               bp->array[1][i][j][k]);
//                  else
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i-1][j][k] +
//                                         (tmp2-tmp1)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i+1][j][k])/
//                                              (beta + alpha[var-mat] +
//                                               bp->array[var][i-1][j][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i+1][j][k] +
//                                               bp->array[0][i][j][k] +
//                                               bp->array[1][i][j][k]);
//      }
//}
//      total_fp_adds += (double) 12*num_active*num_cells;
//      total_fp_muls += (double) 3*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < 3*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  if ((tmp1 = fabs(bp->array[var-mat][i][j][k] -
//                                   bp->array[var-mat][i-1][j][k])) >
//                      (tmp2 = fabs(bp->array[var-mat][i][j][k] -
//                                   bp->array[var-mat][i+1][j][k])))
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i-1][j][k] +
//                                         (tmp1-tmp2)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i+1][j][k])/
//                                              (beta + alpha[var-2*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var+mat][i][j][k] +
//                                               bp->array[var][i-1][j][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i+1][j][k]);
//                  else
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i-1][j][k] +
//                                         (tmp2-tmp1)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i+1][j][k])/
//                                              (beta + alpha[var-2*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var+mat][i][j][k] +
//                                               bp->array[var][i-1][j][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i+1][j][k]);
//      }
//}
//      total_fp_adds += (double) 12*num_active*num_cells;
//      total_fp_muls += (double) 3*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  if ((tmp1 = fabs(bp->array[var-2*mat][i][j][k] -
//                                   bp->array[var-2*mat][i-1][j][k])) >
//                      (tmp2 = fabs(bp->array[var-2*mat][i][j][k] -
//                                   bp->array[var-2*mat][i+1][j][k])))
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i-1][j][k] +
//                                         (tmp1-tmp2)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i+1][j][k])/
//                                              (beta + alpha[var-3*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var-2*mat][i][j][k] +
//                                               bp->array[var][i-1][j][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i+1][j][k]);
//                  else
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i-1][j][k] +
//                                         (tmp2-tmp1)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i+1][j][k])/
//                                              (beta + alpha[var-3*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var-2*mat][i][j][k] +
//                                               bp->array[var][i-1][j][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i+1][j][k]);
//      }
//}
//      total_fp_adds += (double) 12*num_active*num_cells;
//      total_fp_muls += (double) 3*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   }
//}
//
//void stencil_y(int var)
//{
//   int in, i, j, k, v;
//   double tmp1, tmp2;
//   block *bp;
//
//   if (var == 1) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//#pragma omp for
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++) {
//                  for (v = 2; v < mat+2; v++)
//                     bp->array[1][i][j][k] += bp->array[v][i][j][k]*
//                                              bp->array[0][i][j][k];
//                  bp->array[1][i][j][k] /= (beta + bp->array[1][i][j][k]);
//               }
//      }
//}
//      total_fp_adds += (double) (mat+1)*num_active*num_cells;
//      total_fp_muls += (double) mat*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] += bp->array[var][i][j][k]*
//                                             (bp->array[0][i][j][k] +
//                                              bp->array[1][i][j][k] -
//                                              beta*bp->array[var][i][j][k])/
//                                          (alpha[var] + bp->array[1][i][j][k]);
//      }
//}
//      total_fp_adds += (double) 4*num_active*num_cells;
//      total_fp_muls += (double) 2*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < 2*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  if ((tmp1 = fabs(bp->array[var][i][j][k] -
//                                   bp->array[var][i][j-1][k])) >
//                      (tmp2 = fabs(bp->array[var][i][j][k] -
//                                   bp->array[var][i][j+1][k])))
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j-1][k] +
//                                         (tmp1-tmp2)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j+1][k])/
//                                              (beta + alpha[var-mat] +
//                                               bp->array[var][i][j-1][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j+1][k] +
//                                               bp->array[0][i][j][k] +
//                                               bp->array[1][i][j][k]);
//                  else
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j-1][k] +
//                                         (tmp2-tmp1)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j+1][k])/
//                                              (beta + alpha[var-mat] +
//                                               bp->array[var][i][j-1][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j+1][k] +
//                                               bp->array[0][i][j][k] +
//                                               bp->array[1][i][j][k]);
//      }
//}
//      total_fp_adds += (double) 12*num_active*num_cells;
//      total_fp_muls += (double) 3*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < 3*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  if ((tmp1 = fabs(bp->array[var-mat][i][j][k] -
//                                   bp->array[var-mat][i][j-1][k])) >
//                      (tmp2 = fabs(bp->array[var-mat][i][j][k] -
//                                   bp->array[var-mat][i][j+1][k])))
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j-1][k] +
//                                         (tmp1-tmp2)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j+1][k])/
//                                              (beta + alpha[var-2*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var+mat][i][j][k] +
//                                               bp->array[var][i][j-1][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j+1][k]);
//                  else
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j-1][k] +
//                                         (tmp2-tmp1)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j+1][k])/
//                                              (beta + alpha[var-2*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var+mat][i][j][k] +
//                                               bp->array[var][i][j-1][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j+1][k]);
//      }
//}
//      total_fp_adds += (double) 12*num_active*num_cells;
//      total_fp_muls += (double) 3*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  if ((tmp1 = fabs(bp->array[var-2*mat][i][j][k] -
//                                   bp->array[var-2*mat][i][j-1][k])) >
//                      (tmp2 = fabs(bp->array[var-2*mat][i][j][k] -
//                                   bp->array[var-2*mat][i][j+1][k])))
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j-1][k] +
//                                         (tmp1-tmp2)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j+1][k])/
//                                              (beta + alpha[var-3*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var-2*mat][i][j][k] +
//                                               bp->array[var][i][j-1][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j+1][k]);
//                  else
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j-1][k] +
//                                         (tmp2-tmp1)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j+1][k])/
//                                              (beta + alpha[var-3*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var-2*mat][i][j][k] +
//                                               bp->array[var][i][j-1][k] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j+1][k]);
//      }
//}
//      total_fp_adds += (double) 12*num_active*num_cells;
//      total_fp_muls += (double) 3*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   }
//}
//
//void stencil_z(int var)
//{
//   int in, i, j, k, v;
//   double tmp1, tmp2;
//   block *bp;
//
//
//   if (var == 1) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++) {
//                  for (v = 2; v < mat+2; v++)
//                     bp->array[1][i][j][k] += bp->array[v][i][j][k]*
//                                              bp->array[0][i][j][k];
//                  bp->array[1][i][j][k] /= (beta + bp->array[1][i][j][k]);
//               }
//      }
//}
//
//      total_fp_adds += (double) (mat+1)*num_active*num_cells;
//      total_fp_muls += (double) mat*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] += bp->array[var][i][j][k]*
//                                             (bp->array[0][i][j][k] +
//                                              bp->array[1][i][j][k] -
//                                              beta*bp->array[var][i][j][k])/
//                                          (alpha[var] + bp->array[1][i][j][k]);
//      }
//}
//      total_fp_adds += (double) 4*num_active*num_cells;
//      total_fp_muls += (double) 2*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < 2*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  if ((tmp1 = fabs(bp->array[var][i][j][k] -
//                                   bp->array[var][i][j][k-1])) >
//                      (tmp2 = fabs(bp->array[var][i][j][k] -
//                                   bp->array[var][i][j][k+1])))
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j][k-1] +
//                                         (tmp1-tmp2)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j][k+1])/
//                                              (beta + alpha[var-mat] +
//                                               bp->array[var][i][j][k-1] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j][k+1] +
//                                               bp->array[0][i][j][k] +
//                                               bp->array[1][i][j][k]);
//                  else
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j][k-1] +
//                                         (tmp2-tmp1)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j][k+1])/
//                                              (beta + alpha[var-mat] +
//                                               bp->array[var][i][j][k-1] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j][k+1] +
//                                               bp->array[0][i][j][k] +
//                                               bp->array[1][i][j][k]);
//      }
//}
//      total_fp_adds += (double) 12*num_active*num_cells;
//      total_fp_muls += (double) 3*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < 3*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  if ((tmp1 = fabs(bp->array[var-mat][i][j][k] -
//                                   bp->array[var-mat][i][j][k-1])) >
//                      (tmp2 = fabs(bp->array[var-mat][i][j][k] -
//                                   bp->array[var-mat][i][j][k+1])))
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j][k-1] +
//                                         (tmp1-tmp2)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j][k+1])/
//                                              (beta + alpha[var-2*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var+mat][i][j][k] +
//                                               bp->array[var][i][j][k-1] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j][k+1]);
//                  else
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j][k-1] +
//                                         (tmp2-tmp1)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j][k+1])/
//                                              (beta + alpha[var-2*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var+mat][i][j][k] +
//                                               bp->array[var][i][j][k-1] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j][k+1]);
//      }
//}
//      total_fp_adds += (double) 12*num_active*num_cells;
//      total_fp_muls += (double) 3*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else {
//#pragma omp parallel default(shared) private(i, j, k, bp, v, tmp1, tmp2)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  if ((tmp1 = fabs(bp->array[var-2*mat][i][j][k] -
//                                   bp->array[var-2*mat][i][j][k-1])) >
//                      (tmp2 = fabs(bp->array[var-2*mat][i][j][k] -
//                                   bp->array[var-2*mat][i][j][k+1])))
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j][k-1] +
//                                         (tmp1-tmp2)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j][k+1])/
//                                              (beta + alpha[var-3*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var-2*mat][i][j][k] +
//                                               bp->array[var][i][j][k-1] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j][k+1]);
//                  else
//                    bp->array[var][i][j][k] = (tmp1*bp->array[var][i][j][k-1] +
//                                         (tmp2-tmp1)*(bp->array[var][i][j][k] +
//                                                       bp->array[1][i][j][k]) +
//                                               tmp2*bp->array[var][i][j][k+1])/
//                                              (beta + alpha[var-3*mat] +
//                                               bp->array[var-mat][i][j][k] +
//                                               bp->array[var-2*mat][i][j][k] +
//                                               bp->array[var][i][j][k-1] +
//                                               bp->array[var][i][j][k] +
//                                               bp->array[var][i][j][k+1]);
//      }
//}
//      total_fp_adds += (double) 12*num_active*num_cells;
//      total_fp_muls += (double) 3*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   }
//}
//
//void stencil_7(int var)
//{
//   int in, i, j, k, v;
//   double work[x_block_size+2][y_block_size+2][z_block_size+2];
//   block *bp;
//
//   if (var < mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  work[i][j][k] = (bp->array[var      ][i-1][j  ][k  ]*
//                                   bp->array[var+  mat][i-1][j  ][k  ] +
//                                   bp->array[var      ][i  ][j-1][k  ]*
//                                   bp->array[var+2*mat][i  ][j-1][k  ] +
//                                   bp->array[var      ][i  ][j  ][k-1]*
//                                   bp->array[var+3*mat][i  ][j  ][k-1] +
//                                   bp->array[var      ][i  ][j  ][k  ]*
//                                   bp->array[var      ][i  ][j  ][k  ] +
//                                   bp->array[var      ][i  ][j  ][k+1]*
//                                   bp->array[var+3*mat][i  ][j  ][k+1] +
//                                   bp->array[var      ][i  ][j+1][k  ]*
//                                   bp->array[var+2*mat][i  ][j+1][k  ] +
//                                   bp->array[var      ][i+1][j  ][k  ]*
//                                   bp->array[var+  mat][i+1][j  ][k  ])/
//                                   7.0*(beta + bp->array[var][i][j][k]);
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] = work[i][j][k];
//      }
//}
//      total_fp_adds += (double) 7*num_active*num_cells;
//      total_fp_muls += (double) 8*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < 2*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  work[i][j][k] = (bp->array[var      ][i-1][j  ][k  ]*
//                                   bp->array[var+  mat][i-1][j  ][k  ] +
//                                   bp->array[var      ][i  ][j-1][k  ]*
//                                   bp->array[var+2*mat][i  ][j-1][k  ] +
//                                   bp->array[var      ][i  ][j  ][k-1]*
//                                   bp->array[var-  mat][i  ][j  ][k-1] +
//                                   bp->array[var      ][i  ][j  ][k  ]*
//                                   bp->array[var      ][i  ][j  ][k  ] +
//                                   bp->array[var      ][i  ][j  ][k+1]*
//                                   bp->array[var-  mat][i  ][j  ][k+1] +
//                                   bp->array[var      ][i  ][j+1][k  ]*
//                                   bp->array[var+2*mat][i  ][j+1][k  ] +
//                                   bp->array[var      ][i+1][j  ][k  ]*
//                                   bp->array[var+  mat][i+1][j  ][k  ])/
//                                   7.0*(beta + bp->array[var][i][j][k]);
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] = work[i][j][k];
//      }
//}
//      total_fp_adds += (double) 7*num_active*num_cells;
//      total_fp_muls += (double) 8*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < 3*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  work[i][j][k] = (bp->array[var      ][i-1][j  ][k  ]*
//                                   bp->array[var+  mat][i-1][j  ][k  ] +
//                                   bp->array[var      ][i  ][j-1][k  ]*
//                                   bp->array[var-2*mat][i  ][j-1][k  ] +
//                                   bp->array[var      ][i  ][j  ][k-1]*
//                                   bp->array[var-  mat][i  ][j  ][k-1] +
//                                   bp->array[var      ][i  ][j  ][k  ]*
//                                   bp->array[var      ][i  ][j  ][k  ] +
//                                   bp->array[var      ][i  ][j  ][k+1]*
//                                   bp->array[var-  mat][i  ][j  ][k+1] +
//                                   bp->array[var      ][i  ][j+1][k  ]*
//                                   bp->array[var-2*mat][i  ][j+1][k  ] +
//                                   bp->array[var      ][i+1][j  ][k  ]*
//                                   bp->array[var+  mat][i+1][j  ][k  ])/
//                                   7.0*(beta + bp->array[var][i][j][k]);
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] = work[i][j][k];
//      }
//}
//      total_fp_adds += (double) 7*num_active*num_cells;
//      total_fp_muls += (double) 8*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  work[i][j][k] = (bp->array[var      ][i-1][j  ][k  ]*
//                                   bp->array[var-3*mat][i-1][j  ][k  ] +
//                                   bp->array[var      ][i  ][j-1][k  ]*
//                                   bp->array[var-2*mat][i  ][j-1][k  ] +
//                                   bp->array[var      ][i  ][j  ][k-1]*
//                                   bp->array[var-  mat][i  ][j  ][k-1] +
//                                   bp->array[var      ][i  ][j  ][k  ]*
//                                   bp->array[var      ][i  ][j  ][k  ] +
//                                   bp->array[var      ][i  ][j  ][k+1]*
//                                   bp->array[var-  mat][i  ][j  ][k+1] +
//                                   bp->array[var      ][i  ][j+1][k  ]*
//                                   bp->array[var-2*mat][i  ][j+1][k  ] +
//                                   bp->array[var      ][i+1][j  ][k  ]*
//                                   bp->array[var-3*mat][i+1][j  ][k  ])/
//                                   7.0*(beta + bp->array[var][i][j][k]);
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] = work[i][j][k];
//      }
//}
//      total_fp_adds += (double) 7*num_active*num_cells;
//      total_fp_muls += (double) 8*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   }
//}
//
//void stencil_27(int var)
//{
//   int in, i, j, k, v;
//   double work[x_block_size+2][y_block_size+2][z_block_size+2];
//   block *bp;
//
//   if (var < mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//#pragma omp for
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  work[i][j][k] = (bp->array[var+3*mat][i-1][j-1][k-1] +
//                                   bp->array[var+2*mat][i-1][j-1][k  ] +
//                                   bp->array[var+3*mat][i-1][j-1][k+1] +
//                                   bp->array[var+2*mat][i-1][j  ][k-1] +
//                                   bp->array[var+  mat][i-1][j  ][k  ] +
//                                   bp->array[var+2*mat][i-1][j  ][k+1] +
//                                   bp->array[var+3*mat][i-1][j+1][k-1] +
//                                   bp->array[var+2*mat][i-1][j+1][k  ] +
//                                   bp->array[var+3*mat][i-1][j+1][k+1] +
//                                   bp->array[var+2*mat][i  ][j-1][k-1] +
//                                   bp->array[var+  mat][i  ][j-1][k  ] +
//                                   bp->array[var+2*mat][i  ][j-1][k+1] +
//                                   bp->array[var+  mat][i  ][j  ][k-1] +
//                                   bp->array[var      ][i  ][j  ][k  ] +
//                                   bp->array[var+  mat][i  ][j  ][k+1] +
//                                   bp->array[var+2*mat][i  ][j+1][k-1] +
//                                   bp->array[var+  mat][i  ][j+1][k  ] +
//                                   bp->array[var+2*mat][i  ][j+1][k+1] +
//                                   bp->array[var+3*mat][i+1][j-1][k-1] +
//                                   bp->array[var+2*mat][i+1][j-1][k  ] +
//                                   bp->array[var+3*mat][i+1][j-1][k+1] +
//                                   bp->array[var+2*mat][i+1][j  ][k-1] +
//                                   bp->array[var+  mat][i+1][j  ][k  ] +
//                                   bp->array[var+2*mat][i+1][j  ][k+1] +
//                                   bp->array[var+3*mat][i+1][j+1][k-1] +
//                                   bp->array[var+2*mat][i+1][j+1][k  ] +
//                                   bp->array[var+3*mat][i+1][j+1][k+1])/
//                                  (beta+27.0);
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] = work[i][j][k];
//      }
//}
//      total_fp_adds += (double) 27*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < 2*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  work[i][j][k] = (bp->array[var-  mat][i-1][j-1][k-1] +
//                                   bp->array[var+2*mat][i-1][j-1][k  ] +
//                                   bp->array[var-  mat][i-1][j-1][k+1] +
//                                   bp->array[var+2*mat][i-1][j  ][k-1] +
//                                   bp->array[var+  mat][i-1][j  ][k  ] +
//                                   bp->array[var+2*mat][i-1][j  ][k+1] +
//                                   bp->array[var-  mat][i-1][j+1][k-1] +
//                                   bp->array[var+2*mat][i-1][j+1][k  ] +
//                                   bp->array[var-  mat][i-1][j+1][k+1] +
//                                   bp->array[var+2*mat][i  ][j-1][k-1] +
//                                   bp->array[var+  mat][i  ][j-1][k  ] +
//                                   bp->array[var+2*mat][i  ][j-1][k+1] +
//                                   bp->array[var+  mat][i  ][j  ][k-1] +
//                                   bp->array[var      ][i  ][j  ][k  ] +
//                                   bp->array[var+  mat][i  ][j  ][k+1] +
//                                   bp->array[var+2*mat][i  ][j+1][k-1] +
//                                   bp->array[var+  mat][i  ][j+1][k  ] +
//                                   bp->array[var+2*mat][i  ][j+1][k+1] +
//                                   bp->array[var-  mat][i+1][j-1][k-1] +
//                                   bp->array[var+2*mat][i+1][j-1][k  ] +
//                                   bp->array[var-  mat][i+1][j-1][k+1] +
//                                   bp->array[var+2*mat][i+1][j  ][k-1] +
//                                   bp->array[var+  mat][i+1][j  ][k  ] +
//                                   bp->array[var+2*mat][i+1][j  ][k+1] +
//                                   bp->array[var-  mat][i+1][j+1][k-1] +
//                                   bp->array[var+2*mat][i+1][j+1][k  ] +
//                                   bp->array[var-  mat][i+1][j+1][k+1])/
//                                  (beta+27.0);
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] = work[i][j][k];
//      }
//}
//
//      total_fp_adds += (double) 27*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else if (var < 3*mat) {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  work[i][j][k] = (bp->array[var-  mat][i-1][j-1][k-1] +
//                                   bp->array[var-2*mat][i-1][j-1][k  ] +
//                                   bp->array[var-  mat][i-1][j-1][k+1] +
//                                   bp->array[var-2*mat][i-1][j  ][k-1] +
//                                   bp->array[var+  mat][i-1][j  ][k  ] +
//                                   bp->array[var-2*mat][i-1][j  ][k+1] +
//                                   bp->array[var-  mat][i-1][j+1][k-1] +
//                                   bp->array[var-2*mat][i-1][j+1][k  ] +
//                                   bp->array[var-  mat][i-1][j+1][k+1] +
//                                   bp->array[var-2*mat][i  ][j-1][k-1] +
//                                   bp->array[var+  mat][i  ][j-1][k  ] +
//                                   bp->array[var-2*mat][i  ][j-1][k+1] +
//                                   bp->array[var+  mat][i  ][j  ][k-1] +
//                                   bp->array[var      ][i  ][j  ][k  ] +
//                                   bp->array[var+  mat][i  ][j  ][k+1] +
//                                   bp->array[var-2*mat][i  ][j+1][k-1] +
//                                   bp->array[var+  mat][i  ][j+1][k  ] +
//                                   bp->array[var-2*mat][i  ][j+1][k+1] +
//                                   bp->array[var-  mat][i+1][j-1][k-1] +
//                                   bp->array[var-2*mat][i+1][j-1][k  ] +
//                                   bp->array[var-  mat][i+1][j-1][k+1] +
//                                   bp->array[var-2*mat][i+1][j  ][k-1] +
//                                   bp->array[var+  mat][i+1][j  ][k  ] +
//                                   bp->array[var-2*mat][i+1][j  ][k+1] +
//                                   bp->array[var-  mat][i+1][j+1][k-1] +
//                                   bp->array[var-2*mat][i+1][j+1][k  ] +
//                                   bp->array[var-  mat][i+1][j+1][k+1])/
//                                  (beta+27.0);
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] = work[i][j][k];
//      }
//}
//      total_fp_adds += (double) 27*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   } else {
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//      for (in = 0; in < sorted_index[num_refine+1]; in++) {
//         bp = &blocks[sorted_list[in].n];
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  work[i][j][k] = (bp->array[var-  mat][i-1][j-1][k-1] +
//                                   bp->array[var-2*mat][i-1][j-1][k  ] +
//                                   bp->array[var-  mat][i-1][j-1][k+1] +
//                                   bp->array[var-2*mat][i-1][j  ][k-1] +
//                                   bp->array[var-3*mat][i-1][j  ][k  ] +
//                                   bp->array[var-2*mat][i-1][j  ][k+1] +
//                                   bp->array[var-  mat][i-1][j+1][k-1] +
//                                   bp->array[var-2*mat][i-1][j+1][k  ] +
//                                   bp->array[var-  mat][i-1][j+1][k+1] +
//                                   bp->array[var-2*mat][i  ][j-1][k-1] +
//                                   bp->array[var-3*mat][i  ][j-1][k  ] +
//                                   bp->array[var-2*mat][i  ][j-1][k+1] +
//                                   bp->array[var-3*mat][i  ][j  ][k-1] +
//                                   bp->array[var      ][i  ][j  ][k  ] +
//                                   bp->array[var-3*mat][i  ][j  ][k+1] +
//                                   bp->array[var-2*mat][i  ][j+1][k-1] +
//                                   bp->array[var-3*mat][i  ][j+1][k  ] +
//                                   bp->array[var-2*mat][i  ][j+1][k+1] +
//                                   bp->array[var-  mat][i+1][j-1][k-1] +
//                                   bp->array[var-2*mat][i+1][j-1][k  ] +
//                                   bp->array[var-  mat][i+1][j-1][k+1] +
//                                   bp->array[var-2*mat][i+1][j  ][k-1] +
//                                   bp->array[var-3*mat][i+1][j  ][k  ] +
//                                   bp->array[var-2*mat][i+1][j  ][k+1] +
//                                   bp->array[var-  mat][i+1][j+1][k-1] +
//                                   bp->array[var-2*mat][i+1][j+1][k  ] +
//                                   bp->array[var-  mat][i+1][j+1][k+1])/
//                                  (beta+27.0);
//         for (i = 1; i <= x_block_size; i++)
//            for (j = 1; j <= y_block_size; j++)
//               for (k = 1; k <= z_block_size; k++)
//                  bp->array[var][i][j][k] = work[i][j][k];
//      }
//}
//
//      total_fp_adds += (double) 27*num_active*num_cells;
//      total_fp_divs += (double) num_active*num_cells;
//   }
//}
//
//void stencil_check(int var)
//{
//   int in, i, j, k, v;
//   double work[x_block_size+2][y_block_size+2][z_block_size+2];
//   block *bp;
//
//#pragma omp parallel default(shared) private(i, j, k, bp, v)
//{
//   for (in = 0; in < sorted_index[num_refine+1]; in++) {
//      bp = &blocks[sorted_list[in].n];
//      for (i = 1; i <= x_block_size; i++)
//         for (j = 1; j <= y_block_size; j++)
//            for (k = 1; k <= z_block_size; k++) {
//               bp->array[var][i][j][k] = fabs(bp->array[var][i][j][k]);
//               if (bp->array[var][i][j][k] >= 1.0) {
//                  bp->array[var][i][j][k] /= (beta + alpha[0] +
//                                              bp->array[var][i][j][k]);
//                  total_fp_divs += (double) 1;
//                  total_fp_adds += (double) 2;
//               }
//               else if (bp->array[var][i][j][k] < 0.1) {
//                  bp->array[var][i][j][k] *= 10.0 - beta;
//                  total_fp_muls += (double) 1;
//                  total_fp_adds += (double) 1;
//               }
//            }
//   }
//}
//}
