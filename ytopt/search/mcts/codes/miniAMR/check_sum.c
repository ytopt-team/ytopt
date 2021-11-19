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

#include <math.h>
#include <mpi.h>

#include "block.h"
#include "comm.h"
#include "timer.h"
#include "proto.h"

// Generate check sum for a variable over all active blocks.
double check_sum(int var)
{
   int in, i, j, k;
   double sum, gsum, block_sum, t1, t2, t3;
   block *bp;

   t1 = timer();

   sum = 0.0;
//#pragma omp parallel for private (i, j, k, bp, block_sum) reduction(+: sum)
   for (in = 0; in < sorted_index[num_refine+1]; in++) {
      bp = &blocks[sorted_list[in].n];
      block_sum = 0.0;
      typedef double (*block3D_t)[y_block_size+2][z_block_size+2];
      block3D_t array = (block3D_t)&bp->array[var*block3D_size];
      for (i = 1; i <= x_block_size; i++)
         for (j = 1; j <= y_block_size; j++)
            for (k = 1; k <= z_block_size; k++)
               block_sum += array[i][j][k];
//if (!my_pe) printf("cs in %d block %d sum %lf\n", in, sorted_list[in].n, block_sum);
      sum += block_sum;
   }

   t2 = timer();

   MPI_Allreduce(&sum, &gsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   t3 = timer();
   timer_cs_red += t3 - t2;
   timer_cs_calc += t2 - t1;
   total_red++;

   return gsum;
}
