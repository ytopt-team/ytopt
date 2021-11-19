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

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "block.h"
#include "comm.h"
#include "proto.h"

// Write block information (level and center) to plot file.
void plot(int ts)
{
   int i, j, n, *buf, buf_size, size;
   num_sz total_num_blocks;
   char fname[20];
   block *bp;
   MPI_Status status;
   FILE *fp;

   if (!my_pe) {
      fname[0] = 'p';
      fname[1] = 'l';
      fname[2] = 'o';
      fname[3] = 't';
      fname[4] = '.';
      for (n = 1, j = 0; n < num_tsteps; j++, n *= 10) ;
      for (n = 1, i = 0; i <= j; i++, n *= 10)
         fname[5+j-i] = (char) ('0' + (ts/n)%10);
      fname[6+j] = '\0';
      fp = fopen(fname, "w");

      total_num_blocks = 0;
      for (i = 0; i <= num_refine; i++)
         total_num_blocks += num_blocks[i];
      fprintf(fp, "%ld %d %d %d %d\n", (long long) total_num_blocks, num_refine,
                                      npx*init_block_x, npy*init_block_y,
                                      npz*init_block_z);
      buf_size = 0;
      fprintf(fp, "%d\n", num_active);
      for (n = 0; n < max_active_block; n++)
         if ((bp = &blocks[n])->number >= 0)
            fprintf(fp, "%d %d %d %d\n", bp->level, bp->cen[0],
                                         bp->cen[1], bp->cen[2]);
      for (i = 1; i < num_pes; i++) {
         MPI_Send(&size, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
         MPI_Recv(&size, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
         if (size > buf_size) {
            if (i != 1)
               free(buf);
            buf_size = size;
            buf = (int *)ma_malloc(4*buf_size*sizeof(int), __FILE__, __LINE__);
         }
         MPI_Irecv(buf, 4*size, MPI_INT, i, 3, MPI_COMM_WORLD, &request[0]);
         MPI_Send(&size, 1, MPI_INT, i, 4, MPI_COMM_WORLD);
         MPI_Wait(&request[0], &status);
         fprintf(fp, "%d\n", size);
         for (n = j = 0; j < size; j++, n += 4)
            fprintf(fp, "%d %d %d %d\n", buf[n], buf[n+1], buf[n+2], buf[n+3]);
      }
      if (buf_size)
         free(buf);
      fclose(fp);
   } else {
      total_num_blocks = 0;
      for (i = 0; i <= num_refine; i++)
         total_num_blocks += local_num_blocks[i];
      buf = (int *) ma_malloc(4*total_num_blocks*sizeof(int),
                              __FILE__, __LINE__);
      for (i = n = 0; n < max_active_block; n++)
         if ((bp = &blocks[n])->number >= 0) {
            buf[i++] = bp->level;
            buf[i++] = bp->cen[0];
            buf[i++] = bp->cen[1];
            buf[i++] = bp->cen[2];
         }
      MPI_Recv(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
      MPI_Send(&total_num_blocks, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
      MPI_Recv(&size, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, &status);
      MPI_Send(buf, 4*total_num_blocks, MPI_INT, 0, 3, MPI_COMM_WORLD);
      free(buf);
   }
}
