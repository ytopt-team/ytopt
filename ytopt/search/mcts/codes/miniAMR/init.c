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
#include <math.h>
#include <mpi.h>

#include "block.h"
#include "comm.h"
#include "proto.h"

// Initialize the problem and setup initial blocks.
void init(void)
{
   int n, var, i, j, k, l, m, o, size, dir, i1, i2, j1, j2, k1, k2, ib, jb, kb;
   int start[num_pes], pos[3][num_pes], pos1[npx][npy][npz], set,
       npx1, npy1, npz1, pes, fact, fac[25], nfac, f;
   num_sz num;
   block *bp;

   tol = pow(10.0, ((double) -error_tol));

   total_fp_divs = total_fp_adds = total_fp_muls = 0.0;
   p2[0] = p8[0] = 1;
   for (i = 0; i < (num_refine+1); i++) {
      p8[i+1] = p8[i]*8;
      p2[i+1] = p2[i]*2;
      sorted_index[i] = 0;
   }
   sorted_index[num_refine+1] = 0;
   block_start[0] = 0;
   local_max_b = global_max_b =  init_block_x*init_block_y*init_block_z;
   num = num_pes*global_max_b;
   for (i = 1; i <= num_refine; i++) {
      block_start[i] = block_start[i-1] + num;
      num *= 8;
      num_blocks[i] = 0;
      local_num_blocks[i] = 0;
   }

   /* initialize for communication arrays, which are initialized below */
   zero_comm_list();
   par_b.num_comm_part = par_b.num_cases = 0;
   par_p.num_comm_part = par_p.num_cases = 0;

   num_cells = x_block_size*y_block_size*z_block_size;
   x_block_half = x_block_size/2;
   y_block_half = y_block_size/2;
   z_block_half = z_block_size/2;

   if (!code) {
      /* for E/W (X dir) messages:
         0: whole -> whole (7), 1: whole -> whole (27),
         2: whole -> quarter, 3: quarter -> whole */
      msg_len[0][0] = msg_len[0][1] = y_block_size*z_block_size;
      msg_len[0][2] = msg_len[0][3] = y_block_half*z_block_half;
      /* for N/S (Y dir) messages */
      msg_len[1][0] = x_block_size*z_block_size;
      msg_len[1][1] = (x_block_size+2)*z_block_size;
      msg_len[1][2] = msg_len[1][3] = x_block_half*z_block_half;
      /* for U/D (Z dir) messages */
      msg_len[2][0] = x_block_size*y_block_size;
      msg_len[2][1] = (x_block_size+2)*(y_block_size+2);
      msg_len[2][2] = msg_len[2][3] = x_block_half*y_block_half;
   } else if (code == 1) {
      /* for E/W (X dir) messages */
      msg_len[0][0] = msg_len[0][1] = (y_block_size+2)*(z_block_size+2);
      msg_len[0][2] = (y_block_half+1)*(z_block_half+1);
      msg_len[0][3] = (y_block_half+2)*(z_block_half+2);
      /* for N/S (Y dir) messages */
      msg_len[1][0] = msg_len[1][1] = (x_block_size+2)*(z_block_size+2);
      msg_len[1][2] = (x_block_half+1)*(z_block_half+1);
      msg_len[1][3] = (x_block_half+2)*(z_block_half+2);
      /* for U/D (Z dir) messages */
      msg_len[2][0] = msg_len[2][1] = (x_block_size+2)*(y_block_size+2);
      msg_len[2][2] = (x_block_half+1)*(y_block_half+1);
      msg_len[2][3] = (x_block_half+2)*(y_block_half+2);
   } else {
      /* for E/W (X dir) messages */
      msg_len[0][0] = msg_len[0][1] = (y_block_size+2)*(z_block_size+2);
      msg_len[0][2] = (y_block_half+1)*(z_block_half+1);
      msg_len[0][3] = (y_block_size+2)*(z_block_size+2);
      /* for N/S (Y dir) messages */
      msg_len[1][0] = msg_len[1][1] = (x_block_size+2)*(z_block_size+2);
      msg_len[1][2] = (x_block_half+1)*(z_block_half+1);
      msg_len[1][3] = (x_block_size+2)*(z_block_size+2);
      /* for U/D (Z dir) messages */
      msg_len[2][0] = msg_len[2][1] = (x_block_size+2)*(y_block_size+2);
      msg_len[2][2] = (x_block_half+1)*(y_block_half+1);
      msg_len[2][3] = (x_block_size+2)*(y_block_size+2);
   }

   /* Determine position of each core in initial mesh */
   npx1 = npx;
   npy1 = npy;
   npz1 = npz;
   for (i = 0; i < 3; i++)
      for (j = 0; j < num_pes; j++)
         pos[i][j] = 0;
   nfac = factor(num_pes, fac);
   max_num_req = num_pes;
   request = (MPI_Request *) ma_malloc(max_num_req*sizeof(MPI_Request),
                                       __FILE__, __LINE__);
   s_req = (MPI_Request *) ma_malloc(max_num_req*sizeof(MPI_Request),
                                     __FILE__, __LINE__);
   pes = 1;
   start[0] = 0;
   size = num_pes;
   comms = (MPI_Comm *) ma_malloc((nfac+1)*sizeof(MPI_Comm),
                                  __FILE__, __LINE__);
   me = (int *) ma_malloc((nfac+1)*sizeof(int), __FILE__, __LINE__);
   np = (int *) ma_malloc((nfac+1)*sizeof(int), __FILE__, __LINE__);
   comms[0] = MPI_COMM_WORLD;
   me[0] = my_pe;
   np[0] = num_pes;
   // initialize
   for (n = 0, i = nfac; i > 0; i--, n++) {
      fact = fac[i-1];
      dir = find_dir(fact, npx1, npy1, npz1);
      if (dir == 0)
         npx1 /= fact;
      else
         if (dir == 1)
            npy1 /= fact;
         else
            npz1 /= fact;
      size /= fact;
      set = me[n]/size;
      MPI_Comm_split(comms[n], set, me[n], &comms[n+1]);
      MPI_Comm_rank(comms[n+1], &me[n+1]);
      MPI_Comm_size(comms[n+1], &np[n+1]);
      for (j = pes-1; j >= 0; j--)
         for (k = 0; k < fact; k++) {
            m = j*fact + k;
            if (!k)
               start[m] = start[j];
            else
               start[m] = start[m-1] + size;
            for (l = start[m], o = 0; o < size; l++, o++)
               pos[dir][l] = pos[dir][l]*fact + k;
         }
      pes *= fact;
   }
   for (i = 0; i < num_pes; i++)
      pos1[pos[0][i]][pos[1][i]][pos[2][i]] = i;

   if (!stencil) {
      mat = num_vars/4;
      beta = ((double) rand())/((double) RAND_MAX);
      for (i = 0; i < (num_vars/4); i++)
         alpha[i] = ((double) rand())/((double) RAND_MAX);
   }
   max_active_block = init_block_x*init_block_y*init_block_z;
   num_active = max_active_block;
   global_active = num_active*num_pes;
   num_parents = max_active_parent = 0;
   size = p2[num_refine+1];  /* block size is p2[num_refine+1-level]
                              * smallest block is size p2[1], so can find
                              * its center */
   mesh_size[0] = npx*init_block_x*size;
   max_mesh_size = mesh_size[0];
   mesh_size[1] = npy*init_block_y*size;
   if (mesh_size[1] > max_mesh_size)
      max_mesh_size = mesh_size[1];
   mesh_size[2] = npz*init_block_z*size;
   if (mesh_size[2] > max_mesh_size)
      max_mesh_size = mesh_size[2];
   if ((num_pes+1) > max_mesh_size)
      max_mesh_size = num_pes + 1;
   bin  = (int *) ma_malloc(max_mesh_size*sizeof(int), __FILE__, __LINE__);
   gbin = (int *) ma_malloc(max_mesh_size*sizeof(int), __FILE__, __LINE__);
   if (stencil == 7)
      f = 0;
   else
      f = 1;
   for (o = n = k1 = k = 0; k < npz; k++)
      for (k2 = 0; k2 < init_block_z; k1++, k2++)
         for (j1 = j = 0; j < npy; j++)
            for (j2 = 0; j2 < init_block_y; j1++, j2++)
               for (i1 = i = 0; i < npx; i++)
                  for (i2 = 0; i2 < init_block_x; i1++, i2++, n++) {
                     m = pos1[i][j][k];
                     if (m == my_pe) {
                        bp = &blocks[o];
                        bp->level = 0;
                        bp->number = n;
                        bp->parent = -1;
                        bp->parent_node = my_pe;
                        bp->cen[0] = i1*size + size/2;
                        bp->cen[1] = j1*size + size/2;
                        bp->cen[2] = k1*size + size/2;
                        add_sorted_list(o, n, 0);
                        for (var = 0; var < num_vars; var++) {
                           typedef double (*block3D_t)[y_block_size+2][z_block_size+2];
                           block3D_t array = (block3D_t)&bp->array[var*block3D_size];
                           for (ib = 1; ib <= x_block_size; ib++)
                              for (jb = 1; jb <= y_block_size; jb++)
                                 for (kb = 1; kb <= z_block_size; kb++)
                                    array[ib][jb][kb] =
                                       ((double) rand())/((double) RAND_MAX);
                        }
                        if (i2 == 0)
                           if (i == 0) { /* 0 boundary */
                              bp->nei_level[0] = -2;
                              bp->nei[0][0][0] = 0;
                           } else {      /* boundary with neighbor core */
                              bp->nei_level[0] = 0;
                              bp->nei[0][0][0] = -1 - pos1[i-1][j][k];
                              add_comm_list(0, o, pos1[i-1][j][k], 0+f,
                                            bp->cen[2]*mesh_size[1]+bp->cen[1],
                                            bp->cen[0] - size/2);
                           }
                        else {          /* neighbor on core */
                           bp->nei_level[0] = 0;
                           bp->nei[0][0][0] = o - 1;
                        }
                        bp->nei_refine[0] = 0;
                        if (i2 == (init_block_x - 1))
                           if (i == (npx - 1)) { /* 1 boundary */
                              bp->nei_level[1] = -2;
                              bp->nei[1][0][0] = 0;
                           } else {      /* boundary with neighbor core */
                              bp->nei_level[1] = 0;
                              bp->nei[1][0][0] = -1 - pos1[i+1][j][k];
                              add_comm_list(0, o, pos1[i+1][j][k], 10+f,
                                            bp->cen[2]*mesh_size[1]+bp->cen[1],
                                            bp->cen[0] + size/2);
                           }
                        else {          /* neighbor on core */
                           bp->nei_level[1] = 0;
                           bp->nei[1][0][0] = o + 1;
                        }
                        bp->nei_refine[1] = 0;
                        if (j2 == 0)
                           if (j == 0) { /* 0 boundary */
                              bp->nei_level[2] = -2;
                              bp->nei[2][0][0] = 0;
                           } else {      /* boundary with neighbor core */
                              bp->nei_level[2] = 0;
                              bp->nei[2][0][0] = -1 - pos1[i][j-1][k];
                              add_comm_list(1, o, pos1[i][j-1][k], 0+f,
                                            bp->cen[2]*mesh_size[0]+bp->cen[0],
                                            bp->cen[1] - size/2);
                           }
                        else {          /* neighbor on core */
                           bp->nei_level[2] = 0;
                           bp->nei[2][0][0] = o - init_block_x;
                        }
                        bp->nei_refine[2] = 0;
                        if (j2 == (init_block_y - 1))
                           if (j == (npy - 1)) { /* 1 boundary */
                              bp->nei_level[3] = -2;
                              bp->nei[3][0][0] = 0;
                           } else {      /* boundary with neighbor core */
                              bp->nei_level[3] = 0;
                              bp->nei[3][0][0] = -1 - pos1[i][j+1][k];
                              add_comm_list(1, o, pos1[i][j+1][k], 10+f,
                                            bp->cen[2]*mesh_size[0]+bp->cen[0],
                                            bp->cen[1] + size/2);
                           }
                        else {          /* neighbor on core */
                           bp->nei_level[3] = 0;
                           bp->nei[3][0][0] = o + init_block_x;
                        }
                        bp->nei_refine[3] = 0;
                        if (k2 == 0)
                           if (k == 0) { /* 0 boundary */
                              bp->nei_level[4] = -2;
                              bp->nei[4][0][0] = 0;
                           } else {      /* boundary with neighbor core */
                              bp->nei_level[4] = 0;
                              bp->nei[4][0][0] = -1 - pos1[i][j][k-1];
                              add_comm_list(2, o, pos1[i][j][k-1], 0+f,
                                            bp->cen[1]*mesh_size[0]+bp->cen[0],
                                            bp->cen[2] - size/2);
                           }
                        else {          /* neighbor on core */
                           bp->nei_level[4] = 0;
                           bp->nei[4][0][0] = o - init_block_x*init_block_y;
                        }
                        bp->nei_refine[4] = 0;
                        if (k2 == (init_block_z - 1))
                           if (k == (npz - 1)) { /* 1 boundary */
                              bp->nei_level[5] = -2;
                              bp->nei[5][0][0] = 0;
                           } else {      /* boundary with neighbor core */
                              bp->nei_level[5] = 0;
                              bp->nei[5][0][0] = -1 - pos1[i][j][k+1];
                              add_comm_list(2, o, pos1[i][j][k+1], 10+f,
                                            bp->cen[1]*mesh_size[0]+bp->cen[0],
                                            bp->cen[2] + size/2);
                           }
                        else {          /* neighbor on core */
                           bp->nei_level[5] = 0;
                           bp->nei[5][0][0] = o + init_block_x*init_block_y;
                        }
                        bp->nei_refine[5] = 0;
                        o++;
                     }
                  }

   check_buff_size();

   for (var = 0; var < num_vars; var++)
      grid_sum[var] = check_sum(var);
}
