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

#include "block.h"
#include "comm.h"
#include "proto.h"

// Pack and unpack a block to move blocks between processors.
void pack_block(int n)
{
   // Pack a block's values into a buffer to send.  This and the
   // next routine which unpacks the values depend on the assumption
   // that the size of an integer is no larger than a double.
   int v, i, j, k, l;
   int *send_int = (int *) send_buff;
   long long *send_ll = (long long *) send_buff;
   block *bp = &blocks[n];

   send_ll[0] = (long long) bp->number;
   if (bp->parent_node == my_pe && bp->parent != -1)
      // bp->parent converted to parent number (from index) in move_blocks
      send_ll[1] = (long long) (-2 - bp->parent);
   else
      send_ll[1] = (long long) bp->parent;
   l = 4;
   send_int[l++] = bp->level;
   send_int[l++] = bp->refine;
   send_int[l++] = bp->parent_node;
   send_int[l++] = bp->child_number;
   for (i = 0; i < 6; i++) {
      send_int[l++] = bp->nei_refine[i];
      send_int[l++] = bp->nei_level[i];
      for (j = 0; j < 2; j++)
         for (k = 0; k < 2; k++)
            send_int[l++] = bp->nei[i][j][k];
   }
   for (i = 0; i < 3; i++)
      send_int[l++] = bp->cen[i];

   for (v = 0; v < num_vars; v++) {
      typedef double (*block3D_t)[y_block_size+2][z_block_size+2];
      block3D_t array = (block3D_t)&bp->array[v*block3D_size];
      for (i = 1; i <= x_block_size; i++)
         for (j = 1; j <= y_block_size; j++)
            for (k = 1; k <= z_block_size; k++)
               send_buff[l++] = array[i][j][k];
   }
}

void unpack_block(int n)
{
   int v, i, j, k, l;
   int *recv_int = (int *) recv_buff;
   long long *recv_ll = (long long *) recv_buff;
   block *bp = &blocks[n];

   // distinguish blocks that are being moved into a processor
   bp->new_proc = -1;

   bp->number = (num_sz) recv_ll[0];
   bp->parent = (num_sz) recv_ll[1];
   l = 4;
   bp->level = recv_int[l++];
   bp->refine = recv_int[l++];
   bp->parent_node = recv_int[l++];
   bp->child_number = recv_int[l++];
   for (i = 0; i < 6; i++) {
      bp->nei_refine[i] = recv_int[l++];
      bp->nei_level[i] = recv_int[l++];
      for (j = 0; j < 2; j++)
         for (k = 0; k < 2; k++)
            bp->nei[i][j][k] = recv_int[l++];
   }
   for (i = 0; i < 3; i++)
      bp->cen[i] = recv_int[l++];

   for (v = 0; v < num_vars; v++) {
      typedef double (*block3D_t)[y_block_size+2][z_block_size+2];
      block3D_t array = (block3D_t)&bp->array[v*block3D_size];
      for (i = 1; i <= x_block_size; i++)
         for (j = 1; j <= y_block_size; j++)
            for (k = 1; k <= z_block_size; k++)
               array[i][j][k] = recv_buff[l++];
   }
}
