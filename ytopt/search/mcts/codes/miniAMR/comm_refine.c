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

// This routine uses the communication pattern established for exchanging
// ghost values to exchange information about the refinement level and
// plans for refinement for neighboring blocks.
void comm_refine(void)
{
   int i, n, offset, dir, which, face, err, type;
   int *send_int = (int *) send_buff;
   int *recv_int = (int *) recv_buff;
   block *bp;
   MPI_Status status;

   for (dir = 0; dir < 3; dir++) {
      type = 10 + dir;
      for (i = 0; i < num_comm_partners[dir]; i++)
         MPI_Irecv(&recv_int[comm_index[dir][i]], comm_num[dir][i],
                   MPI_INT, comm_partner[dir][i], type, MPI_COMM_WORLD,
                   &request[i]);

      for (i = 0; i < num_comm_partners[dir]; i++) {
         offset = comm_index[dir][i];
         for (n = 0; n < comm_num[dir][i]; n++)
            send_int[offset+n] =
                          blocks[comm_block[dir][comm_index[dir][i]+n]].refine;
         MPI_Isend(&send_int[comm_index[dir][i]], comm_num[dir][i],
                   MPI_INT, comm_partner[dir][i], type, MPI_COMM_WORLD,
                   &s_req[i]);
      }

      for (i = 0; i < num_comm_partners[dir]; i++) {
         err = MPI_Waitany(num_comm_partners[dir], request, &which, &status);
         for (n = 0; n < comm_num[dir][which]; n++) {
            face = dir*2+(comm_face_case[dir][comm_index[dir][which]+n] >= 10);
            bp = &blocks[comm_block[dir][comm_index[dir][which]+n]];
            if (recv_int[comm_index[dir][which]+n] == 1 &&
                bp->nei_level[face] <= bp->level)
               bp->nei_refine[face] = 1;
            else if (recv_int[comm_index[dir][which]+n] >= 0 &&
                     bp->nei_refine[face] == -1)
               bp->nei_refine[face] = 0;
         }
      }

      for (i = 0; i < num_comm_partners[dir]; i++)
         err = MPI_Waitany(num_comm_partners[dir], s_req, &which, &status);
   }
}

void comm_reverse_refine(void)
{
   int i, n, c, offset, dir, which, face, err, type;
   int *send_int = (int *) send_buff;
   int *recv_int = (int *) recv_buff;
   block *bp;
   MPI_Status status;

   for (dir = 0; dir < 3; dir++) {
      type = 13 + dir;
      for (i = 0; i < num_comm_partners[dir]; i++)
         MPI_Irecv(&recv_int[comm_index[dir][i]], comm_num[dir][i],
                   MPI_INT, comm_partner[dir][i], type, MPI_COMM_WORLD,
                   &request[i]);

      for (i = 0; i < num_comm_partners[dir]; i++) {
         offset = comm_index[dir][i];
         for (n = 0; n < comm_num[dir][i]; n++) {
            face = dir*2 + (comm_face_case[dir][comm_index[dir][i]+n] >= 10);
            send_int[offset+n] =
                blocks[comm_block[dir][comm_index[dir][i]+n]].nei_refine[face];
         }
         MPI_Isend(&send_int[comm_index[dir][i]], comm_num[dir][i],
                   MPI_INT, comm_partner[dir][i], type, MPI_COMM_WORLD,
                   &s_req[i]);
      }

      for (i = 0; i < num_comm_partners[dir]; i++) {
         MPI_Waitany(num_comm_partners[dir], request, &which, &status);
         for (n = 0; n < comm_num[dir][which]; n++)
            if (recv_int[comm_index[dir][which]+n] >
                blocks[comm_block[dir][comm_index[dir][which]+n]].refine) {
               bp = &blocks[comm_block[dir][comm_index[dir][which]+n]];
               bp->refine = recv_int[comm_index[dir][which]+n];
               if (bp->parent != -1 && bp->parent_node == my_pe)
                  if (parents[bp->parent].refine == -1) {
                     parents[bp->parent].refine = 0;
                     for (c = 0; c < 8; c++)
                        if (parents[bp->parent].child_node[c] == my_pe &&
                            parents[bp->parent].child[c] >= 0 &&
                            blocks[parents[bp->parent].child[c]].refine == -1)
                           blocks[parents[bp->parent].child[c]].refine = 0;
                  }
            }
      }

      for (i = 0; i < num_comm_partners[dir]; i++)
         err = MPI_Waitany(num_comm_partners[dir], s_req, &which, &status);
   }
}
