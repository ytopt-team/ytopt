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

// This routine uses the communication structure for communicating ghost
// values to communicate the new processors of blocks that will be moving
// in the mesh during load balancing.  This coupled with blocks communicating
// to neighboring blocks on processor allows reconstruction of the structure
// used to communicate ghost values.
void comm_proc(void)
{
   int i, j, k, n, offset, dir, which, face, face_case, err, type;
   int *send_int = (int *) send_buff;
   int *recv_int = (int *) recv_buff;
   block *bp;
   MPI_Status status;

   for (dir = 0; dir < 3; dir++) {
      type = 60 + dir;
      for (i = 0; i < num_comm_partners[dir]; i++)
         MPI_Irecv(&recv_int[comm_index[dir][i]], comm_num[dir][i],
                   MPI_INT, comm_partner[dir][i], type, MPI_COMM_WORLD,
                   &request[i]);

      for (i = 0; i < num_comm_partners[dir]; i++) {
         offset = comm_index[dir][i];
         for (n = 0; n < comm_num[dir][i]; n++)
            send_int[offset+n] =
                        blocks[comm_block[dir][comm_index[dir][i]+n]].new_proc;
         MPI_Isend(&send_int[comm_index[dir][i]], comm_num[dir][i],
                   MPI_INT, comm_partner[dir][i], type, MPI_COMM_WORLD,
                   &s_req[i]);
      }

      for (i = 0; i < num_comm_partners[dir]; i++) {
         err = MPI_Waitany(num_comm_partners[dir], request, &which, &status);
         for (n = 0; n < comm_num[dir][which]; n++) {
            face = dir*2+(comm_face_case[dir][comm_index[dir][which]+n] >= 10);
            bp = &blocks[comm_block[dir][comm_index[dir][which]+n]];
            j = k = 0;
            face_case = comm_face_case[dir][comm_index[dir][which]+n]%10;
            if (face_case >= 6) {
               j = ((face_case+2)/2)%2;
               k = face_case%2;
            }
            bp->nei[face][j][k] = -1 - recv_int[comm_index[dir][which]+n];
         }
      }

      for (i = 0; i < num_comm_partners[dir]; i++)
         err = MPI_Waitany(num_comm_partners[dir], s_req, &which, &status);
   }
}
