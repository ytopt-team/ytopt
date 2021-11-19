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
#include <mpi.h>

#include "block.h"
#include "comm.h"
#include "proto.h"

// Routines to add and delete entries from the communication list that is
// used to exchange values for ghost cells.
void add_comm_list(int dir, int block_f, int pe, int fcase, int pos, int pos1)
{
   int i, j, s_len, r_len, *tmp;

   /* set indexes for send and recieve to determine length of message:
    * for example, if we send a whole face to a quarter face, we will
    * recieve a message sent from a quarter face to a whole face and
    * use 2 as index for the send and 3 for the recv.
    * We can use same index except for offset */
   if (fcase >= 10)    /* +- direction encoded in fcase */
      i = fcase - 10;
   else
      i = fcase;
   switch (i) {
      case 0: s_len = r_len = comm_vars*msg_len[dir][0];
              break;
      case 1: s_len = r_len = comm_vars*msg_len[dir][1];
              break;
      case 2:
      case 3:
      case 4:
      case 5: s_len = comm_vars*msg_len[dir][2];
              r_len = comm_vars*msg_len[dir][3];
              break;
      case 6:
      case 7:
      case 8:
      case 9: s_len = comm_vars*msg_len[dir][3];
              r_len = comm_vars*msg_len[dir][2];
              break;
   }

   for (i = 0; i < num_comm_partners[dir]; i++)
      if (comm_partner[dir][i] >= pe)
         break;

   /* i is being used below as an index where information about this
    * block should go */
   if (i < num_comm_partners[dir] && comm_partner[dir][i] == pe) {
      send_size[dir][i] += s_len;
      recv_size[dir][i] += r_len;
      for (j = num_comm_partners[dir]-1; j > i; j--)
         comm_index[dir][j]++;
      comm_num[dir][i]++;
   } else {
      // make sure arrays are long enough
      // move stuff i and above up one
      if (num_comm_partners[dir] == max_comm_part[dir]) {
         max_comm_part[dir] = (int)(2.0*((double) (num_comm_partners[dir]+1)));
         tmp = (int *) ma_malloc(max_comm_part[dir]*sizeof(int),
                                 __FILE__, __LINE__);
         for (j = 0; j < i; j++)
            tmp[j] = comm_partner[dir][j];
         for (j = i; j < num_comm_partners[dir]; j++)
            tmp[j+1] = comm_partner[dir][j];
         free(comm_partner[dir]);
         comm_partner[dir] = tmp;
         tmp = (int *) ma_malloc(max_comm_part[dir]*sizeof(int),
                                 __FILE__, __LINE__);
         for (j = 0; j < i; j++)
            tmp[j] = send_size[dir][j];
         for (j = i; j < num_comm_partners[dir]; j++)
            tmp[j+1] = send_size[dir][j];
         free(send_size[dir]);
         send_size[dir] = tmp;
         tmp = (int *) ma_malloc(max_comm_part[dir]*sizeof(int),
                                 __FILE__, __LINE__);
         for (j = 0; j < i; j++)
            tmp[j] = recv_size[dir][j];
         for (j = i; j < num_comm_partners[dir]; j++)
            tmp[j+1] = recv_size[dir][j];
         free(recv_size[dir]);
         recv_size[dir] = tmp;
         tmp = (int *) ma_malloc(max_comm_part[dir]*sizeof(int),
                                 __FILE__, __LINE__);
         for (j = 0; j <= i; j++)   // Note that this one is different
            tmp[j] = comm_index[dir][j];
         for (j = i; j < num_comm_partners[dir]; j++)
            tmp[j+1] = comm_index[dir][j] + 1;
         free(comm_index[dir]);
         comm_index[dir] = tmp;
         tmp = (int *) ma_malloc(max_comm_part[dir]*sizeof(int),
                                 __FILE__, __LINE__);
         for (j = 0; j < i; j++)
            tmp[j] = comm_num[dir][j];
         for (j = i; j < num_comm_partners[dir]; j++)
            tmp[j+1] = comm_num[dir][j];
         free(comm_num[dir]);
         comm_num[dir] = tmp;
      } else {
         for (j = num_comm_partners[dir]; j > i; j--) {
            comm_partner[dir][j] = comm_partner[dir][j-1];
            send_size[dir][j] = send_size[dir][j-1];
            recv_size[dir][j] = recv_size[dir][j-1];
            comm_index[dir][j] = comm_index[dir][j-1] + 1;
            comm_num[dir][j] = comm_num[dir][j-1];
         }
      }

      if (i == num_comm_partners[dir])
         if (i == 0)
            comm_index[dir][i] = 0;
         else
            comm_index[dir][i] = comm_index[dir][i-1] + comm_num[dir][i-1];
      num_comm_partners[dir]++;
      comm_partner[dir][i] = pe;
      send_size[dir][i] = s_len;
      recv_size[dir][i] = r_len;
      comm_num[dir][i] = 1;  // still have to put info into arrays
   }

   if ((num_cases[dir]+1) > max_num_cases[dir]) {
      max_num_cases[dir] = (int)(2.0*((double) (num_cases[dir]+1)));
      tmp = (int *) ma_malloc(max_num_cases[dir]*sizeof(int),
                              __FILE__, __LINE__);
      for (j = 0; j < num_cases[dir]; j++)
         tmp[j] = comm_block[dir][j];
      free(comm_block[dir]);
      comm_block[dir] = tmp;
      tmp = (int *) ma_malloc(max_num_cases[dir]*sizeof(int),
                              __FILE__, __LINE__);
      for (j = 0; j < num_cases[dir]; j++)
         tmp[j] = comm_face_case[dir][j];
      free(comm_face_case[dir]);
      comm_face_case[dir] = tmp;
      tmp = (int *) ma_malloc(max_num_cases[dir]*sizeof(int),
                              __FILE__, __LINE__);
      for (j = 0; j < num_cases[dir]; j++)
         tmp[j] = comm_pos[dir][j];
      free(comm_pos[dir]);
      comm_pos[dir] = tmp;
      tmp = (int *) ma_malloc(max_num_cases[dir]*sizeof(int),
                              __FILE__, __LINE__);
      for (j = 0; j < num_cases[dir]; j++)
         tmp[j] = comm_pos1[dir][j];
      free(comm_pos1[dir]);
      comm_pos1[dir] = tmp;
      tmp = (int *) ma_malloc(max_num_cases[dir]*sizeof(int),
                              __FILE__, __LINE__);
      for (j = 0; j < num_cases[dir]; j++)
         tmp[j] = comm_send_off[dir][j];
      free(comm_send_off[dir]);
      comm_send_off[dir] = tmp;
      tmp = (int *) ma_malloc(max_num_cases[dir]*sizeof(int),
                              __FILE__, __LINE__);
      for (j = 0; j < num_cases[dir]; j++)
         tmp[j] = comm_recv_off[dir][j];
      free(comm_recv_off[dir]);
      comm_recv_off[dir] = tmp;
   }
   if (comm_index[dir][i] == num_cases[dir]) {
      // at end
      comm_block[dir][num_cases[dir]] = block_f;
      comm_face_case[dir][num_cases[dir]] = fcase;
      comm_pos[dir][num_cases[dir]] = pos;
      comm_pos1[dir][num_cases[dir]] = pos1;
      comm_send_off[dir][num_cases[dir]] = s_buf_num[dir];
      comm_recv_off[dir][num_cases[dir]] = r_buf_num[dir];
   } else {
      for (j = num_cases[dir]; j > comm_index[dir][i]+comm_num[dir][i]-1; j--){
         comm_block[dir][j] = comm_block[dir][j-1];
         comm_face_case[dir][j] = comm_face_case[dir][j-1];
         comm_pos[dir][j] = comm_pos[dir][j-1];
         comm_pos1[dir][j] = comm_pos1[dir][j-1];
         comm_send_off[dir][j] = comm_send_off[dir][j-1] + s_len;
         comm_recv_off[dir][j] = comm_recv_off[dir][j-1] + r_len;
      }
      for (j = comm_index[dir][i]+comm_num[dir][i]-1;
           j >= comm_index[dir][i]; j--)
         if (j == comm_index[dir][i] || comm_pos[dir][j-1] < pos ||
             (comm_pos[dir][j-1] == pos && comm_pos1[dir][j-1] < pos1)) {
            comm_block[dir][j] = block_f;
            comm_face_case[dir][j] = fcase;
            comm_pos[dir][j] = pos;
            comm_pos1[dir][j] = pos1;
            if (j == num_cases[dir]) {
               comm_send_off[dir][j] = s_buf_num[dir];
               comm_recv_off[dir][j] = r_buf_num[dir];
            }
            // else comm_[send,recv]_off[j] values are correct
            break;
         } else {
            comm_block[dir][j] = comm_block[dir][j-1];
            comm_face_case[dir][j] = comm_face_case[dir][j-1];
            comm_pos[dir][j] = comm_pos[dir][j-1];
            comm_pos1[dir][j] = comm_pos1[dir][j-1];
            comm_send_off[dir][j] = comm_send_off[dir][j-1] + s_len;
            comm_recv_off[dir][j] = comm_recv_off[dir][j-1] + r_len;
         }
   }
   num_cases[dir]++;
   s_buf_num[dir] += s_len;
   r_buf_num[dir] += r_len;
}

void del_comm_list(int dir, int block_f, int pe, int fcase)
{
   int i, j, k, s_len, r_len;

   if (fcase >= 10)    /* +- direction encoded in fcase */
      i = fcase - 10;
   else if (fcase >= 0)
      i = fcase;
   else              /* special case to delete the one in a direction when
                        we don't know which quarter we were sending to */
      i = 2;
   switch (i) {
      case 0: s_len = r_len = comm_vars*msg_len[dir][0];
              break;
      case 1: s_len = r_len = comm_vars*msg_len[dir][1];
              break;
      case 2:
      case 3:
      case 4:
      case 5: s_len = comm_vars*msg_len[dir][2];
              r_len = comm_vars*msg_len[dir][3];
              break;
      case 6:
      case 7:
      case 8:
      case 9: s_len = comm_vars*msg_len[dir][3];
              r_len = comm_vars*msg_len[dir][2];
              break;
   }

   for (i = 0; i < num_comm_partners[dir]; i++)
      if (comm_partner[dir][i] == pe)
         break;

   /* i is being used below as an index where information about this
    * block is located */
   num_cases[dir]--;
   for (j = comm_index[dir][i]; j < comm_index[dir][i] + comm_num[dir][i]; j++)
      if (comm_block[dir][j] == block_f && (comm_face_case[dir][j] == fcase ||
          (fcase==  -1 && comm_face_case[dir][j] >= 2 &&
                          comm_face_case[dir][j] <= 5) ||
          (fcase== -11 && comm_face_case[dir][j] >=12 &&
                          comm_face_case[dir][j] <=15))) {
         for (k = j; k < num_cases[dir]; k++) {
            comm_block[dir][k] = comm_block[dir][k+1];
            comm_face_case[dir][k] = comm_face_case[dir][k+1];
            comm_pos[dir][k] = comm_pos[dir][k+1];
            comm_pos1[dir][k] = comm_pos1[dir][k+1];
            comm_send_off[dir][k] = comm_send_off[dir][k+1] - s_len;
            comm_recv_off[dir][k] = comm_recv_off[dir][k+1] - r_len;
         }
         break;
      }
   comm_num[dir][i]--;
   if (comm_num[dir][i]) {
      send_size[dir][i] -= s_len;
      recv_size[dir][i] -= r_len;
      for (j = i+1; j < num_comm_partners[dir]; j++)
         comm_index[dir][j]--;
   } else {
      num_comm_partners[dir]--;
      for (j = i; j < num_comm_partners[dir]; j++) {
         comm_partner[dir][j] = comm_partner[dir][j+1];
         send_size[dir][j] = send_size[dir][j+1];
         recv_size[dir][j] = recv_size[dir][j+1];
         comm_num[dir][j] = comm_num[dir][j+1];
         comm_index[dir][j] = comm_index[dir][j+1] - 1;
      }
   }

   s_buf_num[dir] -= s_len;
   r_buf_num[dir] -= r_len;
}

void zero_comm_list(void)
{
   int i;

   for (i = 0; i < 3; i++) {
      num_comm_partners[i] = 0;
      s_buf_num[i] = r_buf_num[i] = 0;
      comm_index[i][0] = 0;
      comm_send_off[i][0] = comm_recv_off[i][0] = 0;
   }
}

// check sizes of send and recv buffers and adjust, if necessary
void check_buff_size(void)
{
   int i, j, max_send, max_comm, max_recv;

   for (max_send = max_comm = max_recv = i = 0; i < 3; i++) {
      if (s_buf_num[i] > max_send)
         max_send = s_buf_num[i];
      if (num_comm_partners[i] > max_comm)
         max_comm = num_comm_partners[i];
      if (r_buf_num[i] > max_recv)
         max_recv = r_buf_num[i];
   }

   if (max_send > s_buf_size) {
      s_buf_size = (int) (2.0*((double) max_send));
      free(send_buff);
      send_buff = (double *) ma_malloc(s_buf_size*sizeof(double),
                                       __FILE__, __LINE__);
   }

   if (max_recv > r_buf_size) {
      r_buf_size = (int) (2.0*((double) max_recv));
      free(recv_buff);
      recv_buff = (double *) ma_malloc(r_buf_size*sizeof(double),
                                       __FILE__, __LINE__);
   }

   if (max_comm > max_num_req) {
      free(request);
      max_num_req = (int) (2.0*((double) max_comm));
      request = (MPI_Request *) ma_malloc(max_num_req*sizeof(MPI_Request),
                                          __FILE__, __LINE__);
      free(s_req);
      s_req = (MPI_Request *) ma_malloc(max_num_req*sizeof(MPI_Request),
                                        __FILE__, __LINE__);
   }
}

void update_comm_list(void)
{
   int dir, mcp, ncp, mnc, nc, i, j, n, c, f, i1, j1;
   int *cpe, *cn, *cb, *cf, *cpos, *cpos1;
   int *space = (int *) recv_buff;
   block *bp;

   mcp = num_comm_partners[0];
   if (num_comm_partners[1] > mcp)
      mcp = num_comm_partners[1];
   if (num_comm_partners[2] > mcp)
      mcp = num_comm_partners[2];
   mnc = num_cases[0];
   if (num_cases[1] > mnc)
      mnc = num_cases[1];
   if (num_cases[2] > mnc)
      mnc = num_cases[2];

   cpe = space;
   cn = &cpe[mcp];
   cb = &cn[mnc];
   cf = &cb[mnc];
   cpos = &cf[mnc];
   cpos1 = &cpos[mnc];

   for (dir = 0; dir < 3; dir++) {
      // make copies since original is going to be changed
      ncp = num_comm_partners[dir];
      for (i = 0; i < ncp; i++) {
         cpe[i] = comm_partner[dir][i];
         cn[i] = comm_num[dir][i];
      }
      nc = num_cases[dir];
      for (j = 0; j < nc; j++) {
         cb[j] = comm_block[dir][j];
         cf[j] = comm_face_case[dir][j];
         cpos[j] = comm_pos[dir][j];
         cpos1[j] = comm_pos1[dir][j];
      }

      // Go through communication lists and delete those that are being
      // sent from blocks being moved and change those where the the block
      // being communicated with is being moved (delete if moving here).
      for (n = i = 0; i < ncp; i++)
         for (j = 0; j < cn[i]; j++, n++) {
            bp = &blocks[cb[n]];
            if (bp->new_proc != my_pe)  // block being moved
               del_comm_list(dir, cb[n], cpe[i], cf[n]);
            else {
               if (cf[n] >= 10) {
                  f = cf[n] - 10;
                  c = 2*dir + 1;
               } else {
                  f = cf[n];
                  c = 2*dir;
               }
               if (f <= 5) {
                  if (bp->nei[c][0][0] != (-1 - cpe[i])) {
                     del_comm_list(dir, cb[n], cpe[i], cf[n]);
                     if ((-1 - bp->nei[c][0][0]) != my_pe)
                        add_comm_list(dir, cb[n], (-1 - bp->nei[c][0][0]),
                                      cf[n], cpos[n], cpos1[n]);
                  }
               } else {
                  i1 = (f - 6)/2;
                  j1 = f%2;
                  if (bp->nei[c][i1][j1] != (-1 - cpe[i])) {
                     del_comm_list(dir, cb[n], cpe[i], cf[n]);
                     if ((-1 - bp->nei[c][i1][j1]) != my_pe)
                        add_comm_list(dir, cb[n], (-1 - bp->nei[c][i1][j1]),
                                      cf[n], cpos[n], cpos1[n]);
                  }
               }
            }
         }
   }
}
