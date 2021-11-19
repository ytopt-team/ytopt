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

// These routines are concerned with communicating information between
// parents and their children.  This includes information about refinement
// level and also includes a routine to keep track to where children are
// being moved to.  For on node children, the parent has the index of the
// child block and for off node children has the block number.
void comm_parent(void)
{
   int i, j, b, which, type, offset;
   int *send_int = (int *) send_buff;
   int *recv_int = (int *) recv_buff;
   parent *pp;
   MPI_Status status;

   type = 20;
   for (i = 0; i < par_p.num_comm_part; i++)
      MPI_Irecv(&recv_int[par_p.index[i]], par_p.comm_num[i], MPI_INT,
                 par_p.comm_part[i], type, MPI_COMM_WORLD, &request[i]);

   for (i = 0; i < par_b.num_comm_part; i++) {
      offset = par_b.index[i];
      for (j = 0; j < par_b.comm_num[i]; j++)
         if (par_b.comm_b[par_b.index[i]+j] < 0)
            // parent, so send 0 (its parent can not refine)
            send_int[offset+j] = 0;
         else
            send_int[offset+j] = blocks[par_b.comm_b[par_b.index[i]+j]].refine;
      MPI_Isend(&send_int[par_b.index[i]], par_b.comm_num[i], MPI_INT,
                par_b.comm_part[i], type, MPI_COMM_WORLD, &s_req[i]);
   }

   for (i = 0; i < par_p.num_comm_part; i++) {
      MPI_Waitany(par_p.num_comm_part, request, &which, &status);
      for (j = 0; j < par_p.comm_num[which]; j++)
         if (recv_int[par_p.index[which]+j] > -1) {
            pp = &parents[par_p.comm_p[par_p.index[which]+j]];
            pp->refine = 0;
            for (b = 0; b < 8; b++)
               if (pp->child_node[b] == my_pe && pp->child[b] >= 0 &&
                   blocks[pp->child[b]].refine == -1)
                  blocks[pp->child[b]].refine = 0;
         }
   }

   for (i = 0; i < par_b.num_comm_part; i++)
      MPI_Waitany(par_b.num_comm_part, s_req, &which, &status);
}

void comm_parent_reverse(void)
{
   int i, j, which, type, offset;
   int *send_int = (int *) send_buff;
   int *recv_int = (int *) recv_buff;
   MPI_Status status;

   type = 21;
   for (i = 0; i < par_b.num_comm_part; i++)
      MPI_Irecv(&recv_int[par_b.index[i]], par_b.comm_num[i], MPI_INT,
                 par_b.comm_part[i], type, MPI_COMM_WORLD, &request[i]);

   for (i = 0; i < par_p.num_comm_part; i++) {
      offset = par_p.index[i];
      for (j = 0; j < par_p.comm_num[i]; j++)
         send_int[offset+j] = parents[par_p.comm_p[par_p.index[i]+j]].refine;
      MPI_Isend(&send_int[par_p.index[i]], par_p.comm_num[i], MPI_INT,
                par_p.comm_part[i], type, MPI_COMM_WORLD, &s_req[i]);
   }

   for (i = 0; i < par_b.num_comm_part; i++) {
      MPI_Waitany(par_b.num_comm_part, request, &which, &status);
      for (j = 0; j < par_b.comm_num[which]; j++)
         if (recv_int[par_b.index[which]+j] > -1 &&
             par_b.comm_b[par_b.index[which]+j] >= 0)
            if (blocks[par_b.comm_b[par_b.index[which]+j]].refine == -1)
               blocks[par_b.comm_b[par_b.index[which]+j]].refine = 0;
   }

   for (i = 0; i < par_p.num_comm_part; i++)
      MPI_Waitany(par_p.num_comm_part, s_req, &which, &status);
}

// Communicate new proc to parents - coordinate properly
// As new proc numbers come in, del the current and add the new
void comm_parent_proc(void)
{
   int i, j, which, type, offset;
   int *send_int = (int *) send_buff;
   int *recv_int = (int *) recv_buff;
   parent *pp;
   MPI_Status status;

   // duplicate par_p to par_p1
   if (par_p.num_comm_part > par_p1.max_part) {
      free(par_p1.comm_part);
      free(par_p1.comm_num);
      free(par_p1.index);
      par_p1.max_part = par_p.max_part;
      par_p1.comm_part = (int *) ma_malloc(par_p.max_part*sizeof(int),
                                           __FILE__, __LINE__);
      par_p1.comm_num = (int *) ma_malloc(par_p.max_part*sizeof(int),
                                          __FILE__, __LINE__);
      par_p1.index = (int *) ma_malloc(par_p.max_part*sizeof(int),
                                       __FILE__, __LINE__);
   }
   if (par_p.num_cases > par_p1.max_cases) {
      free(par_p1.comm_b);
      free(par_p1.comm_p);
      free(par_p1.comm_c);
      par_p1.max_cases = par_p.max_cases;
      par_p1.comm_b = (long long *) ma_malloc(par_p.max_cases*sizeof(long long),
                                        __FILE__, __LINE__);
      par_p1.comm_p = (long long *) ma_malloc(par_p.max_cases*sizeof(long long),
                                        __FILE__, __LINE__);
      par_p1.comm_c = (int *) ma_malloc(par_p.max_cases*sizeof(int),
                                        __FILE__, __LINE__);
   }

   par_p1.num_comm_part = par_p.num_comm_part;
   for (i = 0; i < par_p.num_comm_part; i++) {
      par_p1.comm_part[i] = par_p.comm_part[i];
      par_p1.comm_num[i] = par_p.comm_num[i];
      par_p1.index[i] = par_p.index[i];
   }
   par_p1.num_cases = par_p.num_cases;
   for (i = 0; i < par_p.num_cases; i++) {
      par_p1.comm_b[i] = par_p.comm_b[i];
      par_p1.comm_p[i] = par_p.comm_p[i];
      par_p1.comm_c[i] = par_p.comm_c[i];
   }

   type = 23;
   for (i = 0; i < par_p.num_comm_part; i++)
      MPI_Irecv(&recv_int[par_p.index[i]], par_p.comm_num[i], MPI_INT,
                 par_p.comm_part[i], type, MPI_COMM_WORLD, &request[i]);

   for (i = 0; i < par_b.num_comm_part; i++) {
      offset = par_b.index[i];
      for (j = 0; j < par_b.comm_num[i]; j++)
         if (par_b.comm_b[par_b.index[i]+j] < 0)
            // parent and will not move, so send current processor
            send_int[offset+j] = my_pe;
         else
            send_int[offset+j] =
                               blocks[par_b.comm_b[par_b.index[i]+j]].new_proc;
      MPI_Isend(&send_int[par_b.index[i]], par_b.comm_num[i], MPI_INT,
                par_b.comm_part[i], type, MPI_COMM_WORLD, &s_req[i]);
   }

   for (i = 0; i < par_p1.num_comm_part; i++) {
      MPI_Waitany(par_p1.num_comm_part, request, &which, &status);
      for (j = 0; j < par_p1.comm_num[which]; j++)
         if (recv_int[par_p1.index[which]+j] > -1) {
            pp = &parents[par_p1.comm_p[par_p1.index[which]+j]];
            if (pp->child_node[par_p1.comm_c[par_p1.index[which]+j]] !=
                  recv_int[par_p1.index[which]+j]) {
               del_par_list(&par_p, (num_sz) par_p1.comm_p[par_p1.index[which]+j],
                            (num_sz) par_p1.comm_b[par_p1.index[which]+j],
                            par_p1.comm_c[par_p1.index[which]+j],
                            par_p1.comm_part[which]);
               if (recv_int[par_p1.index[which]+j] != my_pe) {
                  add_par_list(&par_p, (num_sz) par_p1.comm_p[par_p1.index[which]+j],
                               (num_sz) par_p1.comm_b[par_p1.index[which]+j],
                               par_p1.comm_c[par_p1.index[which]+j],
                               recv_int[par_p1.index[which]+j], 1);
                  pp->child_node[par_p1.comm_c[par_p1.index[which]+j]] =
                        recv_int[par_p1.index[which]+j];
               } else
                  pp->child_node[par_p1.comm_c[par_p1.index[which]+j]] = my_pe;
            }
         }
   }

   for (i = 0; i < par_b.num_comm_part; i++)
      MPI_Waitany(par_b.num_comm_part, s_req, &which, &status);
}

// Below are routines for adding and deleting from arrays used above

void add_par_list(par_comm *pc, num_sz parent, num_sz block, int child, int pe,
                  int sort)
{
   int i, j, *tmp;
   num_sz *tmpl;

   // first add information into comm_part, comm_num, and index
   // i is being used as an index to where the info goes in the arrays
   for (i = 0; i < pc->num_comm_part; i++)
      if (pc->comm_part[i] >= pe)
         break;

   if (i < pc->num_comm_part && pc->comm_part[i] == pe) {
      for (j = pc->num_comm_part-1; j > i; j--)
         pc->index[j]++;
      pc->comm_num[i]++;
   } else {
      // adding new pe, make sure arrays are large enough
      if (pc->num_comm_part == pc->max_part) {
         pc->max_part = (int)(2.0*((double) (pc->num_comm_part + 1)));
         tmp = (int *) ma_malloc(pc->max_part*sizeof(int), __FILE__, __LINE__);
         for (j = 0; j < i; j++)
            tmp[j] = pc->comm_part[j];
         for (j = i; j < pc->num_comm_part; j++)
            tmp[j+1] = pc->comm_part[j];
         free(pc->comm_part);
         pc->comm_part = tmp;
         tmp = (int *) ma_malloc(pc->max_part*sizeof(int), __FILE__, __LINE__);
         for (j = 0; j < i; j++)
            tmp[j] = pc->comm_num[j];
         for (j = i; j < pc->num_comm_part; j++)
            tmp[j+1] = pc->comm_num[j];
         free(pc->comm_num);
         pc->comm_num = tmp;
         tmp = (int *) ma_malloc(pc->max_part*sizeof(int), __FILE__, __LINE__);
         for (j = 0; j <= i; j++)
            tmp[j] = pc->index[j];
         for (j = i; j < pc->num_comm_part; j++)
            tmp[j+1] = pc->index[j] + 1;
         free(pc->index);
         pc->index = tmp;
      } else {
         for (j = pc->num_comm_part; j > i; j--) {
            pc->comm_part[j] = pc->comm_part[j-1];
            pc->comm_num[j] = pc->comm_num[j-1];
            pc->index[j] = pc->index[j-1] + 1;
         }
      }
      if (i == pc->num_comm_part)
         pc->index[i] = pc->num_cases;
      pc->num_comm_part++;
      pc->comm_part[i] = pe;
      pc->comm_num[i] = 1;
   }

   // now add into to comm_p and comm_b according to index
   // first check if there is room in the arrays
   if (pc->num_cases == pc->max_cases) {
      pc->max_cases = (int)(2.0*((double) (pc->num_cases+1)));
      tmpl = (num_sz *) ma_malloc(pc->max_cases*sizeof(num_sz),
                                  __FILE__, __LINE__);
      for (j = 0; j < pc->num_cases; j++)
         tmpl[j] = pc->comm_p[j];
      free(pc->comm_p);
      pc->comm_p = tmpl;
      tmpl = (num_sz *) ma_malloc(pc->max_cases*sizeof(num_sz),
                                  __FILE__, __LINE__);
      for (j = 0; j < pc->num_cases; j++)
         tmpl[j] = pc->comm_b[j];
      free(pc->comm_b);
      pc->comm_b = tmpl;
      tmp = (int *) ma_malloc(pc->max_cases*sizeof(int), __FILE__, __LINE__);
      for (j = 0; j < pc->num_cases; j++)
         tmp[j] = pc->comm_c[j];
      free(pc->comm_c);
      pc->comm_c = tmp;
   }
   if (pc->index[i] == pc->num_cases) {
      // at end of arrays
      pc->comm_p[pc->num_cases] = parent;
      pc->comm_b[pc->num_cases] = block;
      pc->comm_c[pc->num_cases] = child;
   } else {
      for (j = pc->num_cases; j >= pc->index[i]+pc->comm_num[i]; j--) {
         pc->comm_p[j] = pc->comm_p[j-1];
         pc->comm_b[j] = pc->comm_b[j-1];
         pc->comm_c[j] = pc->comm_c[j-1];
      }
      for (j = pc->index[i]+pc->comm_num[i]-1; j >= pc->index[i]; j--) {
         if (j == pc->index[i] ||
             (sort && (parents[pc->comm_p[j-1]].number < parents[parent].number
                || (pc->comm_p[j-1] == parent && pc->comm_c[j-1] < child))) ||
             (!sort && (pc->comm_p[j-1] < parent
                || (pc->comm_p[j-1] == parent && pc->comm_c[j-1] < child)))) {
            pc->comm_p[j] = parent;
            pc->comm_b[j] = block;
            pc->comm_c[j] = child;
            break;
         } else {
            pc->comm_p[j] = pc->comm_p[j-1];
            pc->comm_b[j] = pc->comm_b[j-1];
            pc->comm_c[j] = pc->comm_c[j-1];
         }
      }
   }
   pc->num_cases++;
}

void del_par_list(par_comm *pc, num_sz parent, num_sz block, int child, int pe)
{
   int i, j, k;

   // find core number in index list and use i below
   for (i = 0; i < pc->num_comm_part; i++)
      if (pc->comm_part[i] == pe)
         break;

   // find and delete case in comm_p, comm_b, and comm_c
   pc->num_cases--;
   for (j = pc->index[i]; j < pc->index[i]+pc->comm_num[i]; j++)
      if (pc->comm_p[j] == parent && pc->comm_c[j] == child) {
         for (k = j; k < pc->num_cases; k++) {
            pc->comm_p[k] = pc->comm_p[k+1];
            pc->comm_b[k] = pc->comm_b[k+1];
            pc->comm_c[k] = pc->comm_c[k+1];
         }
         break;
      }
   // fix index and adjust comm_part and comm_num
   pc->comm_num[i]--;
   if (pc->comm_num[i])
      for (j = i+1; j < pc->num_comm_part; j++)
         pc->index[j]--;
   else {
      pc->num_comm_part--;
      for (j = i; j < pc->num_comm_part; j++) {
         pc->comm_part[j] = pc->comm_part[j+1];
         pc->comm_num[j] = pc->comm_num[j+1];
         pc->index[j] = pc->index[j+1] - 1;
      }
   }
}
