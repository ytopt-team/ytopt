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

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "block.h"
#include "comm.h"
#include "timer.h"
#include "proto.h"

// This file includes routines needed for load balancing.  Load balancing is
// based on RCB.  At each stage, a direction and factor is chosen (factor is
// based on the prime factorization of the number of processors) and the
// blocks in that group are sorted in that direction and divided into factor
// subgroups.  Then dots (corresponding to blocks) are moved into the proper
// subgroup and the process is repeated with the subgroups until each group
// represents a processor.  The dots are then moved back to the originating
// processor, at which point we know where the blocks need to be moved and
// then the blocks are moved.  Some of these routines are also used when
// blocks need to be coarsened - the coarsening routine determines which
// blocks need to be coarsened and those blocks are moved to the processor
// where their parent is.
void load_balance(void)
{
   int npx1, npy1, npz1, nfac, fac[25], fact;
   int i, j, m, n, dir, in;
   double t1, t2, t3, t4, t5, tp, tm, tu;
   block *bp;

   tp = tm = tu = 0.0;

   t3 = t4 = t5 = 0.0;
   t1 = timer();
   for (in = 0, num_dots = 0; in < sorted_index[num_refine+1]; in++) {
      bp = &blocks[n = sorted_list[in].n];
      bp->new_proc = my_pe;
      if ((num_dots+1) > max_num_dots) {
         printf("%d ERROR: need more dots\n", my_pe);
         exit(-1);
      }
      dots[num_dots].cen[0] = bp->cen[0];
      dots[num_dots].cen[1] = bp->cen[1];
      dots[num_dots].cen[2] = bp->cen[2];
      dots[num_dots].number = bp->number;
      dots[num_dots].n = n;
      dots[num_dots].proc = my_pe;
      dots[num_dots++].new_proc = 0;
   }
   max_active_dot = num_dots;
   for (n = num_dots; n < max_num_dots; n++)
      dots[n].number = -1;

   npx1 = npx;
   npy1 = npy;
   npz1 = npz;
   nfac = factor(num_pes, fac);
   for (i = nfac, j = 0; i > 0; i--, j++) {
      fact = fac[i-1];
      dir = find_dir(fact, npx1, npy1, npz1);
      if (dir == 0)
         npx1 /= fact;
      else if (dir == 1)
         npy1 /= fact;
      else
         npz1 /= fact;
      sort(j, fact, dir);
      move_dots(j, fact);
   }
   // first have to move information from dots back to original core,
   // then will update processor block is moving to, and then its neighbors
   for (n = 0; n < num_pes; n++)
      to[n] = 0;
   for (m = i = 0; i < max_active_dot; i++)
      if (dots[i].number >= 0 && dots[i].proc != my_pe) {
         to[dots[i].proc]++;
         m++;
      }

   num_moved_lb += m;
   MPI_Allreduce(&m, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   t4 = timer();
   t2 = t4 - t1;
   if (n) {  // Only move dots and blocks if there is something to move
      MPI_Alltoall(to, 1, MPI_INT, from, 1, MPI_INT, MPI_COMM_WORLD);

      move_dots_back();
      t5 = timer();
      t3 = t5 - t4;
      t4 = t5;

      move_blocks(&tp, &tm, &tu);
   }
   t5 = timer() - t4;
   timer_lb_misc += timer() - t1 - t2 - t3 - tp - tm - tu;
   timer_lb_sort += t2;
   timer_lb_pa += tp;
   timer_lb_mv += tm;
   timer_lb_un += tu;
   timer_lb_mb += t3;
   timer_lb_ma += t5;
}

void exchange(double *tp, double *tm, double *tu)
{
   int f, s, sp, fp, i, j[25], l, rb, lev, block_size, type, type1,
       par[25], start[25];
   double t1, t2, t3, t4;
   MPI_Status status;

   block_size = 47 + num_vars*num_cells;
   type = 40;
   type1 = 41;

   par[0] = 1;
   lev = 0;
   while (par[lev] < num_pes) {
      par[lev+1] = 2*par[lev];
      lev++;
   }
   j[l = 0] = 0;
   start[0] = 0;
   while(j[0] < 2) {
      if (l == lev) {
         t3 = t4 = 0.0;
         t1 = timer();
         sp = fp = s = f = 0;
         // The sense of to and from are reversed in this routine.  They are
         // related to moving the dots from where they ended up back to the
         // processors that they came from and blocks are moved in reverse.
         while (s < from[start[l]] || f < to[start[l]]) {
            if (f < to[start[l]]) {
               if (num_active < max_num_blocks) {
                  MPI_Irecv(recv_buff, block_size, MPI_DOUBLE, start[l], type,
                            MPI_COMM_WORLD, request);
                  rb = 1;
               } else
                  rb = 0;
               MPI_Send(&rb, 1, MPI_INT, start[l], type1, MPI_COMM_WORLD);
            }
            if (s < from[start[l]]) {
               MPI_Recv(&i, 1, MPI_INT, start[l], type1,
                        MPI_COMM_WORLD, &status);
               if (i) {
                  while (sp < max_active_block && blocks[sp].number < 0 ||
                         (blocks[sp].number >= 0 &&
                            (blocks[sp].new_proc != start[l] ||
                             blocks[sp].new_proc == my_pe)))
                     sp++;
                  t2 = timer();
                  pack_block(sp);
                  t3 += timer() - t2;
                  num_active--;
                  local_num_blocks[blocks[sp].level]--;
                  del_sorted_list(blocks[sp].number, blocks[sp].level, 3);
                  blocks[sp].number = -1;
                  MPI_Send(send_buff, block_size, MPI_DOUBLE, start[l], type,
                           MPI_COMM_WORLD);
                  if (fp > sp)
                     fp = sp;
                  sp++;
                  from[start[l]]--;
               } else
                  s = from[start[l]];
            }
            if (f < to[start[l]]) {
               if (rb) {
                  while (fp < max_num_blocks && blocks[fp].number >= 0)
                     fp++;
                  if (fp == max_num_blocks)
                     if (num_active == max_num_blocks) {
                        printf("ERROR: exchange - need more blocks\n");
                        exit(-1);
                     } else {  // there is at least one free block
                        fp = 0;
                        while (blocks[fp].number >= 0)
                           fp++;
                     }
                  MPI_Wait(request, &status);
                  t2 = timer();
                  unpack_block(fp);
                  t4 += timer() - t2;
                  if ((fp+1) > max_active_block)
                     max_active_block = fp + 1;
                  num_active++;
                  local_num_blocks[blocks[fp].level]++;
                  add_sorted_list(fp, blocks[fp].number, blocks[fp].level);
                  fp++;
                  to[start[l]]--;
               } else
                  f = to[start[l]];
            }
         }
         *tm += timer() - t1 - t3 - t4;
         *tp += t3;
         *tu += t4;

         l--;
         j[l]++;
      } else if (j[l] == 0) {
         j[l+1] = 0;
         if (my_pe & par[l])
            if (start[l]+par[l] < num_pes)
               start[l+1] = start[l] + par[l];
            else {
               start[l+1] = start[l];
               j[l] = 1;
            }
         else
            start[l+1] = start[l];
         l++;
      } else if (j[l] == 1) {
         j[l+1] = 0;
         if (my_pe & par[l]) {
            start[l+1] = start[l];
            l++;
         } else
            if (start[l]+par[l] < num_pes) {
               start[l+1] = start[l] + par[l];
               l++;
            } else {
               l--;
               j[l]++;
            }
      } else {
         l--;
         j[l]++;
      }
   }

}

// Sort by binning the dots.  Bin in the desired direction, find the bin
// (or bin) that divides, bin in another direction, and repeat that
// for the third direction.  Then with the three bins (if we need all three
// to get a good binning), we can divide the dots.
// Communicate the dots at the end of each stage.  At the end, use the dots
// to determine where to send the blocks to.
void sort(int div, int fact, int dir)
{
   int i, j, sum, total_dots, part, dir1, point1, extra1,
       bin1[fact], point[fact], extra[fact];

   MPI_Allreduce(&num_dots, &total_dots, 1, MPI_INT, MPI_SUM, comms[div]);

   for (i = 0; i < mesh_size[dir]; i++)
      bin[i] = 0;

   for (i = 0; i < max_active_dot; i++)
      if (dots[i].number >= 0)
         bin[dots[i].cen[dir]]++;

   MPI_Allreduce(bin, gbin, mesh_size[dir], MPI_INT, MPI_SUM, comms[div]);

   part = (total_dots+fact-1)/fact;
   for (sum = j = i = 0; i < mesh_size[dir] && j < (fact-1); i++) {
      sum += gbin[i];
      if (sum >= (j+1)*part) {
         bin1[j] = gbin[i];
         extra[j] = sum - (j+1)*part;
         point[j++] = i;
      }
   }

   for (i = 0; i < max_active_dot; i++)
      if (dots[i].number >= 0) {
         for (j = 0; j < (fact-1); j++)
            if (dots[i].cen[dir] <  point[j]) {
               dots[i].new_proc = j;
               break;
            } else if (dots[i].cen[dir] == point[j]) {
               if (extra[j])
                  dots[i].new_proc = -1 - j;
               else
                  dots[i].new_proc = j;
               break;
            }
         if (j == (fact-1))
            dots[i].new_proc = j;
      }

   for (j = 0; j < (fact-1); j++)
      if (extra[j]) {
         dir1 = (dir+1)%3;
         for (i = 0; i < mesh_size[dir1]; i++)
            bin[i] = 0;
         for (i = 0; i < max_active_dot; i++)
            if (dots[i].number >= 0 && dots[i].new_proc == (-1-j))
               bin[dots[i].cen[dir1]]++;
         MPI_Allreduce(bin, gbin, mesh_size[dir1], MPI_INT, MPI_SUM,
                       comms[div]);
         part = bin1[j] - extra[j];
         for (sum = i = 0; i < mesh_size[dir1]; i++) {
            sum += gbin[i];
            if (sum >= part) {
               extra1 = sum - part;
               point1 = i;
               bin1[j] = gbin[i];
               break;
            }
         }
         for (i = 0; i < max_active_dot; i++)
            if (dots[i].number >= 0)
               if (dots[i].new_proc == (-1-j))
                  if (dots[i].cen[dir1] < point1)
                     dots[i].new_proc = j;
                  else if (dots[i].cen[dir1] == point1) {
                     if (!extra1)
                        dots[i].new_proc = j;
                     // else dots[i].new_proc = (-1-j) - no change
                  } else
                     dots[i].new_proc = j + 1;
         if (extra1) {
            dir1 = (dir+2)%3;
            for (i = 0; i < mesh_size[dir1]; i++)
               bin[i] = 0;
            for (i = 0; i < max_active_dot; i++)
               if (dots[i].number >= 0 && dots[i].new_proc == (-1-j))
                  bin[dots[i].cen[dir1]]++;
            MPI_Allreduce(bin, gbin, mesh_size[dir1], MPI_INT, MPI_SUM,
                          comms[div]);
            part = bin1[j] - extra1;
            for (sum = i = 0; i < mesh_size[dir1]; i++) {
               sum += gbin[i];
               if (sum >= part) {
                  point1 = i;
                  break;
               }
            }
            for (i = 0; i < max_active_dot; i++)
               if (dots[i].number >= 0)
                  if (dots[i].new_proc == (-1-j))
                     if (dots[i].cen[dir1] <= point1)
                        dots[i].new_proc = j;
                     else
                        dots[i].new_proc = j+1;
         }
      }
}

int factor(int np, int *fac)
{
   int nfac = 0, mfac = 2, done = 0;

   while (!done)
      if (np == (np/mfac)*mfac) {
         fac[nfac++] = mfac;
         np /= mfac;
         if (np == 1)
            done = 1;
      } else {
         mfac++;
         if (mfac*mfac > np) {
            fac[nfac++] = np;
            done = 1;
         }
      }

   return nfac;
}

int find_dir(int fact, int npx1, int npy1, int npz1)
{
   /* Find direction with largest number of processors left
    * that is divisible by the factor.
    */
   int dir;

   if (reorder) {
      if (fact > 2)
         if ((npx1/fact)*fact == npx1)
            if ((npy1/fact)*fact == npy1)
               if ((npz1/fact)*fact == npz1)
                  if (npx1 >= npy1)
                     if (npx1 >= npz1)
                        dir = 0;
                     else
                        dir = 2;
                  else
                     if (npy1 >= npz1)
                        dir = 1;
                     else
                        dir = 2;
               else
                  if (npx1 >= npy1)
                     dir = 0;
                  else
                     dir = 1;
            else
               if (((npz1/fact)*fact) == npz1)
                  if (npx1 >= npz1)
                     dir = 0;
                  else
                     dir = 2;
               else
                  dir = 0;
         else
            if ((npy1/fact)*fact == npy1)
               if (((npz1/fact)*fact) == npz1)
                  if (npy1 >= npz1)
                     dir = 1;
                  else
                     dir = 2;
               else
                  dir = 1;
            else
               dir = 2;
      else /* factor is 2 and np[xyz]1 are either 1 or a factor of 2 */
         if (npx1 >= npy1)
            if (npx1 >= npz1)
               dir = 0;
            else
               dir = 2;
         else
            if (npy1 >= npz1)
               dir = 1;
            else
               dir = 2;
   } else {
      /* if not reorder, divide z fist, y second, and x last */
      if (fact > 2)
         if ((npz1/fact)*fact == npz1)
            dir = 2;
         else if ((npy1/fact)*fact == npy1)
            dir = 1;
         else
            dir = 0;
      else
         if (npz1 > 1)
            dir = 2;
         else if (npy1 > 1)
            dir = 1;
         else
            dir = 0;
   }

   return dir;
}

void move_dots(int div, int fact)
{
   int i, j, d, sg, mg, partner, type, off[fact+1], which, err, nr;
   int *send_int = (int *) send_buff;
   int *recv_int = (int *) recv_buff;
   long long *send_ll, *recv_ll;
   MPI_Status status;

   sg = np[div]/fact;
   mg = me[div]/sg;

   for (i = 0; i < fact; i++)
      bin[i] = 0;

   // determine which proc to send dots to
   for (d = 0; d < max_active_dot; d++)
      if (dots[d].number >= 0)
         bin[dots[d].new_proc]++;

   type = 30;
   for (i = 0; i < fact; i++)
      if (i != mg) {
         partner = me[div]%sg + i*sg;
         MPI_Irecv(&gbin[i], 1, MPI_INT, partner, type, comms[div],
                   &request[i]);
      }

   for (i = 0; i < fact; i++)
      if (i != mg) {
         partner = me[div]%sg + i*sg;
         MPI_Send(&bin[i], 1, MPI_INT, partner, type, comms[div]);
      }

   type = 31;
   off[0] = 0;
   for (nr = i = 0; i < fact; i++)
      if (i != mg) {
         err = MPI_Wait(&request[i], &status);
         if (gbin[i] > 0) {
            partner = me[div]%sg + i*sg;
            MPI_Irecv(&recv_int[off[i]], 8*gbin[i], MPI_INT, partner,
                      type, comms[div], &request[i]);
            off[i+1] = off[i] + 8*gbin[i];
            nr++;
         } else {
            off[i+1] = off[i];
            request[i] = MPI_REQUEST_NULL;
         }
      } else {
         off[i+1] = off[i];
         request[i] = MPI_REQUEST_NULL;
      }

   for (i = 0; i < fact; i++)
      if (i != mg && bin[i] > 0) {
         for (j = d = 0; d < max_active_dot; d++)
            if (dots[d].number >= 0 && dots[d].new_proc == i) {
               send_ll = (long long *) &send_int[j];
               j += 2;
               send_ll[0] = (long long) dots[d].number;
               send_int[j++] = dots[d].cen[0];
               send_int[j++] = dots[d].cen[1];
               send_int[j++] = dots[d].cen[2];
               send_int[j++] = dots[d].n;
               send_int[j++] = dots[d].proc;
               j++;
               dots[d].number = -1;
               num_dots--;
            }

         partner = me[div]%sg + i*sg;
         MPI_Send(send_int, 8*bin[i], MPI_INT, partner, type, comms[div]);
      }

   for (d = i = 0; i < nr; i++) {
      err = MPI_Waitany(fact, request, &which, &status);
      for (j = off[which]; j < off[which+1]; ) {
         for ( ; d < max_num_dots; d++)
            if (dots[d].number < 0)
               break;
         if (d == max_num_dots) {
            printf("%d ERROR: need more dots in move_dots %d %d\n",
                   my_pe, max_num_dots, num_dots);
            exit(-1);
         }
         recv_ll = (long long *) &recv_int[j];
         j += 2;
         dots[d].number = (num_sz) recv_ll[0];
         dots[d].cen[0] = recv_int[j++];
         dots[d].cen[1] = recv_int[j++];
         dots[d].cen[2] = recv_int[j++];
         dots[d].n = recv_int[j++];
         dots[d].proc = recv_int[j++];
         j++;
         num_dots++;
         if ((d+1) > max_active_dot)
            max_active_dot = d+1;
      }
   }
}

void move_dots_back()
{
   int i, j, d, nr, err, which;
   int *send_int = (int *) send_buff;
   int *recv_int = (int *) recv_buff;
   MPI_Status status;

   gbin[0] = 0;
   for (nr = i = 0; i < num_pes; i++)
      if (from[i] > 0) {
         gbin[i+1] = gbin[i] + 2*from[i];
         MPI_Irecv(&recv_int[gbin[i]], 2*from[i], MPI_INT, i, 50,
                   MPI_COMM_WORLD, &request[i]);
         nr++;
      } else {
         gbin[i+1] = gbin[i];
         request[i] = MPI_REQUEST_NULL;
      }

   for (i = 0; i < num_pes; i++)
      if (to[i] > 0) {
         for (j = d = 0; d < max_active_dot; d++)
            if (dots[d].number >= 0 && dots[d].proc == i) {
               send_int[j++] = dots[d].n;
               send_int[j++] = my_pe;
            }
         MPI_Send(send_int, 2*to[i], MPI_INT, i, 50, MPI_COMM_WORLD);
      }

   for (i = 0; i < nr; i++) {
      err = MPI_Waitany(num_pes, request, &which, &status);
      for (j = 0; j < from[which]; j++)
         blocks[recv_int[gbin[which]+2*j]].new_proc =
               recv_int[gbin[which]+2*j+1];
   }
}

void move_blocks(double *tp, double *tm, double *tu)
{
   static int mul[3][3] = { {1, 2, 0}, {0, 2, 1}, {0, 1, 2} };
   int n, n1, p, c, c1, dir, i, j, k, i1, j1, k1, in,
       offset, off[3], f, fcase, proc;
   num_sz number, nl, pos[3];
   block *bp, *bp1;

   if (stencil == 7)  // add to face case when diags are needed
      f = 0;
   else
      f = 1;

   comm_proc();
   comm_parent_proc();
   update_comm_list();

   // go through blocks being moved and reset their nei[] list
   // (partially done above with comm_proc) and the lists of their neighbors
   for (in = 0; in < sorted_index[num_refine+1]; in++) {
      bp = &blocks[n = sorted_list[in].n];
      if (bp->new_proc != my_pe) {
         for (c = 0; c < 6; c++) {
            c1 = (c/2)*2 + (c+1)%2;
            dir = c/2;
            fcase = (c1%2)*10;
            if (bp->nei_level[c] == (bp->level-1)) {
               if (bp->nei[c][0][0] >= 0)
                  for (k = fcase+6, i = 0; i < 2; i++)
                     for (j = 0; j < 2; j++, k++)
                        if (blocks[bp->nei[c][0][0]].nei[c1][i][j] == n) {
                           bp1 = &blocks[bp->nei[c][0][0]];
                           offset = p2[num_refine - bp1->level - 1];
                           bp1->nei[c1][i][j] = -1 - bp->new_proc;
                           bp1->nei_refine[c1] = bp->refine;
                           if (bp1->new_proc == my_pe)
                             add_comm_list(dir, bp->nei[c][0][0], bp->new_proc,
                                            k, ((bp1->cen[mul[dir][1]]+(2*i-1)*
                                                offset)*mesh_size[mul[dir][0]]
                                       + bp1->cen[mul[dir][0]]+(2*j-1)*offset),
                                            (bp1->cen[mul[dir][2]] +
                                    (2*(c1%2)-1)*p2[num_refine - bp1->level]));
                           bp->nei_refine[c] = bp1->refine;
                           bp->nei[c][0][0] = -1 - bp1->new_proc;
                           goto done;
                        }
               done: ;
            } else if (bp->nei_level[c] == bp->level) {
               if (bp->nei[c][0][0] >= 0) {
                  bp1 = &blocks[bp->nei[c][0][0]];
                  bp1->nei[c1][0][0] = -1 - bp->new_proc;
                  bp1->nei_refine[c1] = bp->refine;
                  if (bp1->new_proc == my_pe)
                    add_comm_list(dir, bp->nei[c][0][0], bp->new_proc, fcase+f,
                                (bp1->cen[mul[dir][1]]*mesh_size[mul[dir][0]] +
                                    bp1->cen[mul[dir][0]]),
                                   (bp1->cen[mul[dir][2]] +
                                    (2*(c1%2)-1)*p2[num_refine - bp1->level]));
                  bp->nei_refine[c] = bp1->refine;
                  bp->nei[c][0][0] = -1 - bp1->new_proc;
               }
            } else if (bp->nei_level[c] == (bp->level+1)) {
               for (k = fcase+2, i = 0; i < 2; i++)
                  for (j = 0; j < 2; j++, k++)
                     if (bp->nei[c][i][j] >= 0) {
                        bp1 = &blocks[bp->nei[c][i][j]];
                        bp1->nei[c1][0][0] = -1 - bp->new_proc;
                        bp1->nei_refine[c1] = bp->refine;
                        if (bp1->new_proc == my_pe)
                          add_comm_list(dir, bp->nei[c][i][j], bp->new_proc, k,
                                (bp1->cen[mul[dir][1]]*mesh_size[mul[dir][0]] +
                                    bp1->cen[mul[dir][0]]),
                                   (bp1->cen[mul[dir][2]] +
                                    (2*(c1%2)-1)*p2[num_refine- bp1->level]));
                        bp->nei_refine[c] = bp1->refine;
                        bp->nei[c][i][j] = -1 - bp1->new_proc;
                     }
            }
         }
         // move parent connection in blocks being moved
         if (bp->parent != -1)
            if (bp->parent_node == my_pe) {
               parents[bp->parent].child[bp->child_number] = bp->number;
               parents[bp->parent].child_node[bp->child_number] = bp->new_proc;
               add_par_list(&par_p, bp->parent, bp->number, bp->child_number,
                            bp->new_proc, 1);
               bp->parent = parents[bp->parent].number;
            } else
               del_par_list(&par_b, (-2-bp->parent), (num_sz) n,
                            bp->child_number, bp->parent_node);
      }
   }

   /* swap blocks - if space is tight, may take multiple passes */
   n = 0;
   do {
      exchange(tp, tm, tu);
      k = n;
      for (n1 = i = 0; i < num_pes; i++)
         n1 += from[i];
      MPI_Allreduce(&n1, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   } while (n && k != n);

   if (n && !my_pe) {
      printf("Error: exchange blocks not complete - increase max_blocks\n");
      exit(-1);
   }

   // reestablish on-core and off-core comm lists
   for (in = 0; in < sorted_index[num_refine+1]; in++) {
      bp = &blocks[n = sorted_list[in].n];
      if (bp->new_proc == -1) {
         nl = bp->number - block_start[bp->level];
         pos[2] = nl/((p2[bp->level]*npx*init_block_x)*
                      (p2[bp->level]*npy*init_block_y));
         pos[1] = (nl%((p2[bp->level]*npx*init_block_x)*
                       (p2[bp->level]*npy*init_block_y)))/
                  (p2[bp->level]*npx*init_block_x);
         pos[0] = nl%(p2[bp->level]*npx*init_block_x);
         for (c = 0; c < 6; c++) {
            dir = c/2;
            i1 = j1 = k1 = 0;
            if      (c == 0) i1 = -1;
            else if (c == 1) i1 =  1;
            else if (c == 2) j1 = -1;
            else if (c == 3) j1 =  1;
            else if (c == 4) k1 = -1;
            else if (c == 5) k1 =  1;
            c1 = (c/2)*2 + (c+1)%2;
            fcase = (c%2)*10;
            if (bp->nei_level[c] == (bp->level-1)) {
               if (bp->nei[c][0][0] < 0) {
                  proc = -1 - bp->nei[c][0][0];
                  i = pos[mul[dir][1]]%2;
                  j = pos[mul[dir][0]]%2;
                  if (proc == my_pe) {
                     number = (((pos[2]/2+k1)*p2[bp->level-1]*npy*init_block_y)+
                                (pos[1]/2+j1))*p2[bp->level-1]*npx*init_block_x+
                              pos[0]/2 + i1 + block_start[bp->level-1];
                     n1 = find_sorted_list(number, (bp->level-1));
                     bp->nei[c][0][0] = n1;
                     blocks[n1].nei[c1][i][j] = n;
                  } else
                     add_comm_list(dir, n, proc, fcase+2+2*i+j,
                                 (bp->cen[mul[dir][1]]*mesh_size[mul[dir][0]] +
                                  bp->cen[mul[dir][0]]),
                                 (bp->cen[mul[dir][2]] +
                                  (2*(c%2)-1)*p2[num_refine- bp->level]));
               }
            } else if (bp->nei_level[c] == bp->level) {
               if (bp->nei[c][0][0] < 0) {
                  proc = -1 - bp->nei[c][0][0];
                  if (proc == my_pe) {
                     number = (((pos[2]+k1)*p2[bp->level]*npy*init_block_y) +
                                (pos[1]+j1))*p2[bp->level]*npx*init_block_x +
                              pos[0] + i1 + block_start[bp->level];
                     n1 = find_sorted_list(number, bp->level);
                     bp->nei[c][0][0] = n1;
                     blocks[n1].nei[c1][0][0] = n;
                  } else
                     add_comm_list(dir, n, proc, fcase+f,
                                 (bp->cen[mul[dir][1]]*mesh_size[mul[dir][0]] +
                                  bp->cen[mul[dir][0]]),
                                 (bp->cen[mul[dir][2]] +
                                  (2*(c%2)-1)*p2[num_refine- bp->level]));
               }
            } else if (bp->nei_level[c] == (bp->level+1)) {
               offset = p2[num_refine - bp->level - 1];
               off[0] = off[1] = off[2] = 0;
               for (k = fcase+6, i = 0; i < 2; i++)
                  for (j = 0; j < 2; j++, k++)
                     if (bp->nei[c][i][j] < 0) {
                        off[mul[dir][0]] = j;
                        off[mul[dir][1]] = i;
                        proc = -1 - bp->nei[c][i][j];
                        if (proc == my_pe) {
                           number = (((2*(pos[2]+k1)-(k1-1)/2+off[2])*
                                          p2[bp->level+1]*npy*init_block_y) +
                                      (2*(pos[1]+j1)-(j1-1)/2+off[1]))*
                                          p2[bp->level+1]*npx*init_block_x +
                                    2*(pos[0]+i1)-(i1-1)/2 + off[0] +
                                    block_start[bp->level+1];
                           n1 = find_sorted_list(number, (bp->level+1));
                           bp->nei[c][i][j] = n1;
                           blocks[n1].nei[c1][0][0] = n;
                        } else
                           add_comm_list(dir, n, proc, k,
                  ((bp->cen[mul[dir][1]]+(2*i-1)*offset)*mesh_size[mul[dir][0]]
                                   + bp->cen[mul[dir][0]]+(2*j-1)*offset),
                                 (bp->cen[mul[dir][2]] +
                                  (2*(c%2)-1)*p2[num_refine- bp->level]));
                     }
            }
         }
         // connect to parent if moved
         if (bp->parent != -1)
            if (bp->parent_node == my_pe) {
               for (p = 0; p < max_active_parent; p++)
                  if (parents[p].number == -2 - bp->parent) {
                     bp->parent = (num_sz) p;
                     parents[p].child[bp->child_number] = (num_sz) n;
                     parents[p].child_node[bp->child_number] = my_pe;
                     break;
                  }
            } else
               add_par_list(&par_b, (-2-bp->parent), (num_sz) n,
                            bp->child_number, bp->parent_node, 0);
      }
   }
}
