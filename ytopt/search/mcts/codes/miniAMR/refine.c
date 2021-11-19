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
#include <mpi.h>

#include "block.h"
#include "comm.h"
#include "proto.h"
#include "timer.h"

// This file contains routines that determine which blocks are going to
// be refined and which are going to be coarsened.
void refine(int ts)
{
   int i, j, n, in, min_b, max_b, sum_b, num_refine_step, num_split,
       nm_r, nm_c, nm_t;
   double ratio, tp, tm, tu, tp1, tm1, tu1, t1, t2, t3, t4, t5;
   block *bp;

   nrs++;
   nm_r = nm_c = nm_t = 0;
   t4 = tp = tm = tu = tp1 = tm1 = tu1 = 0.0;
   t1 = timer();

   t2 = timer();
   MPI_Allreduce(local_num_blocks, num_blocks, (num_refine+1),
                 MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
   timer_refine_sy += timer() - t2;
   t4 += timer() - t2;

   if (ts)
      num_refine_step = block_change;
   else
      num_refine_step = num_refine;

   for (i = 0; i < num_refine_step; i++) {
      for (j = num_refine; j >= 0; j--)
         if (num_blocks[j]) {
            cur_max_level = j;
            break;
      }
      reset_all();
      if (uniform_refine) {
         for (in = 0; in < sorted_index[num_refine+1]; in++) {
            bp = &blocks[sorted_list[in].n];
            if (bp->level < num_refine)
               bp->refine = 1;
            else
               bp->refine = 0;
         }
      } else {
         t2 = timer();
         check_objects();
         timer_refine_co += timer() - t2;
         t4 += timer() - t2;
      }

      t2 = timer();
      comm_refine();
      comm_parent();
      comm_parent_reverse();
      timer_refine_c1 += timer() - t2;
      t4 += timer() - t2;

      t2 = timer();
      num_split = refine_level();
      t5 = timer();
      timer_refine_mr += t5 - t2;
      t4 += t5 - t2;

      t2 = timer();
      sum_b = num_active + 7*num_split + 1;
      MPI_Allreduce(&sum_b, &max_b, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      sum_b = num_parents + num_split;
      MPI_Allreduce(&sum_b, &min_b, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (max_b > ((int) (0.75*((double) max_num_blocks))) ||
          min_b >= (max_num_parents-1)) {
         redistribute_blocks(&tp1, &tm1, &tu1, &t3, &nm_r, num_split);
         timer_rs_ca += t3;
         nrrs++;
      }
      t5 = timer();
      timer_rs_all += t5 - t2;
      t4 += t5 - t2;

      t2 = timer();
      split_blocks();
      t5 = timer();
      timer_refine_sb += t5 - t2;
      t4 += t5 - t2;

      t2 = timer();
      reset_neighbors();
      comm_parent();
      comm_parent_reverse();
      comm_refine();
      timer_refine_c2 += timer() - t2;
      t4 += timer() - t2;

      t2 = timer();
      redistribute_blocks(&tp, &tm, &tu, &t3, &nm_c, 0);
      t3 = timer() - t3;
      consolidate_blocks();
      t5 = timer();
      timer_cb_cb += t5 - t3;
      timer_cb_all += t5 - t2;
      t4 += t5 - t2;
      check_buff_size();

      t2 = timer();
      MPI_Allreduce(local_num_blocks, num_blocks, (num_refine+1),
                    MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
      timer_refine_sy += timer() - t2;
      t4 += timer() - t2;
      if (lb_opt == 2) {
         t2 = timer();
         if (num_active > local_max_b)
            local_max_b = num_active;
         MPI_Allreduce(&num_active, &min_b, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD);
         MPI_Allreduce(&num_active, &max_b, 1, MPI_INT, MPI_MAX,
                       MPI_COMM_WORLD);
         MPI_Allreduce(&num_active, &sum_b, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD);
         MPI_Allreduce(&local_max_b, &global_max_b, 1, MPI_INT, MPI_MAX,
                       MPI_COMM_WORLD);
         t4 += timer() - t2;
         ratio = ((double) (max_b - min_b)*num_pes)/((double) sum_b);
         if (!uniform_refine && max_b > (min_b + 1) &&
             ratio > ((double) inbalance/100.0)) {
            nlbs++;
            t2 = timer();
            load_balance();
            t5 = timer();
            timer_lb_all += t5 - t2;
            t4 += t5 - t2;

            t2 = timer();
            MPI_Allreduce(local_num_blocks, num_blocks, (num_refine+1),
                          MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
            timer_refine_sy += timer() - t2;
            t4 += timer() - t2;
         }
      }
   }
   timer_rs_pa += tp1;
   timer_rs_mv += tm1;
   timer_rs_un += tu1;
   timer_cb_pa += tp;
   timer_cb_mv += tm;
   timer_cb_un += tu;

   t2 = timer();
   if (num_active > local_max_b)
      local_max_b = num_active;
   MPI_Allreduce(&num_active, &min_b, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(&num_active, &max_b, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(&num_active, &sum_b, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_max_b, &global_max_b, 1, MPI_INT, MPI_MAX,
                 MPI_COMM_WORLD);
   i = nm_r + nm_c + nm_t;
   MPI_Allreduce(&i, &num_split, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   for (j = 0; j <= num_refine; j++) {
      if (!j)
         global_active = num_blocks[0];
      else
         global_active += num_blocks[j];
      if (!my_pe && report_perf & 8)
         printf("Number of blocks at level %d at timestep %d is %ld\n",
                j, ts, num_blocks[j]);
   }
   if (!my_pe && report_perf & 8)
      printf("Total number of blocks at timestep %d is %ld\n\n", ts,
             global_active);
   timer_refine_sy += timer() - t2;
   t4 += timer() - t2;

   if (lb_opt) {
      ratio = ((double) (max_b - min_b)*num_pes)/((double) sum_b);
      if (!uniform_refine &&
          (max_b > (min_b + 1) && ratio > ((double) inbalance/100.0))) {
         nlbs++;
         t2 = timer();
         load_balance();
         t5 = timer();
         timer_lb_all += t5 - t2;
         t4 += t5 - t2;

         t2 = timer();
         MPI_Allreduce(local_num_blocks, num_blocks, (num_refine+1),
                       MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
         timer_refine_sy += timer() - t2;
         t4 += timer() - t2;
      }
   }
   num_moved_rs += nm_r;
   num_moved_coarsen += nm_c;
   num_moved_reduce += nm_t;
   check_buff_size();
   t5 = timer();
   timer_refine_cc += t5 - t1 - t4;
}

int refine_level(void)
{
   int level, nei, i, j, b, c, c1, change, lchange, unrefine, sib, p, in;
   block *bp, *bp1;
   parent *pp;

   /* block states:
    * 1 block should be refined
    * -1 block could be unrefined
    * 0 block at level 0 and can not be unrefined or
    *         at max level and can not be refined
    */

// get list of neighbor blocks (indirect links in from blocks)

   for (level = cur_max_level; level >= 0; level--) {
      /* check for blocks at this level that will refine
         their neighbors at this level can not unrefine
         their neighbors at a lower level must refine
      */
      do {
         lchange = 0;
// think about this - isolate by level #pragma omp parallel for private(in, bp, i, p, pp, b, nei) reduction(+: lchange)
         for (in = sorted_index[level]; in < sorted_index[level+1]; in++) {
            bp = &blocks[sorted_list[in].n];
            if (bp->level == level) {
               if (bp->refine == 1) {
                  if (bp->parent != -1 && bp->parent_node == my_pe) {
                     pp = &parents[bp->parent];
                     if (pp->refine == -1)
                        pp->refine = 0;
                     for (b = 0; b < 8; b++)
                        if (pp->child_node[b] == my_pe && pp->child[b] >= 0)
                           if (blocks[pp->child[b]].refine == -1) {
                              blocks[pp->child[b]].refine = 0;
                              lchange++;
                           }
                  }
                  for (i = 0; i < 6; i++)
                     /* neighbors in level above taken care of already */
                     /* neighbors in this level can not unrefine */
                     if (bp->nei_level[i] == level)
                        if ((nei = bp->nei[i][0][0]) >= 0) { /* on core */
                           if (blocks[nei].refine == -1) {
                              blocks[nei].refine = 0;
                              lchange++;
                              if ((p = blocks[nei].parent) != -1 &&
                                    blocks[nei].parent_node == my_pe) {
                                 if ((pp = &parents[p])->refine == -1)
                                    pp->refine = 0;
                                 for (b = 0; b < 8; b++)
                                    if (pp->child_node[b] == my_pe &&
                                        pp->child[b] >= 0)
                                       if (blocks[pp->child[b]].refine == -1) {
                                          blocks[pp->child[b]].refine = 0;
                                          lchange++;
                                       }
                              }
                           }
                        } else { /* off core */
                           if (bp->nei_refine[i] == -1) {
                              bp->nei_refine[i] = 0;
                              lchange++;
                           }
                        }
                     /* neighbors in level below must refine */
                     else if (bp->nei_level[i] == level-1)
                        if ((nei = bp->nei[i][0][0]) >= 0) {
                           if (blocks[nei].refine != 1) {
                              blocks[nei].refine = 1;
                              lchange++;
                           }
                        } else
                           if (bp->nei_refine[i] != 1) {
                              bp->nei_refine[i] = 1;
                              lchange++;
                           }
               } else if (bp->refine == -1) {
                  // check if block can be unrefined
                  for (i = 0; i < 6; i++)
                     if (bp->nei_level[i] == level+1) {
                        bp->refine = 0;
                        lchange++;
                        if ((p = bp->parent) != -1 &&
                            bp->parent_node == my_pe) {
                           if ((pp = &parents[p])->refine == -1)
                              pp->refine = 0;
                           for (b = 0; b < 8; b++)
                              if (pp->child_node[b] == my_pe &&
                                  pp->child[b] >= 0 &&
                                  blocks[pp->child[b]].refine == -1)
                                 blocks[pp->child[b]].refine = 0;
                        }
                     }
               }
            }
         }

         MPI_Allreduce(&lchange, &change, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD);

         // Communicate these changes if any made
         if (change) {
            comm_reverse_refine();
            // Communicate any changes of which blocks will refine
            comm_refine();
            comm_parent_reverse();
            comm_parent();
         }
      } while (change);

      /* Check for blocks at this level that will remain at this level
         their neighbors at a lower level can not unrefine
      */
      do {
         lchange = 0;
         for (in = sorted_index[level]; in < sorted_index[level+1]; in++) {
            bp = &blocks[sorted_list[in].n];
            if (bp->level == level && bp->refine == 0)
               for (c = 0; c < 6; c++)
                  if (bp->nei_level[c] == level-1) {
                     if ((nei = bp->nei[c][0][0]) >= 0) {
                        if (blocks[nei].refine == -1) {
                           blocks[nei].refine = 0;
                           lchange++;
                           if ((p = blocks[nei].parent) != -1 &&
                                 blocks[nei].parent_node == my_pe)
                              if ((pp = &parents[p])->refine == -1) {
                                 pp->refine = 0;
                                 for (b = 0; b < 8; b++)
                                    if (pp->child_node[b] == my_pe &&
                                        pp->child[b] >= 0 &&
                                        blocks[pp->child[b]].refine == -1)
                                       blocks[pp->child[b]].refine = 0;
                              }
                        }
                     } else
                        if (bp->nei_refine[c] == -1) {
                           bp->nei_refine[c] = 0;
                           lchange++;
                        }
                  } else if (bp->nei_level[c] == level) {
                     if ((nei = bp->nei[c][0][0]) >= 0)
                        blocks[nei].nei_refine[(c/2)*2+(c+1)%2] = 0;
                  } else if (bp->nei_level[c] == level+1) {
                     c1 = (c/2)*2 + (c+1)%2;
                     for (i = 0; i < 2; i++)
                        for (j = 0; j < 2; j++)
                           if ((nei = bp->nei[c][i][j]) >= 0)
                              blocks[nei].nei_refine[c1] = 0;
                  }
         }

         MPI_Allreduce(&lchange, &change, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD);

         // Communicate these changes of any parent that can not refine
         if (change) {
            comm_reverse_refine();
            comm_refine();
            comm_parent();
            // Communicate any changes of which blocks can not unrefine
            comm_parent_reverse();
         }
      } while (change);
   }

   for (i = in = 0; in < sorted_index[num_refine+1]; in++)
     if (blocks[sorted_list[in].n].refine == 1)
        i++;

   return(i);
}

// Reset the neighbor lists on blocks so that matching them against objects
// can set those which can be refined.
void reset_all(void)
{
   int c, in, n;
   block *bp;
   parent *pp;

   for (in = 0; in < sorted_index[num_refine+1]; in++) {
      bp = &blocks[sorted_list[in].n];
      bp->refine = -1;
      for (c = 0; c < 6; c++)
         if (bp->nei_level[c] >= 0)
            bp->nei_refine[c] = -1;
   }

   for (n = 0; n < max_active_parent; n++)
      if ((pp = &parents[n])->number >= 0) {
         pp->refine = -1;
         for (c = 0; c < 8; c++)
            if (pp->child[c] < 0)
               pp->refine = 0;
         if (pp->refine == 0)
            for (c = 0; c < 8; c++)
               if (pp->child_node[c] == my_pe && pp->child[c] >= 0)
                  if (blocks[pp->child[c]].refine == -1)
                     blocks[pp->child[c]].refine = 0;
      }
}

// Reset neighbor lists on blocks since those lists are incorrect on blocks
// that have just been split.
void reset_neighbors(void)
{
   int c, in;
   block *bp;

   for (in = 0; in < sorted_index[num_refine+1]; in++) {
      bp = &blocks[sorted_list[in].n];
      for (c = 0; c < 6; c++)
         if (bp->nei_level[c] >= 0 && bp->nei[c][0][0] < 0)
            bp->nei_refine[c] = -1;
   }
}

// Redistribute blocks so that the number of blocks will not exceed the
// number of available blocks on processors during refinement and coarsening
void redistribute_blocks(double *tp, double *tm, double *tu, double *time,
                         int *num_moved, int num_split)
{
   int i, in, m, n, p, need, excess, my_excess, target, rem, sum, my_active,
       space[num_pes], use[num_pes];
   double t1;
   block *bp;
   parent *pp;

   t1 = timer();

   for (i = 0; i < num_pes; i++)
      bin[i] = 0;
   bin[my_pe] = num_split;

   MPI_Allreduce(bin, gbin, num_pes, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   for (sum = i = 0; i < num_pes; i++) {
      from[i] = 0;
      sum += gbin[i];
   }

   for (i = 0; i < num_pes; i++)
      bin[i] = 0;
   bin[my_pe] = max_num_parents - num_parents - 1 - num_split;

   if (bin[my_pe] < 0)
      bin[my_pe] = 0;

   MPI_Allreduce(bin, space, num_pes, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   for (in = 0; in < sorted_index[num_refine+1]; in++)
      blocks[sorted_list[in].n].new_proc = -1;

   target = sum/num_pes;
   rem = sum - target*num_pes;

   for (excess = i = 0; i < num_pes; i++) {
      need = target + (i < rem);
      if (need > space[i]) {
         use[i] = space[i];
         excess += need - space[i];
      } else
         use[i] = need;
   }
   // loop while there is blocks to be moved and progress is being made
   // if there are blocks to move and no progress, the code will fail later
   while (excess && sum)
      for (sum = i = 0; i < num_pes && excess; i++)
         if (space[i] > use[i]) {
            use[i]++;
            excess--;
            sum++;
         }

   m = in = 0;
   if (num_split > use[my_pe]) {  // have blocks to give
      my_excess = num_split - use[my_pe];
      my_active = num_active - my_excess + 7*use[my_pe] + 1;
      (*num_moved) += my_excess;
      for (excess = i = 0; i < my_pe; i++)
         if (gbin[i] > use[i])
            excess += gbin[i] - use[i];
      for (need = i = 0; i < num_pes && my_excess; i++)
         if (gbin[i] < use[i]) {
            need += use[i] - gbin[i];
            if (need > excess)
               for ( ; in < sorted_index[num_refine+1] && need > excess &&
                       my_excess; in++)
                  if ((bp = &blocks[sorted_list[in].n])->refine == 1) {
                     from[i]++;
                     bp->new_proc = i;
                     need--;
                     my_excess--;
                     m++;
                  }
         }
   } else  // getting blocks
      my_active = num_active + 7*use[my_pe] + 1;

   for (in = 0; in < sorted_index[num_refine+1]; in++) {
      bp = &blocks[sorted_list[in].n];
      if (bp->refine == -1 && bp->parent_node != my_pe) {
         bp->new_proc = bp->parent_node;
         from[bp->parent_node]++;
         my_active--;
         m++;
         (*num_moved)++;
      }
   }
   for (p = 0; p < max_active_parent; p++)
      if ((pp = &parents[p])->number >= 0 && pp->refine == -1)
         for (i = 0; i < 8; i++)
            if (pp->child_node[i] != my_pe)
               my_active++;
            else
               blocks[pp->child[i]].new_proc = my_pe;

   MPI_Allreduce(&m, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   if (n) {
      MPI_Allreduce(&my_active, &sum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

      if (sum > ((int) (0.75*((double) max_num_blocks)))) {
         // even up the expected number of blocks per processor
         for (i = 0; i < num_pes; i++)
            bin[i] = 0;
         bin[my_pe] = my_active;

         MPI_Allreduce(bin, gbin, num_pes, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD);

         for (sum = i = 0; i < num_pes; i++)
            sum += gbin[i];

         target = sum/num_pes;
         rem = sum - target*num_pes;

         in = sorted_index[num_refine+1] - 1;  // don't want to move big blocks
         if (my_active > (target + (my_pe < rem))) {  // have blocks to give
            my_excess = my_active - (target + (my_pe < rem));
            (*num_moved) += my_excess;
            for (excess = i = 0; i < my_pe; i++)
               if (gbin[i] > (target + (i < rem)))
                  excess += gbin[i] - (target + (i < rem));
            for (need = i = 0; i < num_pes && my_excess; i++)
               if (gbin[i] < (target + (i < rem))) {
                  need += (target + (i < rem)) - gbin[i];
                  if (need > excess)
                     for ( ; in >= 0 && need > excess && my_excess; in--)
                        if ((bp = &blocks[sorted_list[in].n])->new_proc == -1){
                           from[i]++;
                           bp->new_proc = i;
                           need--;
                           my_excess--;
                           m++;
                        }
               }
         }
      } else
         in = sorted_index[num_refine+1] - 1;

      // keep the rest of the blocks on this processor
      for ( ; in >= 0; in--)
         if (blocks[sorted_list[in].n].new_proc == -1)
            blocks[sorted_list[in].n].new_proc = my_pe;

      *time = timer() - t1;

      MPI_Alltoall(from, 1, MPI_INT, to, 1, MPI_INT, MPI_COMM_WORLD);
      move_blocks(tp, tm, tu);
   } else
      *time = timer() - t1;
}
