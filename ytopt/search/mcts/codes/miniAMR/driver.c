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
#include <math.h>
#include <mpi.h>

#include "block.h"
#include "comm.h"
#include "timer.h"
#include "proto.h"

// Main driver for program.
void driver(void)
{
   int ts, var, start, number, stage, comm_stage, calc_stage, done, in;
   double t1, t2, t3, t4;
   double sum, delta = 1.0, sim_time;
   block *bp;

   init();
   init_profile();
   counter_malloc_init = counter_malloc;
   size_malloc_init = size_malloc;

   t1 = timer();

   if (num_refine || uniform_refine) refine(0);
   t2 = timer();
   timer_refine_all += t2 - t1;

   if (plot_freq)
      plot(0);
   t3 = timer();
   timer_plot += t3 - t2;

   nb_min = nb_max = global_active;

   if (use_time) delta = calc_time_step();
   for (sim_time = 0.0, done = comm_stage =calc_stage=0, ts = 1; !done; ts++) {
      for (stage=0; stage < stages_per_ts; stage++,comm_stage++,calc_stage++) {
         total_blocks += global_active;
         if (global_active < nb_min)
            nb_min = global_active;
         if (global_active > nb_max)
            nb_max = global_active;
         for (start = 0; start < num_vars; start += comm_vars) {
            if (start+comm_vars > num_vars)
               number = num_vars - start;
            else
               number = comm_vars;
            t3 = timer();
            comm(start, number, comm_stage);
            t4 = timer();
            timer_comm_all += t4 - t3;
            for (var = start; var < (start+number); var ++) {
               stencil_driver(var, calc_stage);
//#pragma omp parallel for private (bp)
//               for (in = 0; in < sorted_index[num_refine+1]; in++) {
//                  bp = &blocks[sorted_list[in].n];
//                  stencil_driver(bp, var, calc_stage);
//               }
               t3 = timer();
//if (!my_pe) printf("var %d took %lf\n", var, (t3 - t4));
               timer_calc_all += t3 - t4;
               if (checksum_freq && !(stage%checksum_freq)) {
                  sum = check_sum(var);
                  if (report_diffusion && !my_pe)
                     printf("%d var %d sum %lf old %lf diff %lf %lf tol %lf\n",
                            ts, var, sum, grid_sum[var], (sum - grid_sum[var]),
                            (fabs(sum - grid_sum[var])/grid_sum[var]), tol);
                  if (stencil || var == 0)
                     if (fabs(sum - grid_sum[var])/grid_sum[var] > tol) {
                        if (!my_pe)
                           printf("Time step %d sum %lf (old %lf) variable %d difference too large\n", ts, sum, grid_sum[var], var);
                           return;
                     }
                  grid_sum[var] = sum;
               }
               t4 = timer();
               timer_cs_all += t4 - t3;
            }
         }
      }

      if (num_refine && !uniform_refine) {
         move(delta);
         if (!(ts%refine_freq))
            refine(ts);
      }
      t2 = timer();
      timer_refine_all += t2 - t4;

      t3 = timer();
      if (plot_freq && !(ts%plot_freq))
         plot(ts);
      timer_plot += timer() - t3;

      if (use_time) {
         delta = calc_time_step();
         if (sim_time >= end_time)
            done = 1;
         else
            sim_time += delta;
      } else
         if (ts >= num_tsteps)
            done = 1;
   }

   end_time = sim_time;
   num_tsteps = ts - 1;
   timer_all = timer() - t1;
}

// Calculate time step increment.  If an object intersects the unit cube
// then look at the rate of movement and size increase compared to the size
// of the smallest cell.  This does not exclude all cases where an object
// would not influence refinement.
double calc_time_step(void)
{
   int o, dir, done;
   double delta, tmp, inv_cell_size[3];
   object *op;

   inv_cell_size[0] = (double) (mesh_size[0]*x_block_size);
   inv_cell_size[1] = (double) (mesh_size[1]*y_block_size);
   inv_cell_size[2] = (double) (mesh_size[2]*z_block_size);
   delta = 0.0;
   for (o = 0; o < num_objects; o++) {
      op = &objects[o];
      if (op->size[0] < 0.0 || op->size[1] < 0.0 || op->size[2] < 0)
         break;
      for (done = dir = 0; dir < 3; dir++)
         if (op->cen[dir] < 0.0) {
            if (op->cen[dir] + op->size[dir] < 0.0)
               done = 1;
         } else if (op->cen[dir] > 1.0) {
            if (op->cen[dir] - op->size[dir] > 1.0)
               done = 1;
         }
      if (done)
         break;
      for (done = dir = 0; dir < 3; dir++) {
         tmp = (fabs(op->move[dir]) + fabs(op->inc[dir]))*inv_cell_size[dir];
         if (tmp > delta)
            delta = tmp;
      }
   }

   if (delta > 0.0)
      delta = 1.0/delta;
   else
      delta = 1.0;

   return delta;
}
