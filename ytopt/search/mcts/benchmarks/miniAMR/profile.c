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
#include <omp.h>

#include "block.h"
#include "comm.h"
#include "proto.h"
#include "timer.h"

// Profiling output.
void profile(void)
{
   int i, ompt;
   double total_gflops, gflops_rank, total_fp_ops, total_fp_adds,
          total_fp_divs;
   object *op;
   char *version = "1.4? w/OpenMP";
   FILE *fp;

#ifdef _OPENMP
   ompt = omp_get_max_threads();
#else
   ompt = 1;
#endif
   calculate_results();
   total_fp_ops = average[128] + average[129] + average[130];
   total_gflops = total_fp_ops/(average[38]*1024.0*1024.0*1024.0);
   gflops_rank = total_gflops/((double) num_pes);

   if (!my_pe) {
      if (report_perf & 1) {
         fp = fopen("results.yaml", "w");
         fprintf(fp, "code: miniAMR\n");
         fprintf(fp, "version: %s\n", version);
         fprintf(fp, "ranks: %d\n", num_pes);
         fprintf(fp, "npx: %d\n", npx);
         fprintf(fp, "npy: %d\n", npy);
         fprintf(fp, "npz: %d\n", npz);
         fprintf(fp, "init_block_x: %d\n", init_block_x);
         fprintf(fp, "init_block_y: %d\n", init_block_y);
         fprintf(fp, "init_block_z: %d\n", init_block_z);
         fprintf(fp, "x_block_size: %d\n", x_block_size);
         fprintf(fp, "y_block_size: %d\n", y_block_size);
         fprintf(fp, "z_block_size: %d\n", z_block_size);
         fprintf(fp, "threads: %d\n", ompt);
         fprintf(fp, "reorder: %d\n", reorder);
         fprintf(fp, "permute: %d\n", permute);
         fprintf(fp, "max_blocks_allowed: %d\n", max_num_blocks);
         fprintf(fp, "code: %d\n", code);
         fprintf(fp, "num_refine: %d\n", num_refine);
         fprintf(fp, "block_change: %d\n", block_change);
         fprintf(fp, "refine_ghost: %d\n", refine_ghost);
         fprintf(fp, "uniform_refine: %d\n", uniform_refine);
         fprintf(fp, "num_objects: %d\n", num_objects);
         for (i = 0; i < num_objects; i++) {
            op = &objects[i];
            fprintf(fp, "obj%dtype: %d\n", i, op->type);
            fprintf(fp, "obj%dbounce: %d\n", i, op->bounce);
            fprintf(fp, "obj%dcenter_x: %lf\n", i, op->cen[0]);
            fprintf(fp, "obj%dcenter_y: %lf\n", i, op->cen[1]);
            fprintf(fp, "obj%dcenter_z: %lf\n", i, op->cen[2]);
            fprintf(fp, "obj%dmove_x: %lf\n", i, op->move[0]);
            fprintf(fp, "obj%dmove_y: %lf\n", i, op->move[1]);
            fprintf(fp, "obj%dmove_z: %lf\n", i, op->move[2]);
            fprintf(fp, "obj%dsize_x: %lf\n", i, op->size[0]);
            fprintf(fp, "obj%dsize_y: %lf\n", i, op->size[1]);
            fprintf(fp, "obj%dsize_z: %lf\n", i, op->size[2]);
            fprintf(fp, "obj%dinc_x: %lf\n", i, op->inc[0]);
            fprintf(fp, "obj%dinc_y: %lf\n", i, op->inc[1]);
            fprintf(fp, "obj%dinc_z: %lf\n", i, op->inc[2]);
         }
         fprintf(fp, "num_tsteps: %d\n", num_tsteps);
         fprintf(fp, "stages_per_timestep: %d\n", stages_per_ts);
         fprintf(fp, "checksum_freq: %d\n", checksum_freq);
         fprintf(fp, "refine_freq: %d\n", refine_freq);
         fprintf(fp, "lb_opt: %d\n", lb_opt);
         fprintf(fp, "inbalance: %d\n", inbalance);
         fprintf(fp, "plot_freq: %d\n", plot_freq);
         fprintf(fp, "num_vars: %d\n", num_vars);
         fprintf(fp, "stencil: %d\n", stencil);
         fprintf(fp, "comm_vars: %d\n", comm_vars);
         fprintf(fp, "error_tol: %d\n", error_tol);

         fprintf(fp, "total_time_ave: %lf\n", average[0]);
         fprintf(fp, "total_time_min: %lf\n", minimum[0]);
         fprintf(fp, "total_time_max: %lf\n", maximum[0]);
         fprintf(fp, "memory_used_ave: %lf\n", average[111]);
         fprintf(fp, "memory_used_min: %lf\n", minimum[111]);
         fprintf(fp, "memory_used_max: %lf\n", maximum[111]);
         fprintf(fp, "compute_time_ave: %lf\n", average[38]);
         fprintf(fp, "compute_time_min: %lf\n", minimum[38]);
         fprintf(fp, "compute_time_max: %lf\n", maximum[38]);
         fprintf(fp, "total_gflops: %lf\n", total_gflops);
         fprintf(fp, "ave_gflops: %lf\n", gflops_rank);

         fprintf(fp, "total_comm_ave: %lf\n", average[37]);
         fprintf(fp, "total_comm_min: %lf\n", minimum[37]);
         fprintf(fp, "total_comm_max: %lf\n", maximum[37]);
         fprintf(fp, "   total_post_recv_ave: %lf\n", average[2]);
         fprintf(fp, "   total_post_recv_min: %lf\n", minimum[2]);
         fprintf(fp, "   total_post_recv_max: %lf\n", maximum[2]);
         fprintf(fp, "   total_pack_faces_ave: %lf\n", average[3]);
         fprintf(fp, "   total_pack_faces_min: %lf\n", minimum[3]);
         fprintf(fp, "   total_pack_faces_max: %lf\n", maximum[3]);
         fprintf(fp, "   total_send_mess_ave: %lf\n", average[4]);
         fprintf(fp, "   total_send_mess_min: %lf\n", minimum[4]);
         fprintf(fp, "   total_send_mess_max: %lf\n", maximum[4]);
         fprintf(fp, "   total_exch_same_ave: %lf\n", average[5]);
         fprintf(fp, "   total_exch_same_min: %lf\n", minimum[5]);
         fprintf(fp, "   total_exch_same_max: %lf\n", maximum[5]);
         fprintf(fp, "   total_exch_diff_ave: %lf\n", average[6]);
         fprintf(fp, "   total_exch_diff_min: %lf\n", minimum[6]);
         fprintf(fp, "   total_exch_diff_max: %lf\n", maximum[6]);
         fprintf(fp, "   total_apply_bc_ave: %lf\n", average[7]);
         fprintf(fp, "   total_apply_bc_min: %lf\n", minimum[7]);
         fprintf(fp, "   total_apply_bc_max: %lf\n", maximum[7]);
         fprintf(fp, "   total_wait_time_ave: %lf\n", average[8]);
         fprintf(fp, "   total_wait_time_min: %lf\n", minimum[8]);
         fprintf(fp, "   total_wait_time_max: %lf\n", maximum[8]);
         fprintf(fp, "   total_unpack_faces_ave: %lf\n", average[9]);
         fprintf(fp, "   total_unpack_faces_min: %lf\n", minimum[9]);
         fprintf(fp, "   total_unpack_faces_max: %lf\n", maximum[9]);
         fprintf(fp, "   total_mess_recv_ave: %lf\n", average[70]);
         fprintf(fp, "   total_mess_recv_min: %lf\n", minimum[70]);
         fprintf(fp, "   total_mess_recv_max: %lf\n", maximum[70]);
         fprintf(fp, "   total_byte_recv_ave: %lf\n", average[68]);
         fprintf(fp, "   total_byte_recv_min: %lf\n", minimum[68]);
         fprintf(fp, "   total_byte_recv_max: %lf\n", maximum[68]);
         fprintf(fp, "   total_face_recv_ave: %lf\n", average[72]);
         fprintf(fp, "   total_face_recv_min: %lf\n", minimum[72]);
         fprintf(fp, "   total_face_recv_max: %lf\n", maximum[72]);
         fprintf(fp, "   total_mess_send_ave: %lf\n", average[71]);
         fprintf(fp, "   total_mess_send_min: %lf\n", minimum[71]);
         fprintf(fp, "   total_mess_send_max: %lf\n", maximum[71]);
         fprintf(fp, "   total_byte_send_ave: %lf\n", average[69]);
         fprintf(fp, "   total_byte_send_min: %lf\n", minimum[69]);
         fprintf(fp, "   total_byte_send_max: %lf\n", maximum[69]);
         fprintf(fp, "   total_face_send_ave: %lf\n", average[73]);
         fprintf(fp, "   total_face_send_min: %lf\n", minimum[73]);
         fprintf(fp, "   total_face_send_max: %lf\n", maximum[73]);
         fprintf(fp, "   total_face_exch_same_ave: %lf\n", average[75]);
         fprintf(fp, "   total_face_exch_same_min: %lf\n", minimum[75]);
         fprintf(fp, "   total_face_exch_same_max: %lf\n", maximum[75]);
         fprintf(fp, "   total_face_exch_diff_ave: %lf\n", average[76]);
         fprintf(fp, "   total_face_exch_diff_min: %lf\n", minimum[76]);
         fprintf(fp, "   total_face_exch_diff_max: %lf\n", maximum[76]);
         fprintf(fp, "   total_face_bc_apply_ave: %lf\n", average[74]);
         fprintf(fp, "   total_face_bc_apply_min: %lf\n", minimum[74]);
         fprintf(fp, "   total_face_bc_apply_max: %lf\n", maximum[74]);

         fprintf(fp, "   x_comm_ave: %lf\n", average[10]);
         fprintf(fp, "   x_comm_min: %lf\n", minimum[10]);
         fprintf(fp, "   x_comm_max: %lf\n", maximum[10]);
         fprintf(fp, "      x_post_recv_ave: %lf\n", average[11]);
         fprintf(fp, "      x_post_recv_min: %lf\n", minimum[11]);
         fprintf(fp, "      x_post_recv_max: %lf\n", maximum[11]);
         fprintf(fp, "      x_pack_faces_ave: %lf\n", average[12]);
         fprintf(fp, "      x_pack_faces_min: %lf\n", minimum[12]);
         fprintf(fp, "      x_pack_faces_max: %lf\n", maximum[12]);
         fprintf(fp, "      x_send_mess_ave: %lf\n", average[13]);
         fprintf(fp, "      x_send_mess_min: %lf\n", minimum[13]);
         fprintf(fp, "      x_send_mess_max: %lf\n", maximum[13]);
         fprintf(fp, "      x_exch_same_ave: %lf\n", average[14]);
         fprintf(fp, "      x_exch_same_min: %lf\n", minimum[14]);
         fprintf(fp, "      x_exch_same_max: %lf\n", maximum[14]);
         fprintf(fp, "      x_exch_diff_ave: %lf\n", average[15]);
         fprintf(fp, "      x_exch_diff_min: %lf\n", minimum[15]);
         fprintf(fp, "      x_exch_diff_max: %lf\n", maximum[15]);
         fprintf(fp, "      x_apply_bc_ave: %lf\n", average[16]);
         fprintf(fp, "      x_apply_bc_min: %lf\n", minimum[16]);
         fprintf(fp, "      x_apply_bc_max: %lf\n", maximum[16]);
         fprintf(fp, "      x_wait_time_ave: %lf\n", average[17]);
         fprintf(fp, "      x_wait_time_min: %lf\n", minimum[17]);
         fprintf(fp, "      x_wait_time_max: %lf\n", maximum[17]);
         fprintf(fp, "      x_unpack_faces_ave: %lf\n", average[18]);
         fprintf(fp, "      x_unpack_faces_min: %lf\n", minimum[18]);
         fprintf(fp, "      x_unpack_faces_max: %lf\n", maximum[18]);
         fprintf(fp, "      x_mess_recv_ave: %lf\n", average[79]);
         fprintf(fp, "      x_mess_recv_min: %lf\n", minimum[79]);
         fprintf(fp, "      x_mess_recv_max: %lf\n", maximum[79]);
         fprintf(fp, "      x_byte_recv_ave: %lf\n", average[77]);
         fprintf(fp, "      x_byte_recv_min: %lf\n", minimum[77]);
         fprintf(fp, "      x_byte_recv_max: %lf\n", maximum[77]);
         fprintf(fp, "      x_face_recv_ave: %lf\n", average[81]);
         fprintf(fp, "      x_face_recv_min: %lf\n", minimum[81]);
         fprintf(fp, "      x_face_recv_max: %lf\n", maximum[81]);
         fprintf(fp, "      x_mess_send_ave: %lf\n", average[80]);
         fprintf(fp, "      x_mess_send_min: %lf\n", minimum[80]);
         fprintf(fp, "      x_mess_send_max: %lf\n", maximum[80]);
         fprintf(fp, "      x_byte_send_ave: %lf\n", average[78]);
         fprintf(fp, "      x_byte_send_min: %lf\n", minimum[78]);
         fprintf(fp, "      x_byte_send_max: %lf\n", maximum[78]);
         fprintf(fp, "      x_face_send_ave: %lf\n", average[82]);
         fprintf(fp, "      x_face_send_min: %lf\n", minimum[82]);
         fprintf(fp, "      x_face_send_max: %lf\n", maximum[82]);
         fprintf(fp, "      x_face_exch_same_ave: %lf\n", average[84]);
         fprintf(fp, "      x_face_exch_same_min: %lf\n", minimum[84]);
         fprintf(fp, "      x_face_exch_same_max: %lf\n", maximum[84]);
         fprintf(fp, "      x_face_exch_diff_ave: %lf\n", average[85]);
         fprintf(fp, "      x_face_exch_diff_min: %lf\n", minimum[85]);
         fprintf(fp, "      x_face_exch_diff_max: %lf\n", maximum[85]);
         fprintf(fp, "      x_face_bc_apply_ave: %lf\n", average[83]);
         fprintf(fp, "      x_face_bc_apply_min: %lf\n", minimum[83]);
         fprintf(fp, "      x_face_bc_apply_max: %lf\n", maximum[83]);

         fprintf(fp, "   y_comm_ave: %lf\n", average[19]);
         fprintf(fp, "   y_comm_min: %lf\n", minimum[19]);
         fprintf(fp, "   y_comm_max: %lf\n", maximum[19]);
         fprintf(fp, "      y_post_recv_ave: %lf\n", average[20]);
         fprintf(fp, "      y_post_recv_min: %lf\n", minimum[20]);
         fprintf(fp, "      y_post_recv_max: %lf\n", maximum[20]);
         fprintf(fp, "      y_pack_faces_ave: %lf\n", average[21]);
         fprintf(fp, "      y_pack_faces_min: %lf\n", minimum[21]);
         fprintf(fp, "      y_pack_faces_max: %lf\n", maximum[21]);
         fprintf(fp, "      y_send_mess_ave: %lf\n", average[22]);
         fprintf(fp, "      y_send_mess_min: %lf\n", minimum[22]);
         fprintf(fp, "      y_send_mess_max: %lf\n", maximum[22]);
         fprintf(fp, "      y_exch_same_ave: %lf\n", average[23]);
         fprintf(fp, "      y_exch_same_min: %lf\n", minimum[23]);
         fprintf(fp, "      y_exch_same_max: %lf\n", maximum[23]);
         fprintf(fp, "      y_exch_diff_ave: %lf\n", average[24]);
         fprintf(fp, "      y_exch_diff_min: %lf\n", minimum[24]);
         fprintf(fp, "      y_exch_diff_max: %lf\n", maximum[24]);
         fprintf(fp, "      y_apply_bc_ave: %lf\n", average[25]);
         fprintf(fp, "      y_apply_bc_min: %lf\n", minimum[25]);
         fprintf(fp, "      y_apply_bc_max: %lf\n", maximum[25]);
         fprintf(fp, "      y_wait_time_ave: %lf\n", average[26]);
         fprintf(fp, "      y_wait_time_min: %lf\n", minimum[26]);
         fprintf(fp, "      y_wait_time_max: %lf\n", maximum[26]);
         fprintf(fp, "      y_unpack_faces_ave: %lf\n", average[27]);
         fprintf(fp, "      y_unpack_faces_min: %lf\n", minimum[27]);
         fprintf(fp, "      y_unpack_faces_max: %lf\n", maximum[27]);
         fprintf(fp, "      y_mess_recv_ave: %lf\n", average[88]);
         fprintf(fp, "      y_mess_recv_min: %lf\n", minimum[88]);
         fprintf(fp, "      y_mess_recv_max: %lf\n", maximum[88]);
         fprintf(fp, "      y_byte_recv_ave: %lf\n", average[86]);
         fprintf(fp, "      y_byte_recv_min: %lf\n", minimum[86]);
         fprintf(fp, "      y_byte_recv_max: %lf\n", maximum[86]);
         fprintf(fp, "      y_face_recv_ave: %lf\n", average[90]);
         fprintf(fp, "      y_face_recv_min: %lf\n", minimum[90]);
         fprintf(fp, "      y_face_recv_max: %lf\n", maximum[90]);
         fprintf(fp, "      y_mess_send_ave: %lf\n", average[89]);
         fprintf(fp, "      y_mess_send_min: %lf\n", minimum[89]);
         fprintf(fp, "      y_mess_send_max: %lf\n", maximum[89]);
         fprintf(fp, "      y_byte_send_ave: %lf\n", average[87]);
         fprintf(fp, "      y_byte_send_min: %lf\n", minimum[87]);
         fprintf(fp, "      y_byte_send_max: %lf\n", maximum[87]);
         fprintf(fp, "      y_face_send_ave: %lf\n", average[91]);
         fprintf(fp, "      y_face_send_min: %lf\n", minimum[91]);
         fprintf(fp, "      y_face_send_max: %lf\n", maximum[91]);
         fprintf(fp, "      y_face_exch_same_ave: %lf\n", average[93]);
         fprintf(fp, "      y_face_exch_same_min: %lf\n", minimum[93]);
         fprintf(fp, "      y_face_exch_same_max: %lf\n", maximum[93]);
         fprintf(fp, "      y_face_exch_diff_ave: %lf\n", average[94]);
         fprintf(fp, "      y_face_exch_diff_min: %lf\n", minimum[94]);
         fprintf(fp, "      y_face_exch_diff_max: %lf\n", maximum[94]);
         fprintf(fp, "      y_face_bc_apply_ave: %lf\n", average[92]);
         fprintf(fp, "      y_face_bc_apply_min: %lf\n", minimum[92]);
         fprintf(fp, "      y_face_bc_apply_max: %lf\n", maximum[92]);

         fprintf(fp, "   z_comm_ave: %lf\n", average[28]);
         fprintf(fp, "   z_comm_min: %lf\n", minimum[28]);
         fprintf(fp, "   z_comm_max: %lf\n", maximum[28]);
         fprintf(fp, "      z_post_recv_ave: %lf\n", average[29]);
         fprintf(fp, "      z_post_recv_min: %lf\n", minimum[29]);
         fprintf(fp, "      z_post_recv_max: %lf\n", maximum[29]);
         fprintf(fp, "      z_pack_faces_ave: %lf\n", average[30]);
         fprintf(fp, "      z_pack_faces_min: %lf\n", minimum[30]);
         fprintf(fp, "      z_pack_faces_max: %lf\n", maximum[30]);
         fprintf(fp, "      z_send_mess_ave: %lf\n", average[31]);
         fprintf(fp, "      z_send_mess_min: %lf\n", minimum[31]);
         fprintf(fp, "      z_send_mess_max: %lf\n", maximum[31]);
         fprintf(fp, "      z_exch_same_ave: %lf\n", average[32]);
         fprintf(fp, "      z_exch_same_min: %lf\n", minimum[32]);
         fprintf(fp, "      z_exch_same_max: %lf\n", maximum[32]);
         fprintf(fp, "      z_exch_diff_ave: %lf\n", average[33]);
         fprintf(fp, "      z_exch_diff_min: %lf\n", minimum[33]);
         fprintf(fp, "      z_exch_diff_max: %lf\n", maximum[33]);
         fprintf(fp, "      z_apply_bc_ave: %lf\n", average[34]);
         fprintf(fp, "      z_apply_bc_min: %lf\n", minimum[34]);
         fprintf(fp, "      z_apply_bc_max: %lf\n", maximum[34]);
         fprintf(fp, "      z_wait_time_ave: %lf\n", average[35]);
         fprintf(fp, "      z_wait_time_min: %lf\n", minimum[35]);
         fprintf(fp, "      z_wait_time_max: %lf\n", maximum[35]);
         fprintf(fp, "      z_unpack_faces_ave: %lf\n", average[36]);
         fprintf(fp, "      z_unpack_faces_min: %lf\n", minimum[36]);
         fprintf(fp, "      z_unpack_faces_max: %lf\n", maximum[36]);
         fprintf(fp, "      z_mess_recv_ave: %lf\n", average[97]);
         fprintf(fp, "      z_mess_recv_min: %lf\n", minimum[97]);
         fprintf(fp, "      z_mess_recv_max: %lf\n", maximum[97]);
         fprintf(fp, "      z_byte_recv_ave: %lf\n", average[95]);
         fprintf(fp, "      z_byte_recv_min: %lf\n", minimum[95]);
         fprintf(fp, "      z_byte_recv_max: %lf\n", maximum[95]);
         fprintf(fp, "      z_face_recv_ave: %lf\n", average[99]);
         fprintf(fp, "      z_face_recv_min: %lf\n", minimum[99]);
         fprintf(fp, "      z_face_recv_max: %lf\n", maximum[99]);
         fprintf(fp, "      z_mess_send_ave: %lf\n", average[98]);
         fprintf(fp, "      z_mess_send_min: %lf\n", minimum[98]);
         fprintf(fp, "      z_mess_send_max: %lf\n", maximum[98]);
         fprintf(fp, "      z_byte_send_ave: %lf\n", average[96]);
         fprintf(fp, "      z_byte_send_min: %lf\n", minimum[96]);
         fprintf(fp, "      z_byte_send_max: %lf\n", maximum[96]);
         fprintf(fp, "      z_face_send_ave: %lf\n", average[100]);
         fprintf(fp, "      z_face_send_min: %lf\n", minimum[100]);
         fprintf(fp, "      z_face_send_max: %lf\n", maximum[100]);
         fprintf(fp, "      z_face_exch_same_ave: %lf\n", average[102]);
         fprintf(fp, "      z_face_exch_same_min: %lf\n", minimum[102]);
         fprintf(fp, "      z_face_exch_same_max: %lf\n", maximum[102]);
         fprintf(fp, "      z_face_exch_diff_ave: %lf\n", average[103]);
         fprintf(fp, "      z_face_exch_diff_min: %lf\n", minimum[103]);
         fprintf(fp, "      z_face_exch_diff_max: %lf\n", maximum[103]);
         fprintf(fp, "      z_face_bc_apply_ave: %lf\n", average[101]);
         fprintf(fp, "      z_face_bc_apply_min: %lf\n", minimum[101]);
         fprintf(fp, "      z_face_bc_apply_max: %lf\n", maximum[101]);

         fprintf(fp, "gridsum_time_ave: %lf\n", average[39]);
         fprintf(fp, "gridsum_time_min: %lf\n", minimum[39]);
         fprintf(fp, "gridsum_time_max: %lf\n", maximum[39]);
         fprintf(fp, "   gridsum_reduce_ave: %lf\n", average[40]);
         fprintf(fp, "   gridsum_reduce_min: %lf\n", minimum[40]);
         fprintf(fp, "   gridsum_reduce_max: %lf\n", maximum[40]);
         fprintf(fp, "   gridsum_calc_ave: %lf\n", average[41]);
         fprintf(fp, "   gridsum_calc_min: %lf\n", minimum[41]);
         fprintf(fp, "   gridsum_calc_max: %lf\n", maximum[41]);

         fprintf(fp, "refine_time_ave: %lf\n", average[42]);
         fprintf(fp, "refine_time_min: %lf\n", minimum[42]);
         fprintf(fp, "refine_time_max: %lf\n", maximum[42]);
         fprintf(fp, "   total_blocks_ts_ave: %lf\n",
                ((double) total_blocks)/((double) (num_tsteps*stages_per_ts)));
         fprintf(fp, "   total_blocks_ts_min: %ld\n", (long long) nb_min);
         fprintf(fp, "   total_blocks_ts_max: %ld\n", (long long) nb_max);
         fprintf(fp, "   blocks_split_ave: %lf\n", average[104]);
         fprintf(fp, "   blocks_split_min: %lf\n", minimum[104]);
         fprintf(fp, "   blocks_split_max: %lf\n", maximum[104]);
         fprintf(fp, "   blocks_reformed_ave: %lf\n", average[105]);
         fprintf(fp, "   blocks_reformed_min: %lf\n", minimum[105]);
         fprintf(fp, "   blocks_reformed_max: %lf\n", maximum[105]);
         fprintf(fp, "   blocks_moved_tot_ave: %lf\n", average[106]);
         fprintf(fp, "   blocks_moved_tot_min: %lf\n", minimum[106]);
         fprintf(fp, "   blocks_moved_tot_max: %lf\n", maximum[106]);
         fprintf(fp, "   blocks_moved_lb_ave: %lf\n", average[107]);
         fprintf(fp, "   blocks_moved_lb_min: %lf\n", minimum[107]);
         fprintf(fp, "   blocks_moved_lb_max: %lf\n", maximum[107]);
         fprintf(fp, "   blocks_moved_redist_ave: %lf\n", average[122]);
         fprintf(fp, "   blocks_moved_redist_min: %lf\n", minimum[122]);
         fprintf(fp, "   blocks_moved_redist_max: %lf\n", maximum[122]);
         fprintf(fp, "   blocks_moved_coarsen_ave: %lf\n", average[109]);
         fprintf(fp, "   blocks_moved_coarsen_min: %lf\n", minimum[109]);
         fprintf(fp, "   blocks_moved_coarsen_max: %lf\n", maximum[109]);
         fprintf(fp, "   time_compare_obj_ave: %lf\n", average[43]);
         fprintf(fp, "   time_compare_obj_min: %lf\n", minimum[43]);
         fprintf(fp, "   time_compare_obj_max: %lf\n", maximum[43]);
         fprintf(fp, "   time_mark_refine_ave: %lf\n", average[44]);
         fprintf(fp, "   time_mark_refine_min: %lf\n", minimum[44]);
         fprintf(fp, "   time_mark_refine_max: %lf\n", maximum[44]);
         fprintf(fp, "   time_comm_block1_ave: %lf\n", average[119]);
         fprintf(fp, "   time_comm_block1_min: %lf\n", minimum[119]);
         fprintf(fp, "   time_comm_block1_max: %lf\n", maximum[119]);
         fprintf(fp, "   time_split_block_ave: %lf\n", average[46]);
         fprintf(fp, "   time_split_block_min: %lf\n", minimum[46]);
         fprintf(fp, "   time_split_block_max: %lf\n", maximum[46]);
         fprintf(fp, "   time_comm_block2_ave: %lf\n", average[120]);
         fprintf(fp, "   time_comm_block2_min: %lf\n", minimum[120]);
         fprintf(fp, "   time_comm_block2_max: %lf\n", maximum[120]);
         fprintf(fp, "   time_sync_ave: %lf\n", average[121]);
         fprintf(fp, "   time_sync_min: %lf\n", minimum[121]);
         fprintf(fp, "   time_sync_max: %lf\n", maximum[121]);
         fprintf(fp, "   time_misc_ave: %lf\n", average[45]);
         fprintf(fp, "   time_misc_min: %lf\n", minimum[45]);
         fprintf(fp, "   time_misc_max: %lf\n", maximum[45]);
         fprintf(fp, "   time_total_coarsen_ave: %lf\n", average[47]);
         fprintf(fp, "   time_total_coarsen_min: %lf\n", minimum[47]);
         fprintf(fp, "   time_total_coarsen_max: %lf\n", maximum[47]);
         fprintf(fp, "      time_coarsen_ave: %lf\n", average[48]);
         fprintf(fp, "      time_coarsen_min: %lf\n", minimum[48]);
         fprintf(fp, "      time_coarsen_max: %lf\n", maximum[48]);
         fprintf(fp, "      time_coarsen_pack_ave: %lf\n", average[49]);
         fprintf(fp, "      time_coarsen_pack_min: %lf\n", minimum[49]);
         fprintf(fp, "      time_coarsen_pack_max: %lf\n", maximum[49]);
         fprintf(fp, "      time_coarsen_move_ave: %lf\n", average[50]);
         fprintf(fp, "      time_coarsen_move_min: %lf\n", minimum[50]);
         fprintf(fp, "      time_coarsen_move_max: %lf\n", maximum[50]);
         fprintf(fp, "      time_coarsen_unpack_ave: %lf\n", average[51]);
         fprintf(fp, "      time_coarsen_unpack_min: %lf\n", minimum[51]);
         fprintf(fp, "      time_coarsen_unpack_max: %lf\n", maximum[51]);
         fprintf(fp, "   time_total_redist_ave: %lf\n", average[123]);
         fprintf(fp, "   time_total_redist_min: %lf\n", minimum[123]);
         fprintf(fp, "   time_total_redist_max: %lf\n", maximum[123]);
         fprintf(fp, "      time_redist_choose_ave: %lf\n", average[124]);
         fprintf(fp, "      time_redist_choose_min: %lf\n", minimum[124]);
         fprintf(fp, "      time_redist_choose_max: %lf\n", maximum[124]);
         fprintf(fp, "      time_redist_pack_ave: %lf\n", average[125]);
         fprintf(fp, "      time_redist_pack_min: %lf\n", minimum[125]);
         fprintf(fp, "      time_redist_pack_max: %lf\n", maximum[125]);
         fprintf(fp, "      time_redist_move_ave: %lf\n", average[126]);
         fprintf(fp, "      time_redist_move_min: %lf\n", minimum[126]);
         fprintf(fp, "      time_redist_move_max: %lf\n", maximum[126]);
         fprintf(fp, "      time_redist_unpack_ave: %lf\n", average[127]);
         fprintf(fp, "      time_redist_unpack_min: %lf\n", minimum[127]);
         fprintf(fp, "      time_redist_unpack_max: %lf\n", maximum[127]);
         fprintf(fp, "   time_total_load_bal_ave: %lf\n", average[62]);
         fprintf(fp, "   time_total_load_bal_min: %lf\n", minimum[62]);
         fprintf(fp, "   time_total_load_bal_max: %lf\n", maximum[62]);
         fprintf(fp, "      time_load_bal_sort_ave: %lf\n", average[63]);
         fprintf(fp, "      time_load_bal_sort_min: %lf\n", minimum[63]);
         fprintf(fp, "      time_load_bal_sort_max: %lf\n", maximum[63]);
         fprintf(fp, "      time_lb_move_dots_ave: %lf\n", average[117]);
         fprintf(fp, "      time_lb_move_dots_min: %lf\n", minimum[117]);
         fprintf(fp, "      time_lb_move_dots_max: %lf\n", maximum[117]);
         fprintf(fp, "      time_lb_move_blocks_ave: %lf\n", average[118]);
         fprintf(fp, "      time_lb_move_blocks_min: %lf\n", minimum[118]);
         fprintf(fp, "      time_lb_move_blocks_max: %lf\n", maximum[118]);
         fprintf(fp, "         time_lb_mb_pack_ave: %lf\n", average[64]);
         fprintf(fp, "         time_lb_mb_pack_min: %lf\n", minimum[64]);
         fprintf(fp, "         time_lb_mb_pack_max: %lf\n", maximum[64]);
         fprintf(fp, "         time_lb_mb_move_ave: %lf\n", average[65]);
         fprintf(fp, "         time_lb_mb_move_min: %lf\n", minimum[65]);
         fprintf(fp, "         time_lb_mb_move_max: %lf\n", maximum[65]);
         fprintf(fp, "         time_lb_mb_unpack_ave: %lf\n", average[66]);
         fprintf(fp, "         time_lb_mb_unpack_min: %lf\n", minimum[66]);
         fprintf(fp, "         time_lb_mb_unpack_max: %lf\n", maximum[66]);
         fprintf(fp, "         time_lb_mb_misc_ave: %lf\n", average[116]);
         fprintf(fp, "         time_lb_mb_misc_min: %lf\n", minimum[116]);
         fprintf(fp, "         time_lb_mb_misc_max: %lf\n", maximum[116]);

         fprintf(fp, "plot_time_ave: %lf\n", average[67]);
         fprintf(fp, "plot_time_min: %lf\n", minimum[67]);
         fprintf(fp, "plot_time_max: %lf\n", maximum[67]);

         fclose(fp);
      }

      if (report_perf & 2) {
         fp = fopen("results.txt", "w");

         fprintf(fp, "\n ================ Start report ===================\n\n");
         fprintf(fp, "          Mantevo miniAMR\n");
         fprintf(fp, "          version %s\n\n", version);

         fprintf(fp, "Run on %d ranks arranged in a %d x %d x %d grid\n", num_pes,
                npx, npy, npz);
         fprintf(fp, "Threads per rank %d\n", ompt);
         fprintf(fp, "initial blocks per rank %d x %d x %d\n", init_block_x,
                init_block_y, init_block_z);
         fprintf(fp, "block size %d x %d x %d\n", x_block_size, y_block_size,
                z_block_size);
         if (reorder)
            fprintf(fp, "Initial ranks arranged by RCB across machine\n\n");
         else
            fprintf(fp, "Initial ranks arranged as a grid across machine\n\n");
         if (permute)
            fprintf(fp, "Order of exchanges permuted\n");
         fprintf(fp, "Maximum number of blocks per rank is %d\n",
                 max_num_blocks);
         if (code)
            fprintf(fp, "Code set to code %d\n", code);
         fprintf(fp, "Number of levels of refinement is %d\n", num_refine);
         fprintf(fp, "Blocks can change by %d levels per refinement step\n",
            block_change);
         if (refine_ghost)
            fprintf(fp, "Ghost cells will be used determine is block is refined\n");
         if (uniform_refine)
            fprintf(fp, "\nBlocks will be uniformly refined\n");
         else {
            fprintf(fp, "\nBlocks will be refined by %d objects\n\n", num_objects);
            for (i = 0; i < num_objects; i++) {
               op = &objects[i];
               if (op->type == 0)
                  fprintf(fp, "Object %d is the surface of a rectangle\n", i);
               else if (op->type == 1)
                  fprintf(fp, "Object %d is the volume of a rectangle\n", i);
               else if (op->type == 2)
                  fprintf(fp, "Object %d is the surface of a spheroid\n", i);
               else if (op->type == 3)
                  fprintf(fp, "Object %d is the volume of a spheroid\n", i);
               else if (op->type == 4)
                  fprintf(fp, "Object %d is the surface of x+ hemispheroid\n", i);
               else if (op->type == 5)
                  fprintf(fp, "Object %d is the volume of x+ hemispheroid\n", i);
               else if (op->type == 6)
                  fprintf(fp, "Object %d is the surface of x- hemispheroid\n", i);
               else if (op->type == 7)
                  fprintf(fp, "Object %d is the volume of x- hemispheroid\n", i);
               else if (op->type == 8)
                  fprintf(fp, "Object %d is the surface of y+ hemispheroid\n", i);
               else if (op->type == 9)
                  fprintf(fp, "Object %d is the volume of y+ hemispheroid\n", i);
               else if (op->type == 10)
                  fprintf(fp, "Object %d is the surface of y- hemispheroid\n", i);
               else if (op->type == 11)
                  fprintf(fp, "Object %d is the volume of y- hemispheroid\n", i);
               else if (op->type == 12)
                  fprintf(fp, "Object %d is the surface of z+ hemispheroid\n", i);
               else if (op->type == 13)
                  fprintf(fp, "Object %d is the volume of z+ hemispheroid\n", i);
               else if (op->type == 14)
                  fprintf(fp, "Object %d is the surface of z- hemispheroid\n", i);
               else if (op->type == 15)
                  fprintf(fp, "Object %d is the volume of z- hemispheroid\n", i);
               else if (op->type == 20)
                  fprintf(fp, "Object %d is the surface of x axis cylinder\n", i);
               else if (op->type == 21)
                  fprintf(fp, "Object %d is the volune of x axis cylinder\n", i);
               else if (op->type == 22)
                  fprintf(fp, "Object %d is the surface of y axis cylinder\n", i);
               else if (op->type == 23)
                  fprintf(fp, "Object %d is the volune of y axis cylinder\n", i);
               else if (op->type == 24)
                  fprintf(fp, "Object %d is the surface of z axis cylinder\n", i);
               else if (op->type == 25)
                  fprintf(fp, "Object %d is the volune of z axis cylinder\n", i);
               if (op->bounce == 0)
                  fprintf(fp, "Oject may leave mesh\n");
               else
                  fprintf(fp, "Oject center will bounce off of walls\n");
               fprintf(fp, "Center starting at %lf %lf %lf\n",
                      op->orig_cen[0], op->orig_cen[1], op->orig_cen[2]);
               fprintf(fp, "Center end at %lf %lf %lf\n",
                      op->cen[0], op->cen[1], op->cen[2]);
               fprintf(fp, "Moving at %lf %lf %lf per timestep\n",
                      op->orig_move[0], op->orig_move[1], op->orig_move[2]);
               fprintf(fp, "   Rate relative to smallest cell size %lf %lf %lf\n",
                      op->orig_move[0]*((double) (mesh_size[0]*x_block_size)),
                      op->orig_move[1]*((double) (mesh_size[1]*y_block_size)),
                      op->orig_move[2]*((double) (mesh_size[2]*z_block_size)));
               fprintf(fp, "Initial size %lf %lf %lf\n",
                      op->orig_size[0], op->orig_size[1], op->orig_size[2]);
               fprintf(fp, "Final size %lf %lf %lf\n",
                      op->size[0], op->size[1], op->size[2]);
               fprintf(fp, "Size increasing %lf %lf %lf per timestep\n",
                      op->inc[0], op->inc[1], op->inc[2]);
               fprintf(fp, "   Rate relative to smallest cell size %lf %lf %lf\n\n",
                      op->inc[0]*((double) (mesh_size[0]*x_block_size)),
                      op->inc[1]*((double) (mesh_size[1]*y_block_size)),
                      op->inc[2]*((double) (mesh_size[2]*z_block_size)));
            }
         }
         if (use_time)
            fprintf(fp, "\nTime %lf in %d timesteps\n", end_time, num_tsteps);
         else
            fprintf(fp, "\nNumber of timesteps is %d\n", num_tsteps);
         fprintf(fp, "Communicaion/computation stages per timestep is %d\n",
                stages_per_ts);
         fprintf(fp, "Communication will be performed with nonblocking sends\n");
         fprintf(fp, "Will perform checksums every %d stages\n", checksum_freq);
         fprintf(fp, "Will refine every %d timesteps\n", refine_freq);
         if (lb_opt == 0)
            fprintf(fp, "Load balance will not be performed\n");
         else
            fprintf(fp, "Load balance when inbalanced by %d%\n", inbalance);
         if (lb_opt == 2)
            fprintf(fp, "Load balance at each phase of refinement step\n");
         if (plot_freq)
            fprintf(fp, "Will plot results every %d timesteps\n", plot_freq);
         else
            fprintf(fp, "Will not plot results\n");
         if (stencil)
            fprintf(fp, "Calculate on %d variables with %d point stencil\n",
                    num_vars, stencil);
         else
            fprintf(fp, "Calculate on %d variables with variable stencils\n",
                    num_vars);
         fprintf(fp, "Communicate %d variables at a time\n", comm_vars);
         fprintf(fp, "Error tolorance for variable sums is 10^(-%d)\n", error_tol);

         fprintf(fp, "\nTotal time for test: ave, std, min, max (sec): %lf %lf %lf %lf\n\n",
                average[0], stddev[0], minimum[0], maximum[0]);

         fprintf(fp, "\nNumber of malloc calls: ave, std, min, max (sec): %lf %lf %lf %lf\n",
                average[110], stddev[110], minimum[110], maximum[110]);
         fprintf(fp, "\nAmount malloced: ave, std, min, max: %lf %lf %lf %lf\n",
                average[111], stddev[111], minimum[111], maximum[111]);
         fprintf(fp, "\nMalloc calls in init: ave, std, min, max (sec): %lf %lf %lf %lf\n",
                average[112], stddev[112], minimum[112], maximum[112]);
         fprintf(fp, "\nAmount malloced in init: ave, std, min, max: %lf %lf %lf %lf\n",
                average[113], stddev[113], minimum[113], maximum[113]);
         fprintf(fp, "\nMalloc calls in timestepping: ave, std, min, max (sec): %lf %lf %lf %lf\n",
                average[114], stddev[114], minimum[114], maximum[114]);
         fprintf(fp, "\nAmount malloced in timestepping: ave, std, min, max: %lf %lf %lf %lf\n\n",
                average[115], stddev[115], minimum[115], maximum[115]);

         fprintf(fp, "---------------------------------------------\n");
         fprintf(fp, "          Computational Performance\n");
         fprintf(fp, "---------------------------------------------\n\n");
         fprintf(fp, "     Time: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[38], stddev[38], minimum[38], maximum[38]);
         fprintf(fp, "     total GFLOPS:             %lf\n", total_gflops);
         fprintf(fp, "     Average GFLOPS per rank:  %lf\n\n", gflops_rank);
         fprintf(fp, "     Total floating point ops: %lf\n\n", total_fp_ops);
         fprintf(fp, "        Adds:                  %lf\n", average[128]);
         fprintf(fp, "        Muls:                  %lf\n", average[129]);
         fprintf(fp, "        Divides:               %lf\n\n", average[130]);

         fprintf(fp, "---------------------------------------------\n");
         fprintf(fp, "           Interblock communication\n");
         fprintf(fp, "---------------------------------------------\n\n");
         fprintf(fp, "     Time: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[37], stddev[37], minimum[37], maximum[37]);
         for (i = 0; i < 4; i++) {
            if (i == 0)
               fprintf(fp, "\nTotal communication:\n\n");
            else if (i == 1)
               fprintf(fp, "\nX direction communication statistics:\n\n");
            else if (i == 2)
               fprintf(fp, "\nY direction communication statistics:\n\n");
            else
               fprintf(fp, "\nZ direction communication statistics:\n\n");
            fprintf(fp, "                              average    stddev  minimum  maximum\n");
            fprintf(fp, "     Total                  : %lf %lf %lf %lf\n",
                   average[1+9*i], stddev[1+9*i], minimum[1+9*i],
                   maximum[1+9*i]);
            fprintf(fp, "     Post IRecv             : %lf %lf %lf %lf\n",
                   average[2+9*i], stddev[2+9*i], minimum[2+9*i],
                   maximum[2+9*i]);
            fprintf(fp, "     Pack faces             : %lf %lf %lf %lf\n",
                   average[3+9*i], stddev[3+9*i], minimum[3+9*i],
                   maximum[3+9*i]);
            fprintf(fp, "     Send messages          : %lf %lf %lf %lf\n",
                   average[4+9*i], stddev[4+9*i], minimum[4+9*i],
                   maximum[4+9*i]);
            fprintf(fp, "     Exchange same level    : %lf %lf %lf %lf\n",
                   average[5+9*i], stddev[5+9*i], minimum[5+9*i],
                   maximum[5+9*i]);
            fprintf(fp, "     Exchange diff level    : %lf %lf %lf %lf\n",
                   average[6+9*i], stddev[6+9*i], minimum[6+9*i],
                   maximum[6+9*i]);
            fprintf(fp, "     Apply BC               : %lf %lf %lf %lf\n",
                   average[7+9*i], stddev[7+9*i], minimum[7+9*i],
                   maximum[7+9*i]);
            fprintf(fp, "     Wait time              : %lf %lf %lf %lf\n",
                   average[8+9*i], stddev[8+9*i], minimum[8+9*i],
                   maximum[8+9*i]);
            fprintf(fp, "     Unpack faces           : %lf %lf %lf %lf\n\n",
                   average[9+9*i], stddev[9+9*i], minimum[9+9*i],
                   maximum[9+9*i]);

            fprintf(fp, "     Messages received      : %lf %lf %lf %lf\n",
               average[70+9*i], stddev[70+9*i], minimum[70+9*i],
                   maximum[70+9*i]);
            fprintf(fp, "     Bytes received         : %lf %lf %lf %lf\n",
               average[68+9*i], stddev[68+9*i], minimum[68+9*i],
                   maximum[68+9*i]);
            fprintf(fp, "     Faces received         : %lf %lf %lf %lf\n",
               average[72+9*i], stddev[72+9*i], minimum[72+9*i],
                   maximum[72+9*i]);
            fprintf(fp, "     Messages sent          : %lf %lf %lf %lf\n",
               average[71+9*i], stddev[71+9*i], minimum[71+9*i],
                   maximum[71+9*i]);
            fprintf(fp, "     Bytes sent             : %lf %lf %lf %lf\n",
               average[69+9*i], stddev[69+9*i], minimum[69+9*i],
                   maximum[69+9*i]);
            fprintf(fp, "     Faces sent             : %lf %lf %lf %lf\n",
               average[73+9*i], stddev[73+9*i], minimum[73+9*i],
                   maximum[73+9*i]);
            fprintf(fp, "     Faces exchanged same   : %lf %lf %lf %lf\n",
               average[75+9*i], stddev[75+9*i], minimum[75+9*i],
                   maximum[75+9*i]);
            fprintf(fp, "     Faces exchanged diff   : %lf %lf %lf %lf\n",
               average[76+9*i], stddev[76+9*i], minimum[76+9*i],
                   maximum[76+9*i]);
            fprintf(fp, "     Faces with BC applied  : %lf %lf %lf %lf\n",
               average[74+9*i], stddev[74+9*i], minimum[74+9*i],
                   maximum[74+9*i]);
         }

         fprintf(fp, "\n---------------------------------------------\n");
         fprintf(fp, "             Gridsum performance\n");
         fprintf(fp, "---------------------------------------------\n\n");
         fprintf(fp, "     Time: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[39], stddev[39], minimum[39], maximum[39]);
         fprintf(fp, "        red : ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[40], stddev[40], minimum[40], maximum[40]);
         fprintf(fp, "        calc: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[41], stddev[41], minimum[41], maximum[41]);
         fprintf(fp, "     total number:             %d\n", total_red);
         fprintf(fp, "     number per timestep:      %d\n\n", num_vars);

         fprintf(fp, "---------------------------------------------\n");
         fprintf(fp, "               Mesh Refinement\n");
         fprintf(fp, "---------------------------------------------\n\n");
         fprintf(fp, "     Time: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[42], stddev[42], minimum[42], maximum[42]);
         fprintf(fp, "     Number of refinement steps: %d\n\n", nrs);
         fprintf(fp, "     Number of load balance steps: %d\n\n", nlbs);
         fprintf(fp, "     Number of redistributing steps: %d\n\n", nrrs);
         fprintf(fp, "     Total blocks           : %ld\n", total_blocks);
         fprintf(fp, "     Blocks/timestep ave, min, max : %lf %ld %ld\n",
                ((double) total_blocks)/((double) (num_tsteps*stages_per_ts)),
                (long long) nb_min, (long long) nb_max);
         fprintf(fp, "     Max blocks on a processor at any time: %d\n",
                global_max_b);
         fprintf(fp, "     total blocks split     : %lf\n",
                 average[104]*num_pes);
         fprintf(fp, "     total blocks reformed  : %lf\n\n",
                 average[105]*num_pes);
         fprintf(fp, "     total blocks moved     : %lf\n",
                 average[106]*num_pes);
         fprintf(fp, "     total moved load bal   : %lf\n",
                 average[107]*num_pes);
         fprintf(fp, "     total moved redistribut: %lf\n",
                 average[122]*num_pes);
         fprintf(fp, "     total moved coasening  : %lf\n",
                 average[109]*num_pes);
         fprintf(fp, "                              average    stddev  minimum  maximum\n");
         fprintf(fp, "     Per processor:\n");
         fprintf(fp, "     total blocks split     : %lf %lf %lf %lf\n",
                average[104], stddev[104], minimum[104], maximum[104]);
         fprintf(fp, "     total blocks reformed  : %lf %lf %lf %lf\n",
                average[105], stddev[105], minimum[105], maximum[105]);
         fprintf(fp, "     Total blocks moved     : %lf %lf %lf %lf\n",
                average[106], stddev[106], minimum[106], maximum[106]);
         fprintf(fp, "     Blocks moved load bal  : %lf %lf %lf %lf\n",
                average[107], stddev[107], minimum[107], maximum[107]);
         fprintf(fp, "     Blocks moved redistribu: %lf %lf %lf %lf\n",
                average[122], stddev[122], minimum[122], maximum[122]);
         fprintf(fp, "     Blocks moved coarsening: %lf %lf %lf %lf\n",
                average[109], stddev[109], minimum[109], maximum[109]);
         fprintf(fp, "     Time:\n");
         fprintf(fp, "        compare objects     : %lf %lf %lf %lf\n",
                average[43], stddev[43], minimum[43], maximum[43]);
         fprintf(fp, "        mark refine/coarsen : %lf %lf %lf %lf\n",
                average[44], stddev[44], minimum[44], maximum[44]);
         fprintf(fp, "        communicate block 1 : %lf %lf %lf %lf\n",
             average[119], stddev[119], minimum[119], maximum[119]);
         fprintf(fp, "        split blocks        : %lf %lf %lf %lf\n",
                average[46], stddev[46], minimum[46], maximum[46]);
         fprintf(fp, "        communicate block 2 : %lf %lf %lf %lf\n",
                average[120], stddev[120], minimum[120], maximum[120]);
         fprintf(fp, "        sync time           : %lf %lf %lf %lf\n",
                average[121], stddev[121], minimum[121], maximum[121]);
         fprintf(fp, "        misc time           : %lf %lf %lf %lf\n",
                average[45], stddev[45], minimum[45], maximum[45]);
         fprintf(fp, "        total coarsen blocks: %lf %lf %lf %lf\n",
                average[47], stddev[47], minimum[47], maximum[47]);
         fprintf(fp, "           coarsen blocks   : %lf %lf %lf %lf\n",
                average[48], stddev[48], minimum[48], maximum[48]);
         fprintf(fp, "           pack blocks      : %lf %lf %lf %lf\n",
                average[49], stddev[49], minimum[49], maximum[49]);
         fprintf(fp, "           move blocks      : %lf %lf %lf %lf\n",
                average[50], stddev[50], minimum[50], maximum[50]);
         fprintf(fp, "           unpack blocks    : %lf %lf %lf %lf\n",
                average[51], stddev[51], minimum[51], maximum[51]);
         fprintf(fp, "        total redistribute  : %lf %lf %lf %lf\n",
                average[123], stddev[123], minimum[123], maximum[123]);
         fprintf(fp, "           choose blocks    : %lf %lf %lf %lf\n",
                average[124], stddev[124], minimum[124], maximum[124]);
         fprintf(fp, "           pack blocks      : %lf %lf %lf %lf\n",
                average[125], stddev[125], minimum[125], maximum[125]);
         fprintf(fp, "           move blocks      : %lf %lf %lf %lf\n",
                average[126], stddev[126], minimum[126], maximum[126]);
         fprintf(fp, "           unpack blocks    : %lf %lf %lf %lf\n",
                average[127], stddev[127], minimum[127], maximum[127]);
         fprintf(fp, "        total load balance  : %lf %lf %lf %lf\n",
                average[62], stddev[62], minimum[62], maximum[62]);
         fprintf(fp, "           sort             : %lf %lf %lf %lf\n",
                average[63], stddev[63], minimum[63], maximum[63]);
         fprintf(fp, "           move dots back   : %lf %lf %lf %lf\n",
                average[117], stddev[117], minimum[117], maximum[117]);
         fprintf(fp, "           move blocks total: %lf %lf %lf %lf\n",
                average[118], stddev[118], minimum[118], maximum[118]);
         fprintf(fp, "              pack blocks   : %lf %lf %lf %lf\n",
                average[64], stddev[64], minimum[64], maximum[64]);
         fprintf(fp, "              move blocks   : %lf %lf %lf %lf\n",
                average[65], stddev[65], minimum[65], maximum[65]);
         fprintf(fp, "              unpack blocks : %lf %lf %lf %lf\n",
                average[66], stddev[66], minimum[66], maximum[66]);
         fprintf(fp, "              misc          : %lf %lf %lf %lf\n\n",
                average[116], stddev[116], minimum[116], maximum[116]);

         fprintf(fp, "---------------------------------------------\n");
         fprintf(fp, "                   Plot\n");
         fprintf(fp, "---------------------------------------------\n\n");
         fprintf(fp, "     Time: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[67], stddev[67], minimum[67], maximum[67]);
         fprintf(fp, "     Number of plot steps: %d\n", nps);
         fprintf(fp, "\n ================== End report ===================\n");

         fclose(fp);
      }

      if (report_perf & 4) {
         printf("\n ================ Start report ===================\n\n");
         printf("          Mantevo miniAMR\n");
         printf("          version %s\n\n", version);

         printf("Run on %d ranks arranged in a %d x %d x %d grid\n", num_pes,
                npx, npy, npz);
         printf("Threads per rank %d\n", ompt);
         printf("initial blocks per rank %d x %d x %d\n", init_block_x,
                init_block_y, init_block_z);
         printf("block size %d x %d x %d\n", x_block_size, y_block_size,
                z_block_size);
         if (reorder)
            printf("Initial ranks arranged by RCB across machine\n\n");
         else
            printf("Initial ranks arranged as a grid across machine\n\n");
         if (permute)
            printf("Order of exchanges permuted\n");
         printf("Maximum number of blocks per rank is %d\n", max_num_blocks);
         if (code)
            printf("Code set to code %d\n", code);
         printf("Number of levels of refinement is %d\n", num_refine);
         printf("Blocks can change by %d levels per refinement step\n",
            block_change);
         if (refine_ghost)
            printf("Ghost cells will be used determine is block is refined\n");
         if (uniform_refine)
            printf("\nBlocks will be uniformly refined\n");
         else {
            printf("\nBlocks will be refined by %d objects\n\n", num_objects);
            for (i = 0; i < num_objects; i++) {
               op = &objects[i];
               if (op->type == 0)
                  printf("Object %d is the surface of a rectangle\n", i);
               else if (op->type == 1)
                  printf("Object %d is the volume of a rectangle\n", i);
               else if (op->type == 2)
                  printf("Object %d is the surface of a spheroid\n", i);
               else if (op->type == 3)
                  printf("Object %d is the volume of a spheroid\n", i);
               else if (op->type == 4)
                  printf("Object %d is the surface of x+ hemispheroid\n", i);
               else if (op->type == 5)
                  printf("Object %d is the volume of x+ hemispheroid\n", i);
               else if (op->type == 6)
                  printf("Object %d is the surface of x- hemispheroid\n", i);
               else if (op->type == 7)
                  printf("Object %d is the volume of x- hemispheroid\n", i);
               else if (op->type == 8)
                  printf("Object %d is the surface of y+ hemispheroid\n", i);
               else if (op->type == 9)
                  printf("Object %d is the volume of y+ hemispheroid\n", i);
               else if (op->type == 10)
                  printf("Object %d is the surface of y- hemispheroid\n", i);
               else if (op->type == 11)
                  printf("Object %d is the volume of y- hemispheroid\n", i);
               else if (op->type == 12)
                  printf("Object %d is the surface of z+ hemispheroid\n", i);
               else if (op->type == 13)
                  printf("Object %d is the volume of z+ hemispheroid\n", i);
               else if (op->type == 14)
                  printf("Object %d is the surface of z- hemispheroid\n", i);
               else if (op->type == 15)
                  printf("Object %d is the volume of z- hemispheroid\n", i);
               else if (op->type == 20)
                  printf("Object %d is the surface of x axis cylinder\n", i);
               else if (op->type == 21)
                  printf("Object %d is the volune of x axis cylinder\n", i);
               else if (op->type == 22)
                  printf("Object %d is the surface of y axis cylinder\n", i);
               else if (op->type == 23)
                  printf("Object %d is the volune of y axis cylinder\n", i);
               else if (op->type == 24)
                  printf("Object %d is the surface of z axis cylinder\n", i);
               else if (op->type == 25)
                  printf("Object %d is the volune of z axis cylinder\n", i);
               if (op->bounce == 0)
                  printf("Oject may leave mesh\n");
               else
                  printf("Oject center will bounce off of walls\n");
               printf("Center starting at %lf %lf %lf\n",
                      op->orig_cen[0], op->orig_cen[1], op->orig_cen[2]);
               printf("Center end at %lf %lf %lf\n",
                      op->cen[0], op->cen[1], op->cen[2]);
               printf("Moving at %lf %lf %lf per timestep\n",
                      op->orig_move[0], op->orig_move[1], op->orig_move[2]);
               printf("   Rate relative to smallest cell size %lf %lf %lf\n",
                      op->orig_move[0]*((double) (mesh_size[0]*x_block_size)),
                      op->orig_move[1]*((double) (mesh_size[1]*y_block_size)),
                      op->orig_move[2]*((double) (mesh_size[2]*z_block_size)));
               printf("Initial size %lf %lf %lf\n",
                      op->orig_size[0], op->orig_size[1], op->orig_size[2]);
               printf("Final size %lf %lf %lf\n",
                      op->size[0], op->size[1], op->size[2]);
               printf("Size increasing %lf %lf %lf per timestep\n",
                      op->inc[0], op->inc[1], op->inc[2]);
               printf("   Rate relative to smallest cell size %lf %lf %lf\n\n",
                      op->inc[0]*((double) (mesh_size[0]*x_block_size)),
                      op->inc[1]*((double) (mesh_size[1]*y_block_size)),
                      op->inc[2]*((double) (mesh_size[2]*z_block_size)));
            }
         }
         if (use_time)
            printf("\nTime %lf in %d timesteps\n", end_time, num_tsteps);
         else
            printf("\nNumber of timesteps is %d\n", num_tsteps);
         printf("Communicaion/computation stages per timestep is %d\n",
                stages_per_ts);
         printf("Communication will be performed with nonblocking sends\n");
         printf("Will perform checksums every %d stages\n", checksum_freq);
         printf("Will refine every %d timesteps\n", refine_freq);
         if (lb_opt == 0)
            printf("Load balance will not be performed\n");
         else
            printf("Load balance when inbalanced by %d%\n", inbalance);
         if (lb_opt == 2)
            printf("Load balance at each phase of refinement step\n");
         if (plot_freq)
            printf("Will plot results every %d timesteps\n", plot_freq);
         else
            printf("Will not plot results\n");
         if (stencil)
            printf("Calculate on %d variables with %d point stencil\n",
                   num_vars, stencil);
         else
            printf("Calculate on %d variables with variable stencils\n",
                   num_vars);
         printf("Communicate %d variables at a time\n", comm_vars);
         printf("Error tolorance for variable sums is 10^(-%d)\n", error_tol);

         printf("\nTotal time for test: ave, std, min, max (sec): %lf %lf %lf %lf\n\n",
                average[0], stddev[0], minimum[0], maximum[0]);

         printf("\nNumber of malloc calls: ave, std, min, max (sec): %lf %lf %lf %lf\n",
                average[110], stddev[110], minimum[110], maximum[110]);
         printf("\nAmount malloced: ave, std, min, max: %lf %lf %lf %lf\n",
                average[111], stddev[111], minimum[111], maximum[111]);
         printf("\nMalloc calls in init: ave, std, min, max (sec): %lf %lf %lf %lf\n",
                average[112], stddev[112], minimum[112], maximum[112]);
         printf("\nAmount malloced in init: ave, std, min, max: %lf %lf %lf %lf\n",
                average[113], stddev[113], minimum[113], maximum[113]);
         printf("\nMalloc calls in timestepping: ave, std, min, max (sec): %lf %lf %lf %lf\n",
                average[114], stddev[114], minimum[114], maximum[114]);
         printf("\nAmount malloced in timestepping: ave, std, min, max: %lf %lf %lf %lf\n\n",
                average[115], stddev[115], minimum[115], maximum[115]);

         printf("---------------------------------------------\n");
         printf("          Computational Performance\n");
         printf("---------------------------------------------\n\n");
         printf("     Time: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[38], stddev[38], minimum[38], maximum[38]);
         printf("     total GFLOPS:             %lf\n", total_gflops);
         printf("     Average GFLOPS per rank:  %lf\n\n", gflops_rank);
         printf("     Total floating point ops: %lf\n\n", total_fp_ops);
         printf("        Adds:                  %lf\n", average[128]);
         printf("        Muls:                  %lf\n", average[129]);
         printf("        Divides:               %lf\n\n", average[130]);

         printf("---------------------------------------------\n");
         printf("           Interblock communication\n");
         printf("---------------------------------------------\n\n");
         printf("     Time: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[37], stddev[37], minimum[37], maximum[37]);
         for (i = 0; i < 4; i++) {
            if (i == 0)
               printf("\nTotal communication:\n\n");
            else if (i == 1)
               printf("\nX direction communication statistics:\n\n");
            else if (i == 2)
               printf("\nY direction communication statistics:\n\n");
            else
               printf("\nZ direction communication statistics:\n\n");
            printf("                              average    stddev  minimum  maximum\n");
            printf("     Total                  : %lf %lf %lf %lf\n",
                   average[1+9*i], stddev[1+9*i], minimum[1+9*i],
                   maximum[1+9*i]);
            printf("     Post IRecv             : %lf %lf %lf %lf\n",
                   average[2+9*i], stddev[2+9*i], minimum[2+9*i],
                   maximum[2+9*i]);
            printf("     Pack faces             : %lf %lf %lf %lf\n",
                   average[3+9*i], stddev[3+9*i], minimum[3+9*i],
                   maximum[3+9*i]);
            printf("     Send messages          : %lf %lf %lf %lf\n",
                   average[4+9*i], stddev[4+9*i], minimum[4+9*i],
                   maximum[4+9*i]);
            printf("     Exchange same level    : %lf %lf %lf %lf\n",
                   average[5+9*i], stddev[5+9*i], minimum[5+9*i],
                   maximum[5+9*i]);
            printf("     Exchange diff level    : %lf %lf %lf %lf\n",
                   average[6+9*i], stddev[6+9*i], minimum[6+9*i],
                   maximum[6+9*i]);
            printf("     Apply BC               : %lf %lf %lf %lf\n",
                   average[7+9*i], stddev[7+9*i], minimum[7+9*i],
                   maximum[7+9*i]);
            printf("     Wait time              : %lf %lf %lf %lf\n",
                   average[8+9*i], stddev[8+9*i], minimum[8+9*i],
                   maximum[8+9*i]);
            printf("     Unpack faces           : %lf %lf %lf %lf\n\n",
                   average[9+9*i], stddev[9+9*i], minimum[9+9*i],
                   maximum[9+9*i]);

            printf("     Messages received      : %lf %lf %lf %lf\n",
               average[70+9*i], stddev[70+9*i], minimum[70+9*i],
                   maximum[70+9*i]);
            printf("     Bytes received         : %lf %lf %lf %lf\n",
               average[68+9*i], stddev[68+9*i], minimum[68+9*i],
                   maximum[68+9*i]);
            printf("     Faces received         : %lf %lf %lf %lf\n",
               average[72+9*i], stddev[72+9*i], minimum[72+9*i],
                   maximum[72+9*i]);
            printf("     Messages sent          : %lf %lf %lf %lf\n",
               average[71+9*i], stddev[71+9*i], minimum[71+9*i],
                   maximum[71+9*i]);
            printf("     Bytes sent             : %lf %lf %lf %lf\n",
               average[69+9*i], stddev[69+9*i], minimum[69+9*i],
                   maximum[69+9*i]);
            printf("     Faces sent             : %lf %lf %lf %lf\n",
               average[73+9*i], stddev[73+9*i], minimum[73+9*i],
                   maximum[73+9*i]);
            printf("     Faces exchanged same   : %lf %lf %lf %lf\n",
               average[75+9*i], stddev[75+9*i], minimum[75+9*i],
                   maximum[75+9*i]);
            printf("     Faces exchanged diff   : %lf %lf %lf %lf\n",
               average[76+9*i], stddev[76+9*i], minimum[76+9*i],
                   maximum[76+9*i]);
            printf("     Faces with BC applied  : %lf %lf %lf %lf\n",
               average[74+9*i], stddev[74+9*i], minimum[74+9*i],
                   maximum[74+9*i]);
         }

         printf("\n---------------------------------------------\n");
         printf("             Gridsum performance\n");
         printf("---------------------------------------------\n\n");
         printf("     Time: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[39], stddev[39], minimum[39], maximum[39]);
         printf("        red : ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[40], stddev[40], minimum[40], maximum[40]);
         printf("        calc: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[41], stddev[41], minimum[41], maximum[41]);
         printf("     total number:             %d\n", total_red);
         printf("     number per timestep:      %d\n\n", num_vars);

         printf("---------------------------------------------\n");
         printf("               Mesh Refinement\n");
         printf("---------------------------------------------\n\n");
         printf("     Time: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[42], stddev[42], minimum[42], maximum[42]);
         printf("     Number of refinement steps: %d\n\n", nrs);
         printf("     Number of load balance steps: %d\n\n", nlbs);
         printf("     Number of redistributing steps: %d\n\n", nrrs);
         printf("     Total blocks           : %ld\n", total_blocks);
         printf("     Blocks/timestep ave, min, max : %lf %ld %ld\n",
                ((double) total_blocks)/((double) (num_tsteps*stages_per_ts)),
                (long long) nb_min, (long long) nb_max);
         printf("     Max blocks on a processor at any time: %d\n",
                global_max_b);
         printf("     total blocks split     : %lf\n", average[104]*num_pes);
         printf("     total blocks reformed  : %lf\n\n", average[105]*num_pes);
         printf("     total blocks moved     : %lf\n", average[106]*num_pes);
         printf("     total moved load bal   : %lf\n", average[107]*num_pes);
         printf("     total moved redistribut: %lf\n", average[122]*num_pes);
         printf("     total moved coasening  : %lf\n", average[109]*num_pes);
         printf("                              average    stddev  minimum  maximum\n");
         printf("     Per processor:\n");
         printf("     total blocks split     : %lf %lf %lf %lf\n",
                average[104], stddev[104], minimum[104], maximum[104]);
         printf("     total blocks reformed  : %lf %lf %lf %lf\n",
                average[105], stddev[105], minimum[105], maximum[105]);
         printf("     Total blocks moved     : %lf %lf %lf %lf\n",
                average[106], stddev[106], minimum[106], maximum[106]);
         printf("     Blocks moved load bal  : %lf %lf %lf %lf\n",
                average[107], stddev[107], minimum[107], maximum[107]);
         printf("     Blocks moved redistribu: %lf %lf %lf %lf\n",
                average[122], stddev[122], minimum[122], maximum[122]);
         printf("     Blocks moved coarsening: %lf %lf %lf %lf\n",
                average[109], stddev[109], minimum[109], maximum[109]);
         printf("     Time:\n");
         printf("        compare objects     : %lf %lf %lf %lf\n",
                average[43], stddev[43], minimum[43], maximum[43]);
         printf("        mark refine/coarsen : %lf %lf %lf %lf\n",
                average[44], stddev[44], minimum[44], maximum[44]);
         printf("        communicate block 1 : %lf %lf %lf %lf\n",
             average[119], stddev[119], minimum[119], maximum[119]);
         printf("        split blocks        : %lf %lf %lf %lf\n",
                average[46], stddev[46], minimum[46], maximum[46]);
         printf("        communicate block 2 : %lf %lf %lf %lf\n",
                average[120], stddev[120], minimum[120], maximum[120]);
         printf("        sync time           : %lf %lf %lf %lf\n",
                average[121], stddev[121], minimum[121], maximum[121]);
         printf("        misc time           : %lf %lf %lf %lf\n",
                average[45], stddev[45], minimum[45], maximum[45]);
         printf("        total coarsen blocks: %lf %lf %lf %lf\n",
                average[47], stddev[47], minimum[47], maximum[47]);
         printf("           coarsen blocks   : %lf %lf %lf %lf\n",
                average[48], stddev[48], minimum[48], maximum[48]);
         printf("           pack blocks      : %lf %lf %lf %lf\n",
                average[49], stddev[49], minimum[49], maximum[49]);
         printf("           move blocks      : %lf %lf %lf %lf\n",
                average[50], stddev[50], minimum[50], maximum[50]);
         printf("           unpack blocks    : %lf %lf %lf %lf\n",
                average[51], stddev[51], minimum[51], maximum[51]);
         printf("        total redistribute  : %lf %lf %lf %lf\n",
                average[123], stddev[123], minimum[123], maximum[123]);
         printf("           choose blocks    : %lf %lf %lf %lf\n",
                average[124], stddev[124], minimum[124], maximum[124]);
         printf("           pack blocks      : %lf %lf %lf %lf\n",
                average[125], stddev[125], minimum[125], maximum[125]);
         printf("           move blocks      : %lf %lf %lf %lf\n",
                average[126], stddev[126], minimum[126], maximum[126]);
         printf("           unpack blocks    : %lf %lf %lf %lf\n",
                average[127], stddev[127], minimum[127], maximum[127]);
         printf("        total load balance  : %lf %lf %lf %lf\n",
                average[62], stddev[62], minimum[62], maximum[62]);
         printf("           sort             : %lf %lf %lf %lf\n",
                average[63], stddev[63], minimum[63], maximum[63]);
         printf("           move dots back   : %lf %lf %lf %lf\n",
                average[117], stddev[117], minimum[117], maximum[117]);
         printf("           move blocks total: %lf %lf %lf %lf\n",
                average[118], stddev[118], minimum[118], maximum[118]);
         printf("              pack blocks   : %lf %lf %lf %lf\n",
                average[64], stddev[64], minimum[64], maximum[64]);
         printf("              move blocks   : %lf %lf %lf %lf\n",
                average[65], stddev[65], minimum[65], maximum[65]);
         printf("              unpack blocks : %lf %lf %lf %lf\n",
                average[66], stddev[66], minimum[66], maximum[66]);
         printf("              misc          : %lf %lf %lf %lf\n\n",
                average[116], stddev[116], minimum[116], maximum[116]);

         printf("---------------------------------------------\n");
         printf("                   Plot\n");
         printf("---------------------------------------------\n\n");
         printf("     Time: ave, stddev, min, max (sec): %lf %lf %lf %lf\n\n",
                average[67], stddev[67], minimum[67], maximum[67]);
         printf("     Number of plot steps: %d\n", nps);
         printf("\n ================== End report ===================\n");
printf("Summary: ranks %d threads %d ts %d time %lf calc %lf max comm %lf min red %lf refine %lf blocks/ts %lf max_blocks %d\n", num_pes, ompt, num_tsteps, average[0], average[38], maximum[37], minimum[39], average[42], ((double) total_blocks)/((double) (num_tsteps*stages_per_ts)), global_max_b);
      }
   }
}

void calculate_results(void)
{
   double results[131], stddev_sum[128];
   int i;

   results[0] = timer_all;
   for (i = 0; i < 9; i++)
      results[i+1] = 0.0;
   for (i = 0; i < 3; i++) {
      results[1] += results[10+9*i] = timer_comm_dir[i];
      results[2] += results[11+9*i] = timer_comm_recv[i];
      results[3] += results[12+9*i] = timer_comm_pack[i];
      results[4] += results[13+9*i] = timer_comm_send[i];
      results[5] += results[14+9*i] = timer_comm_same[i];
      results[6] += results[15+9*i] = timer_comm_diff[i];
      results[7] += results[16+9*i] = timer_comm_bc[i];
      results[8] += results[17+9*i] = timer_comm_wait[i];
      results[9] += results[18+9*i] = timer_comm_unpack[i];
   }
   results[37] = timer_comm_all;
   results[38] = timer_calc_all;
   results[39] = timer_cs_all;
   results[40] = timer_cs_red;
   results[41] = timer_cs_calc;
   results[42] = timer_refine_all;
   results[43] = timer_refine_co;
   results[44] = timer_refine_mr;
   results[45] = timer_refine_cc;
   results[46] = timer_refine_sb;
   results[119] = timer_refine_c1;
   results[120] = timer_refine_c2;
   results[121] = timer_refine_sy;
   results[47] = timer_cb_all;
   results[48] = timer_cb_cb;
   results[49] = timer_cb_pa;
   results[50] = timer_cb_mv;
   results[51] = timer_cb_un;
   results[52] = 0;
   results[53] = 0;
   results[54] = 0;
   results[55] = 0;
   results[56] = 0;
   results[57] = 0;
   results[58] = 0;
   results[59] = 0;
   results[60] = 0;
   results[61] = 0;
   results[62] = timer_lb_all;
   results[63] = timer_lb_sort;
   results[64] = timer_lb_pa;
   results[65] = timer_lb_mv;
   results[66] = timer_lb_un;
   results[116] = timer_lb_misc;
   results[117] = timer_lb_mb;
   results[118] = timer_lb_ma;
   results[67] = timer_plot;
   results[123] = timer_rs_all;
   results[124] = timer_rs_ca;
   results[125] = timer_rs_pa;
   results[126] = timer_rs_mv;
   results[127] = timer_rs_un;
   for (i = 0; i < 9; i++)
      results[68+i] = 0.0;
   for (i = 0; i < 3; i++) {
      results[68] += results[77+9*i] = size_mesg_recv[i];
      results[69] += results[78+9*i] = size_mesg_send[i];
      results[70] += results[79+9*i] = (double) counter_halo_recv[i];
      results[71] += results[80+9*i] = (double) counter_halo_send[i];
      results[72] += results[81+9*i] = (double) counter_face_recv[i];
      results[73] += results[82+9*i] = (double) counter_face_send[i];
      results[74] += results[83+9*i] = (double) counter_bc[i];
      results[75] += results[84+9*i] = (double) counter_same[i];
      results[76] += results[85+9*i] = (double) counter_diff[i];
   }
   results[104] = (double) num_refined;
   results[105] = (double) num_reformed;
   num_moved_all = num_moved_lb + num_moved_reduce + num_moved_coarsen +
                   num_moved_rs;
   results[106] = (double) num_moved_all;
   results[107] = (double) num_moved_lb;
   results[122] = (double) num_moved_rs;
   results[108] = (double) num_moved_reduce;
   results[109] = (double) num_moved_coarsen;
   results[110] = (double) counter_malloc;
   results[111] = size_malloc;
   results[112] = (double) counter_malloc_init;
   results[113] = size_malloc_init;
   results[114] = (double) (counter_malloc - counter_malloc_init);
   results[115] = size_malloc - size_malloc_init;
   results[128] = total_fp_adds;
   results[129] = total_fp_muls;
   results[130] = total_fp_divs;

   MPI_Allreduce(results, average, 131, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(results, minimum, 128, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   MPI_Allreduce(results, maximum, 128, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   for (i = 0; i < 128; i++) {
      average[i] /= (double) num_pes;
      stddev[i] = (results[i] - average[i])*(results[i] - average[i]);
   }
   MPI_Allreduce(stddev, stddev_sum, 128, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   for (i = 0; i < 128; i++)
      stddev[i] = sqrt(stddev_sum[i]/((double) num_pes));
}

void init_profile(void)
{
   int i;

   timer_all = 0.0;

   timer_comm_all = 0.0;
   for (i = 0; i < 3; i++) {
      timer_comm_dir[i] = 0.0;
      timer_comm_recv[i] = 0.0;
      timer_comm_pack[i] = 0.0;
      timer_comm_send[i] = 0.0;
      timer_comm_same[i] = 0.0;
      timer_comm_diff[i] = 0.0;
      timer_comm_bc[i] = 0.0;
      timer_comm_wait[i] = 0.0;
      timer_comm_unpack[i] = 0.0;
   }

   timer_calc_all = 0.0;

   timer_cs_all = 0.0;
   timer_cs_red = 0.0;
   timer_cs_calc = 0.0;

   timer_refine_all = 0.0;
   timer_refine_co = 0.0;
   timer_refine_mr = 0.0;
   timer_refine_cc = 0.0;
   timer_refine_sb = 0.0;
   timer_refine_c1 = 0.0;
   timer_refine_c2 = 0.0;
   timer_refine_sy = 0.0;
   timer_cb_all = 0.0;
   timer_cb_cb = 0.0;
   timer_cb_pa = 0.0;
   timer_cb_mv = 0.0;
   timer_cb_un = 0.0;
   timer_lb_all = 0.0;
   timer_lb_sort = 0.0;
   timer_lb_pa = 0.0;
   timer_lb_mv = 0.0;
   timer_lb_un = 0.0;
   timer_lb_misc = 0.0;
   timer_lb_mb = 0.0;
   timer_lb_ma = 0.0;
   timer_rs_all = 0.0;
   timer_rs_ca = 0.0;
   timer_rs_pa = 0.0;
   timer_rs_mv = 0.0;
   timer_rs_un = 0.0;

   timer_plot = 0.0;

   total_blocks = 0;
   nrrs = 0;
   nrs = 0;
   nps = 0;
   nlbs = 0;
   num_refined = 0;
   num_reformed = 0;
   num_moved_all = 0;
   num_moved_lb = 0;
   num_moved_rs = 0;
   num_moved_reduce = 0;
   num_moved_coarsen = 0;
   for (i = 0; i < 3; i++) {
      counter_halo_recv[i] = 0;
      counter_halo_send[i] = 0;
      size_mesg_recv[i] = 0.0;
      size_mesg_send[i] = 0.0;
      counter_face_recv[i] = 0;
      counter_face_send[i] = 0;
      counter_bc[i] = 0;
      counter_same[i] = 0;
      counter_diff[i] = 0;
   }
   total_red = 0;
}
