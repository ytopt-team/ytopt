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

double average[131];
double stddev[128];
double minimum[128];
double maximum[128];

double timer_all;

double timer_comm_all;
double timer_comm_dir[3];
double timer_comm_recv[3];
double timer_comm_pack[3];
double timer_comm_send[3];
double timer_comm_same[3];
double timer_comm_diff[3];
double timer_comm_bc[3];
double timer_comm_wait[3];
double timer_comm_unpack[3];

double timer_calc_all;

double timer_cs_all;
double timer_cs_red;
double timer_cs_calc;

double timer_refine_all;
double timer_refine_co;
double timer_refine_mr;
double timer_refine_cc;
double timer_refine_sb;
double timer_refine_c1;
double timer_refine_c2;
double timer_refine_sy;
double timer_cb_all;
double timer_cb_cb;
double timer_cb_pa;
double timer_cb_mv;
double timer_cb_un;
double timer_lb_all;
double timer_lb_sort;
double timer_lb_pa;
double timer_lb_mv;
double timer_lb_un;
double timer_lb_misc;
double timer_lb_mb;
double timer_lb_ma;
double timer_rs_all;
double timer_rs_ca;
double timer_rs_pa;
double timer_rs_mv;
double timer_rs_un;

double timer_plot;

long long total_blocks;
num_sz nb_min;
num_sz nb_max;
int nrrs;
int nrs;
int nps;
int nlbs;
int num_refined;
int num_reformed;
int num_moved_all;
int num_moved_lb;
int num_moved_rs;
int num_moved_reduce;
int num_moved_coarsen;
int counter_halo_recv[3];
int counter_halo_send[3];
double size_mesg_recv[3];
double size_mesg_send[3];
int counter_face_recv[3];
int counter_face_send[3];
int counter_bc[3];
int counter_same[3];
int counter_diff[3];
int counter_malloc;
double size_malloc;
int counter_malloc_init;
double size_malloc_init;
int total_red;
