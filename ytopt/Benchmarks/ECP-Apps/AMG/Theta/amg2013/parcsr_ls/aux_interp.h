/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/



void
hypre_ParCSRCommExtendA(hypre_ParCSRMatrix *A, int newoff, int *found,
			int *p_num_recvs, int **p_recv_procs,
			int **p_recv_vec_starts, int *p_num_sends,
			int **p_send_procs, int **p_send_map_starts,
			int **p_send_map_elmts, int **p_node_add);

void insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg, int *IN_marker, 
		      int *node_add, int num_cols_A_offd, 
		      int full_off_procNodes, int num_procs,
		      int *OUT_marker);

/*int new_offd_nodes(int **found, int A_ext_rows, int *A_ext_i, 
		   HYPRE_BigInt *big_A_ext_j, int **A_ext_j, 
		   int num_cols_A_offd, int *col_map_offd, int col_1, 
		   int col_n, int *Sop_i, HYPRE_BigInt big_Sop_j, 
		   int **Sop_j,
		   int *CF_marker, hypre_ParCSRCommPkg *comm_pkg); */
