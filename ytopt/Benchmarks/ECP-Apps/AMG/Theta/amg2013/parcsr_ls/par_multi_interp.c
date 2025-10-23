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



#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_ParAMGBuildMultipass
 * This routine implements Stuben's direct interpolation with multiple passes. 
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGBuildMultipass( hypre_ParCSRMatrix  *A,
                   int                 *CF_marker,
                   hypre_ParCSRMatrix  *S,
                   HYPRE_BigInt        *num_cpts_global,
                   int                  num_functions,
                   int                 *dof_func,
                   int                  debug_flag,
                   double               trunc_factor,
                   int		 	P_max_elmts,
                   int                  weight_option,
                   int                 *col_offd_S_to_A,
                   hypre_ParCSRMatrix **P_ptr )
{
   MPI_Comm	           comm = hypre_ParCSRMatrixComm(A); 
   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg    *tmp_comm_pkg;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   double          *A_diag_data = hypre_CSRMatrixData(A_diag);
   int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   double          *A_offd_data = NULL;
   int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   int             *A_offd_j = NULL;
   HYPRE_BigInt	   *col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   int		    num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   int             *S_diag_i = hypre_CSRMatrixI(S_diag);
   int             *S_diag_j = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);
   int             *S_offd_i = hypre_CSRMatrixI(S_offd);
   int             *S_offd_j = NULL;
   HYPRE_BigInt	   *col_map_offd_S = hypre_ParCSRMatrixColMapOffd(S);
   int		    num_cols_offd_S = hypre_CSRMatrixNumCols(S_offd);
   HYPRE_BigInt	   *col_map_offd = NULL;
   int		    num_cols_offd;

   hypre_ParCSRMatrix *P;
   hypre_CSRMatrix *P_diag;
   double          *P_diag_data;
   int             *P_diag_i; /*at first counter of nonzero cols for each row,
				finally will be pointer to start of row */
   int             *P_diag_j;

   hypre_CSRMatrix *P_offd;
   double          *P_offd_data = NULL;
   int             *P_offd_i; /*at first counter of nonzero cols for each row,
				finally will be pointer to start of row */
   int             *P_offd_j = NULL;

   int              num_sends = 0;
   int             *int_buf_data = NULL;
   HYPRE_BigInt    *big_buf_data = NULL;
   int             *send_map_start;
   int             *send_map_elmt;
   int             *send_procs;
   int              num_recvs = 0;
   int             *recv_vec_start;
   int             *recv_procs;
   int             *new_recv_vec_start = NULL;
   int            **Pext_send_map_start = NULL;
   int            **Pext_recv_vec_start = NULL;
   int             *Pext_start = NULL;
   int             *P_ncols = NULL;
   
   int             *CF_marker_offd = NULL;
   int             *dof_func_offd = NULL;
   int             *P_marker;
   int             *P_marker_offd = NULL;
   int             *C_array;
   int             *C_array_offd = NULL;
   int             *pass_array = NULL; /* contains points ordered according to pass */
   int             *pass_pointer = NULL; /* pass_pointer[j] contains pointer to first
				point of pass j contained in pass_array */
   int             *P_diag_start;
   int             *P_offd_start = NULL;
   int            **P_diag_pass;
   int            **P_offd_pass = NULL;
   int            **Pext_pass = NULL;
   HYPRE_BigInt    *big_temp_pass = NULL;
   HYPRE_BigInt   **new_elmts = NULL; /* new neighbors generated in each pass */
   int             *new_counter = NULL; /* contains no. of new neighbors for
					each pass */
   int             *loc = NULL; /* contains locations for new neighbor 
			connections in int_o_buffer to avoid searching */
   int             *Pext_i = NULL; /*contains P_diag_i and P_offd_i info for nonzero
				cols of off proc neighbors */
   HYPRE_BigInt    *Pext_send_buffer = NULL; /* used to collect global nonzero
				col ids in P_diag for send_map_elmts */

   int             *map_S_to_new = NULL;
   /*int             *map_A_to_new = NULL;*/
   int             *map_A_to_S = NULL;
   HYPRE_BigInt    *new_col_map_offd = NULL;
   HYPRE_BigInt    *col_map_offd_P = NULL;
   HYPRE_BigInt    *big_permute = NULL;
   int             *permute = NULL;

   int              cnt;
   int              cnt_nz;
   int              total_nz;
   int              pass;
   int              num_passes;
   int              max_num_passes = 10;

   int              n_fine;
   int              n_coarse = 0;
   int              n_coarse_offd = 0;
   int              n_SF = 0;
   int              n_SF_offd = 0;

   int             *fine_to_coarse = NULL;
   HYPRE_BigInt    *fine_to_coarse_offd = NULL;

   int             *assigned = NULL;
   int             *assigned_offd = NULL;

   double          *Pext_send_data = NULL;
   double          *Pext_data = NULL;

   double           sum_C, sum_N;
   double           sum_C_pos, sum_C_neg;
   double           sum_N_pos, sum_N_neg;
   double           diagonal;
   double           alfa = 1.0;
   double           beta = 1.0;
   int              j_start;
   int              j_end;

   int              i,i1;
   int              j,j1;
   int              k,k1,k2,k3;
   int              pass_array_size;
   HYPRE_BigInt     global_pass_array_size;
   HYPRE_BigInt     local_pass_array_size;
   int              my_id, num_procs;
   int              index, start;
   HYPRE_BigInt     my_first_cpt;
   HYPRE_BigInt     total_global_cpts;
   HYPRE_BigInt     big_value;
   int              p_cnt;
   int              total_nz_offd;
   int              cnt_nz_offd;
   int              cnt_offd, cnt_new;
   int              no_break;
   int              not_found;
   int              Pext_send_size;
   int              Pext_recv_size;
   int              old_Pext_send_size;
   int              old_Pext_recv_size;
   int              P_offd_size = 0;
   int              local_index = -1;
   int              new_num_cols_offd = 0;
   int              num_cols_offd_P;
   int              num_threads;
   int              ns, ne, size, rest, pass_length;
   int 		   *tmp_marker, *tmp_marker_offd;
   int 		   *tmp_array, *tmp_array_offd;
   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for A and S. Also get size of fine grid.
    *-----------------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   /*   total_global_cpts = 0; */
    if (my_id == (num_procs -1)) total_global_cpts = num_cpts_global[1];
    MPI_Bcast(&total_global_cpts, 1, MPI_HYPRE_BIG_INT, num_procs-1, comm); 
#else
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   if (!comm_pkg)
   {
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      if (!comm_pkg)
      {
#ifdef HYPRE_NO_GLOBAL_PARTITION
          hypre_NewCommPkgCreate(A);
#else
          hypre_MatvecCommPkgCreate(A);
#endif
          comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }
      col_offd_S_to_A = NULL;
   }

   if (col_offd_S_to_A)
   {
      col_map_offd = col_map_offd_S;
      num_cols_offd = num_cols_offd_S;
   }
   else
   {
      col_map_offd = col_map_offd_A;
      num_cols_offd = num_cols_offd_A;
   }

   if (num_cols_offd_A)
   {
      A_offd_data = hypre_CSRMatrixData(A_offd);
      A_offd_j    = hypre_CSRMatrixJ(A_offd);
   }

   if (num_cols_offd)
      S_offd_j    = hypre_CSRMatrixJ(S_offd);

   n_fine = hypre_CSRMatrixNumRows(A_diag);

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   if (n_fine) fine_to_coarse = hypre_CTAlloc(int, n_fine);

   n_coarse = 0;
   n_SF = 0;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) reduction(+:n_coarse,n_SF ) schedule(static)
#endif
   for (i=0; i < n_fine; i++)
      if (CF_marker[i] == 1) n_coarse++;
      else if (CF_marker[i] == -3) n_SF++;

   pass_array_size = n_fine-n_coarse-n_SF;
   if (pass_array_size) pass_array = hypre_CTAlloc(int, pass_array_size);
   pass_pointer = hypre_CTAlloc(int, max_num_passes+1);
   if (n_fine) assigned = hypre_CTAlloc(int, n_fine);

   {
      P_diag_i = hypre_CTAlloc(int, n_fine+1);
      P_offd_i = hypre_CTAlloc(int, n_fine+1);
   }
   if (n_coarse) C_array = hypre_CTAlloc(int, n_coarse);

   if (num_cols_offd)
   {
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      if (num_functions > 1) dof_func_offd = hypre_CTAlloc(int, num_cols_offd);
   }

   if (num_procs > 1)
   {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      send_map_start = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmt = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      recv_vec_start = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      if (send_map_start[num_sends])
      {
         int_buf_data = hypre_CTAlloc(int, send_map_start[num_sends]);
         big_buf_data = hypre_CTAlloc(HYPRE_BigInt, send_map_start[num_sends]);
      }
   }

   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = send_map_start[i];
      for (j = start; j < send_map_start[i+1]; j++)
	 int_buf_data[index++] = CF_marker[send_map_elmt[j]];
   }
   if (num_procs > 1)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
	 CF_marker_offd);
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   if (num_functions > 1)
   {
      index = 0;
      for (i=0; i < num_sends; i++)
      {
         start = send_map_start[i];
         for (j = start; j < send_map_start[i+1]; j++)
	    int_buf_data[index++] = dof_func[send_map_elmt[j]];
      }
      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
	    dof_func_offd);
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }
   }

   n_coarse_offd = 0;
   n_SF_offd = 0;
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) reduction(+:n_coarse_offd,n_SF_offd) schedule(static)
#endif
   for (i=0; i < num_cols_offd; i++)
      if (CF_marker_offd[i] == 1) n_coarse_offd++;
      else if (CF_marker_offd[i] == -3) n_SF_offd++;

   if (num_cols_offd)
   {
      assigned_offd = hypre_CTAlloc(int, num_cols_offd);
      map_S_to_new = hypre_CTAlloc(int, num_cols_offd);
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd);
      new_col_map_offd = hypre_CTAlloc(HYPRE_BigInt, n_coarse_offd);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: determine the maximal size of P, and elementsPerRow[i].
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Assigned points are points for which we know an interpolation
    *  formula already, and which are thus available to interpolate from.
    *  assigned[i]=0 for C points, and 1, 2, 3, ... for F points, depending
    *  in which pass their interpolation formula is determined.
    *  
    *  pass_array contains the points ordered according to its pass, i.e.
    *  |  C-points   |  points of pass 1 | points of pass 2 | ....
    * C_points are points 0 through pass_pointer[1]-1,
    * points of pass k  (0 < k < num_passes) are contained in points
    * pass_pointer[k] through pass_pointer[k+1]-1 of pass_array .
    *
    * pass_array is also used to avoid going through all points for each pass,
    * i,e. at the bginning it contains all points in descending order starting
    * with n_fine-1. Then starting from the last point, we evaluate whether
    * it is a C_point (pass 0). If it is the point is brought to the front
    * and the length of the points to be searched is shortened.  This is
    * done until the parameter cnt (which determines the first point of
    * pass_array to be searched) becomes n_fine. Then all points have been
    * assigned a pass number.
    *-----------------------------------------------------------------------*/


   cnt = 0;
   p_cnt = pass_array_size-1;
   P_diag_i[0] = 0;
   P_offd_i[0] = 0;
   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] == 1)
      {
         fine_to_coarse[i] = cnt; /* this C point is assigned index
					      coarse_counter on coarse grid,
					      and in column of P */
	 C_array[cnt++] = i;
         assigned[i] = 0;
         P_diag_i[i+1] = 1; /* one element in row i1 of P */
         P_offd_i[i+1] = 0;
      }
      else if (CF_marker[i] == -1)
      {
         pass_array[p_cnt--] = i;
         P_diag_i[i+1] = 0;
         P_offd_i[i+1] = 0;
         assigned[i] = -1;
         fine_to_coarse[i] = -1;
      }
      else
      {
         P_diag_i[i+1] = 0;
         P_offd_i[i+1] = 0;
         assigned[i] = -1;
         fine_to_coarse[i] = -1;
      }
   }

   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = send_map_start[i];
      for (j = start; j < send_map_start[i+1]; j++)
      {
	 big_buf_data[index] = (HYPRE_BigInt) fine_to_coarse[send_map_elmt[j]];
	 if (big_buf_data[index] > -1) 
	    big_buf_data[index] += my_first_cpt;
	 index++;
      }
   }
   if (num_procs > 1)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, big_buf_data,
	 fine_to_coarse_offd);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      hypre_TFree(big_buf_data);

   }

   new_recv_vec_start = hypre_CTAlloc(int,num_recvs+1);

   if (n_coarse_offd)
      C_array_offd = hypre_CTAlloc(int,n_coarse_offd);

   cnt = 0;
   new_recv_vec_start[0] = 0;
   for (j = 0; j < num_recvs; j++)
   {
      for (i = recv_vec_start[j]; i < recv_vec_start[j+1]; i++)
      {
         if (CF_marker_offd[i] == 1)
         {
	    map_S_to_new[i] = cnt;
	    C_array_offd[cnt] = i;
	    new_col_map_offd[cnt++] = fine_to_coarse_offd[i];
            assigned_offd[i] = 0; 
         }
         else
         {
            assigned_offd[i] = -1;
            map_S_to_new[i] = -1;
         }
      }
      new_recv_vec_start[j+1] = cnt;
   }

   cnt = 0;
   hypre_TFree(fine_to_coarse_offd);

   if (col_offd_S_to_A)
   {
      map_A_to_S = hypre_CTAlloc(int,num_cols_offd_A);
      for (i=0; i < num_cols_offd_A; i++)
      {
        if (cnt < num_cols_offd && col_map_offd_A[i] == col_map_offd[cnt])
	   map_A_to_S[i] = cnt++;
        else
	   map_A_to_S[i] = -1;
      }
   }

   /*-----------------------------------------------------------------------
    *  Mark all local neighbors of C points as 'assigned'.
    *-----------------------------------------------------------------------*/

   pass_pointer[0] = 0;
   pass_pointer[1] = 0;
   total_nz = n_coarse;  /* accumulates total number of nonzeros in P_diag */
   total_nz_offd = 0; /* accumulates total number of nonzeros in P_offd */

   cnt = 0;
   cnt_offd = 0;
   cnt_nz = 0;
   cnt_nz_offd = 0;
   for (i = pass_array_size-1; i > cnt-1; i--)
   {
     i1 = pass_array[i];
     for (j=S_diag_i[i1]; j < S_diag_i[i1+1]; j++)
     {
        j1 = S_diag_j[j];
        if (CF_marker[j1] == 1)
        {
           P_diag_i[i1+1]++;
           cnt_nz++;
           assigned[i1] = 1;
        }
     }
     for (j=S_offd_i[i1]; j < S_offd_i[i1+1]; j++)
     {
        j1 = S_offd_j[j];
        if (CF_marker_offd[j1] == 1)
        {
           P_offd_i[i1+1]++;
           cnt_nz_offd++;
           assigned[i1] = 1;
        }
     }
     if (assigned[i1] == 1)
     {
        pass_array[i++] = pass_array[cnt];
        pass_array[cnt++] = i1;
     }
   }

   pass_pointer[2] = cnt;

   /*-----------------------------------------------------------------------
    *  All local neighbors are assigned, now need to exchange the boundary
    *  info for assigned strong neighbors.
    *-----------------------------------------------------------------------*/

   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = send_map_start[i];
      for (j = start; j < send_map_start[i+1]; j++)
	 int_buf_data[index++] = assigned[send_map_elmt[j]];
   }
   if (num_procs > 1)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
	 assigned_offd);
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /*-----------------------------------------------------------------------
    *  Now we need to determine strong neighbors of points of pass 1, etc.
    *  we need to update assigned_offd after each pass
    *-----------------------------------------------------------------------*/

   pass = 2;
   local_pass_array_size = (HYPRE_BigInt)(pass_array_size - cnt);
   MPI_Allreduce(&local_pass_array_size, &global_pass_array_size, 1, 
		MPI_HYPRE_BIG_INT, MPI_SUM, comm);

   while (global_pass_array_size && pass < max_num_passes)
   {
      for (i = pass_array_size-1; i > cnt-1; i--)
      {
	 i1 = pass_array[i];
         no_break = 1;
         for (j=S_diag_i[i1]; j < S_diag_i[i1+1]; j++)
	 {
	    j1 = S_diag_j[j];
	    if (assigned[j1] == pass-1)
  	    {
               pass_array[i++] = pass_array[cnt];
               pass_array[cnt++] = i1; 
               assigned[i1] = pass;
               no_break = 0;
               break;
	    }
     	 }
     	 if (no_break)
	 {
            for (j=S_offd_i[i1]; j < S_offd_i[i1+1]; j++)
	    {
	       j1 = S_offd_j[j];
	       if (assigned_offd[j1] == pass-1)
  	       {
                  pass_array[i++] = pass_array[cnt];
                  pass_array[cnt++] = i1; 
                  assigned[i1] = pass;
                  break;
  	       }
	    }
     	 }
      }
      /*printf("pass %d  remaining points %d \n", pass, local_pass_array_size);*/

      pass++;
      pass_pointer[pass] = cnt;

      local_pass_array_size = (HYPRE_BigInt) (pass_array_size - cnt);
      MPI_Allreduce(&local_pass_array_size, &global_pass_array_size, 1, 
		MPI_HYPRE_BIG_INT, MPI_SUM, comm);
      index = 0;
      for (i=0; i < num_sends; i++)
      {
         start = send_map_start[i];
         for (j = start; j < send_map_start[i+1]; j++)
	    int_buf_data[index++] = assigned[send_map_elmt[j]];
      }
      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
	 assigned_offd);
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }
   }

   hypre_TFree(int_buf_data);

   num_passes = pass;

   P_marker = hypre_CTAlloc(int, n_coarse); /* marks points to see if they have
						been counted */
   for (i=0; i < n_coarse; i++)
      P_marker[i] = -1;

   if (n_coarse_offd)
   {
      P_marker_offd = hypre_CTAlloc(int, n_coarse_offd);
      for (i=0; i < n_coarse_offd; i++)
         P_marker_offd[i] = -1;
   }

   P_diag_pass = hypre_CTAlloc(int*,num_passes); /* P_diag_pass[i] will contain
				 all column numbers for points of pass i */

   P_diag_pass[1] = hypre_CTAlloc(int,cnt_nz);

   P_diag_start = hypre_CTAlloc(int, n_fine); /* P_diag_start[i] contains
	   pointer to begin of column numbers in P_pass for point i,
	   P_diag_i[i+1] contains number of columns for point i */

   P_offd_start = hypre_CTAlloc(int, n_fine);

   if (num_procs > 1)
   {
      P_offd_pass = hypre_CTAlloc(int*,num_passes);

      if (cnt_nz_offd)
         P_offd_pass[1] = hypre_CTAlloc(int,cnt_nz_offd);
      else
         P_offd_pass[1] = NULL;

      new_elmts = hypre_CTAlloc(HYPRE_BigInt*,num_passes);

      new_counter = hypre_CTAlloc(int, num_passes+1);

      new_counter[0] = 0;
      new_counter[1] = n_coarse_offd;
      new_num_cols_offd = n_coarse_offd;

      new_elmts[0] = new_col_map_offd;
   }

   /*-----------------------------------------------------------------------
    *  Pass 1: now we consider points of pass 1, with strong C_neighbors,
    *-----------------------------------------------------------------------*/

   cnt_nz = 0;
   cnt_nz_offd = 0;
   for (i=pass_pointer[1]; i < pass_pointer[2]; i++)
   {
      i1 = pass_array[i];
      P_diag_start[i1] = cnt_nz;
      P_offd_start[i1] = cnt_nz_offd;
      for (j=S_diag_i[i1]; j < S_diag_i[i1+1]; j++)
      {
         j1 = S_diag_j[j];
	 if (CF_marker[j1] == 1)
	    P_diag_pass[1][cnt_nz++] = fine_to_coarse[j1];
      }
      for (j=S_offd_i[i1]; j < S_offd_i[i1+1]; j++)
      {
         j1 = S_offd_j[j];
	 if (CF_marker_offd[j1] == 1)
	    P_offd_pass[1][cnt_nz_offd++] = map_S_to_new[j1];
      }
   }


   total_nz += cnt_nz;
   total_nz_offd += cnt_nz_offd;

   if (num_procs > 1)
   {
      tmp_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
      Pext_send_map_start = hypre_CTAlloc(int*,num_passes);
      Pext_recv_vec_start = hypre_CTAlloc(int*,num_passes);
      Pext_pass = hypre_CTAlloc(int*,num_passes);
      Pext_i = hypre_CTAlloc(int, num_cols_offd+1);
      if (num_cols_offd) Pext_start = hypre_CTAlloc(int, num_cols_offd);
      if (send_map_start[num_sends])
         P_ncols = hypre_CTAlloc(int,send_map_start[num_sends]);
      /*for (i=0; i < num_cols_offd+1; i++)
         Pext_i[i] = 0;
      for (i=0; i < send_map_start[num_sends]; i++)
         P_ncols[i] = 0;*/
   }

   old_Pext_send_size = 0;
   old_Pext_recv_size = 0;
   for (pass=2; pass < num_passes; pass++)
   {
      if (num_procs > 1)
      {
         Pext_send_map_start[pass] = hypre_CTAlloc(int, num_sends+1);
         Pext_recv_vec_start[pass] = hypre_CTAlloc(int, num_recvs+1);
         Pext_send_size = 0;
         Pext_send_map_start[pass][0] = 0;

         for (i=0; i < num_sends; i++)
         {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(j,j1) reduction(+:Pext_send_size) schedule(static)
#endif

            for (j=send_map_start[i]; j < send_map_start[i+1]; j++)
            {
               j1 = send_map_elmt[j];
	       if (assigned[j1] == pass-1)
	       {
	          P_ncols[j] = P_diag_i[j1+1] + P_offd_i[j1+1];
	          Pext_send_size += P_ncols[j];
               }
            }
            Pext_send_map_start[pass][i+1] = Pext_send_size;
         }

         comm_handle = hypre_ParCSRCommHandleCreate (11, comm_pkg,
		P_ncols, &Pext_i[1]);
         hypre_ParCSRCommHandleDestroy(comm_handle);

         if (Pext_send_size > old_Pext_send_size)
         {
            hypre_TFree(Pext_send_buffer);
            Pext_send_buffer = hypre_CTAlloc(HYPRE_BigInt, Pext_send_size);
         }
         old_Pext_send_size = Pext_send_size;
      }

      cnt_offd = 0;
      for (i=0; i < num_sends; i++)
      {
         for (j=send_map_start[i]; j < send_map_start[i+1]; j++)
         {
            j1 = send_map_elmt[j];
	    if (assigned[j1] == pass-1)
	    {
	       j_start = P_diag_start[j1];
	       j_end = j_start+P_diag_i[j1+1];
	       for (k=j_start; k < j_end; k++)
	       {
		  Pext_send_buffer[cnt_offd++] = my_first_cpt
			+ (HYPRE_BigInt) P_diag_pass[pass-1][k];
	       }
	       j_start = P_offd_start[j1];
	       j_end = j_start+P_offd_i[j1+1];
	       for (k=j_start; k < j_end; k++)
	       {
		  k1 = P_offd_pass[pass-1][k];
		  k3 = 0;
		  while (k3 < pass-1)
		  {
		     if (k1 < new_counter[k3+1])
		     {
		        k2 = k1-new_counter[k3];
		        Pext_send_buffer[cnt_offd++] = new_elmts[k3][k2];
		        break;
		     }
		     k3++;
		  }
	       }
            }
         }
      }
 
      if (num_procs > 1)
      {
         Pext_recv_size = 0;
         Pext_recv_vec_start[pass][0] = 0;
         cnt_offd = 0;
         for (i=0; i < num_recvs; i++)
         {
 	    for (j=recv_vec_start[i]; j<recv_vec_start[i+1]; j++)
            {
	       if (assigned_offd[j] == pass-1)
               {
                  Pext_start[j] = cnt_offd;
                  cnt_offd += Pext_i[j+1];
               }
            }
            Pext_recv_size = cnt_offd;
            Pext_recv_vec_start[pass][i+1] = Pext_recv_size;
         }

         hypre_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
         hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
         hypre_ParCSRCommPkgSendProcs(tmp_comm_pkg) = send_procs;
         hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = 
		Pext_send_map_start[pass];
         hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
         hypre_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = recv_procs;
         hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = 
		Pext_recv_vec_start[pass];

         if (Pext_recv_size)
         {
            Pext_pass[pass] = hypre_CTAlloc(int, Pext_recv_size);
            new_elmts[pass-1] = hypre_CTAlloc(HYPRE_BigInt,Pext_recv_size);
         }
         else
         {
            Pext_pass[pass] = NULL;
            new_elmts[pass-1] = NULL;
         }

         if (Pext_recv_size > old_Pext_recv_size)
         {
            hypre_TFree(loc);
            hypre_TFree(big_temp_pass);
            loc = hypre_CTAlloc(int,Pext_recv_size);
            big_temp_pass = hypre_CTAlloc(HYPRE_BigInt, Pext_recv_size);
         }
         old_Pext_recv_size = Pext_recv_size;

         comm_handle = hypre_ParCSRCommHandleCreate (21, tmp_comm_pkg,
		Pext_send_buffer, big_temp_pass);
         hypre_ParCSRCommHandleDestroy(comm_handle);

      }

      cnt_new = 0;
      cnt_offd = 0;
      for (i=0; i < num_recvs; i++)
      {
         for (j=recv_vec_start[i]; j < recv_vec_start[i+1]; j++)
	 {
	    if (assigned_offd[j] == pass-1)
            {
	       for (j1 = cnt_offd; j1 < cnt_offd+Pext_i[j+1]; j1++)
               {
	          big_value = big_temp_pass[j1];
	          k2 = (int)(big_value - my_first_cpt);
	          if (k2 > -1 && k2 < n_coarse)
                  {
	             Pext_pass[pass][j1] = -k2-1;
                  }
                  else
                  {
                     not_found = 1;
                     k3 = 0;
                     while (k3 < pass-1 && not_found)
                     {
                        k2 = hypre_BigBinarySearch(new_elmts[k3], big_value, 
				(new_counter[k3+1]-new_counter[k3]));
                        if (k2 > -1)
                        {
	                   Pext_pass[pass][j1] = k2 + new_counter[k3];
	                   not_found = 0;
                        }
                        else
                        {
	                   k3++;
                        }
                     }
                     if (not_found)
                     {
		        new_elmts[pass-1][cnt_new] = big_value;
		        loc[cnt_new++] = j1;
                     }
                  }
               }
	       cnt_offd += Pext_i[j+1];
	    }
	 }
      }

      if (cnt_new)
      {
  	 hypre_BigQsortbi(new_elmts[pass-1],loc,0,cnt_new-1);
         cnt = 0;
         local_index = new_counter[pass-1];
	 Pext_pass[pass][loc[0]] = local_index;

         for (i=1; i < cnt_new; i++)
         {
	    if (new_elmts[pass-1][i] > new_elmts[pass-1][cnt])
	    {
	       new_elmts[pass-1][++cnt] = new_elmts[pass-1][i];
	       local_index++;
	    }
	    Pext_pass[pass][loc[i]] = local_index;
         }
         new_counter[pass] = local_index+1;
      }
      else if (num_procs > 1)
         new_counter[pass] = new_counter[pass-1];

      if (new_num_cols_offd < local_index+1)
      {
         new_num_cols_offd = local_index+1;

         hypre_TFree(P_marker_offd);
         P_marker_offd = hypre_CTAlloc(int,new_num_cols_offd);

         for (i=0; i < new_num_cols_offd; i++)
	    P_marker_offd[i] = -1;
      }
     
      cnt_nz = 0;
      cnt_nz_offd = 0;
      for (i=pass_pointer[pass]; i < pass_pointer[pass+1]; i++)
      {
	 i1 = pass_array[i];
         P_diag_start[i1] = cnt_nz;
         P_offd_start[i1] = cnt_nz_offd;
	 for (j=S_diag_i[i1]; j < S_diag_i[i1+1]; j++)
	 {
	    j1 = S_diag_j[j];
	    if (assigned[j1] == pass-1)
  	    {
	       j_start = P_diag_start[j1];
	       j_end = j_start+P_diag_i[j1+1];
	       for (k=j_start; k < j_end; k++)
	       {
		  k1 = P_diag_pass[pass-1][k];
		  if (P_marker[k1] != i1)
		  {
		     cnt_nz++;
		     P_diag_i[i1+1]++;
		     P_marker[k1] = i1;
		  }
	       }
	       j_start = P_offd_start[j1];
	       j_end = j_start+P_offd_i[j1+1];
	       for (k=j_start; k < j_end; k++)
	       {
		  k1 = P_offd_pass[pass-1][k];
		  if (P_marker_offd[k1] != i1)
		  {
		     cnt_nz_offd++;
		     P_offd_i[i1+1]++;
		     P_marker_offd[k1] = i1;
		  }
	       }
  	    }
  	 }
	 j_start = 0;
	 for (j=S_offd_i[i1]; j < S_offd_i[i1+1]; j++)
	 {
	    j1 = S_offd_j[j];
	    if (assigned_offd[j1] == pass-1)
  	    {
	       j_start = Pext_start[j1];
	       j_end = j_start+Pext_i[j1+1];
	       for (k=j_start; k < j_end; k++)
	       {
		  k1 = Pext_pass[pass][k];
		  if (k1 < 0)
		  {
                     if (P_marker[-k1-1] != i1)
		     {
		        cnt_nz++;
		        P_diag_i[i1+1]++;
		        P_marker[-k1-1] = i1;
		     }
		  }
		  else if (P_marker_offd[k1] != i1)
		  {
		     cnt_nz_offd++;
		     P_offd_i[i1+1]++;
		     P_marker_offd[k1] = i1;
		  }
	       }
  	    }
  	 }
      }

      total_nz += cnt_nz;
      total_nz_offd += cnt_nz_offd;

      P_diag_pass[pass] = hypre_CTAlloc(int, cnt_nz);
      if (cnt_nz_offd)
         P_offd_pass[pass] = hypre_CTAlloc(int, cnt_nz_offd);
      else if (num_procs > 1)
         P_offd_pass[pass] = NULL;

      cnt_nz = 0;
      cnt_nz_offd = 0;
      for (i=pass_pointer[pass]; i < pass_pointer[pass+1]; i++)
      {
         i1 = pass_array[i];
         for (j=S_diag_i[i1]; j < S_diag_i[i1+1]; j++)
         {
            j1 = S_diag_j[j];
	    if (assigned[j1] == pass-1)
  	    {
	       j_start = P_diag_start[j1];
	       j_end = j_start+P_diag_i[j1+1];
	       for (k=j_start; k < j_end; k++)
	       {
		  k1 = P_diag_pass[pass-1][k];
		  if (P_marker[k1] != -i1-1)
		  {
		      P_diag_pass[pass][cnt_nz++] = k1;
		      P_marker[k1] = -i1-1;
		  }
	       }
	       j_start = P_offd_start[j1];
	       j_end = j_start+P_offd_i[j1+1];
	       for (k=j_start; k < j_end; k++)
	       {
		  k1 = P_offd_pass[pass-1][k];
		  if (P_marker_offd[k1] != -i1-1)
		  {
		     P_offd_pass[pass][cnt_nz_offd++] = k1;
		     P_marker_offd[k1] = -i1-1;
		  }
	       }
  	    }
         }
         for (j=S_offd_i[i1]; j < S_offd_i[i1+1]; j++)
         {
            j1 = S_offd_j[j];
	    if (assigned_offd[j1] == pass-1)
  	    {
	       j_start = Pext_start[j1];
	       j_end = j_start+Pext_i[j1+1];
	       for (k=j_start; k < j_end; k++)
	       {
		  k1 = Pext_pass[pass][k];
		  if (k1 < 0)
		  {
 		     if (P_marker[-k1-1] != -i1-1)
		     {
		        P_diag_pass[pass][cnt_nz++] = -k1-1;
		        P_marker[-k1-1] = -i1-1;
		     }
		  }
		  else if (P_marker_offd[k1] != -i1-1)
		  {
		     P_offd_pass[pass][cnt_nz_offd++] = k1;
		     P_marker_offd[k1] = -i1-1;
		  }
	       }
  	    }
         }
      }
   }

   hypre_TFree(loc);
   hypre_TFree(P_ncols);
   hypre_TFree(Pext_send_buffer);
   hypre_TFree(big_temp_pass);
   hypre_TFree(new_recv_vec_start);
   hypre_TFree(P_marker);
   hypre_TFree(P_marker_offd);
   P_marker_offd = NULL;

   P_diag_j = NULL;
   P_offd_j = NULL;
   {

      if (total_nz)
      {
         P_diag_j = hypre_CTAlloc(int,total_nz);
         P_diag_data = hypre_CTAlloc(double,total_nz);
      }
      if (total_nz_offd)
      {
         P_offd_j = hypre_CTAlloc(int,total_nz_offd);
         P_offd_data = hypre_CTAlloc(double,total_nz_offd);
      }
   }

   for (i=0; i < n_fine; i++)
   {
      P_diag_i[i+1] += P_diag_i[i];
      P_offd_i[i+1] += P_offd_i[i];
   }

/* determine P for coarse points */

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,i1) schedule(static)
#endif
   for (i=0; i < n_coarse; i++)
   {
      i1 = C_array[i];
      P_diag_j[P_diag_i[i1]] = fine_to_coarse[i1];
      P_diag_data[P_diag_i[i1]] = 1.0;
   }

   P_marker = hypre_CTAlloc(int,n_fine);
   for (i=0; i < n_fine; i++)
      P_marker[i] = -1;

   if (num_cols_offd)
   {
      P_marker_offd = hypre_CTAlloc(int,num_cols_offd);
      for (i=0; i < num_cols_offd; i++)
         P_marker_offd[i] = -1;
   }

   if (weight_option) /*if this is set, weights are separated into
		negative and positive offdiagonals and accumulated
		accordingly */
   {
   /* determine P for points of pass 1, i.e. neighbors of coarse points */ 
      for (i=pass_pointer[1]; i < pass_pointer[2]; i++) 
      {
         i1 = pass_array[i];
         sum_C_pos = 0;
         sum_C_neg = 0;
         sum_N_pos = 0;
         sum_N_neg = 0;
         j_start = P_diag_start[i1];
         j_end = j_start+P_diag_i[i1+1]-P_diag_i[i1];
         for (j=j_start; j < j_end; j++)
         {
            k1 = P_diag_pass[1][j];
	    P_marker[C_array[k1]] = i1;
         }
         cnt = P_diag_i[i1];
         for (j=A_diag_i[i1]+1; j < A_diag_i[i1+1]; j++)
         {
            j1 = A_diag_j[j];
	    if (CF_marker[j1] != -3 &&
	       (num_functions == 1 || dof_func[i1] == dof_func[j1]))
	    {
	       if (A_diag_data[j] < 0)
	          sum_N_neg += A_diag_data[j];
	       else
	          sum_N_pos += A_diag_data[j];
	    }
	    if (j1 != -1 && P_marker[j1] == i1)
	    {
	       P_diag_data[cnt] = A_diag_data[j];
	       P_diag_j[cnt++] = fine_to_coarse[j1];
	       if (A_diag_data[j] < 0)
	          sum_C_neg += A_diag_data[j];
	       else
	          sum_C_pos += A_diag_data[j];
            }
         }
         j_start = P_offd_start[i1];
         j_end = j_start+P_offd_i[i1+1]-P_offd_i[i1];
         for (j=j_start; j < j_end; j++)
         {
            k1 = P_offd_pass[1][j];
	    P_marker_offd[C_array_offd[k1]] = i1;
         }
         cnt_offd = P_offd_i[i1];
         for (j=A_offd_i[i1]; j < A_offd_i[i1+1]; j++)
         {
	    if (col_offd_S_to_A)
               j1 = map_A_to_S[A_offd_j[j]];
	    else
               j1 = A_offd_j[j];
            if (CF_marker_offd[j1] != -3 &&
	       (num_functions == 1 || dof_func[i1] == dof_func_offd[j1]))
	    {
	       if (A_offd_data[j] < 0)
	          sum_N_neg += A_offd_data[j];
	       else
	          sum_N_pos += A_offd_data[j];
	    }
	    if (j1 != -1 && P_marker_offd[j1] == i1)
	    {
	       P_offd_data[cnt_offd] = A_offd_data[j];
	       P_offd_j[cnt_offd++] = map_S_to_new[j1];
	       if (A_offd_data[j] < 0)
	          sum_C_neg += A_offd_data[j];
	       else
	          sum_C_pos += A_offd_data[j];
            }
         }
         diagonal = A_diag_data[A_diag_i[i1]];
         if (sum_C_neg*diagonal) alfa = -sum_N_neg/(sum_C_neg*diagonal);
         if (sum_C_pos*diagonal) beta = -sum_N_pos/(sum_C_pos*diagonal);
         for (j=P_diag_i[i1]; j < cnt; j++)
            if (P_diag_data[j] < 0)
	       P_diag_data[j] *= alfa;
            else
	       P_diag_data[j] *= beta;
         for (j=P_offd_i[i1]; j < cnt_offd; j++)
            if (P_offd_data[j] < 0)
	       P_offd_data[j] *= alfa;
            else
	       P_offd_data[j] *= beta;
      }

      old_Pext_send_size = 0;
      old_Pext_recv_size = 0;

      /*if (!col_offd_S_to_A) hypre_TFree(map_A_to_new);*/
      hypre_TFree(P_diag_pass[1]);
      if (num_procs > 1) hypre_TFree(P_offd_pass[1]);
	  
      if (new_num_cols_offd > n_coarse_offd)
      {
         hypre_TFree(C_array_offd);
         C_array_offd = hypre_CTAlloc(int, new_num_cols_offd);
      }

      for (pass = 2; pass < num_passes; pass++)
      {

         if (num_procs > 1)
         {
            Pext_send_size = Pext_send_map_start[pass][num_sends];
            if (Pext_send_size > old_Pext_send_size)
            {
               hypre_TFree(Pext_send_data);
               Pext_send_data = hypre_CTAlloc(double, Pext_send_size);
            }
            old_Pext_send_size = Pext_send_size;

            cnt_offd = 0;
            for (i=0; i < num_sends; i++)
            {
               for (j=send_map_start[i]; j < send_map_start[i+1]; j++)
               {
                  j1 = send_map_elmt[j];
	          if (assigned[j1] == pass-1)
	          {
	             j_start = P_diag_i[j1];
	             j_end = P_diag_i[j1+1];
	             for (k=j_start; k < j_end; k++)
	             {
		        Pext_send_data[cnt_offd++] = P_diag_data[k];
	             }
	             j_start = P_offd_i[j1];
	             j_end = P_offd_i[j1+1];
	             for (k=j_start; k < j_end; k++)
	             {
		        Pext_send_data[cnt_offd++] = P_offd_data[k];
	             }
                  }
               }
            }
 
            hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
            hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = 
		Pext_send_map_start[pass];
            hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
            hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = 
		Pext_recv_vec_start[pass];

            Pext_recv_size = Pext_recv_vec_start[pass][num_recvs];

            if (Pext_recv_size > old_Pext_recv_size)
            {
               hypre_TFree(Pext_data);
               Pext_data = hypre_CTAlloc(double, Pext_recv_size);
            }
            old_Pext_recv_size = Pext_recv_size;

            comm_handle = hypre_ParCSRCommHandleCreate (1, tmp_comm_pkg,
		Pext_send_data, Pext_data);
            hypre_ParCSRCommHandleDestroy(comm_handle);

            hypre_TFree(Pext_send_map_start[pass]);
            hypre_TFree(Pext_recv_vec_start[pass]);
         }

         for (i=pass_pointer[pass]; i < pass_pointer[pass+1]; i++)
         {
            i1 = pass_array[i];
            sum_C_neg = 0;
            sum_C_pos = 0;
            sum_N_neg = 0;
            sum_N_pos = 0;
            j_start = P_diag_start[i1];
            j_end = j_start+P_diag_i[i1+1]-P_diag_i[i1];
            cnt = P_diag_i[i1];
            for (j=j_start; j < j_end; j++)
            {
               k1 = P_diag_pass[pass][j];
	       C_array[k1] = cnt;
	       P_diag_data[cnt] = 0;
	       P_diag_j[cnt++] = k1;
            }
            j_start = P_offd_start[i1];
            j_end = j_start+P_offd_i[i1+1]-P_offd_i[i1];
            cnt_offd = P_offd_i[i1];
            for (j=j_start; j < j_end; j++)
            {
               k1 = P_offd_pass[pass][j];
	       C_array_offd[k1] = cnt_offd;
	       P_offd_data[cnt_offd] = 0;
	       P_offd_j[cnt_offd++] = k1;
            }
            for (j=S_diag_i[i1]; j < S_diag_i[i1+1]; j++)
            {
	       j1 = S_diag_j[j];
	       if (assigned[j1] == pass-1)
	          P_marker[j1] = i1;
            }
            for (j=S_offd_i[i1]; j < S_offd_i[i1+1]; j++)
            {
	       j1 = S_offd_j[j];
	       if (assigned_offd[j1] == pass-1)
	          P_marker_offd[j1] = i1; 
            }
            for (j=A_diag_i[i1]+1; j < A_diag_i[i1+1]; j++)
            {
	       j1 = A_diag_j[j];
	       if (P_marker[j1] == i1)
	       {
	          for (k=P_diag_i[j1]; k < P_diag_i[j1+1]; k++)
	          {
		     k1 = P_diag_j[k];
	             alfa = A_diag_data[j]*P_diag_data[k];
	             P_diag_data[C_array[k1]] += alfa;
	             if (alfa < 0)
	             {
	                sum_C_neg += alfa;
	                sum_N_neg += alfa;
	             }
	             else
	             {
	                sum_C_pos += alfa;
	                sum_N_pos += alfa;
	             }
                  }
	          for (k=P_offd_i[j1]; k < P_offd_i[j1+1]; k++)
	          {
		     k1 = P_offd_j[k];
	             alfa = A_diag_data[j]*P_offd_data[k];
	             P_offd_data[C_array_offd[k1]] += alfa;
	             if (alfa < 0)
	             {
	                sum_C_neg += alfa;
	                sum_N_neg += alfa;
	             }
	             else
	             {
	                sum_C_pos += alfa;
	                sum_N_pos += alfa;
	             }
                  }
               }
               else
               {
                  if (CF_marker[j1] != -3 &&
		     (num_functions == 1 || dof_func[i1] == dof_func[j1]))
                  {
		     if (A_diag_data[j] < 0)
		        sum_N_neg += A_diag_data[j];
		     else
		        sum_N_pos += A_diag_data[j];
                  }
               }
            }
            for (j=A_offd_i[i1]; j < A_offd_i[i1+1]; j++)
            {
	       if (col_offd_S_to_A)
	          j1 = map_A_to_S[A_offd_j[j]];
	       else
	          j1 = A_offd_j[j];
 
	       if (j1 > -1 && P_marker_offd[j1] == i1)
	       {
	          j_start = Pext_start[j1];
	          j_end = j_start+Pext_i[j1+1];
	          for (k=j_start; k < j_end; k++)
	          {
		     k1 = Pext_pass[pass][k];
	             alfa = A_offd_data[j]*Pext_data[k];
		     if (k1 < 0) 
	                P_diag_data[C_array[-k1-1]] += alfa;
		     else
	                P_offd_data[C_array_offd[k1]] += alfa;
	             if (alfa < 0)
		     {
	                sum_C_neg += alfa;
	                sum_N_neg += alfa;
                     }
	             else
		     {
	                sum_C_pos += alfa;
	                sum_N_pos += alfa;
                     }
                  }
               }
               else
               {
                  if (CF_marker_offd[j1] != -3 && 
		(num_functions == 1 || dof_func_offd[j1] == dof_func[i1])) 
                  {
		     if ( A_offd_data[j] < 0)
		        sum_N_neg += A_offd_data[j];
		     else
		        sum_N_pos += A_offd_data[j];
                  }
               }
            }
            diagonal = A_diag_data[A_diag_i[i1]];
            if (sum_C_neg*diagonal) alfa = -sum_N_neg/(sum_C_neg*diagonal);
            if (sum_C_pos*diagonal) beta = -sum_N_pos/(sum_C_pos*diagonal);

            for (j=P_diag_i[i1]; j < P_diag_i[i1+1]; j++)
	       if (P_diag_data[j] < 0)
	          P_diag_data[j] *= alfa;
	       else
	          P_diag_data[j] *= beta;
            for (j=P_offd_i[i1]; j < P_offd_i[i1+1]; j++)
	       if (P_offd_data[j] < 0)
	          P_offd_data[j] *= alfa;
	       else
	          P_offd_data[j] *= beta;
         }
         hypre_TFree(P_diag_pass[pass]);
         if (num_procs > 1)
         {
            hypre_TFree(P_offd_pass[pass]);
            hypre_TFree(Pext_pass[pass]);
         }
      }
   }
   else /* no distinction between positive and negative offdiagonal element */
   {
   /* determine P for points of pass 1, i.e. neighbors of coarse points */ 
    pass_length = pass_pointer[2]-pass_pointer[1];
#define HYPRE_SMP_PRIVATE k,k1,i,i1,j,j1,ns,ne,rest,size,sum_C,sum_N,j_start,j_end,cnt,tmp_marker,tmp_marker_offd,cnt_offd,diagonal,alfa
#include "../utilities/hypre_smp_forloop.h"
    for (k = 0; k < num_threads; k++)
    {
      size = pass_length/num_threads;
      rest = pass_length - size*num_threads;
      if (k < rest)
      {
          ns = k*size+k;
          ne = (k+1)*size+k+1;
      }
      else
      {
          ns = k*size+rest;
          ne = (k+1)*size+rest;
      }
      ns = pass_pointer[1]+ns;
      ne = pass_pointer[1]+ne;

      tmp_marker = NULL;
      if (n_fine) tmp_marker = hypre_CTAlloc(int,n_fine);
      tmp_marker_offd = NULL;
      if (num_cols_offd) tmp_marker_offd = hypre_CTAlloc(int,num_cols_offd);
      for (i=0; i < n_fine; i++)
	 tmp_marker[i] = -1;
      for (i=0; i < num_cols_offd; i++)
	 tmp_marker_offd[i] = -1;

      /*for (i=pass_pointer[1]; i < pass_pointer[2]; i++) 
      {*/
      for (i=ns; i < ne; i++) 
      {
         i1 = pass_array[i];
         sum_C = 0;
         sum_N = 0;
         j_start = P_diag_start[i1];
         j_end = j_start+P_diag_i[i1+1]-P_diag_i[i1];
         for (j=j_start; j < j_end; j++)
         {
            k1 = P_diag_pass[1][j];
	    tmp_marker[C_array[k1]] = i1;
         }
         cnt = P_diag_i[i1];
         for (j=A_diag_i[i1]+1; j < A_diag_i[i1+1]; j++)
         {
            j1 = A_diag_j[j];
	    if (CF_marker[j1] != -3 && 
		(num_functions == 1 || dof_func[i1] == dof_func[j1]))
	       sum_N += A_diag_data[j];
	    if (j1 != -1 && tmp_marker[j1] == i1)
	    {
	       P_diag_data[cnt] = A_diag_data[j];
	       P_diag_j[cnt++] = fine_to_coarse[j1];
	       sum_C += A_diag_data[j];
            }
         }
         j_start = P_offd_start[i1];
         j_end = j_start+P_offd_i[i1+1]-P_offd_i[i1];
         for (j=j_start; j < j_end; j++)
         {
            k1 = P_offd_pass[1][j];
	    tmp_marker_offd[C_array_offd[k1]] = i1;
         }
         cnt_offd = P_offd_i[i1];
         for (j=A_offd_i[i1]; j < A_offd_i[i1+1]; j++)
         {
	    if (col_offd_S_to_A)
               j1 = map_A_to_S[A_offd_j[j]];
	    else
               j1 = A_offd_j[j];
            if (CF_marker_offd[j1] != -3 && 
		(num_functions == 1 || dof_func[i1] == dof_func_offd[j1]))
	       sum_N += A_offd_data[j];
	    if (j1 != -1 && tmp_marker_offd[j1] == i1)
	    {
	       P_offd_data[cnt_offd] = A_offd_data[j];
	       P_offd_j[cnt_offd++] = map_S_to_new[j1];
	       sum_C += A_offd_data[j];
            }
         }
         diagonal = A_diag_data[A_diag_i[i1]];
         if (sum_C*diagonal) alfa = -sum_N/(sum_C*diagonal);
         for (j=P_diag_i[i1]; j < cnt; j++)
	    P_diag_data[j] *= alfa;
         for (j=P_offd_i[i1]; j < cnt_offd; j++)
	    P_offd_data[j] *= alfa;
       }
       hypre_TFree(tmp_marker);
       hypre_TFree(tmp_marker_offd);
      }

      old_Pext_send_size = 0;
      old_Pext_recv_size = 0;

      hypre_TFree(P_diag_pass[1]);
      if (num_procs > 1) hypre_TFree(P_offd_pass[1]);
	  
      /*if (new_num_cols_offd > n_coarse_offd)
      {
         hypre_TFree(C_array_offd);
         C_array_offd = hypre_CTAlloc(int, new_num_cols_offd);
      }*/

      for (pass = 2; pass < num_passes; pass++)
      {

         if (num_procs > 1)
         {
            Pext_send_size = Pext_send_map_start[pass][num_sends];
            if (Pext_send_size > old_Pext_send_size)
            {
               hypre_TFree(Pext_send_data);
               Pext_send_data = hypre_CTAlloc(double, Pext_send_size);
            }
            old_Pext_send_size = Pext_send_size;

            cnt_offd = 0;
            for (i=0; i < num_sends; i++)
            {
               for (j=send_map_start[i]; j < send_map_start[i+1]; j++)
               {
                  j1 = send_map_elmt[j];
	          if (assigned[j1] == pass-1)
	          {
	             j_start = P_diag_i[j1];
	             j_end = P_diag_i[j1+1];
	             for (k=j_start; k < j_end; k++)
	             {
		        Pext_send_data[cnt_offd++] = P_diag_data[k];
	             }
	             j_start = P_offd_i[j1];
	             j_end = P_offd_i[j1+1];
	             for (k=j_start; k < j_end; k++)
	             {
		        Pext_send_data[cnt_offd++] = P_offd_data[k];
	             }
                  }
               }
            }
 
            hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
            hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = 
		Pext_send_map_start[pass];
            hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
            hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = 
		Pext_recv_vec_start[pass];

            Pext_recv_size = Pext_recv_vec_start[pass][num_recvs];

            if (Pext_recv_size > old_Pext_recv_size)
            {
               hypre_TFree(Pext_data);
               Pext_data = hypre_CTAlloc(double, Pext_recv_size);
            }
            old_Pext_recv_size = Pext_recv_size;

            comm_handle = hypre_ParCSRCommHandleCreate (1, tmp_comm_pkg,
		Pext_send_data, Pext_data);
            hypre_ParCSRCommHandleDestroy(comm_handle);

            hypre_TFree(Pext_send_map_start[pass]);
            hypre_TFree(Pext_recv_vec_start[pass]);
         }

         pass_length = pass_pointer[pass+1]-pass_pointer[pass];
#define HYPRE_SMP_PRIVATE k,k1,k2,i,i1,j,j1,ns,ne,rest,size,sum_C,sum_N,j_start,j_end,cnt,tmp_marker,tmp_marker_offd,cnt_offd,diagonal,alfa,tmp_array,tmp_array_offd
#include "../utilities/hypre_smp_forloop.h"
         for (k = 0; k < num_threads; k++)
         {
           size = pass_length/num_threads;
           rest = pass_length - size*num_threads;
           if (k < rest)
           {
              ns = k*size+k;
              ne = (k+1)*size+k+1;
           }
           else
           {
              ns = k*size+rest;
              ne = (k+1)*size+rest;
           }
           ns = pass_pointer[pass]+ns;
           ne = pass_pointer[pass]+ne;

           tmp_marker = NULL;
           if (n_fine) tmp_marker = hypre_CTAlloc(int,n_fine);
           tmp_marker_offd = NULL;
           if (num_cols_offd) tmp_marker_offd = hypre_CTAlloc(int,num_cols_offd);
           tmp_array = NULL;
           if (n_coarse) tmp_array = hypre_CTAlloc(int,n_coarse);
      /*if (new_num_cols_offd > n_coarse_offd)
      {
         hypre_TFree(C_array_offd);
         C_array_offd = hypre_CTAlloc(int, new_num_cols_offd);
      }*/
           tmp_array_offd = NULL;
           if (new_num_cols_offd > n_coarse_offd) 
	 	tmp_array_offd = hypre_CTAlloc(int,new_num_cols_offd);
           else
	 	tmp_array_offd = hypre_CTAlloc(int,n_coarse_offd);

           for (i=0; i < n_fine; i++)
	      tmp_marker[i] = -1;
           for (i=0; i < num_cols_offd; i++)
	      tmp_marker_offd[i] = -1;

           for (i=ns; i < ne; i++) 
           {
         /*for (i=pass_pointer[pass]; i < pass_pointer[pass+1]; i++)
         {*/
            i1 = pass_array[i];
            sum_C = 0;
            sum_N = 0;
            j_start = P_diag_start[i1];
            j_end = j_start+P_diag_i[i1+1]-P_diag_i[i1];
            cnt = P_diag_i[i1];
            for (j=j_start; j < j_end; j++)
            {
               k1 = P_diag_pass[pass][j];
	       tmp_array[k1] = cnt;
	       P_diag_data[cnt] = 0;
	       P_diag_j[cnt++] = k1;
            }
            j_start = P_offd_start[i1];
            j_end = j_start+P_offd_i[i1+1]-P_offd_i[i1];
            cnt_offd = P_offd_i[i1];
            for (j=j_start; j < j_end; j++)
            {
               k1 = P_offd_pass[pass][j];
	       tmp_array_offd[k1] = cnt_offd;
	       P_offd_data[cnt_offd] = 0;
	       P_offd_j[cnt_offd++] = k1;
            }
            for (j=S_diag_i[i1]; j < S_diag_i[i1+1]; j++)
            {
	       j1 = S_diag_j[j];
	       if (assigned[j1] == pass-1)
	          /*P_marker[j1] = i1;*/
	          tmp_marker[j1] = i1;
            }
            for (j=S_offd_i[i1]; j < S_offd_i[i1+1]; j++)
            {
	       j1 = S_offd_j[j];
	       if (assigned_offd[j1] == pass-1)
	          /*P_marker_offd[j1] = i1; */
	          tmp_marker_offd[j1] = i1; 
            }
            for (j=A_diag_i[i1]+1; j < A_diag_i[i1+1]; j++)
            {
	       j1 = A_diag_j[j];
	       /*if (P_marker[j1] == i1)*/
	       if (tmp_marker[j1] == i1)
	       {
	          for (k2=P_diag_i[j1]; k2 < P_diag_i[j1+1]; k2++)
	          {
		     k1 = P_diag_j[k2];
	             alfa = A_diag_data[j]*P_diag_data[k2];
	             P_diag_data[tmp_array[k1]] += alfa;
	             sum_C += alfa;
	             sum_N += alfa;
                  }
	          for (k2=P_offd_i[j1]; k2 < P_offd_i[j1+1]; k2++)
	          {
		     k1 = P_offd_j[k2];
	             alfa = A_diag_data[j]*P_offd_data[k2];
	             P_offd_data[tmp_array_offd[k1]] += alfa;
	             sum_C += alfa;
	             sum_N += alfa;
                  }
               }
               else
               {
                  if (CF_marker[j1] != -3 && 
			(num_functions == 1 || dof_func[i1] == dof_func[j1]))
		     sum_N += A_diag_data[j];
               }
            }
            for (j=A_offd_i[i1]; j < A_offd_i[i1+1]; j++)
            {
	       if (col_offd_S_to_A)
	          j1 = map_A_to_S[A_offd_j[j]];
	       else
	          j1 = A_offd_j[j];
 
	       /*if (j1 > -1 && P_marker_offd[j1] == i1)*/
	       if (j1 > -1 && tmp_marker_offd[j1] == i1)
	       {
	          j_start = Pext_start[j1];
	          j_end = j_start+Pext_i[j1+1];
	          for (k2=j_start; k2 < j_end; k2++)
	          {
		     k1 = Pext_pass[pass][k2];
	             alfa = A_offd_data[j]*Pext_data[k2];
		     if (k1 < 0) 
	                P_diag_data[tmp_array[-k1-1]] += alfa;
		     else
	                P_offd_data[tmp_array_offd[k1]] += alfa;
	             sum_C += alfa;
	             sum_N += alfa;
                  }
               }
               else
               {
                  if (CF_marker_offd[j1] != -3 && 
		    (num_functions == 1 || dof_func_offd[j1] == dof_func[i1])) 
		     sum_N += A_offd_data[j];
               }
            }
            diagonal = A_diag_data[A_diag_i[i1]];
            if (sum_C*diagonal) alfa = -sum_N/(sum_C*diagonal);

            for (j=P_diag_i[i1]; j < P_diag_i[i1+1]; j++)
	       P_diag_data[j] *= alfa;
            for (j=P_offd_i[i1]; j < P_offd_i[i1+1]; j++)
	       P_offd_data[j] *= alfa;
          }
          hypre_TFree(tmp_marker);
          hypre_TFree(tmp_marker_offd);
          hypre_TFree(tmp_array);
          hypre_TFree(tmp_array_offd);
         }
         
         hypre_TFree(P_diag_pass[pass]);
         if (num_procs > 1)
         {
            hypre_TFree(P_offd_pass[pass]);
            hypre_TFree(Pext_pass[pass]);
         }
      }
   }

   hypre_TFree(CF_marker_offd);
   hypre_TFree(Pext_send_map_start);
   hypre_TFree(Pext_recv_vec_start);
   if (n_coarse) hypre_TFree(C_array);
   hypre_TFree(C_array_offd);
   hypre_TFree(dof_func_offd);
   hypre_TFree(Pext_send_data);
   hypre_TFree(Pext_data);
   hypre_TFree(P_diag_pass);
   hypre_TFree(P_offd_pass);
   hypre_TFree(Pext_pass);
   hypre_TFree(P_diag_start);
   hypre_TFree(P_offd_start);
   hypre_TFree(Pext_start);
   hypre_TFree(Pext_i);
   hypre_TFree(fine_to_coarse);
   hypre_TFree(assigned);
   hypre_TFree(assigned_offd);
   hypre_TFree(pass_pointer);
   hypre_TFree(pass_array);
   hypre_TFree(map_S_to_new);
   hypre_TFree(map_A_to_S);
   hypre_TFree(P_marker);
   if (num_procs > 1) hypre_TFree(tmp_comm_pkg);

    P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                0,
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);
   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;

   /* Compress P, removing coefficients smaller than trunc_factor * Max 
      and/or keep yat most <P_max_elmts> per row absolutely maximal coefficients */

   if (trunc_factor != 0.0 || P_max_elmts != 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, P_max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
   }
   P_offd_size = P_offd_i[n_fine];

   num_cols_offd_P = 0;
   if (P_offd_size)
   {
      if (new_num_cols_offd > num_cols_offd)
      {
          hypre_TFree(P_marker_offd);
          P_marker_offd = hypre_CTAlloc(int,new_num_cols_offd);
      }

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i=0; i < new_num_cols_offd; i++)
         P_marker_offd[i] = 0;
 
      num_cols_offd_P = 0;
      for (i=0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker_offd[index])
         {
            num_cols_offd_P++;
            P_marker_offd[index] = 1;
         }
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt,num_cols_offd_P);
      permute = hypre_CTAlloc(int, new_counter[num_passes-1]);
      big_permute = hypre_CTAlloc(HYPRE_BigInt, new_counter[num_passes-1]);

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i=0; i < new_counter[num_passes-1]; i++)
	 big_permute[i] = -1;

      cnt = 0;
      for (i=0; i < num_passes-1; i++)
      {
         for (j=new_counter[i]; j < new_counter[i+1]; j++)
         {
	    if (P_marker_offd[j])
	    {
	       col_map_offd_P[cnt] = new_elmts[i][j-new_counter[i]];
	       big_permute[j] = col_map_offd_P[cnt++];
	    }
         }
      }

      hypre_BigQsort0(col_map_offd_P,0,num_cols_offd_P-1);

#define HYPRE_SMP_PRIVATE i, big_value
#include "../utilities/hypre_smp_forloop.h"
      for (i=0; i < new_counter[num_passes-1]; i++)
      {
         big_value = big_permute[i];
         if (big_value != -1)
            permute[i] = hypre_BigBinarySearch(col_map_offd_P,
			big_value,num_cols_offd_P);
      }

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i=0; i < P_offd_size; i++)
         P_offd_j[i] = permute[P_offd_j[i]];
   }
   if (num_procs > 1)
   {
      for (i=0; i < num_passes-1; i++)
         hypre_TFree(new_elmts[i]);
   }
   hypre_TFree(P_marker_offd);
   hypre_TFree(permute);
   hypre_TFree(big_permute);
   hypre_TFree(new_elmts);
   hypre_TFree(new_counter);

   if (num_cols_offd_P)
   {
        hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
        hypre_CSRMatrixNumCols(P_offd) = num_cols_offd_P;
   }

   if (n_SF)
   {
      for (i=0; i < n_fine; i++)
	 if (CF_marker[i] == -3) CF_marker[i] = -1;
   }

   if (num_procs > 1)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
        hypre_NewCommPkgCreate(P);
#else
        hypre_MatvecCommPkgCreate(P);
#endif
   }

   *P_ptr = P;


   /*-----------------------------------------------------------------------
    *  Build and return dof_func array for coarse grid.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Free mapping vector and marker array.
    *-----------------------------------------------------------------------*/


   return(0);
}

