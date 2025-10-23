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
#include "aux_interp.h"

#define MAX_C_CONNECTIONS 100
#define HAVE_COMMON_C 1

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildPartialStdInterp
 *  Comment: The interpolatory weighting can be changed with the sep_weight
 *           variable. This can enable not separating negative and positive
 *           off diagonals in the weight formula.
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGBuildPartialStdInterp(hypre_ParCSRMatrix *A, int *CF_marker,
			      hypre_ParCSRMatrix   *S, HYPRE_BigInt *num_cpts_global,
			      HYPRE_BigInt *num_old_cpts_global,
			      int num_functions, int *dof_func, int debug_flag,
			      double trunc_factor, int max_elmts, 
			      int sep_weight, int *col_offd_S_to_A, 
			      hypre_ParCSRMatrix  **P_ptr)
{
  /* Communication Variables */
  MPI_Comm 	           comm = hypre_ParCSRMatrixComm(A);   
  hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  int              my_id, num_procs;

  /* Variables to store input variables */
  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A); 
  double          *A_diag_data = hypre_CSRMatrixData(A_diag);
  int             *A_diag_i = hypre_CSRMatrixI(A_diag);
  int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

  hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
  double          *A_offd_data = hypre_CSRMatrixData(A_offd);
  int             *A_offd_i = hypre_CSRMatrixI(A_offd);
  int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

  int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
  HYPRE_BigInt    *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
  int              n_fine = hypre_CSRMatrixNumRows(A_diag);
  HYPRE_BigInt     col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
  int              local_numrows = hypre_CSRMatrixNumRows(A_diag);
  HYPRE_BigInt     col_n = col_1 + (HYPRE_BigInt)local_numrows;
  HYPRE_BigInt     total_global_cpts, my_first_cpt;
  HYPRE_BigInt     total_old_global_cpts, my_first_old_cpt;
  int              n_coarse, n_coarse_old;

  /* Variables to store strong connection matrix info */
  hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
  int             *S_diag_i = hypre_CSRMatrixI(S_diag);
  int             *S_diag_j = hypre_CSRMatrixJ(S_diag);
   
  hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
  int             *S_offd_i = hypre_CSRMatrixI(S_offd);
  int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

  /* Interpolation matrix P */
  hypre_ParCSRMatrix *P;
  hypre_CSRMatrix    *P_diag;
  hypre_CSRMatrix    *P_offd;   

  double          *P_diag_data = NULL;
  int             *P_diag_i, *P_diag_j = NULL;
  double          *P_offd_data = NULL;
  int             *P_offd_i, *P_offd_j = NULL;

  HYPRE_BigInt	  *col_map_offd_P;
  int		  *tmp_map_offd;
  int              P_diag_size; 
  int              P_offd_size;
  int             *P_marker = NULL; 
  int             *P_marker_offd = NULL;
  int             *CF_marker_offd = NULL;
  int             *tmp_CF_marker_offd = NULL;
  int             *dof_func_offd = NULL;

  /* Full row information for columns of A that are off diag*/
  hypre_BigCSRMatrix *A_ext;   
  double          *A_ext_data;
  int             *A_ext_i;
  HYPRE_BigInt    *big_A_ext_j;
  int             *A_ext_j;
  
  int             *fine_to_coarse = NULL;
  int             *old_coarse_to_fine = NULL;
  HYPRE_BigInt    *fine_to_coarse_offd = NULL;
  HYPRE_BigInt    *found;

  int              num_cols_P_offd;
  int              newoff, loc_col;
  int              A_ext_rows, full_off_procNodes;
  
  hypre_BigCSRMatrix *Sop;
  int             *Sop_i;
  HYPRE_BigInt    *big_Sop_j;
  int             *Sop_j;
  
  int              Soprows;
  
  /* Variables to keep count of interpolatory points */
  int              jj_counter, jj_counter_offd;
  int              jj_begin_row, jj_end_row;
  int              jj_begin_row_offd = 0;
  int              jj_end_row_offd = 0;
  int              coarse_counter, coarse_counter_offd;
  int             *ihat = NULL; 
  int             *ihat_offd = NULL; 
  int             *ipnt = NULL; 
  int             *ipnt_offd = NULL; 
  int              strong_f_marker = -2;
    
  /* Interpolation weight variables */
  double          *ahat_offd = NULL; 
  double          *ahat = NULL; 
  double           sum_pos, sum_pos_C, sum_neg, sum_neg_C, sum, sum_C;
  double           diagonal, distribute;
  double           alfa, beta;
  
  /* Loop variables */
  int              index, cnt, old_cnt;
  int              start_indexing = 0;
  int              i, ii, i1, j1, jj, kk, k1;
  int              cnt_c, cnt_f, cnt_c_offd, cnt_f_offd, indx;

  /* Definitions */
  double           zero = 0.0;
  double           one  = 1.0;
  double           wall_time;
  double           wall_1 = 0;
  double           wall_2 = 0;
  double           wall_3 = 0;
  

  hypre_ParCSRCommPkg	*extend_comm_pkg = NULL;

  if (debug_flag== 4) wall_time = time_getWallclockSeconds();

  /* BEGIN */
  MPI_Comm_size(comm, &num_procs);   
  MPI_Comm_rank(comm,&my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   my_first_old_cpt = num_old_cpts_global[0];
   n_coarse_old = (int)(num_old_cpts_global[1] - num_old_cpts_global[0]);
   n_coarse = (int)(num_cpts_global[1] - num_cpts_global[0]);
   if (my_id == (num_procs -1))
   {
      total_global_cpts = num_cpts_global[1];
      total_old_global_cpts = num_old_cpts_global[1];
   }
   MPI_Bcast(&total_global_cpts, 1, MPI_HYPRE_BIG_INT, num_procs-1, comm);
   MPI_Bcast(&total_old_global_cpts, 1, MPI_HYPRE_BIG_INT, num_procs-1, comm);
#else
   my_first_cpt = num_cpts_global[my_id];
   my_first_old_cpt = num_old_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
   total_old_global_cpts = num_old_cpts_global[num_procs];
   n_coarse_old = (int)(num_old_cpts_global[my_id+1] - num_old_cpts_global[my_id]);
   n_coarse = (int)(num_cpts_global[my_id+1] - num_cpts_global[my_id]);
#endif

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
     hypre_NewCommPkgCreate(A);
#else
     hypre_MatvecCommPkgCreate(A);
#endif
     comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }
   
   /* Set up off processor information (specifically for neighbors of 
    * neighbors */
   newoff = 0;
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
     /*----------------------------------------------------------------------
      * Get the off processors rows for A and S, associated with columns in 
      * A_offd and S_offd.
      *---------------------------------------------------------------------*/
     A_ext         = hypre_ParCSRMatrixExtractBigExt(A,A,1);
     A_ext_i       = hypre_BigCSRMatrixI(A_ext);
     big_A_ext_j   = hypre_BigCSRMatrixJ(A_ext);
     A_ext_data    = hypre_BigCSRMatrixData(A_ext);
     A_ext_rows    = hypre_BigCSRMatrixNumRows(A_ext);
     
     Sop           = hypre_ParCSRMatrixExtractBigExt(S,A,0);
     Sop_i         = hypre_BigCSRMatrixI(Sop);
     big_Sop_j     = hypre_BigCSRMatrixJ(Sop);
     Soprows       = hypre_BigCSRMatrixNumRows(Sop);

     /* Find nodes that are neighbors of neighbors, not found in offd */
     newoff = new_offd_nodes(&found, A_ext_rows, A_ext_i, big_A_ext_j,
			     &A_ext_j, Soprows, col_map_offd, col_1, 
			     col_n, Sop_i, big_Sop_j, &Sop_j, CF_marker, 
			     comm_pkg);

     hypre_BigCSRMatrixJ(A_ext) = NULL;
     hypre_BigCSRMatrixJ(Sop) = NULL;

     if(newoff >= 0)
       full_off_procNodes = newoff + num_cols_A_offd;
     else
       return(1);

     /* Possibly add new points and new processors to the comm_pkg, all
      * processors need new_comm_pkg */

     /* AHB - create a new comm package just for extended info -
        this will work better with the assumed partition*/
     hypre_ParCSRFindExtendCommPkg(A, newoff, found, 
                                   &extend_comm_pkg);
     
     CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
     
     if (num_functions > 1 && full_off_procNodes > 0)
        dof_func_offd = hypre_CTAlloc(int, full_off_procNodes);
     
     alt_insert_new_nodes(comm_pkg, extend_comm_pkg, CF_marker, 
                          full_off_procNodes, CF_marker_offd);
     
     if(num_functions > 1)
        alt_insert_new_nodes(comm_pkg, extend_comm_pkg, dof_func, 
                             full_off_procNodes, dof_func_offd);
   }


   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(int, n_coarse_old+1);
   P_offd_i    = hypre_CTAlloc(int, n_coarse_old+1);

   if (n_coarse_old)  old_coarse_to_fine = hypre_CTAlloc(int, n_coarse_old);
   if (n_fine)
   {
      fine_to_coarse = hypre_CTAlloc(int, n_fine);
      P_marker = hypre_CTAlloc(int, n_fine);
   }

   if (full_off_procNodes)
   {
      P_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, full_off_procNodes);
      tmp_CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
   }

   for (i=0; i < full_off_procNodes; i++)
   {
      P_marker_offd[i] = -1;
      tmp_CF_marker_offd[i] = -1;
      fine_to_coarse_offd[i] = -1;
   }

   for (i=0; i < n_fine; i++)
   {
      P_marker[i] = -1;
      fine_to_coarse[i] = -1;
   }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;
   coarse_counter_offd = 0;

   cnt = 0;
   old_cnt = 0;
   for (i = 0; i < n_fine; i++)
   {
      fine_to_coarse[i] = -1;
      if (CF_marker[i] == 1)
      {
         fine_to_coarse[i] = cnt++;
         old_coarse_to_fine[old_cnt++] = i;
      }
      else if (CF_marker[i] == -2)
      {
         old_coarse_to_fine[old_cnt++] = i;
      }
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (ii = 0; ii < n_coarse_old; ii++)
   {
     P_diag_i[ii] = jj_counter;
     if (num_procs > 1)
       P_offd_i[ii] = jj_counter_offd;
    
     i = old_coarse_to_fine[ii]; 
     if (CF_marker[i] > 0)
     {
       jj_counter++;
       coarse_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, interpolation is from the C-points that
      *  strongly influence i, or C-points that stronly influence F-points
      *  that strongly influence i.
      *--------------------------------------------------------------------*/
     else if (CF_marker[i] == -2)
     {
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       {
	 i1 = S_diag_j[jj];           
	 if (CF_marker[i1] > 0)
	 { /* i1 is a C point */
	   if (P_marker[i1] < P_diag_i[ii])
	   {
	     P_marker[i1] = jj_counter;
	     jj_counter++;
	   }
	 }
	 else if (CF_marker[i1] != -3)
	 { /* i1 is a F point, loop through it's strong neighbors */
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] > 0)
	     {
	       if(P_marker[k1] < P_diag_i[ii])
	       {
		 P_marker[k1] = jj_counter;
		 jj_counter++;
	       }
	     } 
	   }
	   if(num_procs > 1)
	   {
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];
	       if (CF_marker_offd[k1] > 0)
	       {
		 if(P_marker_offd[k1] < P_offd_i[ii])
		 {
		   tmp_CF_marker_offd[k1] = 1;
		   P_marker_offd[k1] = jj_counter_offd;
		   jj_counter_offd++;
		 }
	       }
	     }
	   }
	 }
       }
       /* Look at off diag strong connections of i */ 
       if (num_procs > 1)
       {
	 for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   i1 = S_offd_j[jj];           
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[i1];
	   if (CF_marker_offd[i1] > 0)
	   {
	     if(P_marker_offd[i1] < P_offd_i[ii])
	     {
	       tmp_CF_marker_offd[i1] = 1;
	       P_marker_offd[i1] = jj_counter_offd;
	       jj_counter_offd++;
	     }
	   }
	   else if (CF_marker_offd[i1] != -3)
	   { /* F point; look at neighbors of i1. Sop contains global col
	      * numbers and entries that could be in S_diag or S_offd or
	      * neither. */
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     {
	       loc_col = Sop_j[kk];
	       if(loc_col > -1)
	       { /* In S_diag */
		 if(CF_marker[loc_col] > 0)
		 {
		   if(P_marker[loc_col] < P_diag_i[ii])
		   {
		     P_marker[loc_col] = jj_counter;
		     jj_counter++;
		   }
		 }
	       }
	       else
	       {
		 loc_col = - loc_col - 1;
		 if(CF_marker_offd[loc_col] > 0)
		 {
		   if(P_marker_offd[loc_col] < P_offd_i[ii])
		   {
		     P_marker_offd[loc_col] = jj_counter_offd;
		     tmp_CF_marker_offd[loc_col] = 1;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       } 
     }
   }
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     determine structure    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
                                                                                               
                                                                                               
   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;
   
   if (P_diag_size)
   {
      P_diag_j    = hypre_CTAlloc(int, P_diag_size);
      P_diag_data = hypre_CTAlloc(double, P_diag_size);
   }

   if (P_offd_size)
   {
      P_offd_j    = hypre_CTAlloc(int, P_offd_size);
      P_offd_data = hypre_CTAlloc(double, P_offd_size);
   }

   P_diag_i[n_coarse_old] = jj_counter; 
   P_offd_i[n_coarse_old] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /* Fine to coarse mapping */
   if(num_procs > 1)
   {
     big_insert_new_nodes(comm_pkg, extend_comm_pkg, fine_to_coarse, 
		      full_off_procNodes, my_first_cpt,
		      fine_to_coarse_offd);
   }

   /* Initialize ahat, which is a modification to a, used in the standard
    * interpolation routine. */
   if (n_fine)
   {
      ahat = hypre_CTAlloc(double, n_fine);
      ihat = hypre_CTAlloc(int, n_fine);
      ipnt = hypre_CTAlloc(int, n_fine);
   }
   if (full_off_procNodes)
   {
     ahat_offd = hypre_CTAlloc(double, full_off_procNodes);
     ihat_offd = hypre_CTAlloc(int, full_off_procNodes);
     ipnt_offd = hypre_CTAlloc(int, full_off_procNodes);
   }

   for (i = 0; i < n_fine; i++)
   {      
     P_marker[i] = -1;
     ahat[i] = 0;
     ihat[i] = -1;
   }
   for (i = 0; i < full_off_procNodes; i++)
   {      
     P_marker_offd[i] = -1;
     ahat_offd[i] = 0;
     ihat_offd[i] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   for (ii = 0; ii < n_coarse_old; ii++)
   {
     jj_begin_row = jj_counter;        
     jj_begin_row_offd = jj_counter_offd;
     i = old_coarse_to_fine[ii];

     /*--------------------------------------------------------------------
      *  If i is a c-point, interpolation is the identity.
      *--------------------------------------------------------------------*/
     
     if (CF_marker[i] > 0)
     {
       P_diag_j[jj_counter]    = fine_to_coarse[i];
       P_diag_data[jj_counter] = one;
       jj_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, build interpolation.
      *--------------------------------------------------------------------*/
     
     else if (CF_marker[i] == -2)
     {         
   if (debug_flag==4) wall_time = time_getWallclockSeconds();
       strong_f_marker--;
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       {
	 i1 = S_diag_j[jj];   
	 
	 /*--------------------------------------------------------------
	  * If neighbor i1 is a C-point, set column number in P_diag_j
	  * and initialize interpolation weight to zero.
	  *--------------------------------------------------------------*/
	 
	 if (CF_marker[i1] > 0)
	 {
	   if (P_marker[i1] < jj_begin_row)
	   {
	     P_marker[i1] = jj_counter;
	     P_diag_j[jj_counter]    = i1;
	     P_diag_data[jj_counter] = zero;
	     jj_counter++;
	   }
	 }
	 else  if (CF_marker[i1] != -3)
	 {
	   P_marker[i1] = strong_f_marker;
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] > 0)
	     {
	       if(P_marker[k1] < jj_begin_row)
	       {
		 P_marker[k1] = jj_counter;
		 P_diag_j[jj_counter] = k1;
		 P_diag_data[jj_counter] = zero;
		 jj_counter++;
	       }
	     }
	   }
	   if(num_procs > 1)
	   {
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];
	       if(CF_marker_offd[k1] > 0)
	       {
		 if(P_marker_offd[k1] < jj_begin_row_offd)
		 {
		   P_marker_offd[k1] = jj_counter_offd;
		   P_offd_j[jj_counter_offd] = k1;
		   P_offd_data[jj_counter_offd] = zero;
		   jj_counter_offd++;
		 }
	       }
	     }
	   }
	 }
       }
       
       if ( num_procs > 1)
       {
	 for (jj=S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   i1 = S_offd_j[jj];
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[i1];
	   if ( CF_marker_offd[i1] > 0)
	   {
	     if(P_marker_offd[i1] < jj_begin_row_offd)
	     {
	       P_marker_offd[i1] = jj_counter_offd;
	       P_offd_j[jj_counter_offd]=i1;
	       P_offd_data[jj_counter_offd] = zero;
	       jj_counter_offd++;
	     }
	   }
	   else if (CF_marker_offd[i1] != -3)
	   {
	     P_marker_offd[i1] = strong_f_marker;
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     {
	       loc_col = Sop_j[kk];
	       if(loc_col > -1)
	       {
		 if(CF_marker[loc_col] > 0)
		 {
		   if(P_marker[loc_col] < jj_begin_row)
		   {		
		     P_marker[loc_col] = jj_counter;
		     P_diag_j[jj_counter] = loc_col;
		     P_diag_data[jj_counter] = zero;
		     jj_counter++;
		   }
		 }
	       }
	       else
	       {
		 loc_col = -loc_col - 1;
		 if(CF_marker_offd[loc_col] > 0)
		 {
		   if(P_marker_offd[loc_col] < jj_begin_row_offd)
		   {
		     P_marker_offd[loc_col] = jj_counter_offd;
		     P_offd_j[jj_counter_offd]=loc_col;
		     P_offd_data[jj_counter_offd] = zero;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       }
       
       jj_end_row = jj_counter;
       jj_end_row_offd = jj_counter_offd;
       
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      wall_1 += wall_time;
      fflush(NULL);
   }
   if (debug_flag==4) wall_time = time_getWallclockSeconds();
       cnt_c = 0;
       cnt_f = jj_end_row-jj_begin_row;
       cnt_c_offd = 0;
       cnt_f_offd = jj_end_row_offd-jj_begin_row_offd;
       ihat[i] = cnt_f;
       ipnt[cnt_f] = i;
       ahat[cnt_f++] = A_diag_data[A_diag_i[i]];
       for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
       { /* i1 is direct neighbor */
	 i1 = A_diag_j[jj];
	 if (P_marker[i1] != strong_f_marker)
	 {
	   indx = ihat[i1];
	   if (indx > -1)
	      ahat[indx] += A_diag_data[jj];
	   else if (P_marker[i1] >= jj_begin_row)
           {
               ihat[i1] = cnt_c;
               ipnt[cnt_c] = i1;
               ahat[cnt_c++] += A_diag_data[jj];
           }
	   else if (CF_marker[i1] != -3)
           {
               ihat[i1] = cnt_f;
               ipnt[cnt_f] = i1;
               ahat[cnt_f++] += A_diag_data[jj];
           }
	 }
	 else 
	 {
	   if(num_functions == 1 || dof_func[i] == dof_func[i1])
	   {
	     distribute = A_diag_data[jj]/A_diag_data[A_diag_i[i1]];
	     for (kk = A_diag_i[i1]+1; kk < A_diag_i[i1+1]; kk++)
	     {
	       k1 = A_diag_j[kk];
	       indx = ihat[k1];
               if (indx > -1) 
	          ahat[indx] -= A_diag_data[kk]*distribute;
               else if (P_marker[k1] >= jj_begin_row)
               {
                  ihat[k1] = cnt_c;
                  ipnt[cnt_c] = k1;
                  ahat[cnt_c++] -= A_diag_data[kk]*distribute;
               }
               else
               {
                  ihat[k1] = cnt_f;
                  ipnt[cnt_f] = k1;
                  ahat[cnt_f++] -= A_diag_data[kk]*distribute;
               }
	     }
	     if(num_procs > 1)
	     {
	       for (kk = A_offd_i[i1]; kk < A_offd_i[i1+1]; kk++)
	       {
		 k1 = A_offd_j[kk];
	         indx = ihat_offd[k1];
		 if(num_functions == 1 || dof_func[i1] == dof_func_offd[k1])
		 {
		   if (indx > -1)
		      ahat_offd[indx] -= A_offd_data[kk]*distribute;
                   else if (P_marker_offd[k1] >= jj_begin_row_offd)
                   {
                      ihat_offd[k1] = cnt_c_offd;
                      ipnt_offd[cnt_c_offd] = k1;
                      ahat_offd[cnt_c_offd++] -= A_offd_data[kk]*distribute;
                   }
                   else
                   {
                      ihat_offd[k1] = cnt_f_offd;
                      ipnt_offd[cnt_f_offd] = k1;
                      ahat_offd[cnt_f_offd++] -= A_offd_data[kk]*distribute;
                   }
		 }
	       }
	     }
	   }
	 }
       }
       if(num_procs > 1)
       {
	 for(jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
	 {
	   i1 = A_offd_j[jj];
	   if(P_marker_offd[i1] != strong_f_marker)
	   {
	     indx = ihat_offd[i1];
	     if (indx > -1)
	        ahat_offd[indx] += A_offd_data[jj];
	     else if (P_marker_offd[i1] >= jj_begin_row_offd)
	     {
	        ihat_offd[i1] = cnt_c_offd;
	        ipnt_offd[cnt_c_offd] = i1;
	        ahat_offd[cnt_c_offd++] += A_offd_data[jj];
	     }
	     else if (CF_marker_offd[i1] != -3)
	     {
	        ihat_offd[i1] = cnt_f_offd;
	        ipnt_offd[cnt_f_offd] = i1;
	        ahat_offd[cnt_f_offd++] += A_offd_data[jj];
	     }
	   }
	   else
	   {
	     if(num_functions == 1 || dof_func[i] == dof_func_offd[i1])
	     {
	       distribute = A_offd_data[jj]/A_ext_data[A_ext_i[i1]];
	       for (kk = A_ext_i[i1]+1; kk < A_ext_i[i1+1]; kk++)
	       {
		 loc_col = A_ext_j[kk];
		 if(loc_col > -1)
		 { /*diag*/
		   indx = ihat[loc_col];
		   if (indx > -1)
		      ahat[indx] -= A_ext_data[kk]*distribute;
		   else if (P_marker[loc_col] >= jj_begin_row)
		   {
		      ihat[loc_col] = cnt_c;
		      ipnt[cnt_c] = loc_col;
		      ahat[cnt_c++] -= A_ext_data[kk]*distribute;
		   }
		   else 
		   {
		      ihat[loc_col] = cnt_f;
		      ipnt[cnt_f] = loc_col;
		      ahat[cnt_f++] -= A_ext_data[kk]*distribute;
		   }
		 }
		 else
		 {
		   loc_col = - loc_col - 1;
		   if(num_functions == 1 || 
		      dof_func_offd[loc_col] == dof_func_offd[i1])
		   {
		     indx = ihat_offd[loc_col];
		     if (indx > -1)
		        ahat_offd[indx] -= A_ext_data[kk]*distribute;
		     else if(P_marker_offd[loc_col] >= jj_begin_row_offd)
		     {
		       ihat_offd[loc_col] = cnt_c_offd;
		       ipnt_offd[cnt_c_offd] = loc_col;
		       ahat_offd[cnt_c_offd++] -= A_ext_data[kk]*distribute;
		     }
		     else
		     {
		       ihat_offd[loc_col] = cnt_f_offd;
		       ipnt_offd[cnt_f_offd] = loc_col;
		       ahat_offd[cnt_f_offd++] -= A_ext_data[kk]*distribute;
		     }
		   }
		 }
	       }
	     }
	   }
	 }
       }
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      wall_2 += wall_time;
      fflush(NULL);
   }

   if (debug_flag==4) wall_time = time_getWallclockSeconds();
       diagonal = ahat[cnt_c];
       ahat[cnt_c] = 0;
       sum_pos = 0;
       sum_pos_C = 0;
       sum_neg = 0;
       sum_neg_C = 0;
       sum = 0;
       sum_C = 0;
       if(sep_weight == 1)
       {
	 for (jj=0; jj < cnt_c; jj++)
	 {
	   if (ahat[jj] > 0)
	   {
	     sum_pos_C += ahat[jj];
	   }
	   else 
	   {
	     sum_neg_C += ahat[jj];
	   }
	 }
	 if(num_procs > 1)
	 {
	   for (jj=0; jj < cnt_c_offd; jj++)
	   {
	     if (ahat_offd[jj] > 0)
	     {
	       sum_pos_C += ahat_offd[jj];
	     }
	     else 
	     {
	       sum_neg_C += ahat_offd[jj];
	     }
	   }
	 }
	 sum_pos = sum_pos_C;
	 sum_neg = sum_neg_C;
	 for (jj=cnt_c+1; jj < cnt_f; jj++)
	 {
	   if (ahat[jj] > 0)
	   {
	     sum_pos += ahat[jj];
	   }
	   else 
	   {
	     sum_neg += ahat[jj];
	   }
	   ahat[jj] = 0;
	 }
	 if(num_procs > 1)
	 {
	   for (jj=cnt_c_offd; jj < cnt_f_offd; jj++)
	   {
	     if (ahat_offd[jj] > 0)
	     {
	       sum_pos += ahat_offd[jj];
	     }
	     else 
	     {
	       sum_neg += ahat_offd[jj];
	     }
	     ahat_offd[jj] = 0;
	   }
	 }
	 if (sum_neg_C) alfa = sum_neg/sum_neg_C/diagonal;
	 if (sum_pos_C) beta = sum_pos/sum_pos_C/diagonal;
       
	 /*-----------------------------------------------------------------
	  * Set interpolation weight by dividing by the diagonal.
	  *-----------------------------------------------------------------*/
       
	 for (jj = jj_begin_row; jj < jj_end_row; jj++)
	 {
	   j1 = ihat[P_diag_j[jj]];
	   if (ahat[j1] > 0)
	     P_diag_data[jj] = -beta*ahat[j1];
	   else 
	     P_diag_data[jj] = -alfa*ahat[j1];
  
	   P_diag_j[jj] = fine_to_coarse[P_diag_j[jj]];
	   ahat[j1] = 0;
	 }
	 for (jj=0; jj < cnt_f; jj++)
	    ihat[ipnt[jj]] = -1;
	 if(num_procs > 1)
	 {
	   for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
	   {
	     j1 = ihat_offd[P_offd_j[jj]];
	     if (ahat_offd[j1] > 0)
	       P_offd_data[jj] = -beta*ahat_offd[j1];
	     else 
	       P_offd_data[jj] = -alfa*ahat_offd[j1];
	   
	     ahat_offd[j1] = 0;
	   }
	   for (jj=0; jj < cnt_f_offd; jj++)
	      ihat_offd[ipnt_offd[jj]] = -1;
	 }
       }
       else
       {
	 for (jj=0; jj < cnt_c; jj++)
	 {
	     sum_C += ahat[jj];
	 }
	 if(num_procs > 1)
	 {
	    for (jj=0; jj < cnt_c_offd; jj++)
	    {
	        sum_C += ahat_offd[jj];
	    }
	 }
	 sum = sum_C;
	 for (jj=cnt_c+1; jj < cnt_f; jj++)
	 {
	     sum += ahat[jj];
	     ahat[jj] = 0;
	 }
	 if(num_procs > 1)
	 {
	    for (jj=cnt_c_offd; jj < cnt_f_offd; jj++)
	    {
	       sum += ahat_offd[jj];
	       ahat_offd[jj] = 0;
	    }
	 }
	 if (sum_C) alfa = sum/sum_C/diagonal;
       
	 /*-----------------------------------------------------------------
	  * Set interpolation weight by dividing by the diagonal.
	  *-----------------------------------------------------------------*/
       
	 for (jj = jj_begin_row; jj < jj_end_row; jj++)
	 {
	   j1 = ihat[P_diag_j[jj]];
	   P_diag_data[jj] = -alfa*ahat[j1];
	   P_diag_j[jj] = fine_to_coarse[P_diag_j[jj]];
	   ahat[j1] = 0;
	 }
	 for (jj=0; jj < cnt_f; jj++)
	    ihat[ipnt[jj]] = -1;
	 if(num_procs > 1)
	 {
	   for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
	   {
	     j1 = ihat_offd[P_offd_j[jj]];
	     P_offd_data[jj] = -alfa*ahat_offd[j1];
	     ahat_offd[j1] = 0;
	   }
	   for (jj=0; jj < cnt_f_offd; jj++)
	      ihat_offd[ipnt_offd[jj]] = -1;
	 }
       }
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      wall_3 += wall_time;
      fflush(NULL);
   }
     }
   }
   
   if (debug_flag==4)
   {
      printf("Proc = %d fill part 1 %f part 2 %f  part 3 %f\n",
                    my_id, wall_1, wall_2, wall_3);
      fflush(NULL);
   }
   P = hypre_ParCSRMatrixCreate(comm,
				total_old_global_cpts,
				total_global_cpts,
				num_old_cpts_global,
				num_cpts_global,
				0,
				P_diag_i[n_coarse_old],
				P_offd_i[n_coarse_old]);
               
   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_coarse_old];
      P_offd_size = P_offd_i[n_coarse_old];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   num_cols_P_offd = 0;
   if(P_offd_size)
   {
     num_cols_P_offd = 0;
     for (i=0; i < P_offd_size; i++)
     {
       index = P_offd_j[i];
       if(tmp_CF_marker_offd[index] == 1)
       {
	   num_cols_P_offd++;
	   tmp_CF_marker_offd[index] = 2;
       }
     }
     
     if (num_cols_P_offd)
     {     
        col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd);
        tmp_map_offd = hypre_CTAlloc(int, num_cols_P_offd);
     }     
     
     index = 0;
     for(i = 0; i < num_cols_P_offd; i++)
     {
       while( tmp_CF_marker_offd[index] == -1) index++;
       col_map_offd_P[i] = fine_to_coarse_offd[index];
       tmp_map_offd[i] = index++;
     }

     hypre_BigQsortbi(col_map_offd_P, tmp_map_offd, 0, num_cols_P_offd-1);

     for (i = 0; i < num_cols_P_offd; i++)
        tmp_CF_marker_offd[tmp_map_offd[i]] = i;

     for(i = 0; i < P_offd_size; i++)
       P_offd_j[i] = tmp_CF_marker_offd[P_offd_j[i]];
     /*index = 0;
     for(i = 0; i < num_cols_P_offd; i++)
     {
       while (P_marker[index] == 0) index++;
       
       col_map_offd_P[i] = fine_to_coarse_offd[index];
       index++;
     }*/

     /* Sort the col_map_offd_P and P_offd_j correctly */
     /* Check if sort actually changed anything */
     /*for(i = 0; i < num_cols_P_offd; i++)
       P_marker[i] = col_map_offd_P[i];

     if(hypre_ssort(col_map_offd_P,num_cols_P_offd))
     {
       for(i = 0; i < P_offd_size; i++)
	 for(j = 0; j < num_cols_P_offd; j++)
	   if(P_marker[P_offd_j[i]] == col_map_offd_P[j])
	   {
	     P_offd_j[i] = j;
	     j = num_cols_P_offd;
	   }
     }
     hypre_TFree(P_marker); */
   }

   if (num_cols_P_offd)
   { 
     hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
     hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
     hypre_TFree(tmp_map_offd);
   } 

#ifdef HYPRE_NO_GLOBAL_PARTITION
     hypre_NewCommPkgCreate(P);
#else
     hypre_MatvecCommPkgCreate(P);
#endif

   for (i=0; i < n_fine; i++)
      if (CF_marker[i] < -1) CF_marker[i] = -1;
 
   *P_ptr = P;

   /* Deallocate memory */   
   hypre_TFree(fine_to_coarse);
   hypre_TFree(old_coarse_to_fine);
   hypre_TFree(P_marker);
   hypre_TFree(ahat);
   hypre_TFree(ihat);
   hypre_TFree(ipnt);

   if (full_off_procNodes)
   {
     hypre_TFree(ahat_offd);
     hypre_TFree(ihat_offd);
     hypre_TFree(ipnt_offd);
   }
   if (num_procs > 1) 
   {
     hypre_BigCSRMatrixDestroy(Sop);
     hypre_BigCSRMatrixDestroy(A_ext);
     hypre_TFree(fine_to_coarse_offd);
     hypre_TFree(P_marker_offd);
     hypre_TFree(CF_marker_offd);
     hypre_TFree(tmp_CF_marker_offd);

     if(num_functions > 1)
       hypre_TFree(dof_func_offd);
     hypre_TFree(found);

     hypre_MatvecCommPkgDestroy(extend_comm_pkg);

   }
   
 
   return hypre_error_flag;  
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildPartialExtPIInterp
 *  Comment: 
 *--------------------------------------------------------------------------*/
int
hypre_BoomerAMGBuildPartialExtPIInterp(hypre_ParCSRMatrix *A, int *CF_marker,
			      hypre_ParCSRMatrix   *S, HYPRE_BigInt *num_cpts_global,
			      HYPRE_BigInt *num_old_cpts_global,
			      int num_functions, int *dof_func, int debug_flag,
			      double trunc_factor, int max_elmts, 
			      int *col_offd_S_to_A,
			      hypre_ParCSRMatrix  **P_ptr)
{
  /* Communication Variables */
  MPI_Comm 	           comm = hypre_ParCSRMatrixComm(A);   
  hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);


  int              my_id, num_procs;

  /* Variables to store input variables */
  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A); 
  double          *A_diag_data = hypre_CSRMatrixData(A_diag);
  int             *A_diag_i = hypre_CSRMatrixI(A_diag);
  int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

  hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
  double          *A_offd_data = hypre_CSRMatrixData(A_offd);
  int             *A_offd_i = hypre_CSRMatrixI(A_offd);
  int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

  int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
  HYPRE_BigInt    *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
  int              n_fine = hypre_CSRMatrixNumRows(A_diag);
  HYPRE_BigInt     col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
  int              local_numrows = hypre_CSRMatrixNumRows(A_diag);
  HYPRE_BigInt     col_n = col_1 + (HYPRE_BigInt) local_numrows;
  HYPRE_BigInt     total_global_cpts, my_first_cpt;
  HYPRE_BigInt     total_old_global_cpts, my_first_old_cpt;
  int              n_coarse, n_coarse_old;

  /* Variables to store strong connection matrix info */
  hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
  int             *S_diag_i = hypre_CSRMatrixI(S_diag);
  int             *S_diag_j = hypre_CSRMatrixJ(S_diag);
   
  hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
  int             *S_offd_i = hypre_CSRMatrixI(S_offd);
  int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

  /* Interpolation matrix P */
  hypre_ParCSRMatrix *P;
  hypre_CSRMatrix    *P_diag;
  hypre_CSRMatrix    *P_offd;   

  double          *P_diag_data = NULL;
  int             *P_diag_i, *P_diag_j = NULL;
  double          *P_offd_data = NULL;
  int             *P_offd_i, *P_offd_j = NULL;

  HYPRE_BigInt	  *col_map_offd_P;
  int	          *tmp_map_offd;
  int              P_diag_size; 
  int              P_offd_size;
  int             *P_marker = NULL; 
  int             *P_marker_offd = NULL;
  int             *CF_marker_offd = NULL;
  int             *tmp_CF_marker_offd = NULL;
  int             *dof_func_offd = NULL;

  /* Full row information for columns of A that are off diag*/
  hypre_BigCSRMatrix *A_ext;   
  double          *A_ext_data;
  int             *A_ext_i;
  HYPRE_BigInt    *big_A_ext_j;
  int             *A_ext_j;
  
  int             *fine_to_coarse = NULL;
  int             *old_coarse_to_fine = NULL;
  HYPRE_BigInt    *fine_to_coarse_offd = NULL;
  HYPRE_BigInt    *found;

  int              num_cols_P_offd;
  int              newoff, loc_col;
  int              A_ext_rows, full_off_procNodes;
  
  hypre_BigCSRMatrix *Sop;
  int             *Sop_i;
  HYPRE_BigInt    *big_Sop_j;
  int             *Sop_j;
  
  int              Soprows, sgn;
  
  /* Variables to keep count of interpolatory points */
  int              jj_counter, jj_counter_offd;
  int              jj_begin_row, jj_end_row;
  int              jj_begin_row_offd = 0;
  int              jj_end_row_offd = 0;
  int              coarse_counter, coarse_counter_offd;
    
  /* Interpolation weight variables */
  double           sum, diagonal, distribute;
  int              strong_f_marker = -2;
 
  /* Loop variables */
  int              index, ii, cnt, old_cnt;
  int              start_indexing = 0;
  int              i, i1, i2, jj, kk, k1, jj1;

  /* Definitions */
  double           zero = 0.0;
  double           one  = 1.0;
  double           wall_time;
  

  hypre_ParCSRCommPkg	*extend_comm_pkg = NULL;

   if (debug_flag==4) wall_time = time_getWallclockSeconds();
                                                                                               
  /* BEGIN */
  MPI_Comm_size(comm, &num_procs);   
  MPI_Comm_rank(comm,&my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   my_first_old_cpt = num_old_cpts_global[0];
   n_coarse_old = (int)(num_old_cpts_global[1] - num_old_cpts_global[0]);
   n_coarse = (int)(num_cpts_global[1] - num_cpts_global[0]);
   if (my_id == (num_procs -1))
   {
      total_global_cpts = num_cpts_global[1];
      total_old_global_cpts = num_old_cpts_global[1];
   }
   MPI_Bcast(&total_global_cpts, 1, MPI_HYPRE_BIG_INT, num_procs-1, comm);
   MPI_Bcast(&total_old_global_cpts, 1, MPI_HYPRE_BIG_INT, num_procs-1, comm);
#else
   my_first_cpt = num_cpts_global[my_id];
   my_first_old_cpt = num_old_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
   total_old_global_cpts = num_old_cpts_global[num_procs];
   n_coarse_old = (int)(num_old_cpts_global[my_id+1] - num_old_cpts_global[my_id]);
   n_coarse = (int)(num_cpts_global[my_id+1] - num_cpts_global[my_id]);
#endif

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
     hypre_NewCommPkgCreate(A);
#else
     hypre_MatvecCommPkgCreate(A);
#endif
     comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   /* Set up off processor information (specifically for neighbors of 
    * neighbors */
   newoff = 0;
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
     /*----------------------------------------------------------------------
      * Get the off processors rows for A and S, associated with columns in 
      * A_offd and S_offd.
      *---------------------------------------------------------------------*/
     A_ext         = hypre_ParCSRMatrixExtractBigExt(A,A,1);
     A_ext_i       = hypre_BigCSRMatrixI(A_ext);
     big_A_ext_j   = hypre_BigCSRMatrixJ(A_ext);
     A_ext_data    = hypre_BigCSRMatrixData(A_ext);
     A_ext_rows    = hypre_BigCSRMatrixNumRows(A_ext);
     
     Sop           = hypre_ParCSRMatrixExtractBigExt(S,A,0);
     Sop_i         = hypre_BigCSRMatrixI(Sop);
     big_Sop_j     = hypre_BigCSRMatrixJ(Sop);
     Soprows       = hypre_BigCSRMatrixNumRows(Sop);

     /* Find nodes that are neighbors of neighbors, not found in offd */
     newoff = new_offd_nodes(&found, A_ext_rows, A_ext_i, big_A_ext_j,
			     &A_ext_j, 
			     Soprows, col_map_offd, col_1, col_n, 
			     Sop_i, big_Sop_j, &Sop_j, CF_marker, comm_pkg);

     hypre_BigCSRMatrixJ(A_ext) = NULL;
     hypre_BigCSRMatrixJ(Sop) = NULL;

     if(newoff >= 0)
       full_off_procNodes = newoff + num_cols_A_offd;
     else
       return(1);

     /* Possibly add new points and new processors to the comm_pkg, all
      * processors need new_comm_pkg */
    
     /* AHB - create a new comm package just for extended info -
        this will work better with the assumed partition*/
     hypre_ParCSRFindExtendCommPkg(A, newoff, found, 
                                   &extend_comm_pkg);
     
     CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
     
     if (num_functions > 1 && full_off_procNodes > 0)
        dof_func_offd = hypre_CTAlloc(int, full_off_procNodes);
     
     alt_insert_new_nodes(comm_pkg, extend_comm_pkg, CF_marker, 
                          full_off_procNodes, CF_marker_offd);
     
     if(num_functions > 1)
        alt_insert_new_nodes(comm_pkg, extend_comm_pkg, dof_func, 
                             full_off_procNodes, dof_func_offd);
   }
  

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(int, n_fine+1);
   P_offd_i    = hypre_CTAlloc(int, n_fine+1);

   if (n_coarse_old)  old_coarse_to_fine = hypre_CTAlloc(int, n_coarse_old);
   if (n_fine)
   {
      fine_to_coarse = hypre_CTAlloc(int, n_fine);
      P_marker = hypre_CTAlloc(int, n_fine);
   }

   if (full_off_procNodes)
   {
      P_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, full_off_procNodes);
      tmp_CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
   }

   for (i=0; i < full_off_procNodes; i++)
   {
      P_marker_offd[i] = -1;
      tmp_CF_marker_offd[i] = -1;
      fine_to_coarse_offd[i] = -1;
   }

   for (i=0; i < n_fine; i++)
   {
      P_marker[i] = -1;
      fine_to_coarse[i] = -1;
   }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;
   coarse_counter_offd = 0;

   cnt = 0;
   old_cnt = 0;
   for (i = 0; i < n_fine; i++)
   {
      fine_to_coarse[i] = -1;
      if (CF_marker[i] == 1)
      {
         fine_to_coarse[i] = cnt++;
         old_coarse_to_fine[old_cnt++] = i;
      }
      else if (CF_marker[i] == -2)
      {
         old_coarse_to_fine[old_cnt++] = i;
      }
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (ii = 0; ii < n_coarse_old; ii++)
   {
     P_diag_i[ii] = jj_counter;
     if (num_procs > 1)
       P_offd_i[ii] = jj_counter_offd;
     
     i = old_coarse_to_fine[ii];
     if (CF_marker[i] >= 0)
     {
       jj_counter++;
       coarse_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, interpolation is from the C-points that
      *  strongly influence i, or C-points that stronly influence F-points
      *  that strongly influence i.
      *--------------------------------------------------------------------*/
     else if (CF_marker[i] == -2)
     {
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       {
	 i1 = S_diag_j[jj];           
	 if (CF_marker[i1] > 0)
	 { /* i1 is a C point */
	   if (P_marker[i1] < P_diag_i[ii])
	   {
	     P_marker[i1] = jj_counter;
	     jj_counter++;
	   }
	 }
	 else if (CF_marker[i1] != -3)
	 { /* i1 is a F point, loop through it's strong neighbors */
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] > 0)
	     {
	       if(P_marker[k1] < P_diag_i[ii])
	       {
		 P_marker[k1] = jj_counter;
		 jj_counter++;
	       }
	     } 
	   }
	   if(num_procs > 1)
	   {
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];
	       if (CF_marker_offd[k1] > 0)
	       {
		 if(P_marker_offd[k1] < P_offd_i[ii])
		 {
		   tmp_CF_marker_offd[k1] = 1;
		   P_marker_offd[k1] = jj_counter_offd;
		   jj_counter_offd++;
		 }
	       }
	     }
	   }
	 }
       }
       /* Look at off diag strong connections of i */ 
       if (num_procs > 1)
       {
	 for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   i1 = S_offd_j[jj];           
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[i1];
	   if (CF_marker_offd[i1] > 0)
	   {
	     if(P_marker_offd[i1] < P_offd_i[ii])
	     {
	       tmp_CF_marker_offd[i1] = 1;
	       P_marker_offd[i1] = jj_counter_offd;
	       jj_counter_offd++;
	     }
	   }
	   else if (CF_marker_offd[i1] != -3)
	   { /* F point; look at neighbors of i1. Sop contains global col
	      * numbers and entries that could be in S_diag or S_offd or
	      * neither. */
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     {
	       loc_col = Sop_j[kk];
	       if(loc_col > -1)
	       { /* In S_diag */
		 if(CF_marker[loc_col] > 0)
		 {
		   if(P_marker[loc_col] < P_diag_i[ii])
		   {
		     P_marker[loc_col] = jj_counter;
		     jj_counter++;
		   }
		 }
	       }
	       else
	       {
		 loc_col = -loc_col - 1; 
		 if(CF_marker_offd[loc_col] > 0)
		 {
		   if(P_marker_offd[loc_col] < P_offd_i[ii])
		   {
		     P_marker_offd[loc_col] = jj_counter_offd;
		     tmp_CF_marker_offd[loc_col] = 1;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       } 
     }
   }
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     determine structure    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
                                                                                               
  if (debug_flag== 4) wall_time = time_getWallclockSeconds();

   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;

   if (P_diag_size)
   {   
      P_diag_j    = hypre_CTAlloc(int, P_diag_size);
      P_diag_data = hypre_CTAlloc(double, P_diag_size);
   }

   if (P_offd_size)
   {   
      P_offd_j    = hypre_CTAlloc(int, P_offd_size);
      P_offd_data = hypre_CTAlloc(double, P_offd_size);
   }

   P_diag_i[n_coarse_old] = jj_counter; 
   P_offd_i[n_coarse_old] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /* Fine to coarse mapping */
   if(num_procs > 1)
   {
     big_insert_new_nodes(comm_pkg, extend_comm_pkg, fine_to_coarse, 
		      full_off_procNodes, my_first_cpt,
		      fine_to_coarse_offd);
   }

   for (i = 0; i < n_fine; i++)     
     P_marker[i] = -1;
     
   for (i = 0; i < full_off_procNodes; i++)
     P_marker_offd[i] = -1;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   for (ii = 0; ii < n_coarse_old; ii++)
   {
     jj_begin_row = jj_counter;        
     jj_begin_row_offd = jj_counter_offd;
     i = old_coarse_to_fine[ii];

     /*--------------------------------------------------------------------
      *  If i is a c-point, interpolation is the identity.
      *--------------------------------------------------------------------*/
     
     if (CF_marker[i] > 0)
     {
       P_diag_j[jj_counter]    = fine_to_coarse[i];
       P_diag_data[jj_counter] = one;
       jj_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, build interpolation.
      *--------------------------------------------------------------------*/
     
     else if (CF_marker[i] == -2)
     {
       strong_f_marker--;
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       {
	 i1 = S_diag_j[jj];   
	 
	 /*--------------------------------------------------------------
	  * If neighbor i1 is a C-point, set column number in P_diag_j
	  * and initialize interpolation weight to zero.
	  *--------------------------------------------------------------*/
	 
	 if (CF_marker[i1] > 0)
	 {
	   if (P_marker[i1] < jj_begin_row)
	   {
	     P_marker[i1] = jj_counter;
	     P_diag_j[jj_counter]    = fine_to_coarse[i1];
	     P_diag_data[jj_counter] = zero;
	     jj_counter++;
	   }
	 }
	 else  if (CF_marker[i1] != -3)
	 {
	   P_marker[i1] = strong_f_marker;
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] > 0)
	     {
	       if(P_marker[k1] < jj_begin_row)
	       {
		 P_marker[k1] = jj_counter;
		 P_diag_j[jj_counter] = fine_to_coarse[k1];
		 P_diag_data[jj_counter] = zero;
		 jj_counter++;
	       }
	     }
	   }
	   if(num_procs > 1)
	   {
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];
	       if(CF_marker_offd[k1] > 0)
	       {
		 if(P_marker_offd[k1] < jj_begin_row_offd)
		 {
		   P_marker_offd[k1] = jj_counter_offd;
		   P_offd_j[jj_counter_offd] = k1;
		   P_offd_data[jj_counter_offd] = zero;
		   jj_counter_offd++;
		 }
	       }
	     }
	   }
	 }
       }
       
       if ( num_procs > 1)
       {
	 for (jj=S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   i1 = S_offd_j[jj];
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[i1];
	   if ( CF_marker_offd[i1] > 0)
	   {
	     if(P_marker_offd[i1] < jj_begin_row_offd)
	     {
	       P_marker_offd[i1] = jj_counter_offd;
	       P_offd_j[jj_counter_offd] = i1;
	       P_offd_data[jj_counter_offd] = zero;
	       jj_counter_offd++;
	     }
	   }
	   else if (CF_marker_offd[i1] != -3)
	   {
	     P_marker_offd[i1] = strong_f_marker;
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     {
	       loc_col = Sop_j[kk];
	       if(loc_col > -1)
	       {
		 if(CF_marker[loc_col] > 0)
		 {
		   if(P_marker[loc_col] < jj_begin_row)
		   {		
		     P_marker[loc_col] = jj_counter;
		     P_diag_j[jj_counter] = fine_to_coarse[loc_col];
		     P_diag_data[jj_counter] = zero;
		     jj_counter++;
		   }
		 }
	       }
	       else
	       { 
		 loc_col = - loc_col - 1;
		 if(CF_marker_offd[loc_col] > 0)
		 {
		   if(P_marker_offd[loc_col] < jj_begin_row_offd)
		   {
		     P_marker_offd[loc_col] = jj_counter_offd;
		     P_offd_j[jj_counter_offd]=loc_col;
		     P_offd_data[jj_counter_offd] = zero;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       }
       
       jj_end_row = jj_counter;
       jj_end_row_offd = jj_counter_offd;
       
       diagonal = A_diag_data[A_diag_i[i]];
       
       for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
       { /* i1 is a c-point and strongly influences i, accumulate
	  * a_(i,i1) into interpolation weight */
	 i1 = A_diag_j[jj];
	 if (P_marker[i1] >= jj_begin_row)
	 {
	   P_diag_data[P_marker[i1]] += A_diag_data[jj];
	 }
	 else if(P_marker[i1] == strong_f_marker)
	 {
	   sum = zero;
           sgn = 1;
           if(A_diag_data[A_diag_i[i1]] < 0) sgn = -1;
	   /* Loop over row of A for point i1 and calculate the sum
            * of the connections to c-points that strongly incluence i. */
	   for(jj1 = A_diag_i[i1]+1; jj1 < A_diag_i[i1+1]; jj1++)
	   {
	     i2 = A_diag_j[jj1];
	     if((P_marker[i2] >= jj_begin_row || i2 == i) && (sgn*A_diag_data[jj1]) < 0)
	       sum += A_diag_data[jj1];
	   }
	   if(num_procs > 1)
	   {
	     for(jj1 = A_offd_i[i1]; jj1< A_offd_i[i1+1]; jj1++)
	     {
	       i2 = A_offd_j[jj1];
	       if(P_marker_offd[i2] >= jj_begin_row_offd &&
		 (sgn*A_offd_data[jj1]) < 0)
		 sum += A_offd_data[jj1];
	     }
	   }
	   if(sum != 0)
	   {
	     distribute = A_diag_data[jj]/sum;
	     /* Loop over row of A for point i1 and do the distribution */
	     for(jj1 = A_diag_i[i1]+1; jj1 < A_diag_i[i1+1]; jj1++)
	     {
	       i2 = A_diag_j[jj1];
	       if(P_marker[i2] >= jj_begin_row && (sgn*A_diag_data[jj1]) < 0)
		 P_diag_data[P_marker[i2]] += 
		   distribute*A_diag_data[jj1];
               if(i2 == i && (sgn*A_diag_data[jj1]) < 0)
                 diagonal += distribute*A_diag_data[jj1];
	     }
	     if(num_procs > 1)
	     {
	       for(jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
	       {
		 i2 = A_offd_j[jj1];
		 if(P_marker_offd[i2] >= jj_begin_row_offd &&
                    (sgn*A_offd_data[jj1]) < 0)
		   P_offd_data[P_marker_offd[i2]] +=
		     distribute*A_offd_data[jj1];
	       }
	     }
	   }
	   else
	   {
	     diagonal += A_diag_data[jj];
	   }
	 }
	 /* neighbor i1 weakly influences i, accumulate a_(i,i1) into
	  * diagonal */
	 else if (CF_marker[i1] != -3)
	 {
	   if(num_functions == 1 || dof_func[i] == dof_func[i1])
	     diagonal += A_diag_data[jj];
	 }
       }
       if(num_procs > 1)
       {
	 for(jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
	 {
	   i1 = A_offd_j[jj];
	   if(P_marker_offd[i1] >= jj_begin_row_offd)
	     P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
	   else if(P_marker_offd[i1] == strong_f_marker)
	   {
	     sum = zero;
             sgn = 1;
             if(A_ext_data[A_ext_i[i1]] < 0) sgn = -1;
	     for(jj1 = A_ext_i[i1]+1; jj1 < A_ext_i[i1+1]; jj1++)
	     {
	       loc_col = A_ext_j[jj1];
	       if(loc_col > -1)
	       { /* diag */
		 if((P_marker[loc_col] >= jj_begin_row || loc_col == i)
			&& (sgn*A_ext_data[jj1]) < 0)
		   sum += A_ext_data[jj1];
	       }
	       else
	       { 
		 loc_col = - loc_col - 1;
		 if(P_marker_offd[loc_col] >= jj_begin_row_offd &&
                    (sgn*A_ext_data[jj1]) < 0)
		   sum += A_ext_data[jj1];
	       }
	     }
	     if(sum != 0)
	     {
	       distribute = A_offd_data[jj] / sum;
	       for(jj1 = A_ext_i[i1]+1; jj1 < A_ext_i[i1+1]; jj1++)
	       {
		 loc_col = A_ext_j[jj1];
		 if(loc_col > -1)
		 { /* diag */
		   if(P_marker[loc_col] >= jj_begin_row &&
                      (sgn*A_ext_data[jj1]) < 0)
		     P_diag_data[P_marker[loc_col]] += distribute*
		      A_ext_data[jj1];
                   if(loc_col == i && (sgn*A_ext_data[jj1]) < 0)
                     diagonal += distribute*A_ext_data[jj1];
		 }
		 else
		 { 
		   loc_col = - loc_col - 1;
		   if(P_marker_offd[loc_col] >= jj_begin_row_offd &&
                      (sgn*A_ext_data[jj1]) < 0)
		     P_offd_data[P_marker_offd[loc_col]] += distribute*
		       A_ext_data[jj1];
		 }
	       }
	     }
	     else
	     {
	       diagonal += A_offd_data[jj];
	     }
	   }
	   else if (CF_marker_offd[i1] != -3)
	   {
	     if(num_functions == 1 || dof_func[i] == dof_func_offd[i1])
	       diagonal += A_offd_data[jj];
	   }
	 }
       }
       for(jj = jj_begin_row; jj < jj_end_row; jj++)
	 P_diag_data[jj] /= -diagonal;
       for(jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
	 P_offd_data[jj] /= -diagonal;
     }
     strong_f_marker--;
   }
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     fill structure    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
                                                                                               
   P = hypre_ParCSRMatrixCreate(comm,
				total_old_global_cpts,
				total_global_cpts,
				num_old_cpts_global,
				num_cpts_global,
				0,
				P_diag_i[n_coarse_old],
				P_offd_i[n_coarse_old]);
               
   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_coarse_old];
      P_offd_size = P_offd_i[n_coarse_old];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   num_cols_P_offd = 0;
   if(P_offd_size)
   {
     num_cols_P_offd = 0;
     for (i=0; i < P_offd_size; i++)
     {
       index = P_offd_j[i];
       if (tmp_CF_marker_offd[index] == 1)
       {
	   num_cols_P_offd++;
	   tmp_CF_marker_offd[index] = 2;
       }
     }

     if (num_cols_P_offd)
     {     
        col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd);
        tmp_map_offd = hypre_CTAlloc(int, num_cols_P_offd);
     }     
     
     index = 0;
     for(i = 0; i < num_cols_P_offd; i++)
     {
       while( tmp_CF_marker_offd[index] != 2) index++;
       /*printf("index %d tmp_CF %d\n", index, tmp_CF_marker_offd[index]);*/
       col_map_offd_P[i] = fine_to_coarse_offd[index];
       tmp_map_offd[i] = index++;
     }

     hypre_BigQsortbi(col_map_offd_P, tmp_map_offd, 0, num_cols_P_offd-1);

     for (i = 0; i < num_cols_P_offd; i++)
        tmp_CF_marker_offd[tmp_map_offd[i]] = i;

     for(i = 0; i < P_offd_size; i++)
       P_offd_j[i] = tmp_CF_marker_offd[P_offd_j[i]];
   }

   if (num_cols_P_offd)
   { 
     hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
     hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
     hypre_TFree(tmp_map_offd);
   } 

#ifdef HYPRE_NO_GLOBAL_PARTITION
     hypre_NewCommPkgCreate(P);
#else
     hypre_MatvecCommPkgCreate(P);
#endif
 
   for (i=0; i < n_fine; i++)
      if (CF_marker[i] < -1) CF_marker[i] = -1;
 
   *P_ptr = P;

   /* Deallocate memory */   
   hypre_TFree(fine_to_coarse);
   hypre_TFree(old_coarse_to_fine);
   hypre_TFree(P_marker);
   
   if (num_procs > 1) 
   {
     hypre_BigCSRMatrixDestroy(Sop);
     hypre_BigCSRMatrixDestroy(A_ext);
     hypre_TFree(fine_to_coarse_offd);
     hypre_TFree(P_marker_offd);
     hypre_TFree(CF_marker_offd);
     hypre_TFree(tmp_CF_marker_offd);
     if(num_functions > 1)
       hypre_TFree(dof_func_offd);
     hypre_TFree(found);


     hypre_MatvecCommPkgDestroy(extend_comm_pkg);
   

   }
   
   return hypre_error_flag;  
}

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBuildPartialExtInterp
 *  Comment: 
 *--------------------------------------------------------------------------*/
int
hypre_BoomerAMGBuildPartialExtInterp(hypre_ParCSRMatrix *A, int *CF_marker,
			      hypre_ParCSRMatrix   *S, HYPRE_BigInt *num_cpts_global,
			      HYPRE_BigInt *num_old_cpts_global,
			      int num_functions, int *dof_func, int debug_flag,
			      double trunc_factor, int max_elmts, 
			      int *col_offd_S_to_A,
			      hypre_ParCSRMatrix  **P_ptr)
{
  /* Communication Variables */
  MPI_Comm 	           comm = hypre_ParCSRMatrixComm(A);   
  hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);


  int              my_id, num_procs;

  /* Variables to store input variables */
  hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A); 
  double          *A_diag_data = hypre_CSRMatrixData(A_diag);
  int             *A_diag_i = hypre_CSRMatrixI(A_diag);
  int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

  hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);   
  double          *A_offd_data = hypre_CSRMatrixData(A_offd);
  int             *A_offd_i = hypre_CSRMatrixI(A_offd);
  int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

  int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
  HYPRE_BigInt    *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
  int              n_fine = hypre_CSRMatrixNumRows(A_diag);
  HYPRE_BigInt     col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
  int              local_numrows = hypre_CSRMatrixNumRows(A_diag);
  HYPRE_BigInt     col_n = col_1 + (HYPRE_BigInt) local_numrows;
  HYPRE_BigInt     total_global_cpts, my_first_cpt;
  HYPRE_BigInt     total_old_global_cpts, my_first_old_cpt;
  int              n_coarse, n_coarse_old;

  /* Variables to store strong connection matrix info */
  hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
  int             *S_diag_i = hypre_CSRMatrixI(S_diag);
  int             *S_diag_j = hypre_CSRMatrixJ(S_diag);
   
  hypre_CSRMatrix *S_offd = hypre_ParCSRMatrixOffd(S);   
  int             *S_offd_i = hypre_CSRMatrixI(S_offd);
  int             *S_offd_j = hypre_CSRMatrixJ(S_offd);

  /* Interpolation matrix P */
  hypre_ParCSRMatrix *P;
  hypre_CSRMatrix    *P_diag;
  hypre_CSRMatrix    *P_offd;   

  double          *P_diag_data = NULL;
  int             *P_diag_i, *P_diag_j = NULL;
  double          *P_offd_data = NULL;
  int             *P_offd_i, *P_offd_j = NULL;

  HYPRE_BigInt	  *col_map_offd_P;
  int	          *tmp_map_offd;
  int              P_diag_size; 
  int              P_offd_size;
  int             *P_marker = NULL; 
  int             *P_marker_offd = NULL;
  int             *CF_marker_offd = NULL;
  int             *tmp_CF_marker_offd = NULL;
  int             *dof_func_offd = NULL;

  /* Full row information for columns of A that are off diag*/
  hypre_BigCSRMatrix *A_ext;   
  double          *A_ext_data;
  int             *A_ext_i;
  HYPRE_BigInt    *big_A_ext_j;
  int             *A_ext_j;
  
  int             *fine_to_coarse = NULL;
  int             *old_coarse_to_fine = NULL;
  HYPRE_BigInt    *fine_to_coarse_offd = NULL;
  HYPRE_BigInt    *found;

  int              num_cols_P_offd;
  int              newoff, loc_col;
  int              A_ext_rows, full_off_procNodes;
  
  hypre_BigCSRMatrix *Sop;
  int             *Sop_i;
  HYPRE_BigInt    *big_Sop_j;
  int             *Sop_j;
  
  int              Soprows, sgn;
  
  /* Variables to keep count of interpolatory points */
  int              jj_counter, jj_counter_offd;
  int              jj_begin_row, jj_end_row;
  int              jj_begin_row_offd = 0;
  int              jj_end_row_offd = 0;
  int              coarse_counter, coarse_counter_offd;
    
  /* Interpolation weight variables */
  double           sum, diagonal, distribute;
  int              strong_f_marker = -2;
 
  /* Loop variables */
  int              index, ii, cnt, old_cnt;
  int              start_indexing = 0;
  int              i, i1, i2, jj, kk, k1, jj1;

  /* Definitions */
  double           zero = 0.0;
  double           one  = 1.0;
  double           wall_time;
  

  hypre_ParCSRCommPkg	*extend_comm_pkg = NULL;

   if (debug_flag==4) wall_time = time_getWallclockSeconds();
                                                                                               
  /* BEGIN */
  MPI_Comm_size(comm, &num_procs);   
  MPI_Comm_rank(comm,&my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   my_first_old_cpt = num_old_cpts_global[0];
   n_coarse_old = (int)(num_old_cpts_global[1] - num_old_cpts_global[0]);
   n_coarse = (int)(num_cpts_global[1] - num_cpts_global[0]);
   if (my_id == (num_procs -1))
   {
      total_global_cpts = num_cpts_global[1];
      total_old_global_cpts = num_old_cpts_global[1];
   }
   MPI_Bcast(&total_global_cpts, 1, MPI_HYPRE_BIG_INT, num_procs-1, comm);
   MPI_Bcast(&total_old_global_cpts, 1, MPI_HYPRE_BIG_INT, num_procs-1, comm);
#else
   my_first_cpt = num_cpts_global[my_id];
   my_first_old_cpt = num_old_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
   total_old_global_cpts = num_old_cpts_global[num_procs];
   n_coarse_old = (int)(num_old_cpts_global[my_id+1] - num_old_cpts_global[my_id]);
   n_coarse = (int)(num_cpts_global[my_id+1] - num_cpts_global[my_id]);
#endif

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
     hypre_NewCommPkgCreate(A);
#else
     hypre_MatvecCommPkgCreate(A);
#endif
     comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   /* Set up off processor information (specifically for neighbors of 
    * neighbors */
   newoff = 0;
   full_off_procNodes = 0;
   if (num_procs > 1)
   {
     /*----------------------------------------------------------------------
      * Get the off processors rows for A and S, associated with columns in 
      * A_offd and S_offd.
      *---------------------------------------------------------------------*/
     A_ext         = hypre_ParCSRMatrixExtractBigExt(A,A,1);
     A_ext_i       = hypre_BigCSRMatrixI(A_ext);
     big_A_ext_j   = hypre_BigCSRMatrixJ(A_ext);
     A_ext_data    = hypre_BigCSRMatrixData(A_ext);
     A_ext_rows    = hypre_BigCSRMatrixNumRows(A_ext);
     
     Sop           = hypre_ParCSRMatrixExtractBigExt(S,A,0);
     Sop_i         = hypre_BigCSRMatrixI(Sop);
     big_Sop_j     = hypre_BigCSRMatrixJ(Sop);
     Soprows       = hypre_BigCSRMatrixNumRows(Sop);

     /* Find nodes that are neighbors of neighbors, not found in offd */
     newoff = new_offd_nodes(&found, A_ext_rows, A_ext_i, big_A_ext_j,
			     &A_ext_j, 
			     Soprows, col_map_offd, col_1, col_n, 
			     Sop_i, big_Sop_j, &Sop_j, CF_marker, comm_pkg);

     hypre_BigCSRMatrixJ(A_ext) = NULL;
     hypre_BigCSRMatrixJ(Sop) = NULL;

     if(newoff >= 0)
       full_off_procNodes = newoff + num_cols_A_offd;
     else
       return(1);

     /* Possibly add new points and new processors to the comm_pkg, all
      * processors need new_comm_pkg */
    
     /* AHB - create a new comm package just for extended info -
        this will work better with the assumed partition*/
     hypre_ParCSRFindExtendCommPkg(A, newoff, found, 
                                   &extend_comm_pkg);
     
     CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
     
     if (num_functions > 1 && full_off_procNodes > 0)
        dof_func_offd = hypre_CTAlloc(int, full_off_procNodes);
     
     alt_insert_new_nodes(comm_pkg, extend_comm_pkg, CF_marker, 
                          full_off_procNodes, CF_marker_offd);
     
     if(num_functions > 1)
        alt_insert_new_nodes(comm_pkg, extend_comm_pkg, dof_func, 
                             full_off_procNodes, dof_func_offd);
   }
  

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/
   P_diag_i    = hypre_CTAlloc(int, n_fine+1);
   P_offd_i    = hypre_CTAlloc(int, n_fine+1);

   if (n_coarse_old)  old_coarse_to_fine = hypre_CTAlloc(int, n_coarse_old);
   if (n_fine)
   {
      fine_to_coarse = hypre_CTAlloc(int, n_fine);
      P_marker = hypre_CTAlloc(int, n_fine);
   }

   if (full_off_procNodes)
   {
      P_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, full_off_procNodes);
      tmp_CF_marker_offd = hypre_CTAlloc(int, full_off_procNodes);
   }

   for (i=0; i < full_off_procNodes; i++)
   {
      P_marker_offd[i] = -1;
      tmp_CF_marker_offd[i] = -1;
      fine_to_coarse_offd[i] = -1;
   }

   for (i=0; i < n_fine; i++)
   {
      P_marker[i] = -1;
      fine_to_coarse[i] = -1;
   }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;
   coarse_counter = 0;
   coarse_counter_offd = 0;

   cnt = 0;
   old_cnt = 0;
   for (i = 0; i < n_fine; i++)
   {
      fine_to_coarse[i] = -1;
      if (CF_marker[i] == 1)
      {
         fine_to_coarse[i] = cnt++;
         old_coarse_to_fine[old_cnt++] = i;
      }
      else if (CF_marker[i] == -2)
      {
         old_coarse_to_fine[old_cnt++] = i;
      }
   }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/
   for (ii = 0; ii < n_coarse_old; ii++)
   {
     P_diag_i[ii] = jj_counter;
     if (num_procs > 1)
       P_offd_i[ii] = jj_counter_offd;
     
     i = old_coarse_to_fine[ii];
     if (CF_marker[i] >= 0)
     {
       jj_counter++;
       coarse_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, interpolation is from the C-points that
      *  strongly influence i, or C-points that stronly influence F-points
      *  that strongly influence i.
      *--------------------------------------------------------------------*/
     else if (CF_marker[i] == -2)
     {
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       {
	 i1 = S_diag_j[jj];           
	 if (CF_marker[i1] > 0)
	 { /* i1 is a C point */
	   if (P_marker[i1] < P_diag_i[ii])
	   {
	     P_marker[i1] = jj_counter;
	     jj_counter++;
	   }
	 }
	 else if (CF_marker[i1] != -3)
	 { /* i1 is a F point, loop through it's strong neighbors */
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] > 0)
	     {
	       if(P_marker[k1] < P_diag_i[ii])
	       {
		 P_marker[k1] = jj_counter;
		 jj_counter++;
	       }
	     } 
	   }
	   if(num_procs > 1)
	   {
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];
	       if (CF_marker_offd[k1] > 0)
	       {
		 if(P_marker_offd[k1] < P_offd_i[ii])
		 {
		   tmp_CF_marker_offd[k1] = 1;
		   P_marker_offd[k1] = jj_counter_offd;
		   jj_counter_offd++;
		 }
	       }
	     }
	   }
	 }
       }
       /* Look at off diag strong connections of i */ 
       if (num_procs > 1)
       {
	 for (jj = S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   i1 = S_offd_j[jj];           
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[i1];
	   if (CF_marker_offd[i1] > 0)
	   {
	     if(P_marker_offd[i1] < P_offd_i[ii])
	     {
	       tmp_CF_marker_offd[i1] = 1;
	       P_marker_offd[i1] = jj_counter_offd;
	       jj_counter_offd++;
	     }
	   }
	   else if (CF_marker_offd[i1] != -3)
	   { /* F point; look at neighbors of i1. Sop contains global col
	      * numbers and entries that could be in S_diag or S_offd or
	      * neither. */
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     {
	       loc_col = Sop_j[kk];
	       if(loc_col > -1)
	       { /* In S_diag */
		 if(CF_marker[loc_col] > 0)
		 {
		   if(P_marker[loc_col] < P_diag_i[ii])
		   {
		     P_marker[loc_col] = jj_counter;
		     jj_counter++;
		   }
		 }
	       }
	       else
	       {
		 loc_col = -loc_col - 1; 
		 if(CF_marker_offd[loc_col] > 0)
		 {
		   if(P_marker_offd[loc_col] < P_offd_i[ii])
		   {
		     P_marker_offd[loc_col] = jj_counter_offd;
		     tmp_CF_marker_offd[loc_col] = 1;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       } 
     }
   }
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     determine structure    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
                                                                                               
  if (debug_flag== 4) wall_time = time_getWallclockSeconds();

   P_diag_size = jj_counter;
   P_offd_size = jj_counter_offd;

   if (P_diag_size)
   {   
      P_diag_j    = hypre_CTAlloc(int, P_diag_size);
      P_diag_data = hypre_CTAlloc(double, P_diag_size);
   }

   if (P_offd_size)
   {   
      P_offd_j    = hypre_CTAlloc(int, P_offd_size);
      P_offd_data = hypre_CTAlloc(double, P_offd_size);
   }

   P_diag_i[n_coarse_old] = jj_counter; 
   P_offd_i[n_coarse_old] = jj_counter_offd;

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /* Fine to coarse mapping */
   if(num_procs > 1)
   {
     big_insert_new_nodes(comm_pkg, extend_comm_pkg, fine_to_coarse, 
		      full_off_procNodes, my_first_cpt,
		      fine_to_coarse_offd);
   }

   for (i = 0; i < n_fine; i++)     
     P_marker[i] = -1;
     
   for (i = 0; i < full_off_procNodes; i++)
     P_marker_offd[i] = -1;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   for (ii = 0; ii < n_coarse_old; ii++)
   {
     jj_begin_row = jj_counter;        
     jj_begin_row_offd = jj_counter_offd;
     i = old_coarse_to_fine[ii];

     /*--------------------------------------------------------------------
      *  If i is a c-point, interpolation is the identity.
      *--------------------------------------------------------------------*/
     
     if (CF_marker[i] > 0)
     {
       P_diag_j[jj_counter]    = fine_to_coarse[i];
       P_diag_data[jj_counter] = one;
       jj_counter++;
     }
     
     /*--------------------------------------------------------------------
      *  If i is an F-point, build interpolation.
      *--------------------------------------------------------------------*/
     
     else if (CF_marker[i] == -2)
     {
       strong_f_marker--;
       for (jj = S_diag_i[i]; jj < S_diag_i[i+1]; jj++)
       {
	 i1 = S_diag_j[jj];   
	 
	 /*--------------------------------------------------------------
	  * If neighbor i1 is a C-point, set column number in P_diag_j
	  * and initialize interpolation weight to zero.
	  *--------------------------------------------------------------*/
	 
	 if (CF_marker[i1] > 0)
	 {
	   if (P_marker[i1] < jj_begin_row)
	   {
	     P_marker[i1] = jj_counter;
	     P_diag_j[jj_counter]    = fine_to_coarse[i1];
	     P_diag_data[jj_counter] = zero;
	     jj_counter++;
	   }
	 }
	 else  if (CF_marker[i1] != -3)
	 {
	   P_marker[i1] = strong_f_marker;
	   for (kk = S_diag_i[i1]; kk < S_diag_i[i1+1]; kk++)
	   {
	     k1 = S_diag_j[kk];
	     if (CF_marker[k1] > 0)
	     {
	       if(P_marker[k1] < jj_begin_row)
	       {
		 P_marker[k1] = jj_counter;
		 P_diag_j[jj_counter] = fine_to_coarse[k1];
		 P_diag_data[jj_counter] = zero;
		 jj_counter++;
	       }
	     }
	   }
	   if(num_procs > 1)
	   {
	     for (kk = S_offd_i[i1]; kk < S_offd_i[i1+1]; kk++)
	     {
	       if(col_offd_S_to_A)
		 k1 = col_offd_S_to_A[S_offd_j[kk]];
	       else
		 k1 = S_offd_j[kk];
	       if(CF_marker_offd[k1] > 0)
	       {
		 if(P_marker_offd[k1] < jj_begin_row_offd)
		 {
		   P_marker_offd[k1] = jj_counter_offd;
		   P_offd_j[jj_counter_offd] = k1;
		   P_offd_data[jj_counter_offd] = zero;
		   jj_counter_offd++;
		 }
	       }
	     }
	   }
	 }
       }
       
       if ( num_procs > 1)
       {
	 for (jj=S_offd_i[i]; jj < S_offd_i[i+1]; jj++)
	 {
	   i1 = S_offd_j[jj];
	   if(col_offd_S_to_A)
	     i1 = col_offd_S_to_A[i1];
	   if ( CF_marker_offd[i1] > 0)
	   {
	     if(P_marker_offd[i1] < jj_begin_row_offd)
	     {
	       P_marker_offd[i1] = jj_counter_offd;
	       P_offd_j[jj_counter_offd] = i1;
	       P_offd_data[jj_counter_offd] = zero;
	       jj_counter_offd++;
	     }
	   }
	   else if (CF_marker_offd[i1] != -3)
	   {
	     P_marker_offd[i1] = strong_f_marker;
	     for(kk = Sop_i[i1]; kk < Sop_i[i1+1]; kk++)
	     {
	       loc_col = Sop_j[kk];
	       if(loc_col > -1)
	       {
		 if(CF_marker[loc_col] > 0)
		 {
		   if(P_marker[loc_col] < jj_begin_row)
		   {		
		     P_marker[loc_col] = jj_counter;
		     P_diag_j[jj_counter] = fine_to_coarse[loc_col];
		     P_diag_data[jj_counter] = zero;
		     jj_counter++;
		   }
		 }
	       }
	       else
	       { 
		 loc_col = - loc_col - 1;
		 if(CF_marker_offd[loc_col] > 0)
		 {
		   if(P_marker_offd[loc_col] < jj_begin_row_offd)
		   {
		     P_marker_offd[loc_col] = jj_counter_offd;
		     P_offd_j[jj_counter_offd]=loc_col;
		     P_offd_data[jj_counter_offd] = zero;
		     jj_counter_offd++;
		   }
		 }
	       }
	     }
	   }
	 }
       }
       
       jj_end_row = jj_counter;
       jj_end_row_offd = jj_counter_offd;
       
       diagonal = A_diag_data[A_diag_i[i]];
       
       for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
       { /* i1 is a c-point and strongly influences i, accumulate
	  * a_(i,i1) into interpolation weight */
	 i1 = A_diag_j[jj];
	 if (P_marker[i1] >= jj_begin_row)
	 {
	   P_diag_data[P_marker[i1]] += A_diag_data[jj];
	 }
	 else if(P_marker[i1] == strong_f_marker)
	 {
	   sum = zero;
           sgn = 1;
           if(A_diag_data[A_diag_i[i1]] < 0) sgn = -1;
	   /* Loop over row of A for point i1 and calculate the sum
            * of the connections to c-points that strongly incluence i. */
	   for(jj1 = A_diag_i[i1]+1; jj1 < A_diag_i[i1+1]; jj1++)
	   {
	     i2 = A_diag_j[jj1];
	     if((P_marker[i2] >= jj_begin_row) && (sgn*A_diag_data[jj1]) < 0)
	       sum += A_diag_data[jj1];
	   }
	   if(num_procs > 1)
	   {
	     for(jj1 = A_offd_i[i1]; jj1< A_offd_i[i1+1]; jj1++)
	     {
	       i2 = A_offd_j[jj1];
	       if(P_marker_offd[i2] >= jj_begin_row_offd &&
		 (sgn*A_offd_data[jj1]) < 0)
		 sum += A_offd_data[jj1];
	     }
	   }
	   if(sum != 0)
	   {
	     distribute = A_diag_data[jj]/sum;
	     /* Loop over row of A for point i1 and do the distribution */
	     for(jj1 = A_diag_i[i1]+1; jj1 < A_diag_i[i1+1]; jj1++)
	     {
	       i2 = A_diag_j[jj1];
	       if(P_marker[i2] >= jj_begin_row && (sgn*A_diag_data[jj1]) < 0)
		 P_diag_data[P_marker[i2]] += 
		   distribute*A_diag_data[jj1];
	     }
	     if(num_procs > 1)
	     {
	       for(jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1+1]; jj1++)
	       {
		 i2 = A_offd_j[jj1];
		 if(P_marker_offd[i2] >= jj_begin_row_offd &&
                    (sgn*A_offd_data[jj1]) < 0)
		   P_offd_data[P_marker_offd[i2]] +=
		     distribute*A_offd_data[jj1];
	       }
	     }
	   }
	   else
	   {
	     diagonal += A_diag_data[jj];
	   }
	 }
	 /* neighbor i1 weakly influences i, accumulate a_(i,i1) into
	  * diagonal */
	 else if (CF_marker[i1] != -3)
	 {
	   if(num_functions == 1 || dof_func[i] == dof_func[i1])
	     diagonal += A_diag_data[jj];
	 }
       }
       if(num_procs > 1)
       {
	 for(jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
	 {
	   i1 = A_offd_j[jj];
	   if(P_marker_offd[i1] >= jj_begin_row_offd)
	     P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
	   else if(P_marker_offd[i1] == strong_f_marker)
	   {
	     sum = zero;
             sgn = 1;
             if(A_ext_data[A_ext_i[i1]] < 0) sgn = -1;
	     for(jj1 = A_ext_i[i1]+1; jj1 < A_ext_i[i1+1]; jj1++)
	     {
	       loc_col = A_ext_j[jj1];
	       if(loc_col > -1)
	       { /* diag */
		 if((P_marker[loc_col] >= jj_begin_row)
			&& (sgn*A_ext_data[jj1]) < 0)
		   sum += A_ext_data[jj1];
	       }
	       else
	       { 
		 loc_col = - loc_col - 1;
		 if(P_marker_offd[loc_col] >= jj_begin_row_offd &&
                    (sgn*A_ext_data[jj1]) < 0)
		   sum += A_ext_data[jj1];
	       }
	     }
	     if(sum != 0)
	     {
	       distribute = A_offd_data[jj] / sum;
	       for(jj1 = A_ext_i[i1]+1; jj1 < A_ext_i[i1+1]; jj1++)
	       {
		 loc_col = A_ext_j[jj1];
		 if(loc_col > -1)
		 { /* diag */
		   if(P_marker[loc_col] >= jj_begin_row &&
                      (sgn*A_ext_data[jj1]) < 0)
		     P_diag_data[P_marker[loc_col]] += distribute*
		      A_ext_data[jj1];
		 }
		 else
		 { 
		   loc_col = - loc_col - 1;
		   if(P_marker_offd[loc_col] >= jj_begin_row_offd &&
                      (sgn*A_ext_data[jj1]) < 0)
		     P_offd_data[P_marker_offd[loc_col]] += distribute*
		       A_ext_data[jj1];
		 }
	       }
	     }
	     else
	     {
	       diagonal += A_offd_data[jj];
	     }
	   }
	   else if (CF_marker_offd[i1] != -3)
	   {
	     if(num_functions == 1 || dof_func[i] == dof_func_offd[i1])
	       diagonal += A_offd_data[jj];
	   }
	 }
       }
       for(jj = jj_begin_row; jj < jj_end_row; jj++)
	 P_diag_data[jj] /= -diagonal;
       for(jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
	 P_offd_data[jj] /= -diagonal;
     }
     strong_f_marker--;
   }
   
   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d     fill structure    %f\n",
                    my_id, wall_time);
      fflush(NULL);
   }
   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
                                                                                               
   P = hypre_ParCSRMatrixCreate(comm,
				total_old_global_cpts,
				total_global_cpts,
				num_old_cpts_global,
				num_cpts_global,
				0,
				P_diag_i[n_coarse_old],
				P_offd_i[n_coarse_old]);
               
   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */
   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_coarse_old];
      P_offd_size = P_offd_i[n_coarse_old];
   }

   /* This builds col_map, col_map should be monotone increasing and contain
    * global numbers. */
   num_cols_P_offd = 0;
   if(P_offd_size)
   {
     num_cols_P_offd = 0;
     for (i=0; i < P_offd_size; i++)
     {
       index = P_offd_j[i];
       if (tmp_CF_marker_offd[index] == 1)
       {
	   num_cols_P_offd++;
	   tmp_CF_marker_offd[index] = 2;
       }
     }

     if (num_cols_P_offd)
     {     
        col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt, num_cols_P_offd);
        tmp_map_offd = hypre_CTAlloc(int, num_cols_P_offd);
     }     
     
     index = 0;
     for(i = 0; i < num_cols_P_offd; i++)
     {
       while( tmp_CF_marker_offd[index] != 2) index++;
       /*printf("index %d tmp_CF %d\n", index, tmp_CF_marker_offd[index]);*/
       col_map_offd_P[i] = fine_to_coarse_offd[index];
       tmp_map_offd[i] = index++;
     }

     hypre_BigQsortbi(col_map_offd_P, tmp_map_offd, 0, num_cols_P_offd-1);

     for (i = 0; i < num_cols_P_offd; i++)
        tmp_CF_marker_offd[tmp_map_offd[i]] = i;

     for(i = 0; i < P_offd_size; i++)
       P_offd_j[i] = tmp_CF_marker_offd[P_offd_j[i]];
   }

   if (num_cols_P_offd)
   { 
     hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
     hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
     hypre_TFree(tmp_map_offd);
   } 

#ifdef HYPRE_NO_GLOBAL_PARTITION
     hypre_NewCommPkgCreate(P);
#else
     hypre_MatvecCommPkgCreate(P);
#endif
 
   for (i=0; i < n_fine; i++)
      if (CF_marker[i] < -1) CF_marker[i] = -1;
 
   *P_ptr = P;

   /* Deallocate memory */   
   hypre_TFree(fine_to_coarse);
   hypre_TFree(old_coarse_to_fine);
   hypre_TFree(P_marker);
   
   if (num_procs > 1) 
   {
     hypre_BigCSRMatrixDestroy(Sop);
     hypre_BigCSRMatrixDestroy(A_ext);
     hypre_TFree(fine_to_coarse_offd);
     hypre_TFree(P_marker_offd);
     hypre_TFree(CF_marker_offd);
     hypre_TFree(tmp_CF_marker_offd);
     if(num_functions > 1)
       hypre_TFree(dof_func_offd);
     hypre_TFree(found);


     hypre_MatvecCommPkgDestroy(extend_comm_pkg);
   

   }
   
   return hypre_error_flag;  
}

