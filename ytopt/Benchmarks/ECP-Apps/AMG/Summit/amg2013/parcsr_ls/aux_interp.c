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

/*---------------------------------------------------------------------------
 * Auxilary routines for the long range interpolation methods.
 *  Implemented: "standard", "extended", "multipass", "FF"
 *--------------------------------------------------------------------------*/
/* AHB 11/06: Modification of the above original - takes two
   communication packages and inserts nodes to position expected for
   OUT_marker
  
   offd nodes from comm_pkg take up first chunk of CF_marker_offd, offd 
   nodes from extend_comm_pkg take up the second chunk 0f CF_marker_offd. */



int alt_insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg, 
                          hypre_ParCSRCommPkg *extend_comm_pkg,
                          int *IN_marker, 
                          int full_off_procNodes,
                          int *OUT_marker)
{   
  hypre_ParCSRCommHandle  *comm_handle;

  int i, j, start, index, shift;

  int num_sends, num_recvs;
  
  int *recv_vec_starts;

  int e_num_sends;

  int *int_buf_data;
  int *e_out_marker;
  

  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  num_recvs =  hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

  e_num_sends = hypre_ParCSRCommPkgNumSends(extend_comm_pkg);


  index = hypre_max(hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                    hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends));

  int_buf_data = hypre_CTAlloc(int, index);

  /* orig commpkg data*/
  index = 0;
  
  for (i = 0; i < num_sends; i++)
  {
    start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
    for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); 
	 j++)
      int_buf_data[index++] 
	= IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
  }
   
  comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
					      OUT_marker);
   
  hypre_ParCSRCommHandleDestroy(comm_handle);
  comm_handle = NULL;
  
  /* now do the extend commpkg */

  /* first we need to shift our position in the OUT_marker */
  shift = recv_vec_starts[num_recvs];
  e_out_marker = OUT_marker + shift;
  
  index = 0;

  for (i = 0; i < e_num_sends; i++)
  {
    start = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, i);
    for (j = start; j < hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, i+1); 
	 j++)
       int_buf_data[index++] 
	= IN_marker[hypre_ParCSRCommPkgSendMapElmt(extend_comm_pkg,j)];
  }
   
  comm_handle = hypre_ParCSRCommHandleCreate( 11, extend_comm_pkg, int_buf_data, 
					      e_out_marker);
   
  hypre_ParCSRCommHandleDestroy(comm_handle);
  comm_handle = NULL;
  
  hypre_TFree(int_buf_data);
    
  return hypre_error_flag;
} 


int big_insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg, 
                          hypre_ParCSRCommPkg *extend_comm_pkg,
                          int *IN_marker, 
                          int full_off_procNodes,
                          HYPRE_BigInt offset,
                          HYPRE_BigInt *OUT_marker)
{   
  hypre_ParCSRCommHandle  *comm_handle;

  int i, j, start, index, shift;

  int num_sends, num_recvs;
  
  int *recv_vec_starts;

  int e_num_sends;

  HYPRE_BigInt *big_buf_data;
  HYPRE_BigInt *e_out_marker;
  

  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  num_recvs =  hypre_ParCSRCommPkgNumRecvs(comm_pkg);
  recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

  e_num_sends = hypre_ParCSRCommPkgNumSends(extend_comm_pkg);


  index = hypre_max(hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                    hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends));

  big_buf_data = hypre_CTAlloc(HYPRE_BigInt, index);

  /* orig commpkg data*/
  index = 0;
  
  for (i = 0; i < num_sends; i++)
  {
    start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
    for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); 
	 j++)
      big_buf_data[index++] = offset
	+ (HYPRE_BigInt) IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
  }
   
  comm_handle = hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_buf_data, 
					      OUT_marker);
   
  hypre_ParCSRCommHandleDestroy(comm_handle);
  comm_handle = NULL;
  
  /* now do the extend commpkg */

  /* first we need to shift our position in the OUT_marker */
  shift = recv_vec_starts[num_recvs];
  e_out_marker = OUT_marker + shift;
  
  index = 0;

  for (i = 0; i < e_num_sends; i++)
  {
    start = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, i);
    for (j = start; j < hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, i+1); 
	 j++)
       big_buf_data[index++] = offset 
	+ (HYPRE_BigInt) IN_marker[hypre_ParCSRCommPkgSendMapElmt(extend_comm_pkg,j)];
  }
   
  comm_handle = hypre_ParCSRCommHandleCreate( 21, extend_comm_pkg, big_buf_data, 
					      e_out_marker);
   
  hypre_ParCSRCommHandleDestroy(comm_handle);
  comm_handle = NULL;
  
  hypre_TFree(big_buf_data);
    
  return hypre_error_flag;
} 



/* AHB 11/06 : alternate to the extend function below - creates a
 * second comm pkg based on found - this makes it easier to use the
 * global partition*/
int
hypre_ParCSRFindExtendCommPkg(hypre_ParCSRMatrix *A, int newoff, HYPRE_BigInt *found, 
                              hypre_ParCSRCommPkg **extend_comm_pkg)

{
   

   int			num_sends;
   int			*send_procs;
   int			*send_map_starts;
   int			*send_map_elmts;
 
   int			num_recvs;
   int			*recv_procs;
   int			*recv_vec_starts;

   hypre_ParCSRCommPkg	*new_comm_pkg;

   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);

   HYPRE_BigInt first_col_diag = hypre_ParCSRMatrixFirstColDiag(A);
  /* use found instead of col_map_offd in A, and newoff instead 
      of num_cols_offd*/

#if HYPRE_NO_GLOBAL_PARTITION

   HYPRE_BigInt        row_start=0, row_end=0, col_start = 0, col_end = 0;
   HYPRE_BigInt        global_num_cols;
   hypre_IJAssumedPart   *apart;
   
   hypre_ParCSRMatrixGetLocalRange( A,
                                    &row_start, &row_end ,
                                    &col_start, &col_end );
   

   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A); 

   /* Create the assumed partition */
   if  (hypre_ParCSRMatrixAssumedPartition(A) == NULL)
   {
      hypre_ParCSRMatrixCreateAssumedPartition(A);
   }

   apart = hypre_ParCSRMatrixAssumedPartition(A);
   
   hypre_NewCommPkgCreate_core( comm, found, first_col_diag, 
                                col_start, col_end, 
                                newoff, global_num_cols,
                                &num_recvs, &recv_procs, &recv_vec_starts,
                                &num_sends, &send_procs, &send_map_starts, 
                                &send_map_elmts, apart);

#else   
   HYPRE_BigInt  *col_starts = hypre_ParCSRMatrixColStarts(A);
   int	num_cols_diag = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A));
   
   hypre_MatvecCommPkgCreate_core
      (
         comm, found, first_col_diag, col_starts,
         num_cols_diag, newoff,
         first_col_diag, found,
         1,
         &num_recvs, &recv_procs, &recv_vec_starts,
         &num_sends, &send_procs, &send_map_starts,
         &send_map_elmts
         );

#endif

   new_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);

   hypre_ParCSRCommPkgComm(new_comm_pkg) = comm;

   hypre_ParCSRCommPkgNumRecvs(new_comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(new_comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(new_comm_pkg) = recv_vec_starts;
   hypre_ParCSRCommPkgNumSends(new_comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(new_comm_pkg) = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(new_comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(new_comm_pkg) = send_map_elmts;



   *extend_comm_pkg = new_comm_pkg;
   

   return hypre_error_flag;
   
}

/* Find nodes that are offd and are not contained in original offd
 * (neighbors of neighbors) */
int new_offd_nodes(HYPRE_BigInt **found, int num_cols_A_offd, int *A_ext_i, 
		   HYPRE_BigInt *big_A_ext_j, int **A_ext_j_ptr,
		   int num_cols_S_offd, HYPRE_BigInt *col_map_offd, HYPRE_BigInt col_1, 
		   HYPRE_BigInt col_n, int *Sop_i, HYPRE_BigInt *big_Sop_j,
		   int **Sop_j_ptr,
		   int *CF_marker, hypre_ParCSRCommPkg *comm_pkg)
{
  int i, ii, j, ifound, kk;
  int got_loc, loc_col;

  int min;

  int size_offP;

  HYPRE_BigInt *tmp_found, big_ifound;
  HYPRE_BigInt big_i1, big_k1;
  int *CF_marker_offd = NULL;
  int *int_buf_data;
  int *loc;
  int *A_ext_j;
  int *Sop_j;
  int newoff = 0;
  hypre_ParCSRCommHandle *comm_handle;
                                                                                                                                         
  CF_marker_offd = hypre_CTAlloc(int, num_cols_A_offd);
  int_buf_data = hypre_CTAlloc(int, 
		hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                hypre_ParCSRCommPkgNumSends(comm_pkg)));
  ii = 0;
  for (i=0; i < hypre_ParCSRCommPkgNumSends(comm_pkg); i++)
  {
      for (j=hypre_ParCSRCommPkgSendMapStart(comm_pkg,i);
                j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        int_buf_data[ii++]
          = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
  }
  comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg,int_buf_data,
        CF_marker_offd);
  hypre_ParCSRCommHandleDestroy(comm_handle);
  hypre_TFree(int_buf_data);

  size_offP = A_ext_i[num_cols_A_offd];
  tmp_found = hypre_CTAlloc(HYPRE_BigInt, size_offP);
  A_ext_j = hypre_CTAlloc(int, size_offP);
  loc = hypre_CTAlloc(int, size_offP);

  /* Find nodes that will be added to the off diag list */ 
  for (i = 0; i < num_cols_A_offd; i++)
  {
   if (CF_marker_offd[i] < 0)
   {
    for (j = A_ext_i[i]; j < A_ext_i[i+1]; j++)
    {
      big_i1 = big_A_ext_j[j];
      if(big_i1 < col_1 || big_i1 >= col_n)
      {
	  ifound = hypre_BigBinarySearch(col_map_offd,big_i1,num_cols_A_offd);
	  if(ifound == -1)
	  {
	      tmp_found[newoff]=big_i1;
	      loc[newoff++] = j;
	  }
	  else
	  {
	      A_ext_j[j] = -ifound-1;
	  }
      }
      else
	  A_ext_j[j] = (int)(big_i1 - col_1);
    }
   }
  }
  /* Put found in monotone increasing order */
  if (newoff > 0)
  {
     hypre_BigQsort0(tmp_found,0,newoff-1);
     big_ifound = tmp_found[0];
     min = 1;
     for (i=1; i < newoff; i++)
     {
       if (tmp_found[i] > big_ifound)
       {
          big_ifound = tmp_found[i];
          tmp_found[min++] = big_ifound;
       }
     }
  }

  for (i=0; i < newoff; i++)
  {
      kk = hypre_BigBinarySearch(tmp_found,big_A_ext_j[loc[i]],min);
      A_ext_j[loc[i]] = -kk - num_cols_A_offd - 1;
  }

  if (newoff > 0) newoff = min;
  hypre_TFree(big_A_ext_j);
  hypre_TFree(loc);
  
  /* Set column indices for Sop and A_ext such that offd nodes are
   * negatively indexed */

  Sop_j = hypre_CTAlloc(int,Sop_i[num_cols_S_offd]);

  if (newoff < num_cols_A_offd)
  {
   for(i = 0; i < num_cols_S_offd; i++)
   {
    if (CF_marker_offd[i] < 0)
    {
     for(kk = Sop_i[i]; kk < Sop_i[i+1]; kk++)
     {
       big_k1 = big_Sop_j[kk];
       if(big_k1 < col_1 || big_k1 >= col_n)
       { 
	 got_loc = hypre_BigBinarySearch(tmp_found,big_k1,newoff);
	 if(got_loc > -1)
	     loc_col = got_loc + num_cols_A_offd;
	 else
	     loc_col = hypre_BigBinarySearch(col_map_offd,big_k1,
					  num_cols_A_offd);
	 if(loc_col < 0)
	 {
	   printf("Could not find node: STOP\n");
	   return(-1);
	 }
	 Sop_j[kk] = -loc_col - 1;
       }
       else
	 Sop_j[kk] = (int)(big_k1-col_1);
     }
    }
   }
  }
  else
  {
   for(i = 0; i < num_cols_S_offd; i++)
   {
    if (CF_marker_offd[i] < 0)
    {
     for(kk = Sop_i[i]; kk < Sop_i[i+1]; kk++)
     {
       big_k1 = big_Sop_j[kk];
       if(big_k1 < col_1 || big_k1 >= col_n)
       { 
	 loc_col = hypre_BigBinarySearch(col_map_offd,big_k1,
					num_cols_A_offd);
	 if(loc_col == -1)
	     loc_col = hypre_BigBinarySearch(tmp_found,big_k1,newoff) +
	       num_cols_A_offd;
	 if(loc_col < 0)
	 {
	   printf("Could not find node: STOP\n");
	   return(-1);
	 }
	 Sop_j[kk] = -loc_col - 1;
       }
       else
	 Sop_j[kk] = (int)(big_k1-col_1);
     }
    }
   }
  }

  hypre_TFree(big_Sop_j);
  hypre_TFree(CF_marker_offd);
  

  *found = tmp_found;
  *A_ext_j_ptr = A_ext_j; 
  *Sop_j_ptr = Sop_j; 


  return newoff;
}
