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



/******************************************************************************
 *
 *****************************************************************************/

/* following should be in a header file */


#include "headers.h"



/*==========================================================================*/
/*==========================================================================*/
/**
  Generates nodal norm matrix for use with nodal systems version

  {\bf Input files:}
  headers.h

  @return Error code.
  
  @param A [IN]
  coefficient matrix
  @param AN_ptr [OUT]
  nodal norm matrix
  
  @see */
/*--------------------------------------------------------------------------*/

int
hypre_BoomerAMGCreateNodalA(hypre_ParCSRMatrix    *A,
                       int                    num_functions,
                       int                   *dof_func,
                       int                    option,
                       hypre_ParCSRMatrix   **AN_ptr)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix    *A_diag          = hypre_ParCSRMatrixDiag(A);
   int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   double             *A_diag_data     = hypre_CSRMatrixData(A_diag);


   hypre_CSRMatrix    *A_offd          = hypre_ParCSRMatrixOffd(A);
   int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   double             *A_offd_data     = hypre_CSRMatrixData(A_offd);
   int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   HYPRE_BigInt	      *row_starts      = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt	      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(A);
   int                 num_variables   = hypre_CSRMatrixNumRows(A_diag);
   int 		       num_nonzeros_offd = 0;
   int 		       num_cols_offd = 0;
                  
   hypre_ParCSRMatrix *AN;
   hypre_CSRMatrix    *AN_diag;
   int                *AN_diag_i;
   int                *AN_diag_j;
   double             *AN_diag_data; 
   hypre_CSRMatrix    *AN_offd;
   int                *AN_offd_i;
   int                *AN_offd_j = NULL;
   double             *AN_offd_data = NULL; 
   HYPRE_BigInt	      *col_map_offd_AN = NULL;
   HYPRE_BigInt	      *new_col_map_offd = NULL;
   HYPRE_BigInt	      *row_starts_AN;
   int		       AN_num_nonzeros_diag = 0;
   int		       AN_num_nonzeros_offd = 0;
   int		       num_cols_offd_AN;
   int		       new_num_cols_offd;
                 
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   int		       num_sends;
   int		       num_recvs;
   int		      *send_procs;
   int		      *send_map_starts;
   int		      *send_map_elmts;
   int		      *new_send_map_elmts;
   int		      *recv_procs;
   int		      *recv_vec_starts;

   hypre_ParCSRCommPkg *comm_pkg_AN;
   int		      *send_procs_AN;
   int		      *send_map_starts_AN;
   int		      *send_map_elmts_AN;
   int		      *recv_procs_AN;
   int		      *recv_vec_starts_AN;

   int                 i, j, k, k_map;
                      
   int                 ierr = 0;

   int		       index, row;
   int		       start_index;
   int		       num_procs;
   int		       mode, node, cnt;
   HYPRE_BigInt	       big_node;
   int		       new_send_elmts_size;

   HYPRE_BigInt	       global_num_nodes;
   int		       num_nodes;
   int		       num_fun2;
   HYPRE_BigInt	      *big_map_to_node = NULL;
   int		      *map_to_node = NULL;
   int		      *map_to_map;
   int		      *counter;

   MPI_Comm_size(comm,&num_procs);

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      hypre_NewCommPkgCreate(A);
#else
      hypre_MatvecCommPkgCreate(A);
#endif
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   mode = fabs(option);

   comm_pkg_AN = NULL;
   col_map_offd_AN = NULL;

#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_starts_AN = hypre_CTAlloc(HYPRE_BigInt, 2);

   for (i=0; i < 2; i++)
   {
      row_starts_AN[i] = row_starts[i]/(HYPRE_BigInt)num_functions;
      if (row_starts_AN[i]*(HYPRE_BigInt)num_functions < row_starts[i])
      {
	  printf("nodes not properly aligned or incomplete info!\n");
	  return (87);
      }
   }
   
   global_num_nodes = hypre_ParCSRMatrixGlobalNumRows(A)/(HYPRE_BigInt)num_functions;


#else
   row_starts_AN = hypre_CTAlloc(HYPRE_BigInt, num_procs+1);

  for (i=0; i < num_procs+1; i++)
   {
      row_starts_AN[i] = row_starts[i]/(HYPRE_BigInt)num_functions;
      if (row_starts_AN[i]*(HYPRE_BigInt)num_functions < row_starts[i])
      {
	  printf("nodes not properly aligned or incomplete info!\n");
	  return (87);
      }
   }
   
   global_num_nodes = row_starts_AN[num_procs];

#endif

 
   num_nodes =  num_variables/num_functions;
   num_fun2 = num_functions*num_functions;

   map_to_node = hypre_CTAlloc(int, num_variables);
   AN_diag_i = hypre_CTAlloc(int, num_nodes+1);
   counter = hypre_CTAlloc(int, num_nodes);
   for (i=0; i < num_variables; i++)
      map_to_node[i] = i/num_functions;
   for (i=0; i < num_nodes; i++)
      counter[i] = -1;

   AN_num_nonzeros_diag = 0;
   row = 0;
   for (i=0; i < num_nodes; i++)
   {
      AN_diag_i[i] = AN_num_nonzeros_diag;
      for (j=0; j < num_functions; j++)
      {
	 for (k=A_diag_i[row]; k < A_diag_i[row+1]; k++)
	 {
	    k_map = map_to_node[A_diag_j[k]];
	    if (counter[k_map] < i)
	    {
	       counter[k_map] = i;
	       AN_num_nonzeros_diag++;
	    }
	 }
	 row++;
      }
   }
   AN_diag_i[num_nodes] = AN_num_nonzeros_diag;

   AN_diag_j = hypre_CTAlloc(int, AN_num_nonzeros_diag);	
   AN_diag_data = hypre_CTAlloc(double, AN_num_nonzeros_diag);	

   AN_diag = hypre_CSRMatrixCreate(num_nodes,num_nodes,AN_num_nonzeros_diag);
   hypre_CSRMatrixI(AN_diag) = AN_diag_i;
   hypre_CSRMatrixJ(AN_diag) = AN_diag_j;
   hypre_CSRMatrixData(AN_diag) = AN_diag_data;
       
   for (i=0; i < num_nodes; i++)
      counter[i] = -1;
   index = 0;
   start_index = 0;
   row = 0;

   switch (mode)
   {
      case 7:  /* frobenius norm with signs*/
      {
         for (i=0; i < num_nodes; i++)
         {
            for (j=0; j < num_functions; j++)
            {
	       for (k=A_diag_i[row]; k < A_diag_i[row+1]; k++)
	       {
	          k_map = map_to_node[A_diag_j[k]];
	          if (counter[k_map] < start_index)
	          {
	             counter[k_map] = index;
	             AN_diag_j[index] = k_map;
	             AN_diag_data[index] = A_diag_data[k]*A_diag_data[k];
	             index++;
	          }
	          else
	          {
	             AN_diag_data[counter[k_map]] += 
				A_diag_data[k]*A_diag_data[k];
	          }
	       }
	       row++;
            }
            start_index = index;
         }
         for (i=0; i < AN_num_nonzeros_diag; i++)
            AN_diag_data[i] = sqrt(AN_diag_data[i]);

         /* temp for testing - make all diagonal entries negative */
         /* the diagonal is the first element listed in each row -
            this is the same as serial code */

         for (i=0; i < num_nodes; i++)
         {
            index = AN_diag_i[i];
            AN_diag_data[index] = - AN_diag_data[index];
         }

      }
      break;
      
      case 1:  /* frobenius norm */
      {
         for (i=0; i < num_nodes; i++)
         {
            for (j=0; j < num_functions; j++)
            {
	       for (k=A_diag_i[row]; k < A_diag_i[row+1]; k++)
	       {
	          k_map = map_to_node[A_diag_j[k]];
	          if (counter[k_map] < start_index)
	          {
	             counter[k_map] = index;
	             AN_diag_j[index] = k_map;
	             AN_diag_data[index] = A_diag_data[k]*A_diag_data[k];
	             index++;
	          }
	          else
	          {
	             AN_diag_data[counter[k_map]] += 
				A_diag_data[k]*A_diag_data[k];
	          }
	       }
	       row++;
            }
            start_index = index;
         }
         for (i=0; i < AN_num_nonzeros_diag; i++)
            AN_diag_data[i] = sqrt(AN_diag_data[i]);

#if 0
         /* temp for testing - make all diagonal entries negative */
         /* the diagonal is the first element listed in each row -
            this is the same as serial code */

         for (i=0; i < num_nodes; i++)
         {
            index = AN_diag_i[i];
            AN_diag_data[index] = - AN_diag_data[index];
         }
#endif         

      }
      break;
      
      case 2:  /* sum of abs. value of all elements in each block */
      {
         for (i=0; i < num_nodes; i++)
         {
            for (j=0; j < num_functions; j++)
            {
	       for (k=A_diag_i[row]; k < A_diag_i[row+1]; k++)
	       {
	          k_map = map_to_node[A_diag_j[k]];
	          if (counter[k_map] < start_index)
	          {
	             counter[k_map] = index;
	             AN_diag_j[index] = k_map;
	             AN_diag_data[index] = fabs(A_diag_data[k]);
	             index++;
	          }
	          else
	          {
	             AN_diag_data[counter[k_map]] += fabs(A_diag_data[k]);
	          }
	       }
	       row++;
            }
            start_index = index;
         }
         for (i=0; i < AN_num_nonzeros_diag; i++)
            AN_diag_data[i] /= num_fun2;
      }
      break;

      case 3:  /* largest element of each block (sets true value - not abs. value) */
      {

         for (i=0; i < num_nodes; i++)
         {
            for (j=0; j < num_functions; j++)
            {
      	       for (k=A_diag_i[row]; k < A_diag_i[row+1]; k++)
      	       {
      	          k_map = map_to_node[A_diag_j[k]];
      	          if (counter[k_map] < start_index)
      	          {
      	             counter[k_map] = index;
      	             AN_diag_j[index] = k_map;
      	             AN_diag_data[index] = A_diag_data[k];
      	             index++;
      	          }
      	          else
      	          {
      	             if (fabs(A_diag_data[k]) > 
				fabs(AN_diag_data[counter[k_map]]))
      	                AN_diag_data[counter[k_map]] = A_diag_data[k];
      	          }
      	       }
      	       row++;
            }
            start_index = index;
         }
      }
      break;
   }

   num_nonzeros_offd = A_offd_i[num_variables];
   AN_offd_i = hypre_CTAlloc(int, num_nodes+1);

   num_cols_offd_AN = 0;

   if (comm_pkg)
   {
      comm_pkg_AN = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
      hypre_ParCSRCommPkgComm(comm_pkg_AN) = comm;
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      hypre_ParCSRCommPkgNumSends(comm_pkg_AN) = num_sends;
      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      hypre_ParCSRCommPkgNumRecvs(comm_pkg_AN) = num_recvs;
      send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      send_procs_AN = NULL;
      send_map_elmts_AN = NULL;
      if (num_sends) 
      {
         send_procs_AN = hypre_CTAlloc(int,num_sends);
         send_map_elmts_AN = hypre_CTAlloc(int,send_map_starts[num_sends]);
      }
      send_map_starts_AN = hypre_CTAlloc(int,num_sends+1);
      recv_vec_starts_AN = hypre_CTAlloc(int,num_recvs+1);
      recv_procs_AN = NULL;
      if (num_recvs) recv_procs_AN = hypre_CTAlloc(int,num_recvs);
      for (i=0; i < num_sends; i++)
         send_procs_AN[i] = send_procs[i];
      for (i=0; i < num_recvs; i++)
         recv_procs_AN[i] = recv_procs[i];

      send_map_starts_AN[0] = 0;
      cnt = 0;
      for (i=0; i < num_sends; i++)
      {
	 k_map = send_map_starts[i];
	 if (send_map_starts[i+1]-k_map)
            send_map_elmts_AN[cnt++] = send_map_elmts[k_map]/num_functions;
         for (j=send_map_starts[i]+1; j < send_map_starts[i+1]; j++)
         {
            node = send_map_elmts[j]/num_functions;
            if (node > send_map_elmts_AN[cnt-1])
	       send_map_elmts_AN[cnt++] = node; 
         }
         send_map_starts_AN[i+1] = cnt;
      }
      hypre_ParCSRCommPkgSendProcs(comm_pkg_AN) = send_procs_AN;
      hypre_ParCSRCommPkgSendMapStarts(comm_pkg_AN) = send_map_starts_AN;
      hypre_ParCSRCommPkgSendMapElmts(comm_pkg_AN) = send_map_elmts_AN;
      hypre_ParCSRCommPkgRecvProcs(comm_pkg_AN) = recv_procs_AN;
      hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_AN) = recv_vec_starts_AN;
   }

   num_cols_offd = hypre_CSRMatrixNumCols(A_offd);
   hypre_TFree(map_to_node);
   if (num_cols_offd)
   {
      if (num_cols_offd > num_variables)
      {
         big_map_to_node = hypre_CTAlloc(HYPRE_BigInt,num_cols_offd);
      }

      num_cols_offd_AN = 1;
      big_map_to_node[0] = col_map_offd[0]/(HYPRE_BigInt)num_functions;
      for (i=1; i < num_cols_offd; i++)
      {
         big_map_to_node[i] = col_map_offd[i]/(HYPRE_BigInt)num_functions;
         if (big_map_to_node[i] > big_map_to_node[i-1]) num_cols_offd_AN++;
      }
      
      if (num_cols_offd_AN > num_nodes)
      {
         hypre_TFree(counter);
         counter = hypre_CTAlloc(int,num_cols_offd_AN);
      }

      map_to_map = NULL;
      col_map_offd_AN = NULL;
      map_to_map = hypre_CTAlloc(int, num_cols_offd);
      col_map_offd_AN = hypre_CTAlloc(HYPRE_BigInt,num_cols_offd_AN);
      col_map_offd_AN[0] = big_map_to_node[0];
      recv_vec_starts_AN[0] = 0;
      cnt = 1;
      for (i=0; i < num_recvs; i++)
      {
         for (j=recv_vec_starts[i]; j < recv_vec_starts[i+1]; j++)
         {
            big_node = big_map_to_node[j];
	    if (big_node > col_map_offd_AN[cnt-1])
	    {
	       col_map_offd_AN[cnt++] = big_node; 
	    }
	    map_to_map[j] = cnt-1;
         }
         recv_vec_starts_AN[i+1] = cnt;
      }

      for (i=0; i < num_cols_offd_AN; i++)
         counter[i] = -1;

      AN_num_nonzeros_offd = 0;
      row = 0;
      for (i=0; i < num_nodes; i++)
      {
         AN_offd_i[i] = AN_num_nonzeros_offd;
         for (j=0; j < num_functions; j++)
         {
	    for (k=A_offd_i[row]; k < A_offd_i[row+1]; k++)
	    {
	       k_map = map_to_map[A_offd_j[k]];
	       if (counter[k_map] < i)
	       {
	          counter[k_map] = i;
	          AN_num_nonzeros_offd++;
	       }
	    }
	    row++;
         }
      }
      AN_offd_i[num_nodes] = AN_num_nonzeros_offd;
   }

       
   AN_offd = hypre_CSRMatrixCreate(num_nodes,num_cols_offd_AN,	
		AN_num_nonzeros_offd);
   hypre_CSRMatrixI(AN_offd) = AN_offd_i;
   if (AN_num_nonzeros_offd)
   {
      AN_offd_j = hypre_CTAlloc(int, AN_num_nonzeros_offd);	
      AN_offd_data = hypre_CTAlloc(double, AN_num_nonzeros_offd);	
      hypre_CSRMatrixJ(AN_offd) = AN_offd_j;
      hypre_CSRMatrixData(AN_offd) = AN_offd_data;
   
      for (i=0; i < num_cols_offd_AN; i++)
         counter[i] = -1;
      index = 0;
      row = 0;
      AN_offd_i[0] = 0;
      start_index = 0;
      switch (mode)
      {
         case 1: /* frobenius norm */
         {
            for (i=0; i < num_nodes; i++)
            {
               for (j=0; j < num_functions; j++)
               {
	          for (k=A_offd_i[row]; k < A_offd_i[row+1]; k++)
	          {
	             k_map = map_to_map[A_offd_j[k]];
	             if (counter[k_map] < start_index)
	             {
	                counter[k_map] = index;
	                AN_offd_j[index] = k_map;
	                AN_offd_data[index] = A_offd_data[k]*A_offd_data[k];
	                index++;
	             }
	             else
	             {
	                AN_offd_data[counter[k_map]] += 
				A_offd_data[k]*A_offd_data[k];
	             }
	          }
	          row++;
               }
               start_index = index;
            }
            for (i=0; i < AN_num_nonzeros_offd; i++)
	       AN_offd_data[i] = sqrt(AN_offd_data[i]);
         }
         break;
      
         case 2:  /* sum of abs. value of all elements in block */
         {
            for (i=0; i < num_nodes; i++)
            {
               for (j=0; j < num_functions; j++)
               {
	          for (k=A_offd_i[row]; k < A_offd_i[row+1]; k++)
	          {
	             k_map = map_to_map[A_offd_j[k]];
	             if (counter[k_map] < start_index)
	             {
	                counter[k_map] = index;
	                AN_offd_j[index] = k_map;
	                AN_offd_data[index] = fabs(A_offd_data[k]);
	                index++;
	             }
	             else
	             {
	                AN_offd_data[counter[k_map]] += fabs(A_offd_data[k]);
	             }
	          }
	          row++;
               }
               start_index = index;
            }
            for (i=0; i < AN_num_nonzeros_offd; i++)
               AN_offd_data[i] /= num_fun2;
         }
         break;

         case 3: /* largest element in each block (not abs. value ) */
         {
            for (i=0; i < num_nodes; i++)
            {
               for (j=0; j < num_functions; j++)
               {
      	          for (k=A_offd_i[row]; k < A_offd_i[row+1]; k++)
      	          {
      	             k_map = map_to_map[A_offd_j[k]];
      	             if (counter[k_map] < start_index)
      	             {
      	                counter[k_map] = index;
      	                AN_offd_j[index] = k_map;
      	                AN_offd_data[index] = A_offd_data[k];
      	                index++;
      	             }
      	             else
      	             {
      	                if (fabs(A_offd_data[k]) > 
				fabs(AN_offd_data[counter[k_map]]))
      	                   AN_offd_data[counter[k_map]] = A_offd_data[k];
      	             }
      	          }
      	          row++;
               }
               start_index = index;
            }
         }
         break;
      }
      hypre_TFree(map_to_map);
   }
   
    
   AN = hypre_ParCSRMatrixCreate(comm, global_num_nodes, global_num_nodes,
		row_starts_AN, row_starts_AN, num_cols_offd_AN,
		AN_num_nonzeros_diag, AN_num_nonzeros_offd);

   /* we already created the diag and offd matrices - so we don't need the ones
      created above */
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(AN));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(AN));
   hypre_ParCSRMatrixDiag(AN) = AN_diag;
   hypre_ParCSRMatrixOffd(AN) = AN_offd;


   hypre_ParCSRMatrixColMapOffd(AN) = col_map_offd_AN;
   hypre_ParCSRMatrixCommPkg(AN) = comm_pkg_AN;

   new_num_cols_offd = num_functions*num_cols_offd_AN;

   if (new_num_cols_offd > num_cols_offd)
   {
      new_col_map_offd = hypre_CTAlloc(HYPRE_BigInt, new_num_cols_offd);
      cnt = 0;
      for (i=0; i < num_cols_offd_AN; i++)
      {
	 for (j=0; j < num_functions; j++)
         {
 	    new_col_map_offd[cnt++] = (HYPRE_BigInt)num_functions*
		col_map_offd_AN[i]+(HYPRE_BigInt)j;
         }
      }
      cnt = 0;
      for (i=0; i < num_cols_offd; i++)
      {
         while (col_map_offd[i] >  new_col_map_offd[cnt])
            cnt++;
         col_map_offd[i] = cnt++;
      }
      for (i=0; i < num_recvs+1; i++)
      {
         recv_vec_starts[i] = num_functions*recv_vec_starts_AN[i];
      }

      for (i=0; i < num_nonzeros_offd; i++)
      {
         j = A_offd_j[i];
	 A_offd_j[i] = col_map_offd[j];
      }
      hypre_ParCSRMatrixColMapOffd(A) = new_col_map_offd;
      hypre_CSRMatrixNumCols(A_offd) = new_num_cols_offd;
      hypre_TFree(col_map_offd);
   }
 
   hypre_TFree(big_map_to_node);
   new_send_elmts_size = send_map_starts_AN[num_sends]*num_functions;

   if (new_send_elmts_size > send_map_starts[num_sends])
   {
      new_send_map_elmts = hypre_CTAlloc(int,new_send_elmts_size);
      cnt = 0;
      send_map_starts[0] = 0;
      for (i=0; i < num_sends; i++)
      {
         send_map_starts[i+1] = send_map_starts_AN[i+1]*num_functions;
         for (j=send_map_starts_AN[i]; j < send_map_starts_AN[i+1]; j++)
	 {
            for (k=0; k < num_functions; k++)
	       new_send_map_elmts[cnt++] = send_map_elmts_AN[j]*num_functions+k;
	 }
      }
      hypre_TFree(send_map_elmts);
      hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = new_send_map_elmts;
   }
 
   *AN_ptr        = AN;

   hypre_TFree(counter);

   return (ierr);
}


/* This creates a scalar version of the CF_marker, dof_array and strength matrix (SN) */

int
hypre_BoomerAMGCreateScalarCFS(hypre_ParCSRMatrix    *SN,
                       int                   *CFN_marker,
                       int                   *col_offd_SN_to_AN,
                       int                    num_functions,
                       int                    nodal,
                       int                    data,
                       int                  **dof_func_ptr,
                       int                  **CF_marker_ptr,
                       int                  **col_offd_S_to_A_ptr,
                       hypre_ParCSRMatrix   **S_ptr)
{
   MPI_Comm	       comm = hypre_ParCSRMatrixComm(SN);
   hypre_ParCSRMatrix *S;
   hypre_CSRMatrix    *S_diag;
   int		      *S_diag_i;
   int		      *S_diag_j;
   double	      *S_diag_data;
   hypre_CSRMatrix    *S_offd;
   int		      *S_offd_i;
   int		      *S_offd_j;
   double	      *S_offd_data;
   HYPRE_BigInt	      *row_starts_S;
   HYPRE_BigInt	      *col_starts_S;
   HYPRE_BigInt	      *row_starts_SN = hypre_ParCSRMatrixRowStarts(SN);
   HYPRE_BigInt	      *col_starts_SN = hypre_ParCSRMatrixColStarts(SN);
   hypre_CSRMatrix    *SN_diag = hypre_ParCSRMatrixDiag(SN);
   int		      *SN_diag_i = hypre_CSRMatrixI(SN_diag);
   int		      *SN_diag_j = hypre_CSRMatrixJ(SN_diag);
   double	      *SN_diag_data;
   hypre_CSRMatrix    *SN_offd = hypre_ParCSRMatrixOffd(SN);
   int		      *SN_offd_i = hypre_CSRMatrixI(SN_offd);
   int		      *SN_offd_j = hypre_CSRMatrixJ(SN_offd);
   double	      *SN_offd_data;
   int		      *CF_marker;
   HYPRE_BigInt	      *col_map_offd_SN = hypre_ParCSRMatrixColMapOffd(SN);
   HYPRE_BigInt	      *col_map_offd_S;
   int		      *dof_func;
   int		       num_nodes = hypre_CSRMatrixNumRows(SN_diag);
   int		       num_variables;
   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(SN);
   int		       num_sends;
   int		       num_recvs;
   int		      *send_procs;
   int		      *send_map_starts;
   int		      *send_map_elmts;
   int		      *recv_procs;
   int		      *recv_vec_starts;
   hypre_ParCSRCommPkg *comm_pkg_S;
   int		      *send_procs_S;
   int		      *send_map_starts_S;
   int		      *send_map_elmts_S;
   int		      *recv_procs_S;
   int		      *recv_vec_starts_S;
   int		      *col_offd_S_to_A = NULL;
   
   int		       num_coarse_nodes;
   int		       i,j,k,k1,jj,cnt;
   int		       row, start, end;
   int		       num_procs;
   int		       num_cols_offd_SN = hypre_CSRMatrixNumCols(SN_offd);
   int		       num_cols_offd_S;
   int		       SN_num_nonzeros_diag;
   int		       SN_num_nonzeros_offd;
   int		       S_num_nonzeros_diag;
   int		       S_num_nonzeros_offd;
   HYPRE_BigInt	       global_num_vars;
   HYPRE_BigInt	       global_num_cols;
   HYPRE_BigInt	       global_num_nodes;
   int		       ierr = 0;
 
   MPI_Comm_size(comm, &num_procs);
 
   num_variables = num_functions*num_nodes;
   CF_marker = hypre_CTAlloc(int, num_variables);

   if (nodal < 0)
   {
      cnt = 0;
      num_coarse_nodes = 0;
      for (i=0; i < num_nodes; i++)
      {
	 if (CFN_marker[i] == 1) num_coarse_nodes++;
         for (j=0; j < num_functions; j++)
	    CF_marker[cnt++] = CFN_marker[i];
      }

      dof_func = hypre_CTAlloc(int,num_coarse_nodes*num_functions);
      cnt = 0;
      for (i=0; i < num_nodes; i++)
      {
	 if (CFN_marker[i] == 1)
	 {
	    for (k=0; k < num_functions; k++)
	       dof_func[cnt++] = k;
	 }
      }
      *dof_func_ptr = dof_func;
   }
   else
   {
      cnt = 0;
      for (i=0; i < num_nodes; i++)
         for (j=0; j < num_functions; j++)
	    CF_marker[cnt++] = CFN_marker[i];
   }

   *CF_marker_ptr = CF_marker;


#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_starts_S = hypre_CTAlloc(HYPRE_BigInt,2);
   for (i=0; i < 2; i++)
      row_starts_S[i] = num_functions*row_starts_SN[i];

   if (row_starts_SN != col_starts_SN)
   {
      col_starts_S = hypre_CTAlloc(HYPRE_BigInt,2);
      for (i=0; i < 2; i++)
         col_starts_S[i] = num_functions*col_starts_SN[i];
   }
   else
   {
      col_starts_S = row_starts_S;
   }
#else
   row_starts_S = hypre_CTAlloc(HYPRE_BigInt,num_procs+1);
   for (i=0; i < num_procs+1; i++)
      row_starts_S[i] = num_functions*row_starts_SN[i];

   if (row_starts_SN != col_starts_SN)
   {
      col_starts_S = hypre_CTAlloc(HYPRE_BigInt,num_procs+1);
      for (i=0; i < num_procs+1; i++)
         col_starts_S[i] = num_functions*col_starts_SN[i];
   }
   else
   {
      col_starts_S = row_starts_S;
   }
#endif


   SN_num_nonzeros_diag = SN_diag_i[num_nodes];
   SN_num_nonzeros_offd = SN_offd_i[num_nodes];
 
   global_num_nodes = hypre_ParCSRMatrixGlobalNumRows(SN);
   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(SN)*num_functions;
 
   global_num_vars = global_num_nodes*num_functions;
   S_num_nonzeros_diag = num_functions*SN_num_nonzeros_diag;
   S_num_nonzeros_offd = num_functions*SN_num_nonzeros_offd;
   num_cols_offd_S = num_functions*num_cols_offd_SN;
   S = hypre_ParCSRMatrixCreate(comm, global_num_vars, global_num_cols,
		row_starts_S, col_starts_S, num_cols_offd_S,
		S_num_nonzeros_diag, S_num_nonzeros_offd);

   S_diag = hypre_ParCSRMatrixDiag(S);
   S_offd = hypre_ParCSRMatrixOffd(S);
   S_diag_i = hypre_CTAlloc(int, num_variables+1);
   S_offd_i = hypre_CTAlloc(int, num_variables+1);
   S_diag_j = hypre_CTAlloc(int, S_num_nonzeros_diag);
   hypre_CSRMatrixI(S_diag) = S_diag_i;
   hypre_CSRMatrixJ(S_diag) = S_diag_j;
   if (data) 
   {
      SN_diag_data = hypre_CSRMatrixData(SN_diag);
      S_diag_data = hypre_CTAlloc(double, S_num_nonzeros_diag);
      hypre_CSRMatrixData(S_diag) = S_diag_data;
      if (num_cols_offd_S)
      {
         SN_offd_data = hypre_CSRMatrixData(SN_offd);
         S_offd_data = hypre_CTAlloc(double, S_num_nonzeros_offd);
         hypre_CSRMatrixData(S_offd) = S_offd_data;
      }

   }
   hypre_CSRMatrixI(S_offd) = S_offd_i;

   if (comm_pkg)
   {
      comm_pkg_S = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
      hypre_ParCSRCommPkgComm(comm_pkg_S) = comm;
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      hypre_ParCSRCommPkgNumSends(comm_pkg_S) = num_sends;
      num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      hypre_ParCSRCommPkgNumRecvs(comm_pkg_S) = num_recvs;
      send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      send_procs_S = NULL;
      send_map_elmts_S = NULL;
      if (num_sends) 
      {
         send_procs_S = hypre_CTAlloc(int,num_sends);
         send_map_elmts_S = hypre_CTAlloc(int,
		num_functions*send_map_starts[num_sends]);
      }
      send_map_starts_S = hypre_CTAlloc(int,num_sends+1);
      recv_vec_starts_S = hypre_CTAlloc(int,num_recvs+1);
      recv_procs_S = NULL;
      if (num_recvs) recv_procs_S = hypre_CTAlloc(int,num_recvs);
      send_map_starts_S[0] = 0;
      for (i=0; i < num_sends; i++)
      {
         send_procs_S[i] = send_procs[i];
         send_map_starts_S[i+1] = num_functions*send_map_starts[i+1];
      }
      recv_vec_starts_S[0] = 0;
      for (i=0; i < num_recvs; i++)
      {
         recv_procs_S[i] = recv_procs[i];
         recv_vec_starts_S[i+1] = num_functions*recv_vec_starts[i+1];
      }

      cnt = 0;
      for (i=0; i < send_map_starts[num_sends]; i++)
      {
	 k1 = num_functions*send_map_elmts[i];
         for (j=0; j < num_functions; j++)
         {
	    send_map_elmts_S[cnt++] = k1+j;
         }
      }
      hypre_ParCSRCommPkgSendProcs(comm_pkg_S) = send_procs_S;
      hypre_ParCSRCommPkgSendMapStarts(comm_pkg_S) = send_map_starts_S;
      hypre_ParCSRCommPkgSendMapElmts(comm_pkg_S) = send_map_elmts_S;
      hypre_ParCSRCommPkgRecvProcs(comm_pkg_S) = recv_procs_S;
      hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_S) = recv_vec_starts_S;
      hypre_ParCSRMatrixCommPkg(S) = comm_pkg_S;
   }

   if (num_cols_offd_S)
   {
      S_offd_j = hypre_CTAlloc(int, S_num_nonzeros_offd);
      hypre_CSRMatrixJ(S_offd) = S_offd_j;

      col_map_offd_S = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_S);

      cnt = 0;
      for (i=0; i < num_cols_offd_SN; i++)
      {
         k1 = col_map_offd_SN[i]*num_functions;
         for (j=0; j < num_functions; j++)
            col_map_offd_S[cnt++] = k1+j;
      }
      hypre_ParCSRMatrixColMapOffd(S) = col_map_offd_S;
   }
   
   if (col_offd_SN_to_AN)
   {
      col_offd_S_to_A = hypre_CTAlloc(int, num_cols_offd_S);

      cnt = 0;
      for (i=0; i < num_cols_offd_SN; i++)
      {
         k1 = col_offd_SN_to_AN[i]*num_functions;
         for (j=0; j < num_functions; j++)
            col_offd_S_to_A[cnt++] = k1+j;
      }
      *col_offd_S_to_A_ptr = col_offd_S_to_A;
   }
   


   cnt = 0;
   row = 0;
   for (i=0; i < num_nodes; i++)
   {
      row++;
      start = cnt;
      for (j=SN_diag_i[i]; j < SN_diag_i[i+1]; j++)
      {
         jj = SN_diag_j[j];
	 if (data) S_diag_data[cnt] = SN_diag_data[j];
	 S_diag_j[cnt++] = jj*num_functions;
      }
      end = cnt;
      S_diag_i[row] = cnt;
      for (k1=1; k1 < num_functions; k1++)
      {
         row++;
	 for (k=start; k < end; k++)
	 {
	    if (data) S_diag_data[cnt] = S_diag_data[k];
	    S_diag_j[cnt++] = S_diag_j[k]+k1;
	 }
         S_diag_i[row] = cnt;
      }
   } 

   cnt = 0;
   row = 0;
   for (i=0; i < num_nodes; i++)
   {
      row++;
      start = cnt;
      for (j=SN_offd_i[i]; j < SN_offd_i[i+1]; j++)
      {
         jj = SN_offd_j[j];
	 if (data) S_offd_data[cnt] = SN_offd_data[j];
	 S_offd_j[cnt++] = jj*num_functions;
      }
      end = cnt;
      S_offd_i[row] = cnt;
      for (k1=1; k1 < num_functions; k1++)
      {
         row++;
	 for (k=start; k < end; k++)
	 {
	    if (data) S_offd_data[cnt] = S_offd_data[k];
	    S_offd_j[cnt++] = S_offd_j[k]+k1;
	 }
         S_offd_i[row] = cnt;
      }
   } 

   *S_ptr = S; 

   return (ierr);
}


/* This function just finds the scalaer CF_marker and dof_func */

int
hypre_BoomerAMGCreateScalarCF(int                   *CFN_marker,
                              int                    num_functions,
                              int                    num_nodes,
                              int                  **dof_func_ptr,
                              int                  **CF_marker_ptr)

{
   int		      *CF_marker;
   int		      *dof_func;
   int		       num_variables;
   int		       num_coarse_nodes;
   int		       i,j,k,cnt;
   int		       ierr = 0;
 
 
   num_variables = num_functions*num_nodes;
   CF_marker = hypre_CTAlloc(int, num_variables);

   cnt = 0;
   num_coarse_nodes = 0;
   for (i=0; i < num_nodes; i++)
   {
      if (CFN_marker[i] == 1) num_coarse_nodes++;
      for (j=0; j < num_functions; j++)
         CF_marker[cnt++] = CFN_marker[i];
   }

   
   dof_func = hypre_CTAlloc(int,num_coarse_nodes*num_functions);
   cnt = 0;
   for (i=0; i < num_nodes; i++)
   {
      if (CFN_marker[i] == 1)
      {
         for (k=0; k < num_functions; k++)
            dof_func[cnt++] = k;
      }
   }
   

   *dof_func_ptr = dof_func;
   *CF_marker_ptr = CF_marker;


   return (ierr);
}
