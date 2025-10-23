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




#include "utilities.h"
#include "../parcsr_mv/parcsr_mv.h"
                                                                                                               
/* The following function was formerly part of hypre_ParMatmul
   but was removed so it can also be used for multiplication of
   Boolean matrices
*/

void hypre_ParMatmul_RowSizes
( int ** C_diag_i, int ** C_offd_i, int ** B_marker,
  int * A_diag_i, int * A_diag_j, int * A_offd_i, int * A_offd_j,
  int * B_diag_i, int * B_diag_j, int * B_offd_i, int * B_offd_j,
  int * B_ext_diag_i, int * B_ext_diag_j, 
  int * B_ext_offd_i, int * B_ext_offd_j, int * map_B_to_C,
  int *C_diag_size, int *C_offd_size,
  int num_rows_diag_A, int num_cols_offd_A, int allsquare,
  int num_cols_diag_B, int num_cols_offd_B, int num_cols_offd_C
)
{
   int i1, i2, i3, jj2, jj3;
   int jj_count_diag, jj_count_offd, jj_row_begin_diag, jj_row_begin_offd;
   int start_indexing = 0; /* start indexing for C_data at 0 */
   /* First pass begins here.  Computes sizes of C rows.
      Arrays computed: C_diag_i, C_offd_i, B_marker
      Arrays needed: (11, all int*)
        A_diag_i, A_diag_j, A_offd_i, A_offd_j,
        B_diag_i, B_diag_j, B_offd_i, B_offd_j,
        B_ext_i, B_ext_j, col_map_offd_B,
        col_map_offd_B, B_offd_i, B_offd_j, B_ext_i, B_ext_j,
      Scalars computed: C_diag_size, C_offd_size
      Scalars needed:
      num_rows_diag_A, num_rows_diag_A, num_cols_offd_A, allsquare,
      first_col_diag_B, n_cols_B, num_cols_offd_B, num_cols_diag_B
   */

   *C_diag_i = hypre_CTAlloc(int, num_rows_diag_A+1);
   *C_offd_i = hypre_CTAlloc(int, num_rows_diag_A+1);

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for (i1 = 0; i1 < num_cols_diag_B+num_cols_offd_C; i1++)
   {      
      (*B_marker)[i1] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over rows of A
    *-----------------------------------------------------------------------*/
   
   for (i1 = 0; i1 < num_rows_diag_A; i1++)
   {
      
      /*--------------------------------------------------------------------
       *  Set marker for diagonal entry, C_{i1,i1} (for square matrices). 
       *--------------------------------------------------------------------*/
 
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      if ( allsquare ) {
         (*B_marker)[i1] = jj_count_diag;
         jj_count_diag++;
      }

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/
         
	 if (num_cols_offd_A)
	 {
           for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
           {
            i2 = A_offd_j[jj2];
 
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_ext.
                *-----------------------------------------------------------*/
 
               for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2+1]; jj3++)
               {
                  i3 = num_cols_diag_B+B_ext_offd_j[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if ((*B_marker)[i3] < jj_row_begin_offd)
                  {
                     	(*B_marker)[i3] = jj_count_offd;
                     	jj_count_offd++;
		  } 
               }
               for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2+1]; jj3++)
               {
                  i3 = B_ext_diag_j[jj3];
                  
                  if ((*B_marker)[i3] < jj_row_begin_diag)
                  {
                  	(*B_marker)[i3] = jj_count_diag;
                     	jj_count_diag++;
		  } 
               }
            }
         }
         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/
         
         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];
 
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_diag.
                *-----------------------------------------------------------*/
 
               for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2+1]; jj3++)
               {
                  i3 = B_diag_j[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/
 
                  if ((*B_marker)[i3] < jj_row_begin_diag)
                  {
                     (*B_marker)[i3] = jj_count_diag;
                     jj_count_diag++;
                  }
               }
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_offd.
                *-----------------------------------------------------------*/

	       if (num_cols_offd_B)
	       { 
                 for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2+1]; jj3++)
                 {
                  i3 = num_cols_diag_B+map_B_to_C[B_offd_j[jj3]];
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/
 
                  if ((*B_marker)[i3] < jj_row_begin_offd)
                  {
                     (*B_marker)[i3] = jj_count_offd;
                     jj_count_offd++;
                  }
                 }
            }
      }
            
      /*--------------------------------------------------------------------
       * Set C_diag_i and C_offd_i for this row.
       *--------------------------------------------------------------------*/
 
      (*C_diag_i)[i1] = jj_row_begin_diag;
      (*C_offd_i)[i1] = jj_row_begin_offd;
      
   }
  
   (*C_diag_i)[num_rows_diag_A] = jj_count_diag;
   (*C_offd_i)[num_rows_diag_A] = jj_count_offd;
 
   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   *C_diag_size = jj_count_diag;
   *C_offd_size = jj_count_offd;

   /* End of First Pass */
}

/*--------------------------------------------------------------------------
 * hypre_ParMatmul : multiplies two ParCSRMatrices A and B and returns
 * the product in ParCSRMatrix C
 * Note that C does not own the partitionings since its row_starts
 * is owned by A and col_starts by B.
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *hypre_ParMatmul( hypre_ParCSRMatrix  *A,
				     hypre_ParCSRMatrix  *B)
{
   MPI_Comm 	   comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   
   double          *A_diag_data = hypre_CSRMatrixData(A_diag);
   int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   
   double          *A_offd_data = hypre_CSRMatrixData(A_offd);
   int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_BigInt *row_starts_A = hypre_ParCSRMatrixRowStarts(A);
   int	num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);
   int	num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);
   int	num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);
   
   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);
   
   double          *B_diag_data = hypre_CSRMatrixData(B_diag);
   int             *B_diag_i = hypre_CSRMatrixI(B_diag);
   int             *B_diag_j = hypre_CSRMatrixJ(B_diag);

   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   HYPRE_BigInt	   *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);
   
   double          *B_offd_data = hypre_CSRMatrixData(B_offd);
   int             *B_offd_i = hypre_CSRMatrixI(B_offd);
   int             *B_offd_j = hypre_CSRMatrixJ(B_offd);

   HYPRE_BigInt	first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_BigInt	last_col_diag_B;
   HYPRE_BigInt *col_starts_B = hypre_ParCSRMatrixColStarts(B);
   int	num_rows_diag_B = hypre_CSRMatrixNumRows(B_diag);
   int	num_cols_diag_B = hypre_CSRMatrixNumCols(B_diag);
   int	num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);

   hypre_ParCSRMatrix *C;
   HYPRE_BigInt	      *col_map_offd_C;
   int		      *map_B_to_C;

   hypre_CSRMatrix *C_diag;

   double          *C_diag_data;
   int             *C_diag_i;
   int             *C_diag_j;

   hypre_CSRMatrix *C_offd;

   double          *C_offd_data=NULL;
   int             *C_offd_i=NULL;
   int             *C_offd_j=NULL;

   int              C_diag_size;
   int              C_offd_size;
   int		    num_cols_offd_C = 0;
   
   hypre_BigCSRMatrix *Bs_ext;
   
   double          *Bs_ext_data;
   int             *Bs_ext_i;
   HYPRE_BigInt    *Bs_ext_j;
   HYPRE_BigInt    *temp;

   double          *B_ext_diag_data;
   int             *B_ext_diag_i;
   int             *B_ext_diag_j;
   int              B_ext_diag_size;

   double          *B_ext_offd_data;
   int             *B_ext_offd_i;
   int             *B_ext_offd_j;
   int              B_ext_offd_size;

   int		   *B_marker;

   int              i, j;
   int              i1, i2, i3;
   int              jj2, jj3;
   
   int              jj_count_diag, jj_count_offd;
   int              jj_row_begin_diag, jj_row_begin_offd;
   int              start_indexing = 0; /* start indexing for C_data at 0 */
   HYPRE_BigInt	    n_rows_A, n_cols_A;
   HYPRE_BigInt	    n_rows_B, n_cols_B;
   HYPRE_BigInt	    value;
   int              allsquare = 0;
   int              cnt, cnt_offd, cnt_diag;
   int 		    num_procs;

   double           a_entry;
   double           a_b_product;
   
   double           zero = 0.0;

   n_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = hypre_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);

   if (n_cols_A != n_rows_B || num_cols_diag_A != num_rows_diag_B)
   {
        hypre_error_in_arg(1);
	printf(" Error! Incompatible matrix dimensions!\n");
	return NULL;
   }
   if ( num_rows_diag_A==num_cols_diag_B) allsquare = 1;

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product 
    *-----------------------------------------------------------------------*/

   MPI_Comm_size(comm, &num_procs);

   if (num_procs > 1)
   {
       /*---------------------------------------------------------------------
    	* If there exists no CommPkg for A, a CommPkg is generated using
    	* equally load balanced partitionings within 
	* hypre_ParCSRMatrixExtractBExt
    	*--------------------------------------------------------------------*/
   	Bs_ext = hypre_ParCSRMatrixExtractBigExt(B,A,1);
   	Bs_ext_data = hypre_BigCSRMatrixData(Bs_ext);
   	Bs_ext_i    = hypre_BigCSRMatrixI(Bs_ext);
   	Bs_ext_j    = hypre_BigCSRMatrixJ(Bs_ext);
   }
   B_ext_diag_i = hypre_CTAlloc(int, num_cols_offd_A+1);
   B_ext_offd_i = hypre_CTAlloc(int, num_cols_offd_A+1);
   B_ext_diag_size = 0;
   B_ext_offd_size = 0;
   last_col_diag_B = first_col_diag_B + (HYPRE_BigInt) num_cols_diag_B -1;

   for (i=0; i < num_cols_offd_A; i++)
   {
      for (j=Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
            B_ext_offd_size++;
         else
            B_ext_diag_size++;
      B_ext_diag_i[i+1] = B_ext_diag_size;
      B_ext_offd_i[i+1] = B_ext_offd_size;
   }

   if (B_ext_diag_size)
   {
      B_ext_diag_j = hypre_CTAlloc(int, B_ext_diag_size);
      B_ext_diag_data = hypre_CTAlloc(double, B_ext_diag_size);
   }
   if (B_ext_offd_size)
   {
      B_ext_offd_j = hypre_CTAlloc(int, B_ext_offd_size);
      B_ext_offd_data = hypre_CTAlloc(double, B_ext_offd_size);
   }

   cnt_offd = 0;
   cnt_diag = 0;
   for (i=0; i < num_cols_offd_A; i++)
   {
      for (j=Bs_ext_i[i]; j < Bs_ext_i[i+1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
         {
            Bs_ext_j[cnt_offd] = Bs_ext_j[j];
            B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
         }
         else
         {
            B_ext_diag_j[cnt_diag] = (int) (Bs_ext_j[j] - first_col_diag_B);
            B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
         }
   }

   cnt = 0;
   if (B_ext_offd_size || num_cols_offd_B)
   {
      temp = hypre_CTAlloc(HYPRE_BigInt, B_ext_offd_size+num_cols_offd_B);
      for (i=0; i < B_ext_offd_size; i++)
         temp[i] = Bs_ext_j[i];
      cnt = B_ext_offd_size;
      for (i=0; i < num_cols_offd_B; i++)
         temp[cnt++] = col_map_offd_B[i];
   }
   if (cnt)
   {
      hypre_BigQsort0(temp, 0, cnt-1);

      num_cols_offd_C = 1;
      value = temp[0];
      for (i=1; i < cnt; i++)
      {
         if (temp[i] > value)
         {
            value = temp[i];
            temp[num_cols_offd_C++] = value;
         }
      }
   }

   if (num_cols_offd_C)
        col_map_offd_C = hypre_CTAlloc(HYPRE_BigInt,num_cols_offd_C);

   for (i=0; i < num_cols_offd_C; i++)
      col_map_offd_C[i] = temp[i];

   if (B_ext_offd_size || num_cols_offd_B)
      hypre_TFree(temp);

   for (i=0 ; i < B_ext_offd_size; i++)
      B_ext_offd_j[i] = hypre_BigBinarySearch(col_map_offd_C,
                                           Bs_ext_j[i],
                                           num_cols_offd_C);
   if (num_procs > 1)
   {
      hypre_BigCSRMatrixDestroy(Bs_ext);
      Bs_ext = NULL;
   }

   if (num_cols_offd_B)
   {
      map_B_to_C = hypre_CTAlloc(int,num_cols_offd_B);

      cnt = 0;
      for (i=0; i < num_cols_offd_C; i++)
         if (col_map_offd_C[i] == col_map_offd_B[cnt])
         {
            map_B_to_C[cnt++] = i;
            if (cnt == num_cols_offd_B) break;
         }
   }

   /*-----------------------------------------------------------------------
   *  Allocate marker array.
    *-----------------------------------------------------------------------*/

   B_marker = hypre_CTAlloc(int, num_cols_diag_B+num_cols_offd_C);

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   for (i1 = 0; i1 < num_cols_diag_B+num_cols_offd_C; i1++)
   {      
      B_marker[i1] = -1;
   }


   hypre_ParMatmul_RowSizes(
      &C_diag_i, &C_offd_i, &B_marker,
      A_diag_i, A_diag_j, A_offd_i, A_offd_j,
      B_diag_i, B_diag_j, B_offd_i, B_offd_j,
      B_ext_diag_i, B_ext_diag_j, B_ext_offd_i, B_ext_offd_j,
      map_B_to_C,
      &C_diag_size, &C_offd_size,
      num_rows_diag_A, num_cols_offd_A, allsquare,
      num_cols_diag_B, num_cols_offd_B,
      num_cols_offd_C
      );


   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/
 
   last_col_diag_B = first_col_diag_B + (HYPRE_BigInt) num_cols_diag_B - 1;
   C_diag_data = hypre_CTAlloc(double, C_diag_size);
   C_diag_j    = hypre_CTAlloc(int, C_diag_size);
   if (C_offd_size)
   { 
   	C_offd_data = hypre_CTAlloc(double, C_offd_size);
   	C_offd_j    = hypre_CTAlloc(int, C_offd_size);
   } 


   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_data and C_diag_j.
    *  Second Pass: Fill in C_offd_data and C_offd_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for (i1 = 0; i1 < num_cols_diag_B+num_cols_offd_C; i1++)
   {      
      B_marker[i1] = -1;
   }
   
   /*-----------------------------------------------------------------------
    *  Loop over interior c-points.
    *-----------------------------------------------------------------------*/
    
   for (i1 = 0; i1 < num_rows_diag_A; i1++)
   {
      
      /*--------------------------------------------------------------------
       *  Create diagonal entry, C_{i1,i1} 
       *--------------------------------------------------------------------*/

      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      if ( allsquare ) {
         B_marker[i1] = jj_count_diag;
         C_diag_data[jj_count_diag] = zero;
         C_diag_j[jj_count_diag] = i1;
         jj_count_diag++;
      }

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/
         
	 if (num_cols_offd_A)
	 {
	  for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
          {
            i2 = A_offd_j[jj2];
            a_entry = A_offd_data[jj2];
            
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_ext.
                *-----------------------------------------------------------*/

               for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2+1]; jj3++)
               {
                  i3 = num_cols_diag_B+B_ext_offd_j[jj3];
                  a_b_product = a_entry * B_ext_offd_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/
                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     	B_marker[i3] = jj_count_offd;
                     	C_offd_data[jj_count_offd] = a_b_product;
                     	C_offd_j[jj_count_offd] = i3-num_cols_diag_B;
                     	jj_count_offd++;
		  }
		  else
                    	C_offd_data[B_marker[i3]] += a_b_product;
               }
               for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2+1]; jj3++)
               {
                  i3 = B_ext_diag_j[jj3];
                  a_b_product = a_entry * B_ext_diag_data[jj3];
                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     	B_marker[i3] = jj_count_diag;
                     	C_diag_data[jj_count_diag] = a_b_product;
                     	C_diag_j[jj_count_diag] = i3;
                     	jj_count_diag++;
		  }
		  else
                     	C_diag_data[B_marker[i3]] += a_b_product;
               }
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/

         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];
            a_entry = A_diag_data[jj2];
            
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_diag.
                *-----------------------------------------------------------*/

               for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2+1]; jj3++)
               {
                  i3 = B_diag_j[jj3];
                  a_b_product = a_entry * B_diag_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     C_diag_data[jj_count_diag] = a_b_product;
                     C_diag_j[jj_count_diag] = i3;
                     jj_count_diag++;
                  }
                  else
                  {
                     C_diag_data[B_marker[i3]] += a_b_product;
                  }
               }
               if (num_cols_offd_B)
	       {
		for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2+1]; jj3++)
                {
                  i3 = num_cols_diag_B+map_B_to_C[B_offd_j[jj3]];
                  a_b_product = a_entry * B_offd_data[jj3];
                  
                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     C_offd_data[jj_count_offd] = a_b_product;
                     C_offd_j[jj_count_offd] = i3-num_cols_diag_B;
                     jj_count_offd++;
                  }
                  else
                  {
                     C_offd_data[B_marker[i3]] += a_b_product;
                  }
                }
               }
         }
   }

   C = hypre_ParCSRMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
	col_starts_B, num_cols_offd_C, C_diag_size, C_offd_size);

/* Note that C does not own the partitionings */
   hypre_ParCSRMatrixSetRowStartsOwner(C,0);
   hypre_ParCSRMatrixSetColStartsOwner(C,0);

   C_diag = hypre_ParCSRMatrixDiag(C);
   hypre_CSRMatrixData(C_diag) = C_diag_data; 
   hypre_CSRMatrixI(C_diag) = C_diag_i; 
   hypre_CSRMatrixJ(C_diag) = C_diag_j; 

   C_offd = hypre_ParCSRMatrixOffd(C);
   hypre_CSRMatrixI(C_offd) = C_offd_i; 
   hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      hypre_CSRMatrixData(C_offd) = C_offd_data; 
      hypre_CSRMatrixJ(C_offd) = C_offd_j; 
      hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   }

   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/

   hypre_TFree(B_marker);   
   hypre_TFree(B_ext_diag_i);
   if (B_ext_diag_size)
   {
      hypre_TFree(B_ext_diag_j);
      hypre_TFree(B_ext_diag_data);
   }
   hypre_TFree(B_ext_offd_i);
   if (B_ext_offd_size)
   {
      hypre_TFree(B_ext_offd_j);
      hypre_TFree(B_ext_offd_data);
   }
   if (num_cols_offd_B) hypre_TFree(map_B_to_C);

   return C;
   
}            

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractConvBExt : extracts rows from B which are located on 
 * other processors and needed for multiplication with A locally. The rows
 * are returned as CSRMatrix. Column indices are converted to local
 * indices with columns belonging to immediate neighbors marked as negative;
 * indices belonging to points on neighbor processors that are more than
 * distance one away are eliminated.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix * 
hypre_ParCSRMatrixExtractConvBExt( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A, int data)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(B);
   HYPRE_BigInt first_col_diag = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(B);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   int num_recvs;
   int *recv_vec_starts;
   int num_sends;
   int *send_map_starts;
   int *send_map_elmts;
 
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(B);

   int *diag_i = hypre_CSRMatrixI(diag);
   int *diag_j = hypre_CSRMatrixJ(diag);
   double *diag_data = hypre_CSRMatrixData(diag);

   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(B);

   int *offd_i = hypre_CSRMatrixI(offd);
   int *offd_j = hypre_CSRMatrixJ(offd);
   double *offd_data = hypre_CSRMatrixData(offd);
   int num_cols_offd = hypre_CSRMatrixNumCols(offd);

   int num_cols_B, num_nonzeros;
   int num_rows_B_ext;

   hypre_CSRMatrix *B_ext;

   int *B_ext_i;
   HYPRE_BigInt *B_tmp_j;
   int *B_ext_j;
   double *B_ext_data;

   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg *tmp_comm_pkg;

   int *B_int_i;
   HYPRE_BigInt *B_int_j;
   double * B_int_data;

   int num_procs, my_id;
   int *jdata_recv_vec_starts;
   int *jdata_send_map_starts;
 
   int i, j, k, counter, cnt;
   int start_index;
   int j_cnt, j_cnt_rm, jrow;
   HYPRE_BigInt big_k;
   HYPRE_BigInt col_1, col_n;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   num_cols_B = hypre_CSRMatrixNumRows(diag);
   col_1 = first_col_diag;
   col_n = first_col_diag + (HYPRE_BigInt) num_cols_B;
   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings 
    *--------------------------------------------------------------------*/
   if (!hypre_ParCSRMatrixCommPkg(A))
   {
              hypre_MatvecCommPkgCreate(A);
   }
    
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
 
   num_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];

   num_rows_B_ext = recv_vec_starts[num_recvs];
   if ( num_rows_B_ext < 0 )
   {  /* no B_ext, no communication */
      B_ext_i = NULL;
      B_ext_j = NULL;
      if ( data ) B_ext_data = NULL;
      num_nonzeros = 0;
      return 0;
   };
   B_int_i = hypre_CTAlloc(int, send_map_starts[num_sends]+1);
   B_ext_i = hypre_CTAlloc(int, num_rows_B_ext+1);

/*--------------------------------------------------------------------------
 * generate B_int_i through adding number of row-elements of offd and diag
 * for corresponding rows. B_int_i[j+1] contains the number of elements of
 * a row j (which is determined through send_map_elmts) 
 *--------------------------------------------------------------------------*/
   B_int_i[0] = 0;
   j_cnt = 0;
   j_cnt_rm = 0;
   num_nonzeros = 0;
   for (i=0; i < num_sends; i++)
   {
      for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
      {
         jrow = send_map_elmts[j];
         B_int_i[++j_cnt] = offd_i[jrow+1] - offd_i[jrow]
			  + diag_i[jrow+1] - diag_i[jrow];
	 num_nonzeros += B_int_i[j_cnt];
      }
   }

/*--------------------------------------------------------------------------
 * initialize communication 
 *--------------------------------------------------------------------------*/
   comm_handle = hypre_ParCSRCommHandleCreate(11,comm_pkg,
		&B_int_i[1],&(B_ext_i[1]) );

   B_int_j = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros);
   if (data) B_int_data = hypre_CTAlloc(double, num_nonzeros);

   jdata_send_map_starts = hypre_CTAlloc(int, num_sends+1);
   jdata_recv_vec_starts = hypre_CTAlloc(int, num_recvs+1);
   start_index = B_int_i[0];
   jdata_send_map_starts[0] = start_index;
   counter = 0;
   for (i=0; i < num_sends; i++)
   {
	num_nonzeros = counter;
	for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
	{
	    jrow = send_map_elmts[j];
	    for (k=diag_i[jrow]; k < diag_i[jrow+1]; k++) 
	    {
		B_int_j[counter] = (HYPRE_BigInt) diag_j[k]+first_col_diag;
		if (data) B_int_data[counter] = diag_data[k];
		counter++;
  	    }
	    for (k=offd_i[jrow]; k < offd_i[jrow+1]; k++) 
	    {
		B_int_j[counter] = col_map_offd[offd_j[k]];
		if (data) B_int_data[counter] = offd_data[k];
		counter++;
  	    }
	   
	}
	num_nonzeros = counter - num_nonzeros;
	start_index += num_nonzeros;
        jdata_send_map_starts[i+1] = start_index;
   }

   tmp_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
   hypre_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
   hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
   hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgSendProcs(tmp_comm_pkg) = hypre_ParCSRCommPkgSendProcs(comm_pkg);
   hypre_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = jdata_send_map_starts; 

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

/*--------------------------------------------------------------------------
 * after communication exchange B_ext_i[j+1] contains the number of elements
 * of a row j ! 
 * evaluate B_ext_i and compute *num_nonzeros for B_ext 
 *--------------------------------------------------------------------------*/

   for (i=0; i < num_recvs; i++)
	for (j = recv_vec_starts[i]; j < recv_vec_starts[i+1]; j++)
		B_ext_i[j+1] += B_ext_i[j];

   num_nonzeros = B_ext_i[num_rows_B_ext];

   B_tmp_j = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros);
   B_ext_j = hypre_CTAlloc(int, num_nonzeros);

   if (data) 
      B_ext_data = hypre_CTAlloc(double, num_nonzeros);

   for (i=0; i < num_recvs; i++)
   {
	start_index = B_ext_i[recv_vec_starts[i]];
	num_nonzeros = B_ext_i[recv_vec_starts[i+1]]-start_index;
	jdata_recv_vec_starts[i+1] = B_ext_i[recv_vec_starts[i+1]];
   }


   hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = jdata_recv_vec_starts;

   comm_handle = hypre_ParCSRCommHandleCreate(21,tmp_comm_pkg,B_int_j,B_tmp_j);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   if (data)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(1,tmp_comm_pkg,B_int_data,
						B_ext_data);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   hypre_TFree(jdata_send_map_starts);
   hypre_TFree(jdata_recv_vec_starts);
   hypre_TFree(tmp_comm_pkg);
   hypre_TFree(B_int_i);
   hypre_TFree(B_int_j);
   if (data) hypre_TFree(B_int_data);

   cnt = 0;
   for (i=0; i < num_rows_B_ext; i++)
   {
      for (j=B_ext_i[i]; j < B_ext_i[i+1]; j++)
      {
         big_k = B_tmp_j[j];
         if (big_k >= col_1 && big_k < col_n)
         {
            if (data) B_ext_data[cnt] = B_ext_data[j];
            B_ext_j[cnt++] = (int)(big_k - col_1);
         }
         else
         {
            k = hypre_BigBinarySearch(col_map_offd,big_k,num_cols_offd);
            if (k > -1)
            {
               if (data) B_ext_data[cnt] = B_ext_data[j];
               B_ext_j[cnt++] = -k-1;
            }
         }
      }
      B_ext_i[i] = cnt;
   }
   for (i = num_rows_B_ext; i > 0; i--)
      B_ext_i[i] = B_ext_i[i-1];
   if (num_procs > 1) B_ext_i[0] = 0;


   B_ext = hypre_CSRMatrixCreate(num_rows_B_ext,num_cols_B,num_nonzeros);
   hypre_CSRMatrixI(B_ext) = B_ext_i;
   hypre_CSRMatrixJ(B_ext) = B_ext_j;
   if (data) hypre_CSRMatrixData(B_ext) = B_ext_data;

   return B_ext;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractBigExt : extracts rows from B which are located on 
 * other processors and needed for multiplication with A locally. The rows
 * are returned as BigCSRMatrix.
 *--------------------------------------------------------------------------*/

hypre_BigCSRMatrix * 
hypre_ParCSRMatrixExtractBigExt( hypre_ParCSRMatrix *B, hypre_ParCSRMatrix *A, int data)
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(B);
   HYPRE_BigInt first_col_diag = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(B);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   int num_recvs;
   int *recv_vec_starts;
   int num_sends;
   int *send_map_starts;
   int *send_map_elmts;
 
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(B);

   int *diag_i = hypre_CSRMatrixI(diag);
   int *diag_j = hypre_CSRMatrixJ(diag);
   double *diag_data = hypre_CSRMatrixData(diag);

   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(B);

   int *offd_i = hypre_CSRMatrixI(offd);
   int *offd_j = hypre_CSRMatrixJ(offd);
   double *offd_data = hypre_CSRMatrixData(offd);

   int num_cols_B, num_nonzeros;

   hypre_BigCSRMatrix *B_ext;

   int *B_ext_i;
   HYPRE_BigInt *B_ext_j;
   double *B_ext_data;

   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg *tmp_comm_pkg;

   int *B_int_i;
   HYPRE_BigInt *B_int_j;
   double * B_int_data;

   int num_procs, my_id;
   int *jdata_recv_vec_starts;
   int *jdata_send_map_starts;
 
   int i, j, k, counter;
   int start_index;
   int j_cnt, j_cnt_rm, jrow;
   int num_rows_B_ext;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);
   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings 
    *--------------------------------------------------------------------*/
   if (!hypre_ParCSRMatrixCommPkg(A))
   {
              hypre_MatvecCommPkgCreate(A);
   }
    
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
 
   num_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];

   num_rows_B_ext = recv_vec_starts[num_recvs];
   if ( num_rows_B_ext < 0 )
   {  /* no B_ext, no communication */
      B_ext_i = NULL;
      B_ext_j = NULL;
      if ( data ) B_ext_data = NULL;
      num_nonzeros = 0;
      return 0;
   };
   B_int_i = hypre_CTAlloc(int, send_map_starts[num_sends]+1);
   B_ext_i = hypre_CTAlloc(int, num_rows_B_ext+1);

/*--------------------------------------------------------------------------
 * generate B_int_i through adding number of row-elements of offd and diag
 * for corresponding rows. B_int_i[j+1] contains the number of elements of
 * a row j (which is determined through send_map_elmts) 
 *--------------------------------------------------------------------------*/
   B_int_i[0] = 0;
   j_cnt = 0;
   j_cnt_rm = 0;
   num_nonzeros = 0;
   for (i=0; i < num_sends; i++)
   {
      for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
      {
         jrow = send_map_elmts[j];
         B_int_i[++j_cnt] = offd_i[jrow+1] - offd_i[jrow]
			  + diag_i[jrow+1] - diag_i[jrow];
	 num_nonzeros += B_int_i[j_cnt];
      }
   }

/*--------------------------------------------------------------------------
 * initialize communication 
 *--------------------------------------------------------------------------*/
   comm_handle = hypre_ParCSRCommHandleCreate(11,comm_pkg,
		&B_int_i[1],&(B_ext_i[1]) );

   B_int_j = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros);
   if (data) B_int_data = hypre_CTAlloc(double, num_nonzeros);

   jdata_send_map_starts = hypre_CTAlloc(int, num_sends+1);
   jdata_recv_vec_starts = hypre_CTAlloc(int, num_recvs+1);
   start_index = B_int_i[0];
   jdata_send_map_starts[0] = start_index;
   counter = 0;
   for (i=0; i < num_sends; i++)
   {
	num_nonzeros = counter;
	for (j = send_map_starts[i]; j < send_map_starts[i+1]; j++)
	{
	    jrow = send_map_elmts[j];
	    for (k=diag_i[jrow]; k < diag_i[jrow+1]; k++) 
	    {
		B_int_j[counter] = (HYPRE_BigInt) diag_j[k]+first_col_diag;
		if (data) B_int_data[counter] = diag_data[k];
		counter++;
  	    }
	    for (k=offd_i[jrow]; k < offd_i[jrow+1]; k++) 
	    {
		B_int_j[counter] = col_map_offd[offd_j[k]];
		if (data) B_int_data[counter] = offd_data[k];
		counter++;
  	    }
	   
	}
	num_nonzeros = counter - num_nonzeros;
	start_index += num_nonzeros;
        jdata_send_map_starts[i+1] = start_index;
   }

   tmp_comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg,1);
   hypre_ParCSRCommPkgComm(tmp_comm_pkg) = comm;
   hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
   hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgSendProcs(tmp_comm_pkg) = hypre_ParCSRCommPkgSendProcs(comm_pkg);
   hypre_ParCSRCommPkgRecvProcs(tmp_comm_pkg) = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) = jdata_send_map_starts; 

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

/*--------------------------------------------------------------------------
 * after communication exchange B_ext_i[j+1] contains the number of elements
 * of a row j ! 
 * evaluate B_ext_i and compute *num_nonzeros for B_ext 
 *--------------------------------------------------------------------------*/

   for (i=0; i < num_recvs; i++)
	for (j = recv_vec_starts[i]; j < recv_vec_starts[i+1]; j++)
		B_ext_i[j+1] += B_ext_i[j];

   num_nonzeros = B_ext_i[num_rows_B_ext];

   B_ext_j = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros);

   if (data) 
      B_ext_data = hypre_CTAlloc(double, num_nonzeros);

   for (i=0; i < num_recvs; i++)
   {
	start_index = B_ext_i[recv_vec_starts[i]];
	num_nonzeros = B_ext_i[recv_vec_starts[i+1]]-start_index;
	jdata_recv_vec_starts[i+1] = B_ext_i[recv_vec_starts[i+1]];
   }


   hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) = jdata_recv_vec_starts;

   comm_handle = hypre_ParCSRCommHandleCreate(21,tmp_comm_pkg,B_int_j,B_ext_j);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   if (data)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(1,tmp_comm_pkg,B_int_data,
						B_ext_data);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   hypre_TFree(jdata_send_map_starts);
   hypre_TFree(jdata_recv_vec_starts);
   hypre_TFree(tmp_comm_pkg);
   hypre_TFree(B_int_i);
   hypre_TFree(B_int_j);
   if (data) hypre_TFree(B_int_data);

   B_ext = hypre_BigCSRMatrixCreate(num_rows_B_ext,num_cols_B,num_nonzeros);
   hypre_CSRMatrixI(B_ext) = B_ext_i;
   hypre_CSRMatrixJ(B_ext) = B_ext_j;
   if (data) hypre_CSRMatrixData(B_ext) = B_ext_data;

   return B_ext;
}
