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
 * IJMatrix_ParCSR interface
 *
 *****************************************************************************/
 
#include "headers.h"

#include "../HYPRE.h"

/******************************************************************************
 *
 * hypre_IJMatrixCreateParCSR
 *
 *****************************************************************************/

int
hypre_IJMatrixCreateParCSR(hypre_IJMatrix *matrix)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   hypre_ParCSRMatrix *par_matrix;
   HYPRE_BigInt *row_starts;
   HYPRE_BigInt *col_starts;
   int num_procs;
   int i;

   MPI_Comm_size(comm,&num_procs);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   row_starts = hypre_CTAlloc(HYPRE_BigInt,2);
   if (hypre_IJMatrixGlobalFirstRow(matrix))
      for (i=0; i < 2; i++)
	 row_starts[i] = row_partitioning[i]- hypre_IJMatrixGlobalFirstRow(matrix);
   else
      for (i=0; i < 2; i++)
	 row_starts[i] = row_partitioning[i];
   if (row_partitioning != col_partitioning)
   {
      col_starts = hypre_CTAlloc(HYPRE_BigInt,2);
      if (hypre_IJMatrixGlobalFirstCol(matrix))
	 for (i=0; i < 2; i++)
	    col_starts[i] = col_partitioning[i]-hypre_IJMatrixGlobalFirstCol(matrix);
      else
	 for (i=0; i < 2; i++)
	    col_starts[i] = col_partitioning[i];
   }
   else
      col_starts = row_starts;

   par_matrix = hypre_ParCSRMatrixCreate(comm, hypre_IJMatrixGlobalNumRows(matrix),
		 hypre_IJMatrixGlobalNumCols(matrix),row_starts, col_starts, 0, 0, 0);

#else
   row_starts = hypre_CTAlloc(HYPRE_BigInt,num_procs+1);
   if (row_partitioning[0])
	 for (i=0; i < num_procs+1; i++)
	 row_starts[i] = row_partitioning[i]-row_partitioning[0];
   else
	 for (i=0; i < num_procs+1; i++)
	 row_starts[i] = row_partitioning[i];
   if (row_partitioning != col_partitioning)
   {
	 col_starts = hypre_CTAlloc(HYPRE_BigInt,num_procs+1);
      if (col_partitioning[0])
	 for (i=0; i < num_procs+1; i++)
	    col_starts[i] = col_partitioning[i]-col_partitioning[0];
      else
	 for (i=0; i < num_procs+1; i++)
	    col_starts[i] = col_partitioning[i];
   }
   else
	 col_starts = row_starts;
   par_matrix = hypre_ParCSRMatrixCreate(comm,row_starts[num_procs],
		col_starts[num_procs],row_starts, col_starts, 0, 0, 0);
#endif

   hypre_IJMatrixObject(matrix) = par_matrix;

   
   return hypre_error_flag;
 
}


/******************************************************************************
 *
 * hypre_IJMatrixSetRowSizesParCSR
 *
 *****************************************************************************/
int
hypre_IJMatrixSetRowSizesParCSR(hypre_IJMatrix *matrix,
			      	const int      *sizes)
{
   int local_num_rows, local_num_cols;
   int i, my_id;
   int *row_space;
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   hypre_AuxParCSRMatrix *aux_matrix;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);

   MPI_Comm_rank(comm,&my_id);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   local_num_rows = (int) (row_partitioning[1]-row_partitioning[0]);
   local_num_cols = (int)(col_partitioning[1]-col_partitioning[0]);
#else
   local_num_rows = (int)(row_partitioning[my_id+1]-row_partitioning[my_id]);
   local_num_cols = (int)(col_partitioning[my_id+1]-col_partitioning[my_id]);
#endif
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   row_space = NULL;
   if (aux_matrix)
      row_space =  hypre_AuxParCSRMatrixRowSpace(aux_matrix);
   if (!row_space)
      row_space = hypre_CTAlloc(int, local_num_rows);
   for (i = 0; i < local_num_rows; i++)
      row_space[i] = sizes[i];
   if (!aux_matrix)
   {
      hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, 
				local_num_cols, row_space);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }
   hypre_AuxParCSRMatrixRowSpace(aux_matrix) = row_space;

   return hypre_error_flag;

}

/******************************************************************************
 *
 * hypre_IJMatrixSetDiagOffdSizesParCSR
 * sets diag_i inside the diag part of the ParCSRMatrix
 * and offd_i inside the offd part,
 * requires exact row sizes for diag and offd
 *
 *****************************************************************************/
int
hypre_IJMatrixSetDiagOffdSizesParCSR(hypre_IJMatrix *matrix,
			      	     const int	   *diag_sizes,
			      	     const int	   *offdiag_sizes)
{
   int local_num_rows;
   int i;
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixObject(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = hypre_IJMatrixTranslator(matrix);
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;
   int *diag_i;
   int *offd_i;

   if (!par_matrix)
   {
      hypre_IJMatrixCreateParCSR(matrix);
      par_matrix = hypre_IJMatrixObject(matrix);
   }
   
   diag =  hypre_ParCSRMatrixDiag(par_matrix);
   diag_i =  hypre_CSRMatrixI(diag); 
   local_num_rows = hypre_CSRMatrixNumRows(diag); 
   if (!diag_i) 
      diag_i = hypre_CTAlloc(int, local_num_rows+1); 
   for (i = 0; i < local_num_rows; i++) 
      diag_i[i+1] = diag_i[i] + diag_sizes[i]; 
   hypre_CSRMatrixI(diag) = diag_i; 
   hypre_CSRMatrixNumNonzeros(diag) = diag_i[local_num_rows]; 
   offd =  hypre_ParCSRMatrixOffd(par_matrix); 
   offd_i =  hypre_CSRMatrixI(offd); 
   if (!offd_i)
      offd_i = hypre_CTAlloc(int, local_num_rows+1);
   for (i = 0; i < local_num_rows; i++)
      offd_i[i+1] = offd_i[i] + offdiag_sizes[i];
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixNumNonzeros(offd) = offd_i[local_num_rows];
   if (!aux_matrix)
   {
      hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, 
				hypre_CSRMatrixNumCols(diag), NULL);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }
   hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
   if (offd_i[local_num_rows])
	hypre_AuxParCSRMatrixAuxOffdJ(aux_matrix) = 
		hypre_CTAlloc(HYPRE_BigInt, offd_i[local_num_rows]);

   return hypre_error_flag;

}

/******************************************************************************
 *
 * hypre_IJMatrixSetMaxOffProcElmtsParCSR
 *
 *****************************************************************************/

int
hypre_IJMatrixSetMaxOffProcElmtsParCSR(hypre_IJMatrix *matrix,
			      	       int max_off_proc_elmts)
{
   hypre_AuxParCSRMatrix *aux_matrix;
   int local_num_rows, local_num_cols, my_id;
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   MPI_Comm comm = hypre_IJMatrixComm(matrix);

   MPI_Comm_rank(comm,&my_id);
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   if (!aux_matrix)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      local_num_rows = (int)( row_partitioning[1]-row_partitioning[0]);
      local_num_cols = (int)( col_partitioning[1]-col_partitioning[0]);
#else
      local_num_rows = (int)( row_partitioning[my_id+1]-row_partitioning[my_id]);
      local_num_cols = (int)( col_partitioning[my_id+1]-col_partitioning[my_id]);
#endif
      hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, 
				local_num_cols, NULL);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }
   hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = max_off_proc_elmts;

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixInitializeParCSR
 *
 * initializes AuxParCSRMatrix and ParCSRMatrix as necessary
 *
 *****************************************************************************/

int
hypre_IJMatrixInitializeParCSR(hypre_IJMatrix *matrix)
{
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixObject(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = hypre_IJMatrixTranslator(matrix);
   int local_num_rows;

   if (hypre_IJMatrixAssembleFlag(matrix) == 0)
   {
      if (!par_matrix)
      {
         hypre_IJMatrixCreateParCSR(matrix);
         par_matrix = hypre_IJMatrixObject(matrix);
      }
      local_num_rows = 
		hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(par_matrix));
      if (!aux_matrix)
      {
         hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, 
		hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(par_matrix)), 
		NULL);
         hypre_IJMatrixTranslator(matrix) = aux_matrix;
      }
     
      hypre_ParCSRMatrixInitialize(par_matrix);
      hypre_AuxParCSRMatrixInitialize(aux_matrix);
      if (! hypre_AuxParCSRMatrixNeedAux(aux_matrix))
      {
         int i, *indx_diag, *indx_offd, *diag_i, *offd_i;
         diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix));
         offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix));
         indx_diag = hypre_AuxParCSRMatrixIndxDiag(aux_matrix);
 
         indx_offd = hypre_AuxParCSRMatrixIndxOffd(aux_matrix);
         for (i=0; i < local_num_rows; i++)
         {
	    indx_diag[i] = diag_i[i];
	    indx_offd[i] = offd_i[i];
         }
      }
   }
   else /* AB 4/06 - the assemble routine destroys the aux matrix - so we need
           to recreate if initialize is called again*/
   {
      if (!aux_matrix)
      {
         local_num_rows = 
		hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(par_matrix));
         hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, 
                                            hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(par_matrix)), 
                                            NULL);
         hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
         hypre_IJMatrixTranslator(matrix) = aux_matrix;
      }

   }
   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixGetRowCountsParCSR
 *
 * gets the number of columns for rows specified by the user
 * 
 *****************************************************************************/

int hypre_IJMatrixGetRowCountsParCSR( hypre_IJMatrix *matrix,
                               int	       nrows,
                               HYPRE_BigInt   *rows,
                               int	      *ncols)
{
   HYPRE_BigInt row_index;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixObject(matrix);

   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(par_matrix);
   int *diag_i = hypre_CSRMatrixI(diag);

   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(par_matrix);
   int *offd_i = hypre_CSRMatrixI(offd);

   int i, my_id;

   MPI_Comm_rank(comm,&my_id);

   for (i=0; i < nrows; i++)
   {
      row_index = rows[i];
#ifdef HYPRE_NO_GLOBAL_PARTITION
      if (row_index >= row_partitioning[0] && 
		row_index < row_partitioning[1])
      {
   	/* compute local row number */
         row_index -= row_partitioning[0]; 
#else
      if (row_index >= row_partitioning[my_id] && 
		row_index < row_partitioning[my_id+1])
      {
   	/* compute local row number */
         row_index -= row_partitioning[my_id]; 
#endif
         ncols[i] = diag_i[row_index+1]-diag_i[row_index]+offd_i[row_index+1]
		-offd_i[row_index];
      }
      else
      {
         ncols[i] = 0;
#ifdef HYPRE_LONG_LONG
	 printf ("Warning! Row %lld is not on Proc. %d!\n", row_index, my_id);
#else
	 printf ("Warning! Row %d is not on Proc. %d!\n", row_index, my_id);
#endif
      }
   }

  return hypre_error_flag;

}
/******************************************************************************
 *
 * hypre_IJMatrixGetValuesParCSR
 *
 * gets values of an IJMatrix
 * 
 *****************************************************************************/
int
hypre_IJMatrixGetValuesParCSR( hypre_IJMatrix *matrix,
                               int	       nrows,
                               int	      *ncols,
                               HYPRE_BigInt   *rows,
                               HYPRE_BigInt   *cols,
                               double         *values)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixObject(matrix);
   int assemble_flag = hypre_IJMatrixAssembleFlag(matrix);

   hypre_CSRMatrix *diag;
   int *diag_i;
   int *diag_j;
   double *diag_data;

   hypre_CSRMatrix *offd;
   int *offd_i;
   int *offd_j;
   double *offd_data;

   HYPRE_BigInt *col_map_offd;
   HYPRE_BigInt *col_starts = hypre_ParCSRMatrixColStarts(par_matrix);

   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);

#ifndef HYPRE_NO_GLOBAL_PARTITION
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
#endif

   int i, j, n, ii, indx;
   HYPRE_BigInt col_indx;
   int num_procs, my_id;
   HYPRE_BigInt col_0, col_n, row;
   int row_local, row_size;
   int warning = 0;
   int *counter;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (assemble_flag == 0)
   {
      hypre_error_in_arg(1);
      printf("Error! Matrix not assembled yet! HYPRE_IJMatrixGetValues\n");
   }

#ifdef HYPRE_NO_GLOBAL_PARTITION
   col_0 = col_starts[0];
   col_n = col_starts[1]-1;
#else
   col_0 = col_starts[my_id];
   col_n = col_starts[my_id+1]-1;
#endif

   diag = hypre_ParCSRMatrixDiag(par_matrix);
   diag_i = hypre_CSRMatrixI(diag);
   diag_j = hypre_CSRMatrixJ(diag);
   diag_data = hypre_CSRMatrixData(diag);
   
   offd = hypre_ParCSRMatrixOffd(par_matrix);
   offd_i = hypre_CSRMatrixI(offd);
   if (num_procs > 1)
   {
      offd_j = hypre_CSRMatrixJ(offd);
      offd_data = hypre_CSRMatrixData(offd);
      col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
   }

   if (nrows < 0)
   {
      nrows = -nrows;
      
      counter = hypre_CTAlloc(int,nrows+1);
      counter[0] = 0;
      for (i=0; i < nrows; i++)
         counter[i+1] = counter[i]+ncols[i];

      indx = 0;   
      for (i=0; i < nrows; i++)
      {
         row = rows[i];
#ifdef HYPRE_NO_GLOBAL_PARTITION
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (int)(row - row_partitioning[0]); 
#else
         if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = (int)(row - row_partitioning[my_id]); 
#endif
            row_size = diag_i[row_local+1]-diag_i[row_local]+
			offd_i[row_local+1]-offd_i[row_local];
            if (counter[i]+row_size > counter[nrows])
	    {
               hypre_error_in_arg(1);
	       printf ("Error! Not enough memory! HYPRE_IJMatrixGetValues\n");
	    }
            if (ncols[i] < row_size)
		warning = 1;
	    for (j = diag_i[row_local]; j < diag_i[row_local+1]; j++)
 	    {
	       cols[indx] = (HYPRE_BigInt) diag_j[j] + col_0;
	       values[indx++] = diag_data[j];
	    }
	    for (j = offd_i[row_local]; j < offd_i[row_local+1]; j++)
 	    {
	       cols[indx] = col_map_offd[offd_j[j]];
	       values[indx++] = offd_data[j];
	    }
	    counter[i+1] = indx;
         }
         else
#ifdef HYPRE_LONG_LONG
	    printf ("Warning! Row %lld is not on Proc. %d!\n", row, my_id);
#else
	    printf ("Warning! Row %d is not on Proc. %d!\n", row, my_id);
#endif
      }
      if (warning)
      {
         for (i=0; i < nrows; i++)
	    ncols[i] = counter[i+1] - counter[i];
	 printf ("Warning!  ncols has been changed!\n");
      }
      hypre_TFree(counter);
   }
   else
   {
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
#ifdef HYPRE_NO_GLOBAL_PARTITION
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (int)(row - row_partitioning[0]); 
            /* compute local row number */
     	    for (i=0; i < n; i++)
   	    {
   	       col_indx = cols[indx] -  hypre_IJMatrixGlobalFirstCol(matrix);

#else
         if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = (int)(row - row_partitioning[my_id]); 
   				/* compute local row number */
     	    for (i=0; i < n; i++)
   	    {
   	       col_indx = cols[indx] - col_partitioning[0];
#endif



   	       values[indx] = 0.0;
   	       if (col_indx < col_0 || col_indx > col_n)
   				/* search in offd */	
   	       {
   	          for (j=offd_i[row_local]; j < offd_i[row_local+1]; j++)
   	          {
   		     if (col_map_offd[offd_j[j]] == col_indx)
   		     {
                        values[indx] = offd_data[j];
   		        break;
   		     }
   	          }
	       }
	       else  /* search in diag */
	       {
   	          col_indx = col_indx - col_0;
	          for (j=diag_i[row_local]; j < diag_i[row_local+1]; j++)
	          {
		     if (diag_j[j] == (int)col_indx)
		     {
                        values[indx] = diag_data[j];
		        break;
		     }
	          } 
	       }
	       indx++;
	    }
         }
         else
#ifdef HYPRE_LONG_LONG
	    printf ("Warning! Row %lld is not on Proc. %d!\n", row, my_id);
#else
	    printf ("Warning! Row %d is not on Proc. %d!\n", row, my_id);
#endif
      }
   }

   return hypre_error_flag;
   
}
/******************************************************************************
 *
 * hypre_IJMatrixSetValuesParCSR
 *
 * sets or adds row values to an IJMatrix before assembly, 
 * 
 *****************************************************************************/
int
hypre_IJMatrixSetValuesParCSR( hypre_IJMatrix *matrix,
                               int	       nrows,
                               int	      *ncols,
                               const HYPRE_BigInt *rows,
                               const HYPRE_BigInt *cols,
                               const double   *values)
{
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag, *offd;
   hypre_AuxParCSRMatrix *aux_matrix;
   HYPRE_BigInt *row_partitioning;
   HYPRE_BigInt *col_partitioning;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   int num_procs, my_id;
   int row_local;
   HYPRE_BigInt col_0, col_n, row;
   int i, ii, j, n, not_found;
   HYPRE_BigInt **aux_j;
   HYPRE_BigInt *local_j;
   HYPRE_BigInt *tmp_j;
   double **aux_data;
   double *local_data;
   double *tmp_data;
   int diag_space, offd_space;
   int *row_length, *row_space;
   int need_aux;
   int tmp_indx, indx;
   int space, size, old_size;
   int cnt, cnt_diag, cnt_offd;
   int pos_diag, pos_offd;
   int len_diag, len_offd;
   int offd_indx, diag_indx;
   int *diag_i;
   int *diag_j;
   double *diag_data;
   int *offd_i;
   int *offd_j;
   HYPRE_BigInt *aux_offd_j;
   double *offd_data;
   HYPRE_BigInt first;
   int current_num_elmts;
   int max_off_proc_elmts;
   int off_proc_i_indx;
   HYPRE_BigInt *off_proc_i;
   HYPRE_BigInt *off_proc_j;
   double *off_proc_data;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);
   par_matrix = hypre_IJMatrixObject( matrix );
   row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   col_partitioning = hypre_IJMatrixColPartitioning(matrix);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   col_0 = col_partitioning[0];
   col_n = col_partitioning[1]-1;
   first =  hypre_IJMatrixGlobalFirstCol(matrix);
#else
   col_0 = col_partitioning[my_id];
   col_n = col_partitioning[my_id+1]-1;
   first = col_partitioning[0];
#endif
   if (nrows < 0)
   {
      hypre_error_in_arg(2);
      printf("Error! nrows negative! HYPRE_IJMatrixSetValues\n");
   }

   if (hypre_IJMatrixAssembleFlag(matrix))
   {
      HYPRE_BigInt *col_map_offd;
      int num_cols_offd;
      int j_offd;
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];

         /* processor owns the row */ 

#ifdef HYPRE_NO_GLOBAL_PARTITION
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (int)(row - row_partitioning[0]); 
#else
         if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = (int)(row - row_partitioning[my_id]); 
#endif

				/* compute local row number */
            diag = hypre_ParCSRMatrixDiag(par_matrix);
            diag_i = hypre_CSRMatrixI(diag);
            diag_j = hypre_CSRMatrixJ(diag);
            diag_data = hypre_CSRMatrixData(diag);
            offd = hypre_ParCSRMatrixOffd(par_matrix);
            offd_i = hypre_CSRMatrixI(offd);
            num_cols_offd = hypre_CSRMatrixNumCols(offd);
            if (num_cols_offd)
            {
               col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
               offd_j = hypre_CSRMatrixJ(offd);
               offd_data = hypre_CSRMatrixData(offd);
            }
            size = diag_i[row_local+1] - diag_i[row_local]
      			+ offd_i[row_local+1] - offd_i[row_local];
      
            if (n > size)
            {
               hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
      	       printf (" row %lld too long! \n", row);
#else
      	       printf (" row %d too long! \n", row);
#endif
               return hypre_error_flag;
            }
       
            pos_diag = diag_i[row_local];
            pos_offd = offd_i[row_local];
            len_diag = diag_i[row_local+1];
            len_offd = offd_i[row_local+1];
            not_found = 1;
      	
            for (i=0; i < n; i++)
            {
               if (cols[indx] < col_0 || cols[indx] > col_n)
				/* insert into offd */	
               {
      	          j_offd = hypre_BigBinarySearch(col_map_offd,cols[indx]-first,
						num_cols_offd);
      	          if (j_offd == -1)
      	          {
                     hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
      	             printf (" Error, element %lld %lld does not exist\n",
				row, cols[indx]);
#else
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
#endif
                     return hypre_error_flag;
      	          }
      	          for (j=pos_offd; j < len_offd; j++)
      	          {
      	             if (offd_j[j] == j_offd)
      	             {
                        offd_data[j] = values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
                     hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
      	             printf (" Error, element %lld %lld does not exist\n",
				row, cols[indx]);
#else
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
#endif
                     return hypre_error_flag;
      	          }
      	          not_found = 1;
               }
               /* diagonal element */
      	       else if (cols[indx] == row)
      	       {
      	          if (diag_j[pos_diag] != row_local)
      	          {
                     hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
      	             printf (" Error, element %lld %lld does not exist\n",
				row, cols[indx]);
#else
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
#endif
      	             /* return -1;*/
                     return hypre_error_flag;
      	          }
      	          diag_data[pos_diag] = values[indx];
      	       }
               else  /* insert into diag */
               {
      	          for (j=pos_diag; j < len_diag; j++)
      	          {
      	             if (diag_j[j] == (int)(cols[indx]-col_0))
      	             {
                        diag_data[j] = values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
                     hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
      	             printf (" Error, element %lld %lld does not exist\n",
				row, cols[indx]);
#else
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
#endif
      	             /* return -1; */
                     return hypre_error_flag;
      	          }
               }
               indx++;
            }
         }
         
         /* processor does not own the row */  
        
         else
	 {
            aux_matrix = hypre_IJMatrixTranslator(matrix);
   	    if (!aux_matrix)
            {
#ifdef HYPRE_NO_GLOBAL_PARTITION
               size = (int)(row_partitioning[1]-row_partitioning[0]);
#else
               size = (int)(row_partitioning[my_id+1]-row_partitioning[my_id]);
#endif
	       hypre_AuxParCSRMatrixCreate(&aux_matrix, size, size, NULL);
      	       hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
      	       hypre_IJMatrixTranslator(matrix) = aux_matrix;
            }
   	    current_num_elmts 
		= hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	    max_off_proc_elmts
		= hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
   	    off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	    off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	    off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	    off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
   	    
	    if (!max_off_proc_elmts)
	    {
	       max_off_proc_elmts = hypre_max(n,1000);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) =
	       		max_off_proc_elmts;
   	       hypre_AuxParCSRMatrixOffProcI(aux_matrix)
			= hypre_CTAlloc(HYPRE_BigInt,2*max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix)
			= hypre_CTAlloc(HYPRE_BigInt,max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcData(aux_matrix)
			= hypre_CTAlloc(double,max_off_proc_elmts);
   	       off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	       off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
	    }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3*n;
               off_proc_i = hypre_TReAlloc(off_proc_i,HYPRE_BigInt,2*max_off_proc_elmts);
               off_proc_j = hypre_TReAlloc(off_proc_j,HYPRE_BigInt,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
				max_off_proc_elmts);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix)
   	    		= max_off_proc_elmts;
	       hypre_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
	       hypre_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
	    }
            off_proc_i[off_proc_i_indx++] = row; 
            off_proc_i[off_proc_i_indx++] = n; 
	    for (i=0; i < n; i++)
	    {
	       off_proc_j[current_num_elmts] = cols[indx];
	       off_proc_data[current_num_elmts++] = values[indx++];
	    }
	    hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx; 
	    hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix)
   	    	= current_num_elmts; 
	 }
      } /* end of for loop */
   }
   else
   {
      aux_matrix = hypre_IJMatrixTranslator(matrix);
      row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
         /* processor owns the row */ 
#ifdef HYPRE_NO_GLOBAL_PARTITION
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (int)(row - row_partitioning[0]); 
#else
         if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = (int)(row - row_partitioning[my_id]); 

#endif
			/* compute local row number */
            if (need_aux)
            {
               aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
               aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
               local_j = aux_j[row_local];
               local_data = aux_data[row_local];
   	       space = row_space[row_local]; 
   	       old_size = row_length[row_local]; 
   	       size = space - old_size;
   	       if (size < n)
      	       {
      	          size = n - size;
      	          tmp_j = hypre_CTAlloc(HYPRE_BigInt,size);
      	          tmp_data = hypre_CTAlloc(double,size);
      	       }
      	       else
      	       {
      	          tmp_j = NULL;
      	       }
      	       tmp_indx = 0;
      	       not_found = 1;
      	       size = old_size;
               for (i=0; i < n; i++)
      	       {
      	          for (j=0; j < old_size; j++)
      	          {
      	             if (local_j[j] == cols[indx])
      	             {
                        local_data[j] = values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
      	             if (size < space)
      	             {
      	                local_j[size] = cols[indx];
      	                local_data[size++] = values[indx];
      	             }
      	             else
      	             {
      	                tmp_j[tmp_indx] = cols[indx];
      	                tmp_data[tmp_indx++] = values[indx];
      	             }
      	          }
      	          not_found = 1;
        	  indx++;
      	       }
      	    
               row_length[row_local] = size+tmp_indx;
               
               if (tmp_indx)
               {
   	          aux_j[row_local] = hypre_TReAlloc(aux_j[row_local],HYPRE_BigInt,
   				size+tmp_indx);
   	          aux_data[row_local] = hypre_TReAlloc(aux_data[row_local],
   					double,size+tmp_indx);
                  row_space[row_local] = size+tmp_indx;
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
               }
   
   	       cnt = size; 
   
   	       for (i=0; i < tmp_indx; i++)
   	       {
   	          local_j[cnt] = tmp_j[i];
   	          local_data[cnt++] = tmp_data[i];
	       }
  
	       if (tmp_j)
	       { 
	          hypre_TFree(tmp_j); 
	          hypre_TFree(tmp_data); 
	       } 
            }
            else /* insert immediately into data in ParCSRMatrix structure */
            {
	       int col_j;
	       offd_indx =hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
	       diag_indx =hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
               diag = hypre_ParCSRMatrixDiag(par_matrix);
               diag_i = hypre_CSRMatrixI(diag);
               diag_j = hypre_CSRMatrixJ(diag);
               diag_data = hypre_CSRMatrixData(diag);
               offd = hypre_ParCSRMatrixOffd(par_matrix);
               offd_i = hypre_CSRMatrixI(offd);
               if (num_procs > 1)
	       {
	          offd_j = hypre_CSRMatrixJ(offd);
                  offd_data = hypre_CSRMatrixData(offd);
	          aux_offd_j = hypre_AuxParCSRMatrixAuxOffdJ(aux_matrix);
               }
	       cnt_diag = diag_indx;
	       cnt_offd = offd_indx;
	       diag_space = diag_i[row_local+1];
	       offd_space = offd_i[row_local+1];
	       not_found = 1;
  	       for (i=0; i < n; i++)
	       {
	          if (cols[indx] < col_0 || cols[indx] > col_n)
				/* insert into offd */	
	          {
	             for (j=offd_i[row_local]; j < offd_indx; j++)
	             {
		        if (aux_offd_j[j] == cols[indx])
		        {
                           offd_data[j] = values[indx];
		           not_found = 0;
		           break;
		        }
	             }
	             if (not_found)
	             { 
	                if (cnt_offd < offd_space) 
	                { 
	                   aux_offd_j[cnt_offd] = cols[indx];
	                   offd_data[cnt_offd++] = values[indx];
	                } 
	                else 
	 	        {
                           hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
	    	           printf("Error in row %lld ! Too many elements!\n", 
				row);
#else
	    	           printf("Error in row %d ! Too many elements!\n", 
				row);
#endif
	    	           /* return 1; */
                           return hypre_error_flag;
	 	        }
	             }  
	             not_found = 1;
	          }
	          else  /* insert into diag */
	          {
	             for (j=diag_i[row_local]; j < diag_indx; j++)
	             {
		        col_j = (int)(cols[indx]-col_0);
		        if (diag_j[j] == col_j)
		        {
                           diag_data[j] = values[indx];
		           not_found = 0;
		           break;
		        }
	             } 
	             if (not_found)
	             { 
	                if (cnt_diag < diag_space) 
	                { 
	                   diag_j[cnt_diag] = col_j;
	                   diag_data[cnt_diag++] = values[indx];
	                } 
	                else 
	 	        {
                           hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
	    	           printf("Error in row %lld ! Too many elements !\n", 
				row);
#else
	    	           printf("Error in row %d ! Too many elements !\n", 
				row);
#endif
	    	           /* return 1; */
                           return hypre_error_flag;
	 	        }
	             } 
	             not_found = 1;
	          }
	          indx++;
	       }

               hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
               hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;

            }
         }

         /* processor does not own the row */
         else
	 {

   	    current_num_elmts 
		= hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	    max_off_proc_elmts
		= hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
   	    off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	    off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	    off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	    off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
   	    
	    if (!max_off_proc_elmts)
	    {
	       max_off_proc_elmts = hypre_max(n,1000);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) =
	       		max_off_proc_elmts;
   	       hypre_AuxParCSRMatrixOffProcI(aux_matrix)
			= hypre_CTAlloc(HYPRE_BigInt,2*max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix)
			= hypre_CTAlloc(HYPRE_BigInt,max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcData(aux_matrix)
			= hypre_CTAlloc(double,max_off_proc_elmts);
   	       off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	       off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
	    }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3*n;
               off_proc_i = hypre_TReAlloc(off_proc_i,HYPRE_BigInt,2*max_off_proc_elmts);
               off_proc_j = hypre_TReAlloc(off_proc_j,HYPRE_BigInt,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
				max_off_proc_elmts);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix)
   	    		= max_off_proc_elmts;
	       hypre_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
	       hypre_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
	    }
            off_proc_i[off_proc_i_indx++] = row; 
            off_proc_i[off_proc_i_indx++] = n; 
	    for (i=0; i < n; i++)
	    {
	       off_proc_j[current_num_elmts] = cols[indx];
	       off_proc_data[current_num_elmts++] = values[indx++];
	    }
	    hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx; 
	    hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix)
   	    	= current_num_elmts; 
	 }
      }
   }
   return hypre_error_flag;

}

/******************************************************************************
 *
 * hypre_IJMatrixAddToValuesParCSR
 *
 * adds row values to an IJMatrix 
 * 
 *****************************************************************************/
int
hypre_IJMatrixAddToValuesParCSR( hypre_IJMatrix *matrix,
                                 int	       nrows,
                                 int	      *ncols,
                                 const HYPRE_BigInt *rows,
                                 const HYPRE_BigInt *cols,
                                 const double   *values)
{
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag, *offd;
   hypre_AuxParCSRMatrix *aux_matrix;
   HYPRE_BigInt *row_partitioning;
   HYPRE_BigInt *col_partitioning;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   int num_procs, my_id;
   int row_local;
   HYPRE_BigInt row;
   HYPRE_BigInt col_0, col_n;
   int i, ii, j, n, not_found, col_j;
   HYPRE_BigInt **aux_j;
   HYPRE_BigInt *local_j;
   HYPRE_BigInt *tmp_j;
   double **aux_data;
   double *local_data;
   double *tmp_data;
   int diag_space, offd_space;
   int *row_length, *row_space;
   int need_aux;
   int tmp_indx, indx;
   int space, size, old_size;
   int cnt, cnt_diag, cnt_offd;
   int pos_diag, pos_offd;
   int len_diag, len_offd;
   int offd_indx, diag_indx;
   int first;
   int *diag_i;
   int *diag_j;
   double *diag_data;
   int *offd_i;
   int *offd_j;
   HYPRE_BigInt *aux_offd_j;
   double *offd_data;
   int current_num_elmts;
   int max_off_proc_elmts;
   int off_proc_i_indx;
   HYPRE_BigInt *off_proc_i;
   HYPRE_BigInt *off_proc_j;
   double *off_proc_data;

   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &my_id);
   par_matrix = hypre_IJMatrixObject( matrix );
   row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   col_partitioning = hypre_IJMatrixColPartitioning(matrix);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   col_0 = col_partitioning[0];
   col_n = col_partitioning[1]-1;
   first = hypre_IJMatrixGlobalFirstCol(matrix);
#else
   col_0 = col_partitioning[my_id];
   col_n = col_partitioning[my_id+1]-1;
   first = col_partitioning[0];
#endif
   if (hypre_IJMatrixAssembleFlag(matrix))
   {


      int num_cols_offd;
      HYPRE_BigInt *col_map_offd;
      int j_offd;
      indx = 0;   


      /* AB - 4/06 - need to get this object*/
      aux_matrix = hypre_IJMatrixTranslator(matrix);

      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
#ifdef HYPRE_NO_GLOBAL_PARTITION
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (int)(row - row_partitioning[0]); 
#else
         if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = (int)(row - row_partitioning[my_id]); 
#endif
				/* compute local row number */
            diag = hypre_ParCSRMatrixDiag(par_matrix);
            diag_i = hypre_CSRMatrixI(diag);
            diag_j = hypre_CSRMatrixJ(diag);
            diag_data = hypre_CSRMatrixData(diag);
            offd = hypre_ParCSRMatrixOffd(par_matrix);
            offd_i = hypre_CSRMatrixI(offd);
            num_cols_offd = hypre_CSRMatrixNumCols(offd);
            if (num_cols_offd)
            {
               col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
               offd_j = hypre_CSRMatrixJ(offd);
               offd_data = hypre_CSRMatrixData(offd);
            }
            size = diag_i[row_local+1] - diag_i[row_local]
      			+ offd_i[row_local+1] - offd_i[row_local];
      
            if (n > size)
            {
               hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
      	       printf (" row %lld too long! \n", row);
#else
      	       printf (" row %d too long! \n", row);
#endif
      	       /* return -1; */
               return hypre_error_flag;
            }
       
            pos_diag = diag_i[row_local];
            pos_offd = offd_i[row_local];
            len_diag = diag_i[row_local+1];
            len_offd = offd_i[row_local+1];
            not_found = 1;
      	
            for (i=0; i < n; i++)
            {
               if (cols[indx] < col_0 || cols[indx] > col_n)
				/* insert into offd */	
               {
      	          j_offd = hypre_BigBinarySearch(col_map_offd,cols[indx]-first,
						num_cols_offd);
      	          if (j_offd == -1)
      	          {
                     hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
      	             printf (" Error, element %lld %lld does not exist\n",
				row, cols[indx]);
#else
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
#endif
                     return hypre_error_flag;
      	             /* return -1; */
      	          }
      	          for (j=pos_offd; j < len_offd; j++)
      	          {
      	             if (offd_j[j] == j_offd)
      	             {
                        offd_data[j] += values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
                     hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
      	             printf (" Error, element %lld %lld does not exist\n",
				row, cols[indx]);
#else
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
#endif
      	             /* return -1;*/
                     return hypre_error_flag;
      	          }
      	          not_found = 1;
               }
               /* diagonal element */
      	       else if (cols[indx] == row)
      	       {
      	          if (diag_j[pos_diag] != row_local)
      	          {
                     hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
      	             printf (" Error, element %lld %lld does not exist\n",
				row, cols[indx]);
#else
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
#endif
      	             /* return -1; */
                     return hypre_error_flag;
      	          }
      	          diag_data[pos_diag] += values[indx];
      	       }
               else  /* insert into diag */
               {
      	          for (j=pos_diag; j < len_diag; j++)
      	          {
      	             if (diag_j[j] == (int)(cols[indx]-col_0))
      	             {
                        diag_data[j] += values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
                     hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
      	             printf (" Error, element %lld %lld does not exist\n",
				row, cols[indx]);
#else
      	             printf (" Error, element %d %d does not exist\n",
				row, cols[indx]);
#endif
      	             /* return -1;*/
                     return hypre_error_flag;
      	          }
               }
            indx++;
            }
         }
         /* not my row */
         else
	 {
   	    if (!aux_matrix)
            {
#ifdef HYPRE_NO_GLOBAL_PARTITION
               size = (int)( row_partitioning[1]-row_partitioning[0]);
#else
               size = (int)( row_partitioning[my_id+1]-row_partitioning[my_id]);
#endif
	       hypre_AuxParCSRMatrixCreate(&aux_matrix, size, size, NULL);
      	       hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
      	       hypre_IJMatrixTranslator(matrix) = aux_matrix;
            }
   	    current_num_elmts 
		= hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	    max_off_proc_elmts
		= hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
   	    off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	    off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	    off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	    off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
   	    
	    if (!max_off_proc_elmts)
	    {
	       max_off_proc_elmts = hypre_max(n,1000);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) =
	       		max_off_proc_elmts;
   	       hypre_AuxParCSRMatrixOffProcI(aux_matrix)
			= hypre_CTAlloc(HYPRE_BigInt,2*max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix)
			= hypre_CTAlloc(HYPRE_BigInt,max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcData(aux_matrix)
			= hypre_CTAlloc(double,max_off_proc_elmts);
   	       off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	       off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
	    }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3*n;
               off_proc_i = hypre_TReAlloc(off_proc_i,HYPRE_BigInt,2*max_off_proc_elmts);
               off_proc_j = hypre_TReAlloc(off_proc_j,HYPRE_BigInt,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
				max_off_proc_elmts);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix)
   	    		= max_off_proc_elmts;
	       hypre_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
	       hypre_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
	    }

            /* AB - 4/6 - the row should be negative to indicate an add */
            /* off_proc_i[off_proc_i_indx++] = row; */
            off_proc_i[off_proc_i_indx++] = -row-1;
            
            off_proc_i[off_proc_i_indx++] = n; 
	    for (i=0; i < n; i++)
	    {
	       off_proc_j[current_num_elmts] = cols[indx];
	       off_proc_data[current_num_elmts++] = values[indx++];
	    }
	    hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx; 
	    hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix)
   	    	= current_num_elmts; 
	 }
      }
   }
   
/* not assembled */
   else
   {
      aux_matrix = hypre_IJMatrixTranslator(matrix);
      row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);
      indx = 0;   
      for (ii=0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
#ifdef HYPRE_NO_GLOBAL_PARTITION
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (int)(row - row_partitioning[0]); 
#else
         if (row >= row_partitioning[my_id] && row < row_partitioning[my_id+1])
         {
            row_local = (int)(row - row_partitioning[my_id]); 
#endif
			/* compute local row number */
            if (need_aux)
            {
               aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
               aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
               local_j = aux_j[row_local];
               local_data = aux_data[row_local];
   	       space = row_space[row_local]; 
   	       old_size = row_length[row_local]; 
   	       size = space - old_size;
   	       if (size < n)
      	       {
      	          size = n - size;
      	          tmp_j = hypre_CTAlloc(HYPRE_BigInt,size);
      	          tmp_data = hypre_CTAlloc(double,size);
      	       }
      	       else
      	       {
      	          tmp_j = NULL;
      	       }
      	       tmp_indx = 0;
      	       not_found = 1;
      	       size = old_size;
               for (i=0; i < n; i++)
      	       {
      	          for (j=0; j < old_size; j++)
      	          {
      	             if (local_j[j] == cols[indx])
      	             {
                        local_data[j] += values[indx];
      		        not_found = 0;
      		        break;
      	             }
      	          }
      	          if (not_found)
      	          {
      	             if (size < space)
      	             {
      	                local_j[size] = cols[indx];
      	                local_data[size++] = values[indx];
      	             }
      	             else
      	             {
      	                tmp_j[tmp_indx] = cols[indx];
      	                tmp_data[tmp_indx++] = values[indx];
      	             }
      	          }
      	          not_found = 1;
        	  indx++;
      	       }
      	    
               row_length[row_local] = size+tmp_indx;
               
               if (tmp_indx)
               {
   	          aux_j[row_local] = hypre_TReAlloc(aux_j[row_local],HYPRE_BigInt,
   				size+tmp_indx);
   	          aux_data[row_local] = hypre_TReAlloc(aux_data[row_local],
   					double,size+tmp_indx);
                  row_space[row_local] = size+tmp_indx;
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
               }
   
   	       cnt = size; 
   
   	       for (i=0; i < tmp_indx; i++)
   	       {
   	          local_j[cnt] = tmp_j[i];
   	          local_data[cnt++] = tmp_data[i];
	       }
  
	       if (tmp_j)
	       { 
	          hypre_TFree(tmp_j); 
	          hypre_TFree(tmp_data); 
	       } 
            }
            else /* insert immediately into data in ParCSRMatrix structure */
            {
	       offd_indx = hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
	       diag_indx = hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
               diag = hypre_ParCSRMatrixDiag(par_matrix);
               diag_i = hypre_CSRMatrixI(diag);
               diag_j = hypre_CSRMatrixJ(diag);
               diag_data = hypre_CSRMatrixData(diag);
               offd = hypre_ParCSRMatrixOffd(par_matrix);
               offd_i = hypre_CSRMatrixI(offd);
               if (num_procs > 1)
	       {
	          offd_j = hypre_CSRMatrixJ(offd);
                  offd_data = hypre_CSRMatrixData(offd);
	          aux_offd_j = hypre_AuxParCSRMatrixAuxOffdJ(aux_matrix);
               }
	       cnt_diag = diag_indx;
	       cnt_offd = offd_indx;
	       diag_space = diag_i[row_local+1];
	       offd_space = offd_i[row_local+1];
	       not_found = 1;
  	       for (i=0; i < n; i++)
	       {
	          if (cols[indx] < col_0 || cols[indx] > col_n)
				/* insert into offd */	
	          {
	             for (j=offd_i[row_local]; j < offd_indx; j++)
	             {
		        if (aux_offd_j[j] == cols[indx])
		        {
                           offd_data[j] += values[indx];
		           not_found = 0;
		           break;
		        }
	             }
	             if (not_found)
	             { 
	                if (cnt_offd < offd_space) 
	                { 
	                   aux_offd_j[cnt_offd] = cols[indx];
	                   offd_data[cnt_offd++] = values[indx];
	                } 
	                else 
	 	        {
                           hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
	    	           printf("Error in row %lld ! Too many elements!\n", 
				row);
#else
	    	           printf("Error in row %d ! Too many elements!\n", 
				row);
#endif
	    	           /* return 1;*/
                           return hypre_error_flag;
	 	        }
	             }  
	             not_found = 1;
	          }
	          else  /* insert into diag */
	          {
	             for (j=diag_i[row_local]; j < diag_indx; j++)
	             {
		        col_j = (int)(cols[indx]-col_0);
		        if (diag_j[j] == col_j)
		        {
                           diag_data[j] += values[indx];
		           not_found = 0;
		           break;
		        }
	             } 
	             if (not_found)
	             { 
	                if (cnt_diag < diag_space) 
	                { 
	                   diag_j[cnt_diag] = col_j;
	                   diag_data[cnt_diag++] = values[indx];
	                } 
	                else 
	 	        {
                           hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_LONG_LONG
	    	           printf("Error in row %lld ! Too many elements !\n", 
				row);
#else
	    	           printf("Error in row %d ! Too many elements !\n", 
				row);
#endif
	    	           /* return 1; */
                           return hypre_error_flag;
	 	        }
	             } 
	             not_found = 1;
	          }
	          indx++;
	       }

               hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
               hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;

            }
         }
         /* not my row */
         else
         {
   	    current_num_elmts 
		= hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
   	    max_off_proc_elmts
		= hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
   	    off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
   	    off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	    off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	    off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
   	    
	    if (!max_off_proc_elmts)
	    {
	       max_off_proc_elmts = hypre_max(n,1000);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) =
	       		max_off_proc_elmts;
   	       hypre_AuxParCSRMatrixOffProcI(aux_matrix)
			= hypre_CTAlloc(HYPRE_BigInt,2*max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix)
			= hypre_CTAlloc(HYPRE_BigInt,max_off_proc_elmts);
   	       hypre_AuxParCSRMatrixOffProcData(aux_matrix)
			= hypre_CTAlloc(double,max_off_proc_elmts);
   	       off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
   	       off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
   	       off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
	    }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3*n;
               off_proc_i = hypre_TReAlloc(off_proc_i,HYPRE_BigInt,2*max_off_proc_elmts);
               off_proc_j = hypre_TReAlloc(off_proc_j,HYPRE_BigInt,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
				max_off_proc_elmts);
	       hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix)
   	    		= max_off_proc_elmts;
	       hypre_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
	       hypre_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
	       hypre_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
	    }
            off_proc_i[off_proc_i_indx++] = -row-1; 
            off_proc_i[off_proc_i_indx++] = n; 
	    for (i=0; i < n; i++)
	    {
	       off_proc_j[current_num_elmts] = cols[indx];
	       off_proc_data[current_num_elmts++] = values[indx++];
	    }
	    hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx; 
	    hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix)
   	    	= current_num_elmts; 
         }
      }
   }
   return hypre_error_flag;

}

/******************************************************************************
 *
 * hypre_IJMatrixAssembleParCSR
 *
 * assembles IJMatrix from AuxParCSRMatrix auxiliary structure
 *****************************************************************************/
int
hypre_IJMatrixAssembleParCSR(hypre_IJMatrix *matrix)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixObject(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = hypre_IJMatrixTranslator(matrix);
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(par_matrix);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(par_matrix);
   int *diag_i = hypre_CSRMatrixI(diag);
   int *offd_i = hypre_CSRMatrixI(offd);
   int *diag_j;
   int *offd_j = NULL;
   double *diag_data;
   double *offd_data = NULL;
   int i, j, j0;
   int num_cols_offd;
   int *diag_pos;
   HYPRE_BigInt *col_map_offd;
   int *row_length;
   HYPRE_BigInt **aux_j;
   double **aux_data;
   int my_id, num_procs;
   int num_rows;
   int i_diag, i_offd;
   HYPRE_BigInt *local_j;
   double *local_data;
   HYPRE_BigInt col_0, col_n;
   int nnz_offd;
   HYPRE_BigInt *aux_offd_j;
   HYPRE_BigInt *tmp_j;
   double temp; 
#ifdef HYPRE_NO_GLOBAL_PARTITION
   HYPRE_BigInt base = hypre_IJMatrixGlobalFirstCol(matrix);
#else
   HYPRE_BigInt base = col_partitioning[0];
#endif
   int off_proc_i_indx;
   int max_off_proc_elmts;
   int current_num_elmts;
   HYPRE_BigInt *off_proc_i;
   HYPRE_BigInt *off_proc_j;
   double *off_proc_data;
   int offd_proc_elmts;

   if (aux_matrix)
   {
      off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
      MPI_Allreduce(&off_proc_i_indx,&offd_proc_elmts,1,MPI_INT, MPI_SUM,comm);
      if (offd_proc_elmts)
      {
          max_off_proc_elmts=hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
          current_num_elmts=hypre_AuxParCSRMatrixCurrentNumElmts(aux_matrix);
          off_proc_i=hypre_AuxParCSRMatrixOffProcI(aux_matrix);
          off_proc_j=hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
          off_proc_data=hypre_AuxParCSRMatrixOffProcData(aux_matrix);
          hypre_IJMatrixAssembleOffProcValsParCSR(matrix,off_proc_i_indx,
		max_off_proc_elmts, current_num_elmts, off_proc_i,
		off_proc_j, off_proc_data);
      }
   }

   if (hypre_IJMatrixAssembleFlag(matrix) == 0)
   {
      MPI_Comm_size(comm, &num_procs); 
      MPI_Comm_rank(comm, &my_id);
#ifdef HYPRE_NO_GLOBAL_PARTITION
      num_rows = (int)(row_partitioning[1] - row_partitioning[0]); 
      col_0 = col_partitioning[0];
      col_n = col_partitioning[1]-1;
#else
      num_rows = (int)(row_partitioning[my_id+1] - row_partitioning[my_id]); 
      col_0 = col_partitioning[my_id];
      col_n = col_partitioning[my_id+1]-1;
#endif
   /* Got till here !! Warning!!! Check this carefully!! Problems!! ***/
   
   /* move data into ParCSRMatrix if not there already */ 
      if (hypre_AuxParCSRMatrixNeedAux(aux_matrix))
      {
         aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
         aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
         row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
         diag_pos = hypre_CTAlloc(int, num_rows);
         i_diag = 0;
         i_offd = 0;
         for (i=0; i < num_rows; i++)
         {
   	    local_j = aux_j[i];
   	    local_data = aux_data[i];
   	    diag_pos[i] = -1;
   	    for (j=0; j < row_length[i]; j++)
   	    {
   	       if (local_j[j] < col_0 || local_j[j] > col_n)
   	          i_offd++;
   	       else
   	       {
   	          i_diag++;
   	          if ((int)(local_j[j]-col_0) == i) diag_pos[i] = j;
   	       }
   	    }
   	    diag_i[i+1] = i_diag;
   	    offd_i[i+1] = i_offd;
         }

         if (hypre_CSRMatrixJ(diag))
            hypre_TFree(hypre_CSRMatrixJ(diag));
         if (hypre_CSRMatrixData(diag))
            hypre_TFree(hypre_CSRMatrixData(diag));
         if (hypre_CSRMatrixJ(offd))
            hypre_TFree(hypre_CSRMatrixJ(offd));
         if (hypre_CSRMatrixData(offd))
            hypre_TFree(hypre_CSRMatrixData(offd));
         diag_j = hypre_CTAlloc(int,i_diag);
         diag_data = hypre_CTAlloc(double,i_diag);
         if (i_offd > 0)
         {
    	    offd_j = hypre_CTAlloc(int,i_offd);
            offd_data = hypre_CTAlloc(double,i_offd);
            aux_offd_j = hypre_CTAlloc(HYPRE_BigInt, i_offd);
            tmp_j = hypre_CTAlloc(HYPRE_BigInt, i_offd);
         }

         i_diag = 0;
         i_offd = 0;
         for (i=0; i < num_rows; i++)
         {
   	    local_j = aux_j[i];
   	    local_data = aux_data[i];
            if (diag_pos[i] > -1)
            {
   	       diag_j[i_diag] = (int)(local_j[diag_pos[i]] - col_0);
               diag_data[i_diag++] = local_data[diag_pos[i]];
            }
   	    for (j=0; j < row_length[i]; j++)
   	    {
   	       if (local_j[j] < col_0 || local_j[j] > col_n)
   	       {
                  aux_offd_j[i_offd] = local_j[j];
                  tmp_j[i_offd] = aux_offd_j[i_offd];
   	          offd_data[i_offd++] = local_data[j];
   	       }
   	       else if (j != diag_pos[i])
   	       {
   	          diag_j[i_diag] = (int)(local_j[j] - col_0);
   	          diag_data[i_diag++] = local_data[j];
   	       }
   	    }
         }
         hypre_CSRMatrixJ(diag) = diag_j;      
         hypre_CSRMatrixData(diag) = diag_data;      
         hypre_CSRMatrixNumNonzeros(diag) = diag_i[num_rows];      
         if (i_offd > 0)
         {
            hypre_CSRMatrixJ(offd) = offd_j;      
            hypre_CSRMatrixData(offd) = offd_data;      
         }
         hypre_CSRMatrixNumNonzeros(offd) = offd_i[num_rows];      
         hypre_TFree(diag_pos);
      }
      else
      {
         /* move diagonal element into first space */
   
         diag_j = hypre_CSRMatrixJ(diag);
         diag_data = hypre_CSRMatrixData(diag);
         for (i=0; i < num_rows; i++)
         {
   	    j0 = diag_i[i];
   	    for (j=j0; j < diag_i[i+1]; j++)
   	    {
   	       if (diag_j[j] == i)
   	       {
   	          temp = diag_data[j0];
   	          diag_data[j0] = diag_data[j];
   	          diag_data[j] = temp;
   	          diag_j[j] = diag_j[j0];
   	          diag_j[j0] = i;
   	       }
   	    }
         }
   
         offd_j = hypre_CSRMatrixJ(offd);
      }

            /*  generate the nonzero rows inside offd and diag by calling */

      hypre_CSRMatrixSetRownnz(diag);
      hypre_CSRMatrixSetRownnz(offd);
      nnz_offd = offd_i[num_rows];
   
   /*  generate col_map_offd */
      if (nnz_offd)
      {
         hypre_BigQsort0(tmp_j,0,nnz_offd-1);
         num_cols_offd = 1;
         for (i=0; i < nnz_offd-1; i++)
         {
            if (tmp_j[i+1] > tmp_j[i])
            {
               tmp_j[num_cols_offd] = tmp_j[i+1];
               num_cols_offd++;
            }
         }
         col_map_offd = hypre_CTAlloc(HYPRE_BigInt,num_cols_offd);
         for (i=0; i < num_cols_offd; i++)
   	    col_map_offd[i] = tmp_j[i];
        
         for (i=0; i < nnz_offd; i++)
         {
            offd_j[i]=hypre_BigBinarySearch(col_map_offd,aux_offd_j[i],num_cols_offd);
         }
 	 if (base)
 	 {
	    for (i=0; i < num_cols_offd; i++)
	       col_map_offd[i] -= base;
	 } 
         hypre_ParCSRMatrixColMapOffd(par_matrix) = col_map_offd;    
         hypre_CSRMatrixNumCols(offd) = num_cols_offd;    
         hypre_TFree(aux_offd_j);
         hypre_TFree(tmp_j);
      }
      hypre_AuxParCSRMatrixDestroy(aux_matrix);
      hypre_IJMatrixTranslator(matrix) = NULL;
      hypre_IJMatrixAssembleFlag(matrix) = 1;
   }
   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixDestroyParCSR
 *
 * frees an IJMatrix
 *
 *****************************************************************************/
int
hypre_IJMatrixDestroyParCSR(hypre_IJMatrix *matrix)
{
   hypre_ParCSRMatrixDestroy(hypre_IJMatrixObject(matrix));
   hypre_AuxParCSRMatrixDestroy(hypre_IJMatrixTranslator(matrix));

   return hypre_error_flag;
}







/******************************************************************************
 *
 * hypre_IJMatrixAssembleOffProcValsParCSR
 *
 * This is for handling set and get values calls to off-proc. entries -
 * it is called from matrix assemble.  There is an alternate version for
 * when the assumed partition is being used.
 *
 *****************************************************************************/

#ifndef HYPRE_NO_GLOBAL_PARTITION

int
hypre_IJMatrixAssembleOffProcValsParCSR( hypre_IJMatrix *matrix, 
   					 int off_proc_i_indx,
   					 int max_off_proc_elmts,
   					 int current_num_elmts,
   					 HYPRE_BigInt *off_proc_i,
   					 HYPRE_BigInt *off_proc_j,
   					 double *off_proc_data)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   MPI_Request *requests;
   MPI_Status *status;
   int i, ii, j, j2, jj, n;
   HYPRE_BigInt row;
   int iii, iid, indx, ip;
   int proc_id, num_procs, my_id;
   int num_sends, num_sends3;
   int num_recvs;
   int num_requests;
   int vec_start, vec_len;
   int *send_procs;
   int *chunks;
   HYPRE_BigInt *send_i;
   int *send_map_starts;
   int *dbl_send_map_starts;
   int *recv_procs;
   int *recv_chunks;
   HYPRE_BigInt *recv_i;
   int *recv_vec_starts;
   int *dbl_recv_vec_starts;
   int *info;
   int *int_buffer;
   int *proc_id_mem;
   HYPRE_BigInt *partitioning;
   int *displs;
   int *recv_buf;
   double *send_data;
   double *recv_data;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm, &my_id);
   partitioning = hypre_IJMatrixRowPartitioning(matrix);

   info = hypre_CTAlloc(int,num_procs);  
   chunks = hypre_CTAlloc(int,num_procs);  
   proc_id_mem = hypre_CTAlloc(int,off_proc_i_indx/2);
   j=0;
   for (i=0; i < off_proc_i_indx; i++)
   {
      row = off_proc_i[i++];
      if (row < 0) row = -row-1; 
      n = (int) off_proc_i[i];
      proc_id = hypre_FindProc(partitioning,row,num_procs);
      proc_id_mem[j++] = proc_id; 
      info[proc_id] += n;
      chunks[proc_id]++;
   }

   /* determine send_procs and amount of data to be sent */   
   num_sends = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         num_sends++;
      }
   }
   send_procs =  hypre_CTAlloc(int,num_sends);
   send_map_starts =  hypre_CTAlloc(int,num_sends+1);
   dbl_send_map_starts =  hypre_CTAlloc(int,num_sends+1);
   num_sends3 = 3*num_sends;
   int_buffer =  hypre_CTAlloc(int,3*num_sends);
   j = 0;
   j2 = 0;
   send_map_starts[0] = 0;
   dbl_send_map_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         send_procs[j++] = i;
         send_map_starts[j] = send_map_starts[j-1]+2*chunks[i]+info[i];
         dbl_send_map_starts[j] = dbl_send_map_starts[j-1]+info[i];
         int_buffer[j2++] = i;
	 int_buffer[j2++] = chunks[i];
	 int_buffer[j2++] = info[i];
      }
   }

   hypre_TFree(chunks);

   MPI_Allgather(&num_sends3,1,MPI_INT,info,1,MPI_INT,comm);

   displs = hypre_CTAlloc(int, num_procs+1);
   displs[0] = 0;
   for (i=1; i < num_procs+1; i++)
        displs[i] = displs[i-1]+info[i-1];
   recv_buf = hypre_CTAlloc(int, displs[num_procs]);

   MPI_Allgatherv(int_buffer,num_sends3,MPI_INT,recv_buf,info,displs,
			MPI_INT,comm);

   hypre_TFree(int_buffer);
   hypre_TFree(info);

   /* determine recv procs and amount of data to be received */
   num_recvs = 0;
   for (j=0; j < displs[num_procs]; j+=3)
   {
      if (recv_buf[j] == my_id)
	 num_recvs++;
   }

   recv_procs = hypre_CTAlloc(int,num_recvs);
   recv_chunks = hypre_CTAlloc(int,num_recvs);
   recv_vec_starts = hypre_CTAlloc(int,num_recvs+1);
   dbl_recv_vec_starts = hypre_CTAlloc(int,num_recvs+1);

   j2 = 0;
   recv_vec_starts[0] = 0;
   dbl_recv_vec_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      for (j=displs[i]; j < displs[i+1]; j+=3)
      {
         if (recv_buf[j] == my_id)
         {
	    recv_procs[j2] = i;
	    recv_chunks[j2++] = recv_buf[j+1];
	    recv_vec_starts[j2] = recv_vec_starts[j2-1]+2*recv_buf[j+1]
			+recv_buf[j+2];
	    dbl_recv_vec_starts[j2] = dbl_recv_vec_starts[j2-1]+recv_buf[j+2];
         }
         if (j2 == num_recvs) break;
      }
   }
   hypre_TFree(recv_buf);
   hypre_TFree(displs);

   /* set up data to be sent to send procs */
   /* send_i contains for each send proc : row no., no. of elmts and column
      indices, send_data contains corresponding values */
      
   send_i = hypre_CTAlloc(HYPRE_BigInt,send_map_starts[num_sends]);
   send_data = hypre_CTAlloc(double,dbl_send_map_starts[num_sends]);
   recv_i = hypre_CTAlloc(HYPRE_BigInt,recv_vec_starts[num_recvs]);
   recv_data = hypre_CTAlloc(double,dbl_recv_vec_starts[num_recvs]);
    
   j=0;
   jj=0;
   for (i=0; i < off_proc_i_indx; i++)
   {
      row = off_proc_i[i++]; 
      n = (int) off_proc_i[i];
      proc_id = proc_id_mem[i/2];
      indx = hypre_BinarySearch(send_procs,proc_id,num_sends);
      iii = send_map_starts[indx];
      iid = dbl_send_map_starts[indx];
      send_i[iii++] = row;
      send_i[iii++] = (HYPRE_BigInt) n;
      for (ii = 0; ii < n; ii++)
      {
          send_i[iii++] = off_proc_j[jj];
          send_data[iid++] = off_proc_data[jj++];
      }
      send_map_starts[indx] = iii;
      dbl_send_map_starts[indx] = iid;
   }

   hypre_TFree(proc_id_mem);

   for (i=num_sends; i > 0; i--)
   {
      send_map_starts[i] = send_map_starts[i-1];
      dbl_send_map_starts[i] = dbl_send_map_starts[i-1];
   }
   send_map_starts[0] = 0;
   dbl_send_map_starts[0] = 0;

   num_requests = num_recvs+num_sends;

   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status = hypre_CTAlloc(MPI_Status, num_requests);

   j=0; 
   for (i=0; i < num_recvs; i++)
   {
       vec_start = recv_vec_starts[i];
       vec_len = recv_vec_starts[i+1] - vec_start;
       ip = recv_procs[i];
       MPI_Irecv(&recv_i[vec_start], vec_len, MPI_HYPRE_BIG_INT, ip, 0, comm, 
			&requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
       vec_start = send_map_starts[i];
       vec_len = send_map_starts[i+1] - vec_start;
       ip = send_procs[i];
       MPI_Isend(&send_i[vec_start], vec_len, MPI_HYPRE_BIG_INT, ip, 0, comm, 
			&requests[j++]);
   }
  
   if (num_requests)
   {
      MPI_Waitall(num_requests, requests, status);
   }

   j=0;
   for (i=0; i < num_recvs; i++)
   {
       vec_start = dbl_recv_vec_starts[i];
       vec_len = dbl_recv_vec_starts[i+1] - vec_start;
       ip = recv_procs[i];
       MPI_Irecv(&recv_data[vec_start], vec_len, MPI_DOUBLE, ip, 0, comm, 
			&requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
       vec_start = dbl_send_map_starts[i];
       vec_len = dbl_send_map_starts[i+1] - vec_start;
       ip = send_procs[i];
       MPI_Isend(&send_data[vec_start], vec_len, MPI_DOUBLE, ip, 0, comm, 
			&requests[j++]);
   }
  
   if (num_requests)
   {
      MPI_Waitall(num_requests, requests, status);
      hypre_TFree(requests);
      hypre_TFree(status);
   }

   hypre_TFree(send_i);
   hypre_TFree(send_data);
   hypre_TFree(send_procs);
   hypre_TFree(send_map_starts);
   hypre_TFree(dbl_send_map_starts);
   hypre_TFree(recv_procs);
   hypre_TFree(recv_vec_starts);
   hypre_TFree(dbl_recv_vec_starts);

   j = 0;
   j2 = 0;
   for (i=0; i < num_recvs; i++)
   {
      for (ii=0; ii < recv_chunks[i]; ii++)
      {
         row = recv_i[j];
	 if (row < 0) 
	 {
	    row = -row-1;
 	    hypre_IJMatrixAddToValuesParCSR(matrix,1,(int*)&recv_i[j+1],&row,
		&recv_i[j+2],&recv_data[j2]);
	    j2 += recv_i[j+1]; 
	    j += recv_i[j+1]+2; 
	 }
	 else
	 {
 	    hypre_IJMatrixSetValuesParCSR(matrix,1,(int*)&recv_i[j+1],&row,
		&recv_i[j+2],&recv_data[j2]);
	    j2 += recv_i[j+1]; 
	    j += recv_i[j+1]+2; 
	 }
      }
   }
   hypre_TFree(recv_chunks);
   hypre_TFree(recv_i);
   hypre_TFree(recv_data);

   return hypre_error_flag;

}

#else


/* assumed partition version */

int
hypre_IJMatrixAssembleOffProcValsParCSR( hypre_IJMatrix *matrix, 
   					 int off_proc_i_indx,
   					 int max_off_proc_elmts,
   					 int current_num_elmts,
   					 HYPRE_BigInt *off_proc_i,
   					 HYPRE_BigInt *off_proc_j,
   					 double *off_proc_data)
{
  
   MPI_Comm comm = hypre_IJMatrixComm(matrix);

   int i, j, k, in_i;
   int myid;

   int proc_id, last_proc, prev_id, tmp_id;
   int max_response_size;
   HYPRE_BigInt global_num_cols;
   int ex_num_contacts = 0, num_rows = 0;
   HYPRE_BigInt range_start, range_end;
   int num_elements;
   int storage;
   int indx;
   HYPRE_BigInt row;
   int num_ranges;
   int num_recvs;
   int counter;
   HYPRE_BigInt upper_bound;
   int num_real_procs;
   

   HYPRE_BigInt *row_list=NULL;
   int *row_list_num_elements=NULL;
   int *a_proc_id=NULL, *orig_order=NULL;
   int *real_proc_id = NULL, *us_real_proc_id = NULL;
   int *ex_contact_procs = NULL, *ex_contact_vec_starts = NULL; 
   HYPRE_BigInt *ex_contact_buf = NULL;
   int  *recv_starts=NULL;
   HYPRE_BigInt *response_buf = NULL;
   int *response_buf_starts=NULL;
   int *num_rows_per_proc = NULL, *num_elements_total = NULL;

   int  obj_size_bytes, int_size, double_size, big_int_size;
   int  tmp_int;
   HYPRE_BigInt tmp_big_int;
   HYPRE_BigInt *col_ptr;
   HYPRE_BigInt *big_int_data = NULL;
   int big_int_data_size = 0, double_data_size = 0;

   void *void_contact_buf = NULL;
   void *index_ptr;
   void *recv_data_ptr;
   
   double tmp_double;
   double *col_data_ptr;
   double *double_data = NULL;
  
   hypre_DataExchangeResponse      response_obj1, response_obj2;
   hypre_ProcListElements          send_proc_obj; 

   hypre_IJAssumedPart   *apart;

   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixObject(matrix);

   MPI_Comm_rank(comm, &myid);
   global_num_cols = hypre_IJMatrixGlobalNumCols(matrix);
   
   /* verify that we have created the assumed partition */
   if  (hypre_ParCSRMatrixAssumedPartition(par_matrix) == NULL)
   {
      hypre_ParCSRMatrixCreateAssumedPartition(par_matrix);
   }

   apart = hypre_ParCSRMatrixAssumedPartition(par_matrix);

   num_rows = off_proc_i_indx/2;
   
   row_list = hypre_CTAlloc(HYPRE_BigInt, num_rows);
   row_list_num_elements = hypre_CTAlloc(int, num_rows);
   a_proc_id = hypre_CTAlloc(int, num_rows);
   orig_order =  hypre_CTAlloc(int, num_rows);
   real_proc_id = hypre_CTAlloc(int, num_rows);

   /* get the assumed processor id for each row */
   if (num_rows > 0 )
   {
      for (i=0; i < num_rows; i++)
      {
         row = off_proc_i[i*2];
         if (row < 0) row = -row-1; 
         row_list[i] = row;
         row_list_num_elements[i] = (int) off_proc_i[i*2+1];
         
         hypre_GetAssumedPartitionProcFromRow (comm, row, global_num_cols, &proc_id);
         a_proc_id[i] = proc_id;
         orig_order[i] = i;
      }
      
      /* now we need to find the actual order of each row  - sort on row -
         this will result in proc ids sorted also...*/
      
      hypre_BigQsortb2i(row_list, a_proc_id, orig_order, 0, num_rows -1);


      /* calculate the number of contacts */
      ex_num_contacts = 1;
      last_proc = a_proc_id[0];
      for (i=1; i < num_rows; i++)
      {
         if (a_proc_id[i] > last_proc)      
         {
            ex_num_contacts++;
            last_proc = a_proc_id[i];
         }
      }
      
   } 
   

   /* now we will go through a create a contact list - need to contact
      assumed processors and find out who the actual row owner is - we
      will contact with a range (2 numbers) */

   ex_contact_procs = hypre_CTAlloc(int, ex_num_contacts);
   ex_contact_vec_starts =  hypre_CTAlloc(int, ex_num_contacts+1);
   ex_contact_buf =  hypre_CTAlloc(HYPRE_BigInt, ex_num_contacts*2);

   counter = 0;
   range_end = -1;
   for (i=0; i< num_rows; i++) 
   {
      if (row_list[i] > range_end)
      {
         /* assumed proc */
         proc_id = a_proc_id[i];

         /* end of prev. range */
         if (counter > 0)  ex_contact_buf[counter*2 - 1] = row_list[i-1];
         
        /*start new range*/
    	 ex_contact_procs[counter] = proc_id;
         ex_contact_vec_starts[counter] = counter*2;
         ex_contact_buf[counter*2] =  row_list[i];
         counter++;
         
         hypre_GetAssumedPartitionRowRange(comm, proc_id, global_num_cols, 
                                           &range_start, &range_end); 


      }
   }
   /*finish the starts*/
   ex_contact_vec_starts[counter] =  counter*2;
   /*finish the last range*/
   if (counter > 0)  
      ex_contact_buf[counter*2 - 1] = row_list[num_rows - 1];


   /*don't allocate space for responses */
    

   /*create response object - can use same fill response as used in the commpkg
     routine */
   response_obj1.fill_response = hypre_RangeFillResponseIJDetermineRecvProcs;
   response_obj1.data1 =  apart; /* this is necessary so we can fill responses*/ 
   response_obj1.data2 = NULL;
   

   max_response_size = 6;  /* 6 means we can fit 3 ranges*/
   
   
   hypre_DataExchangeList(ex_num_contacts, ex_contact_procs, 
                    ex_contact_buf, ex_contact_vec_starts, sizeof(HYPRE_BigInt), 
                     sizeof(HYPRE_BigInt), &response_obj1, max_response_size, 1, 
                     comm, (void**) &response_buf, &response_buf_starts);


   /* now response_buf contains
     a proc_id followed by an upper bound for the range.  */

   hypre_TFree(ex_contact_procs);
   hypre_TFree(ex_contact_buf);
   hypre_TFree(ex_contact_vec_starts);

   hypre_TFree(a_proc_id);


   /*how many ranges were returned?*/
   num_ranges = response_buf_starts[ex_num_contacts];   
   num_ranges = num_ranges/2;


  prev_id = -1;
  j = 0;
  counter = 0;
  num_real_procs = 0;
  

 /* loop through ranges - create a list of actual processor ids*/
   for (i=0; i<num_ranges; i++)
   {
      upper_bound = response_buf[i*2+1];
      counter = 0;
      tmp_id = response_buf[i*2];

      /* loop through row_list entries - counting how many are in the range */
      while (row_list[j] <= upper_bound && j < num_rows)     
      {
         real_proc_id[j] = tmp_id;
         j++;
         counter++;       
      }
      if (counter > 0 && tmp_id != prev_id)        
      {
         num_real_procs++;
      }

      prev_id = tmp_id;
      
   }

 

   /* now we have the list of real procesors ids (real_proc_id) - and
      the number of distinct ones - so now we can set up data to be
      sent - we have int data and double data.  that we will need to
      pack together */
   

   /*first find out how many rows and 
     elements we need to send per proc - so we can do storage */
   
   ex_contact_procs = hypre_CTAlloc(int, num_real_procs);
   num_rows_per_proc = hypre_CTAlloc(int, num_real_procs);
   num_elements_total  =  hypre_CTAlloc(int, num_real_procs); 
   
   counter = 0;
   
   if (num_real_procs > 0 )
   {
      ex_contact_procs[0] = real_proc_id[0];
      num_rows_per_proc[0] = 1;
      num_elements_total[0] = row_list_num_elements[orig_order[0]];
      
      for (i=1; i < num_rows; i++) /* loop through real procs - these are sorted
                                      (row_list is sorted also)*/
      {
         if (real_proc_id[i] == ex_contact_procs[counter]) /* same processor */
         {
            num_rows_per_proc[counter] += 1; /*another row */
            num_elements_total[counter] += row_list_num_elements[orig_order[i]];
         }
         else /* new processor */
         {
            counter++;
            ex_contact_procs[counter] = real_proc_id[i];
            num_rows_per_proc[counter] = 1;
            num_elements_total[counter] = row_list_num_elements[orig_order[i]];
            
         }
      }
   }
   
   /* to pack together, we need to use the largest obj. size of
      (int) and (double) - if these are much different, then we are
      wasting some storage, but I do not think that it will be a
      large amount since this function should not be used on really
      large amounts of data anyway*/
   int_size = sizeof(int);
   double_size = sizeof(double);
   big_int_size = sizeof(HYPRE_BigInt);
   

   obj_size_bytes = hypre_max(big_int_size, double_size);

   /* set up data to be sent to send procs */
   /* for each proc, ex_contact_buf contains #rows, row #,
      no. elements, col indicies, col data, row #, no. elements, col
      indicies, col data, etc. */
      
   /* first calculate total storage and make vec_starts arrays */
   storage = 0;
   ex_contact_vec_starts = hypre_CTAlloc(int, num_real_procs + 1);
   ex_contact_vec_starts[0] = -1;
   
   for (i=0; i < num_real_procs; i++)
   {
      storage += 1 + 2 * num_rows_per_proc[i] + 2* num_elements_total[i];
      ex_contact_vec_starts[i+1] = -storage-1; /* need negative for next loop */
   }      

   hypre_TFree(num_elements_total);

   void_contact_buf = hypre_MAlloc(storage*obj_size_bytes);
   index_ptr = void_contact_buf; /* step through with this index */
   

   /* for each proc: #rows, row #, no. elements, 
      col indicies, col data, row #, no. elements, col indicies, col data, etc. */
      
   /* un-sort real_proc_id  - we want to access data arrays in order, so 
      cheaper to do this*/
   us_real_proc_id =  hypre_CTAlloc(int, num_rows);
   for (i=0; i < num_rows; i++)
   {
      us_real_proc_id[orig_order[i]] = real_proc_id[i];
   }
   hypre_TFree(real_proc_id);

   counter = 0; /* index into data arrays */
   prev_id = -1;
   for (i=0; i < num_rows; i++)
   {
      proc_id = us_real_proc_id[i];
      row = off_proc_i[i*2]; /*can't use row list[i] - you loose the negative signs that 
                               differentiate add/set values */
      num_elements = row_list_num_elements[i];
      /* find position of this processor */
      indx = hypre_BinarySearch(ex_contact_procs, proc_id, num_real_procs);
      in_i = ex_contact_vec_starts[indx];

      index_ptr = (void *) ((char *) void_contact_buf + in_i*obj_size_bytes);

      if (in_i < 0) /* first time for this processor - add the number of rows to the buffer */
      {
         in_i = -in_i - 1;
         /* re-calc. index_ptr since in_i was negative */
         index_ptr = (void *) ((char *) void_contact_buf + in_i*obj_size_bytes);

         tmp_int =  num_rows_per_proc[indx];
         memcpy( index_ptr, &tmp_int, int_size);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);

         in_i++;
      }
      /* add row # */   
      memcpy( index_ptr, &row, big_int_size);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in_i++;
            
      /* add number of elements */   
      memcpy( index_ptr, &num_elements, int_size);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in_i++;

      /* now add col indices  */
      for (j=0; j< num_elements; j++)
      {
         tmp_big_int = off_proc_j[counter+j]; /* col number */

         memcpy( index_ptr, &tmp_big_int, big_int_size);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
         in_i ++;
         
      }

      /* now add data */
      for (j=0; j< num_elements; j++)
      {
         tmp_double = off_proc_data[counter++]; /* value */

         memcpy( index_ptr, &tmp_double, double_size);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
         in_i++;
         
      }


      /* increment the indexes to keep track of where we are - we
       * adjust below to be actual starts*/
      ex_contact_vec_starts[indx] = in_i;
      
   }
   
   /* some clean up */
 
   hypre_TFree(response_buf);
   hypre_TFree(response_buf_starts);

   hypre_TFree(us_real_proc_id);
   hypre_TFree(orig_order);
   hypre_TFree(row_list);
   hypre_TFree(row_list_num_elements);
   hypre_TFree(num_rows_per_proc);
   

   for (i=num_real_procs; i > 0; i--)
   {
      ex_contact_vec_starts[i] =   ex_contact_vec_starts[i-1];
   }

   ex_contact_vec_starts[0] = 0;


   /* now send the data */

   /***********************************/

   /* first get the interger info in send_proc_obj */

   /* the response we expect is just a confirmation*/
   response_buf = NULL;
   response_buf_starts = NULL;


   /*build the response object*/

   /* use the send_proc_obj for the info kept from contacts */
   /*estimate inital storage allocation */
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = num_real_procs + 5;
   send_proc_obj.id = NULL; /* don't care who send it to us */
   send_proc_obj.vec_starts = hypre_CTAlloc(int, send_proc_obj.storage_length + 1); 
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = storage + 20;
   send_proc_obj.v_elements = hypre_MAlloc(obj_size_bytes*send_proc_obj.element_storage_length);

   response_obj2.fill_response = hypre_FillResponseIJOffProcVals;
   response_obj2.data1 = NULL;
   response_obj2.data2 = &send_proc_obj;

   max_response_size = 0;

   hypre_DataExchangeList(num_real_procs, ex_contact_procs, 
                          void_contact_buf, ex_contact_vec_starts, obj_size_bytes,
                          0, &response_obj2, max_response_size, 2, 
                          comm,  (void **) &response_buf, &response_buf_starts);





   hypre_TFree(response_buf);
   hypre_TFree(response_buf_starts);

   hypre_TFree(ex_contact_procs);
   hypre_TFree(void_contact_buf);
   hypre_TFree(ex_contact_vec_starts);


   /* Now we can unpack the send_proc_objects and call set 
      and add to values functions */

   num_recvs = send_proc_obj.length; 

   /* alias */
   recv_data_ptr = send_proc_obj.v_elements;
   recv_starts = send_proc_obj.vec_starts;
   
   for (i=0; i < num_recvs; i++)
   {

      indx = recv_starts[i];

      /* get the number of rows for this recv */
      memcpy( &num_rows, recv_data_ptr, int_size);
      recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
      indx++;
      
      
      for (j=0; j < num_rows; j++) /* for each row: unpack info */
      {
         /* row # */
         memcpy( &row, recv_data_ptr, big_int_size);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         /* num elements for this row */
         memcpy( &num_elements, recv_data_ptr, int_size);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         /* col indices */
         if (big_int_size == obj_size_bytes)
         {
            col_ptr = (HYPRE_BigInt *) recv_data_ptr;
            recv_data_ptr = (void *) ((char *)recv_data_ptr + num_elements*obj_size_bytes);
         }
         else /* copy data */
         {
            if (big_int_data_size < num_elements)
            {
               big_int_data = hypre_TReAlloc(big_int_data, HYPRE_BigInt, num_elements + 10);
            }
            for (k=0; k< num_elements; k++)
            { 
               memcpy( &big_int_data[k], recv_data_ptr, big_int_size);
               recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
            }
            col_ptr = big_int_data;
         }
         
         /* col data */
         if (double_size == obj_size_bytes)
         {
            col_data_ptr = (double *) recv_data_ptr;
            recv_data_ptr = (void *) ((char *)recv_data_ptr + num_elements*obj_size_bytes);
         }
         else /* copy data */
         {
            if (double_data_size < num_elements)
            {
               double_data = hypre_TReAlloc(double_data, double, num_elements + 10);
            }
            for (k=0; k< num_elements; k++)
            { 
               memcpy( &double_data[k], recv_data_ptr, double_size);
               recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
            }
            col_data_ptr = double_data;
            
         }

	 if (row < 0)  /* Add */
	 {
            row = -row-1;
 	    hypre_IJMatrixAddToValuesParCSR(matrix,1,&num_elements,&row,
		col_ptr,col_data_ptr);
	 }
	 else /* Set */
	 {
 	    hypre_IJMatrixSetValuesParCSR(matrix,1,&num_elements,&row,
		col_ptr,col_data_ptr);
	 }
         indx += (num_elements*2); 

      }
   }
   hypre_TFree(send_proc_obj.v_elements);
   hypre_TFree(send_proc_obj.vec_starts);
 
   if (big_int_data) hypre_TFree(big_int_data);
   if (double_data) hypre_TFree(double_data);
   


   return hypre_error_flag;
}

#endif


/*--------------------------------------------------------------------
 * hypre_FillResponseIJOffProcVals
 * Fill response function for the previous function (2nd data exchange)
 *--------------------------------------------------------------------*/

int
hypre_FillResponseIJOffProcVals(void *p_recv_contact_buf, 
                                 int contact_size, int contact_proc, void *ro, 
                                 MPI_Comm comm, void **p_send_response_buf, 
                                 int *response_message_size )


{
   int    myid;
   int    index, count, elength;

   int object_size;
   void *index_ptr;

   hypre_DataExchangeResponse  *response_obj = ro;  

   hypre_ProcListElements      *send_proc_obj = response_obj->data2;   

   object_size = hypre_max(sizeof(HYPRE_BigInt), sizeof(double));

   MPI_Comm_rank(comm, &myid );


   /*check to see if we need to allocate more space in send_proc_obj for vec starts*/
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length +=20; /*add space for 20 more contact*/
      send_proc_obj->vec_starts = hypre_TReAlloc(send_proc_obj->vec_starts,int, 
                                  send_proc_obj->storage_length + 1);
   }
  
   /*initialize*/ 
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /*this is the current number of elements*/

   /*do we need more storage for the elements?*/
   if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = hypre_max(contact_size, 100);   
      elength += index;
      send_proc_obj->v_elements = hypre_ReAlloc(send_proc_obj->v_elements, 
					       elength*object_size);
      send_proc_obj->element_storage_length = elength; 
   }
   /*populate send_proc_obj*/
   index_ptr = (void *) ((char *) send_proc_obj->v_elements + index*object_size);

   memcpy(index_ptr, p_recv_contact_buf , object_size*contact_size);


   send_proc_obj->vec_starts[count+1] = index + contact_size;
   send_proc_obj->length++;
   

  /*output - no message to return (confirmation) */
   *response_message_size = 0; 
  
   
   return hypre_error_flag;

}




/*--------------------------------------------------------------------*/

int hypre_FindProc(HYPRE_BigInt *list, HYPRE_BigInt value, int list_length)
{
   int low, high, m;

   low = 0;
   high = list_length;
   if (value >= list[high] || value < list[low])
      return -1;
   else
   {
      while (low+1 < high)
      {
         m = (low + high) / 2;
         if (value < list[m])
         {
            high = m;
         }
         else if (value >= list[m])
         {
            low = m;
         }
      }
      return low;
   }
}
 
