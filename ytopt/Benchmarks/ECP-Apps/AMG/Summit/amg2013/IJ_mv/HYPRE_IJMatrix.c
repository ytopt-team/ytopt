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
 * HYPRE_IJMatrix interface
 *
 *****************************************************************************/

#include "./IJ_mv.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixCreate
 *--------------------------------------------------------------------------*/

int HYPRE_IJMatrixCreate( MPI_Comm comm, HYPRE_BigInt ilower, HYPRE_BigInt iupper, 
			  HYPRE_BigInt jlower, HYPRE_BigInt jupper, HYPRE_IJMatrix *matrix) 

{

   HYPRE_BigInt *row_partitioning;
   HYPRE_BigInt *col_partitioning;
   HYPRE_BigInt *info;
   int num_procs;
   int myid;
   

   hypre_IJMatrix *ijmatrix;

#ifdef HYPRE_NO_GLOBAL_PARTITION
   HYPRE_BigInt  row0, col0, rowN, colN;
#else
   HYPRE_BigInt *recv_buf;
   int i, i4;
   int square;
#endif

   ijmatrix = hypre_CTAlloc(hypre_IJMatrix, 1);

   hypre_IJMatrixComm(ijmatrix)         = comm;
   hypre_IJMatrixObject(ijmatrix)       = NULL;
   hypre_IJMatrixTranslator(ijmatrix)   = NULL;
   hypre_IJMatrixObjectType(ijmatrix)   = HYPRE_UNITIALIZED;
   hypre_IJMatrixAssembleFlag(ijmatrix) = 0;

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm, &myid);
   

   if (ilower > iupper+1 || ilower < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (iupper < -1)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (jlower > jupper+1 || jlower < 0)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if (jupper < -1)
   {
      hypre_error_in_arg(5);
      return hypre_error_flag;
   }


#ifdef HYPRE_NO_GLOBAL_PARTITION

   info = hypre_CTAlloc(HYPRE_BigInt,2);

   row_partitioning = hypre_CTAlloc(HYPRE_BigInt, 2);
   col_partitioning = hypre_CTAlloc(HYPRE_BigInt, 2);

   row_partitioning[0] = ilower;
   row_partitioning[1] = iupper+1;
   col_partitioning[0] = jlower;
   col_partitioning[1] = jupper+1;

   /* now we need the global number of rows and columns as well
      as the global first row and column index */

   /* proc 0 has the first row and col */
   if (myid==0) 
   {
      info[0] = ilower;
      info[1] = jlower;
   }
   MPI_Bcast(info, 2, MPI_HYPRE_BIG_INT, 0, comm);
   row0 = info[0];
   col0 = info[1];
   
   /* proc (num_procs-1) has the last row and col */   
   if (myid == (num_procs-1))
   {
      info[0] = iupper;
      info[1] = jupper;
   }
   MPI_Bcast(info, 2, MPI_HYPRE_BIG_INT, num_procs-1, comm);

   rowN = info[0];
   colN = info[1];

   hypre_IJMatrixGlobalFirstRow(ijmatrix) = row0;
   hypre_IJMatrixGlobalFirstCol(ijmatrix) = col0;
   hypre_IJMatrixGlobalNumRows(ijmatrix) = rowN - row0 + 1;
   hypre_IJMatrixGlobalNumCols(ijmatrix) = colN - col0 + 1;
   
   hypre_TFree(info);
   

#else

   info = hypre_CTAlloc(HYPRE_BigInt,4);
   recv_buf = hypre_CTAlloc(HYPRE_BigInt,4*num_procs);
   row_partitioning = hypre_CTAlloc(HYPRE_BigInt, num_procs+1);

   info[0] = ilower;
   info[1] = iupper;
   info[2] = jlower;
   info[3] = jupper;

  
   /* Generate row- and column-partitioning through information exchange
      across all processors, check whether the matrix is square, and
      if the partitionings match. i.e. no overlaps or gaps,
      if there are overlaps or gaps in the row partitioning or column
      partitioning , ierr will be set to -9 or -10, respectively */

   MPI_Allgather(info,4,MPI_HYPRE_BIG_INT,recv_buf,4,MPI_HYPRE_BIG_INT,comm);

   row_partitioning[0] = recv_buf[0];
   square = 1;
   for (i=0; i < num_procs-1; i++)
   {
      i4 = 4*i;
      if ( recv_buf[i4+1] != (recv_buf[i4+4]-1) )
      {
         printf("Warning -- row partitioning does not line up! Partitioning incomplete!\n");
         hypre_error(HYPRE_ERROR_GENERIC);
   	 return hypre_error_flag;
      }
      else
	 row_partitioning[i+1] = recv_buf[i4+4];
	 
      if ((square && (recv_buf[i4]   != recv_buf[i4+2])) ||
                    (recv_buf[i4+1] != recv_buf[i4+3])  )
      {
         square = 0;
      }
   }	
   i4 = (num_procs-1)*4;
   row_partitioning[num_procs] = recv_buf[i4+1]+1;

   if ((recv_buf[i4] != recv_buf[i4+2]) || (recv_buf[i4+1] != recv_buf[i4+3])) 
      square = 0;

   if (square)
      col_partitioning = row_partitioning;
   else
   {   
      col_partitioning = hypre_CTAlloc(HYPRE_BigInt,num_procs+1);
      col_partitioning[0] = recv_buf[2];
      for (i=0; i < num_procs-1; i++)
      {
         i4 = 4*i;
         if (recv_buf[i4+3] != recv_buf[i4+6]-1)
         {
           printf("Warning -- col partitioning does not line up! Partitioning incomplete!\n");
           hypre_error(HYPRE_ERROR_GENERIC);
   	   return hypre_error_flag;
         }
         else
   	   col_partitioning[i+1] = recv_buf[i4+6];
      }
      col_partitioning[num_procs] = recv_buf[num_procs*4-1]+1;
   }



   hypre_IJMatrixGlobalFirstRow(ijmatrix) = row_partitioning[0];
   hypre_IJMatrixGlobalFirstCol(ijmatrix) = col_partitioning[0];
   hypre_IJMatrixGlobalNumRows(ijmatrix) = row_partitioning[num_procs] - 
      row_partitioning[0];
   hypre_IJMatrixGlobalNumCols(ijmatrix) = col_partitioning[num_procs] - 
      col_partitioning[0];
   


   hypre_TFree(info);
   hypre_TFree(recv_buf);
   
#endif


   hypre_IJMatrixRowPartitioning(ijmatrix) = row_partitioning;
   hypre_IJMatrixColPartitioning(ijmatrix) = col_partitioning;

   *matrix = (HYPRE_IJMatrix) ijmatrix;
  
   return hypre_error_flag;
   /* return ierr; */
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixDestroy( HYPRE_IJMatrix matrix )
{
   int ierr = 0;
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixDestroy\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (ijmatrix)
   {
      if (hypre_IJMatrixRowPartitioning(ijmatrix) ==
                      hypre_IJMatrixColPartitioning(ijmatrix))
         hypre_TFree(hypre_IJMatrixRowPartitioning(ijmatrix));
      else
      {
         hypre_TFree(hypre_IJMatrixRowPartitioning(ijmatrix));
         hypre_TFree(hypre_IJMatrixColPartitioning(ijmatrix));
      }

      /*
      if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
         ierr = hypre_IJMatrixDestroyPETSc( ijmatrix );
      else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
         ierr = hypre_IJMatrixDestroyISIS( ijmatrix );
      else */ 

      if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
         ierr = hypre_IJMatrixDestroyParCSR( ijmatrix );
      else if ( hypre_IJMatrixObjectType(ijmatrix) != -1 )
      {
         printf("Unrecognized object type -- HYPRE_IJMatrixDestroy\n");
         hypre_error_in_arg(1);
         return hypre_error_flag;
      }
   }

   hypre_TFree(ijmatrix); 

   return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixInitialize( HYPRE_IJMatrix matrix )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;
   int ierr = 0;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixInitialize\n");

      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixInitializePETSc( ijmatrix ) ;
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixInitializeISIS( ijmatrix ) ;
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixInitializeParCSR( ijmatrix ) ;
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixInitialize\n");
      hypre_error_in_arg(1);
   }
  
   return hypre_error_flag;

}
  


/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetValues( HYPRE_IJMatrix matrix, int nrows,
                         int *ncols, const HYPRE_BigInt *rows,
                         const HYPRE_BigInt *cols, const double *values)
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (nrows == 0)
      return hypre_error_flag;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixSetValues\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nrows < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!ncols)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (!rows)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if (!cols)
   {
      hypre_error_in_arg(5);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(6);
      return hypre_error_flag;
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      return( hypre_IJMatrixSetValuesPETSc( ijmatrix, nrows, ncols, 
                                            rows, cols, values ) );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      return( hypre_IJMatrixSetValuesISIS( ijmatrix, nrows, ncols, 
                                           rows, cols, values ) );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      return( hypre_IJMatrixSetValuesParCSR( ijmatrix, nrows, ncols,
                                             rows, cols, values ) );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixSetValues\n");
      hypre_error_in_arg(1);
   }
    
   return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAddToValues
 *--------------------------------------------------------------------------*/


int 
HYPRE_IJMatrixAddToValues( HYPRE_IJMatrix matrix, int nrows,
                           int *ncols, const HYPRE_BigInt *rows,
                           const HYPRE_BigInt *cols, const double *values)
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (nrows == 0)
      return hypre_error_flag;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixAddToValues\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nrows < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!ncols)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (!rows)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if (!cols)
   {
      hypre_error_in_arg(5);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(6);
      return hypre_error_flag;
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      return( hypre_IJMatrixAddToValuesPETSc( ijmatrix, nrows, ncols, 
                                            rows, cols, values ) );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      return( hypre_IJMatrixAddToValuesISIS( ijmatrix, nrows, ncols, 
                                           rows, cols, values ) );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      return( hypre_IJMatrixAddToValuesParCSR( ijmatrix, nrows, ncols,
                                             rows, cols, values ) );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixAddToValues\n");
      hypre_error_in_arg(1);
   }
    
   return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixAssemble( HYPRE_IJMatrix matrix )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixAssemble\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      return( hypre_IJMatrixAssemblePETSc( ijmatrix ) );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      return( hypre_IJMatrixAssembleISIS( ijmatrix ) );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      return( hypre_IJMatrixAssembleParCSR( ijmatrix ) );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixAssemble\n");
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetRowCounts
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixGetRowCounts( HYPRE_IJMatrix matrix, int nrows, 
                         HYPRE_BigInt *rows, int *ncols )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (nrows == 0) return hypre_error_flag;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixGetRowCounts\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nrows < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!rows)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (!ncols)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixGetRowCountsPETSc( ijmatrix, nrows, rows, ncols );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixGetRowCountsISIS( ijmatrix, nrows, rows, ncols );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      hypre_IJMatrixGetRowCountsParCSR( ijmatrix, nrows, rows, ncols );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixGetRowCounts\n");
      hypre_error_in_arg(1);
   }

    return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixGetValues( HYPRE_IJMatrix matrix, int nrows, int *ncols,
                         HYPRE_BigInt *rows, HYPRE_BigInt *cols, double *values)
{
   int ierr = 0;
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (nrows == 0) return hypre_error_flag;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixGetValues\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nrows < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!ncols)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (!rows)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   if (!cols)
   {
      hypre_error_in_arg(5);
      return hypre_error_flag;
   }

   if (!values)
   {
      hypre_error_in_arg(6);
      return hypre_error_flag;
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixGetValuesPETSc( ijmatrix, nrows, ncols, 
					   rows, cols, values );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixGetValuesISIS( ijmatrix, nrows, ncols, 
					  rows, cols, values );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      ierr = hypre_IJMatrixGetValuesParCSR( ijmatrix, nrows, ncols,
					    rows, cols, values );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixGetValues\n");
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetObjectType
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetObjectType( HYPRE_IJMatrix matrix, int type )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixSetObjectType\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_IJMatrixObjectType(ijmatrix) = type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetObjectType
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixGetObjectType( HYPRE_IJMatrix matrix, int *type )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixGetObjectType\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *type = hypre_IJMatrixObjectType(ijmatrix);
   return hypre_error_flag;
   /* return 0; */
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixGetLocalRange( HYPRE_IJMatrix matrix, HYPRE_BigInt *ilower, 
			HYPRE_BigInt *iupper,
			HYPRE_BigInt *jlower, HYPRE_BigInt *jupper )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;
   MPI_Comm comm;
   HYPRE_BigInt *row_partitioning;
   HYPRE_BigInt *col_partitioning;
   int my_id;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixGetObjectType\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   comm = hypre_IJMatrixComm(ijmatrix);
   row_partitioning = hypre_IJMatrixRowPartitioning(ijmatrix);
   col_partitioning = hypre_IJMatrixColPartitioning(ijmatrix);

   MPI_Comm_rank(comm, &my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   *ilower = row_partitioning[0];
   *iupper = row_partitioning[1]-1;
   *jlower = col_partitioning[0];
   *jupper = col_partitioning[1]-1;
#else
   *ilower = row_partitioning[my_id];
   *iupper = row_partitioning[my_id+1]-1;
   *jlower = col_partitioning[my_id];
   *jupper = col_partitioning[my_id+1]-1;
#endif

   return hypre_error_flag;
   /* return 0; */
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixGetObject
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to an underlying ijmatrix type used to implement IJMatrix.
Assumes that the implementation has an underlying matrix, so it would not
work with a direct implementation of IJMatrix. 

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

int
HYPRE_IJMatrixGetObject( HYPRE_IJMatrix matrix, void **object )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixGetObject\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *object = hypre_IJMatrixObject( ijmatrix );

   return hypre_error_flag;

   /* return 0; */
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetRowSizes
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetRowSizes( HYPRE_IJMatrix matrix, const int *sizes )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixSetRowSizes\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      return( hypre_IJMatrixSetRowSizesPETSc( ijmatrix , sizes ) );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      return( hypre_IJMatrixSetRowSizesISIS( ijmatrix , sizes ) );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      return( hypre_IJMatrixSetRowSizesParCSR( ijmatrix , sizes ) );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixSetRowSizes\n");
      hypre_error_in_arg(1);
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetDiagOffdSizes
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetDiagOffdSizes( HYPRE_IJMatrix matrix, 
				const int *diag_sizes,
				const int *offdiag_sizes )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixSetDiagOffdSizes\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      ierr = hypre_IJMatrixSetDiagOffdSizesPETSc( ijmatrix , diag_sizes ,
						offdiag_sizes );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      ierr = hypre_IJMatrixSetDiagOffdSizesISIS( ijmatrix , diag_sizes ,
						offdiag_sizes );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      hypre_IJMatrixSetDiagOffdSizesParCSR( ijmatrix , diag_sizes ,
							offdiag_sizes );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixSetDiagOffdSizes\n");
      hypre_error_in_arg(1);
   }
   return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixSetMaxOffProcElmts( HYPRE_IJMatrix matrix, 
				  int max_off_proc_elmts)
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- HYPRE_IJMatrixSetMaxOffProcElmts\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PETSC )
      return( hypre_IJMatrixSetMaxOffProcElmtsPETSc( ijmatrix , 
			max_off_proc_elmts_set, max_off_proc_elmts_add ) );
   else if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_ISIS )
      return( hypre_IJMatrixSetMaxOffProcElmtsISIS( ijmatrix , 
			max_off_proc_elmts_set, max_off_proc_elmts_add ) );
   else */

   if ( hypre_IJMatrixObjectType(ijmatrix) == HYPRE_PARCSR )
      return( hypre_IJMatrixSetMaxOffProcElmtsParCSR( ijmatrix , 
			max_off_proc_elmts) );
   else
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixSetMaxOffProcElmts\n");
      hypre_error_in_arg(1);
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixRead
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixRead( const char     *filename,
                    MPI_Comm        comm,
                    int             type,
		    HYPRE_IJMatrix *matrix_ptr )
{
   HYPRE_IJMatrix  matrix;
   HYPRE_BigInt    ilower, iupper, jlower, jupper;
   HYPRE_BigInt    I, J;
   int             ncols;
   double          value;
   int             myid;
   char            new_filename[255];
   FILE           *file;

   MPI_Comm_rank(comm, &myid);
   
   sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      printf("Error: can't open input file %s\n", new_filename);

      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

#ifdef HYPRE_LONG_LONG
   fscanf(file, "%lld %lld %lld %lld", &ilower, &iupper, &jlower, &jupper);
#else
   fscanf(file, "%d %d %d %d", &ilower, &iupper, &jlower, &jupper);
#endif
   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &matrix);

   HYPRE_IJMatrixSetObjectType(matrix, type);
   HYPRE_IJMatrixInitialize(matrix);

   ncols = 1;
#ifdef HYPRE_LONG_LONG
   while ( fscanf(file, "%lld %lld %le", &I, &J, &value) != EOF )
#else
   while ( fscanf(file, "%d %d %le", &I, &J, &value) != EOF )
#endif
   {
      HYPRE_IJMatrixSetValues(matrix, 1, &ncols, &I, &J, &value);
   }

   HYPRE_IJMatrixAssemble(matrix);

   fclose(file);

   *matrix_ptr = matrix;

   return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 * HYPRE_IJMatrixPrint
 *--------------------------------------------------------------------------*/

int 
HYPRE_IJMatrixPrint( HYPRE_IJMatrix  matrix,
                     const char     *filename )
{
   MPI_Comm  comm = hypre_IJMatrixComm(matrix);
   HYPRE_BigInt *row_partitioning;
   HYPRE_BigInt *col_partitioning;
   HYPRE_BigInt  ilower, iupper, jlower, jupper;
   HYPRE_BigInt  i, ii;
   int       j;
   int       ncols;
   HYPRE_BigInt *cols;
   double   *values;
   int       myid;
   char      new_filename[255];
   FILE     *file;
   void     *object;

   if (!matrix)
   {
      printf("Variable matrix is NULL -- HYPRE_IJMatrixPrint\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( (hypre_IJMatrixObjectType(matrix) != HYPRE_PARCSR) )
   {
      printf("Unrecognized object type -- HYPRE_IJMatrixPrint\n");
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   MPI_Comm_rank(comm, &myid);
   
   sprintf(new_filename,"%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      printf("Error: can't open output file %s\n", new_filename);
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   col_partitioning = hypre_IJMatrixColPartitioning(matrix);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   ilower = row_partitioning[0];
   iupper = row_partitioning[1] - 1;
   jlower = col_partitioning[0];
   jupper = col_partitioning[1] - 1;
#else
   ilower = row_partitioning[myid];
   iupper = row_partitioning[myid+1] - 1;
   jlower = col_partitioning[myid];
   jupper = col_partitioning[myid+1] - 1;
#endif
#ifdef HYPRE_LONG_LONG
   fprintf(file, "%lld %lld %lld %lld\n", ilower, iupper, jlower, jupper);
#else
   fprintf(file, "%d %d %d %d\n", ilower, iupper, jlower, jupper);
#endif

   HYPRE_IJMatrixGetObject(matrix, &object);

   for (i = ilower; i <= iupper; i++)
   {
      if ( hypre_IJMatrixObjectType(matrix) == HYPRE_PARCSR )
      {
#ifdef HYPRE_NO_GLOBAL_PARTITION
         ii = i -  hypre_IJMatrixGlobalFirstRow(matrix);
#else
         ii = i - row_partitioning[0];
#endif
         HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) object,
                                          ii, &ncols, &cols, &values);
         for (j = 0; j < ncols; j++)
         {
#ifdef HYPRE_NO_GLOBAL_PARTITION
            cols[j] +=  hypre_IJMatrixGlobalFirstCol(matrix);
#else
            cols[j] += col_partitioning[0];
#endif
         }
      }

      for (j = 0; j < ncols; j++)
      {
#ifdef HYPRE_LONG_LONG
         fprintf(file, "%lld %lld %.14e\n", i, cols[j], values[j]);
#else
         fprintf(file, "%d %d %.14e\n", i, cols[j], values[j]);
#endif
      }

      if ( hypre_IJMatrixObjectType(matrix) == HYPRE_PARCSR )
      {
         for (j = 0; j < ncols; j++)
         {
#ifdef HYPRE_NO_GLOBAL_PARTITION
            cols[j] -=  hypre_IJMatrixGlobalFirstCol(matrix);
#else
            cols[j] -= col_partitioning[0];
#endif
         }
         HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) object,
                                              ii, &ncols, &cols, &values);
      }
   }

   fclose(file);

   return hypre_error_flag;
}
