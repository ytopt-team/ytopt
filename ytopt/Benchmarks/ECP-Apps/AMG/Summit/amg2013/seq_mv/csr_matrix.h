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
 * Header info for CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_CSR_MATRIX_HEADER
#define hypre_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   double  *data;
   int     *i;
   int     *j;
   int      num_rows;
   int      num_cols;
   int      num_nonzeros;

  /* for compressing rows in matrix multiplication  */
   int     *rownnz;
   int      num_rownnz;

   /* Does the CSRMatrix create/destroy `data', `i', `j'? */
   int      owns_data;

} hypre_CSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRMatrixData(matrix)         ((matrix) -> data)
#define hypre_CSRMatrixI(matrix)            ((matrix) -> i)
#define hypre_CSRMatrixJ(matrix)            ((matrix) -> j)
#define hypre_CSRMatrixNumRows(matrix)      ((matrix) -> num_rows)
#define hypre_CSRMatrixNumCols(matrix)      ((matrix) -> num_cols)
#define hypre_CSRMatrixNumNonzeros(matrix)  ((matrix) -> num_nonzeros)
#define hypre_CSRMatrixRownnz(matrix)       ((matrix) -> rownnz)
#define hypre_CSRMatrixNumRownnz(matrix)    ((matrix) -> num_rownnz)
#define hypre_CSRMatrixOwnsData(matrix)     ((matrix) -> owns_data)


/*--------------------------------------------------------------------------
 * BigCSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   int    *i;
   HYPRE_BigInt *j;
   double *data;
   int     num_rows;
   int     num_cols;
   int     num_nonzeros;
   int     owns_data;

} hypre_BigCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the BigCSRMatrix structure
 *--------------------------------------------------------------------------*/

#define hypre_BigCSRMatrixI(matrix)          ((matrix)->i)
#define hypre_BigCSRMatrixJ(matrix)          ((matrix)->j)
#define hypre_BigCSRMatrixData(matrix)       ((matrix)->data)
#define hypre_BigCSRMatrixNumRows(matrix)    ((matrix)->num_rows)
#define hypre_BigCSRMatrixNumCols(matrix)    ((matrix)->num_cols)
#define hypre_BigCSRMatrixNumNonzeros(matrix)((matrix)->num_nonzeros)
#define hypre_BigCSRMatrixOwnsData(matrix)   ((matrix)->owns_data)

#endif


