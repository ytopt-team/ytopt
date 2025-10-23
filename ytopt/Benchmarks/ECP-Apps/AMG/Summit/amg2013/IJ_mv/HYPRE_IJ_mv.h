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




#ifndef HYPRE_IJ_MV_HEADER
#define HYPRE_IJ_MV_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ System Interface
 *
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 *
 * @memo A linear-algebraic conceptual interface
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ Matrices
 **/
/*@{*/

struct hypre_IJMatrix_struct;
/**
 * The matrix object.
 **/
typedef struct hypre_IJMatrix_struct *HYPRE_IJMatrix;

/**
 * Create a matrix object.  Each process owns some unique consecutive
 * range of rows, indicated by the global row indices {\tt ilower} and
 * {\tt iupper}.  The row data is required to be such that the value
 * of {\tt ilower} on any process $p$ be exactly one more than the
 * value of {\tt iupper} on process $p-1$.  Note that the first row of
 * the global matrix may start with any integer value.  In particular,
 * one may use zero- or one-based indexing.
 *
 * For square matrices, {\tt jlower} and {\tt jupper} typically should
 * match {\tt ilower} and {\tt iupper}, respectively.  For rectangular
 * matrices, {\tt jlower} and {\tt jupper} should define a
 * partitioning of the columns.  This partitioning must be used for
 * any vector $v$ that will be used in matrix-vector products with the
 * rectangular matrix.  The matrix data structure may use {\tt jlower}
 * and {\tt jupper} to store the diagonal blocks (rectangular in
 * general) of the matrix separately from the rest of the matrix.
 *
 * Collective.
 **/
int HYPRE_IJMatrixCreate(MPI_Comm        comm,
                         HYPRE_BigInt    ilower,
                         HYPRE_BigInt    iupper,
                         HYPRE_BigInt    jlower,
                         HYPRE_BigInt    jupper,
                         HYPRE_IJMatrix *matrix);

/**
 * Destroy a matrix object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_IJMatrixDestroy(HYPRE_IJMatrix matrix);

/**
 * Prepare a matrix object for setting coefficient values.  This
 * routine will also re-initialize an already assembled matrix,
 * allowing users to modify coefficient values.
 **/
int HYPRE_IJMatrixInitialize(HYPRE_IJMatrix matrix);

/**
 * Sets values for {\tt nrows} rows or partial rows of the matrix.  
 * The arrays {\tt ncols}
 * and {\tt rows} are of dimension {\tt nrows} and contain the number
 * of columns in each row and the row indices, respectively.  The
 * array {\tt cols} contains the column indices for each of the {\tt
 * rows}, and is ordered by rows.  The data in the {\tt values} array
 * corresponds directly to the column entries in {\tt cols}.  Erases
 * any previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before, inserts a
 * new one.
 *
 * Not collective.
 **/
int HYPRE_IJMatrixSetValues(HYPRE_IJMatrix  matrix,
                            int             nrows,
                            int            *ncols,
                            const HYPRE_BigInt *rows,
                            const HYPRE_BigInt *cols,
                            const double   *values);

/**
 * Adds to values for {\tt nrows} rows or partial rows of the matrix.  
 * Usage details are
 * analogous to \Ref{HYPRE_IJMatrixSetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value there
 * before, inserts a new one.
 *
 * Not collective.
 **/
int HYPRE_IJMatrixAddToValues(HYPRE_IJMatrix  matrix,
                              int             nrows,
                              int            *ncols,
                              const HYPRE_BigInt *rows,
                              const HYPRE_BigInt *cols,
                              const double   *values);

/**
 * Finalize the construction of the matrix before using.
 **/
int HYPRE_IJMatrixAssemble(HYPRE_IJMatrix matrix);

/**
 * Gets number of nonzeros elements for {\tt nrows} rows specified in {\tt rows}
 * and returns them in {\tt ncols}, which needs to be allocated by the
 * user.
 **/
int HYPRE_IJMatrixGetRowCounts(HYPRE_IJMatrix  matrix,
                               int             nrows,
                               HYPRE_BigInt   *rows,
                               int            *ncols);

/**
 * Gets values for {\tt nrows} rows or partial rows of the matrix.  
 * Usage details are
 * analogous to \Ref{HYPRE_IJMatrixSetValues}.
 **/
int HYPRE_IJMatrixGetValues(HYPRE_IJMatrix  matrix,
                            int             nrows,
                            int            *ncols,
                            HYPRE_BigInt   *rows,
                            HYPRE_BigInt   *cols,
                            double         *values);

/**
 * Set the storage type of the matrix object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR}.
 *
 * Not collective, but must be the same on all processes.
 *
 * @see HYPRE_IJMatrixGetObject
 **/
int HYPRE_IJMatrixSetObjectType(HYPRE_IJMatrix matrix,
                                int            type);

/**
 * Get the storage type of the constructed matrix object.
 **/
int HYPRE_IJMatrixGetObjectType(HYPRE_IJMatrix  matrix,
                                int            *type);

/**
 * Gets range of rows owned by this processor and range
 * of column partitioning for this processor.
 **/
int HYPRE_IJMatrixGetLocalRange(HYPRE_IJMatrix  matrix,
                                HYPRE_BigInt   *ilower,
                                HYPRE_BigInt   *iupper,
                                HYPRE_BigInt   *jlower,
                                HYPRE_BigInt   *jupper);

/**
 * Get a reference to the constructed matrix object.
 *
 * @see HYPRE_IJMatrixSetObjectType
 **/
int HYPRE_IJMatrixGetObject(HYPRE_IJMatrix   matrix,
                            void           **object);

/**
 * (Optional) Set the max number of nonzeros to expect in each row.
 * The array {\tt sizes} contains estimated sizes for each row on this
 * process.  This call can significantly improve the efficiency of
 * matrix construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
int HYPRE_IJMatrixSetRowSizes(HYPRE_IJMatrix  matrix,
                              const int      *sizes);

/**
 * (Optional) Set the max number of nonzeros to expect in each row of
 * the diagonal and off-diagonal blocks.  The diagonal block is the
 * submatrix whose column numbers correspond to rows owned by this
 * process, and the off-diagonal block is everything else.  The arrays
 * {\tt diag\_sizes} and {\tt offdiag\_sizes} contain estimated sizes
 * for each row of the diagonal and off-diagonal blocks, respectively.
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
int HYPRE_IJMatrixSetDiagOffdSizes(HYPRE_IJMatrix  matrix,
                                   const int      *diag_sizes,
                                   const int      *offdiag_sizes);

/**
 * (Optional) Sets the maximum number of elements that are expected to be set
 * (or added) on other processors from this processor
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
int HYPRE_IJMatrixSetMaxOffProcElmts(HYPRE_IJMatrix  matrix,
                                     int max_off_proc_elmts);

/**
 * Read the matrix from file.  This is mainly for debugging purposes.
 **/
int HYPRE_IJMatrixRead(const char     *filename,
		       MPI_Comm        comm,
		       int             type,
		       HYPRE_IJMatrix *matrix);

/**
 * Print the matrix to file.  This is mainly for debugging purposes.
 **/
int HYPRE_IJMatrixPrint(HYPRE_IJMatrix  matrix,
                        const char     *filename);

/*@}*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name IJ Vectors
 **/
/*@{*/

struct hypre_IJVector_struct;
/**
 * The vector object.
 **/
typedef struct hypre_IJVector_struct *HYPRE_IJVector;

/**
 * Create a vector object.  Each process owns some unique consecutive
 * range of vector unknowns, indicated by the global indices {\tt
 * jlower} and {\tt jupper}.  The data is required to be such that the
 * value of {\tt jlower} on any process $p$ be exactly one more than
 * the value of {\tt jupper} on process $p-1$.  Note that the first
 * index of the global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 *
 * Collective.
 **/
int HYPRE_IJVectorCreate(MPI_Comm        comm,
                         HYPRE_BigInt    jlower,
                         HYPRE_BigInt    jupper,
                         HYPRE_IJVector *vector);

/**
 * Destroy a vector object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  Once destroyed, the object must not be referenced
 * again.  Note that the object may not be deallocated at the
 * completion of this call, since there may be internal package
 * references to the object.  The object will then be destroyed when
 * all internal reference counts go to zero.
 **/
int HYPRE_IJVectorDestroy(HYPRE_IJVector vector);

/**
 * Prepare a vector object for setting coefficient values.  This
 * routine will also re-initialize an already assembled vector,
 * allowing users to modify coefficient values.
 **/
int HYPRE_IJVectorInitialize(HYPRE_IJVector vector);

/**
 * (Optional) Sets the maximum number of elements that are expected to be set
 * (or added) on other processors from this processor
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 *
 * Not collective.
 **/
int HYPRE_IJVectorSetMaxOffProcElmts(HYPRE_IJVector  vector,
                                     int max_off_proc_elmts);

/**
 * Sets values in vector.  The arrays {\tt values} and {\tt indices}
 * are of dimension {\tt nvalues} and contain the vector values to be
 * set and the corresponding global vector indices, respectively.
 * Erases any previous values at the specified locations and replaces
 * them with new ones.
 *
 * Not collective.
 **/
int HYPRE_IJVectorSetValues(HYPRE_IJVector  vector,
                            int             nvalues,
                            const HYPRE_BigInt *indices,
                            const double   *values);

/**
 * Adds to values in vector.  Usage details are analogous to
 * \Ref{HYPRE_IJVectorSetValues}.
 *
 * Not collective.
 **/
int HYPRE_IJVectorAddToValues(HYPRE_IJVector  vector,
                              int             nvalues,
                              const HYPRE_BigInt *indices,
                              const double   *values);

/**
 * Finalize the construction of the vector before using.
 **/
int HYPRE_IJVectorAssemble(HYPRE_IJVector vector);

/**
 * Gets values in vector.  Usage details are analogous to
 * \Ref{HYPRE_IJVectorSetValues}.
 *
 * Not collective.
 **/
int HYPRE_IJVectorGetValues(HYPRE_IJVector  vector,
                            int             nvalues,
                            const HYPRE_BigInt *indices,
                            double         *values);

/**
 * Set the storage type of the vector object to be constructed.
 * Currently, {\tt type} can only be {\tt HYPRE\_PARCSR}.
 *
 * Not collective, but must be the same on all processes.
 *
 * @see HYPRE_IJVectorGetObject
 **/
int HYPRE_IJVectorSetObjectType(HYPRE_IJVector vector,
                                int            type);

/**
 * Get the storage type of the constructed vector object.
 **/
int HYPRE_IJVectorGetObjectType(HYPRE_IJVector  vector,
                                int            *type);

/**
 * Returns range of the part of the vector owned by this processor.
 **/
int HYPRE_IJVectorGetLocalRange(HYPRE_IJVector  vector,
                                HYPRE_BigInt   *jlower,
                                HYPRE_BigInt   *jupper);

/**
 * Get a reference to the constructed vector object.
 *
 * @see HYPRE_IJVectorSetObjectType
 **/
int HYPRE_IJVectorGetObject(HYPRE_IJVector   vector,
                            void           **object);

/**
 * Read the vector from file.  This is mainly for debugging purposes.
 **/
int HYPRE_IJVectorRead(const char     *filename,
		       MPI_Comm        comm,
		       int             type,
                       HYPRE_IJVector *vector);

/**
 * Print the vector to file.  This is mainly for debugging purposes.
 **/
int HYPRE_IJVectorPrint(HYPRE_IJVector  vector,
                        const char     *filename);

/*@}*/
/*@}*/

#ifdef __cplusplus
}
#endif

#endif
