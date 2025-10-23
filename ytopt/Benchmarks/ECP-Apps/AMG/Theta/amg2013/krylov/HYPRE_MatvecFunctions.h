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



#ifndef HYPRE_MATVEC_FUNCTIONS
#define HYPRE_MATVEC_FUNCTIONS

typedef struct
{
  void*  (*MatvecCreate)  ( void *A, void *x );
  int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
  int    (*MatvecDestroy) ( void *matvec_data );

  void*  (*MatMultiVecCreate)  ( void *A, void *x );
  int    (*MatMultiVec)        ( void *data, double alpha, void *A,
				 void *x, double beta, void *y );
  int    (*MatMultiVecDestroy)  ( void *data );

} HYPRE_MatvecFunctions;

#endif
