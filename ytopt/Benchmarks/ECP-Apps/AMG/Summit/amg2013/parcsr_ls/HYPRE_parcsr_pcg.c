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
 * HYPRE_ParCSRPCG interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   /* The function names with a PCG in them are in
      parcsr_ls/pcg_par.c .  These functions do rather little -
      e.g., cast to the correct type - before calling something else.
      These names should be called, e.g., hypre_struct_Free, to reduce the
      chance of name conflicts. */
   hypre_PCGFunctions * pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );

   *solver = ( (HYPRE_Solver) hypre_PCGCreate( pcg_functions ) );
   if (!solver) hypre_error_in_arg(2);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPCGDestroy( HYPRE_Solver solver )
{
   return( hypre_PCGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPCGSetup( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   return( HYPRE_PCGSetup( solver,
                           (HYPRE_Matrix) A,
                           (HYPRE_Vector) b,
                           (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPCGSolve( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   return( HYPRE_PCGSolve( solver,
                           (HYPRE_Matrix) A,
                           (HYPRE_Vector) b,
                           (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetTol( HYPRE_Solver solver,
                       double             tol    )
{
   return( HYPRE_PCGSetTol( solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetMaxIter( HYPRE_Solver solver,
                           int                max_iter )
{
   return( HYPRE_PCGSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetStopCrit( HYPRE_Solver solver,
                            int          stop_crit )
{
   return( HYPRE_PCGSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetTwoNorm( HYPRE_Solver solver,
                           int                two_norm )
{
   return( HYPRE_PCGSetTwoNorm( solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetRelChange( HYPRE_Solver solver,
                             int                rel_change )
{
   return( HYPRE_PCGSetRelChange( solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetPrecond( HYPRE_Solver         solver,
                           HYPRE_PtrToParSolverFcn precond,
                           HYPRE_PtrToParSolverFcn precond_setup,
                           HYPRE_Solver         precond_solver )
{
   return( HYPRE_PCGSetPrecond( solver,
                                (HYPRE_PtrToSolverFcn) precond,
                                (HYPRE_PtrToSolverFcn) precond_setup,
                                precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGGetPrecond( HYPRE_Solver  solver,
                           HYPRE_Solver *precond_data_ptr )
{
   return( HYPRE_PCGGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrintLevel
 * an obsolete function; use HYPRE_PCG* functions instead
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetPrintLevel( HYPRE_Solver solver,
                              int level )
{
   return( HYPRE_PCGSetPrintLevel( solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetLogging
 * an obsolete function; use HYPRE_PCG* functions instead
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetLogging( HYPRE_Solver solver,
                           int level )
{
   return( HYPRE_PCGSetLogging( solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGGetNumIterations( HYPRE_Solver  solver,
                                 int                *num_iterations )
{
   return( HYPRE_PCGGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                             double             *norm   )
{
   return( HYPRE_PCGGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScaleSetup
 *--------------------------------------------------------------------------*/
 
int 
HYPRE_ParCSRDiagScaleSetup( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector y,
                            HYPRE_ParVector x      )
{
   return 0;
}
 
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScale
 *--------------------------------------------------------------------------*/
 
int 
HYPRE_ParCSRDiagScale( HYPRE_Solver solver,
                       HYPRE_ParCSRMatrix HA,
                       HYPRE_ParVector Hy,
                       HYPRE_ParVector Hx      )
{
   hypre_ParCSRMatrix *A = (hypre_ParCSRMatrix *) HA;
   hypre_ParVector    *y = (hypre_ParVector *) Hy;
   hypre_ParVector    *x = (hypre_ParVector *) Hx;
   double *x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   double *y_data = hypre_VectorData(hypre_ParVectorLocalVector(y));
   double *A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
   int *A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A));
   int local_size = hypre_VectorSize(hypre_ParVectorLocalVector(x));
   int i, ierr = 0;

   for (i=0; i < local_size; i++)
   {
	x_data[i] = y_data[i]/A_data[A_i[i]];
   } 
 
   return ierr;
}

