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
 * Preconditioned conjugate gradient (Omin) headers
 *
 *****************************************************************************/

#ifndef HYPRE_KRYLOV_PCG_HEADER
#define HYPRE_KRYLOV_PCG_HEADER

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name Generic PCG Interface
 *
 * A general description of the interface goes here...
 *
 * @memo A generic PCG linear solver interface
 * @version 0.1
 * @author Jeffrey F. Painter
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_PCGData and hypre_PCGFunctions
 *--------------------------------------------------------------------------*/


/**
 * @name PCG structs
 *
 * Description...
 **/
/*@{*/

/**
 * The {\tt hypre\_PCGSFunctions} object ...
 **/

typedef struct
{
   char * (*CAlloc)        ( int count, int elt_size );
   int    (*Free)          ( char *ptr );
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs );
   void * (*CreateVector)  ( void *vector );
   int    (*DestroyVector) ( void *vector );
   void * (*MatvecCreate)  ( void *A, void *x );
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y );
   int    (*MatvecDestroy) ( void *matvec_data );
   double (*InnerProd)     ( void *x, void *y );
   int    (*CopyVector)    ( void *x, void *y );
   int    (*ClearVector)   ( void *x );
   int    (*ScaleVector)   ( double alpha, void *x );
   int    (*Axpy)          ( double alpha, void *x, void *y );

   int    (*precond)();
   int    (*precond_setup)();
} hypre_PCGFunctions;

/**
 * The {\tt hypre\_PCGData} object ...
 **/

/*
 Summary of Parameters to Control Stopping Test:
 - Standard (default) error tolerance: |delta-residual|/|right-hand-side|<tol
 where the norm is an energy norm wrt preconditioner, |r|=sqrt(<Cr,r>).
 - two_norm!=0 means: the norm is the L2 norm, |r|=sqrt(<r,r>)
 - rel_change!=0 means: if pass the other stopping criteria, also check the
 relative change in the solution x.
 - stop_crit!=0 means: pure absolute error tolerance rather than a pure relative
 error tolerance on the residual.  Never applies if rel_change!=0 or atolf!=0.
 - atolf = absolute error tolerance factor to be used _together_ with the
 relative error tolerance, |delta-residual| / ( atolf + |right-hand-side| ) < tol
 - tol = relative error tolerance, as above
 - cf_tol = convergence factor tolerance; if >0 used for special test
 for slow convergence
*/

typedef struct
{
   double   tol;
   double   atolf;
   double   cf_tol;
   int      max_iter;
   int      two_norm;
   int      rel_change;
   int      stop_crit;
   int      converged;

   void    *A;
   void    *p;
   void    *s;
   void    *r; /* ...contains the residual.  This is currently kept permanently.
                  If that is ever changed, it still must be kept if logging>1 */

   int      owns_matvec_data;  /* normally 1; if 0, don't delete it */
   void    *matvec_data;
   void    *precond_data;

   hypre_PCGFunctions * functions;

   /* log info (always logged) */
   int      num_iterations;
   double   rel_residual_norm;

   int     print_level; /* printing when print_level>0 */
   int     logging;  /* extra computations for logging when logging>0 */
   double  *norms;
   double  *rel_norms;

} hypre_PCGData;

#define hypre_PCGDataOwnsMatvecData(pcgdata)  ((pcgdata) -> owns_matvec_data)

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @name generic PCG Solver
 *
 * Description...
 **/
/*@{*/

/**
 * Description...
 *
 * @param param [IN] ...
 **/

hypre_PCGFunctions *
hypre_PCGFunctionsCreate(
   char * (*CAlloc)        ( int count, int elt_size ),
   int    (*Free)          ( char *ptr ),
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   int    (*DestroyVector) ( void *vector ),
   void * (*MatvecCreate)  ( void *A, void *x ),
   int    (*Matvec)        ( void *matvec_data, double alpha, void *A,
                             void *x, double beta, void *y ),
   int    (*MatvecDestroy) ( void *matvec_data ),
   double (*InnerProd)     ( void *x, void *y ),
   int    (*CopyVector)    ( void *x, void *y ),
   int    (*ClearVector)   ( void *x ),
   int    (*ScaleVector)   ( double alpha, void *x ),
   int    (*Axpy)          ( double alpha, void *x, void *y ),
   int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
   );

/**
 * Description...
 *
 * @param param [IN] ...
 **/

void *
hypre_PCGCreate( hypre_PCGFunctions *pcg_functions );

#ifdef __cplusplus
}
#endif

#endif
