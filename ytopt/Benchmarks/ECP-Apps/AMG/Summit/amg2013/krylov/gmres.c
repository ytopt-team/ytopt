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
 * GMRES gmres
 *
 *****************************************************************************/

#include "krylov.h"
#include "utilities.h"

/*--------------------------------------------------------------------------
 * hypre_GMRESFunctionsCreate
 *--------------------------------------------------------------------------*/

hypre_GMRESFunctions *
hypre_GMRESFunctionsCreate(
   char * (*CAlloc)        ( int count, int elt_size ),
   int    (*Free)          ( char *ptr ),
   int    (*CommInfo)      ( void  *A, int   *my_id, int   *num_procs ),
   void * (*CreateVector)  ( void *vector ),
   void * (*CreateVectorArray)  ( int size, void *vectors ),
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
   )
{
   hypre_GMRESFunctions * gmres_functions;
   gmres_functions = (hypre_GMRESFunctions *)
      CAlloc( 1, sizeof(hypre_GMRESFunctions) );

   gmres_functions->CAlloc = CAlloc;
   gmres_functions->Free = Free;
   gmres_functions->CommInfo = CommInfo; /* not in PCGFunctionsCreate */
   gmres_functions->CreateVector = CreateVector;
   gmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
   gmres_functions->DestroyVector = DestroyVector;
   gmres_functions->MatvecCreate = MatvecCreate;
   gmres_functions->Matvec = Matvec;
   gmres_functions->MatvecDestroy = MatvecDestroy;
   gmres_functions->InnerProd = InnerProd;
   gmres_functions->CopyVector = CopyVector;
   gmres_functions->ClearVector = ClearVector;
   gmres_functions->ScaleVector = ScaleVector;
   gmres_functions->Axpy = Axpy;
/* default preconditioner must be set here but can be changed later... */
   gmres_functions->precond_setup = PrecondSetup;
   gmres_functions->precond       = Precond;

   return gmres_functions;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESCreate
 *--------------------------------------------------------------------------*/
 
void *
hypre_GMRESCreate( hypre_GMRESFunctions *gmres_functions )
{
   hypre_GMRESData *gmres_data;
 
   gmres_data = hypre_CTAllocF(hypre_GMRESData, 1, gmres_functions);

   gmres_data->functions = gmres_functions;
 
   /* set defaults */
   (gmres_data -> k_dim)          = 5;
   (gmres_data -> tol)            = 1.0e-06;
   (gmres_data -> cf_tol)         = 0.0;
   (gmres_data -> min_iter)       = 0;
   (gmres_data -> max_iter)       = 1000;
   (gmres_data -> rel_change)     = 0;
   (gmres_data -> stop_crit)      = 0; /* rel. residual norm */
   (gmres_data -> converged)      = 0;
   (gmres_data -> precond_data)   = NULL;
   (gmres_data -> print_level)    = 0;
   (gmres_data -> logging)        = 0;
   (gmres_data -> p)              = NULL;
   (gmres_data -> r)              = NULL;
   (gmres_data -> w)              = NULL;
   (gmres_data -> matvec_data)    = NULL;
   (gmres_data -> norms)          = NULL;
   (gmres_data -> log_file_name)  = NULL;
 
   return (void *) gmres_data;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESDestroy
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESDestroy( void *gmres_vdata )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;
   int i, ierr = 0;
 
   if (gmres_data)
   {
      if ( (gmres_data->logging>0) || (gmres_data->print_level) > 0 )
      {
         if ( (gmres_data -> norms) != NULL )
            hypre_TFreeF( gmres_data -> norms, gmres_functions );
      }
 
      if ( (gmres_data -> matvec_data) != NULL )
         (*(gmres_functions->MatvecDestroy))(gmres_data -> matvec_data);
 
      if ( (gmres_data -> r) != NULL )
         (*(gmres_functions->DestroyVector))(gmres_data -> r);
      if ( (gmres_data -> w) != NULL )
         (*(gmres_functions->DestroyVector))(gmres_data -> w);
      if ( (gmres_data -> p) != NULL )
      {
         for (i = 0; i < (gmres_data -> k_dim+1); i++)
         {
            if ( (gmres_data -> p)[i] != NULL )
	       (*(gmres_functions->DestroyVector))( (gmres_data -> p) [i]);
         }
         hypre_TFreeF( gmres_data->p, gmres_functions );
      }
      hypre_TFreeF( gmres_data, gmres_functions );
      hypre_TFreeF( gmres_functions, gmres_functions );
   }
 
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetResidual
 *--------------------------------------------------------------------------*/

int hypre_GMRESGetResidual( void *gmres_vdata, void **residual )
{
   /* returns a pointer to the residual vector */
   int ierr = 0;
   hypre_GMRESData  *gmres_data     = gmres_vdata;
   *residual = gmres_data->r;
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetup
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetup( void *gmres_vdata,
                  void *A,
                  void *b,
                  void *x         )
{
   hypre_GMRESData *gmres_data     = gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;

   int            k_dim            = (gmres_data -> k_dim);
   int            max_iter         = (gmres_data -> max_iter);
   int          (*precond_setup)() = (gmres_functions->precond_setup);
   void          *precond_data     = (gmres_data -> precond_data);
   int            ierr = 0;
 
   (gmres_data -> A) = A;
 
   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/
 
   if ((gmres_data -> p) == NULL)
      (gmres_data -> p) = (*(gmres_functions->CreateVectorArray))(k_dim+1,x);
   if ((gmres_data -> r) == NULL)
      (gmres_data -> r) = (*(gmres_functions->CreateVector))(b);
   if ((gmres_data -> w) == NULL)
      (gmres_data -> w) = (*(gmres_functions->CreateVector))(b);
 
   if ((gmres_data -> matvec_data) == NULL)
      (gmres_data -> matvec_data) = (*(gmres_functions->MatvecCreate))(A, x);
 
   ierr = precond_setup(precond_data, A, b, x);
 
   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/
 
   if ( (gmres_data->logging)>0 || (gmres_data->print_level) > 0 )
   {
      if ((gmres_data -> norms) == NULL)
         (gmres_data -> norms) = hypre_CTAllocF(double, max_iter + 1,gmres_functions);
   }
   if ( (gmres_data->print_level) > 0 ) {
      if ((gmres_data -> log_file_name) == NULL)
         (gmres_data -> log_file_name) = "gmres.out.log";
   }
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESSolve
 *-------------------------------------------------------------------------*/

int
hypre_GMRESSolve(void  *gmres_vdata,
                 void  *A,
                 void  *b,
		 void  *x)
{
   hypre_GMRESData  *gmres_data   = gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;
   int 		     k_dim        = (gmres_data -> k_dim);
   int               min_iter     = (gmres_data -> min_iter);
   int 		     max_iter     = (gmres_data -> max_iter);
   int               rel_change   = (gmres_data -> rel_change);
   int 		     stop_crit    = (gmres_data -> stop_crit);
   double 	     accuracy     = (gmres_data -> tol);
   double 	     cf_tol       = (gmres_data -> cf_tol);
   void             *matvec_data  = (gmres_data -> matvec_data);

   void             *r            = (gmres_data -> r);
   void             *w            = (gmres_data -> w);
   void            **p            = (gmres_data -> p);

   int 	           (*precond)()   = (gmres_functions -> precond);
   int 	            *precond_data = (gmres_data -> precond_data);

   int             print_level    = (gmres_data -> print_level);
   int             logging        = (gmres_data -> logging);

   double         *norms          = (gmres_data -> norms);
/* not used yet   char           *log_file_name  = (gmres_data -> log_file_name);*/
/*   FILE           *fp; */
   
   int        ierr = 0;
   int        break_value = 0;
   int	      i, j, k;
   double     *rs, **hh, *c, *s;
   int        iter; 
   int        my_id, num_procs;
   double     epsilon, gamma, t, r_norm, b_norm, den_norm, x_norm;
   double     epsmac = 1.e-16; 
   double     ieee_check = 0.;

   double     guard_zero_residual; 
   double     cf_ave_0 = 0.0;
   double     cf_ave_1 = 0.0;
   double     weight;
   double     r_norm_0;
   double     relative_error;

   (gmres_data -> converged) = 0;
   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/
   guard_zero_residual = 0.0;

   (*(gmres_functions->CommInfo))(A,&my_id,&num_procs);
   if ( logging>0 || print_level>0 )
   {
      norms          = (gmres_data -> norms);
      /* not used yet      log_file_name  = (gmres_data -> log_file_name);*/
      /* fp = fopen(log_file_name,"w"); */
   }

   /* initialize work arrays */
   rs = hypre_CTAllocF(double,k_dim+1,gmres_functions); 
   c = hypre_CTAllocF(double,k_dim,gmres_functions); 
   s = hypre_CTAllocF(double,k_dim,gmres_functions); 

   hh = hypre_CTAllocF(double*,k_dim+1,gmres_functions); 
   for (i=0; i < k_dim+1; i++)
   {	
   	hh[i] = hypre_CTAllocF(double,k_dim,gmres_functions); 
   }

   (*(gmres_functions->CopyVector))(b,p[0]);

   /* compute initial residual */
   (*(gmres_functions->Matvec))(matvec_data,-1.0, A, x, 1.0, p[0]);

   b_norm = sqrt((*(gmres_functions->InnerProd))(b,b));

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (b_norm != 0.) ieee_check = b_norm/b_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
        printf("\n\nERROR detected by Hypre ... BEGIN\n");
        printf("ERROR -- hypre_GMRESSolve: INFs and/or NaNs detected in input.\n");
        printf("User probably placed non-numerics in supplied b.\n");
        printf("Returning error flag += 101.  Program not terminated.\n");
        printf("ERROR detected by Hypre ... END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

   r_norm = sqrt((*(gmres_functions->InnerProd))(p[0],p[0]));
   r_norm_0 = r_norm;

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (r_norm != 0.) ieee_check = r_norm/r_norm; /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
        printf("\n\nERROR detected by Hypre ... BEGIN\n");
        printf("ERROR -- hypre_GMRESSolve: INFs and/or NaNs detected in input.\n");
        printf("User probably placed non-numerics in supplied A or x_0.\n");
        printf("Returning error flag += 101.  Program not terminated.\n");
        printf("ERROR detected by Hypre ... END\n\n\n");
      }
      ierr += 101;
      return ierr;
   }

   if ( logging>0 || print_level > 0)
   {
      norms[0] = r_norm;
      if ( print_level>1 && my_id == 0 )
      {
  	 printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
            printf("Rel_resid_norm actually contains the residual norm\n");
         printf("Initial L2 norm of residual: %e\n", r_norm);
      
      }
   }
   iter = 0;

   if (b_norm > 0.0)
   {
/* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
     den_norm= b_norm;
   }
   else
   {
/* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
     den_norm= r_norm;
   };

   epsilon= accuracy;

/* convergence criterion |r_i| <= accuracy , absolute residual norm*/
   if ( stop_crit && !rel_change )
      epsilon = accuracy;

   if ( print_level>1 && my_id == 0 )
   {
      if (b_norm > 0.0)
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
          printf("-----    ------------    ---------- ------------\n");
      
          }

      else
         {printf("=============================================\n\n");
          printf("Iters     resid.norm     conv.rate\n");
          printf("-----    ------------    ----------\n");
      
          };
   }

  /* set the relative_error to initially bypass the stopping criterion */
   if (rel_change)
   {
      relative_error= epsilon + 1.;
   }

   while (iter < max_iter)
   {
   /* initialize first term of hessenberg system */

	rs[0] = r_norm;
        if (r_norm == 0.0)
        {
           hypre_TFreeF(c,gmres_functions); 
           hypre_TFreeF(s,gmres_functions); 
           hypre_TFreeF(rs,gmres_functions);
           for (i=0; i < k_dim+1; i++) hypre_TFreeF(hh[i],gmres_functions);
           hypre_TFreeF(hh,gmres_functions); 
	   ierr = 0;
	   return ierr;
	}

	if (r_norm/den_norm <= epsilon && iter >= min_iter) 
        {
                if (rel_change)
                {
                   if (relative_error <= epsilon)
                   {
		      (*(gmres_functions->CopyVector))(b,r);
          	      (*(gmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
		      r_norm = sqrt((*(gmres_functions->InnerProd))(r,r));
		      if (r_norm/den_norm <= epsilon)
                      {
                         if ( print_level>1 && my_id == 0)
                         {
                            printf("\n\n");
                            printf("Final L2 norm of residual: %e\n\n", r_norm);
                         }
                         break;
                      }
                      else
                      if ( print_level>0 && my_id == 0)
                           printf("false convergence 1\n");
                   }
                }
                else
                {
                   (*(gmres_functions->CopyVector))(b,r);
                   (*(gmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
                   r_norm = sqrt((*(gmres_functions->InnerProd))(r,r));
                   if (r_norm/den_norm <= epsilon)
                   {
                       if ( print_level>1 && my_id == 0)
                       {
                            printf("\n\n");
                            printf("Final L2 norm of residual: %e\n\n", r_norm);
                       }
                       break;
                   }
                   else
                      if ( print_level>0 && my_id == 0)
                           printf("false convergence 1\n");
                }

	}

      	t = 1.0 / r_norm;
	(*(gmres_functions->ScaleVector))(t,p[0]);
	i = 0;
	while (i < k_dim && ( (r_norm/den_norm > epsilon || iter < min_iter)
                         || ((rel_change) && relative_error > epsilon) )
                         && iter < max_iter)
	{
		i++;
		iter++;
		(*(gmres_functions->ClearVector))(r);
		precond(precond_data, A, p[i-1], r);
		(*(gmres_functions->Matvec))(matvec_data, 1.0, A, r, 0.0, p[i]);
		/* modified Gram_Schmidt */
		for (j=0; j < i; j++)
		{
			hh[j][i-1] = (*(gmres_functions->InnerProd))(p[j],p[i]);
			(*(gmres_functions->Axpy))(-hh[j][i-1],p[j],p[i]);
		}
		t = sqrt((*(gmres_functions->InnerProd))(p[i],p[i]));
		hh[i][i-1] = t;	
		if (t != 0.0)
		{
			t = 1.0/t;
			(*(gmres_functions->ScaleVector))(t,p[i]);
		}
		/* done with modified Gram_schmidt and Arnoldi step.
		   update factorization of hh */
		for (j = 1; j < i; j++)
		{
			t = hh[j-1][i-1];
			hh[j-1][i-1] = c[j-1]*t + s[j-1]*hh[j][i-1];		
			hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
		}
		gamma = sqrt(hh[i-1][i-1]*hh[i-1][i-1] + hh[i][i-1]*hh[i][i-1]);
		if (gamma == 0.0) gamma = epsmac;
		c[i-1] = hh[i-1][i-1]/gamma;
		s[i-1] = hh[i][i-1]/gamma;
		rs[i] = -s[i-1]*rs[i-1];
		rs[i-1] = c[i-1]*rs[i-1];
		/* determine residual norm */
		hh[i-1][i-1] = c[i-1]*hh[i-1][i-1] + s[i-1]*hh[i][i-1];
		r_norm = fabs(rs[i]);
		if ( print_level>0 )
		{
		   norms[iter] = r_norm;
                   if ( print_level>1 && my_id == 0 )
   		   {
      		      if (b_norm > 0.0)
             	         printf("% 5d    %e    %f   %e\n", iter, 
				norms[iter],norms[iter]/norms[iter-1],
 	             		norms[iter]/b_norm);
      		      else
             	         printf("% 5d    %e    %f\n", iter, norms[iter],
				norms[iter]/norms[iter-1]);
   		   }
		}
                if (cf_tol > 0.0)
                {
                   cf_ave_0 = cf_ave_1;
           	   cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));

           	   weight   = fabs(cf_ave_1 - cf_ave_0);
           	   weight   = weight / hypre_max(cf_ave_1, cf_ave_0);
           	   weight   = 1.0 - weight;
#if 0
           	   printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
                	i, cf_ave_1, cf_ave_0, weight );
#endif
           	   if (weight * cf_ave_1 > cf_tol) 
		   {
		      break_value = 1;
		      break;
		   }
        	}

	}
	/* now compute solution, first solve upper triangular system */

	if (break_value) break;
	
	rs[i-1] = rs[i-1]/hh[i-1][i-1];
	for (k = i-2; k >= 0; k--)
	{
		t = rs[k];
		for (j = k+1; j < i; j++)
		{
			t -= hh[k][j]*rs[j];
		}
		rs[k] = t/hh[k][k];
	}
	
	(*(gmres_functions->CopyVector))(p[0],w);
	(*(gmres_functions->ScaleVector))(rs[0],w);
	for (j = 1; j < i; j++)
		(*(gmres_functions->Axpy))(rs[j], p[j], w);

	(*(gmres_functions->ClearVector))(r);
	precond(precond_data, A, w, r);

	(*(gmres_functions->Axpy))(1.0,r,x);

/* check for convergence, evaluate actual residual */
	if (r_norm/den_norm <= epsilon && iter >= min_iter) 
        {
                if (rel_change)
                {
                   x_norm = sqrt( (*(gmres_functions->InnerProd))(x,x) );
                   if ( x_norm<=guard_zero_residual ) break; /* don't divide by 0 */
		   r_norm = sqrt( (*(gmres_functions->InnerProd))(r,r) );
                   relative_error= r_norm/x_norm;
                }

		(*(gmres_functions->CopyVector))(b,r);
          	(*(gmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
		r_norm = sqrt( (*(gmres_functions->InnerProd))(r,r) );
		if (r_norm/den_norm <= epsilon)
                {
                   if ( print_level>1 && my_id == 0 )
                   {
                      printf("\n\n");
                      printf("Final L2 norm of residual: %e\n\n", r_norm);
                   }
                   if (rel_change && r_norm > guard_zero_residual)
                      /* Also test on relative change of iterates, x_i - x_(i-1) */
                   {  /* At this point r = x_i - x_(i-1) */
                      x_norm = sqrt( (*(gmres_functions->InnerProd))(x,x) );
                      if ( x_norm<=guard_zero_residual ) break; /* don't divide by 0 */
                      if ( relative_error < epsilon )
                      {
                         (gmres_data -> converged) = 1;
                         break;
                      }
                   }
                   else
                   {
                      (gmres_data -> converged) = 1;
                      break;
                   }
                }
		else 
                {
                   if ( print_level>0 && my_id == 0)
                      printf("false convergence 2\n");
		   (*(gmres_functions->CopyVector))(r,p[0]);
		   i = 0;
		}
	}

/* compute residual vector and continue loop */

	for (j=i ; j > 0; j--)
	{
		rs[j-1] = -s[j-1]*rs[j];
		rs[j] = c[j-1]*rs[j];
	}

	if (i) (*(gmres_functions->Axpy))(rs[0]-1.0,p[0],p[0]);
	for (j=1; j < i+1; j++)
		(*(gmres_functions->Axpy))(rs[j],p[j],p[0]);	
   }

   if ( print_level>1 && my_id == 0 )
          printf("\n\n"); 

   (gmres_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (gmres_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (gmres_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm/den_norm > epsilon) ierr = 1;

   hypre_TFreeF(c,gmres_functions); 
   hypre_TFreeF(s,gmres_functions); 
   hypre_TFreeF(rs,gmres_functions);
 
   for (i=0; i < k_dim+1; i++)
   {	
   	hypre_TFreeF(hh[i],gmres_functions);
   }
   hypre_TFreeF(hh,gmres_functions); 

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetKDim, hypre_GMRESGetKDim
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetKDim( void   *gmres_vdata,
                    int   k_dim )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> k_dim) = k_dim;
 
   return ierr;
}

int
hypre_GMRESGetKDim( void   *gmres_vdata,
                    int * k_dim )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   *k_dim = (gmres_data -> k_dim);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetTol, hypre_GMRESGetTol
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetTol( void   *gmres_vdata,
                   double  tol       )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> tol) = tol;
 
   return ierr;
}

int
hypre_GMRESGetTol( void   *gmres_vdata,
                   double  * tol      )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   *tol = (gmres_data -> tol);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetConvergenceFactorTol, hypre_GMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetConvergenceFactorTol( void   *gmres_vdata,
                   double  cf_tol       )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> cf_tol) = cf_tol;
 
   return ierr;
}

int
hypre_GMRESGetConvergenceFactorTol( void   *gmres_vdata,
                   double * cf_tol       )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   *cf_tol = (gmres_data -> cf_tol);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetMinIter, hypre_GMRESGetMinIter
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetMinIter( void *gmres_vdata,
                       int   min_iter  )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   (gmres_data -> min_iter) = min_iter;
 
   return ierr;
}

int
hypre_GMRESGetMinIter( void *gmres_vdata,
                       int * min_iter  )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   *min_iter = (gmres_data -> min_iter);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetMaxIter, hypre_GMRESGetMaxIter
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetMaxIter( void *gmres_vdata,
                       int   max_iter  )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   (gmres_data -> max_iter) = max_iter;
 
   return ierr;
}

int
hypre_GMRESGetMaxIter( void *gmres_vdata,
                       int * max_iter  )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   *max_iter = (gmres_data -> max_iter);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetRelChange, hypre_GMRESGetRelChange
 *--------------------------------------------------------------------------*/

int
hypre_GMRESSetRelChange( void *gmres_vdata,
                         int   rel_change  )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> rel_change) = rel_change;
 
   return ierr;
}

int
hypre_GMRESGetRelChange( void *gmres_vdata,
                         int * rel_change  )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   *rel_change = (gmres_data -> rel_change);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetStopCrit, hypre_GMRESGetStopCrit
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetStopCrit( void   *gmres_vdata,
                        int  stop_crit       )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> stop_crit) = stop_crit;
 
   return ierr;
}

int
hypre_GMRESGetStopCrit( void   *gmres_vdata,
                        int * stop_crit       )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   *stop_crit = (gmres_data -> stop_crit);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetPrecond
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESSetPrecond( void  *gmres_vdata,
                       int  (*precond)(),
                       int  (*precond_setup)(),
                       void  *precond_data )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   hypre_GMRESFunctions *gmres_functions = gmres_data->functions;
   int              ierr = 0;
 
   (gmres_functions -> precond)        = precond;
   (gmres_functions -> precond_setup)  = precond_setup;
   (gmres_data -> precond_data)   = precond_data;
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESGetPrecond
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESGetPrecond( void         *gmres_vdata,
                       HYPRE_Solver *precond_data_ptr )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   *precond_data_ptr = (HYPRE_Solver)(gmres_data -> precond_data);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESSetPrintLevel, hypre_GMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

int
hypre_GMRESSetPrintLevel( void *gmres_vdata,
                        int   level)
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> print_level) = level;
 
   return ierr;
}

int
hypre_GMRESGetPrintLevel( void *gmres_vdata,
                        int * level)
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   *level = (gmres_data -> print_level);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESSetLogging, hypre_GMRESGetLogging
 *--------------------------------------------------------------------------*/

int
hypre_GMRESSetLogging( void *gmres_vdata,
                      int   level)
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   (gmres_data -> logging) = level;
 
   return ierr;
}

int
hypre_GMRESGetLogging( void *gmres_vdata,
                      int * level)
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int            ierr = 0;
 
   *level = (gmres_data -> logging);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_GMRESGetNumIterations
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESGetNumIterations( void *gmres_vdata,
                             int  *num_iterations )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   *num_iterations = (gmres_data -> num_iterations);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESGetConverged
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESGetConverged( void *gmres_vdata,
                             int  *converged )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int              ierr = 0;
 
   *converged = (gmres_data -> converged);
 
   return ierr;
}
 
/*--------------------------------------------------------------------------
 * hypre_GMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/
 
int
hypre_GMRESGetFinalRelativeResidualNorm( void   *gmres_vdata,
                                         double *relative_residual_norm )
{
   hypre_GMRESData *gmres_data = gmres_vdata;
   int 		ierr = 0;
 
   *relative_residual_norm = (gmres_data -> rel_residual_norm);
   
   return ierr;
} 
