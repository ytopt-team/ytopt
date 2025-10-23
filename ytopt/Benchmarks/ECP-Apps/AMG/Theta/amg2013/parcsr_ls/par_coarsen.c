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
  Selects a coarse "grid" based on the graph of a matrix.

  Notes:
  \begin{itemize}
  \item The underlying matrix storage scheme is a hypre_ParCSR matrix.
  \item The routine returns the following:
  \begin{itemize}
  \item S - a ParCSR matrix representing the "strength matrix".  This is
  used in the "build interpolation" routine.
  \item CF\_marker - an array indicating both C-pts (value = 1) and
  F-pts (value = -1)
  \end{itemize}
  \item We define the following temporary storage:
  \begin{itemize}
  \item measure\_array - an array containing the "measures" for each
  of the fine-grid points
  \item graph\_array - an array containing the list of points in the
  "current subgraph" being considered in the coarsening process.
  \end{itemize}
  \item The graph of the "strength matrix" for A is a subgraph of the
  graph of A, but requires nonsymmetric storage even if A is
  symmetric.  This is because of the directional nature of the
  "strengh of dependence" notion (see below).  Since we are using
  nonsymmetric storage for A right now, this is not a problem.  If we
  ever add the ability to store A symmetrically, then we could store
  the strength graph as floats instead of doubles to save space.
  \item This routine currently "compresses" the strength matrix.  We
  should consider the possibility of defining this matrix to have the
  same "nonzero structure" as A.  To do this, we could use the same
  A\_i and A\_j arrays, and would need only define the S\_data array.
  There are several pros and cons to discuss.
  \end{itemize}

  Terminology:
  \begin{itemize}
  \item Ruge's terminology: A point is "strongly connected to" $j$, or
  "strongly depends on" $j$, if $-a_ij >= \theta max_{l != j} \{-a_il\}$.
  \item Here, we retain some of this terminology, but with a more
  generalized notion of "strength".  We also retain the "natural"
  graph notation for representing the directed graph of a matrix.
  That is, the nonzero entry $a_ij$ is represented as: i --> j.  In
  the strength matrix, S, the entry $s_ij$ is also graphically denoted
  as above, and means both of the following:
  \begin{itemize}
  \item $i$ "depends on" $j$ with "strength" $s_ij$
  \item $j$ "influences" $i$ with "strength" $s_ij$
  \end{itemize}
  \end{itemize}

  {\bf Input files:}
  headers.h

  @return Error code.
  
  @param A [IN]
  coefficient matrix
  @param strength_threshold [IN]
  threshold parameter used to define strength
  @param S_ptr [OUT]
  strength matrix
  @param CF_marker_ptr [OUT]
  array indicating C/F points
  
  @see */
/*--------------------------------------------------------------------------*/

#define C_PT  1
#define F_PT -1
#define SF_PT -3
#define COMMON_C_PT  2
#define Z_PT -2

int
hypre_BoomerAMGCoarsen( hypre_ParCSRMatrix    *S,
                        hypre_ParCSRMatrix    *A,
                        int                    CF_init,
                        int                    debug_flag,
                        int                  **CF_marker_ptr)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j;

   /*HYPRE_BigInt	      *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);*/
   int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   /*HYPRE_BigInt	       col_1 = hypre_ParCSRMatrixFirstColDiag(S);
   HYPRE_BigInt	       col_n = col_1 + (HYPRE_BigInt)hypre_CSRMatrixNumCols(S_diag);*/
   int 		       num_cols_offd = 0;
                  
   hypre_CSRMatrix    *S_ext;
   int                *S_ext_i;
   int                *S_ext_j;

   int		       num_sends = 0;
   int  	      *int_buf_data;
   double	      *buf_data;

   int                *CF_marker;
   int                *CF_marker_offd;
                      
   double             *measure_array;
   int                *graph_array;
   int                *graph_array_offd;
   int                 graph_size;
   HYPRE_BigInt        big_graph_size;
   int                 graph_offd_size;
   HYPRE_BigInt        global_graph_size;
                      
   int                 i, j, k, kc, jS, kS, ig, elmt;
   int		       index, start, my_id, num_procs, jrow, cnt;
                      
   int                 ierr = 0;
   int                 use_commpkg_A = 0;
   int                 break_var = 1;

   double	    wall_time;
   int   iter = 0;

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

   /*--------------------------------------------------------------
    * Compute a  ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   S_ext = NULL;
   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
        use_commpkg_A = 1;
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
        hypre_NewCommPkgCreate(A);
#else
        hypre_MatvecCommPkgCreate(A);
#endif
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = hypre_CSRMatrixJ(S_offd);
   }
   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are currently given by the column sums of S.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    *
    * The measures are augmented by a random number
    * between 0 and 1.
    *----------------------------------------------------------*/

   measure_array = hypre_CTAlloc(double, num_variables+num_cols_offd);

   for (i=0; i < S_offd_i[num_variables]; i++)
   { 
      measure_array[num_variables + S_offd_j[i]] += 1.0;
   }
   if (num_procs > 1)
   comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg, 
                        &measure_array[num_variables], buf_data);

   for (i=0; i < S_diag_i[num_variables]; i++)
   { 
      measure_array[S_diag_j[i]] += 1.0;
   }

   if (num_procs > 1)
   hypre_ParCSRCommHandleDestroy(comm_handle);
      
   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += buf_data[index++];
   }

   for (i=num_variables; i < num_variables+num_cols_offd; i++)
   { 
      measure_array[i] = 0;
   }

   /* this augments the measures */
   if (CF_init == 2)
      hypre_BoomerAMGIndepSetInit(S, measure_array, 1);
   else
      hypre_BoomerAMGIndepSetInit(S, measure_array, 0);

   /*---------------------------------------------------
    * Initialize the graph array
    * graph_array contains interior points in elements 0 ... num_variables-1
    * followed by boundary values
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(int, num_variables);
   if (num_cols_offd) 
      graph_array_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      graph_array_offd = NULL;

   /* initialize measure array and graph array */

   for (ig = 0; ig < num_cols_offd; ig++)
      graph_array_offd[ig] = ig;

   /*---------------------------------------------------
    * Initialize the C/F marker array
    * C/F marker array contains interior points in elements 0 ... 
    * num_variables-1  followed by boundary values
    *---------------------------------------------------*/

   graph_offd_size = num_cols_offd;

   if (CF_init==1)
   {
      CF_marker = *CF_marker_ptr;
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
         if ( (S_offd_i[i+1]-S_offd_i[i]) > 0
                 || CF_marker[i] == -1)
         {
            CF_marker[i] = 0;
         }
         if ( CF_marker[i] == Z_PT)
         {
            if (measure_array[i] >= 1.0 ||
                (S_diag_i[i+1]-S_diag_i[i]) > 0)
            {
               CF_marker[i] = 0;
               graph_array[cnt++] = i;
            }
            else
            {
               CF_marker[i] = F_PT;
            }
         }
         else if (CF_marker[i] == SF_PT)
	    measure_array[i] = 0;
         else 
            graph_array[cnt++] = i;
      }
   }
   else
   {
      CF_marker = hypre_CTAlloc(int, num_variables);
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
	 CF_marker[i] = 0;
	 if ( (S_diag_i[i+1]-S_diag_i[i]) == 0 
		&& (S_offd_i[i+1]-S_offd_i[i]) == 0)
	 {
	    CF_marker[i] = SF_PT;
	    measure_array[i] = 0;
	 }
	 else
            graph_array[cnt++] = i;
      }
   }
   graph_size = cnt;
   if (num_cols_offd)
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      CF_marker_offd = NULL;
   for (i=0; i < num_cols_offd; i++)
	CF_marker_offd[i] = 0;
  
   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (num_procs > 1)
   {
      if (use_commpkg_A)
         S_ext      = hypre_ParCSRMatrixExtractConvBExt(S,A,0);
      else
         S_ext      = hypre_ParCSRMatrixExtractConvBExt(S,S,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
   }

   /*  compress S_ext  and convert column numbers*/

   /* index = 0;
   for (i=0; i < num_cols_offd; i++)
   {
      for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++)
      {
	 big_k = S_ext_j[j];
	 if (big_k >= col_1 && big_k < col_n)
	 {
	    S_ext_j[index++] = (int)(big_k - col_1);
	 }
	 else
	 {
	    kc = hypre_BigBinarySearch(col_map_offd,big_k,num_cols_offd);
	    if (kc > -1) S_ext_j[index++] = -kc-1;
	 }
      }
      S_ext_i[i] = index;
   }
   for (i = num_cols_offd; i > 0; i--)
      S_ext_i[i] = S_ext_i[i-1];
   if (num_procs > 1) S_ext_i[0] = 0; */
   
   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Initialize CLJP phase = %f\n",
                     my_id, wall_time); 
   }

   while (1)
   {
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures and S_ext_data
       *------------------------------------------------*/

      if (num_procs > 1)
   	 comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg, 
                        &measure_array[num_variables], buf_data);

      if (num_procs > 1)
   	 hypre_ParCSRCommHandleDestroy(comm_handle);
      
      index = 0;
      for (i=0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += buf_data[index++];
      }

      /*------------------------------------------------
       * Set F-pts and update subgraph
       *------------------------------------------------*/
 
      if (iter || (CF_init != 1))
      {
         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];

            if ( (CF_marker[i] != C_PT) && (measure_array[i] < 1) )
            {
               /* set to be an F-pt */
               CF_marker[i] = F_PT;
 
	       /* make sure all dependencies have been accounted for */
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     CF_marker[i] = 0;
                  }
               }
            }
            if (CF_marker[i])
            {
               measure_array[i] = 0;
 
               /* take point out of the subgraph */
               graph_size--;
               graph_array[ig] = graph_array[graph_size];
               graph_array[graph_size] = i;
               ig--;
            }
         }
      }
 
      /*------------------------------------------------
       * Exchange boundary data, i.i. get measures 
       *------------------------------------------------*/

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        {
            jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            buf_data[index++] = measure_array[jrow];
         }
      }

      if (num_procs > 1)
      { 
         comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data, 
        	&measure_array[num_variables]);
 
         hypre_ParCSRCommHandleDestroy(comm_handle);   
 
      } 
      /*------------------------------------------------
       * Debugging:
       *
       * Uncomment the sections of code labeled
       * "debugging" to generate several files that
       * can be visualized using the `coarsen.m'
       * matlab routine.
       *------------------------------------------------*/

#if 0 /* debugging */
      /* print out measures */
      sprintf(filename, "coarsen.out.measures.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%f\n", measure_array[i]);
      }
      fclose(fp);

      /* print out strength matrix */
      sprintf(filename, "coarsen.out.strength.%04d", iter);
      hypre_CSRMatrixPrint(S, filename);

      /* print out C/F marker */
      sprintf(filename, "coarsen.out.CF.%04d", iter);
      fp = fopen(filename, "w");
      for (i = 0; i < num_variables; i++)
      {
         fprintf(fp, "%d\n", CF_marker[i]);
      }
      fclose(fp);

      iter++;
#endif

      /*------------------------------------------------
       * Test for convergence
       *------------------------------------------------*/
      big_graph_size = (HYPRE_BigInt) graph_size;

      MPI_Allreduce(&big_graph_size,&global_graph_size,1,MPI_HYPRE_BIG_INT,MPI_SUM,comm);

      if (global_graph_size == 0)
         break;

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       *------------------------------------------------*/
      if (iter || (CF_init != 1))
      {
         hypre_BoomerAMGIndepSet(S, measure_array, graph_array, 
				graph_size, 
				graph_array_offd, graph_offd_size, 
				CF_marker, CF_marker_offd);
         if (num_procs > 1)
         {
            comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
                CF_marker_offd, int_buf_data);

            hypre_ParCSRCommHandleDestroy(comm_handle);
         }

         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1);j++)            {
               elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
               if (!int_buf_data[index++] && CF_marker[elmt] > 0)
               {
                  CF_marker[elmt] = 0;
               }
            }
         }
      }

      iter++;
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/


      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
        {
           elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
          int_buf_data[index++] = CF_marker[elmt];
        }
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, 
        	CF_marker_offd);
 
         hypre_ParCSRCommHandleDestroy(comm_handle);   
      }
 
      for (ig = 0; ig < graph_offd_size; ig++)
      {
         i = graph_array_offd[ig];

         if (CF_marker_offd[i] < 0)
         {
            /* take point out of the subgraph */
            graph_offd_size--;
            graph_array_offd[ig] = graph_array_offd[graph_offd_size];
            graph_array_offd[graph_offd_size] = i;
            ig--;
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d  iter %d  comm. and subgraph update = %f\n",
                     my_id, iter, wall_time); 
      }
      /*------------------------------------------------
       * Set C_pts and apply heuristics.
       *------------------------------------------------*/

      for (i=num_variables; i < num_variables+num_cols_offd; i++)
      { 
         measure_array[i] = 0;
      }

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();
      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * Heuristic: C-pts don't interpolate from
          * neighbors that influence them.
          *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {  
            /* set to be a C-pt */
            CF_marker[i] = C_PT;

            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               j = S_diag_j[jS];
               if (j > -1)
               {
               
                  /* "remove" edge from S */
                  S_diag_j[jS] = -S_diag_j[jS]-1;
             
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker[j])
                  {
                     measure_array[j]--;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
               if (j > -1)
               {
             
                  /* "remove" edge from S */
                  S_offd_j[jS] = -S_offd_j[jS]-1;
               
                  /* decrement measures of unmarked neighbors */
                  if (!CF_marker_offd[j])
                  {
                     measure_array[j+num_variables]--;
                  }
               }
            }
         }
	 else
    	 {
            /* marked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               j = S_diag_j[jS];
	       if (j < 0) j = -j-1;
   
               if (CF_marker[j] > 0)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker[j] = COMMON_C_PT;
               }
               else if (CF_marker[j] == SF_PT)
               {
                  if (S_diag_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_diag_j[jS] = -S_diag_j[jS]-1;
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               j = S_offd_j[jS];
	       if (j < 0) j = -j-1;
   
               if (CF_marker_offd[j] > 0)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
                  }
   
                  /* IMPORTANT: consider all dependencies */
                  /* temporarily modify CF_marker */
                  CF_marker_offd[j] = COMMON_C_PT;
               }
               else if (CF_marker_offd[j] == SF_PT)
               {
                  if (S_offd_j[jS] > -1)
                  {
                     /* "remove" edge from S */
                     S_offd_j[jS] = -S_offd_j[jS]-1;
                  }
               }
            }
   
            /* unmarked dependencies */
            for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
            {
               if (S_diag_j[jS] > -1)
               {
                  j = S_diag_j[jS];
   	          break_var = 1;
                  /* check for common C-pt */
                  for (kS = S_diag_i[j]; kS < S_diag_i[j+1]; kS++)
                  {
                     k = S_diag_j[kS];
		     if (k < 0) k = -k-1;
   
                     /* IMPORTANT: consider all dependencies */
                     if (CF_marker[k] == COMMON_C_PT)
                     {
                        /* "remove" edge from S and update measure*/
                        S_diag_j[jS] = -S_diag_j[jS]-1;
                        measure_array[j]--;
                        break_var = 0;
                        break;
                     }
                  }
   		  if (break_var)
                  {
                     for (kS = S_offd_i[j]; kS < S_offd_i[j+1]; kS++)
                     {
                        k = S_offd_j[kS];
		        if (k < 0) k = -k-1;
   
                        /* IMPORTANT: consider all dependencies */
                        if ( CF_marker_offd[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_diag_j[jS] = -S_diag_j[jS]-1;
                           measure_array[j]--;
                           break;
                        }
                     }
                  }
               }
            }
            for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
            {
               if (S_offd_j[jS] > -1)
               {
                  j = S_offd_j[jS];
   
                  /* check for common C-pt */
                  for (kS = S_ext_i[j]; kS < S_ext_i[j+1]; kS++)
                  {
                     k = S_ext_j[kS];
   	             if (k >= 0)
   		     {
                        /* IMPORTANT: consider all dependencies */
                        if (CF_marker[k] == COMMON_C_PT)
                        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS]-1;
                           measure_array[j+num_variables]--;
                           break;
                        }
                     }
   		     else
   		     {
   		        kc = -k-1;
   		        if (kc > -1 && CF_marker_offd[kc] == COMMON_C_PT)
   		        {
                           /* "remove" edge from S and update measure*/
                           S_offd_j[jS] = -S_offd_j[jS]-1;
                           measure_array[j+num_variables]--;
                           break;
   		        }
   		     }
                  }
               }
            }
         }

         /* reset CF_marker */
	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
	 {
            j = S_diag_j[jS];
	    if (j < 0) j = -j-1;

            if (CF_marker[j] == COMMON_C_PT)
            {
               CF_marker[j] = C_PT;
            }
         }
         for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
         {
            j = S_offd_j[jS];
	    if (j < 0) j = -j-1;

            if (CF_marker_offd[j] == COMMON_C_PT)
            {
               CF_marker_offd[j] = C_PT;
            }
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    CLJP phase = %f graph_size = %d nc_offd = %d\n",
                     my_id, wall_time, graph_size, num_cols_offd); 
      }
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   /* Reset S_matrix */
   for (i=0; i < S_diag_i[num_variables]; i++)
   {
      if (S_diag_j[i] < 0)
         S_diag_j[i] = -S_diag_j[i]-1;
   }
   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      if (S_offd_j[i] < 0)
         S_offd_j[i] = -S_offd_j[i]-1;
   }
   /*for (i=0; i < num_variables; i++)
      if (CF_marker[i] == SF_PT) CF_marker[i] = F_PT;*/

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   if (num_cols_offd) hypre_TFree(graph_array_offd);
   hypre_TFree(buf_data);
   hypre_TFree(int_buf_data);
   hypre_TFree(CF_marker_offd);
   if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);

   *CF_marker_ptr   = CF_marker;

   return (ierr);
}

/*==========================================================================
 * Ruge's coarsening algorithm                        
 *==========================================================================*/

#define C_PT 1
#define F_PT -1
#define Z_PT -2
#define SF_PT -3  /* special fine points */
#define SC_PT 3  /* special coarse points */
#define UNDECIDED 0 


/**************************************************************
 *
 *      Ruge Coarsening routine
 *
 **************************************************************/
int
hypre_BoomerAMGCoarsenRuge( hypre_ParCSRMatrix    *S,
                            hypre_ParCSRMatrix    *A,
                            int                    measure_type,
                            int                    coarsen_type,
                            int                    debug_flag,
                            int                  **CF_marker_ptr)
{
   MPI_Comm         comm          = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg   *comm_pkg      = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle *comm_handle;
   hypre_CSRMatrix *S_diag        = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix *S_offd        = hypre_ParCSRMatrixOffd(S);
   int             *S_i           = hypre_CSRMatrixI(S_diag);
   int             *S_j           = hypre_CSRMatrixJ(S_diag);
   int             *S_offd_i      = hypre_CSRMatrixI(S_offd);
   int             *S_offd_j;
   int              num_variables = hypre_CSRMatrixNumRows(S_diag);
   int              num_cols_offd = hypre_CSRMatrixNumCols(S_offd);
   /*HYPRE_BigInt    *col_map_offd    = hypre_ParCSRMatrixColMapOffd(S);*/
                  
   hypre_CSRMatrix *S_ext;
   int             *S_ext_i;
   int             *S_ext_j;
                 
   hypre_CSRMatrix *ST;
   int             *ST_i;
   int             *ST_j;
                 
   int             *CF_marker;
   int             *CF_marker_offd;
   int              ci_tilde = -1;
   int              ci_tilde_mark = -1;
   int              ci_tilde_offd = -1;
   int              ci_tilde_offd_mark = -1;

   int             *measure_array;
   int             *graph_array;
   int 	           *int_buf_data = NULL;
   int 	           *ci_array;

   int              i, j, k, jS;
   int		    ji, jj, jk, jm, index;
   int		    set_empty = 1;
   int		    C_i_nonempty = 0;
   /*int		    num_nonzeros;*/
   int		    num_procs, my_id;
   int		    num_sends = 0;
   int		    start;
   /*HYPRE_BigInt	    first_col;
   HYPRE_BigInt	    col_0, col_n;*/

   hypre_LinkList   LoL_head;
   hypre_LinkList   LoL_tail;

   int             *lists, *where;
   int              measure, new_meas;
   int              meas_type = 0;
   int              agg_2 = 0;
   int              num_left, elmt;
   int              nabor, nabor_two;

   int              ierr = 0;
   int              use_commpkg_A = 0;
   int              break_var = 0;
   int              f_pnt = F_PT;
   double	    wall_time;

   if (coarsen_type < 0) coarsen_type = -coarsen_type;
   if (measure_type == 1 || measure_type == 4) meas_type = 1;
   if (measure_type == 4 || measure_type == 3) agg_2 = 1;

   /*-------------------------------------------------------
    * Initialize the C/F marker, LoL_head, LoL_tail  arrays
    *-------------------------------------------------------*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = hypre_CTAlloc(int, num_variables);
   where = hypre_CTAlloc(int, num_variables);

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

   /*--------------------------------------------------------------
    * Compute a CSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
        use_commpkg_A = 1;
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      hypre_NewCommPkgCreate(A);
#else
        hypre_MatvecCommPkgCreate(A);
#endif
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (num_cols_offd) S_offd_j = hypre_CSRMatrixJ(S_offd);

   jS = S_i[num_variables];

   ST = hypre_CSRMatrixCreate(num_variables, num_variables, jS);
   ST_i = hypre_CTAlloc(int,num_variables+1);
   ST_j = hypre_CTAlloc(int,jS);
   hypre_CSRMatrixI(ST) = ST_i;
   hypre_CSRMatrixJ(ST) = ST_j;

   /*----------------------------------------------------------
    * generate transpose of S, ST
    *----------------------------------------------------------*/

   for (i=0; i <= num_variables; i++)
      ST_i[i] = 0;
 
   for (i=0; i < jS; i++)
   {
	 ST_i[S_j[i]+1]++;
   }
   for (i=0; i < num_variables; i++)
   {
      ST_i[i+1] += ST_i[i];
   }
   for (i=0; i < num_variables; i++)
   {
      for (j=S_i[i]; j < S_i[i+1]; j++)
      {
	 index = S_j[j];
       	 ST_j[ST_i[index]] = i;
       	 ST_i[index]++;
      }
   }      
   for (i = num_variables; i > 0; i--)
   {
      ST_i[i] = ST_i[i-1];
   }
   ST_i[0] = 0;

   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are given by the row sums of ST.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    * correct actual measures through adding influences from
    * neighbor processors
    *----------------------------------------------------------*/

   measure_array = hypre_CTAlloc(int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] = ST_i[i+1]-ST_i[i];
   }

   /* special case for Falgout coarsening */
   if (coarsen_type == 6) 
   {
      f_pnt = Z_PT;
      coarsen_type = 1;
   }
   if (coarsen_type == 10)
   {
      f_pnt = Z_PT;
      coarsen_type = 11;
   }

   if (meas_type && num_procs > 1)
   {
      int *measure_offd;
      measure_offd = hypre_CTAlloc(int, num_cols_offd);
      int_buf_data = hypre_CTAlloc(int, 
		hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

      for (i=0; i < S_offd_i[num_variables]; i++)
         measure_offd[S_offd_j[i]]++;

      comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg, 
                        measure_offd, int_buf_data);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      
      index = 0;
      for (i=0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += int_buf_data[index++];
      }
      hypre_TFree(measure_offd);
   }

   if ((coarsen_type != 1 && coarsen_type != 11) 
		&& num_procs > 1)
   {
      if (use_commpkg_A)
         S_ext      = hypre_ParCSRMatrixExtractConvBExt(S,A,0);
      else
         S_ext      = hypre_ParCSRMatrixExtractConvBExt(S,S,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
      /*num_nonzeros = S_ext_i[num_cols_offd];
      col_0 = hypre_ParCSRMatrixFirstColDiag(S)-1;
      col_n = col_0 + (HYPRE_BigInt) num_variables;
      if (measure_type)
      {
	 for (i=0; i < num_nonzeros; i++)
         {
	    index = S_ext_j[i] - first_col;
	    if (index > -1 && index < num_variables)
		measure_array[index]++;
         } 
      } */
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   /* first coarsening phase */

  /*************************************************************
   *
   *   Initialize the lists
   *
   *************************************************************/

   CF_marker = hypre_CTAlloc(int, num_variables);
   
   num_left = 0;
   for (j = 0; j < num_variables; j++)
   {
      if ((S_i[j+1]-S_i[j])== 0 &&
		(S_offd_i[j+1]-S_offd_i[j]) == 0)
      {
         CF_marker[j] = SF_PT;
         if (agg_2) CF_marker[j] = SC_PT;
         measure_array[j] = 0;
      }
      else
      {
         CF_marker[j] = UNDECIDED;
         num_left++;
      }
   } 

   for (j = 0; j < num_variables; j++) 
   {    
      measure = measure_array[j];
      if (CF_marker[j] != SF_PT && CF_marker[j] != SC_PT)
      {
         if (measure > 0)
         {
            enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
         }
         else
         {
            if (measure < 0) printf("negative measure!\n");
            CF_marker[j] = f_pnt;
            for (k = S_i[j]; k < S_i[j+1]; k++)
            {
               nabor = S_j[k];
               if (CF_marker[nabor] != SF_PT && CF_marker[nabor] != SC_PT)
               {
                  if (nabor < j)
                  {
                     new_meas = measure_array[nabor];
	             if (new_meas > 0)
                        remove_point(&LoL_head, &LoL_tail, new_meas, 
                               nabor, lists, where);

                     new_meas = ++(measure_array[nabor]);
                     enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                 nabor, lists, where);
                  }
	          else
                  {
                     new_meas = ++(measure_array[nabor]);
                  }
               }
            }
            --num_left;
         }
      }
   }

   /****************************************************************
    *
    *  Main loop of Ruge-Stueben first coloring pass.
    *
    *  WHILE there are still points to classify DO:
    *        1) find first point, i,  on list with max_measure
    *           make i a C-point, remove it from the lists
    *        2) For each point, j,  in S_i^T,
    *           a) Set j to be an F-point
    *           b) For each point, k, in S_j
    *                  move k to the list in LoL with measure one
    *                  greater than it occupies (creating new LoL
    *                  entry if necessary)
    *        3) For each point, j,  in S_i,
    *                  move j to the list in LoL with measure one
    *                  smaller than it occupies (creating new LoL
    *                  entry if necessary)
    *
    ****************************************************************/

   while (num_left > 0)
   {
      index = LoL_head -> head;

      CF_marker[index] = C_PT;
      measure = measure_array[index];
      measure_array[index] = 0;
      --num_left;
      
      remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);
  
      for (j = ST_i[index]; j < ST_i[index+1]; j++)
      {
         nabor = ST_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            CF_marker[nabor] = F_PT;
            measure = measure_array[nabor];

            remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            --num_left;

            for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
            {
               nabor_two = S_j[k];
               if (CF_marker[nabor_two] == UNDECIDED)
               {
                  measure = measure_array[nabor_two];
                  remove_point(&LoL_head, &LoL_tail, measure, 
                               nabor_two, lists, where);

                  new_meas = ++(measure_array[nabor_two]);
                 
                  enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                 nabor_two, lists, where);
               }
            }
         }
      }
      for (j = S_i[index]; j < S_i[index+1]; j++)
      {
         nabor = S_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            measure = measure_array[nabor];

            remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

            measure_array[nabor] = --measure;
	
	    if (measure > 0)
               enter_on_lists(&LoL_head, &LoL_tail, measure, nabor, 
				lists, where);
	    else
	    {
               CF_marker[nabor] = F_PT;
               --num_left;

               for (k = S_i[nabor]; k < S_i[nabor+1]; k++)
               {
                  nabor_two = S_j[k];
                  if (CF_marker[nabor_two] == UNDECIDED)
                  {
                     new_meas = measure_array[nabor_two];
                     remove_point(&LoL_head, &LoL_tail, new_meas, 
                               nabor_two, lists, where);

                     new_meas = ++(measure_array[nabor_two]);
                 
                     enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                 nabor_two, lists, where);
                  }
               }
	    }
         }
      }
   }

   hypre_TFree(measure_array);
   hypre_CSRMatrixDestroy(ST);

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Coarsen 1st pass = %f\n",
                     my_id, wall_time); 
   }

   hypre_TFree(lists);
   hypre_TFree(where);
   hypre_TFree(LoL_head);
   hypre_TFree(LoL_tail);

   for (i=0; i < num_variables; i++)
      if (CF_marker[i] == SC_PT) CF_marker[i] = C_PT;

   if (coarsen_type == 11)
   {
      hypre_TFree(int_buf_data);
      int_buf_data = NULL;
      *CF_marker_ptr = CF_marker;
      return 0;
   }

   /* second pass, check fine points for coarse neighbors 
      for coarsen_type = 2, the second pass includes
      off-processore boundary points */

   /*---------------------------------------------------
    * Initialize the graph array
    *---------------------------------------------------*/

   graph_array = hypre_CTAlloc(int, num_variables);

   for (i = 0; i < num_variables; i++)
   {
      graph_array[i] = -1;
   }

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   if (coarsen_type == 2)
   {
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/
    
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      if (int_buf_data == NULL)
         int_buf_data = hypre_CTAlloc(int, 
		hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
    
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++]
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
    
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
        CF_marker_offd);
    
      hypre_ParCSRCommHandleDestroy(comm_handle);
      }
    
      ci_array = hypre_CTAlloc(int,num_cols_offd);
      for (i=0; i < num_cols_offd; i++)
	 ci_array[i] = -1;
	
      for (i=0; i < num_variables; i++)
      {
	 if (ci_tilde_mark |= i) ci_tilde = -1;
	 if (ci_tilde_offd_mark |= i) ci_tilde_offd = -1;
         if (CF_marker[i] == -1)
         {
            break_var = 1;
            for (ji = S_i[i]; ji < S_i[i+1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] > 0)
                  graph_array[j] = i;
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] > 0)
                  ci_array[j] = i;
            }
            for (ji = S_i[i]; ji < S_i[i+1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] == -1)
               {
                  set_empty = 1;
                  for (jj = S_i[j]; jj < S_i[j+1]; jj++)
                  {
                     index = S_j[jj];
                     if (graph_array[index] == i)
                     {
                        set_empty = 0;
                        break;
                     }
                  }
		  if (set_empty)
                  {
                     for (jj = S_offd_i[j]; jj < S_offd_i[j+1]; jj++)
                     {
                        index = S_offd_j[jj];
                        if (ci_array[index] == i)
                        {
                           set_empty = 0;
                           break;
                        }
                     }
                  } 
                  if (set_empty)
                  {
                     if (C_i_nonempty)
                     {
                        CF_marker[i] = 1;
                        if (ci_tilde > -1)
                        {
                           CF_marker[ci_tilde] = -1;
                           ci_tilde = -1;
                        }
                        if (ci_tilde_offd > -1)
                        {
                           CF_marker_offd[ci_tilde_offd] = -1;
                           ci_tilde_offd = -1;
                        }
                        C_i_nonempty = 0;
                        break_var = 0;
                        break;
                     }
                     else
                     {
                        ci_tilde = j;
                        ci_tilde_mark = i;
                        CF_marker[j] = 1;
                        C_i_nonempty = 1;
                        i--;
                        break_var = 0;
                        break;
                     }
                  }
               }
            }
            if (break_var)
            {
               for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji++)
               {
                  j = S_offd_j[ji];
                  if (CF_marker_offd[j] == -1)
                  {
                     set_empty = 1;
                     for (jj = S_ext_i[j]; jj < S_ext_i[j+1]; jj++)
                     {
                        index = S_ext_j[jj];
                        if (index >= 0) /* index interior */
                        {
                           if (graph_array[index] == i)
                           {
                              set_empty = 0;
                              break;
                           }
                        }
                        else
                        {
   		           jk = -index-1;
                           if (ci_array[jk] == i)
                           {
                                 set_empty = 0;
                                 break;
                           }
                        }
                     }
                     if (set_empty)
                     {
                        if (C_i_nonempty)
                        {
                           CF_marker[i] = 1;
                           if (ci_tilde > -1)
                           {
                              CF_marker[ci_tilde] = -1;
                              ci_tilde = -1;
                           }
                           if (ci_tilde_offd > -1)
                           {
                              CF_marker_offd[ci_tilde_offd] = -1;
                              ci_tilde_offd = -1;
                           }
                           C_i_nonempty = 0;
                           break;
                        }
                        else
                        {
                           ci_tilde_offd = j;
                           ci_tilde_offd_mark = i;
                           CF_marker_offd[j] = 1;
                           C_i_nonempty = 1;
                           i--;
                           break;
                        }
                     }
                  }
               }
            }
         }
      }
   }
   else
   {
      for (i=0; i < num_variables; i++)
      {
	 if (ci_tilde_mark |= i) ci_tilde = -1;
         if (CF_marker[i] == -1)
         {
   	    for (ji = S_i[i]; ji < S_i[i+1]; ji++)
   	    {
   	       j = S_j[ji];
   	       if (CF_marker[j] > 0)
   	          graph_array[j] = i;
    	    }
   	    for (ji = S_i[i]; ji < S_i[i+1]; ji++)
   	    {
   	       j = S_j[ji];
   	       if (CF_marker[j] == -1)
   	       {
   	          set_empty = 1;
   	          for (jj = S_i[j]; jj < S_i[j+1]; jj++)
   	          {
   		     index = S_j[jj];
   		     if (graph_array[index] == i)
   		     {
   		        set_empty = 0;
   		        break;
   		     }
   	          }
   	          if (set_empty)
   	          {
   		     if (C_i_nonempty)
   		     {
   		        CF_marker[i] = 1;
   		        if (ci_tilde > -1)
   		        {
   			   CF_marker[ci_tilde] = -1;
   		           ci_tilde = -1;
   		        }
   	    		C_i_nonempty = 0;
   		        break;
   		     }
   		     else
   		     {
   		        ci_tilde = j;
   		        ci_tilde_mark = i;
   		        CF_marker[j] = 1;
   		        C_i_nonempty = 1;
		        i--;
		        break;
		     }
	          }
	       }
	    }
	 }
      }
   }

   if (debug_flag == 3 && coarsen_type != 2)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Coarsen 2nd pass = %f\n",
                       my_id, wall_time); 
   }

   /* third pass, check boundary fine points for coarse neighbors */

   if (coarsen_type == 3 || coarsen_type == 4)
   {
      if (debug_flag == 3) wall_time = time_getWallclockSeconds();

      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      if (int_buf_data == NULL)
         int_buf_data = hypre_CTAlloc(int, 
		hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
             int_buf_data[index++] 
              = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
 
      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, 
		int_buf_data, CF_marker_offd);
 
         hypre_ParCSRCommHandleDestroy(comm_handle);   
      }

      ci_array = hypre_CTAlloc(int,num_cols_offd);
      for (i=0; i < num_cols_offd; i++)
	 ci_array[i] = -1;
   }

   if (coarsen_type > 1 && coarsen_type < 5)
   { 
      for (i=0; i < num_variables; i++)
	 graph_array[i] = -1;
      for (i=0; i < num_cols_offd; i++)
      {
	 if (ci_tilde_mark |= i) ci_tilde = -1;
	 if (ci_tilde_offd_mark |= i) ci_tilde_offd = -1;
         if (CF_marker_offd[i] == -1)
         {
   	    for (ji = S_ext_i[i]; ji < S_ext_i[i+1]; ji++)
   	    {
   	       j = S_ext_j[ji];
   	       if (j >= 0)
   	       {
   	          if (CF_marker[j] > 0)
   	             graph_array[j] = i;
   	       }
   	       else
   	       {
   		  jj = -j-1;
   		  if (CF_marker_offd[jj] > 0)
   	                ci_array[jj] = i;
    	       }	
    	    }
   	    for (ji = S_ext_i[i]; ji < S_ext_i[i+1]; ji++)
   	    {
   	       j = S_ext_j[ji];
   	       if (j >= 0)
   	       {
   	          if ( CF_marker[j] == -1)
   	          {
   	             set_empty = 1;
   	             for (jj = S_i[j]; jj < S_i[j+1]; jj++)
   	             {
   		        index = S_j[jj];
   		        if (graph_array[index] == i)
   		        {
   		           set_empty = 0;
   		           break;
   		        }
   	             }
   	             for (jj = S_offd_i[j]; jj < S_offd_i[j+1]; jj++)
   	             {
   		        index = S_offd_j[jj];
   		        if (ci_array[index] == i)
   		        {
   		           set_empty = 0;
   		           break;
   		        }
   	             }
   	             if (set_empty)
   	             {
   		        if (C_i_nonempty)
   		        {
   		           CF_marker_offd[i] = 1;
   		           if (ci_tilde > -1)
   		           {
   			      CF_marker[ci_tilde] = -1;
			      ci_tilde = -1;
   		           }
   		           if (ci_tilde_offd > -1)
   		           {
   			      CF_marker_offd[ci_tilde_offd] = -1;
			      ci_tilde_offd = -1;
   		           }
                           C_i_nonempty = 0;
   		           break;
   		        }
   		        else
   		        {
   		           ci_tilde = j;
   		           ci_tilde_mark = i;
   		           CF_marker[j] = 1;
   		           C_i_nonempty = 1;
   		           i--;
   		           break;
   		        }
   	             }
   	          }
   	       }
   	       else
   	       {
   		  jm = -j-1;
   		  if (CF_marker_offd[jm] == -1)
   	          {
   	             set_empty = 1;
   	             for (jj = S_ext_i[jm]; jj < S_ext_i[jm+1]; jj++)
   	             {
   		        index = S_ext_j[jj];
   		        if (index >= 0) 
   		  	{
   		           if (graph_array[index] == i)
   		           {
   		              set_empty = 0;
   		              break;
   		           }
   	                }
   			else
   			{
   		           jk = -index-1;
   		           if (ci_array[jk] == i)
   		           {
   		              set_empty = 0;
   		              break;
   		           }
   	                }
   	             }
   	             if (set_empty)
   	             {
   		        if (C_i_nonempty)
   		        {
   		           CF_marker_offd[i] = 1;
   		           if (ci_tilde > -1)
   		           {
   			      CF_marker[ci_tilde] = -1;
   			      ci_tilde = -1;
   		           }
   		           if (ci_tilde_offd > -1)
   		           {
   			      CF_marker_offd[ci_tilde_offd] = -1;
   			      ci_tilde_offd = -1;
   		           }
                           C_i_nonempty = 0;
   		           break;
   		        }
   		        else
   		        {
   		           ci_tilde_offd = jm;
   		           ci_tilde_offd_mark = i;
   		           CF_marker_offd[jm] = 1;
   		           C_i_nonempty = 1;
   		           i--;
   		           break;
   		        }
   		     }
   	          }
   	       }
   	    }
         }
      }
      /*------------------------------------------------
       * Send boundary data for CF_marker back
       *------------------------------------------------*/
      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg, 
		CF_marker_offd, int_buf_data);
    
         hypre_ParCSRCommHandleDestroy(comm_handle);   
      }
   
      /* only CF_marker entries from larger procs are accepted  
	if coarsen_type = 4 coarse points are not overwritten  */
 
      index = 0;
      if (coarsen_type != 4)
      {
         for (i = 0; i < num_sends; i++)
         {
	    start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            if (hypre_ParCSRCommPkgSendProc(comm_pkg,i) > my_id)
	    {
              for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                   CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)] =
                   int_buf_data[index++]; 
            }
	    else
	    {
	       index += hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - start;
	    }
         }
      }
      else
      {
         for (i = 0; i < num_sends; i++)
         {
	    start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            if (hypre_ParCSRCommPkgSendProc(comm_pkg,i) > my_id)
	    {
              for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
              {
                 elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
                 if (CF_marker[elmt] != 1)
                   CF_marker[elmt] = int_buf_data[index];
		 index++; 
              }
            }
	    else
	    {
	       index += hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1) - start;
	    }
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         if (coarsen_type == 4)
		printf("Proc = %d    Coarsen 3rd pass = %f\n",
                my_id, wall_time); 
         if (coarsen_type == 3)
		printf("Proc = %d    Coarsen 3rd pass = %f\n",
                my_id, wall_time); 
         if (coarsen_type == 2)
		printf("Proc = %d    Coarsen 2nd pass = %f\n",
                my_id, wall_time); 
      }
   }
   if (coarsen_type == 5)
   {
      /*------------------------------------------------
       * Exchange boundary data for CF_marker
       *------------------------------------------------*/

      if (debug_flag == 3) wall_time = time_getWallclockSeconds();
    
      CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      if (int_buf_data == NULL)
	 int_buf_data = hypre_CTAlloc(int, 
		hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));
    
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++]
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
    
      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, 
		int_buf_data, CF_marker_offd);
    
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }
    
      ci_array = hypre_CTAlloc(int,num_cols_offd);
      for (i=0; i < num_cols_offd; i++)
   	 ci_array[i] = -1;
      for (i=0; i < num_variables; i++)
   	 graph_array[i] = -1;

      for (i=0; i < num_variables; i++)
      {
         if (CF_marker[i] == -1 && (S_offd_i[i+1]-S_offd_i[i]) > 0)
         {
            break_var = 1;
            for (ji = S_i[i]; ji < S_i[i+1]; ji++)
            {
               j = S_j[ji];
               if (CF_marker[j] > 0)
                  graph_array[j] = i;
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] > 0)
                  ci_array[j] = i;
            }
            for (ji = S_offd_i[i]; ji < S_offd_i[i+1]; ji++)
            {
               j = S_offd_j[ji];
               if (CF_marker_offd[j] == -1)
               {
                  set_empty = 1;
                  for (jj = S_ext_i[j]; jj < S_ext_i[j+1]; jj++)
                  {
                     index = S_ext_j[jj];
                     if (index >= 0) /* index interior */
                     {
                        if (graph_array[index] == i)
                        {
                           set_empty = 0;
                           break;
                        }
                     }
                     else
                     {
   		        jk = -index-1;
                        if (ci_array[jk] == i)
                        {
                           set_empty = 0;
                           break;
                        }
                     }
                  }
                  if (set_empty)
                  {
                     if (C_i_nonempty)
                     {
                        CF_marker[i] = -2;
                        C_i_nonempty = 0;
                        break;
                     }
                     else
                     {
                        C_i_nonempty = 1;
                        i--;
                        break;
                     }
                  }
               }
            }
         }
      }
      if (debug_flag == 3)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    Coarsen special points = %f\n",
                       my_id, wall_time); 
      }

   }
   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(int_buf_data);
   if (coarsen_type != 1)
   {   
      hypre_TFree(CF_marker_offd);
      hypre_TFree(ci_array);
   }   
   hypre_TFree(graph_array);
   if ((meas_type || coarsen_type != 1) && num_procs > 1)
   	hypre_CSRMatrixDestroy(S_ext); 
   
   *CF_marker_ptr   = CF_marker;
   
   return (ierr);
}


int
hypre_BoomerAMGCoarsenFalgout( hypre_ParCSRMatrix    *S,
                               hypre_ParCSRMatrix    *A,
                               int                    measure_type,
                               int                    debug_flag,
                               int                  **CF_marker_ptr)
{
   int              ierr = 0;

   /*-------------------------------------------------------
    * Perform Ruge coarsening followed by CLJP coarsening
    *-------------------------------------------------------*/

   ierr += hypre_BoomerAMGCoarsenRuge (S, A, measure_type, 6, debug_flag, 
				CF_marker_ptr);

   ierr += hypre_BoomerAMGCoarsen (S, A, 1, debug_flag, 
				CF_marker_ptr);

   return (ierr);
}

int
hypre_BoomerAMGCoarsenHMIS( hypre_ParCSRMatrix    *S,
                            hypre_ParCSRMatrix    *A,
                            int                    measure_type,
                            int                    debug_flag,
                            int                  **CF_marker_ptr)
{
   int              ierr = 0;

   /*-------------------------------------------------------
    * Perform Ruge coarsening followed by CLJP coarsening
    *-------------------------------------------------------*/

   ierr += hypre_BoomerAMGCoarsenRuge (S, A, measure_type, 10, debug_flag,
                                CF_marker_ptr);

   ierr += hypre_BoomerAMGCoarsenPMIS (S, A, 1, debug_flag,
                                CF_marker_ptr);

   return (ierr);
}

/*--------------------------------------------------------------------------*/

#define C_PT  1
#define F_PT -1
#define SF_PT -3
#define COMMON_C_PT  2
#define Z_PT -2

      /* begin HANS added */
/**************************************************************
 *
 *      Modified Independent Set Coarsening routine
 *          (don't worry about strong F-F connections
 *           without a common C point)
 *
 **************************************************************/
int
hypre_BoomerAMGCoarsenPMIS( hypre_ParCSRMatrix    *S,
			    hypre_ParCSRMatrix    *A,
                        int                    CF_init,
                        int                    debug_flag,
                        int                  **CF_marker_ptr)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j;

   int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   int 		       num_cols_offd = 0;
                  
   /* hypre_CSRMatrix    *S_ext;
   int                *S_ext_i;
   int                *S_ext_j; */

   int		       num_sends = 0;
   int  	      *int_buf_data;
   double	      *buf_data;

   int                *CF_marker;
   int                *CF_marker_offd;
                      
   double             *measure_array;
   int                *graph_array;
   int                *graph_array_offd;
   int                 graph_size;
   HYPRE_BigInt        big_graph_size;
   int                 graph_offd_size;
   HYPRE_BigInt        global_graph_size;
                      
   int                 i, j, jS, ig;
   int		       index, start, my_id, num_procs, jrow, cnt, elmt;
                      
   int                 ierr = 0;

   double	    wall_time;
   int   iter = 0;



#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

   /*******************************************************************************
    BEFORE THE INDEPENDENT SET COARSENING LOOP:
      measure_array: calculate the measures, and communicate them
        (this array contains measures for both local and external nodes)
      CF_marker, CF_marker_offd: initialize CF_marker
        (separate arrays for local and external; 0=unassigned, negative=F point, positive=C point)
   ******************************************************************************/      

   /*--------------------------------------------------------------
    * Use the ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: S_data is not used; in stead, only strong columns are retained
    *       in S_j, which can then be used like S_data
    *----------------------------------------------------------------*/

   /*S_ext = NULL; */
   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      hypre_NewCommPkgCreate(A);
#else
        hypre_MatvecCommPkgCreate(A);
#endif
        comm_pkg = hypre_ParCSRMatrixCommPkg(A); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = hypre_CSRMatrixJ(S_offd);
   }

   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are currently given by the column sums of S.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    *
    * The measures are augmented by a random number
    * between 0 and 1.
    *----------------------------------------------------------*/

   measure_array = hypre_CTAlloc(double, num_variables+num_cols_offd);

   /* first calculate the local part of the sums for the external nodes */
   for (i=0; i < S_offd_i[num_variables]; i++)
   { 
      measure_array[num_variables + S_offd_j[i]] += 1.0;
   }

   /* now send those locally calculated values for the external nodes to the neighboring processors */
   if (num_procs > 1)
   comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg, 
                        &measure_array[num_variables], buf_data);

   /* calculate the local part for the local nodes */
   for (i=0; i < S_diag_i[num_variables]; i++)
   { 
      measure_array[S_diag_j[i]] += 1.0;
   }

   /* finish the communication */
   if (num_procs > 1)
   hypre_ParCSRCommHandleDestroy(comm_handle);
      
   /* now add the externally calculated part of the local nodes to the local nodes */
   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += buf_data[index++];
   }

   /* set the measures of the external nodes to zero */
   for (i=num_variables; i < num_variables+num_cols_offd; i++)
   { 
      measure_array[i] = 0;
   }

   /* this augments the measures with a random number between 0 and 1 */
   /* (only for the local part) */
   /* this augments the measures */
   if (CF_init == 2 || CF_init == 4)
      hypre_BoomerAMGIndepSetInit(S, measure_array, 1);
   else
      hypre_BoomerAMGIndepSetInit(S, measure_array, 0);

   /*---------------------------------------------------
    * Initialize the graph arrays, and CF_marker arrays
    *---------------------------------------------------*/

   /* first the off-diagonal part of the graph array */
   if (num_cols_offd) 
      graph_array_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      graph_array_offd = NULL;

   for (ig = 0; ig < num_cols_offd; ig++)
      graph_array_offd[ig] = ig;

   graph_offd_size = num_cols_offd;

   /* now the local part of the graph array, and the local CF_marker array */
   graph_array = hypre_CTAlloc(int, num_variables);

   if (CF_init==1)
   { 
      CF_marker = *CF_marker_ptr;
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
         if ( (S_offd_i[i+1]-S_offd_i[i]) > 0 || CF_marker[i] == -1)
	 {
	   CF_marker[i] = 0;
	 }
         if ( CF_marker[i] == Z_PT)
         {
            if (measure_array[i] >= 1.0 ||
                (S_diag_i[i+1]-S_diag_i[i]) > 0)
            {
               CF_marker[i] = 0;
               graph_array[cnt++] = i;
            }
            else
            {
               CF_marker[i] = F_PT;
            }
         }
         else if (CF_marker[i] == SF_PT)
            measure_array[i] = 0;
         else
            graph_array[cnt++] = i;
      }
   }
   else
   {
      CF_marker = hypre_CTAlloc(int, num_variables);
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
         CF_marker[i] = 0;
         if ( (S_diag_i[i+1]-S_diag_i[i]) == 0
                && (S_offd_i[i+1]-S_offd_i[i]) == 0)
         {
            CF_marker[i] = SF_PT;
            if (CF_init == 3 || CF_init == 4) CF_marker[i] = C_PT;
            measure_array[i] = 0;
         }
         else
            graph_array[cnt++] = i;
      }
   }
   graph_size = cnt;

   /* now the off-diagonal part of CF_marker */
   if (num_cols_offd)
     CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
   else
     CF_marker_offd = NULL;

   for (i=0; i < num_cols_offd; i++)
	CF_marker_offd[i] = 0;
  
   /*------------------------------------------------
    * Communicate the local measures, which are complete,
      to the external nodes
    *------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++)
     {
       start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
       for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
	 {
	   jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
	   buf_data[index++] = measure_array[jrow];
         }
     }
   
   if (num_procs > 1)
     { 
       comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data, 
						  &measure_array[num_variables]);
       
       hypre_ParCSRCommHandleDestroy(comm_handle);   
       
     } 
      
   /* we need S_ext: the columns of the S matrix for the local nodes */
   /* we need this because the independent set routine can only decide
      which local nodes are in it when it knows both the rows and columns
      of S */

   /* if (num_procs > 1)
   {
      S_ext      = hypre_ParCSRMatrixExtractBExt(S,A,0);
      S_ext_i    = hypre_CSRMatrixI(S_ext);
      S_ext_j    = hypre_CSRMatrixJ(S_ext);
   } */

   /*  compress S_ext and convert column numbers*/

   /* index = 0;
   for (i=0; i < num_cols_offd; i++)
   {
      for (j=S_ext_i[i]; j < S_ext_i[i+1]; j++)
      {
	 k = S_ext_j[j];
	 if (k >= col_1 && k < col_n)
	 {
	    S_ext_j[index++] = k - col_1;
	 }
	 else
	 {
	    kc = hypre_BinarySearch(col_map_offd,k,num_cols_offd);
	    if (kc > -1) S_ext_j[index++] = -kc-1;
	 }
      }
      S_ext_i[i] = index;
   }
   for (i = num_cols_offd; i > 0; i--)
      S_ext_i[i] = S_ext_i[i-1];
   if (num_procs > 1) S_ext_i[0] = 0; */
 
   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Initialize CLJP phase = %f\n",
                     my_id, wall_time); 
   }

   /*******************************************************************************
    THE INDEPENDENT SET COARSENING LOOP:
   ******************************************************************************/      

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   while (1)
   {

     big_graph_size = (HYPRE_BigInt) graph_size;
     /* stop the coarsening if nothing left to be coarsened */
     MPI_Allreduce(&big_graph_size,&global_graph_size,1,MPI_HYPRE_BIG_INT,MPI_SUM,comm);

     if (global_graph_size == 0)
       break;

     /*     printf("\n");
     printf("*** MIS iteration %d\n",iter);
     printf("graph_size remaining %d\n",graph_size);*/

     /*------------------------------------------------
      * Pick an independent set of points with
      * maximal measure.
        At the end, CF_marker is complete, but still needs to be
        communicated to CF_marker_offd
      *------------------------------------------------*/
      if (!CF_init || iter)
      {
          hypre_BoomerAMGIndepSet(S, measure_array, graph_array, 
				graph_size, 
				graph_array_offd, graph_offd_size, 
				CF_marker, CF_marker_offd);

      /*------------------------------------------------
       * Exchange boundary data for CF_marker: send internal
         points to external points
       *------------------------------------------------*/

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg, 
		CF_marker_offd, int_buf_data);
 
         hypre_ParCSRCommHandleDestroy(comm_handle);   
      }

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            if (!int_buf_data[index] && CF_marker[elmt] > 0)
            {
               CF_marker[elmt] = 0; 
               index++;
            }
            else
            {
               int_buf_data[index++] = CF_marker[elmt];
            }
         }
      }
 
      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, 
        	CF_marker_offd);
 
         hypre_ParCSRCommHandleDestroy(comm_handle);   
      }
      }

      iter++;
     /*------------------------------------------------
      * Set C-pts and F-pts.
      *------------------------------------------------*/

     for (ig = 0; ig < graph_size; ig++) 
     {
       i = graph_array[ig];

       /*---------------------------------------------
	* If the measure of i is smaller than 1, then
        * make i and F point (because it does not influence
        * any other point), and remove all edges of
	* equation i.
	*---------------------------------------------*/

       if(measure_array[i]<1.)
       {
	 /* make point i an F point*/
	 CF_marker[i]= F_PT;

         /* remove the edges in equation i */
	 /* first the local part */
	 /*for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) {
	   j = S_diag_j[jS];
	   if (j > -1){ 
	     S_diag_j[jS]  = -S_diag_j[jS]-1;
	   }
	 }*/
	 /* now the external part */
	 /*for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) {
	   j = S_offd_j[jS];
	   if (j > -1){ 
	     S_offd_j[jS]  = -S_offd_j[jS]-1;
	   }
	 }*/
       }

       /*---------------------------------------------
	* First treat the case where point i is in the
	* independent set: make i a C point, 
        * take out all the graph edges for
        * equation i.
	*---------------------------------------------*/
       
       if (CF_marker[i] > 0) 
       {
	 /* set to be a C-pt */
	 CF_marker[i] = C_PT;

         /* remove the edges in equation i */
	 /* first the local part */
	 /*for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) {
	   j = S_diag_j[jS];
	   if (j > -1){ 
	     S_diag_j[jS]  = -S_diag_j[jS]-1;
	   }
	 }*/
	 /* now the external part */
	 /*for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) {
	   j = S_offd_j[jS];
	   if (j > -1){ 
	     S_offd_j[jS]  = -S_offd_j[jS]-1;
	   }
	 }*/
       }  

       /*---------------------------------------------
	* Now treat the case where point i is not in the
	* independent set: loop over
	* all the points j that influence equation i; if
	* j is a C point, then make i an F point.
	* If i is a new F point, then remove all the edges
        * from the graph for equation i.
	*---------------------------------------------*/

       else 
       {

	 /* first the local part */
	 for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) 
         {
	   /* j is the column number, or the local number of the point influencing i */
	   j = S_diag_j[jS];
           /*if(j<0) j=-j-1;*/

	   if (CF_marker[j] > 0)
           { /* j is a C-point */
	     CF_marker[i] = F_PT;
	   }
	 }
	 /* now the external part */
	 for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) 
         {
	   j = S_offd_j[jS];
           /*if(j<0) j=-j-1;*/
	   if (CF_marker_offd[j] > 0)
           { /* j is a C-point */
	     CF_marker[i] = F_PT;
	   }
	 }

         /* remove all the edges for equation i if i is a new F point */
	 /*if (CF_marker[i] == F_PT){
	   for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) {
	     j = S_diag_j[jS];
	     if (j > -1){ 
	       S_diag_j[jS]  = -S_diag_j[jS]-1;
	     }
	   }
	   for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) {
	     j = S_offd_j[jS];
	     if (j > -1){ 
	       S_offd_j[jS]  = -S_offd_j[jS]-1;
	     }
	   }
	 } */  
       } /* end else */
     } /* end first loop over graph */

     /* now communicate CF_marker to CF_marker_offd, to make
        sure that new external F points are known on this processor */

      /*------------------------------------------------
       * Exchange boundary data for CF_marker: send internal
         points to external points
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++] 
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
 
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, 
        CF_marker_offd);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);   
      }

     /*---------------------------------------------
      * Now loop over the points i in the unassigned
      * graph again. For all points i that are no new C or
      * F points, remove the edges in equation i that
      * connect to C or F points.
      * (We have removed the rows for the new C and F
      * points above; now remove the columns.)
      *---------------------------------------------*/

     /*for (ig = 0; ig < graph_size; ig++) {
       i = graph_array[ig];

       if(CF_marker[i]==0) {*/

	 /* first the local part */
	 /*for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++) {
	   j = S_diag_j[jS];
           if(j<0) j=-j-1;

	   if (!CF_marker[j]==0 && S_diag_j[jS] > -1){ */ /* connection to C or F point, and
                                                 column number is still positive; not accounted for yet */
	     /*S_diag_j[jS]  = -S_diag_j[jS]-1;
	   }
	 }*/
	 /* now the external part */
	 /*for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) {
	   j = S_offd_j[jS];
           if(j<0) j=-j-1;

	   if (!CF_marker_offd[j]==0 && S_offd_j[jS] > -1){*/ /* connection to C or F point, and
                                                 column number is still positive; not accounted for yet */
	     /*S_offd_j[jS]  = -S_offd_j[jS]-1;
	   }
	 }
       }
     } *//* end second loop over graph */

     /*------------------------------------------------
      * Update subgraph
      *------------------------------------------------*/

     for (ig = 0; ig < graph_size; ig++) 
     {
       i = graph_array[ig];
       
       if (!CF_marker[i]==0) /* C or F point */
	 {
	   /* the independent set subroutine needs measure 0 for
              removed nodes */
	   measure_array[i] = 0;
	   /* take point out of the subgraph */
	   graph_size--;
	   graph_array[ig] = graph_array[graph_size];
	   graph_array[graph_size] = i;
	   ig--;
	 }
     }
     for (ig = 0; ig < graph_offd_size; ig++) 
     {
       i = graph_array_offd[ig];
       
       if (!CF_marker_offd[i]==0) /* C or F point */
	 {
	   /* the independent set subroutine needs measure 0 for
              removed nodes */
	   measure_array[i+num_variables] = 0;
	   /* take point out of the subgraph */
	   graph_offd_size--;
	   graph_array_offd[ig] = graph_array_offd[graph_offd_size];
	   graph_array_offd[graph_offd_size] = i;
	   ig--;
	 }
     }
     
   } /* end while */

   /*   printf("*** MIS iteration %d\n",iter);
   printf("graph_size remaining %d\n",graph_size);

   printf("num_cols_offd %d\n",num_cols_offd);
   for (i=0;i<num_variables;i++)
     {
              if(CF_marker[i]==1)
       printf("node %d CF %d\n",i,CF_marker[i]);
       }*/


   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   /* Reset S_matrix */
   /*for (i=0; i < S_diag_i[num_variables]; i++)
   {
      if (S_diag_j[i] < 0)
         S_diag_j[i] = -S_diag_j[i]-1;
   }
   for (i=0; i < S_offd_i[num_variables]; i++)
   {
      if (S_offd_j[i] < 0)
         S_offd_j[i] = -S_offd_j[i]-1;
   }*/
   /*for (i=0; i < num_variables; i++)
      if (CF_marker[i] == SF_PT) CF_marker[i] = F_PT;*/

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   if (num_cols_offd) hypre_TFree(graph_array_offd);
   hypre_TFree(buf_data);
   hypre_TFree(int_buf_data);
   hypre_TFree(CF_marker_offd);
   /*if (num_procs > 1) hypre_CSRMatrixDestroy(S_ext);*/

   *CF_marker_ptr   = CF_marker;

   return (ierr);
}
