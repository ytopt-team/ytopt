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
#include "par_amg.h"

/*****************************************************************************
 *
 * Routine for getting matrix statistics from setup
 *
 *****************************************************************************/


int
hypre_BoomerAMGSetupStats( void               *amg_vdata,
                        hypre_ParCSRMatrix *A         )
{
   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A);   

   hypre_ParAMGData *amg_data = amg_vdata;

   /*hypre_SeqAMGData *seq_data = hypre_ParAMGDataSeqData(amg_data);*/

   /* Data Structure variables */

   hypre_ParCSRMatrix **A_array;
   hypre_ParCSRMatrix **P_array;

   hypre_CSRMatrix *A_diag;
   double          *A_diag_data;
   int             *A_diag_i;

   hypre_CSRMatrix *A_offd;   
   double          *A_offd_data;
   int             *A_offd_i;

   hypre_CSRMatrix *P_diag;
   double          *P_diag_data;
   int             *P_diag_i;

   hypre_CSRMatrix *P_offd;   
   double          *P_offd_data;
   int             *P_offd_i;


   int	    numrows;

   HYPRE_BigInt	    *row_starts;

 
   int      num_levels; 
   int      coarsen_type;
   int      interp_type;
   int      measure_type;
   double   global_nonzeros;

   double  *send_buff;
   double  *gather_buff;
 
   /* Local variables */

   int       level;
   int       j;
   HYPRE_BigInt fine_size;
 
   int       min_entries;
   int       max_entries;

   int       num_procs,my_id, num_threads;


   double    min_rowsum;
   double    max_rowsum;
   double    sparse;


   int       i;
   

   HYPRE_BigInt coarse_size;
   int       entries;

   double    avg_entries;
   double    rowsum;

   double    min_weight;
   double    max_weight;

   int       global_min_e;
   int       global_max_e;
   double    global_min_rsum;
   double    global_max_rsum;
   double    global_min_wt;
   double    global_max_wt;

   double  *num_coeffs;
   double  *num_variables;
   double   total_variables; 
   double   operat_cmplxty;
   double   grid_cmplxty;

   /* amg solve params */
   int      max_iter;
   int      cycle_type;    
   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int      relax_order;
   int    **grid_relax_points; 
   double  *relax_weight;
   double  *omega;
   double   tol;


   int one = 1;
   int minus_one = -1;
   int zero = 0;
   int smooth_type;
   int smooth_num_levels;
   int agg_num_levels;
   /*int seq_cg = 0;*/
   
   /*if (seq_data)
      seq_cg = 1;*/


   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();

   if (my_id == 0)
      printf("\nNumber of MPI processes: %d , Number of OpenMP threads: %d\n", num_procs, num_threads);
   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);
   interp_type = hypre_ParAMGDataInterpType(amg_data);
   measure_type = hypre_ParAMGDataMeasureType(amg_data);
   smooth_type = hypre_ParAMGDataSmoothType(amg_data);
   smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   agg_num_levels = hypre_ParAMGDataAggNumLevels(amg_data);


   /*----------------------------------------------------------
    * Get the amg_data data
    *----------------------------------------------------------*/

   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   max_iter   = hypre_ParAMGDataMaxIter(amg_data);
   cycle_type = hypre_ParAMGDataCycleType(amg_data);    
   num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);  
   grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_weight = hypre_ParAMGDataRelaxWeight(amg_data); 
   relax_order = hypre_ParAMGDataRelaxOrder(amg_data); 
   omega = hypre_ParAMGDataOmega(amg_data); 
   tol = hypre_ParAMGDataTol(amg_data);

   /*block_mode = hypre_ParAMGDataBlockMode(amg_data);*/

   send_buff     = hypre_CTAlloc(double, 6);
#ifdef HYPRE_NO_GLOBAL_PARTITION
   gather_buff = hypre_CTAlloc(double,6);    
#else
   gather_buff = hypre_CTAlloc(double,6*num_procs);    
#endif

   if (my_id==0)
   {
      printf("\nBoomerAMG SETUP PARAMETERS:\n\n");
      printf(" Max levels = %d\n",hypre_ParAMGDataMaxLevels(amg_data));
      printf(" Num levels = %d\n\n",num_levels);
      printf(" Strength Threshold = %f\n", 
                         hypre_ParAMGDataStrongThreshold(amg_data));
      printf(" Interpolation Truncation Factor = %f\n", 
                         hypre_ParAMGDataTruncFactor(amg_data));
      printf(" Maximum Row Sum Threshold for Dependency Weakening = %f\n\n", 
                         hypre_ParAMGDataMaxRowSum(amg_data));

      if (coarsen_type == 0)
      {
	printf(" Coarsening Type = Cleary-Luby-Jones-Plassman\n");
      }
      else if (abs(coarsen_type) == 1) 
      {
	printf(" Coarsening Type = Ruge\n");
      }
      else if (abs(coarsen_type) == 2) 
      {
	printf(" Coarsening Type = Ruge2B\n");
      }
      else if (abs(coarsen_type) == 3) 
      {
	printf(" Coarsening Type = Ruge3\n");
      }
      else if (abs(coarsen_type) == 4) 
      {
	printf(" Coarsening Type = Ruge 3c \n");
      }
      else if (abs(coarsen_type) == 5) 
      {
	printf(" Coarsening Type = Ruge relax special points \n");
      }
      else if (abs(coarsen_type) == 6) 
      {
	printf(" Coarsening Type = Falgout-CLJP \n");
      }
      else if (abs(coarsen_type) == 8) 
      {
	printf(" Coarsening Type = PMIS \n");
      }
      else if (abs(coarsen_type) == 10) 
      {
	printf(" Coarsening Type = HMIS \n");
      }
      else if (abs(coarsen_type) == 11) 
      {
	printf(" Coarsening Type = Ruge 1st pass only \n");
      }
      else if (abs(coarsen_type) == 9) 
      {
	printf(" Coarsening Type = PMIS fixed random \n");
      }
      else if (abs(coarsen_type) == 7) 
      {
	printf(" Coarsening Type = CLJP, fixed random \n");
      }
      if (coarsen_type > 0) 
      {
	printf(" Hybrid Coarsening (switch to CLJP when coarsening slows)\n");
      }
      

      if (coarsen_type)
      	printf(" measures are determined %s\n\n", 
                  (measure_type ? "globally" : "locally"));

      if (agg_num_levels)
	printf(" no. of levels of aggressive coarsening: %d\n\n", agg_num_levels);

#ifdef HYPRE_NO_GLOBAL_PARTITION
      printf( "\n No global partition option chosen.\n\n");
#endif

      if (interp_type == 0)
      {
	printf(" Interpolation = modified classical interpolation\n");
      }
      else if (interp_type == 1) 
      {
	printf(" Interpolation = LS interpolation \n");
      }
      else if (interp_type == 2) 
      {
	printf(" Interpolation = modified classical interpolation for hyperbolic PDEs\n");
      }
      else if (interp_type == 3) 
      {
	printf(" Interpolation = direct interpolation with separation of weights\n");
      }
      else if (interp_type == 4) 
      {
	printf(" Interpolation = multipass interpolation\n");
      }
      else if (interp_type == 5) 
      {
	printf(" Interpolation = multipass interpolation with separation of weights\n");
      }
      else if (interp_type == 6) 
      {
	printf(" Interpolation = extended+i interpolation\n");
      }
      else if (interp_type == 7) 
      {
	printf(" Interpolation = extended+i interpolation (only when needed)\n");
      }
      else if (interp_type == 8) 
      {
	printf(" Interpolation = standard interpolation\n");
      }
      else if (interp_type == 9) 
      {
	printf(" Interpolation = standard interpolation with separation of weights\n");
      }
      else if (interp_type == 12) 
      {
	printf(" FF interpolation \n");
      }
      else if (interp_type == 13) 
      {
	printf(" FF1 interpolation \n");
      }

      {
         printf( "\nOperator Matrix Information:\n\n");
      }
#if HYPRE_LONG_LONG
      printf("                  nonzero         entries p");
      printf("er row        row sums\n");
      printf("lev        rows   entries  sparse  min  max   ");
      printf("avg       min         max\n");
      printf("=======================================");
      printf("==================================\n");
#else      
      printf("            nonzero         entries p");
      printf("er row        row sums\n");
      printf("lev   rows  entries  sparse  min  max   ");
      printf("avg       min         max\n");
      printf("=======================================");
      printf("============================\n");
#endif
   }
  
   /*-----------------------------------------------------
    *  Enter Statistics Loop
    *-----------------------------------------------------*/

   num_coeffs = hypre_CTAlloc(double,num_levels);

   num_variables = hypre_CTAlloc(double,num_levels);

   for (level = 0; level < num_levels; level++)
   { 

      {
         A_diag = hypre_ParCSRMatrixDiag(A_array[level]);
         A_diag_data = hypre_CSRMatrixData(A_diag);
         A_diag_i = hypre_CSRMatrixI(A_diag);
         
         A_offd = hypre_ParCSRMatrixOffd(A_array[level]);   
         A_offd_data = hypre_CSRMatrixData(A_offd);
         A_offd_i = hypre_CSRMatrixI(A_offd);
         
         row_starts = hypre_ParCSRMatrixRowStarts(A_array[level]);
         
         fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
         global_nonzeros = hypre_ParCSRMatrixDNumNonzeros(A_array[level]);
         num_coeffs[level] = global_nonzeros;
         num_variables[level] = (double) fine_size;
         
         sparse = global_nonzeros /((double) fine_size * (double) fine_size);

         min_entries = 0;
         max_entries = 0;
         min_rowsum = 0.0;
         max_rowsum = 0.0;
         
         if (hypre_CSRMatrixNumRows(A_diag))
         {
            min_entries = (A_diag_i[1]-A_diag_i[0])+(A_offd_i[1]-A_offd_i[0]);
            for (j = A_diag_i[0]; j < A_diag_i[1]; j++)
               min_rowsum += A_diag_data[j];
            for (j = A_offd_i[0]; j < A_offd_i[1]; j++)
               min_rowsum += A_offd_data[j];
            
            max_rowsum = min_rowsum;
            
            for (j = 0; j < hypre_CSRMatrixNumRows(A_diag); j++)
            {
               entries = (A_diag_i[j+1]-A_diag_i[j])+(A_offd_i[j+1]-A_offd_i[j]);
               min_entries = hypre_min(entries, min_entries);
               max_entries = hypre_max(entries, max_entries);
               
               rowsum = 0.0;
               for (i = A_diag_i[j]; i < A_diag_i[j+1]; i++)
                  rowsum += A_diag_data[i];
               
               for (i = A_offd_i[j]; i < A_offd_i[j+1]; i++)
                  rowsum += A_offd_data[i];
               
               min_rowsum = hypre_min(rowsum, min_rowsum);
               max_rowsum = hypre_max(rowsum, max_rowsum);
            }
         }
         avg_entries = global_nonzeros / ((double) fine_size);
      }
      
#ifdef HYPRE_NO_GLOBAL_PARTITION       

       numrows = (int)(row_starts[1]-row_starts[0]);
       if (!numrows) /* if we don't have any rows, then don't have this count toward
                         min row sum or min num entries */
       {
          min_entries = 1000000;
          min_rowsum =  1.0e7;
       }
       
       send_buff[0] = - (double) min_entries;
       send_buff[1] = (double) max_entries;
       send_buff[2] = - min_rowsum;
       send_buff[3] = max_rowsum;

       MPI_Reduce(send_buff, gather_buff, 4, MPI_DOUBLE, MPI_MAX, 0, comm);
       
       if (my_id ==0)
       {
          global_min_e = - gather_buff[0];
          global_max_e = gather_buff[1];
          global_min_rsum = - gather_buff[2];
          global_max_rsum = gather_buff[3];
#ifdef HYPRE_LONG_LONG
          printf( "%2d %12lld %8.0f  %0.3f  %4d %4d",
                  level, fine_size, global_nonzeros, sparse, global_min_e, 
                  global_max_e);
#else          
          printf( "%2d %7d %8.0f  %0.3f  %4d %4d",
                  level, fine_size, global_nonzeros, sparse, global_min_e, 
                  global_max_e);
#endif          
          printf("  %4.1f  %10.3e  %10.3e\n", avg_entries,
                 global_min_rsum, global_max_rsum);
       }
       
#else

       send_buff[0] = (double) min_entries;
       send_buff[1] = (double) max_entries;
       send_buff[2] = min_rowsum;
       send_buff[3] = max_rowsum;
       
       MPI_Gather(send_buff,4,MPI_DOUBLE,gather_buff,4,MPI_DOUBLE,0,comm);

       if (my_id == 0)
       {
          global_min_e = 1000000;
          global_max_e = 0;
          global_min_rsum = 1.0e7;
          global_max_rsum = 0.0;
          for (j = 0; j < num_procs; j++)
          {
             numrows = row_starts[j+1]-row_starts[j];
             if (numrows)
             {
                global_min_e = hypre_min(global_min_e, (int) gather_buff[j*4]);
                global_min_rsum = hypre_min(global_min_rsum, gather_buff[j*4 +2]);
             }
             global_max_e = hypre_max(global_max_e, (int) gather_buff[j*4 +1]);
             global_max_rsum = hypre_max(global_max_rsum, gather_buff[j*4 +3]);
          }

#ifdef HYPRE_LONG_LONG
          printf( "%2d %12lld %8.0f  %0.3f  %4d %4d",
                  level, fine_size, global_nonzeros, sparse, global_min_e, 
                  global_max_e);
#else          
          printf( "%2d %7d %8.0f  %0.3f  %4d %4d",
                  level, fine_size, global_nonzeros, sparse, global_min_e, 
                  global_max_e);
#endif          
          printf("  %4.1f  %10.3e  %10.3e\n", avg_entries,
                 global_min_rsum, global_max_rsum);
       }

#endif

        
   }

       
   if (my_id == 0)
   {
      {
         printf( "\n\nInterpolation Matrix Information:\n\n");
      }
#if HYPRE_LONG_LONG
      printf("                             entries/row    min     max");
      printf("         row sums\n");
      printf("lev        rows x cols          min max  ");
      printf("   weight   weight     min       max \n");
      printf("=======================================");
      printf("======================================\n");
#else      
      printf("                 entries/row    min     max");
      printf("         row sums\n");
      printf("lev  rows cols    min max  ");
      printf("   weight   weight     min       max \n");
      printf("=======================================");
      printf("==========================\n");
#endif
   }
  
   /*-----------------------------------------------------
    *  Enter Statistics Loop
    *-----------------------------------------------------*/


   for (level = 0; level < num_levels-1; level++)
   {
    
      {
         P_diag = hypre_ParCSRMatrixDiag(P_array[level]);
         P_diag_data = hypre_CSRMatrixData(P_diag);
         P_diag_i = hypre_CSRMatrixI(P_diag);
         
         P_offd = hypre_ParCSRMatrixOffd(P_array[level]);   
         P_offd_data = hypre_CSRMatrixData(P_offd);
         P_offd_i = hypre_CSRMatrixI(P_offd);
         
         row_starts = hypre_ParCSRMatrixRowStarts(P_array[level]);
         
         fine_size = hypre_ParCSRMatrixGlobalNumRows(P_array[level]);
         coarse_size = hypre_ParCSRMatrixGlobalNumCols(P_array[level]);
         global_nonzeros = hypre_ParCSRMatrixNumNonzeros(P_array[level]);
         
         min_weight = 1.0;
         max_weight = 0.0;
         max_rowsum = 0.0;
         min_rowsum = 0.0;
         min_entries = 0;
         max_entries = 0;
         
         if (hypre_CSRMatrixNumRows(P_diag))
         {
            if (hypre_CSRMatrixNumCols(P_diag)) min_weight = P_diag_data[0];
            for (j = P_diag_i[0]; j < P_diag_i[1]; j++)
            {
               min_weight = hypre_min(min_weight, P_diag_data[j]);
               if (P_diag_data[j] != 1.0)
                  max_weight = hypre_max(max_weight, P_diag_data[j]);
               min_rowsum += P_diag_data[j];
            }
            for (j = P_offd_i[0]; j < P_offd_i[1]; j++)
            {        
               min_weight = hypre_min(min_weight, P_offd_data[j]); 
               if (P_offd_data[j] != 1.0)
                  max_weight = hypre_max(max_weight, P_offd_data[j]);     
               min_rowsum += P_offd_data[j];
            }
            
            max_rowsum = min_rowsum;
            
            min_entries = (P_diag_i[1]-P_diag_i[0])+(P_offd_i[1]-P_offd_i[0]); 
            max_entries = 0;
            
            for (j = 0; j < hypre_CSRMatrixNumRows(P_diag); j++)
            {
               entries = (P_diag_i[j+1]-P_diag_i[j])+(P_offd_i[j+1]-P_offd_i[j]);
               min_entries = hypre_min(entries, min_entries);
               max_entries = hypre_max(entries, max_entries);
               
               rowsum = 0.0;
               for (i = P_diag_i[j]; i < P_diag_i[j+1]; i++)
               {
                  min_weight = hypre_min(min_weight, P_diag_data[i]);
                  if (P_diag_data[i] != 1.0)
                     max_weight = hypre_max(max_weight, P_diag_data[i]);
                  rowsum += P_diag_data[i];
               }
               
               for (i = P_offd_i[j]; i < P_offd_i[j+1]; i++)
               {
                  min_weight = hypre_min(min_weight, P_offd_data[i]);
                  if (P_offd_data[i] != 1.0) 
                     max_weight = hypre_max(max_weight, P_offd_data[i]);
                  rowsum += P_offd_data[i];
               }
               
               min_rowsum = hypre_min(rowsum, min_rowsum);
               max_rowsum = hypre_max(rowsum, max_rowsum);
            }
         
         }
         avg_entries = ((double) global_nonzeros) / ((double) fine_size);
      }

#ifdef HYPRE_NO_GLOBAL_PARTITION

      numrows = (int)(row_starts[1]-row_starts[0]);
      if (!numrows) /* if we don't have any rows, then don't have this count toward
                       min row sum or min num entries */
      {
         min_entries = 1000000;
         min_rowsum =  1.0e7;
         min_weight = 1.0e7;
       }
       
      send_buff[0] = - (double) min_entries;
      send_buff[1] = (double) max_entries;
      send_buff[2] = - min_rowsum;
      send_buff[3] = max_rowsum;
      send_buff[4] = - min_weight;
      send_buff[5] = max_weight;

      MPI_Reduce(send_buff, gather_buff, 6, MPI_DOUBLE, MPI_MAX, 0, comm);

      if (my_id == 0)
      {
         global_min_e = - gather_buff[0];
         global_max_e = gather_buff[1];
         global_min_rsum = -gather_buff[2];
         global_max_rsum = gather_buff[3];
         global_min_wt = -gather_buff[4];
         global_max_wt = gather_buff[5];

#ifdef HYPRE_LONG_LONG
          printf( "%2d %12lld x %-12lld %3d %3d",
                 level, fine_size, coarse_size,  global_min_e, global_max_e);
#else          
          printf( "%2d %5d x %-5d %3d %3d",
                 level, fine_size, coarse_size,  global_min_e, global_max_e);
#endif          
         printf("  %10.3e %9.3e %9.3e %9.3e\n",
                global_min_wt, global_max_wt, 
                global_min_rsum, global_max_rsum);
      }


#else
      
      send_buff[0] = (double) min_entries;
      send_buff[1] = (double) max_entries;
      send_buff[2] = min_rowsum;
      send_buff[3] = max_rowsum;
      send_buff[4] = min_weight;
      send_buff[5] = max_weight;
      
      MPI_Gather(send_buff,6,MPI_DOUBLE,gather_buff,6,MPI_DOUBLE,0,comm);
      
      if (my_id == 0)
      {
         global_min_e = 1000000;
         global_max_e = 0;
         global_min_rsum = 1.0e7;
         global_max_rsum = 0.0;
         global_min_wt = 1.0e7;
         global_max_wt = 0.0;
         
         for (j = 0; j < num_procs; j++)
         {
            numrows = row_starts[j+1] - row_starts[j];
            if (numrows)
            {
               global_min_e = hypre_min(global_min_e, (int) gather_buff[j*6]);
               global_min_rsum = hypre_min(global_min_rsum, gather_buff[j*6+2]);
               global_min_wt = hypre_min(global_min_wt, gather_buff[j*6+4]);
            }
            global_max_e = hypre_max(global_max_e, (int) gather_buff[j*6+1]);
            global_max_rsum = hypre_max(global_max_rsum, gather_buff[j*6+3]);
            global_max_wt = hypre_max(global_max_wt, gather_buff[j*6+5]);
         }
         
#ifdef HYPRE_LONG_LONG
         printf( "%2d %12lld x %-12lld %3d %3d",
                 level, fine_size, coarse_size,  global_min_e, global_max_e);
#else          
         printf( "%2d %5d x %-5d %3d %3d",
                 level, fine_size, coarse_size,  global_min_e, global_max_e);
#endif          
         printf("  %10.3e %9.3e %9.3e %9.3e\n",
                global_min_wt, global_max_wt, 
                global_min_rsum, global_max_rsum);
      }

#endif

   }


   total_variables = 0;
   operat_cmplxty = 0;
   for (j=0;j<hypre_ParAMGDataNumLevels(amg_data);j++)
   {
      operat_cmplxty +=  num_coeffs[j] / num_coeffs[0];
      total_variables += num_variables[j];
   }
   if (num_variables[0] != 0)
      grid_cmplxty = total_variables / num_variables[0];
 
   if (my_id == 0 )
   {
      printf("\n\n     Complexity:    grid = %f\n",grid_cmplxty);
      printf("                operator = %f\n",operat_cmplxty);
   }

   if (my_id == 0) printf("\n\n");

   if (my_id == 0)
   { 
      printf("\n\nBoomerAMG SOLVER PARAMETERS:\n\n");
      printf( "  Maximum number of cycles:         %d \n",max_iter);
      printf( "  Stopping Tolerance:               %e \n",tol); 
      printf( "  Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      printf( "  Relaxation Parameters:\n");
      printf( "   Visiting Grid:                     down   up  coarse\n");
      printf( "            Number of partial sweeps: %4d   %2d  %4d \n",
              num_grid_sweeps[1],
              num_grid_sweeps[2],num_grid_sweeps[3]);
      printf( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:   %4d   %2d  %4d \n",
              grid_relax_type[1],
              grid_relax_type[2],grid_relax_type[3]);
#if 1 /* TO DO: may not want this to print if CG in the coarse grid */
      printf( "   Point types, partial sweeps (1=C, -1=F):\n");
      if (grid_relax_points)
      {
         printf( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
              printf("  %2d", grid_relax_points[1][j]);
         printf( "\n");
         printf( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
              printf("  %2d", grid_relax_points[2][j]);
         printf( "\n");
         printf( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
              printf("  %2d", grid_relax_points[3][j]);
         printf( "\n\n");
      }
      else if (relax_order == 1)
      {
         printf( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
              printf("  %2d  %2d", one, minus_one);
         printf( "\n");
         printf( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
              printf("  %2d  %2d", minus_one, one);
         printf( "\n");
         printf( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
              printf("  %2d", zero);
         printf( "\n\n");
      }
      else 
      {
         printf( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
              printf("  %2d", zero);
         printf( "\n");
         printf( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
              printf("  %2d", zero);
         printf( "\n");
         printf( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
              printf("  %2d", zero);
         printf( "\n\n");
      }
#endif
      if (smooth_type == 6)
         for (j=0; j < smooth_num_levels; j++)
            printf( " Schwarz Relaxation Weight %f level %d\n",
			hypre_ParAMGDataSchwarzRlxWeight(amg_data),j);
      for (j=0; j < num_levels; j++)
         if (relax_weight[j] != 1)
	       printf( " Relaxation Weight %f level %d\n",relax_weight[j],j);
      for (j=0; j < num_levels; j++)
         if (omega[j] != 1)
               printf( " Outer relaxation weight %f level %d\n",omega[j],j);
   }


   /*if (seq_cg) 
   {
      hypre_seqAMGSetupStats(amg_data,num_coeffs[0],num_variables[0], 
                             operat_cmplxty, grid_cmplxty );
   }*/
   




   hypre_TFree(num_coeffs);
   hypre_TFree(num_variables);
   hypre_TFree(send_buff);
   hypre_TFree(gather_buff);
   
   return(0);
}  




/*---------------------------------------------------------------
 * hypre_BoomerAMGWriteSolverParams
 *---------------------------------------------------------------*/


int    hypre_BoomerAMGWriteSolverParams(data)
void    *data;
 
{ 
   hypre_ParAMGData  *amg_data = data;
 
   /* amg solve params */
   int      num_levels; 
   int      max_iter;
   int      cycle_type;    
   int     *num_grid_sweeps;  
   int     *grid_relax_type;   
   int    **grid_relax_points; 
   int      relax_order;
   double  *relax_weight;
   double  *omega;
   double   tol;
   int      smooth_type; 
   int      smooth_num_levels; 
   /* amg output params */
   int      amg_print_level;
 
   int      j;
   int      one = 1;
   int      minus_one = -1;
   int      zero = 0;
 
 
   /*----------------------------------------------------------
    * Get the amg_data data
    *----------------------------------------------------------*/

   num_levels = hypre_ParAMGDataNumLevels(amg_data);
   max_iter   = hypre_ParAMGDataMaxIter(amg_data);
   cycle_type = hypre_ParAMGDataCycleType(amg_data);    
   num_grid_sweeps = hypre_ParAMGDataNumGridSweeps(amg_data);  
   grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_order = hypre_ParAMGDataRelaxOrder(amg_data);
   relax_weight = hypre_ParAMGDataRelaxWeight(amg_data); 
   omega = hypre_ParAMGDataOmega(amg_data); 
   smooth_type = hypre_ParAMGDataSmoothType(amg_data); 
   smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data); 
   tol = hypre_ParAMGDataTol(amg_data);
 
   amg_print_level = hypre_ParAMGDataPrintLevel(amg_data);
 
   /*----------------------------------------------------------
    * AMG info
    *----------------------------------------------------------*/
 
   if (amg_print_level == 1 || amg_print_level == 3)
   { 
      printf("\n\nBoomerAMG SOLVER PARAMETERS:\n\n");
      printf( "  Maximum number of cycles:         %d \n",max_iter);
      printf( "  Stopping Tolerance:               %e \n",tol); 
      printf( "  Cycle type (1 = V, 2 = W, etc.):  %d\n\n", cycle_type);
      printf( "  Relaxation Parameters:\n");
      printf( "   Visiting Grid:                     down   up  coarse\n");
      printf( "   Visiting Grid:                     down   up  coarse\n");
      printf( "            Number of partial sweeps: %4d   %2d  %4d \n",
              num_grid_sweeps[1],
              num_grid_sweeps[2],num_grid_sweeps[3]);
      printf( "   Type 0=Jac, 3=hGS, 6=hSGS, 9=GE:   %4d   %2d  %4d \n",
              grid_relax_type[1],
              grid_relax_type[2],grid_relax_type[3]);
      printf( "   Point types, partial sweeps (1=C, -1=F):\n");
      if (grid_relax_points)
      {
         printf( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
              printf("  %2d", grid_relax_points[1][j]);
         printf( "\n");
         printf( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
              printf("  %2d", grid_relax_points[2][j]);
         printf( "\n");
         printf( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
              printf("  %2d", grid_relax_points[3][j]);
         printf( "\n\n");
      }
      else if (relax_order == 1)
      {
         printf( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
              printf("  %2d  %2d", one, minus_one);
         printf( "\n");
         printf( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
              printf("  %2d  %2d", minus_one, one);
         printf( "\n");
         printf( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
              printf("  %2d", zero);
         printf( "\n\n");
      }
      else 
      {
         printf( "                  Pre-CG relaxation (down):");
         for (j = 0; j < num_grid_sweeps[1]; j++)
              printf("  %2d", zero);
         printf( "\n");
         printf( "                   Post-CG relaxation (up):");
         for (j = 0; j < num_grid_sweeps[2]; j++)
              printf("  %2d", zero);
         printf( "\n");
         printf( "                             Coarsest grid:");
         for (j = 0; j < num_grid_sweeps[3]; j++)
              printf("  %2d", zero);
         printf( "\n\n");
      }
      if (smooth_type == 6)
         for (j=0; j < smooth_num_levels; j++)
            printf( " Schwarz Relaxation Weight %f level %d\n",
			hypre_ParAMGDataSchwarzRlxWeight(amg_data),j);
      for (j=0; j < num_levels; j++)
         if (relax_weight[j] != 1)
	       printf( " Relaxation Weight %f level %d\n",relax_weight[j],j);
      for (j=0; j < num_levels; j++)
         if (omega[j] != 1)
               printf( " Outer relaxation weight %f level %d\n",omega[j],j);

      printf( " Output flag (print_level): %d \n", amg_print_level);
   }
 
   return 0;
}
