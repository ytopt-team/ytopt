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
/*#include "par_csr_block_matrix.h"*/

#define DEBUG 0

#define THETA 0

#define MPIP_SETUP_ON 0

/*****************************************************************************
 *
 * Routine for driving the setup phase of AMG
 *
 *****************************************************************************/

/*****************************************************************************
 * hypre_BoomerAMGSetup
 *****************************************************************************/

int
hypre_BoomerAMGSetup( void               *amg_vdata,
                   hypre_ParCSRMatrix *A,
                   hypre_ParVector    *f,
                   hypre_ParVector    *u         )
{
   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(A); 

   hypre_ParAMGData   *amg_data = amg_vdata;

   /* Data Structure variables */

   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;
   hypre_ParVector     *Vtemp;
   hypre_ParVector     *Rtemp;
   hypre_ParVector     *Ptemp;
   hypre_ParVector     *Ztemp;
   hypre_ParCSRMatrix **P_array;
   hypre_ParVector    *Residual_array;
   int                **CF_marker_array;   
   int                **dof_func_array;   
   int                 *dof_func;
   int                 *dof_func1;
   int                 *col_offd_S_to_A;
   int                 *col_offd_SN_to_AN;
   double              *relax_weight;
   double              *omega;
   double               strong_threshold;
   double               max_row_sum;
   double               trunc_factor, jacobi_trunc_threshold;
   double               agg_trunc_factor;
   double               S_commpkg_switch;
   /*double      		CR_rate;*/

   int      max_levels; 
   int      amg_logging;
   int      amg_print_level;
   int      debug_flag;
   int      local_num_vars;
   int      P_max_elmts;
   int      P_max1;
   int      P_max2;
   int      agg_P_max_elmts;
   /*int      IS_type;
   int      num_CR_relax_steps;
   int      CR_use_CG; */


   /*hypre_ParCSRBlockMatrix **A_block_array, **P_block_array;*/
 
   /* Local variables */
   int                 *CF_marker;
   int                 *CFN_marker;
   int                 *CF2_marker;
   hypre_ParCSRMatrix  *S;
   hypre_ParCSRMatrix  *S2;
   hypre_ParCSRMatrix  *SN;
   hypre_ParCSRMatrix  *P;
   hypre_ParCSRMatrix  *P1;
   hypre_ParCSRMatrix  *P2;
   hypre_ParCSRMatrix  *PN;
   hypre_ParCSRMatrix  *A_H;
   hypre_ParCSRMatrix  *AN;
   double             **l1_norms = NULL;
   /*double              *SmoothVecs = NULL;*/

   int       old_num_levels, num_levels;
   int       level;
   int       local_size, i;
   HYPRE_BigInt first_local_row, num_fun;
   HYPRE_BigInt coarse_size;
   int       coarsen_type;
   int       measure_type;
   int       setup_type;
   HYPRE_BigInt fine_size;
   int       rest, tms, indx;
   double    size;
   int       not_finished_coarsening = 1;
   int       Setup_err_flag = 0;
   int       seq_threshold = hypre_ParAMGDataSeqThreshold(amg_data);
   int       coarse_threshold = hypre_ParAMGDataMaxCoarseSize(amg_data);
   int       j, k;
   int       num_procs,my_id;
   int       num_threads;
   int      *grid_relax_type = hypre_ParAMGDataGridRelaxType(amg_data);
   int       num_functions = hypre_ParAMGDataNumFunctions(amg_data);
   int       nodal = hypre_ParAMGDataNodal(amg_data);
   int       num_paths = hypre_ParAMGDataNumPaths(amg_data);
   int       agg_num_levels = hypre_ParAMGDataAggNumLevels(amg_data);
   int	    *coarse_dof_func = NULL;
   HYPRE_BigInt *coarse_pnts_global = NULL;
   HYPRE_BigInt *coarse_pnts_global1 = NULL;
   int       num_cg_sweeps;

   double *max_eig_est = NULL;
   double *min_eig_est = NULL;

   HYPRE_Solver *smoother = NULL;
   HYPRE_Solver *smoother_prec = NULL;
   /* 
   int       smooth_type = hypre_ParAMGDataSmoothType(amg_data);
   int       smooth_num_levels = hypre_ParAMGDataSmoothNumLevels(amg_data);
   int	     sym;
   int	     nlevel;
   double    thresh;
   double    filter;
   double    drop_tol;
   int	     max_nz_per_row;
   char     *euclidfile;*/

   int interp_type;
   int agg_interp_type;
   int post_interp_type;  /* what to do after computing the interpolation matrix
                             0 for nothing, 1 for a Jacobi step */

   /*hypre_ParCSRBlockMatrix *A_H_block;*/

   int block_mode = 0;

   double    wall_time;   /* for debugging instrumentation */

   MPI_Comm_size(comm, &num_procs);   
   MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();

   old_num_levels = hypre_ParAMGDataNumLevels(amg_data);
   max_levels = hypre_ParAMGDataMaxLevels(amg_data);
   amg_logging = hypre_ParAMGDataLogging(amg_data);
   amg_print_level = hypre_ParAMGDataPrintLevel(amg_data);
   coarsen_type = hypre_ParAMGDataCoarsenType(amg_data);
   measure_type = hypre_ParAMGDataMeasureType(amg_data);
   setup_type = hypre_ParAMGDataSetupType(amg_data);
   debug_flag = hypre_ParAMGDataDebugFlag(amg_data);
   relax_weight = hypre_ParAMGDataRelaxWeight(amg_data);
   omega = hypre_ParAMGDataOmega(amg_data);
   dof_func = hypre_ParAMGDataDofFunc(amg_data);
   /*sym = hypre_ParAMGDataSym(amg_data);
   nlevel = hypre_ParAMGDataLevel(amg_data);
   filter = hypre_ParAMGDataFilter(amg_data);
   thresh = hypre_ParAMGDataThreshold(amg_data);
   drop_tol = hypre_ParAMGDataDropTol(amg_data);
   max_nz_per_row = hypre_ParAMGDataMaxNzPerRow(amg_data);
   euclidfile = hypre_ParAMGDataEuclidFile(amg_data);*/
   agg_interp_type = hypre_ParAMGDataAggInterpType(amg_data);
   interp_type = hypre_ParAMGDataInterpType(amg_data);
   post_interp_type = hypre_ParAMGDataPostInterpType(amg_data);
   /*IS_type = hypre_ParAMGDataISType(amg_data);
   num_CR_relax_steps = hypre_ParAMGDataNumCRRelaxSteps(amg_data);
   CR_rate = hypre_ParAMGDataCRRate(amg_data);
   CR_use_CG = hypre_ParAMGDataCRUseCG(amg_data);*/

   hypre_ParCSRMatrixSetNumNonzeros(A);
   hypre_ParCSRMatrixSetDNumNonzeros(A);
   hypre_ParAMGDataNumVariables(amg_data) = hypre_ParCSRMatrixNumRows(A);

   if (setup_type == 0) return Setup_err_flag;

   S = NULL;

   A_array = hypre_ParAMGDataAArray(amg_data);
   P_array = hypre_ParAMGDataPArray(amg_data);
   CF_marker_array = hypre_ParAMGDataCFMarkerArray(amg_data);
   dof_func_array = hypre_ParAMGDataDofFuncArray(amg_data);
   local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

 
   /*A_block_array = hypre_ParAMGDataABlockArray(amg_data);
   P_block_array = hypre_ParAMGDataPBlockArray(amg_data);*/

   grid_relax_type[3] = hypre_ParAMGDataUserCoarseRelaxType(amg_data); 

   

   hypre_ParAMGDataBlockMode(amg_data) = block_mode;
   /* end of systems checks */



   /*if (A_array || A_block_array || P_array || P_block_array || CF_marker_array || dof_func_array)*/
   if (A_array || P_array || CF_marker_array || dof_func_array)
   {
      for (j = 1; j < old_num_levels; j++)
      {
         if (A_array[j])
         {
            hypre_ParCSRMatrixDestroy(A_array[j]);
            A_array[j] = NULL;
         }

         /*if (A_block_array[j])
         {
            hypre_ParCSRBlockMatrixDestroy(A_block_array[j]);
            A_block_array[j] = NULL;
         }*/
        


         if (dof_func_array[j])
         {
            hypre_TFree(dof_func_array[j]);
            dof_func_array[j] = NULL;
         }
      }

      for (j = 0; j < old_num_levels-1; j++)
      {
         if (P_array[j])
         {
            hypre_ParCSRMatrixDestroy(P_array[j]);
            P_array[j] = NULL;
         }

         /*if (P_block_array[j])
         {
            hypre_ParCSRBlockMatrixDestroy(P_block_array[j]);
            P_array[j] = NULL;
         }*/

      }

/* Special case use of CF_marker_array when old_num_levels == 1
   requires us to attempt this deallocation every time */
      if (CF_marker_array[0])
      {
        hypre_TFree(CF_marker_array[0]);
        CF_marker_array[0] = NULL;
      }

      for (j = 1; j < old_num_levels-1; j++)
      {
         if (CF_marker_array[j])
         {
            hypre_TFree(CF_marker_array[j]);
            CF_marker_array[j] = NULL;
         }
      }
   }

   if (A_array == NULL)
      A_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels);
   /*if (A_block_array == NULL)
      A_block_array = hypre_CTAlloc(hypre_ParCSRBlockMatrix*, max_levels);*/


   if (P_array == NULL && max_levels > 1)
      P_array = hypre_CTAlloc(hypre_ParCSRMatrix*, max_levels-1);
   /*if (P_block_array == NULL && max_levels > 1)
      P_block_array = hypre_CTAlloc(hypre_ParCSRBlockMatrix*, max_levels-1);*/


   if (CF_marker_array == NULL)
      CF_marker_array = hypre_CTAlloc(int*, max_levels);
   if (dof_func_array == NULL)
      dof_func_array = hypre_CTAlloc(int*, max_levels);
   if (num_functions > 1 && dof_func == NULL)
   {
      first_local_row = hypre_ParCSRMatrixFirstRowIndex(A);
      dof_func = hypre_CTAlloc(int,local_size);
      num_fun = (HYPRE_BigInt) num_functions;
      rest = (int)(first_local_row-((first_local_row/num_fun)*num_fun));
      indx = num_functions-rest;
      if (rest == 0) indx = 0;
      k = num_functions - 1;
      for (j = indx-1; j > -1; j--)
         dof_func[j] = k--;
      tms = local_size/num_functions;
      if (tms*num_functions+indx > local_size) tms--;
      for (j=0; j < tms; j++)
      {
         for (k=0; k < num_functions; k++)
            dof_func[indx++] = k;
      }
      k = 0;
      while (indx < local_size)
         dof_func[indx++] = k++;
      hypre_ParAMGDataDofFunc(amg_data) = dof_func;
   }

   A_array[0] = A;

   

   dof_func_array[0] = dof_func;
   hypre_ParAMGDataCFMarkerArray(amg_data) = CF_marker_array;
   hypre_ParAMGDataDofFuncArray(amg_data) = dof_func_array;
   hypre_ParAMGDataAArray(amg_data) = A_array;
   hypre_ParAMGDataPArray(amg_data) = P_array;
   hypre_ParAMGDataRArray(amg_data) = P_array;


   Vtemp = hypre_ParAMGDataVtemp(amg_data);

   if (Vtemp != NULL)
   {
      hypre_ParVectorDestroy(Vtemp);
      Vtemp = NULL;
   }

   Vtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
   hypre_ParVectorInitialize(Vtemp);
   hypre_ParVectorSetPartitioningOwner(Vtemp,0);
   hypre_ParAMGDataVtemp(amg_data) = Vtemp;

   /*if ((smooth_num_levels > 0 && smooth_type > 9) 
		|| relax_weight[0] < 0 || omega[0] < 0 ||
                hypre_ParAMGDataSchwarzRlxWeight(amg_data) < 0)*/
   if ( relax_weight[0] < 0 || omega[0] < 0 )
   {
      Ptemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize(Ptemp);
      hypre_ParVectorSetPartitioningOwner(Ptemp,0);
      hypre_ParAMGDataPtemp(amg_data) = Ptemp;
      Rtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize(Rtemp);
      hypre_ParVectorSetPartitioningOwner(Rtemp,0);
      hypre_ParAMGDataRtemp(amg_data) = Rtemp;
   }
   /*if ((smooth_num_levels > 0 && smooth_type > 6)
		 || relax_weight[0] < 0 || omega[0] < 0 ||
                hypre_ParAMGDataSchwarzRlxWeight(amg_data) < 0)*/
   if ( relax_weight[0] < 0 || omega[0] < 0 )
   {
      Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize(Ztemp);
      hypre_ParVectorSetPartitioningOwner(Ztemp,0);
      hypre_ParAMGDataZtemp(amg_data) = Ztemp;
   }
   F_array = hypre_ParAMGDataFArray(amg_data);
   U_array = hypre_ParAMGDataUArray(amg_data);

   

   if (F_array != NULL || U_array != NULL)
   {
      for (j = 1; j < old_num_levels; j++)
      {
         if (F_array[j] != NULL)
         {
            hypre_ParVectorDestroy(F_array[j]);
            F_array[j] = NULL;
         }
         if (U_array[j] != NULL)
         {
            hypre_ParVectorDestroy(U_array[j]);
            U_array[j] = NULL;
         }
      }
   }

   if (F_array == NULL)
      F_array = hypre_CTAlloc(hypre_ParVector*, max_levels);
   if (U_array == NULL)
      U_array = hypre_CTAlloc(hypre_ParVector*, max_levels);

   F_array[0] = f;
   U_array[0] = u;

   hypre_ParAMGDataFArray(amg_data) = F_array;
   hypre_ParAMGDataUArray(amg_data) = U_array;

   /*----------------------------------------------------------
    * Initialize hypre_ParAMGData
    *----------------------------------------------------------*/

   not_finished_coarsening = 1;
   level = 0;
  
   strong_threshold = hypre_ParAMGDataStrongThreshold(amg_data);
   max_row_sum = hypre_ParAMGDataMaxRowSum(amg_data);
   trunc_factor = hypre_ParAMGDataTruncFactor(amg_data);
   agg_trunc_factor = hypre_ParAMGDataAggTruncFactor(amg_data);
   P_max_elmts = hypre_ParAMGDataPMaxElmts(amg_data);
   P_max1 = hypre_ParAMGDataPMax1(amg_data);
   P_max2 = hypre_ParAMGDataPMax2(amg_data);
   agg_P_max_elmts = hypre_ParAMGDataAggPMaxElmts(amg_data);
   jacobi_trunc_threshold = hypre_ParAMGDataJacobiTruncThreshold(amg_data);
   S_commpkg_switch = hypre_ParAMGDataSCommPkgSwitch(amg_data);
   /*if (smooth_num_levels > level)
   {
      smoother = hypre_CTAlloc(HYPRE_Solver, smooth_num_levels);
      hypre_ParAMGDataSmoother(amg_data) = smoother;
   }*/

   /*-----------------------------------------------------
    *  Enter Coarsening Loop
    *-----------------------------------------------------*/

   while (not_finished_coarsening)
   {

#if MPIP_SETUP_ON
      MPI_Pcontrol(2); 
      MPI_Pcontrol(1); 
#endif

      {
         fine_size = hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
      }
      
      if (level > 0)
      {   
         {
            F_array[level] =
               hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                     hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                     hypre_ParCSRMatrixRowStarts(A_array[level]));
            hypre_ParVectorInitialize(F_array[level]);
            hypre_ParVectorSetPartitioningOwner(F_array[level],0);
            
            U_array[level] =
               hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                     hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                     hypre_ParCSRMatrixRowStarts(A_array[level]));
            hypre_ParVectorInitialize(U_array[level]);
            hypre_ParVectorSetPartitioningOwner(U_array[level],0);
         }
         
      }

      /*-------------------------------------------------------------
       * Select coarse-grid points on 'level' : returns CF_marker
       * for the level.  Returns strength matrix, S  
       *--------------------------------------------------------------*/
     
      if (debug_flag==1) wall_time = time_getWallclockSeconds();
      if (debug_flag==3)
      {
          printf("\n ===== Proc = %d     Level = %d  =====\n",
                        my_id, level);
          fflush(NULL);
      }

      /*if ((smooth_type == 6 || smooth_type == 16) && smooth_num_levels > level)
      {

	 schwarz_relax_wt = hypre_ParAMGDataSchwarzRlxWeight(amg_data);
         HYPRE_SchwarzCreate(&smoother[level]);
         HYPRE_SchwarzSetNumFunctions(smoother[level],num_functions);
         HYPRE_SchwarzSetVariant(smoother[level],
		hypre_ParAMGDataVariant(amg_data));
         HYPRE_SchwarzSetOverlap(smoother[level],
		hypre_ParAMGDataOverlap(amg_data));
         HYPRE_SchwarzSetDomainType(smoother[level],
		hypre_ParAMGDataDomainType(amg_data));
	 if (schwarz_relax_wt > 0)
            HYPRE_SchwarzSetRelaxWeight(smoother[level],schwarz_relax_wt);
         HYPRE_SchwarzSetup(smoother[level],
                        (HYPRE_ParCSRMatrix) A_array[level],
                        (HYPRE_ParVector) f,
                        (HYPRE_ParVector) u);
      }*/

      if (max_levels > 1)
      {
         {
            local_num_vars =
               hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_array[level]));
         }


         /**** Get the Strength Matrix ****/        

         if (hypre_ParAMGDataGSMG(amg_data) == 0)
	 {
	    if (nodal) /* if we are solving systems and 
                          not using the unknown approach then we need to 
                          convert A to a nodal matrix - values that represent the
                          blocks  - before getting the strength matrix*/
	    {

               {
                  hypre_BoomerAMGCreateNodalA(A_array[level],num_functions,
                                              dof_func_array[level], abs(nodal), &AN);
               }

               /* dof array not needed for creating S because we pass in that 
                  the number of functions is 1 */
               if (nodal == 3 || nodal == -3 || nodal == 7)  /* option 3 may have negative entries in AN - 
                                                all other options are pos numbers only */
                  hypre_BoomerAMGCreateS(AN, strong_threshold, max_row_sum,
                                   1, NULL,&SN);
               else
		  hypre_BoomerAMGCreateSabs(AN, strong_threshold, max_row_sum,
                                   1, NULL,&SN);
#if 1
/* TEMP - to compare with serial */
               hypre_BoomerAMGCreateS(AN, strong_threshold, max_row_sum,
                                      1, NULL,&SN);
#endif
            

               col_offd_S_to_A = NULL;
	       col_offd_SN_to_AN = NULL;
	       if (strong_threshold > S_commpkg_switch)
                  hypre_BoomerAMGCreateSCommPkg(AN,SN,&col_offd_SN_to_AN);
	    }
	    else
	    {
	       hypre_BoomerAMGCreateS(A_array[level], 
				   strong_threshold, max_row_sum, 
				   num_functions, dof_func_array[level],&S);
	       col_offd_S_to_A = NULL;
	       if (strong_threshold > S_commpkg_switch)
                  hypre_BoomerAMGCreateSCommPkg(A_array[level],S,
				&col_offd_S_to_A);
	    }
	 }
	 /*else
	 {
	    hypre_BoomerAMGCreateSmoothDirs(amg_data, A_array[level],
	       SmoothVecs, strong_threshold, 
               num_functions, dof_func_array[level], &S);
	 }*/


         /**** Do the appropriate coarsening ****/ 


         if (coarsen_type == 6)  /* falgout */
         {
            if (nodal == 0) /* nonsystems or unknown approach for systems*/
            {
               hypre_BoomerAMGCoarsenFalgout(S, A_array[level], measure_type,
                                             debug_flag, &CF_marker);

               if (level < agg_num_levels && (agg_interp_type < 30 || agg_interp_type > 32))
               {
                 /* set num_functions 1 in CoarseParms, since coarse_dof_func
                    is not needed here */
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             1, dof_func_array[level], CF_marker,
                                             &coarse_dof_func,&coarse_pnts_global1);
                  hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                             coarse_pnts_global1, &S2);
                  hypre_BoomerAMGCoarsenFalgout(S2, S2, measure_type,
                                                debug_flag, &CFN_marker);
                  hypre_ParCSRMatrixDestroy(S2);
                  hypre_BoomerAMGCorrectCFMarker (CF_marker, local_num_vars, CFN_marker);
                  hypre_TFree(coarse_pnts_global1);
                  hypre_TFree(CFN_marker);
               }
            }
            else if (nodal < 0 || block_mode ) /*       nodal interpolation
                                                       or if nodal < 0 then we 
                                                       build interpolation normally 
                                                       using the nodal matrix */
            {
               hypre_BoomerAMGCoarsenFalgout(SN, SN, measure_type,
                                    debug_flag, &CF_marker);
            }
            
            else if (nodal > 0)  /* nodal = 1,2,3: here we convert our nodal coarsening 
                                    so that we can perform regular interpolation */
            {
               hypre_BoomerAMGCoarsenFalgout(SN, SN, measure_type,
                                             debug_flag, &CFN_marker);
               
               if (level < agg_num_levels)
               {
                  /* set num_functions 1 in CoarseParms, since coarse_dof_func
                 is not needed here */
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                        1, dof_func_array[level], CFN_marker,
                        &coarse_dof_func,&coarse_pnts_global1);
                  hypre_BoomerAMGCreate2ndS (SN, CFN_marker, num_paths,
                                coarse_pnts_global1, &S2);
                  hypre_BoomerAMGCoarsenFalgout(S2, S2, measure_type,
                                debug_flag, &CF2_marker);
                  hypre_ParCSRMatrixDestroy(S2);
                  hypre_BoomerAMGCorrectCFMarker (CFN_marker, local_num_vars, CF2_marker);
                  hypre_TFree(coarse_pnts_global1);
                  hypre_TFree(CF2_marker);
               }
               
               col_offd_S_to_A = NULL;
               hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker, col_offd_SN_to_AN,
                                              num_functions, nodal, 0, NULL, &CF_marker, 
                                              &col_offd_S_to_A, &S);
               if (col_offd_SN_to_AN == NULL)
                  col_offd_S_to_A = NULL;
               hypre_TFree(CFN_marker);
               hypre_TFree(col_offd_SN_to_AN);
               hypre_ParCSRMatrixDestroy(AN);
               hypre_ParCSRMatrixDestroy(SN);
               
            }
         }
         else if (coarsen_type == 7) /* cljp1 */
         {
            if (nodal == 0) /* nonsystems or unknown approach for systems*/
            {
               hypre_BoomerAMGCoarsen(S, A_array[level], 2,
                                      debug_flag, &CF_marker);
               if (level < agg_num_levels && (agg_interp_type < 30 || agg_interp_type > 32))
               {
                  /* set num_functions 1 in CoarseParms, since coarse_dof_func
                     is not needed here */
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             1, dof_func_array[level], CF_marker,
                                             &coarse_dof_func,&coarse_pnts_global1);
                  hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                             coarse_pnts_global1, &S2);
                  hypre_BoomerAMGCoarsen(S2, S2, 2, debug_flag, &CFN_marker);
                  hypre_ParCSRMatrixDestroy(S2);
                  hypre_BoomerAMGCorrectCFMarker (CF_marker, local_num_vars, CFN_marker);
                  hypre_TFree(CFN_marker);
                  hypre_TFree(coarse_pnts_global1);
               }
            }
            else if (nodal < 0 || block_mode ) /*        nodal interpolation
                                                        or if nodal < 0 then we 
                                                        build interpolation normally 
                                                        using the nodal matrix */
            {
               hypre_BoomerAMGCoarsen(SN, SN, 2, debug_flag, &CF_marker);
            }
            
            else if (nodal > 0) /* nodal = 1,2,3: here we convert our nodal coarsening 
                                   so that we can perform regualr interpolation */
            {
               hypre_BoomerAMGCoarsen(SN, SN, 2, debug_flag, &CFN_marker);
               
               col_offd_S_to_A = NULL;
               hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker, col_offd_SN_to_AN,
                                              num_functions, nodal, 0, NULL, &CF_marker, 
                                              &col_offd_S_to_A, &S);
               if (col_offd_SN_to_AN == NULL)
                  col_offd_S_to_A = NULL;
               hypre_TFree(CFN_marker);
               hypre_TFree(col_offd_SN_to_AN);
               
               hypre_ParCSRMatrixDestroy(AN);
               hypre_ParCSRMatrixDestroy(SN);
               
            }
         }

         else if (coarsen_type == 8) /* pmis */
         {
            if (nodal == 0) /* nonsystems or unknown approach for systems*/
            {
               hypre_BoomerAMGCoarsenPMIS(S, A_array[level], 0,
                                          debug_flag, &CF_marker);

               if (level < agg_num_levels && (agg_interp_type < 30 || agg_interp_type > 32))
               {
                  /* set num_functions 1 in CoarseParms, since coarse_dof_func
                     is not needed here */
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             1, dof_func_array[level], CF_marker,
                                             &coarse_dof_func,&coarse_pnts_global1);
                  hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                             coarse_pnts_global1, &S2);
                  hypre_BoomerAMGCoarsenPMIS(S2, S2, 0,
                                             debug_flag, &CFN_marker);
                  hypre_ParCSRMatrixDestroy(S2);
                  hypre_BoomerAMGCorrectCFMarker (CF_marker, local_num_vars, CFN_marker);
                  hypre_TFree(CFN_marker);
                  hypre_TFree(coarse_pnts_global1);
               }
            }
            else if (nodal < 0 || block_mode ) /*       nodal interpolation
                                                       or if nodal < 0 then we 
                                                       build interpolation normally 
                                                       using the nodal matrix */
            {
               hypre_BoomerAMGCoarsenPMIS(SN, SN, 0,
                                    debug_flag, &CF_marker);
            }
            else if (nodal > 0) /* nodal = 1,2,3: here we convert our nodal coarsening 
                                  so that we can perform regualr interpolation */
           {
              hypre_BoomerAMGCoarsenPMIS(SN, SN, 0,
                                    debug_flag, &CFN_marker);
             /*hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker,
                             num_functions, nodal, 0, NULL, &CF_marker, &S);*/
	     col_offd_S_to_A = NULL;
             hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker, col_offd_SN_to_AN,
                             num_functions, nodal, 0, NULL, &CF_marker, 
			     &col_offd_S_to_A, &S);
             if (col_offd_SN_to_AN == NULL)
	        col_offd_S_to_A = NULL;
             hypre_TFree(CFN_marker);
             hypre_TFree(col_offd_SN_to_AN);
             hypre_ParCSRMatrixDestroy(AN);
             hypre_ParCSRMatrixDestroy(SN);
           }
         }

         else if (coarsen_type == 9) /* pmis1 */
         {
            if (nodal == 0) /* nonsystems or unknown approach for systems*/
            {
               hypre_BoomerAMGCoarsenPMIS(S, A_array[level], 2,
                                          debug_flag, &CF_marker);
               if (level < agg_num_levels && (agg_interp_type < 30 || agg_interp_type > 32))
               {
                  /* set num_functions 1 in CoarseParms, since coarse_dof_func
                     is not needed here */
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             1, dof_func_array[level], CF_marker,
                                             &coarse_dof_func,&coarse_pnts_global1);
                  hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                             coarse_pnts_global1, &S2);
                  hypre_BoomerAMGCoarsenPMIS(S2, S2, 2,
                                             debug_flag, &CFN_marker);
                  hypre_ParCSRMatrixDestroy(S2);
                  hypre_BoomerAMGCorrectCFMarker (CF_marker, local_num_vars, CFN_marker);
                  hypre_TFree(CFN_marker);
                  hypre_TFree(coarse_pnts_global1);
               }
            }
            
            else if (nodal < 0 || block_mode ) /*        nodal interpolation
                                                        or if nodal < 0 then we 
                                                        build interpolation normally 
                                                        using the nodal matrix */
            {
               hypre_BoomerAMGCoarsenPMIS(SN, SN, 2,
                                          debug_flag, &CF_marker);
               
            }
            else if (nodal > 0) /* nodal = 1,2,3: here we convert our nodal coarsening 
                                   so that we can perform regualr interpolation */
            {
               hypre_BoomerAMGCoarsenPMIS(SN, SN, 2,
                                          debug_flag, &CFN_marker);
               /*hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker,
                 num_functions, nodal, 0, NULL, &CF_marker, &S);*/
               col_offd_S_to_A = NULL;
               hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker, col_offd_SN_to_AN,
                                              num_functions, nodal, 0, NULL, &CF_marker, 
                                              &col_offd_S_to_A, &S);
               if (col_offd_SN_to_AN == NULL)
                  col_offd_S_to_A = NULL;
               hypre_TFree(CFN_marker);
               hypre_TFree(col_offd_SN_to_AN);
               hypre_ParCSRMatrixDestroy(AN);
               hypre_ParCSRMatrixDestroy(SN);
            }

         }
         else if (coarsen_type == 10) /*hmis */
         {
            if (nodal == 0) /* nonsystems or unknown approach for systems*/
            {
               hypre_BoomerAMGCoarsenHMIS(S, A_array[level], measure_type,
                                          debug_flag, &CF_marker);
               if (level < agg_num_levels && (agg_interp_type < 30 || agg_interp_type > 32))
               {
                  /* set num_functions 1 in CoarseParms, since coarse_dof_func
                     is not needed here */
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             1, dof_func_array[level], CF_marker,
                                             &coarse_dof_func,&coarse_pnts_global1);
                  hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                             coarse_pnts_global1, &S2);
                  hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type,
                                             debug_flag, &CFN_marker);
                  hypre_ParCSRMatrixDestroy(S2);
                  hypre_BoomerAMGCorrectCFMarker (CF_marker, local_num_vars, CFN_marker);
                  hypre_TFree(CFN_marker);
                  hypre_TFree(coarse_pnts_global1);
               }
            }
            else if (nodal < 0 || block_mode ) /*           nodal interpolation
                                                        or if nodal < 0 then we 
                                                        build interpolation normally 
                                                        using the nodal matrix */
            {
               hypre_BoomerAMGCoarsenHMIS(SN, SN, measure_type,
                                          debug_flag, &CF_marker);
            }
            else if (nodal > 0) /* nodal = 1,2,3: here we convert our nodal coarsening 
                                  so that we can perform regular interpolation */
            {
               hypre_BoomerAMGCoarsenHMIS(SN, SN, measure_type,
                                          debug_flag, &CFN_marker);
               /*hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker,
                 num_functions, nodal, 0, NULL, &CF_marker, &S);*/
               if (level < agg_num_levels)
               {
                /* set num_functions 1 in CoarseParms, since coarse_dof_func
                 is not needed here */
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                        1, dof_func_array[level], CFN_marker,
                        &coarse_dof_func,&coarse_pnts_global1);
                  hypre_BoomerAMGCreate2ndS (SN, CFN_marker, num_paths,
                                coarse_pnts_global1, &S2);
                  hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type,
                                    debug_flag, &CF2_marker);
                  hypre_ParCSRMatrixDestroy(S2);
                  hypre_BoomerAMGCorrectCFMarker (CFN_marker, local_num_vars, CF2_marker);
                  hypre_TFree(CF2_marker);
                  hypre_TFree(coarse_pnts_global1);
               }
               col_offd_S_to_A = NULL;
               hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker, col_offd_SN_to_AN,
                                              num_functions, nodal, 0, NULL, &CF_marker, 
                                              &col_offd_S_to_A, &S);
               if (col_offd_SN_to_AN == NULL)
                  col_offd_S_to_A = NULL;
               hypre_TFree(CFN_marker);
               hypre_TFree(col_offd_SN_to_AN);
               hypre_ParCSRMatrixDestroy(AN);
               hypre_ParCSRMatrixDestroy(SN);
            }

         }
         
         else if (coarsen_type) /* ruge, ruge1p, ruge2b, ruge3, ruge3c, ruge1x */
         {
            if (nodal == 0) /* nonsystems or unknown approach for systems*/
            {
               hypre_BoomerAMGCoarsenRuge(S, A_array[level],
                                          measure_type, coarsen_type, debug_flag,
                                          &CF_marker);
               if (level < agg_num_levels && (agg_interp_type < 30 || agg_interp_type > 32))
               {
                  /* set num_functions 1 in CoarseParms, since coarse_dof_func
		 is not needed here */
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             1, dof_func_array[level], CF_marker,
                                             &coarse_dof_func,&coarse_pnts_global1);
                  hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                             coarse_pnts_global1, &S2);
                  hypre_BoomerAMGCoarsenRuge(S2, S2, measure_type, coarsen_type, 
                                             debug_flag, &CFN_marker);
                  hypre_ParCSRMatrixDestroy(S2);
                  hypre_BoomerAMGCorrectCFMarker (CF_marker, local_num_vars, CFN_marker);
                  hypre_TFree(CFN_marker);
                  hypre_TFree(coarse_pnts_global1);
               }
            }
            else if (nodal < 0 || block_mode ) /*        nodal interpolation
                                                        or if nodal < 0 then we 
                                                        build interpolation normally 
                                                        using the nodal matrix */
           {
             hypre_BoomerAMGCoarsenRuge(SN, SN,
                                 measure_type, coarsen_type, debug_flag,
                                 &CF_marker);
           }
           else if (nodal > 0) /* nodal = 1,2,3: here we convert our nodal coarsening 
                                  so that we can perform regualr interpolation */
           {
             hypre_BoomerAMGCoarsenRuge(SN, SN,
                                 measure_type, coarsen_type, debug_flag,
                                 &CFN_marker);
             /*hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker,
                             num_functions, 0, &CF_marker, &S);*/
             /*hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker,
                             num_functions, nodal, 0, NULL, &CF_marker, &S);*/
	     col_offd_S_to_A = NULL;
             hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker, col_offd_SN_to_AN,
                             num_functions, nodal, 0, NULL, &CF_marker, 
			     &col_offd_S_to_A, &S);
             if (col_offd_SN_to_AN == NULL)
	        col_offd_S_to_A = NULL;
             hypre_TFree(CFN_marker);
             hypre_TFree(col_offd_SN_to_AN);
             hypre_ParCSRMatrixDestroy(AN);
             hypre_ParCSRMatrixDestroy(SN);
           }
         }
         else /* coarsen_type = 0 (or anything negative) - cljp */
         {
          
            if (nodal == 0) /* nonsystems or unknown approach for systems*/
            {
               
               hypre_BoomerAMGCoarsen(S, A_array[level], 0,
                                      debug_flag, &CF_marker);
               if (level < agg_num_levels && (agg_interp_type < 30 || agg_interp_type > 32))
               {
                  /* set num_functions 1 in CoarseParms, since coarse_dof_func
                     is not needed here */
                  hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                             1, dof_func_array[level], CF_marker,
                                             &coarse_dof_func,&coarse_pnts_global1);
                  hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                             coarse_pnts_global1, &S2);
                  hypre_BoomerAMGCoarsen(S2, S2, 0, debug_flag, &CFN_marker);
                  hypre_ParCSRMatrixDestroy(S2);
                  hypre_BoomerAMGCorrectCFMarker (CF_marker, local_num_vars, CFN_marker);
                  hypre_TFree(CFN_marker);
                  hypre_TFree(coarse_pnts_global1);
               }
            }
            else if (nodal < 0 || block_mode ) /*       nodal interpolation
                                                       or if nodal < 0 then we 
                                                       build interpolation normally 
                                                       using the nodal matrix */
           {
              hypre_BoomerAMGCoarsen(SN, SN, 0, debug_flag, &CF_marker);
           }
           else if (nodal > 0) /* nodal = 1,2,3: here we convert our nodal coarsening 
                                  so that we can perform regualr interpolation */
           {
             hypre_BoomerAMGCoarsen(SN, SN, 0, debug_flag, &CFN_marker);

	     col_offd_S_to_A = NULL;
             hypre_BoomerAMGCreateScalarCFS(SN, CFN_marker, col_offd_SN_to_AN,
                             num_functions, nodal, 0, NULL, &CF_marker, 
			     &col_offd_S_to_A, &S);
             if (col_offd_SN_to_AN == NULL)
	        col_offd_S_to_A = NULL;
             hypre_TFree(CFN_marker);
             hypre_TFree(col_offd_SN_to_AN);

             hypre_ParCSRMatrixDestroy(AN);
             hypre_ParCSRMatrixDestroy(SN);
           }
         }
          
         /** end of coarsen type options **/

         /* store the CF array */
         CF_marker_array[level] = CF_marker;


         if (relax_weight[level] == 0.0)
         {
	    hypre_ParCSRMatrixScaledNorm(A_array[level], &relax_weight[level]);
	    if (relax_weight[level] != 0.0)
	       relax_weight[level] = 4.0/3.0/relax_weight[level];
	    else
	       printf (" Warning ! Matrix norm is zero !!!");
         }
         if (relax_weight[level] < 0 )
         {
	    num_cg_sweeps = (int) (-relax_weight[level]);
 	    hypre_BoomerAMGCGRelaxWt(amg_data, level, num_cg_sweeps,
			&relax_weight[level]);
         }
         if (omega[level] < 0 )
         {
	    num_cg_sweeps = (int) (-omega[level]);
 	    hypre_BoomerAMGCGRelaxWt(amg_data, level, num_cg_sweeps,
			&omega[level]);
         }
         /*if (schwarz_relax_wt < 0 )
         {
	    num_cg_sweeps = (int) (-schwarz_relax_wt);
 	    hypre_BoomerAMGCGRelaxWt(amg_data, level, num_cg_sweeps,
			&schwarz_relax_wt);*/
	    /*printf (" schwarz weight %f \n", schwarz_relax_wt);*/
	    /*HYPRE_SchwarzSetRelaxWeight(smoother[level], schwarz_relax_wt);
 	    if (hypre_ParAMGDataVariant(amg_data) > 0)
            {
               local_size = hypre_CSRMatrixNumRows
			(hypre_ParCSRMatrixDiag(A_array[level]));
 	       hypre_SchwarzReScale(smoother[level], local_size, 
				schwarz_relax_wt);
            }
	    schwarz_relax_wt = 1;
         }*/
         if (debug_flag==1)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            printf("Proc = %d    Level = %d    Coarsen Time = %f\n",
                       my_id,level, wall_time); 
	    fflush(NULL);
         }

         /**** Get the coarse parameters ****/
         if (nodal < 0 || block_mode )
         {
            /* here we will determine interpolation using a nodal matrix */  
	    hypre_BoomerAMGCoarseParms(comm,
		hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(AN)),
			1, NULL, CF_marker, NULL, &coarse_pnts_global);
         }
         else
         {
	    hypre_BoomerAMGCoarseParms(comm, local_num_vars,
			num_functions, dof_func_array[level], CF_marker,
			&coarse_dof_func,&coarse_pnts_global);
         }
         dof_func_array[level+1] = NULL;
         if (num_functions > 1 && nodal > -1 && (!block_mode) )
	    dof_func_array[level+1] = coarse_dof_func;

#ifdef HYPRE_NO_GLOBAL_PARTITION
         if (my_id == (num_procs -1)) coarse_size = coarse_pnts_global[1];
         MPI_Bcast(&coarse_size, 1, MPI_HYPRE_BIG_INT, num_procs-1, comm);
#else
	 coarse_size = coarse_pnts_global[num_procs];
#endif
      
      }
      else
      {
	 S = NULL;
	 coarse_pnts_global = NULL;
         CF_marker = hypre_CTAlloc(int, local_size );
	 for (i=0; i < local_size ; i++)
	    CF_marker[i] = 1;
         /*CF_marker_array = hypre_CTAlloc(int*, 1);*/
	 CF_marker_array[level] = CF_marker;
	 coarse_size = fine_size;
      }

      /* if no coarse-grid, stop coarsening, and set the
       * coarsest solve to be a single sweep of Jacobi */
      if ((coarse_size == 0) ||
          (coarse_size == fine_size))
      {
         int     *num_grid_sweeps =
            hypre_ParAMGDataNumGridSweeps(amg_data);
         int    **grid_relax_points =
            hypre_ParAMGDataGridRelaxPoints(amg_data);
         if (grid_relax_type[3] == 9)
	 {
	    grid_relax_type[3] = grid_relax_type[0];
	    num_grid_sweeps[3] = 1;
	    if (grid_relax_points) grid_relax_points[3][0] = 0; 
	 }
	 if (S)
            hypre_ParCSRMatrixDestroy(S);
	 hypre_TFree(coarse_pnts_global);
         if (level > 0)
         {
            /* note special case treatment of CF_marker is necessary
             * to do CF relaxation correctly when num_levels = 1 */
            hypre_TFree(CF_marker_array[level]);
            hypre_ParVectorDestroy(F_array[level]);
            hypre_ParVectorDestroy(U_array[level]);
         }

#if MPIP_SETUP_ON
      MPI_Pcontrol(3); 
      MPI_Pcontrol(0); 
#endif

         break; 
      }

      /*-------------------------------------------------------------
       * Build prolongation matrix, P, and place in P_array[level] 
       *--------------------------------------------------------------*/

      if (debug_flag==1) wall_time = time_getWallclockSeconds();

      if (hypre_ParAMGDataAggInterpType(amg_data) == 30 && (level < agg_num_levels
                                                        && (!block_mode)))
      {
          hypre_BoomerAMGBuildExtPIInterp(A_array[level], CF_marker_array[level],
                 S, coarse_pnts_global, num_functions, dof_func_array[level],
                 debug_flag, 0, P_max1, col_offd_S_to_A, &P1);
          hypre_TFree(col_offd_S_to_A);
         /*if (my_id == 0 && debug_flag==1)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            printf("Proc = %d    Level = %d    Interp P1 Time = %f\n",
                       my_id,level, wall_time);
            fflush(NULL);
         }
      if (my_id == 0 && debug_flag==1) wall_time = time_getWallclockSeconds();
          hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                     coarse_pnts_global, &S2);
         if (my_id == 0 && debug_flag==1)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            printf("Proc = %d    Level = %d    Create S2 Time = %f\n",
                       my_id,level, wall_time);
            fflush(NULL);
         }
      if (my_id == 0 && debug_flag==1) wall_time = time_getWallclockSeconds();*/
          hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                     coarse_pnts_global, &S2);
          if (coarsen_type == 10)
                hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 8)
                hypre_BoomerAMGCoarsenPMIS(S2, S2, 3,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 9)
                hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type+3,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 11)
                hypre_BoomerAMGCoarsenPMIS(S2, S2, 4,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 6)
                hypre_BoomerAMGCoarsenFalgout(S2, S2, measure_type,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 0)
                hypre_BoomerAMGCoarsen(S2, S2, 0, debug_flag, &CFN_marker);
          hypre_ParCSRMatrixDestroy(S2);
          hypre_BoomerAMGCorrectCFMarker2 (CF_marker, local_num_vars, CFN_marker);
          hypre_TFree(CFN_marker);
          hypre_TFree(coarse_dof_func);
          coarse_dof_func = NULL;
          hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                        num_functions, dof_func_array[level], CF_marker,
                        &coarse_dof_func,&coarse_pnts_global1);
          if (num_functions > 1 && nodal > -1 && (!block_mode) )
            dof_func_array[level+1] = coarse_dof_func;
         /*if (my_id == 0 && debug_flag==1)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            printf("Proc = %d    Level = %d    Coarsen 2 Time = %f\n",
                       my_id,level, wall_time);
            fflush(NULL);
         }
      if (my_id == 0 && debug_flag==1) wall_time = time_getWallclockSeconds();*/
          hypre_BoomerAMGBuildPartialExtPIInterp(A_array[level], CF_marker_array[level],
                 S, coarse_pnts_global1, coarse_pnts_global, num_functions, dof_func_array[level],
                 debug_flag, 0, P_max2, col_offd_S_to_A, &P2);
         /*if (my_id == 0 && debug_flag==1)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            printf("Proc = %d    Level = %d    Interp P2 Time = %f\n",
                       my_id,level, wall_time);
            fflush(NULL);
         }
      if (my_id == 0 && debug_flag==1) wall_time = time_getWallclockSeconds(); */
          P = hypre_ParMatmul(P1,P2);
        /*if (my_id == 0 && debug_flag==1)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            printf("Proc = %d    Level = %d    P1*P2 Time = %f\n",
                       my_id,level, wall_time);
            fflush(NULL);
         }
      if (my_id == 0 && debug_flag==1) wall_time = time_getWallclockSeconds();*/
          hypre_BoomerAMGInterpTruncation(P,agg_trunc_factor,agg_P_max_elmts);
          hypre_ParCSRMatrixDestroy(P1);
          hypre_ParCSRMatrixOwnsColStarts(P2) = 0;
          hypre_ParCSRMatrixDestroy(P2);
          hypre_ParCSRMatrixOwnsColStarts(P) = 1;
          coarse_pnts_global = coarse_pnts_global1;
#ifdef HYPRE_NO_GLOBAL_PARTITION
          hypre_NewCommPkgCreate(P);
          if (my_id == (num_procs -1)) coarse_size = coarse_pnts_global[1];
          MPI_Bcast(&coarse_size, 1, MPI_HYPRE_BIG_INT, num_procs-1, comm);
#else
          hypre_MatvecCommPkgCreate(P);
          coarse_size = coarse_pnts_global[num_procs];
#endif

      }
      else if (hypre_ParAMGDataAggInterpType(amg_data) == 31 && (level < agg_num_levels
                                                        && (!block_mode)))
      {
          hypre_BoomerAMGBuildExtInterp(A_array[level], CF_marker_array[level],
                 S, coarse_pnts_global, num_functions, dof_func_array[level],
                 debug_flag, 0, P_max1, col_offd_S_to_A, &P1);
          hypre_TFree(col_offd_S_to_A);
          hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                     coarse_pnts_global, &S2);
          if (coarsen_type == 10)
                hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 8)
                hypre_BoomerAMGCoarsenPMIS(S2, S2, 3,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 9)
                hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type+3,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 11)
                hypre_BoomerAMGCoarsenPMIS(S2, S2, 4,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 6)
                hypre_BoomerAMGCoarsenFalgout(S2, S2, measure_type,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 0)
                hypre_BoomerAMGCoarsen(S2, S2, 0, debug_flag, &CFN_marker);
          hypre_ParCSRMatrixDestroy(S2);
          hypre_BoomerAMGCorrectCFMarker2 (CF_marker, local_num_vars, CFN_marker);
          hypre_TFree(CFN_marker);
          hypre_TFree(coarse_dof_func);
          coarse_dof_func = NULL;
          hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                        num_functions, dof_func_array[level], CF_marker,
                        &coarse_dof_func,&coarse_pnts_global1);
          if (num_functions > 1 && nodal > -1 && (!block_mode) )
            dof_func_array[level+1] = coarse_dof_func;
         /*if (my_id == 0 && debug_flag==1)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            printf("Proc = %d    Level = %d    Coarsen 2 Time = %f\n",
                       my_id,level, wall_time);
            fflush(NULL);
         }
      if (my_id == 0 && debug_flag==1) wall_time = time_getWallclockSeconds();*/
          hypre_BoomerAMGBuildPartialExtInterp(A_array[level], CF_marker_array[level],
                 S, coarse_pnts_global1, coarse_pnts_global, num_functions, dof_func_array[level],
                 debug_flag, 0, P_max2, col_offd_S_to_A, &P2);
         /*if (my_id == 0 && debug_flag==1)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            printf("Proc = %d    Level = %d    Interp P2 Time = %f\n",
                       my_id,level, wall_time);
            fflush(NULL);
         }
      if (my_id == 0 && debug_flag==1) wall_time = time_getWallclockSeconds(); */
          P = hypre_ParMatmul(P1,P2);
        /*if (my_id == 0 && debug_flag==1)
         {
            wall_time = time_getWallclockSeconds() - wall_time;
            printf("Proc = %d    Level = %d    P1*P2 Time = %f\n",
                       my_id,level, wall_time);
            fflush(NULL);
         }
      if (my_id == 0 && debug_flag==1) wall_time = time_getWallclockSeconds();*/
          hypre_BoomerAMGInterpTruncation(P,agg_trunc_factor,agg_P_max_elmts);
          hypre_ParCSRMatrixDestroy(P1);
          hypre_ParCSRMatrixOwnsColStarts(P2) = 0;
          hypre_ParCSRMatrixDestroy(P2);
          hypre_ParCSRMatrixOwnsColStarts(P) = 1;
          coarse_pnts_global = coarse_pnts_global1;
#ifdef HYPRE_NO_GLOBAL_PARTITION
          hypre_NewCommPkgCreate(P);
          if (my_id == (num_procs -1)) coarse_size = coarse_pnts_global[1];
          MPI_Bcast(&coarse_size, 1, MPI_HYPRE_BIG_INT, num_procs-1, comm);
#else
          hypre_MatvecCommPkgCreate(P);
          coarse_size = coarse_pnts_global[num_procs];
#endif

      }
      else if (hypre_ParAMGDataAggInterpType(amg_data) == 32 && (level < agg_num_levels
                                                        && (!block_mode)))
      {
          hypre_BoomerAMGBuildStdInterp(A_array[level], CF_marker_array[level],
                 S, coarse_pnts_global, num_functions, dof_func_array[level],
                 debug_flag, 0, P_max1, 0, col_offd_S_to_A, &P1);
          hypre_TFree(col_offd_S_to_A);
          hypre_BoomerAMGCreate2ndS (S, CF_marker, num_paths,
                                     coarse_pnts_global, &S2);
          if (coarsen_type == 10)
                hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 8)
                hypre_BoomerAMGCoarsenPMIS(S2, S2, 3,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 9)
                hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type+3,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 11)
                hypre_BoomerAMGCoarsenPMIS(S2, S2, 4,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 6)
                hypre_BoomerAMGCoarsenFalgout(S2, S2, measure_type,
                                     debug_flag, &CFN_marker);
          else if (coarsen_type == 0)
                hypre_BoomerAMGCoarsen(S2, S2, 0, debug_flag, &CFN_marker);
          hypre_ParCSRMatrixDestroy(S2);
          hypre_BoomerAMGCorrectCFMarker2 (CF_marker, local_num_vars, CFN_marker);
          hypre_TFree(CFN_marker);
          hypre_TFree(coarse_dof_func);
          coarse_dof_func = NULL;
          hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                        num_functions, dof_func_array[level], CF_marker,
                        &coarse_dof_func,&coarse_pnts_global1);
          if (num_functions > 1 && nodal > -1 && (!block_mode) )
            dof_func_array[level+1] = coarse_dof_func;
          hypre_BoomerAMGBuildPartialStdInterp(A_array[level], CF_marker_array[level],
                 S, coarse_pnts_global1, coarse_pnts_global, num_functions, dof_func_array[level],
                 debug_flag, 0, P_max2, 0, col_offd_S_to_A, &P2);
          P = hypre_ParMatmul(P1,P2);
          hypre_BoomerAMGInterpTruncation(P,agg_trunc_factor,agg_P_max_elmts);
          hypre_ParCSRMatrixDestroy(P1);
          hypre_ParCSRMatrixOwnsColStarts(P2) = 0;
          hypre_ParCSRMatrixDestroy(P2);
          hypre_ParCSRMatrixOwnsColStarts(P) = 1;
          coarse_pnts_global = coarse_pnts_global1;
#ifdef HYPRE_NO_GLOBAL_PARTITION
         hypre_NewCommPkgCreate(P);
         if (my_id == (num_procs -1)) coarse_size = coarse_pnts_global[1];
         MPI_Bcast(&coarse_size, 1, MPI_HYPRE_BIG_INT, num_procs-1, comm);
#else
         hypre_MatvecCommPkgCreate(P);
         coarse_size = coarse_pnts_global[num_procs];
#endif

      }
      else if ((hypre_ParAMGDataAggInterpType(amg_data) == 4 && 
	(level < agg_num_levels && (!block_mode)))||
	(hypre_ParAMGDataInterpType(amg_data) == 4 && 
	level >= agg_num_levels))
      {
	 if (nodal > -1)
	 {
            hypre_BoomerAMGBuildMultipass(A_array[level], CF_marker_array[level], 
                 S, coarse_pnts_global, num_functions, dof_func_array[level], 
		 debug_flag, agg_trunc_factor, agg_P_max_elmts, 0, col_offd_S_to_A, &P);
	    hypre_TFree(col_offd_S_to_A);
         }
         else
         {
            CFN_marker = CF_marker_array[level];
            hypre_BoomerAMGBuildMultipass(AN, CFN_marker,
                 SN, coarse_pnts_global, 1, NULL,
		 debug_flag, agg_trunc_factor, agg_P_max_elmts, 0, col_offd_SN_to_AN, &PN);
	    hypre_TFree(col_offd_SN_to_AN);
            hypre_BoomerAMGCreateScalarCFS(PN, CFN_marker, NULL,
                num_functions, nodal, 1, &dof_func1,
                &CF_marker, NULL, &P);
            dof_func_array[level+1] = dof_func1;
            CF_marker_array[level] = CF_marker;
            hypre_TFree (CFN_marker);
            hypre_ParCSRMatrixDestroy(AN);
            hypre_ParCSRMatrixDestroy(PN);
            hypre_ParCSRMatrixDestroy(SN);
         }
      }
      else if (hypre_ParAMGDataInterpType(amg_data) == 5)
      {
	 if (nodal > -1)
	 {
            hypre_BoomerAMGBuildMultipass(A_array[level], CF_marker_array[level], 
                 S, coarse_pnts_global, num_functions, dof_func_array[level], 
		 debug_flag, trunc_factor, P_max_elmts, 1, col_offd_S_to_A, &P);
	    hypre_TFree(col_offd_S_to_A);
         }
         else
         {
            CFN_marker = CF_marker_array[level];
            hypre_BoomerAMGBuildMultipass(AN, CFN_marker,
                 SN, coarse_pnts_global, 1, NULL,
		 debug_flag, trunc_factor, P_max_elmts, 1, col_offd_SN_to_AN, &PN);
	    hypre_TFree(col_offd_SN_to_AN);
            hypre_BoomerAMGCreateScalarCFS(PN, CFN_marker, NULL,
                num_functions, nodal, 1, &dof_func1,
                &CF_marker, NULL, &P);
            dof_func_array[level+1] = dof_func1;
            CF_marker_array[level] = CF_marker;
            hypre_TFree (CFN_marker);
            hypre_ParCSRMatrixDestroy(AN);
            hypre_ParCSRMatrixDestroy(PN);
            hypre_ParCSRMatrixDestroy(SN);
         }
      }
      else if (hypre_ParAMGDataInterpType(amg_data) == 6) /*Extended+i interpolation */
      {
          hypre_BoomerAMGBuildExtPIInterp(A_array[level], CF_marker_array[level], 
                 S, coarse_pnts_global, num_functions, dof_func_array[level], 
		 debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
	  hypre_TFree(col_offd_S_to_A);
      }
      else if (hypre_ParAMGDataInterpType(amg_data) == 7) /*Extended+i interpolation (common C only)*/
      {
          hypre_BoomerAMGBuildExtPICCInterp(A_array[level], CF_marker_array[level], 
                 S, coarse_pnts_global, num_functions, dof_func_array[level], 
		 debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
	  hypre_TFree(col_offd_S_to_A);
      }
      /*else if (hypre_ParAMGDataInterpType(amg_data) == 12)*/ /*FF interpolation  */
      /*{
          hypre_BoomerAMGBuildFFInterp(A_array[level], CF_marker_array[level], 
                 S, coarse_pnts_global, num_functions, dof_func_array[level], 
		 debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
	  hypre_TFree(col_offd_S_to_A);
      }*/
      else if (hypre_ParAMGDataInterpType(amg_data) == 13) /*FF1 interpolation */
      {
          hypre_BoomerAMGBuildFF1Interp(A_array[level], CF_marker_array[level], 
                 S, coarse_pnts_global, num_functions, dof_func_array[level], 
		 debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
	  hypre_TFree(col_offd_S_to_A);
      }
      /*else if (hypre_ParAMGDataInterpType(amg_data) == 8)*/ /*Standard interpolation */
      /*{
          hypre_BoomerAMGBuildStdInterp(A_array[level], CF_marker_array[level], 
                 S, coarse_pnts_global, num_functions, dof_func_array[level], 
		 debug_flag, trunc_factor, P_max_elmts, 0, col_offd_S_to_A, &P);
	  hypre_TFree(col_offd_S_to_A);
      }
      else if (hypre_ParAMGDataInterpType(amg_data) == 9) */ /*Standard interpolation with separation of
							   negative and positive weights*/
      /*{
          hypre_BoomerAMGBuildStdInterp(A_array[level], CF_marker_array[level], 
                 S, coarse_pnts_global, num_functions, dof_func_array[level], 
		 debug_flag, trunc_factor, P_max_elmts, 1, col_offd_S_to_A, &P);
	  hypre_TFree(col_offd_S_to_A);
      }*/
      else if (hypre_ParAMGDataGSMG(amg_data) == 0)
      {
         /* no nodal interp OR nodal =0 */
         {
            if (nodal > -1) /* 0, 1, 2, 3  - unknown approach for interpolation */
            {

               hypre_BoomerAMGBuildInterp(A_array[level], CF_marker_array[level], 
                                          S, coarse_pnts_global, num_functions, 
                                          dof_func_array[level], 
                                          debug_flag, trunc_factor, P_max_elmts, col_offd_S_to_A, &P);
  
               hypre_TFree(col_offd_S_to_A);
            }
            else /* -1, -2, -3:  here we build interp using the nodal matrix and then convert 
                    to regular size */
            {
               CFN_marker = CF_marker_array[level];
               hypre_BoomerAMGBuildInterp(AN, CFN_marker,
                                          SN, coarse_pnts_global, 1, NULL,
                                          debug_flag, trunc_factor, P_max_elmts, col_offd_SN_to_AN, &PN);
               hypre_TFree(col_offd_SN_to_AN);
               hypre_BoomerAMGCreateScalarCFS(PN, CFN_marker, NULL,
                                              num_functions, nodal, 1, &dof_func1,
                                              &CF_marker, NULL, &P);
               dof_func_array[level+1] = dof_func1;
               CF_marker_array[level] = CF_marker;
               hypre_TFree (CFN_marker);
               hypre_ParCSRMatrixDestroy(AN);
               hypre_ParCSRMatrixDestroy(PN);
               hypre_ParCSRMatrixDestroy(SN);
            }
         }
      }

      /*if ( post_interp_type>=1 && level < agg_num_levels)*/
      for (i=0; i < post_interp_type; i++)
         /* Improve on P with Jacobi interpolation */
         hypre_BoomerAMGJacobiInterp( A_array[level], &P, S,
                                      num_functions, dof_func,
                                      CF_marker_array[level],
                                      level, jacobi_trunc_threshold, 0.5*jacobi_trunc_threshold );


      if (debug_flag==1)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    Level = %d    Build Interp Time = %f\n",
                     my_id,level, wall_time);
         fflush(NULL);
      }

      if (!block_mode)
      {
         P_array[level] = P; 
      }
      
      if (S) hypre_ParCSRMatrixDestroy(S);
      S = NULL;


      /*-------------------------------------------------------------
       * Build coarse-grid operator, A_array[level+1] by R*A*P
       *--------------------------------------------------------------*/

      if (debug_flag==1) wall_time = time_getWallclockSeconds();

         
      hypre_BoomerAMGBuildCoarseOperator(P_array[level], A_array[level] , 
                                            P_array[level], &A_H);
 
      if (debug_flag==1)
      {
         wall_time = time_getWallclockSeconds() - wall_time;
         printf("Proc = %d    Level = %d    Build Coarse Operator Time = %f\n",
                       my_id,level, wall_time);
	 fflush(NULL);
      }

      ++level;

      if (!block_mode)
      {
         hypre_ParCSRMatrixSetNumNonzeros(A_H);
         hypre_ParCSRMatrixSetDNumNonzeros(A_H);
         A_array[level] = A_H;
      }
      
      size = ((double) fine_size )*.75;
      if (coarsen_type > 0 && coarse_size >= (HYPRE_BigInt) size)
      {
	coarsen_type = 0;      
      }
      
      {
         
         int thresh = hypre_max(coarse_threshold, seq_threshold);
      
         //stop coarsening ?
         if ( (level == max_levels-1) || (coarse_size <= (HYPRE_BigInt) thresh) )
         {
            not_finished_coarsening = 0;
         }

      }
      
#if MPIP_SETUP_ON
      MPI_Pcontrol(3); 
      MPI_Pcontrol(0); 
#endif

   } /* end of coarsening */ 

   //more seq coarsening?
   if (  (seq_threshold > coarse_threshold) && (coarse_size > coarse_threshold) && (level != max_levels-1))
   {
      hypre_seqAMGSetup( amg_data, level, coarse_threshold);
   }



   if (level > 0)
   {
         F_array[level] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                  hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                  hypre_ParCSRMatrixRowStarts(A_array[level]));
         hypre_ParVectorInitialize(F_array[level]);
         hypre_ParVectorSetPartitioningOwner(F_array[level],0);
         
         U_array[level] =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[level]),
                                  hypre_ParCSRMatrixGlobalNumRows(A_array[level]),
                                  hypre_ParCSRMatrixRowStarts(A_array[level]));
         hypre_ParVectorInitialize(U_array[level]);
         hypre_ParVectorSetPartitioningOwner(U_array[level],0);
   }
   
   /*-----------------------------------------------------------------------
    * enter all the stuff created, A[level], P[level], CF_marker[level],
    * for levels 1 through coarsest, into amg_data data structure
    *-----------------------------------------------------------------------*/

   num_levels = level+1;
   hypre_ParAMGDataNumLevels(amg_data) = num_levels;

   
   /*-----------------------------------------------------------------------
    * Setup F and U arrays
    *-----------------------------------------------------------------------*/
   
   if (grid_relax_type[1] == 8 || grid_relax_type[1] == 19 || grid_relax_type[1] == 20 )
   {
      l1_norms = hypre_CTAlloc(double *, num_levels);
      hypre_ParAMGDataL1Norms(amg_data) = l1_norms;
      
      Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize(Ztemp);
      hypre_ParVectorSetPartitioningOwner(Ztemp,0);
      hypre_ParAMGDataZtemp(amg_data) = Ztemp;
   }
   else if (grid_relax_type[1] == 18)
   {
      l1_norms = hypre_CTAlloc(double *, num_levels);
      hypre_ParAMGDataL1Norms(amg_data) = l1_norms;
   }
   
   if (grid_relax_type[1] == 11 || grid_relax_type[1] == 15 
       || grid_relax_type[1] == 16  || grid_relax_type[1] == 17 ) /* Chebyshev */
   {
      max_eig_est = hypre_CTAlloc(double, num_levels);
      min_eig_est = hypre_CTAlloc(double, num_levels);
      hypre_ParAMGDataMaxEigEst(amg_data) = max_eig_est;
      hypre_ParAMGDataMinEigEst(amg_data) = min_eig_est;

      Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                 hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                 hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize(Ztemp);
      hypre_ParVectorSetPartitioningOwner(Ztemp,0);
      hypre_ParAMGDataZtemp(amg_data) = Ztemp;

   }
   if (grid_relax_type[1] == 13 || grid_relax_type[3] == 13 ||
       grid_relax_type[1] == 14 || grid_relax_type[3] == 14 ) /* CG */
   {
      smoother = hypre_CTAlloc(HYPRE_Solver, num_levels);
      hypre_ParAMGDataSmoother(amg_data) = smoother;
      if (grid_relax_type[1] == 14 || grid_relax_type[3] == 14)
      {
         smoother_prec = hypre_CTAlloc(HYPRE_Solver, num_levels);
         hypre_ParAMGDataSmootherPrec(amg_data) = smoother_prec;
         
      }
   }
   if (     grid_relax_type[1] == 3 || 
            grid_relax_type[1] == 6) /* GS (3 and 6) all need a temp vector*/
   {
      Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                                    hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                    hypre_ParCSRMatrixRowStarts(A_array[0]));
      hypre_ParVectorInitialize(Ztemp);
      hypre_ParVectorSetPartitioningOwner(Ztemp,0);
      hypre_ParAMGDataZtemp(amg_data) = Ztemp;
   }
   for (j = 0; j < num_levels; j++)
   {
    if (num_threads == 1)
    {
      if ((grid_relax_type[1] == 8 ) && j < num_levels-1)
      {
         hypre_ParCSRComputeL1Norms(A_array[j], 4, NULL, &l1_norms[j]);
      }
      else if ((grid_relax_type[3] == 8 ) && j == num_levels-1)
      {
         hypre_ParCSRComputeL1Norms(A_array[j], 4, NULL, &l1_norms[j]);
      }
      if (grid_relax_type[1] == 18 && j < num_levels-1)
      {
         if (hypre_ParAMGDataRelaxOrder(amg_data))
            hypre_ParCSRComputeL1Norms(A_array[j], 4, CF_marker_array[j], &l1_norms[j]);
         else
            hypre_ParCSRComputeL1Norms(A_array[j], 1, NULL, &l1_norms[j]);
      }
      else if (grid_relax_type[3] == 18 && j == num_levels-1)
      {
         hypre_ParCSRComputeL1Norms(A_array[j], 1, NULL, &l1_norms[j]);
      }
    }
    else
    {
      if ((grid_relax_type[1] == 8 ) && j < num_levels-1)
      {
         hypre_ParCSRComputeL1NormsThreads(A_array[j], 4, num_threads, NULL, &l1_norms[j]);
      }
      else if ((grid_relax_type[3] == 8 ) && j == num_levels-1)
      {
         hypre_ParCSRComputeL1NormsThreads(A_array[j], 4, num_threads, NULL, &l1_norms[j]);
      }
      if (grid_relax_type[1] == 18 && j < num_levels-1)
      {
         if (hypre_ParAMGDataRelaxOrder(amg_data))
            hypre_ParCSRComputeL1NormsThreads(A_array[j], 4, num_threads, CF_marker_array[j], &l1_norms[j]);
         else
            hypre_ParCSRComputeL1NormsThreads(A_array[j], 1, num_threads, NULL, &l1_norms[j]);
      }
      else if (grid_relax_type[3] == 18 && j == num_levels-1)
      {
         hypre_ParCSRComputeL1NormsThreads(A_array[j], 1, num_threads, NULL, &l1_norms[j]);
      }
    }
      if (grid_relax_type[1] == 11 || grid_relax_type[1] == 15   
          || grid_relax_type[1] == 16  || grid_relax_type[1] == 17 )
      {
         int scale = 0;
         double temp_d, temp_d2;
         /* hypre_ParCSRMaxEigEstimate(A_array[j], 0, &temp_d);*/
         if (grid_relax_type[1] == 16  || grid_relax_type[1] == 17)
            scale = 1;

         hypre_ParCSRMaxEigEstimateCG(A_array[j], scale, 10, &temp_d, &temp_d2);
         /* if (my_id ==0) printf("cg est: max =  %g, min = %g\n", temp_d, temp_d2); */
         
         max_eig_est[j] = temp_d;
         min_eig_est[j] = temp_d2;
      }
      
      if (grid_relax_type[1] == 13 || (grid_relax_type[3] == 13 && j == (num_levels-1))  )
      {
         
         HYPRE_ParCSRPCGCreate(comm, &smoother[j]);
         HYPRE_ParCSRPCGSetup(smoother[j],
                             (HYPRE_ParCSRMatrix) A_array[j],
                             (HYPRE_ParVector) F_array[j],
                             (HYPRE_ParVector) U_array[j]);

         HYPRE_PCGSetTol(smoother[j], 1e-9); /* do a fixed number of iterations, so the
                                                 conv. tolerance is small*/
         HYPRE_PCGSetTwoNorm(smoother[j], 1); /* use 2-norm*/

         /* HYPRE_PCGSetLogging(smoother[j], 1);*/ /* needed to get run info later */

         /* Do we want diagonal scaling? */
         /*
           HYPRE_PCGSetPrecond(smoother[j],
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                             (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                             NULL);
         */
         





         HYPRE_ParCSRPCGSetup(smoother[j], 
                              (HYPRE_ParCSRMatrix) A_array[j], 
                              (HYPRE_ParVector) F_array[j], 
                              (HYPRE_ParVector) U_array[j]);
  
         
      }

      if (grid_relax_type[1] == 14 || (grid_relax_type[3] == 14 && j == (num_levels-1))  )
      {
         
         HYPRE_ParCSRPCGCreate(comm, &smoother[j]);
         HYPRE_ParCSRPCGSetup(smoother[j],
                             (HYPRE_ParCSRMatrix) A_array[j],
                             (HYPRE_ParVector) F_array[j],
                             (HYPRE_ParVector) U_array[j]);

         HYPRE_PCGSetTol(smoother[j], 1e-9); /* want to do a fixed number of iterations, so the
                                                 conv. tolerance is very small*/
         HYPRE_PCGSetTwoNorm(smoother[j], 1); /* use 2-norm*/

         HYPRE_BoomerAMGCreate(&smoother_prec[j]); 
         HYPRE_BoomerAMGSetCoarsenType(smoother_prec[j], 10); /* hmis*/
         HYPRE_BoomerAMGSetInterpType(smoother_prec[j], 6); /* ext+i (5) */
         HYPRE_BoomerAMGSetPMaxElmts(smoother_prec[j], 5);
         HYPRE_BoomerAMGSetRelaxType(smoother_prec[j], 8); /* L1-SGS */
         HYPRE_BoomerAMGSetMaxIter(smoother_prec[j], 1);
         
         HYPRE_BoomerAMGSetup(smoother_prec[j], 
                              (HYPRE_ParCSRMatrix) A_array[j], 
                              (HYPRE_ParVector) F_array[j], 
                              (HYPRE_ParVector) U_array[j]);


         HYPRE_PCGSetPrecond( smoother[j],
                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                              smoother_prec[j]);


         HYPRE_ParCSRPCGSetup(smoother[j], 
                              (HYPRE_ParCSRMatrix) A_array[j], 
                              (HYPRE_ParVector) F_array[j], 
                              (HYPRE_ParVector) U_array[j]);
  
         
      }


   }
   
   if ( amg_logging > 1 ) {

      Residual_array= 
	hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_array[0]),
                              hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                              hypre_ParCSRMatrixRowStarts(A_array[0]) );
      hypre_ParVectorInitialize(Residual_array);
      hypre_ParVectorSetPartitioningOwner(Residual_array,0);
      hypre_ParAMGDataResidual(amg_data) = Residual_array;
   }
   else
      hypre_ParAMGDataResidual(amg_data) = NULL;

   /*-----------------------------------------------------------------------
    * Print some stuff
    *-----------------------------------------------------------------------*/

   if (amg_print_level == 1 || amg_print_level == 3)
      hypre_BoomerAMGSetupStats(amg_data,A);


/* print out matrices on all levels  */
#if 0
{
   char  filename[256];
      
      for (level = 0; level < num_levels; level++)
      {
         sprintf(filename, "BoomerAMG.out.A.%02d", level);
         hypre_ParCSRMatrixPrint(A_array[level], filename);
            
      }
}

#endif



   return(Setup_err_flag);
}  
