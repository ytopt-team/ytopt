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

#define MPIP_CYCLE_ON 0



/******************************************************************************
 *
 * ParAMG cycling routine
 *
 *****************************************************************************/

#include "headers.h"
#include "par_amg.h"
/*--------------------------------------------------------------------------
 * hypre_BoomerAMGCycle
 *--------------------------------------------------------------------------*/

int
hypre_BoomerAMGCycle( void              *amg_vdata, 
                   hypre_ParVector  **F_array,
                   hypre_ParVector  **U_array   )
{
   hypre_ParAMGData *amg_data = amg_vdata;

   hypre_SeqAMGData *seq_data = hypre_ParAMGDataSeqData(amg_data);

   MPI_Comm comm;
   HYPRE_Solver *smoother;
   /* Data Structure variables */

   hypre_ParCSRMatrix    **A_array;
   hypre_ParCSRMatrix    **P_array;
   hypre_ParCSRMatrix    **R_array;
   /*hypre_ParVector    *Utemp;*/
   hypre_ParVector    *Vtemp;
   hypre_ParVector    *Rtemp;
   hypre_ParVector    *Ptemp;
   hypre_ParVector    *Ztemp;
   hypre_ParVector    *Aux_U;
   hypre_ParVector    *Aux_F;

   int     **CF_marker_array;
   /* int     **unknown_map_array;
   int     **point_map_array;
   int     **v_at_point_array; */

   double    cycle_op_count;   
   int       cycle_type;
   int       num_levels;
   int       max_levels;

   double   *num_coeffs;
   int      *num_grid_sweeps;   
   int      *grid_relax_type;   
   int     **grid_relax_points;  

   /*int     block_mode;*/

   double  *max_eig_est;
   double  *min_eig_est;
   int      cheby_order;
   double   cheby_eig_ratio;
   

 /* Local variables  */ 
   int      *lev_counter;
   int       Solve_err_flag;
   int       k;
   int       j, jj;
   int       level;
   int       cycle_param;
   int       coarse_grid;
   int       fine_grid;
   int       Not_Finished;
   int       num_sweep;
   int       cg_num_sweep = 1;
   int       relax_type;
   int       relax_points;
   int       relax_order;
   int       relax_local;
   int       old_version = 0;
   double   *relax_weight;
   double   *omega;
   double    beta;
   int       local_size;

   double    alpha;
   double  **l1_norms = NULL;
   double   *l1_norms_level;

#if 0
   double   *D_mat;
   double   *S_vec;
#endif
   

   int myid;
   int seq_cg = 0;
   
   if (seq_data)
      seq_cg = 1;
   

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   /* Acquire data and allocate storage */

   A_array           = hypre_ParAMGDataAArray(amg_data);
   P_array           = hypre_ParAMGDataPArray(amg_data);
   R_array           = hypre_ParAMGDataRArray(amg_data);
   CF_marker_array   = hypre_ParAMGDataCFMarkerArray(amg_data);
   Vtemp             = hypre_ParAMGDataVtemp(amg_data);
   Rtemp             = hypre_ParAMGDataRtemp(amg_data);
   Ptemp             = hypre_ParAMGDataPtemp(amg_data);
   Ztemp             = hypre_ParAMGDataZtemp(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   max_levels        = hypre_ParAMGDataMaxLevels(amg_data);
   cycle_type        = hypre_ParAMGDataCycleType(amg_data);

   num_grid_sweeps     = hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type     = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points   = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_order         = hypre_ParAMGDataRelaxOrder(amg_data);
   relax_weight        = hypre_ParAMGDataRelaxWeight(amg_data); 
   omega               = hypre_ParAMGDataOmega(amg_data); 
   l1_norms            = hypre_ParAMGDataL1Norms(amg_data);

   max_eig_est = hypre_ParAMGDataMaxEigEst(amg_data);
   min_eig_est = hypre_ParAMGDataMinEigEst(amg_data);
   cheby_order = hypre_ParAMGDataChebyOrder(amg_data);
   cheby_eig_ratio = hypre_ParAMGDataChebyEigRatio(amg_data);

   cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

   lev_counter = hypre_CTAlloc(int, num_levels);

   /* Initialize */

   Solve_err_flag = 0;

   if (grid_relax_points) old_version = 1;

   num_coeffs = hypre_CTAlloc(double, num_levels);
   num_coeffs[0]    = hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
   comm = hypre_ParCSRMatrixComm(A_array[0]);

   for (j = 1; j < num_levels; j++)
         num_coeffs[j] = hypre_ParCSRMatrixDNumNonzeros(A_array[j]);
   
   
   /*---------------------------------------------------------------------
    *    Initialize cycling control counter
    *
    *     Cycling is controlled using a level counter: lev_counter[k]
    *     
    *     Each time relaxation is performed on level k, the
    *     counter is decremented by 1. If the counter is then
    *     negative, we go to the next finer level. If non-
    *     negative, we go to the next coarser level. The
    *     following actions control cycling:
    *     
    *     a. lev_counter[0] is initialized to 1.
    *     b. lev_counter[k] is initialized to cycle_type for k>0.
    *     
    *     c. During cycling, when going down to level k, lev_counter[k]
    *        is set to the max of (lev_counter[k],cycle_type)
    *---------------------------------------------------------------------*/

   Not_Finished = 1;

   smoother = hypre_ParAMGDataSmoother(amg_data);

   lev_counter[0] = 1;
   for (k = 1; k < num_levels; ++k) 
   {
      lev_counter[k] = cycle_type;
   }

   level = 0;
   cycle_param = 1;

   /*---------------------------------------------------------------------
    * Main loop of cycling
    *--------------------------------------------------------------------*/

  
   while (Not_Finished)
   {

      if (num_levels > 1) 
      {
        local_size 
            = hypre_VectorSize(hypre_ParVectorLocalVector(F_array[level]));
        hypre_VectorSize(hypre_ParVectorLocalVector(Vtemp)) = local_size;
        cg_num_sweep = 1;
        num_sweep = num_grid_sweeps[cycle_param];
        Aux_U = U_array[level];
        Aux_F = F_array[level];
        relax_type = grid_relax_type[cycle_param];
      }
      else /* do this for max levels = 1 also */
      {
        /* If no coarsening occurred, apply a simple smoother once */
        Aux_U = U_array[level];
        Aux_F = F_array[level];
        num_sweep = 1;
        relax_type = 0;
      }

      if (l1_norms != NULL)
         l1_norms_level = l1_norms[level];
      else
         l1_norms_level = NULL;


      if (cycle_param == 3 && seq_cg)
      {
         //do a seq amg solve
         hypre_seqAMGCycle(amg_data,
                           level,
                           F_array,
                           U_array);
         


      }
      else
      {
         /*------------------------------------------------------------------
          * Do the relaxation num_sweep times
          *-----------------------------------------------------------------*/
         for (jj = 0; jj < cg_num_sweep; jj++)
         {
            
            for (j = 0; j < num_sweep; j++)
            {
               if (num_levels == 1 && max_levels > 1)
               {
                  relax_points = 0;
                  relax_local = 0;
               }
               else
               {
                  if (old_version)
                     relax_points = grid_relax_points[cycle_param][j];
                  relax_local = relax_order;
               }
               
               /*-----------------------------------------------
                * VERY sloppy approximation to cycle complexity
                *-----------------------------------------------*/
               
               if (old_version && level < num_levels -1)
               {
                  switch (relax_points)
                  {
                     case 1:
                        cycle_op_count += num_coeffs[level+1];
                        break;
                        
                     case -1: 
                        cycle_op_count += (num_coeffs[level]-num_coeffs[level+1]); 
                        break;
                  }
               }
               else
               {
                  cycle_op_count += num_coeffs[level]; 
               }
               if (relax_type == 11 ||  relax_type == 15  || relax_type == 16
                   || relax_type == 17 )
                  
               { /* Chebyshev */
                  
                  int scale = 0;
                  int variant = 0;
                  
                  if (relax_type == 15 || relax_type == 17) /*modified Cheby */
                  {
                     variant = 1;
                     
                  }
                  
                  if (relax_type == 16 || relax_type == 17 ) /* scaled Cheby */
                  {
                     scale = 1;
                  }
                  
                  hypre_ParCSRRelax_Cheby(A_array[level], 
                                          Aux_F,
                                          max_eig_est[level],     
                                          min_eig_est[level],     
                                          cheby_eig_ratio, cheby_order, scale,
                                          variant, Aux_U, Vtemp, Ztemp );
               }
               
               else if (relax_type ==12)
               {
                  hypre_BoomerAMGRelax_FCFJacobi(A_array[level], 
                                                 Aux_F,
                                                 CF_marker_array[level],
                                                 relax_weight[level],
                                                 Aux_U,
                                                 Vtemp);
               }
               else if (relax_type == 13 || relax_type == 14)
               {
                  if (j ==0) /* do num sweep iterations of CG */
                     hypre_ParCSRRelax_CG( smoother[level],
                                           A_array[level], 
                                           Aux_F,      
                                           Aux_U,
                                           num_sweep);
               }
               else if (relax_type == 8)
               {
                  hypre_ParCSRRelax_L1(A_array[level],
                                       Aux_F,
                                       relax_weight[level],
                                       omega[level],
                                       l1_norms_level,
                                       Aux_U,
                                       Vtemp, 
                                       Ztemp);
               }
               else if (relax_type == 19)
               {
                  hypre_ParCSRRelax_L1_GS(A_array[level],
                                          Aux_F,
                                          relax_weight[level],
                                          omega[level],
                                          l1_norms_level,
                                          Aux_U,
                                          Vtemp, 
                                          Ztemp);
               }
               else if (relax_type == 18)
               {
                  if (relax_order == 1 && cycle_type < 3)
                  {
                     int i;
                     int loc_relax_points[2];
                     
                     if (cycle_type < 2)
                     {
                        loc_relax_points[0] = 1;
                        loc_relax_points[1] = -1;
                     }
                     else
                     {
                        loc_relax_points[0] = -1;
                        loc_relax_points[1] = 1;
                     }
                     for (i=0; i < 2; i++)
                        hypre_ParCSRRelax_L1_Jacobi(A_array[level],
                                                    Aux_F,
                                                    CF_marker_array[level],
                                                    loc_relax_points[i],
                                                    relax_weight[level],
                                                    l1_norms_level,
                                                    Aux_U,
                                                    Vtemp);
                  }
                  else
                  {
                     hypre_ParCSRRelax_L1_Jacobi(A_array[level],
                                                 Aux_F,
                                                 CF_marker_array[level],
                                                 0,
                                                 relax_weight[level],
                                                 l1_norms_level,
                                                 Aux_U,
                                                 Vtemp);
                  }
               }
               else
               {
                  
                  if (old_version)
                  {
                     Solve_err_flag = hypre_BoomerAMGRelax(A_array[level], 
                                                           Aux_F,
                                                           CF_marker_array[level],
                                                           relax_type,
                                                           relax_points,
                                                           relax_weight[level],
                                                           omega[level],
                                                           l1_norms_level,
                                                           Aux_U,
                                                           Vtemp, Ztemp);
                  }
                  else 
                  {
                     Solve_err_flag = hypre_BoomerAMGRelaxIF(A_array[level], 
                                                             Aux_F,
                                                             CF_marker_array[level],
                                                             relax_type,
                                                             relax_local,
                                                             cycle_param,
                                                             relax_weight[level],
                                                             omega[level],
                                                             l1_norms_level,
                                                             Aux_U,
                                                             Vtemp, Ztemp);
                     
                  }
               }
               
               if (Solve_err_flag != 0)
                  return(Solve_err_flag);
            }
         }
      }//end of relaxation
      

      /*------------------------------------------------------------------
       * Decrement the control counter and determine which grid to visit next
       *-----------------------------------------------------------------*/

      --lev_counter[level];
       
      if (lev_counter[level] >= 0 && level != num_levels-1)
      {
                               
         /*---------------------------------------------------------------
          * Visit coarser level next.  
 	  * Compute residual using hypre_ParCSRMatrixMatvec.
          * Perform restriction using hypre_ParCSRMatrixMatvecT.
          * Reset counters and cycling parameters for coarse level
          *--------------------------------------------------------------*/

         fine_grid = level;
         coarse_grid = level + 1;

         hypre_ParVectorSetConstantValues(U_array[coarse_grid], 0.0); 
          
         hypre_ParVectorCopy(F_array[fine_grid],Vtemp);
         alpha = -1.0;
         beta = 1.0;

         hypre_ParCSRMatrixMatvec(alpha, A_array[fine_grid], U_array[fine_grid],
                                     beta, Vtemp);

         alpha = 1.0;
         beta = 0.0;

         hypre_ParCSRMatrixMatvecT(alpha,R_array[fine_grid],Vtemp,
                                      beta,F_array[coarse_grid]);

         ++level;
         lev_counter[level] = hypre_max(lev_counter[level],cycle_type);
         cycle_param = 1;
         if (level == num_levels-1) cycle_param = 3;
      }

      else if (level != 0)
      {
         /*---------------------------------------------------------------
          * Visit finer level next.
          * Interpolate and add correction using hypre_ParCSRMatrixMatvec.
          * Reset counters and cycling parameters for finer level.
          *--------------------------------------------------------------*/

         fine_grid = level - 1;
         coarse_grid = level;
         alpha = 1.0;
         beta = 1.0;
         hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid], 
                                     U_array[coarse_grid],
                                     beta, U_array[fine_grid]);            

         --level;
         cycle_param = 2;

      }
      else
      {
         Not_Finished = 0;
      }
      
   }


   hypre_ParAMGDataCycleOpCount(amg_data) = cycle_op_count;

   hypre_TFree(lev_counter);
   hypre_TFree(num_coeffs);

   return(Solve_err_flag);
}
