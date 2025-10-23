
#include "HYPRE_parcsr_ls.h"

#ifndef hypre_PARCSR_LS_HEADER
#define hypre_PARCSR_LS_HEADER

#include "utilities.h"
#include "krylov.h"
#include "seq_mv.h"
#include "parcsr_mv.h"
#include "par_amg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int prev; int next; } Link;


/* aux_interp.c */
int alt_insert_new_nodes ( hypre_ParCSRCommPkg *comm_pkg , hypre_ParCSRCommPkg *extend_comm_pkg , int *IN_marker , int full_off_procNodes , int *OUT_marker );
int big_insert_new_nodes ( hypre_ParCSRCommPkg *comm_pkg , hypre_ParCSRCommPkg *extend_comm_pkg , int *IN_marker , int full_off_procNodes , HYPRE_BigInt offset , HYPRE_BigInt *OUT_marker );
int hypre_ParCSRFindExtendCommPkg ( hypre_ParCSRMatrix *A , int newoff , HYPRE_BigInt *found , hypre_ParCSRCommPkg **extend_comm_pkg );
int new_offd_nodes ( HYPRE_BigInt **found , int num_cols_A_offd , int *A_ext_i , HYPRE_BigInt *big_A_ext_j , int **A_ext_j_ptr , int num_cols_S_offd , HYPRE_BigInt *col_map_offd , HYPRE_BigInt col_1 , HYPRE_BigInt col_n , int *Sop_i , HYPRE_BigInt *big_Sop_j , int **Sop_j_ptr , int *CF_marker , hypre_ParCSRCommPkg *comm_pkg );

/* gen_redcs_mat.c */
int hypre_seqAMGSetup ( hypre_ParAMGData *amg_data , int p_level , int coarse_threshold );
int hypre_seqAMGCycle ( hypre_ParAMGData *amg_data , int p_level , hypre_ParVector **Par_F_array , hypre_ParVector **Par_U_array );
int hypre_GenerateSubComm ( MPI_Comm comm , int participate , MPI_Comm *new_comm_ptr );
void hypre_merge_lists ( int *list1 , int *list2 , int *np1 , MPI_Datatype *dptr );

/* HYPRE_parcsr_amg.c */
int HYPRE_BoomerAMGCreate ( HYPRE_Solver *solver );
int HYPRE_BoomerAMGDestroy ( HYPRE_Solver solver );
int HYPRE_BoomerAMGSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_BoomerAMGSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_BoomerAMGSetRestriction ( HYPRE_Solver solver , int restr_par );
int HYPRE_BoomerAMGSetMaxLevels ( HYPRE_Solver solver , int max_levels );
int HYPRE_BoomerAMGGetMaxLevels ( HYPRE_Solver solver , int *max_levels );
int HYPRE_BoomerAMGSetStrongThreshold ( HYPRE_Solver solver , double strong_threshold );
int HYPRE_BoomerAMGGetStrongThreshold ( HYPRE_Solver solver , double *strong_threshold );
int HYPRE_BoomerAMGSetMaxRowSum ( HYPRE_Solver solver , double max_row_sum );
int HYPRE_BoomerAMGGetMaxRowSum ( HYPRE_Solver solver , double *max_row_sum );
int HYPRE_BoomerAMGSetTruncFactor ( HYPRE_Solver solver , double trunc_factor );
int HYPRE_BoomerAMGSetAggTruncFactor ( HYPRE_Solver solver , double agg_trunc_factor );
int HYPRE_BoomerAMGGetTruncFactor ( HYPRE_Solver solver , double *trunc_factor );
int HYPRE_BoomerAMGSetPMaxElmts ( HYPRE_Solver solver , int P_max_elmts );
int HYPRE_BoomerAMGGetPMaxElmts ( HYPRE_Solver solver , int *P_max_elmts );
int HYPRE_BoomerAMGSetPMax1 ( HYPRE_Solver solver , int P_max1 );
int HYPRE_BoomerAMGSetPMax2 ( HYPRE_Solver solver , int P_max2 );
int HYPRE_BoomerAMGSetJacobiTruncThreshold ( HYPRE_Solver solver , double jacobi_trunc_threshold );
int HYPRE_BoomerAMGGetJacobiTruncThreshold ( HYPRE_Solver solver , double *jacobi_trunc_threshold );
int HYPRE_BoomerAMGSetPostInterpType ( HYPRE_Solver solver , int post_interp_type );
int HYPRE_BoomerAMGGetPostInterpType ( HYPRE_Solver solver , int *post_interp_type );
int HYPRE_BoomerAMGSetSCommPkgSwitch ( HYPRE_Solver solver , double S_commpkg_switch );
int HYPRE_BoomerAMGSetInterpType ( HYPRE_Solver solver , int interp_type );
int HYPRE_BoomerAMGSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_BoomerAMGSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_BoomerAMGGetMaxIter ( HYPRE_Solver solver , int *max_iter );
int HYPRE_BoomerAMGSetCoarsenType ( HYPRE_Solver solver , int coarsen_type );
int HYPRE_BoomerAMGGetCoarsenType ( HYPRE_Solver solver , int *coarsen_type );
int HYPRE_BoomerAMGSetMeasureType ( HYPRE_Solver solver , int measure_type );
int HYPRE_BoomerAMGGetMeasureType ( HYPRE_Solver solver , int *measure_type );
int HYPRE_BoomerAMGSetSetupType ( HYPRE_Solver solver , int setup_type );
int HYPRE_BoomerAMGSetCycleType ( HYPRE_Solver solver , int cycle_type );
int HYPRE_BoomerAMGGetCycleType ( HYPRE_Solver solver , int *cycle_type );
int HYPRE_BoomerAMGSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_BoomerAMGGetTol ( HYPRE_Solver solver , double *tol );
int HYPRE_BoomerAMGSetNumGridSweeps ( HYPRE_Solver solver , int *num_grid_sweeps );
int HYPRE_BoomerAMGSetNumSweeps ( HYPRE_Solver solver , int num_sweeps );
int HYPRE_BoomerAMGSetCycleNumSweeps ( HYPRE_Solver solver , int num_sweeps , int k );
int HYPRE_BoomerAMGGetCycleNumSweeps ( HYPRE_Solver solver , int *num_sweeps , int k );
int HYPRE_BoomerAMGInitGridRelaxation ( int **num_grid_sweeps_ptr , int **grid_relax_type_ptr , int ***grid_relax_points_ptr , int coarsen_type , double **relax_weights_ptr , int max_levels );
int HYPRE_BoomerAMGSetGridRelaxType ( HYPRE_Solver solver , int *grid_relax_type );
int HYPRE_BoomerAMGSetRelaxType ( HYPRE_Solver solver , int relax_type );
int HYPRE_BoomerAMGSetCycleRelaxType ( HYPRE_Solver solver , int relax_type , int k );
int HYPRE_BoomerAMGGetCycleRelaxType ( HYPRE_Solver solver , int *relax_type , int k );
int HYPRE_BoomerAMGSetRelaxOrder ( HYPRE_Solver solver , int relax_order );
int HYPRE_BoomerAMGSetGridRelaxPoints ( HYPRE_Solver solver , int **grid_relax_points );
int HYPRE_BoomerAMGSetRelaxWeight ( HYPRE_Solver solver , double *relax_weight );
int HYPRE_BoomerAMGSetRelaxWt ( HYPRE_Solver solver , double relax_wt );
int HYPRE_BoomerAMGSetLevelRelaxWt ( HYPRE_Solver solver , double relax_wt , int level );
int HYPRE_BoomerAMGSetOmega ( HYPRE_Solver solver , double *omega );
int HYPRE_BoomerAMGSetOuterWt ( HYPRE_Solver solver , double outer_wt );
int HYPRE_BoomerAMGSetLevelOuterWt ( HYPRE_Solver solver , double outer_wt , int level );
int HYPRE_BoomerAMGSetSmoothType ( HYPRE_Solver solver , int smooth_type );
int HYPRE_BoomerAMGGetSmoothType ( HYPRE_Solver solver , int *smooth_type );
int HYPRE_BoomerAMGSetSmoothNumLevels ( HYPRE_Solver solver , int smooth_num_levels );
int HYPRE_BoomerAMGGetSmoothNumLevels ( HYPRE_Solver solver , int *smooth_num_levels );
int HYPRE_BoomerAMGSetSmoothNumSweeps ( HYPRE_Solver solver , int smooth_num_sweeps );
int HYPRE_BoomerAMGGetSmoothNumSweeps ( HYPRE_Solver solver , int *smooth_num_sweeps );
int HYPRE_BoomerAMGSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_BoomerAMGGetLogging ( HYPRE_Solver solver , int *logging );
int HYPRE_BoomerAMGSetPrintLevel ( HYPRE_Solver solver , int print_level );
int HYPRE_BoomerAMGGetPrintLevel ( HYPRE_Solver solver , int *print_level );
int HYPRE_BoomerAMGSetPrintFileName ( HYPRE_Solver solver , const char *print_file_name );
int HYPRE_BoomerAMGSetDebugFlag ( HYPRE_Solver solver , int debug_flag );
int HYPRE_BoomerAMGGetDebugFlag ( HYPRE_Solver solver , int *debug_flag );
int HYPRE_BoomerAMGGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_BoomerAMGGetCumNumIterations ( HYPRE_Solver solver , int *cum_num_iterations );
int HYPRE_BoomerAMGGetResidual ( HYPRE_Solver solver , HYPRE_ParVector *residual );
int HYPRE_BoomerAMGGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *rel_resid_norm );
int HYPRE_BoomerAMGSetVariant ( HYPRE_Solver solver , int variant );
int HYPRE_BoomerAMGGetVariant ( HYPRE_Solver solver , int *variant );
int HYPRE_BoomerAMGSetOverlap ( HYPRE_Solver solver , int overlap );
int HYPRE_BoomerAMGGetOverlap ( HYPRE_Solver solver , int *overlap );
int HYPRE_BoomerAMGSetDomainType ( HYPRE_Solver solver , int domain_type );
int HYPRE_BoomerAMGGetDomainType ( HYPRE_Solver solver , int *domain_type );
int HYPRE_BoomerAMGSetSchwarzRlxWeight ( HYPRE_Solver solver , double schwarz_rlx_weight );
int HYPRE_BoomerAMGGetSchwarzRlxWeight ( HYPRE_Solver solver , double *schwarz_rlx_weight );
int HYPRE_BoomerAMGSetNumFunctions ( HYPRE_Solver solver , int num_functions );
int HYPRE_BoomerAMGGetNumFunctions ( HYPRE_Solver solver , int *num_functions );
int HYPRE_BoomerAMGSetNodal ( HYPRE_Solver solver , int nodal );
int HYPRE_BoomerAMGSetDofFunc ( HYPRE_Solver solver , int *dof_func );
int HYPRE_BoomerAMGSetNumPaths ( HYPRE_Solver solver , int num_paths );
int HYPRE_BoomerAMGSetAggInterpType ( HYPRE_Solver solver , int agg_interp_type );
int HYPRE_BoomerAMGSetAggPMaxElmts ( HYPRE_Solver solver , int agg_P_max_elmts );
int HYPRE_BoomerAMGSetAggNumLevels ( HYPRE_Solver solver , int agg_num_levels );
int HYPRE_BoomerAMGSetNumCRRelaxSteps ( HYPRE_Solver solver , int num_CR_relax_steps );
int HYPRE_BoomerAMGSetCRRate ( HYPRE_Solver solver , double CR_rate );
int HYPRE_BoomerAMGSetISType ( HYPRE_Solver solver , int IS_type );
int HYPRE_BoomerAMGSetCRUseCG ( HYPRE_Solver solver , int CR_use_CG );
int HYPRE_BoomerAMGSetGSMG ( HYPRE_Solver solver , int gsmg );
int HYPRE_BoomerAMGSetNumSamples ( HYPRE_Solver solver , int gsmg );
int HYPRE_BoomerAMGSetMaxCoarseSize ( HYPRE_Solver solver , int max_coarse_size );
int HYPRE_BoomerAMGSetSeqThreshold ( HYPRE_Solver solver , int seq_threshold );

/* HYPRE_parcsr_gmres.c */
int HYPRE_ParCSRGMRESCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRGMRESDestroy ( HYPRE_Solver solver );
int HYPRE_ParCSRGMRESSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRGMRESSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRGMRESSetKDim ( HYPRE_Solver solver , int k_dim );
int HYPRE_ParCSRGMRESSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRGMRESSetMinIter ( HYPRE_Solver solver , int min_iter );
int HYPRE_ParCSRGMRESSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRGMRESSetStopCrit ( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRGMRESSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRGMRESGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRGMRESSetLogging ( HYPRE_Solver solver , int logging );
int HYPRE_ParCSRGMRESSetPrintLevel ( HYPRE_Solver solver , int print_level );
int HYPRE_ParCSRGMRESGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );


/* HYPRE_parcsr_pcg.c */
int HYPRE_ParCSRPCGCreate ( MPI_Comm comm , HYPRE_Solver *solver );
int HYPRE_ParCSRPCGDestroy ( HYPRE_Solver solver );
int HYPRE_ParCSRPCGSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPCGSolve ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector b , HYPRE_ParVector x );
int HYPRE_ParCSRPCGSetTol ( HYPRE_Solver solver , double tol );
int HYPRE_ParCSRPCGSetMaxIter ( HYPRE_Solver solver , int max_iter );
int HYPRE_ParCSRPCGSetStopCrit ( HYPRE_Solver solver , int stop_crit );
int HYPRE_ParCSRPCGSetTwoNorm ( HYPRE_Solver solver , int two_norm );
int HYPRE_ParCSRPCGSetRelChange ( HYPRE_Solver solver , int rel_change );
int HYPRE_ParCSRPCGSetPrecond ( HYPRE_Solver solver , HYPRE_PtrToParSolverFcn precond , HYPRE_PtrToParSolverFcn precond_setup , HYPRE_Solver precond_solver );
int HYPRE_ParCSRPCGGetPrecond ( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );
int HYPRE_ParCSRPCGSetPrintLevel ( HYPRE_Solver solver , int level );
int HYPRE_ParCSRPCGSetLogging ( HYPRE_Solver solver , int level );
int HYPRE_ParCSRPCGGetNumIterations ( HYPRE_Solver solver , int *num_iterations );
int HYPRE_ParCSRPCGGetFinalRelativeResidualNorm ( HYPRE_Solver solver , double *norm );
int HYPRE_ParCSRDiagScaleSetup ( HYPRE_Solver solver , HYPRE_ParCSRMatrix A , HYPRE_ParVector y , HYPRE_ParVector x );
int HYPRE_ParCSRDiagScale ( HYPRE_Solver solver , HYPRE_ParCSRMatrix HA , HYPRE_ParVector Hy , HYPRE_ParVector Hx );

/* par_amg.c */
void *hypre_BoomerAMGCreate ( void );
int hypre_BoomerAMGDestroy ( void *data );
int hypre_BoomerAMGSetRestriction ( void *data , int restr_par );
int hypre_BoomerAMGSetMaxLevels ( void *data , int max_levels );
int hypre_BoomerAMGGetMaxLevels ( void *data , int *max_levels );
int hypre_BoomerAMGSetStrongThreshold ( void *data , double strong_threshold );
int hypre_BoomerAMGGetStrongThreshold ( void *data , double *strong_threshold );
int hypre_BoomerAMGSetMaxRowSum ( void *data , double max_row_sum );
int hypre_BoomerAMGGetMaxRowSum ( void *data , double *max_row_sum );
int hypre_BoomerAMGSetTruncFactor ( void *data , double trunc_factor );
int hypre_BoomerAMGSetAggTruncFactor ( void *data , double agg_trunc_factor );
int hypre_BoomerAMGGetTruncFactor ( void *data , double *trunc_factor );
int hypre_BoomerAMGSetPMaxElmts ( void *data , int P_max_elmts );
int hypre_BoomerAMGSetPMax1 ( void *data , int P_max1 );
int hypre_BoomerAMGSetPMax2 ( void *data , int P_max2 );
int hypre_BoomerAMGGetPMaxElmts ( void *data , int *P_max_elmts );
int hypre_BoomerAMGSetJacobiTruncThreshold ( void *data , double jacobi_trunc_threshold );
int hypre_BoomerAMGGetJacobiTruncThreshold ( void *data , double *jacobi_trunc_threshold );
int hypre_BoomerAMGSetPostInterpType ( void *data , int post_interp_type );
int hypre_BoomerAMGGetPostInterpType ( void *data , int *post_interp_type );
int hypre_BoomerAMGSetSCommPkgSwitch ( void *data , double S_commpkg_switch );
int hypre_BoomerAMGGetSCommPkgSwitch ( void *data , double *S_commpkg_switch );
int hypre_BoomerAMGSetInterpType ( void *data , int interp_type );
int hypre_BoomerAMGGetInterpType ( void *data , int *interp_type );
int hypre_BoomerAMGSetMinIter ( void *data , int min_iter );
int hypre_BoomerAMGGetMinIter ( void *data , int *min_iter );
int hypre_BoomerAMGSetMaxIter ( void *data , int max_iter );
int hypre_BoomerAMGGetMaxIter ( void *data , int *max_iter );
int hypre_BoomerAMGSetCoarsenType ( void *data , int coarsen_type );
int hypre_BoomerAMGGetCoarsenType ( void *data , int *coarsen_type );
int hypre_BoomerAMGSetMeasureType ( void *data , int measure_type );
int hypre_BoomerAMGGetMeasureType ( void *data , int *measure_type );
int hypre_BoomerAMGSetSetupType ( void *data , int setup_type );
int hypre_BoomerAMGGetSetupType ( void *data , int *setup_type );
int hypre_BoomerAMGSetCycleType ( void *data , int cycle_type );
int hypre_BoomerAMGGetCycleType ( void *data , int *cycle_type );
int hypre_BoomerAMGSetTol ( void *data , double tol );
int hypre_BoomerAMGGetTol ( void *data , double *tol );
int hypre_BoomerAMGSetNumSweeps ( void *data , int num_sweeps );
int hypre_BoomerAMGSetCycleNumSweeps ( void *data , int num_sweeps , int k );
int hypre_BoomerAMGGetCycleNumSweeps ( void *data , int *num_sweeps , int k );
int hypre_BoomerAMGSetNumGridSweeps ( void *data , int *num_grid_sweeps );
int hypre_BoomerAMGGetNumGridSweeps ( void *data , int **num_grid_sweeps );
int hypre_BoomerAMGSetRelaxType ( void *data , int relax_type );
int hypre_BoomerAMGSetCycleRelaxType ( void *data , int relax_type , int k );
int hypre_BoomerAMGGetCycleRelaxType ( void *data , int *relax_type , int k );
int hypre_BoomerAMGSetRelaxOrder ( void *data , int relax_order );
int hypre_BoomerAMGGetRelaxOrder ( void *data , int *relax_order );
int hypre_BoomerAMGSetGridRelaxType ( void *data , int *grid_relax_type );
int hypre_BoomerAMGGetGridRelaxType ( void *data , int **grid_relax_type );
int hypre_BoomerAMGSetGridRelaxPoints ( void *data , int **grid_relax_points );
int hypre_BoomerAMGGetGridRelaxPoints ( void *data , int ***grid_relax_points );
int hypre_BoomerAMGSetRelaxWeight ( void *data , double *relax_weight );
int hypre_BoomerAMGGetRelaxWeight ( void *data , double **relax_weight );
int hypre_BoomerAMGSetRelaxWt ( void *data , double relax_weight );
int hypre_BoomerAMGSetLevelRelaxWt ( void *data , double relax_weight , int level );
int hypre_BoomerAMGGetLevelRelaxWt ( void *data , double *relax_weight , int level );
int hypre_BoomerAMGSetOmega ( void *data , double *omega );
int hypre_BoomerAMGGetOmega ( void *data , double **omega );
int hypre_BoomerAMGSetOuterWt ( void *data , double omega );
int hypre_BoomerAMGSetLevelOuterWt ( void *data , double omega , int level );
int hypre_BoomerAMGGetLevelOuterWt ( void *data , double *omega , int level );
int hypre_BoomerAMGSetSmoothType ( void *data , int smooth_type );
int hypre_BoomerAMGGetSmoothType ( void *data , int *smooth_type );
int hypre_BoomerAMGSetSmoothNumLevels ( void *data , int smooth_num_levels );
int hypre_BoomerAMGGetSmoothNumLevels ( void *data , int *smooth_num_levels );
int hypre_BoomerAMGSetSmoothNumSweeps ( void *data , int smooth_num_sweeps );
int hypre_BoomerAMGGetSmoothNumSweeps ( void *data , int *smooth_num_sweeps );
int hypre_BoomerAMGSetLogging ( void *data , int logging );
int hypre_BoomerAMGGetLogging ( void *data , int *logging );
int hypre_BoomerAMGSetPrintLevel ( void *data , int print_level );
int hypre_BoomerAMGGetPrintLevel ( void *data , int *print_level );
int hypre_BoomerAMGSetPrintFileName ( void *data , const char *print_file_name );
int hypre_BoomerAMGGetPrintFileName ( void *data , char **print_file_name );
int hypre_BoomerAMGSetNumIterations ( void *data , int num_iterations );
int hypre_BoomerAMGSetDebugFlag ( void *data , int debug_flag );
int hypre_BoomerAMGGetDebugFlag ( void *data , int *debug_flag );
int hypre_BoomerAMGSetGSMG ( void *data , int par );
int hypre_BoomerAMGSetNumSamples ( void *data , int par );
int hypre_BoomerAMGSetNumFunctions ( void *data , int num_functions );
int hypre_BoomerAMGGetNumFunctions ( void *data , int *num_functions );
int hypre_BoomerAMGSetNodal ( void *data , int nodal );
int hypre_BoomerAMGSetNumPaths ( void *data , int num_paths );
int hypre_BoomerAMGSetAggInterpType ( void *data , int agg_interp_type );
int hypre_BoomerAMGSetAggPMaxElmts ( void *data , int agg_P_max_elmts );
int hypre_BoomerAMGSetAggNumLevels ( void *data , int agg_num_levels );
int hypre_BoomerAMGSetNumCRRelaxSteps ( void *data , int num_CR_relax_steps );
int hypre_BoomerAMGSetCRRate ( void *data , double CR_rate );
int hypre_BoomerAMGSetISType ( void *data , int IS_type );
int hypre_BoomerAMGSetCRUseCG ( void *data , int CR_use_CG );
int hypre_BoomerAMGSetNumPoints ( void *data , int num_points );
int hypre_BoomerAMGSetDofFunc ( void *data , int *dof_func );
int hypre_BoomerAMGSetPointDofMap ( void *data , int *point_dof_map );
int hypre_BoomerAMGSetDofPoint ( void *data , int *dof_point );
int hypre_BoomerAMGGetNumIterations ( void *data , int *num_iterations );
int hypre_BoomerAMGGetCumNumIterations ( void *data , int *cum_num_iterations );
int hypre_BoomerAMGGetResidual ( void *data , hypre_ParVector **resid );
int hypre_BoomerAMGGetRelResidualNorm ( void *data , double *rel_resid_norm );
int hypre_BoomerAMGSetVariant ( void *data , int variant );
int hypre_BoomerAMGGetVariant ( void *data , int *variant );
int hypre_BoomerAMGSetOverlap ( void *data , int overlap );
int hypre_BoomerAMGGetOverlap ( void *data , int *overlap );
int hypre_BoomerAMGSetDomainType ( void *data , int domain_type );
int hypre_BoomerAMGGetDomainType ( void *data , int *domain_type );
int hypre_BoomerAMGSetSchwarzRlxWeight ( void *data , double schwarz_rlx_weight );
int hypre_BoomerAMGGetSchwarzRlxWeight ( void *data , double *schwarz_rlx_weight );
int hypre_BoomerAMGSetChebyOrder( void     *data, int       order);
int hypre_BoomerAMGSetChebyEigRatio( void     *data, double       ratio);
int hypre_BoomerAMGSetMaxCoarseSize ( void *data , int max_coarse_size );
int hypre_BoomerAMGSetSeqThreshold ( void *data , int seq_threshold );



/* par_amg_setup.c */
int hypre_BoomerAMGSetup ( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );

/* par_amg_solve.c */
int hypre_BoomerAMGSolve ( void *amg_vdata , hypre_ParCSRMatrix *A , hypre_ParVector *f , hypre_ParVector *u );

/* par_cg_relax_wt.c */
int hypre_BoomerAMGCGRelaxWt ( void *amg_vdata , int level , int num_cg_sweeps , double *rlx_wt_ptr );
int hypre_Bisection ( int n , double *diag , double *offd , double y , double z , double tol , int k , double *ev_ptr );

/* par_coarsen.c */
int hypre_BoomerAMGCoarsen ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int CF_init , int debug_flag , int **CF_marker_ptr );
int hypre_BoomerAMGCoarsenRuge ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int measure_type , int coarsen_type , int debug_flag , int **CF_marker_ptr );
int hypre_BoomerAMGCoarsenFalgout ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int measure_type , int debug_flag , int **CF_marker_ptr );
int hypre_BoomerAMGCoarsenHMIS ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int measure_type , int debug_flag , int **CF_marker_ptr );
int hypre_BoomerAMGCoarsenPMIS ( hypre_ParCSRMatrix *S , hypre_ParCSRMatrix *A , int CF_init , int debug_flag , int **CF_marker_ptr );

/* par_coarse_parms.c */
int hypre_BoomerAMGCoarseParms ( MPI_Comm comm , int local_num_variables , int num_functions , int *dof_func , int *CF_marker , int **coarse_dof_func_ptr , HYPRE_BigInt **coarse_pnts_global_ptr );

/* par_cycle.c */
int hypre_BoomerAMGCycle ( void *amg_vdata , hypre_ParVector **F_array , hypre_ParVector **U_array );

/* par_indepset.c */
int hypre_BoomerAMGIndepSetInit ( hypre_ParCSRMatrix *S , double *measure_array , int seq_rand );
int hypre_BoomerAMGIndepSet ( hypre_ParCSRMatrix *S , double *measure_array , int *graph_array , int graph_array_size , int *graph_array_offd , int graph_array_offd_size , int *IS_marker , int *IS_marker_offd );

/* par_interp.c */
int hypre_BoomerAMGBuildInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGInterpTruncation ( hypre_ParCSRMatrix *P , double trunc_factor , int max_elmts );
void hypre_qsort2abs ( int *v , double *w , int left , int right );

/* par_jacobi_interp.c */
void hypre_BoomerAMGJacobiInterp ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **P , hypre_ParCSRMatrix *S , int num_functions , int *dof_func , int *CF_marker , int level , double truncation_threshold , double truncation_threshold_minus );
void hypre_BoomerAMGJacobiInterp_1 ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix **P , hypre_ParCSRMatrix *S , int *CF_marker , int level , double truncation_threshold , double truncation_threshold_minus , int *dof_func , int *dof_func_offd , double weight_AF );
void hypre_BoomerAMGTruncateInterp ( hypre_ParCSRMatrix *P , double eps , double dlt , int *CF_marker );
int hypre_ParCSRMatrix_dof_func_offd ( hypre_ParCSRMatrix *A , int num_functions , int *dof_func , int **dof_func_offd );

/* par_laplace_27pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian27pt ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_BigInt nz , int P , int Q , int R , int p , int q , int r , double *value , HYPRE_ParVector *rhs_ptr , HYPRE_ParVector *x_ptr );

/* par_laplace.c */
HYPRE_ParCSRMatrix GenerateLaplacian ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_BigInt nz , int P , int Q , int R , int p , int q , int r , double *value , HYPRE_ParVector *rhs_ptr , HYPRE_ParVector *x_ptr );
int hypre_map ( int ix , int iy , int iz , int p , int q , int r , int P , int Q , int R , HYPRE_BigInt *nx_part , HYPRE_BigInt *ny_part , HYPRE_BigInt *nz_part , HYPRE_BigInt *global_part , HYPRE_BigInt *value_ptr );

/* par_laplace_9pt.c */
HYPRE_ParCSRMatrix GenerateLaplacian9pt ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , int P , int Q , int p , int q , double *value , HYPRE_ParVector *rhs_ptr , HYPRE_ParVector *x_ptr );
int hypre_map2 ( int ix , int iy , int p , int q , int P , int Q , HYPRE_BigInt *nx_part , HYPRE_BigInt *ny_part , HYPRE_BigInt *global_part , HYPRE_BigInt *value_ptr );

/* par_lr_interp.c */
int hypre_BoomerAMGBuildStdInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int sep_weight , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildExtPIInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildExtPICCInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildFFInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildFF1Interp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildExtInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );

/* par_multi_interp.c */
int hypre_BoomerAMGBuildMultipass ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int P_max_elmts , int weight_option , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );

/* par_nodal_systems.c */
int hypre_BoomerAMGCreateNodalA ( hypre_ParCSRMatrix *A , int num_functions , int *dof_func , int option , hypre_ParCSRMatrix **AN_ptr );
int hypre_BoomerAMGCreateScalarCFS ( hypre_ParCSRMatrix *SN , int *CFN_marker , int *col_offd_SN_to_AN , int num_functions , int nodal , int data , int **dof_func_ptr , int **CF_marker_ptr , int **col_offd_S_to_A_ptr , hypre_ParCSRMatrix **S_ptr );
int hypre_BoomerAMGCreateScalarCF ( int *CFN_marker , int num_functions , int num_nodes , int **dof_func_ptr , int **CF_marker_ptr );

/* par_rap.c */
hypre_BigCSRMatrix *hypre_ExchangeRAPData ( hypre_BigCSRMatrix *RAP_int , hypre_ParCSRCommPkg *comm_pkg_RT );
int hypre_BoomerAMGBuildCoarseOperator ( hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , hypre_ParCSRMatrix **RAP_ptr );

/* par_rap_communication.c */
int hypre_GetCommPkgRTFromCommPkgA ( hypre_ParCSRMatrix *RT , hypre_ParCSRMatrix *A , int *fine_to_coarse , int *tmp_map_offd );
int hypre_GenerateSendMapAndCommPkg ( MPI_Comm comm , int num_sends , int num_recvs , int *recv_procs , int *send_procs , int *recv_vec_starts , hypre_ParCSRMatrix *A );

/* par_relax.c */
int hypre_BoomerAMGRelax ( hypre_ParCSRMatrix *A , hypre_ParVector *f , int *cf_marker , int relax_type , int relax_points , double relax_weight , double omega ,  double  *l1_norms, hypre_ParVector *u , hypre_ParVector *Vtemp , hypre_ParVector *Ztemp );
int gselim ( double *A , double *x , int n );

/* par_relax_interface.c */
int hypre_BoomerAMGRelaxIF ( hypre_ParCSRMatrix *A , hypre_ParVector *f , int *cf_marker , int relax_type , int relax_order , int cycle_type , double relax_weight , double omega ,  double *l1_norms, hypre_ParVector *u , hypre_ParVector *Vtemp, hypre_ParVector *Ztemp );

/* par_rotate_7pt.c */
HYPRE_ParCSRMatrix GenerateRotate7pt( MPI_Comm comm, HYPRE_BigInt nx, HYPRE_BigInt ny, int P, int Q, int p, int q, double alpha, double eps, HYPRE_ParVector *rhs_ptr, HYPRE_ParVector *x_ptr);

/* par_scaled_matnorm.c */
int hypre_ParCSRMatrixScaledNorm ( hypre_ParCSRMatrix *A , double *scnorm );

/* par_stats.c */
int hypre_BoomerAMGSetupStats ( void *amg_vdata , hypre_ParCSRMatrix *A );
int hypre_BoomerAMGWriteSolverParams ( void *data );

/* par_strength.c */
int hypre_BoomerAMGCreateS ( hypre_ParCSRMatrix *A , double strength_threshold , double max_row_sum , int num_functions , int *dof_func , hypre_ParCSRMatrix **S_ptr );
int hypre_BoomerAMGCreateSabs ( hypre_ParCSRMatrix *A , double strength_threshold , double max_row_sum , int num_functions , int *dof_func , hypre_ParCSRMatrix **S_ptr );
int hypre_BoomerAMGCreateSCommPkg ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *S , int **col_offd_S_to_A_ptr );
int hypre_BoomerAMGCreate2ndS ( hypre_ParCSRMatrix *S , int *CF_marker , int num_paths , HYPRE_BigInt *coarse_row_starts , hypre_ParCSRMatrix **C_ptr );
int hypre_BoomerAMGCorrectCFMarker ( int *CF_marker , int num_var , int *new_CF_marker );
int hypre_BoomerAMGCorrectCFMarker2 ( int *CF_marker , int num_var , int *new_CF_marker );

/* par_vardifconv.c */
HYPRE_ParCSRMatrix GenerateVarDifConv ( MPI_Comm comm , HYPRE_BigInt nx , HYPRE_BigInt ny , HYPRE_BigInt nz , int P , int Q , int R , int p , int q , int r , double eps , HYPRE_ParVector *rhs_ptr , HYPRE_ParVector *x_ptr );
double afun ( double xx , double yy , double zz );
double bfun ( double xx , double yy , double zz );
double cfun ( double xx , double yy , double zz );
double dfun ( double xx , double yy , double zz );
double efun ( double xx , double yy , double zz );
double ffun ( double xx , double yy , double zz );
double gfun ( double xx , double yy , double zz );
double rfun ( double xx , double yy , double zz );
double bndfun ( double xx , double yy , double zz );

/* partial.c */
int hypre_BoomerAMGBuildPartialStdInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_BigInt *num_old_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int sep_weight , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildPartialExtPIInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_BigInt *num_old_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );
int hypre_BoomerAMGBuildPartialExtInterp ( hypre_ParCSRMatrix *A , int *CF_marker , hypre_ParCSRMatrix *S , HYPRE_BigInt *num_cpts_global , HYPRE_BigInt *num_old_cpts_global , int num_functions , int *dof_func , int debug_flag , double trunc_factor , int max_elmts , int *col_offd_S_to_A , hypre_ParCSRMatrix **P_ptr );

/* pcg_par.c */
char *hypre_ParKrylovCAlloc ( int count , int elt_size );
int hypre_ParKrylovFree ( char *ptr );
void *hypre_ParKrylovCreateVector ( void *vvector );
void *hypre_ParKrylovCreateVectorArray ( int n , void *vvector );
int hypre_ParKrylovDestroyVector ( void *vvector );
void *hypre_ParKrylovMatvecCreate ( void *A , void *x );
int hypre_ParKrylovMatvec ( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_ParKrylovMatvecT ( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_ParKrylovMatvecDestroy ( void *matvec_data );
double hypre_ParKrylovInnerProd ( void *x , void *y );
int hypre_ParKrylovCopyVector ( void *x , void *y );
int hypre_ParKrylovClearVector ( void *x );
int hypre_ParKrylovScaleVector ( double alpha , void *x );
int hypre_ParKrylovAxpy ( double alpha , void *x , void *y );
int hypre_ParKrylovCommInfo ( void *A , int *my_id , int *num_procs );
int hypre_ParKrylovIdentitySetup ( void *vdata , void *A , void *b , void *x );
int hypre_ParKrylovIdentity ( void *vdata , void *A , void *b , void *x );

/* par_more_relax.c */
   int hypre_ParCSRMaxEigEstimate(hypre_ParCSRMatrix *A, int scale, double *max_eig);
int hypre_ParCSRMaxEigEstimateCG(hypre_ParCSRMatrix *A, /* matrix to relax with */
                                 int scale, /* scale by diagonal?*/
                                 int max_iter,
                                 double *max_eig, double *min_eig);
int hypre_ParCSRRelax_Cheby(hypre_ParCSRMatrix *A, hypre_ParVector *f, double max_eig,  double min_eig, double eig_ratio, int order, int scale,  int variant, hypre_ParVector *u,  hypre_ParVector *v, hypre_ParVector *v2);
int hypre_BoomerAMGRelax_FCFJacobi( hypre_ParCSRMatrix *A,  hypre_ParVector *f, int *cf_marker, double relax_weight, hypre_ParVector *u, hypre_ParVector *Vtemp );
int hypre_ParCSRRelax_CG( HYPRE_Solver solver, hypre_ParCSRMatrix *A, hypre_ParVector    *f, hypre_ParVector    *u, int num_its);
int hypre_ParCSRRelax_SD( hypre_ParCSRMatrix *A,  hypre_ParVector    *f,  hypre_ParVector    *u, hypre_ParVector *r,    hypre_ParVector *p,    int num_its);
int hypre_ParCSRComputeTheta(hypre_ParCSRMatrix *A, double *theta_est);
int hypre_ParCSRComputeL1Norms ( hypre_ParCSRMatrix *A , int option , int *cf_marker , double **l1_norm_ptr);
int hypre_ParCSRComputeL1NormsThreads ( hypre_ParCSRMatrix *A , int option , int num_threads , int *cf_marker , double **l1_norm_ptr);
int hypre_ParCSRRelax_L1 ( hypre_ParCSRMatrix *A , hypre_ParVector *f , double relax_weight , double omega , double *l1_norms , hypre_ParVector *u , hypre_ParVector *v ,  hypre_ParVector *z);
int hypre_ParCSRRelax_L1_GS ( hypre_ParCSRMatrix *A , hypre_ParVector *f , double relax_weight , double omega , double *l1_norms , hypre_ParVector *u , hypre_ParVector *v ,  hypre_ParVector *z);
int hypre_ParCSRRelax_L1_Jacobi( hypre_ParCSRMatrix *A, hypre_ParVector    *f,  int                *cf_marker,  int                 relax_points,  double              relax_weight, double             *l1_norms,  hypre_ParVector    *u, hypre_ParVector    *Vtemp );
   
#ifdef __cplusplus
}
#endif

#endif

