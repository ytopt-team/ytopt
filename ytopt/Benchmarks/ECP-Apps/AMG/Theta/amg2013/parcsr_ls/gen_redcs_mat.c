#include "parcsr_ls.h"
#include "parcsr_mv.h"
#include "seq_mv.h"
#include "par_amg.h"


/* here we have the sequential setup and solve - called from the
 * parallel one - for the coarser levels */

int hypre_seqAMGSetup( hypre_ParAMGData *amg_data,
                      int p_level,
                      int coarse_threshold)


{

   /* Par Data Structure variables */
   hypre_ParCSRMatrix **Par_A_array = hypre_ParAMGDataAArray(amg_data);

   MPI_Comm 	      comm = hypre_ParCSRMatrixComm(Par_A_array[0]); 
   MPI_Comm 	      new_comm, seq_comm;

   hypre_ParCSRMatrix   *A_seq = NULL;
   hypre_CSRMatrix  *A_seq_diag;
   hypre_CSRMatrix  *A_seq_offd;
   hypre_ParVector   *F_seq = NULL;
   hypre_ParVector   *U_seq = NULL;
 
   hypre_ParCSRMatrix *A;

   int               **dof_func_array;   
   int                num_procs, my_id;

   int                not_finished_coarsening;
   int                level;
   int                redundant;

   HYPRE_Solver  coarse_solver;

   /* misc */
   dof_func_array = hypre_ParAMGDataDofFuncArray(amg_data);
   redundant = hypre_ParAMGDataRedundant(amg_data);

   /*MPI Stuff */
   MPI_Comm_size(comm, &num_procs);   
  
   /*initial */
   level = p_level;
   
   not_finished_coarsening = 1;
  
   /* convert A at this level to sequential */
   A = Par_A_array[level];

   {
      double *A_seq_data = NULL;
      int *A_seq_i = NULL;
      int *A_seq_offd_i = NULL;
      int *A_seq_j = NULL;

      double *A_tmp_data = NULL;
      int *A_tmp_i = NULL;
      HYPRE_BigInt *A_tmp_j = NULL;

      int *info = NULL;
      int *displs = NULL;
      int *displs2 = NULL;
      int i, j, size, num_nonzeros, total_nnz, cnt;
  
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
      HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
      int *A_diag_i = hypre_CSRMatrixI(A_diag);
      int *A_offd_i = hypre_CSRMatrixI(A_offd);
      int *A_diag_j = hypre_CSRMatrixJ(A_diag);
      int *A_offd_j = hypre_CSRMatrixJ(A_offd);
      double *A_diag_data = hypre_CSRMatrixData(A_diag);
      double *A_offd_data = hypre_CSRMatrixData(A_offd);
      int num_rows = hypre_CSRMatrixNumRows(A_diag);
      HYPRE_BigInt first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
      int new_num_procs;
      HYPRE_BigInt *row_starts;

      hypre_GenerateSubComm(comm, num_rows, &new_comm); 

      if (num_rows)
      {
         hypre_ParAMGDataParticipate(amg_data) = 1;
         MPI_Comm_size(new_comm, &new_num_procs); 
         MPI_Comm_rank(new_comm, &my_id);
         info = hypre_CTAlloc(int, new_num_procs);

         if (redundant)
            MPI_Allgather(&num_rows, 1, MPI_INT, info, 1, MPI_INT, new_comm);
         else
            MPI_Gather(&num_rows, 1, MPI_INT, info, 1, MPI_INT, 0, new_comm);

         /* alloc space in seq data structure only for participating procs*/
         if (redundant || my_id == 0)
         {
            HYPRE_BoomerAMGCreate(&coarse_solver);
            HYPRE_BoomerAMGSetMaxRowSum(coarse_solver,
		hypre_ParAMGDataMaxRowSum(amg_data)); 
            HYPRE_BoomerAMGSetStrongThreshold(coarse_solver,
		hypre_ParAMGDataStrongThreshold(amg_data)); 
            HYPRE_BoomerAMGSetCoarsenType(coarse_solver,
		hypre_ParAMGDataCoarsenType(amg_data)); 
            HYPRE_BoomerAMGSetInterpType(coarse_solver,
		hypre_ParAMGDataInterpType(amg_data)); 
            HYPRE_BoomerAMGSetTruncFactor(coarse_solver, 
		hypre_ParAMGDataTruncFactor(amg_data)); 
            HYPRE_BoomerAMGSetPMaxElmts(coarse_solver, 
		hypre_ParAMGDataPMaxElmts(amg_data)); 
            HYPRE_BoomerAMGSetGridRelaxType(coarse_solver, 
		hypre_ParAMGDataGridRelaxType(amg_data)); 
            HYPRE_BoomerAMGSetGridRelaxPoints(coarse_solver, 
		hypre_ParAMGDataGridRelaxPoints(amg_data)); 
            HYPRE_BoomerAMGSetRelaxOrder(coarse_solver, 
		hypre_ParAMGDataRelaxOrder(amg_data)); 
            HYPRE_BoomerAMGSetRelaxWeight(coarse_solver, 
		&hypre_ParAMGDataRelaxWeight(amg_data)[p_level]); 
            HYPRE_BoomerAMGSetNumGridSweeps(coarse_solver, 
		hypre_ParAMGDataNumGridSweeps(amg_data)); 
            HYPRE_BoomerAMGSetNumFunctions(coarse_solver, 
		hypre_ParAMGDataNumFunctions(amg_data)); 
            HYPRE_BoomerAMGSetMaxIter(coarse_solver, 1); 
            HYPRE_BoomerAMGSetTol(coarse_solver, 0); 
         }

         /* Create CSR Matrix, will be Diag part of new matrix */
         A_tmp_i = hypre_CTAlloc(int, num_rows+1);

         A_tmp_i[0] = 0;
         for (i=1; i < num_rows+1; i++)
            A_tmp_i[i] = A_diag_i[i]-A_diag_i[i-1]+A_offd_i[i]-A_offd_i[i-1];

         num_nonzeros = A_offd_i[num_rows]+A_diag_i[num_rows];

         A_tmp_j = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros);
         A_tmp_data = hypre_CTAlloc(double, num_nonzeros);

         cnt = 0;
         for (i=0; i < num_rows; i++)
         {
            for (j=A_diag_i[i]; j < A_diag_i[i+1]; j++)
	    {
	       A_tmp_j[cnt] = (HYPRE_BigInt) A_diag_j[j]+first_row_index;
	       A_tmp_data[cnt++] = A_diag_data[j];
	    }
            for (j=A_offd_i[i]; j < A_offd_i[i+1]; j++)
	    {
	       A_tmp_j[cnt] = col_map_offd[A_offd_j[j]];
	       A_tmp_data[cnt++] = A_offd_data[j];
	    }
         }

         displs = hypre_CTAlloc(int, new_num_procs+1);
         displs[0] = 0;
         for (i=1; i < new_num_procs+1; i++)
            displs[i] = displs[i-1]+info[i-1];
         size = displs[new_num_procs];
  
         if (redundant || my_id == 0)
         {
            A_seq_i = hypre_CTAlloc(int, size+1);
            A_seq_offd_i = hypre_CTAlloc(int, size+1);
         }

         if (redundant)
            MPI_Allgatherv ( &A_tmp_i[1], num_rows, MPI_INT, &A_seq_i[1], info,
                        displs, MPI_INT, new_comm );
         else
            MPI_Gatherv ( &A_tmp_i[1], num_rows, MPI_INT, &A_seq_i[1], info,
                        displs, MPI_INT, 0, new_comm );

         if (redundant || my_id == 0)
         {
            displs2 = hypre_CTAlloc(int, new_num_procs+1);

            A_seq_i[0] = 0;
            displs2[0] = 0;
            for (j=1; j < displs[1]; j++)
               A_seq_i[j] = A_seq_i[j]+A_seq_i[j-1];
            for (i=1; i < new_num_procs; i++)
            {
               for (j=displs[i]; j < displs[i+1]; j++)
               {
                  A_seq_i[j] = A_seq_i[j]+A_seq_i[j-1];
               }
            }
            A_seq_i[size] = A_seq_i[size]+A_seq_i[size-1];
            displs2[new_num_procs] = A_seq_i[size];
            for (i=1; i < new_num_procs+1; i++)
            {
               displs2[i] = A_seq_i[displs[i]];
               info[i-1] = displs2[i] - displs2[i-1];
            }

            total_nnz = displs2[new_num_procs];
            A_seq_j = hypre_CTAlloc(int, total_nnz);
            A_seq_data = hypre_CTAlloc(double, total_nnz);
         }
         if (redundant)
         {
            MPI_Allgatherv ( A_tmp_j, num_nonzeros, MPI_HYPRE_BIG_INT,
                       A_seq_j, info, displs2,
                       MPI_INT, new_comm );

            MPI_Allgatherv ( A_tmp_data, num_nonzeros, MPI_DOUBLE,
                       A_seq_data, info, displs2,
                       MPI_DOUBLE, new_comm );
         }
         else
         {
            MPI_Gatherv ( A_tmp_j, num_nonzeros, MPI_HYPRE_BIG_INT,
                       A_seq_j, info, displs2,
                       MPI_INT, 0, new_comm );

            MPI_Gatherv ( A_tmp_data, num_nonzeros, MPI_DOUBLE,
                       A_seq_data, info, displs2,
                       MPI_DOUBLE, 0, new_comm );
         }

         hypre_TFree(info);
         hypre_TFree(displs);
         hypre_TFree(A_tmp_i);
         hypre_TFree(A_tmp_j);
         hypre_TFree(A_tmp_data);

         if (redundant || my_id == 0)
         {
            hypre_TFree(displs2);
   
            row_starts = hypre_CTAlloc(HYPRE_BigInt,2);
            row_starts[0] = 0; 
            row_starts[1] = size;
 
            /* Create 1 proc communicator */
            seq_comm = MPI_COMM_SELF;

            A_seq = hypre_ParCSRMatrixCreate(seq_comm,size,size,
					  row_starts, row_starts,
						0,total_nnz,0); 

            A_seq_diag = hypre_ParCSRMatrixDiag(A_seq);
            A_seq_offd = hypre_ParCSRMatrixOffd(A_seq);

            hypre_CSRMatrixData(A_seq_diag) = A_seq_data;
            hypre_CSRMatrixI(A_seq_diag) = A_seq_i;
            hypre_CSRMatrixJ(A_seq_diag) = A_seq_j;
            hypre_CSRMatrixI(A_seq_offd) = A_seq_offd_i;

            F_seq = hypre_ParVectorCreate(seq_comm, size, row_starts);
            U_seq = hypre_ParVectorCreate(seq_comm, size, row_starts);
            hypre_ParVectorOwnsPartitioning(F_seq) = 0;
            hypre_ParVectorOwnsPartitioning(U_seq) = 0;
            hypre_ParVectorInitialize(F_seq);
            hypre_ParVectorInitialize(U_seq);

            hypre_BoomerAMGSetup(coarse_solver,A_seq,F_seq,U_seq);

            hypre_ParAMGDataCoarseSolver(amg_data) = coarse_solver;
            hypre_ParAMGDataACoarse(amg_data) = A_seq;
            hypre_ParAMGDataFCoarse(amg_data) = F_seq;
            hypre_ParAMGDataUCoarse(amg_data) = U_seq;
         }
         hypre_ParAMGDataNewComm(amg_data) = new_comm;
      }
   }
   return 0;
}



/*--------------------------------------------------------------------------
 * hypre_seqAMGCycle
 *--------------------------------------------------------------------------*/

int
hypre_seqAMGCycle( hypre_ParAMGData *amg_data,
                   int p_level,
                   hypre_ParVector  **Par_F_array,
                   hypre_ParVector  **Par_U_array   )
{
   
   hypre_ParVector    *Aux_U;
   hypre_ParVector    *Aux_F;

   /* Local variables  */

   int       Solve_err_flag = 0;

   int n;
   int i;
   
   hypre_Vector   *u_local;
   double         *u_data;
   
   HYPRE_BigInt	   first_index;
   
   /* Acquire seq data */
   MPI_Comm new_comm = hypre_ParAMGDataNewComm(amg_data);
   HYPRE_Solver coarse_solver = hypre_ParAMGDataCoarseSolver(amg_data);
   hypre_ParCSRMatrix *A_coarse = hypre_ParAMGDataACoarse(amg_data);
   hypre_ParVector *F_coarse = hypre_ParAMGDataFCoarse(amg_data);
   hypre_ParVector *U_coarse = hypre_ParAMGDataUCoarse(amg_data);
   int redundant = hypre_ParAMGDataRedundant(amg_data);

   Aux_U = Par_U_array[p_level];
   Aux_F = Par_F_array[p_level];

   first_index = hypre_ParVectorFirstIndex(Aux_U);
   u_local = hypre_ParVectorLocalVector(Aux_U);
   u_data  = hypre_VectorData(u_local);
   n =  hypre_VectorSize(u_local);


   /*if (A_coarse)*/
   if (hypre_ParAMGDataParticipate(amg_data))
   {
      double         *f_data;
      hypre_Vector   *f_local;
      hypre_Vector   *tmp_vec;
      
      int nf;
      int local_info;
      double *recv_buf;
      int *displs = NULL;
      int *info = NULL;
      int size;
      int new_num_procs, my_id;
      
      MPI_Comm_size(new_comm, &new_num_procs);
      MPI_Comm_rank(new_comm, &my_id);

      f_local = hypre_ParVectorLocalVector(Aux_F);
      f_data = hypre_VectorData(f_local);
      nf =  hypre_VectorSize(f_local);

      /* first f */
      info = hypre_CTAlloc(int, new_num_procs);
      local_info = nf;
      if (redundant)
         MPI_Allgather(&local_info, 1, MPI_INT, info, 1, MPI_INT, new_comm);
      else
         MPI_Gather(&local_info, 1, MPI_INT, info, 1, MPI_INT, 0, new_comm);

      if (redundant || my_id ==0)
      {
         displs = hypre_CTAlloc(int, new_num_procs+1);
         displs[0] = 0;
         for (i=1; i < new_num_procs+1; i++)
            displs[i] = displs[i-1]+info[i-1]; 
         size = displs[new_num_procs];
      
         tmp_vec =  hypre_ParVectorLocalVector(F_coarse);
         recv_buf = hypre_VectorData(tmp_vec);
      }

      if (redundant)
         MPI_Allgatherv ( f_data, nf, MPI_DOUBLE,
                          recv_buf, info, displs,
                          MPI_DOUBLE, new_comm );
      else
         MPI_Gatherv ( f_data, nf, MPI_DOUBLE,
                          recv_buf, info, displs,
                          MPI_DOUBLE, 0, new_comm );

      if (redundant || my_id ==0)
      {
         tmp_vec =  hypre_ParVectorLocalVector(U_coarse);
         recv_buf = hypre_VectorData(tmp_vec);
      }
      
      /*then u */
      if (redundant)
      {
         MPI_Allgatherv ( u_data, n, MPI_DOUBLE,
                       recv_buf, info, displs,
                       MPI_DOUBLE, new_comm );
         hypre_TFree(displs);
         hypre_TFree(info);
      }
      else
         MPI_Gatherv ( u_data, n, MPI_DOUBLE,
                       recv_buf, info, displs,
                       MPI_DOUBLE, 0, new_comm );
         
      /* clean up */
      if (redundant || my_id ==0)
      {
         hypre_BoomerAMGSolve(coarse_solver, A_coarse, F_coarse, U_coarse);
      }

      /*copy my part of U to parallel vector */
      if (redundant)
      {
         double *local_data;

         local_data =  hypre_VectorData(hypre_ParVectorLocalVector(U_coarse));

         for (i = 0; i < n; i++)
         {
            u_data[i] = local_data[(int)first_index+i];
         }
      }
      else
      {
         double *local_data;

         if (my_id == 0)
            local_data =  hypre_VectorData(hypre_ParVectorLocalVector(U_coarse));

         MPI_Scatterv ( local_data, info, displs, MPI_DOUBLE,
                       u_data, n, MPI_DOUBLE, 0, new_comm );
         /*if (my_id == 0)
            local_data =  hypre_VectorData(hypre_ParVectorLocalVector(F_coarse));
         hypre_ MPI_Scatterv ( local_data, info, displs, MPI_DOUBLE,
                       f_data, n, MPI_DOUBLE, 0, new_comm );*/
         if (my_id == 0) hypre_TFree(displs);
         hypre_TFree(info);
      }
   }

   return(Solve_err_flag);
}

/* generate sub communicator, which contains only idle processors */

int hypre_GenerateSubComm(MPI_Comm comm, int participate, MPI_Comm *new_comm_ptr) 
{
   MPI_Comm new_comm;
   MPI_Group orig_group, new_group; 
   MPI_Op hypre_MPI_MERGE;
   int *info, *ranks, new_num_procs, my_info, my_id, num_procs;
   int *list_len;

   MPI_Comm_rank(comm,&my_id);

   if (participate) 
      my_info = 1;
   else
      my_info = 0;

   MPI_Allreduce(&my_info, &new_num_procs, 1, MPI_INT, MPI_SUM, comm);

   if (new_num_procs == 0)
   {
      new_comm = MPI_COMM_NULL;
      *new_comm_ptr = new_comm;
      return 0;
   }
   ranks = hypre_CTAlloc(int, new_num_procs+2);
   if (new_num_procs == 1)
   {
      if (participate) my_info = my_id;
      MPI_Allreduce(&my_info, &ranks[2], 1, MPI_INT, MPI_SUM, comm);
   }
   else
   {
      info = hypre_CTAlloc(int, new_num_procs+2);
      list_len = hypre_CTAlloc(int, 1);

      if (participate) 
      {
         info[0] = 1;
         info[1] = 1;
         info[2] = my_id;
      }
      else
         info[0] = 0;

      list_len[0] = new_num_procs + 2;

      MPI_Op_create((MPI_User_function *)hypre_merge_lists, 0, &hypre_MPI_MERGE);

      MPI_Allreduce(info, ranks, list_len[0], MPI_INT, hypre_MPI_MERGE, comm);

      MPI_Op_free (&hypre_MPI_MERGE);
      hypre_TFree(list_len);
      hypre_TFree(info);
   }
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_group(comm, &orig_group);
   MPI_Group_incl(orig_group, new_num_procs, &ranks[2], &new_group);
   MPI_Comm_create(comm, new_group, &new_comm);
   MPI_Group_free(&new_group);
   MPI_Group_free(&orig_group);

   hypre_TFree(ranks);
   
   *new_comm_ptr = new_comm;
   
   return 0;
}


void hypre_merge_lists (int *list1, int* list2, int *np1, MPI_Datatype *dptr)
{
   int i, len1, len2, indx1, indx2;

   if (list1[0] == 0 || (list2[0] == 0 && list1[0] == 0))
   {
      return;
   }
   else
   {
      list2[0] = 1;
      len1 = list1[1];
      len2 = list2[1];
      list2[1] = len1+len2;
      if (list2[1] > *np1+2) printf("segfault in MPI User function merge_list\n");
      indx1 = len1+1;
      indx2 = len2+1;
      for (i=len1+len2+1; i > 1; i--)
      {
	 if (indx2 > 1 && indx1 > 1 && list1[indx1] > list2[indx2])
         {
            list2[i] = list1[indx1];
            indx1--;
         }
         else if (indx2 > 1)
         {
            list2[i] = list2[indx2];
            indx2--;
         }
         else if (indx1 > 1)
         {
            list2[i] = list1[indx1];
            indx1--;
         }
      }
   }
}

