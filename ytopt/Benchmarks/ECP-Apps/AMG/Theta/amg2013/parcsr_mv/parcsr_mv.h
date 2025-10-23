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

#include "HYPRE_parcsr_mv.h"

#ifndef hypre_PARCSR_MV_HEADER
#define hypre_PARCSR_MV_HEADER

#include "utilities.h"
#include "seq_mv.h"

#ifdef __cplusplus
extern "C" {
#endif




#ifndef HYPRE_PAR_CSR_COMMUNICATION_HEADER
#define HYPRE_PAR_CSR_COMMUNICATION_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommPkg:
 *   Structure containing information for doing communications
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm               comm;

   int                    num_sends;
   int                   *send_procs;
   int			 *send_map_starts;
   int			 *send_map_elmts;

   int                    num_recvs;
   int                   *recv_procs;
   int                   *recv_vec_starts;

   /* remote communication information */
   MPI_Datatype          *send_mpi_types;
   MPI_Datatype          *recv_mpi_types;

} hypre_ParCSRCommPkg;

/*--------------------------------------------------------------------------
 * hypre_ParCSRCommHandle:
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_ParCSRCommPkg  *comm_pkg;
   void 	  *send_data;
   void 	  *recv_data;

   int             num_requests;
   MPI_Request    *requests;

} hypre_ParCSRCommHandle;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommPkg
 *--------------------------------------------------------------------------*/
 
#define hypre_ParCSRCommPkgComm(comm_pkg)          (comm_pkg -> comm)
                                               
#define hypre_ParCSRCommPkgNumSends(comm_pkg)      (comm_pkg -> num_sends)
#define hypre_ParCSRCommPkgSendProcs(comm_pkg)     (comm_pkg -> send_procs)
#define hypre_ParCSRCommPkgSendProc(comm_pkg, i)   (comm_pkg -> send_procs[i])
#define hypre_ParCSRCommPkgSendMapStarts(comm_pkg) (comm_pkg -> send_map_starts)
#define hypre_ParCSRCommPkgSendMapStart(comm_pkg,i)(comm_pkg -> send_map_starts[i])
#define hypre_ParCSRCommPkgSendMapElmts(comm_pkg)  (comm_pkg -> send_map_elmts)
#define hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i) (comm_pkg -> send_map_elmts[i])

#define hypre_ParCSRCommPkgNumRecvs(comm_pkg)      (comm_pkg -> num_recvs)
#define hypre_ParCSRCommPkgRecvProcs(comm_pkg)     (comm_pkg -> recv_procs)
#define hypre_ParCSRCommPkgRecvProc(comm_pkg, i)   (comm_pkg -> recv_procs[i])
#define hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) (comm_pkg -> recv_vec_starts)
#define hypre_ParCSRCommPkgRecvVecStart(comm_pkg,i)(comm_pkg -> recv_vec_starts[i])

#define hypre_ParCSRCommPkgSendMPITypes(comm_pkg)  (comm_pkg -> send_mpi_types)
#define hypre_ParCSRCommPkgSendMPIType(comm_pkg,i) (comm_pkg -> send_mpi_types[i])

#define hypre_ParCSRCommPkgRecvMPITypes(comm_pkg)  (comm_pkg -> recv_mpi_types)
#define hypre_ParCSRCommPkgRecvMPIType(comm_pkg,i) (comm_pkg -> recv_mpi_types[i])

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_ParCSRCommHandle
 *--------------------------------------------------------------------------*/
 
#define hypre_ParCSRCommHandleCommPkg(comm_handle)     (comm_handle -> comm_pkg)
#define hypre_ParCSRCommHandleSendData(comm_handle)    (comm_handle -> send_data)
#define hypre_ParCSRCommHandleRecvData(comm_handle)    (comm_handle -> recv_data)
#define hypre_ParCSRCommHandleNumRequests(comm_handle) (comm_handle -> num_requests)
#define hypre_ParCSRCommHandleRequests(comm_handle)    (comm_handle -> requests)
#define hypre_ParCSRCommHandleRequest(comm_handle, i)  (comm_handle -> requests[i])

#endif /* HYPRE_PAR_CSR_COMMUNICATION_HEADER */

#ifndef hypre_PARCSR_ASSUMED_PART
#define  hypre_PARCSR_ASSUMED_PART

typedef struct
{
   int                   length;
   HYPRE_BigInt          row_start;
   HYPRE_BigInt          row_end;
   int                   storage_length;
   int                   *proc_list;
   HYPRE_BigInt	         *row_start_list;
   HYPRE_BigInt          *row_end_list;  
  int                    *sort_index;
} hypre_IJAssumedPart;




#endif /* hypre_PARCSR_ASSUMED_PART */




#ifndef hypre_NEW_COMMPKG
#define hypre_NEW_COMMPKG


typedef struct
{
  int                   length;
  int                   storage_length; 
  int                   *id;
  int                   *vec_starts;
  int                   element_storage_length; 
  HYPRE_BigInt          *elements;
  double                *d_elements;
   void                  *v_elements;
}  hypre_ProcListElements;   




#endif /* hypre_NEW_COMMPKG */





/******************************************************************************
 *
 * Header info for Parallel Vector data structure
 *
 *****************************************************************************/

#ifndef hypre_PAR_VECTOR_HEADER
#define hypre_PAR_VECTOR_HEADER


/*--------------------------------------------------------------------------
 * hypre_ParVector
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm	 comm;

   HYPRE_BigInt  global_size;
   HYPRE_BigInt  first_index;
   HYPRE_BigInt  last_index;
   HYPRE_BigInt *partitioning;
   hypre_Vector	*local_vector; 

   /* Does the Vector create/destroy `data'? */
   int      	 owns_data;
   int      	 owns_partitioning;

   hypre_IJAssumedPart *assumed_partition; /* only populated if no_global_partition option
                                              is used (compile-time option) AND this partition
                                              needed
                                              (for setting off-proc elements, for example)*/


} hypre_ParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_ParVectorComm(vector)  	        ((vector) -> comm)
#define hypre_ParVectorGlobalSize(vector)       ((vector) -> global_size)
#define hypre_ParVectorFirstIndex(vector)       ((vector) -> first_index)
#define hypre_ParVectorLastIndex(vector)        ((vector) -> last_index)
#define hypre_ParVectorPartitioning(vector)     ((vector) -> partitioning)
#define hypre_ParVectorLocalVector(vector)      ((vector) -> local_vector)
#define hypre_ParVectorOwnsData(vector)         ((vector) -> owns_data)
#define hypre_ParVectorOwnsPartitioning(vector) ((vector) -> owns_partitioning)
#define hypre_ParVectorNumVectors(vector)\
 (hypre_VectorNumVectors( hypre_ParVectorLocalVector(vector) ))

#define hypre_ParVectorAssumedPartition(vector) ((vector) -> assumed_partition)


#endif




/******************************************************************************
 *
 * Header info for Parallel CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_PAR_CSR_MATRIX_HEADER
#define hypre_PAR_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Parallel CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm		comm;

   HYPRE_BigInt     	global_num_rows;
   HYPRE_BigInt     	global_num_cols;
   HYPRE_BigInt		first_row_index;
   HYPRE_BigInt		first_col_diag;
   /* need to know entire local range in case row_starts and col_starts 
      are null  (i.e., bgl) AHB 6/05*/
   HYPRE_BigInt         last_row_index;
   HYPRE_BigInt         last_col_diag;

   hypre_CSRMatrix	*diag;
   hypre_CSRMatrix	*offd;
   HYPRE_BigInt		*col_map_offd; 
	/* maps columns of offd to global columns */
   HYPRE_BigInt 	*row_starts; 
	/* array of length num_procs+1, row_starts[i] contains the 
	   global number of the first row on proc i,  
	   first_row_index = row_starts[my_id],
	   row_starts[num_procs] = global_num_rows */
   HYPRE_BigInt		*col_starts;
	/* array of length num_procs+1, col_starts[i] contains the 
	   global number of the first column of diag on proc i,  
	   first_col_diag = col_starts[my_id],
	   col_starts[num_procs] = global_num_cols */

   hypre_ParCSRCommPkg	*comm_pkg;
   hypre_ParCSRCommPkg	*comm_pkgT;
   
   /* Does the ParCSRMatrix create/destroy `diag', `offd', `col_map_offd'? */
   int      owns_data;
   /* Does the ParCSRMatrix create/destroy `row_starts', `col_starts'? */
   int      owns_row_starts;
   int      owns_col_starts;

   HYPRE_BigInt num_nonzeros;
   double   d_num_nonzeros;

   /* Buffers used by GetRow to hold row currently being accessed. AJC, 4/99 */
   HYPRE_BigInt *rowindices;
   double  *rowvalues;
   int      getrowactive;

   hypre_IJAssumedPart *assumed_partition; /* only populated if no_global_partition option
                                              is used (compile-time option)*/


} hypre_ParCSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_ParCSRMatrixComm(matrix)		  ((matrix) -> comm)
#define hypre_ParCSRMatrixGlobalNumRows(matrix)   ((matrix) -> global_num_rows)
#define hypre_ParCSRMatrixGlobalNumCols(matrix)   ((matrix) -> global_num_cols)
#define hypre_ParCSRMatrixFirstRowIndex(matrix)   ((matrix) -> first_row_index)
#define hypre_ParCSRMatrixFirstColDiag(matrix)    ((matrix) -> first_col_diag)
#define hypre_ParCSRMatrixLastRowIndex(matrix)    ((matrix) -> last_row_index)
#define hypre_ParCSRMatrixLastColDiag(matrix)     ((matrix) -> last_col_diag)
#define hypre_ParCSRMatrixDiag(matrix)  	  ((matrix) -> diag)
#define hypre_ParCSRMatrixOffd(matrix)  	  ((matrix) -> offd)
#define hypre_ParCSRMatrixColMapOffd(matrix)  	  ((matrix) -> col_map_offd)
#define hypre_ParCSRMatrixRowStarts(matrix)       ((matrix) -> row_starts)
#define hypre_ParCSRMatrixColStarts(matrix)       ((matrix) -> col_starts)
#define hypre_ParCSRMatrixCommPkg(matrix)	  ((matrix) -> comm_pkg)
#define hypre_ParCSRMatrixCommPkgT(matrix)	  ((matrix) -> comm_pkgT)
#define hypre_ParCSRMatrixOwnsData(matrix)        ((matrix) -> owns_data)
#define hypre_ParCSRMatrixOwnsRowStarts(matrix)   ((matrix) -> owns_row_starts)
#define hypre_ParCSRMatrixOwnsColStarts(matrix)   ((matrix) -> owns_col_starts)
#define hypre_ParCSRMatrixNumRows(matrix) \
hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(matrix))
#define hypre_ParCSRMatrixNumCols(matrix) \
hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(matrix))
#define hypre_ParCSRMatrixNumNonzeros(matrix)     ((matrix) -> num_nonzeros)
#define hypre_ParCSRMatrixDNumNonzeros(matrix)    ((matrix) -> d_num_nonzeros)
#define hypre_ParCSRMatrixRowindices(matrix)      ((matrix) -> rowindices)
#define hypre_ParCSRMatrixRowvalues(matrix)       ((matrix) -> rowvalues)
#define hypre_ParCSRMatrixGetrowactive(matrix)    ((matrix) -> getrowactive)
#define hypre_ParCSRMatrixAssumedPartition(matrix) ((matrix) -> assumed_partition)

#endif

/* HYPRE_parcsr_matrix.c */
int HYPRE_ParCSRMatrixCreate ( MPI_Comm comm , HYPRE_BigInt global_num_rows , HYPRE_BigInt global_num_cols , HYPRE_BigInt *row_starts , HYPRE_BigInt *col_starts , int num_cols_offd , int num_nonzeros_diag , int num_nonzeros_offd , HYPRE_ParCSRMatrix *matrix );
int HYPRE_ParCSRMatrixDestroy ( HYPRE_ParCSRMatrix matrix );
int HYPRE_ParCSRMatrixInitialize ( HYPRE_ParCSRMatrix matrix );
int HYPRE_ParCSRMatrixRead ( MPI_Comm comm , const char *file_name , HYPRE_ParCSRMatrix *matrix );
int HYPRE_ParCSRMatrixPrint ( HYPRE_ParCSRMatrix matrix , const char *file_name );
int HYPRE_ParCSRMatrixPrintIJ ( HYPRE_ParCSRMatrix matrix , int base_i , int base_j , const char *file_name );
int HYPRE_ParCSRMatrixGetComm ( HYPRE_ParCSRMatrix matrix , MPI_Comm *comm );
int HYPRE_ParCSRMatrixGetDims ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt *M , HYPRE_BigInt *N );
int HYPRE_ParCSRMatrixGetRowPartitioning ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt **row_partitioning_ptr );
int HYPRE_ParCSRMatrixGetColPartitioning ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt **col_partitioning_ptr );
int HYPRE_ParCSRMatrixGetLocalRange ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt *row_start , HYPRE_BigInt *row_end , HYPRE_BigInt *col_start , HYPRE_BigInt *col_end );
int HYPRE_ParCSRMatrixGetRow ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt row , int *size , HYPRE_BigInt **col_ind , double **values );
int HYPRE_ParCSRMatrixRestoreRow ( HYPRE_ParCSRMatrix matrix , HYPRE_BigInt row , int *size , HYPRE_BigInt **col_ind , double **values );
int HYPRE_CSRMatrixToParCSRMatrix ( MPI_Comm comm , HYPRE_CSRMatrix A_CSR , HYPRE_BigInt *row_partitioning , HYPRE_BigInt *col_partitioning , HYPRE_ParCSRMatrix *matrix );
int HYPRE_ParCSRMatrixMatvec ( double alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , double beta , HYPRE_ParVector y );
int HYPRE_ParCSRMatrixMatvecT ( double alpha , HYPRE_ParCSRMatrix A , HYPRE_ParVector x , double beta , HYPRE_ParVector y );

/* HYPRE_parcsr_vector.c */
int HYPRE_ParVectorCreate ( MPI_Comm comm , HYPRE_BigInt global_size , HYPRE_BigInt *partitioning , HYPRE_ParVector *vector );
int HYPRE_ParVectorDestroy ( HYPRE_ParVector vector );
int HYPRE_ParVectorInitialize ( HYPRE_ParVector vector );
int HYPRE_ParVectorRead ( MPI_Comm comm , const char *file_name , HYPRE_ParVector *vector );
int HYPRE_ParVectorPrint ( HYPRE_ParVector vector , const char *file_name );
int HYPRE_ParVectorPrintIJ ( HYPRE_ParVector vector , int base_i , const char *file_name );
int HYPRE_ParVectorSetConstantValues ( HYPRE_ParVector vector , double value );
int HYPRE_ParVectorSetRandomValues ( HYPRE_ParVector vector , int seed );
int HYPRE_ParVectorCopy ( HYPRE_ParVector x , HYPRE_ParVector y );
HYPRE_ParVector HYPRE_ParVectorCloneShallow ( HYPRE_ParVector x );
int HYPRE_ParVectorScale ( double value , HYPRE_ParVector x );
int HYPRE_ParVectorAxpy ( double alpha , HYPRE_ParVector x , HYPRE_ParVector y );
int HYPRE_ParVectorInnerProd ( HYPRE_ParVector x , HYPRE_ParVector y , double *prod );

/* new_commpkg.c */
int PrintCommpkg ( hypre_ParCSRMatrix *A , const char *file_name );
int hypre_NewCommPkgCreate_core ( MPI_Comm comm , HYPRE_BigInt *col_map_off_d , HYPRE_BigInt first_col_diag , HYPRE_BigInt col_start , HYPRE_BigInt col_end , int num_cols_off_d , HYPRE_BigInt global_num_cols , int *p_num_recvs , int **p_recv_procs , int **p_recv_vec_starts , int *p_num_sends , int **p_send_procs , int **p_send_map_starts , int **p_send_map_elements , hypre_IJAssumedPart *apart );
int hypre_NewCommPkgCreate ( hypre_ParCSRMatrix *parcsr_A );
int hypre_NewCommPkgDestroy ( hypre_ParCSRMatrix *parcsr_A );
int hypre_RangeFillResponseIJDetermineRecvProcs ( void *p_recv_contact_buf , int contact_size , int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , int *response_message_size );
int hypre_FillResponseIJDetermineSendProcs ( void *p_recv_contact_buf , int contact_size , int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , int *response_message_size );

/* par_csr_assumed_part.c */
int hypre_LocateAssummedPartition ( MPI_Comm comm , HYPRE_BigInt row_start , HYPRE_BigInt row_end , HYPRE_BigInt global_num_rows , hypre_IJAssumedPart *part , int myid );
int hypre_ParCSRMatrixCreateAssumedPartition ( hypre_ParCSRMatrix *matrix );
int hypre_ParCSRMatrixDestroyAssumedPartition ( hypre_ParCSRMatrix *matrix );
int hypre_GetAssumedPartitionProcFromRow ( MPI_Comm comm , HYPRE_BigInt row , HYPRE_BigInt global_num_rows , int *proc_id );
int hypre_GetAssumedPartitionRowRange ( MPI_Comm comm , int proc_id , HYPRE_BigInt global_num_rows , HYPRE_BigInt *row_start , HYPRE_BigInt *row_end );
int hypre_ParVectorCreateAssumedPartition ( hypre_ParVector *vector );
int hypre_ParVectorDestroyAssumedPartition ( hypre_ParVector *vector );

/* par_csr_communication.c */
hypre_ParCSRCommHandle *hypre_ParCSRCommHandleCreate ( int job , hypre_ParCSRCommPkg *comm_pkg , void *send_data , void *recv_data );
int hypre_ParCSRCommHandleDestroy ( hypre_ParCSRCommHandle *comm_handle );
void hypre_MatvecCommPkgCreate_core ( MPI_Comm comm , HYPRE_BigInt *col_map_offd , HYPRE_BigInt first_col_diag , HYPRE_BigInt *col_starts , int num_cols_diag , int num_cols_offd , HYPRE_BigInt firstColDiag , HYPRE_BigInt *colMapOffd , int data , int *p_num_recvs , int **p_recv_procs , int **p_recv_vec_starts , int *p_num_sends , int **p_send_procs , int **p_send_map_starts , int **p_send_map_elmts );
int hypre_MatvecCommPkgCreate ( hypre_ParCSRMatrix *A );
int hypre_MatvecCommPkgDestroy ( hypre_ParCSRCommPkg *comm_pkg );
int hypre_BuildCSRMatrixMPIDataType ( int num_nonzeros , int num_rows , double *a_data , int *a_i , int *a_j , MPI_Datatype *csr_matrix_datatype );
int hypre_BuildCSRJDataType ( int num_nonzeros , double *a_data , int *a_j , MPI_Datatype *csr_jdata_datatype );

/* par_csr_matop.c */
void hypre_ParMatmul_RowSizes ( int **C_diag_i , int **C_offd_i , int **B_marker , int *A_diag_i , int *A_diag_j , int *A_offd_i , int *A_offd_j , int *B_diag_i , int *B_diag_j , int *B_offd_i , int *B_offd_j , int *B_ext_diag_i , int *B_ext_diag_j , int *B_ext_offd_i , int *B_ext_offd_j , int *map_B_to_C , int *C_diag_size , int *C_offd_size , int num_rows_diag_A , int num_cols_offd_A , int allsquare , int num_cols_diag_B , int num_cols_offd_B , int num_cols_offd_C );
hypre_ParCSRMatrix *hypre_ParMatmul ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B );
hypre_CSRMatrix *hypre_ParCSRMatrixExtractConvBExt ( hypre_ParCSRMatrix *B , hypre_ParCSRMatrix *A , int data );
hypre_BigCSRMatrix *hypre_ParCSRMatrixExtractBigExt ( hypre_ParCSRMatrix *B , hypre_ParCSRMatrix *A , int data );
int hypre_ParCSRMatrixTranspose ( hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **AT_ptr, int data);

/* par_csr_matop_marked.c */
void hypre_ParMatmul_RowSizes_Marked ( int **C_diag_i , int **C_offd_i , int **B_marker , int *A_diag_i , int *A_diag_j , int *A_offd_i , int *A_offd_j , int *B_diag_i , int *B_diag_j , int *B_offd_i , int *B_offd_j , int *B_ext_diag_i , int *B_ext_diag_j , int *B_ext_offd_i , int *B_ext_offd_j , int *map_B_to_C , int *C_diag_size , int *C_offd_size , int num_rows_diag_A , int num_cols_offd_A , int allsquare , int num_cols_diag_B , int num_cols_offd_B , int num_cols_offd_C , int *CF_marker , int *dof_func , int *dof_func_offd );
hypre_ParCSRMatrix *hypre_ParMatmul_FC ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *P , int *CF_marker , int *dof_func , int *dof_func_offd );
void hypre_ParMatScaleDiagInv_F ( hypre_ParCSRMatrix *C , hypre_ParCSRMatrix *A , double weight , int *CF_marker );
hypre_ParCSRMatrix *hypre_ParMatMinus_F ( hypre_ParCSRMatrix *P , hypre_ParCSRMatrix *C , int *CF_marker );
void hypre_ParCSRMatrixZero_F ( hypre_ParCSRMatrix *P , int *CF_marker );
void hypre_ParCSRMatrixCopy_C ( hypre_ParCSRMatrix *P , hypre_ParCSRMatrix *C , int *CF_marker );
void hypre_ParCSRMatrixDropEntries ( hypre_ParCSRMatrix *C , hypre_ParCSRMatrix *P , int *CF_marker );

/* par_csr_matrix.c */
hypre_ParCSRMatrix *hypre_ParCSRMatrixCreate ( MPI_Comm comm , HYPRE_BigInt global_num_rows , HYPRE_BigInt global_num_cols , HYPRE_BigInt *row_starts , HYPRE_BigInt *col_starts , int num_cols_offd , int num_nonzeros_diag , int num_nonzeros_offd );
int hypre_ParCSRMatrixDestroy ( hypre_ParCSRMatrix *matrix );
int hypre_ParCSRMatrixInitialize ( hypre_ParCSRMatrix *matrix );
int hypre_ParCSRMatrixSetNumNonzeros ( hypre_ParCSRMatrix *matrix );
int hypre_ParCSRMatrixSetDNumNonzeros ( hypre_ParCSRMatrix *matrix );
int hypre_ParCSRMatrixSetDataOwner ( hypre_ParCSRMatrix *matrix , int owns_data );
int hypre_ParCSRMatrixSetRowStartsOwner ( hypre_ParCSRMatrix *matrix , int owns_row_starts );
int hypre_ParCSRMatrixSetColStartsOwner ( hypre_ParCSRMatrix *matrix , int owns_col_starts );
hypre_ParCSRMatrix *hypre_ParCSRMatrixRead ( MPI_Comm comm , const char *file_name );
int hypre_ParCSRMatrixPrint ( hypre_ParCSRMatrix *matrix , const char *file_name );
int hypre_ParCSRMatrixPrintIJ ( const hypre_ParCSRMatrix *matrix , const int base_i , const int base_j , const char *filename );
int hypre_ParCSRMatrixGetLocalRange ( hypre_ParCSRMatrix *matrix , HYPRE_BigInt *row_start , HYPRE_BigInt *row_end , HYPRE_BigInt *col_start , HYPRE_BigInt *col_end );
int hypre_ParCSRMatrixGetRow ( hypre_ParCSRMatrix *mat , HYPRE_BigInt row , int *size , HYPRE_BigInt **col_ind , double **values );
int hypre_ParCSRMatrixRestoreRow ( hypre_ParCSRMatrix *matrix , HYPRE_BigInt row , int *size , HYPRE_BigInt **col_ind , double **values );
hypre_ParCSRMatrix *hypre_CSRMatrixToParCSRMatrix ( MPI_Comm comm , hypre_CSRMatrix *A , HYPRE_BigInt *row_starts , HYPRE_BigInt *col_starts );
int GenerateDiagAndOffd ( hypre_CSRMatrix *A , hypre_ParCSRMatrix *matrix , HYPRE_BigInt first_col_diag , HYPRE_BigInt last_col_diag );
hypre_CSRMatrix *hypre_MergeDiagAndOffd ( hypre_ParCSRMatrix *par_matrix );
hypre_CSRMatrix *hypre_ParCSRMatrixToCSRMatrixAll ( hypre_ParCSRMatrix *par_matrix );
int hypre_ParCSRMatrixCopy ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B , int copy_data );
int hypre_FillResponseParToCSRMatrix ( void *p_recv_contact_buf , int contact_size , int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , int *response_message_size );
hypre_ParCSRMatrix *hypre_ParCSRMatrixCompleteClone ( hypre_ParCSRMatrix *A );
hypre_ParCSRMatrix *hypre_ParCSRMatrixUnion ( hypre_ParCSRMatrix *A , hypre_ParCSRMatrix *B );

/* par_csr_matvec.c */
int hypre_ParCSRMatrixMatvec ( double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y );
int hypre_ParCSRMatrixMatvecT ( double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y );
int hypre_ParCSRMatrixMatvec_FF ( double alpha , hypre_ParCSRMatrix *A , hypre_ParVector *x , double beta , hypre_ParVector *y , int *CF_marker , int fpt );

/* par_vector.c */
hypre_ParVector *hypre_ParVectorCreate ( MPI_Comm comm , HYPRE_BigInt global_size , HYPRE_BigInt *partitioning );
int hypre_ParVectorDestroy ( hypre_ParVector *vector );
int hypre_ParVectorInitialize ( hypre_ParVector *vector );
int hypre_ParVectorSetDataOwner ( hypre_ParVector *vector , int owns_data );
int hypre_ParVectorSetPartitioningOwner ( hypre_ParVector *vector , int owns_partitioning );
int hypre_ParVectorSetNumVectors ( hypre_ParVector *vector , int num_vectors );
hypre_ParVector *hypre_ParVectorRead ( MPI_Comm comm , const char *file_name );
int hypre_ParVectorPrint ( hypre_ParVector *vector , const char *file_name );
int hypre_ParVectorSetConstantValues ( hypre_ParVector *v , double value );
int hypre_ParVectorSetRandomValues ( hypre_ParVector *v , int seed );
int hypre_ParVectorCopy ( hypre_ParVector *x , hypre_ParVector *y );
hypre_ParVector *hypre_ParVectorCloneShallow ( hypre_ParVector *x );
int hypre_ParVectorScale ( double alpha , hypre_ParVector *y );
int hypre_ParVectorAxpy ( double alpha , hypre_ParVector *x , hypre_ParVector *y );
double hypre_ParVectorInnerProd ( hypre_ParVector *x , hypre_ParVector *y );
hypre_Vector *hypre_ParVectorToVectorAll ( hypre_ParVector *par_v );
int hypre_ParVectorPrintIJ ( hypre_ParVector *vector , int base_j , const char *filename );
int hypre_ParVectorReadIJ ( MPI_Comm comm , const char *filename , int *base_j_ptr , hypre_ParVector **vector_ptr );
int hypre_FillResponseParToVectorAll ( void *p_recv_contact_buf , int contact_size , int contact_proc , void *ro , MPI_Comm comm , void **p_send_response_buf , int *response_message_size );
double hypre_ParVectorLocalSumElts ( hypre_ParVector *vector );

#ifdef __cplusplus
}
#endif

#endif

