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
 
/*--------------------------------------------------------------------------
 * hypre_GenerateLaplacian
 *--------------------------------------------------------------------------*/

HYPRE_ParCSRMatrix 
GenerateLaplacian( MPI_Comm comm,
                   HYPRE_BigInt      nx,
                   HYPRE_BigInt      ny,
                   HYPRE_BigInt      nz, 
                   int      P,
                   int      Q,
                   int      R,
                   int      p,
                   int      q,
                   int      r,
                   double  *value,
                   HYPRE_ParVector *rhs_ptr,
                   HYPRE_ParVector *x_ptr)
{
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;
   hypre_ParVector *par_rhs;
   hypre_ParVector *par_x;
   hypre_Vector *rhs;
   hypre_Vector *x;
   double *rhs_data;
   double *x_data;

   int    *diag_i;
   int    *diag_j;
   double *diag_data;

   int    *offd_i;
   int    *offd_j = NULL;
   double *offd_data = NULL;

   HYPRE_BigInt *global_part;
   HYPRE_BigInt *global_part_rhs;
   HYPRE_BigInt *global_part_x;
   int    *tmp_j = NULL;
   int ix, iy, iz;
   int cnt, o_cnt;
   int local_num_rows; 
   HYPRE_BigInt *col_map_offd = NULL;
   int row_index;
   int i, j;
   int ip, iq, ir;

   int nx_local, ny_local, nz_local;
   int num_cols_offd;
   HYPRE_BigInt grid_size;

   HYPRE_BigInt *nx_part;
   HYPRE_BigInt *ny_part;
   HYPRE_BigInt *nz_part;

   int num_procs, my_id;
   int P_busy, Q_busy, R_busy;

#ifdef HYPRE_NO_GLOBAL_PARTITION
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   hypre_GenerateLocalPartitioning(nx,P,p,&nx_part);
   hypre_GenerateLocalPartitioning(ny,Q,q,&ny_part);
   hypre_GenerateLocalPartitioning(nz,R,r,&nz_part);

   nx_local = (int)(nx_part[1] - nx_part[0]);
   ny_local = (int)(ny_part[1] - ny_part[0]);
   nz_local = (int)(nz_part[1] - nz_part[0]);
   local_num_rows = nx_local*ny_local*nz_local;

   global_part = hypre_CTAlloc(HYPRE_BigInt,2);
   global_part_rhs = hypre_CTAlloc(HYPRE_BigInt,2);
   global_part_x = hypre_CTAlloc(HYPRE_BigInt,2);

   global_part[0] = (HYPRE_BigInt)my_id*(HYPRE_BigInt)local_num_rows;
   global_part[1] = global_part[0]+(HYPRE_BigInt)local_num_rows;

   for (i=0; i< 2; i++)
   {
      global_part_x[i] = global_part[i];
      global_part_rhs[i] = global_part[i];
   }
   ip = p;
   iq = q;
   ir = r;
#else
   int nx_size, ny_size, nz_size;
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   hypre_GeneratePartitioning(nx,P,&nx_part);
   hypre_GeneratePartitioning(ny,Q,&ny_part);
   hypre_GeneratePartitioning(nz,R,&nz_part);

   global_part = hypre_CTAlloc(HYPRE_BigInt,P*Q*R+1);
   global_part_rhs = hypre_CTAlloc(HYPRE_BigInt,P*Q*R+1);
   global_part_x = hypre_CTAlloc(HYPRE_BigInt,P*Q*R+1);

   global_part[0] = 0;
   global_part_rhs[0] = 0;
   global_part_x[0] = 0;
   cnt = 1;
   for (iz = 0; iz < R; iz++)
   {
      nz_size = (int)(nz_part[iz+1]-nz_part[iz]);
      for (iy = 0; iy < Q; iy++)
      {
         ny_size = (int)(ny_part[iy+1]-ny_part[iy]);
         for (ix = 0; ix < P; ix++)
         {
            nx_size = (int)(nx_part[ix+1] - nx_part[ix]);
            global_part[cnt] = global_part[cnt-1];
            global_part[cnt] += (HYPRE_BigInt)(nx_size*ny_size*nz_size);
            global_part_x[cnt] = global_part[cnt];
            global_part_rhs[cnt] = global_part[cnt];
            cnt++;
         }
      }
   }

   nx_local = (int)(nx_part[p+1] - nx_part[p]);
   ny_local = (int)(ny_part[q+1] - ny_part[q]);
   nz_local = (int)(nz_part[r+1] - nz_part[r]);
   local_num_rows = nx_local*ny_local*nz_local;
   ip = p;
   iq = q;
   ir = r;
#endif

   grid_size = nx*ny*nz;

   diag_i = hypre_CTAlloc(int, local_num_rows+1);
   offd_i = hypre_CTAlloc(int, local_num_rows+1);
   rhs_data = hypre_CTAlloc(double, local_num_rows);
   x_data = hypre_CTAlloc(double, local_num_rows);

   for (i=0; i < local_num_rows; i++)
   {
      x_data[i] = 0.0;
      rhs_data[i] = 1.0;
   }

   P_busy = hypre_min(nx,P);
   Q_busy = hypre_min(ny,Q);
   R_busy = hypre_min(nz,R);

   num_cols_offd = 0;
   if (p) num_cols_offd += ny_local*nz_local;
   if (p < P_busy-1) num_cols_offd += ny_local*nz_local;
   if (q) num_cols_offd += nx_local*nz_local;
   if (q < Q_busy-1) num_cols_offd += nx_local*nz_local;
   if (r) num_cols_offd += nx_local*ny_local;
   if (r < R_busy-1) num_cols_offd += nx_local*ny_local;

   if (!local_num_rows) num_cols_offd = 0;


   if (num_cols_offd)
   {
      col_map_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd);
   }
   
   cnt = 1;
   o_cnt = 1;
   diag_i[0] = 0;
   offd_i[0] = 0;
   for (iz = 0; iz < nz_local; iz++)
   {
      for (iy = 0;  iy < ny_local; iy++)
      {
         for (ix = 0; ix < nx_local; ix++)
         {
            diag_i[cnt] = diag_i[cnt-1];
            offd_i[o_cnt] = offd_i[o_cnt-1];
            diag_i[cnt]++;
            if (iz) 
               diag_i[cnt]++;
            else
            {
               if (ir) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iy) 
               diag_i[cnt]++;
            else
            {
               if (iq) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (ix) 
               diag_i[cnt]++;
            else
            {
               if (ip) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (ix+1 < nx_local) 
               diag_i[cnt]++;
            else
            {
               if (ip+1 < P) 
               {
                  offd_i[o_cnt]++; 
               }
            }
            if (iy+1 < ny_local) 
               diag_i[cnt]++;
            else
            {
               if (iq+1 < Q) 
               {
                  offd_i[o_cnt]++;
               }
            }
            if (iz+1 < nz_local) 
               diag_i[cnt]++;
            else
            {
               if (ir+1 < R) 
               {
                  offd_i[o_cnt]++;
               }
            }
            cnt++;
            o_cnt++;
         }
      }
   }


   diag_j = hypre_CTAlloc(int, diag_i[local_num_rows]);
   diag_data = hypre_CTAlloc(double, diag_i[local_num_rows]);

   if (num_procs > 1)
   {
      offd_j = hypre_CTAlloc(int, offd_i[local_num_rows]);
      offd_data = hypre_CTAlloc(double, offd_i[local_num_rows]);
   }

   row_index = 0;
   cnt = 0;
   o_cnt = 0;
   for (iz = 0; iz < nz_local; iz++)
   {
      for (iy = 0;  iy < ny_local; iy++)
      {
         for (ix = 0; ix < nx_local; ix++)
         {
            diag_j[cnt] = row_index;
            diag_data[cnt++] = value[0];
            if (iz) 
            {
               diag_j[cnt] = row_index-nx_local*ny_local;
               diag_data[cnt++] = value[3];
            }
            else
            {
               if (ir) 
               {
                  offd_j[o_cnt] = o_cnt;
		  hypre_map(ix,iy,iz-1,p,q,r-1,P,Q,R,
                         nx_part,ny_part,nz_part,global_part,
			 &col_map_offd[o_cnt]);
                  offd_data[o_cnt++] = value[3];
               }
            }
            if (iy) 
            {
               diag_j[cnt] = row_index-nx_local;
               diag_data[cnt++] = value[2];
            }
            else
            {
               if (iq) 
               {
                  offd_j[o_cnt] = o_cnt;
		  hypre_map(ix,iy-1,iz,p,q-1,r,P,Q,R,
                         nx_part,ny_part,nz_part,global_part,
			&col_map_offd[o_cnt]);
                  offd_data[o_cnt++] = value[2];
               }
            }
            if (ix) 
            {
               diag_j[cnt] = row_index-1;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ip) 
               {
                  offd_j[o_cnt] = o_cnt;
		  hypre_map(ix-1,iy,iz,p-1,q,r,P,Q,R,
                         nx_part,ny_part,nz_part,global_part,
			&col_map_offd[o_cnt]);
                  offd_data[o_cnt++] = value[1];
               }
            }
            if (ix+1 < nx_local) 
            {
               diag_j[cnt] = row_index+1;
               diag_data[cnt++] = value[1];
            }
            else
            {
               if (ip+1 < P) 
               {
                  offd_j[o_cnt] = o_cnt;
		  hypre_map(ix+1,iy,iz,p+1,q,r,P,Q,R,
                         nx_part,ny_part,nz_part,global_part,
			&col_map_offd[o_cnt]);
                  offd_data[o_cnt++] = value[1];
               }
            }
            if (iy+1 < ny_local) 
            {
               diag_j[cnt] = row_index+nx_local;
               diag_data[cnt++] = value[2];
            }
            else
            {
               if (iq+1 < Q) 
               {
                  offd_j[o_cnt] = o_cnt;
		  hypre_map(ix,iy+1,iz,p,q+1,r,P,Q,R,
                         nx_part,ny_part,nz_part,global_part,
			&col_map_offd[o_cnt]);
                  offd_data[o_cnt++] = value[2];
               }
            }
            if (iz+1 < nz_local) 
            {
               diag_j[cnt] = row_index+nx_local*ny_local;
               diag_data[cnt++] = value[3];
            }
            else
            {
               if (ir+1 < R) 
               {
                  offd_j[o_cnt] = o_cnt;
		  hypre_map(ix,iy,iz+1,p,q,r+1,P,Q,R,
                         nx_part,ny_part,nz_part,global_part,
			&col_map_offd[o_cnt]);
                  offd_data[o_cnt++] = value[3];
               }
            }
            row_index++;
         }
      }
   }


   if (num_cols_offd) tmp_j = hypre_CTAlloc(int, num_cols_offd);
   for (i=0; i < num_cols_offd; i++)
      tmp_j[i] = offd_j[i];


   if (num_procs > 1)
   {
      hypre_BigQsortbi(col_map_offd, tmp_j, 0, num_cols_offd-1);

   for (i=0; i < num_cols_offd; i++)
      for (j=0; j < num_cols_offd; j++)
	 if (offd_j[i] == tmp_j[j])
	 {
	    offd_j[i] = j;
	    break;
         }
   }


   hypre_TFree(tmp_j);

   A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
                                global_part, global_part, num_cols_offd,
                                diag_i[local_num_rows],
                                offd_i[local_num_rows]);

   par_rhs = hypre_ParVectorCreate(comm, grid_size, global_part_rhs);
   rhs = hypre_ParVectorLocalVector(par_rhs);
   hypre_VectorData(rhs) = rhs_data;

   par_x = hypre_ParVectorCreate(comm, grid_size, global_part_x);
   x = hypre_ParVectorLocalVector(par_x);
   hypre_VectorData(x) = x_data;

   hypre_ParCSRMatrixColMapOffd(A) = col_map_offd;

   diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_data;

   offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrixI(offd) = offd_i;
   if (num_cols_offd)
   {
      hypre_CSRMatrixJ(offd) = offd_j;
      hypre_CSRMatrixData(offd) = offd_data;
   }

   hypre_TFree(nx_part);
   hypre_TFree(ny_part);
   hypre_TFree(nz_part);

   *rhs_ptr = (HYPRE_ParVector) par_rhs;
   *x_ptr = (HYPRE_ParVector) par_x;

   return (HYPRE_ParCSRMatrix) A;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
hypre_map( int  ix,
     int  iy,
     int  iz,
     int  p,
     int  q,
     int  r,
     int  P,
     int  Q,
     int  R, 
     HYPRE_BigInt *nx_part,
     HYPRE_BigInt *ny_part,
     HYPRE_BigInt *nz_part,
     HYPRE_BigInt *global_part,
     HYPRE_BigInt *value_ptr)
{
   int nx_local;
   int ny_local;
   int nz_local;
   HYPRE_BigInt global_index;
   int proc_num;
 
#ifdef HYPRE_NO_GLOBAL_PARTITION
   proc_num = r*P*Q + q*P + p;
   nx_local = (int)(nx_part[1] - nx_part[0]);
   ny_local = (int)(ny_part[1] - ny_part[0]);
   nz_local = (int)(nz_part[1] - nz_part[0]);
   if (ix < 0) ix += nx_local;
   if (ix >= nx_local) ix -= nx_local;
   if (iy < 0) iy += ny_local;
   if (iy >= ny_local) iy -= ny_local;
   if (iz < 0) iz += nz_local;
   if (iz >= nz_local) iz -= nz_local;
   global_index = (HYPRE_BigInt)proc_num
	*(HYPRE_BigInt)(nx_local*ny_local*nz_local)
      + (HYPRE_BigInt)((iz*ny_local+iy)*nx_local + ix);
#else
   proc_num = r*P*Q + q*P + p;
   nx_local = (int)(nx_part[p+1] - nx_part[p]);
   ny_local = (int)(ny_part[q+1] - ny_part[q]);
   nz_local = (int)(nz_part[r+1] - nz_part[r]);
   if (ix < 0) ix += nx_local;
   if (ix >= nx_local) ix  = 0;
   if (iy < 0) iy += ny_local;
   if (iy >= ny_local) iy  = 0;
   if (iz < 0) iz += nz_local;
   if (iz >= nz_local) iz  = 0;
   global_index = global_part[proc_num] 
      + (HYPRE_BigInt)((iz*ny_local+iy)*nx_local + ix);
#endif
   *value_ptr = global_index;
 
   return 0;
}
