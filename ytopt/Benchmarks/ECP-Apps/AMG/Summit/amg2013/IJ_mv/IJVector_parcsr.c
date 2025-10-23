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
 * IJVector_Par interface
 *
 *****************************************************************************/
 
#include "headers.h"
#include "../HYPRE.h"

/******************************************************************************
 *
 * hypre_IJVectorCreatePar
 *
 * creates ParVector if necessary, and leaves a pointer to it as the
 * hypre_IJVector object
 *
 *****************************************************************************/
int
hypre_IJVectorCreatePar(hypre_IJVector *vector, HYPRE_BigInt *IJpartitioning)
{
   MPI_Comm comm = hypre_IJVectorComm(vector);


   int num_procs, j;
   HYPRE_BigInt global_n, *partitioning, jmin;
   MPI_Comm_size(comm, &num_procs);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   jmin = hypre_IJVectorGlobalFirstRow(vector);
   global_n = hypre_IJVectorGlobalNumRows(vector);

   partitioning = hypre_CTAlloc(HYPRE_BigInt, 2); 

/* Shift to zero-based partitioning for ParVector object */
   for (j = 0; j < 2; j++) 
      partitioning[j] = IJpartitioning[j] - jmin;

#else
   jmin = IJpartitioning[0];
   global_n = IJpartitioning[num_procs] - jmin;

   partitioning = hypre_CTAlloc(HYPRE_BigInt, num_procs+1); 

/* Shift to zero-based partitioning for ParVector object */
   for (j = 0; j < num_procs+1; j++) 
      partitioning[j] = IJpartitioning[j] - jmin;

#endif
  

   hypre_IJVectorObject(vector) = hypre_ParVectorCreate(comm,
            global_n, (HYPRE_BigInt *) partitioning); 

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorDestroyPar
 *
 * frees ParVector local storage of an IJVectorPar 
 *
 *****************************************************************************/
int
hypre_IJVectorDestroyPar(hypre_IJVector *vector)
{
   return hypre_ParVectorDestroy(hypre_IJVectorObject(vector));
}

/******************************************************************************
 *
 * hypre_IJVectorInitializePar
 *
 * initializes ParVector of IJVectorPar
 *
 *****************************************************************************/
int
hypre_IJVectorInitializePar(hypre_IJVector *vector)
{
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   hypre_AuxParVector *aux_vector = hypre_IJVectorTranslator(vector);
   HYPRE_BigInt *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);
   int my_id;
   
   MPI_Comm  comm = hypre_IJVectorComm(vector);

   MPI_Comm_rank(comm,&my_id);
  
   if (!partitioning)
   {
      printf("No ParVector partitioning for initialization -- ");
      printf("hypre_IJVectorInitializePar\n"); 
      hypre_error_in_arg(1);
   }

#ifdef HYPRE_NO_GLOBAL_PARTITION
   hypre_VectorSize(local_vector) = (int) (partitioning[1] -
                                    partitioning[0]);
#else
   hypre_VectorSize(local_vector) = (int) (partitioning[my_id+1] -
                                    partitioning[my_id]);
#endif


   hypre_ParVectorInitialize(par_vector);

   if (!aux_vector)
   {  
      hypre_AuxParVectorCreate(&aux_vector);
      hypre_IJVectorTranslator(vector) = aux_vector;
   }
   hypre_AuxParVectorInitialize(aux_vector);

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorSetMaxOffProcElmtsPar
 *
 *****************************************************************************/

int
hypre_IJVectorSetMaxOffProcElmtsPar(hypre_IJVector *vector,
                                    int max_off_proc_elmts)
{
  
   hypre_AuxParVector *aux_vector;

   aux_vector = hypre_IJVectorTranslator(vector);
   if (!aux_vector)
   {
      hypre_AuxParVectorCreate(&aux_vector);
      hypre_IJVectorTranslator(vector) = aux_vector;
   }
   hypre_AuxParVectorMaxOffProcElmts(aux_vector) = max_off_proc_elmts;
   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorDistributePar
 *
 * takes an IJVector generated for one processor and distributes it
 * across many processors according to vec_starts,
 * if vec_starts is NULL, it distributes them evenly?
 *
 *****************************************************************************/
/*int
hypre_IJVectorDistributePar(hypre_IJVector *vector,
			    const int	   *vec_starts)
{

   hypre_ParVector *old_vector = hypre_IJVectorObject(vector);
   hypre_ParVector *par_vector;
   
   if (!old_vector)
   {
      printf("old_vector == NULL -- ");
      printf("hypre_IJVectorDistributePar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }

   par_vector = hypre_VectorToParVector(hypre_ParVectorComm(old_vector),
		                        hypre_ParVectorLocalVector(old_vector),
                                        (int *)vec_starts);
   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorDistributePar\n");
      printf("**** Vector storage is unallocated ****\n");
      hypre_error_in_arg(1);
   }

   hypre_ParVectorDestroy(old_vector);

   hypre_IJVectorObject(vector) = par_vector;

   return hypre_error_flag;
}*/

/******************************************************************************
 *
 * hypre_IJVectorZeroValuesPar
 *
 * zeroes all local components of an IJVectorPar
 *
 *****************************************************************************/
int
hypre_IJVectorZeroValuesPar(hypre_IJVector *vector)
{
   int my_id;
   int i;
   HYPRE_BigInt vec_start, vec_stop;
   double *data;

   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);
   HYPRE_BigInt *partitioning = hypre_ParVectorPartitioning(par_vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorZeroValuesPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }
   if (!partitioning)
   {
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorZeroValuesPar\n");
      printf("**** Vector partitioning is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorZeroValuesPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }

#ifdef HYPRE_NO_GLOBAL_PARTITION
   vec_start = partitioning[0];
   vec_stop  = partitioning[1];
#else
   vec_start = partitioning[my_id];
   vec_stop  = partitioning[my_id+1];
#endif


   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorZeroValuesPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      hypre_error_in_arg(1);
   
   }

   data = hypre_VectorData( local_vector );
   for (i = 0; i < (int)(vec_stop - vec_start); i++)
      data[i] = 0.;
  
   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorSetValuesPar
 *
 * sets a potentially noncontiguous set of components of an IJVectorPar
 *
 *****************************************************************************/
int
hypre_IJVectorSetValuesPar(hypre_IJVector *vector,
                                    int             num_values,
                                    const HYPRE_BigInt *indices,
                                    const double   *values            )
{

   int my_id;
   int j, k;
   HYPRE_BigInt i, vec_start, vec_stop;
   double *data;

   HYPRE_BigInt *IJpartitioning = hypre_IJVectorPartitioning(vector);
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   hypre_AuxParVector *aux_vector = hypre_IJVectorTranslator(vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be set, perform no checking and return */
   if (num_values < 1) return 0;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorSetValuesPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }
   if (!IJpartitioning)
   {
      printf("IJpartitioning == NULL -- ");
      printf("hypre_IJVectorSetValuesPar\n");
      printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorSetValuesPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }

#ifdef HYPRE_NO_GLOBAL_PARTITION
   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1]-1;
#else
   vec_start = IJpartitioning[my_id];
   vec_stop  = IJpartitioning[my_id+1]-1;
#endif

   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorSetValuesPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      hypre_error_in_arg(1);
   }

/* Determine whether indices points to local indices only,
   and if not, store indices and values into auxiliary vector structure
   If indices == NULL, assume that num_values components are to be
   set in a block starting at vec_start. 
   NOTE: If indices == NULL off processor values are ignored!!! */

   data = hypre_VectorData(local_vector);

   if (indices)
   {
      int current_num_elmts
                = hypre_AuxParVectorCurrentNumElmts(aux_vector);
      int max_off_proc_elmts
                = hypre_AuxParVectorMaxOffProcElmts(aux_vector);
      HYPRE_BigInt *off_proc_i = hypre_AuxParVectorOffProcI(aux_vector);
      double *off_proc_data = hypre_AuxParVectorOffProcData(aux_vector);

      for (j = 0; j < num_values; j++)
      {
	 i = indices[j];
	 if (i < vec_start || i > vec_stop)
         {
	 /* if elements outside processor boundaries, store in off processor
		stash */
	    if (!max_off_proc_elmts)
            {
               max_off_proc_elmts = 100;
               hypre_AuxParVectorMaxOffProcElmts(aux_vector) =
                        max_off_proc_elmts;
               hypre_AuxParVectorOffProcI(aux_vector)
                        = hypre_CTAlloc(HYPRE_BigInt,max_off_proc_elmts);
               hypre_AuxParVectorOffProcData(aux_vector)
                        = hypre_CTAlloc(double,max_off_proc_elmts);
               off_proc_i = hypre_AuxParVectorOffProcI(aux_vector);
               off_proc_data = hypre_AuxParVectorOffProcData(aux_vector);
            }
            else if (current_num_elmts + 1 > max_off_proc_elmts)
            {
               max_off_proc_elmts += 10;
               off_proc_i = hypre_TReAlloc(off_proc_i,HYPRE_BigInt,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
                                max_off_proc_elmts);
               hypre_AuxParVectorMaxOffProcElmts(aux_vector)
                        = max_off_proc_elmts;
               hypre_AuxParVectorOffProcI(aux_vector) = off_proc_i;
               hypre_AuxParVectorOffProcData(aux_vector) = off_proc_data;
            }
            off_proc_i[current_num_elmts] = i;
            off_proc_data[current_num_elmts++] = values[j];
            hypre_AuxParVectorCurrentNumElmts(aux_vector)=current_num_elmts;
         }
         else /* local values are inserted into the vector */
         {
            k = (int) (i - vec_start);
            data[k] = values[j];
         }
      } 
   }
   else 
   {
      if (num_values > (int)(vec_stop - vec_start) + 1)
      {
         printf("Warning! Indices beyond local range  not identified!\n ");
         printf("Off processor values have been ignored!\n");
	 num_values = (int)(vec_stop - vec_start) +1;
      }

      for (j = 0; j < num_values; j++)
         data[j] = values[j];
   } 
  
   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorAddToValuesPar
 *
 * adds to a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/
int
hypre_IJVectorAddToValuesPar(hypre_IJVector *vector,
                             int             num_values,
                             const HYPRE_BigInt *indices,
                             const double   *values      )
{
   int my_id;
   int j, k;
   HYPRE_BigInt i, vec_start, vec_stop;
   double *data;

   HYPRE_BigInt *IJpartitioning = hypre_IJVectorPartitioning(vector);
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   hypre_AuxParVector *aux_vector = hypre_IJVectorTranslator(vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1) return 0;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorAddToValuesPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }
   if (!IJpartitioning)
   {
      printf("IJpartitioning == NULL -- ");
      printf("hypre_IJVectorAddToValuesPar\n");
      printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorAddToValuesPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }

#ifdef HYPRE_NO_GLOBAL_PARTITION
   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1]-1;
#else
   vec_start = IJpartitioning[my_id];
   vec_stop  = IJpartitioning[my_id+1]-1;
#endif

   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorAddToValuesPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      hypre_error_in_arg(1);
   }

/* Determine whether indices points to local indices only,
   and if not, store indices and values into auxiliary vector structure
   If indices == NULL, assume that num_values components are to be
   set in a block starting at vec_start. 
   NOTE: If indices == NULL off processor values are ignored!!! */

   /* if (indices)
   {
      for (i = 0; i < num_values; i++)
      { 	
	ierr += (indices[i] <  vec_start);
        ierr += (indices[i] >= vec_stop);
      }
   }

   if (ierr)
   {
      printf("indices beyond local range -- ");
      printf("hypre_IJVectorAddToValuesPar\n");
      printf("**** Indices specified are unusable ****\n");
      exit(1);
   } */
    
   data = hypre_VectorData(local_vector);

   if (indices)
   {
      int current_num_elmts
                = hypre_AuxParVectorCurrentNumElmts(aux_vector);
      int max_off_proc_elmts
                = hypre_AuxParVectorMaxOffProcElmts(aux_vector);
      HYPRE_BigInt *off_proc_i = hypre_AuxParVectorOffProcI(aux_vector);
      double *off_proc_data = hypre_AuxParVectorOffProcData(aux_vector);

      for (j = 0; j < num_values; j++)
      {
	 i = indices[j];
	 if (i < vec_start || i > vec_stop)
         {
	 /* if elements outside processor boundaries, store in off processor
		stash */
	    if (!max_off_proc_elmts)
            {
               max_off_proc_elmts = 100;
               hypre_AuxParVectorMaxOffProcElmts(aux_vector) =
                        max_off_proc_elmts;
               hypre_AuxParVectorOffProcI(aux_vector)
                        = hypre_CTAlloc(HYPRE_BigInt,max_off_proc_elmts);
               hypre_AuxParVectorOffProcData(aux_vector)
                        = hypre_CTAlloc(double,max_off_proc_elmts);
               off_proc_i = hypre_AuxParVectorOffProcI(aux_vector);
               off_proc_data = hypre_AuxParVectorOffProcData(aux_vector);
            }
            else if (current_num_elmts + 1 > max_off_proc_elmts)
            {
               max_off_proc_elmts += 10;
               off_proc_i = hypre_TReAlloc(off_proc_i,HYPRE_BigInt,max_off_proc_elmts);
               off_proc_data = hypre_TReAlloc(off_proc_data,double,
                                max_off_proc_elmts);
               hypre_AuxParVectorMaxOffProcElmts(aux_vector)
                        = max_off_proc_elmts;
               hypre_AuxParVectorOffProcI(aux_vector) = off_proc_i;
               hypre_AuxParVectorOffProcData(aux_vector) = off_proc_data;
            }
            off_proc_i[current_num_elmts] = -i-1;
            off_proc_data[current_num_elmts++] = values[j];
            hypre_AuxParVectorCurrentNumElmts(aux_vector)=current_num_elmts;
         }
         else /* local values are added to the vector */
         {
            k = (int) (i - vec_start);
            data[k] += values[j];
         }
      } 
   }
   else 
   {
      if (num_values > (int)(vec_stop - vec_start) + 1)
      {
         printf("Warning! Indices beyond local range  not identified!\n ");
         printf("Off processor values have been ignored!\n");
	 num_values = (int)(vec_stop - vec_start) +1;
      }

      for (j = 0; j < num_values; j++)
         data[j] += values[j];
   } 
  
   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJVectorAssemblePar
 *
 * currently tests existence of of ParVector object and its partitioning
 *
 *****************************************************************************/
int
hypre_IJVectorAssemblePar(hypre_IJVector *vector)
{
   HYPRE_BigInt *IJpartitioning = hypre_IJVectorPartitioning(vector);
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   hypre_AuxParVector *aux_vector = hypre_IJVectorTranslator(vector);
   HYPRE_BigInt *partitioning = hypre_ParVectorPartitioning(par_vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorAssemblePar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   } 
   if (!IJpartitioning)
   { 
      printf("IJpartitioning == NULL -- ");
      printf("hypre_IJVectorAssemblePar\n");
      printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }
   if (!partitioning)
   { 
      printf("partitioning == NULL -- ");
      printf("hypre_IJVectorAssemblePar\n");
      printf("**** ParVector partitioning is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }

   if (aux_vector)
   {
      int off_proc_elmts, current_num_elmts;
      int max_off_proc_elmts;
      HYPRE_BigInt *off_proc_i;
      double *off_proc_data;
      current_num_elmts = hypre_AuxParVectorCurrentNumElmts(aux_vector);
      MPI_Allreduce(&current_num_elmts,&off_proc_elmts,1,MPI_INT, MPI_SUM,comm);
      if (off_proc_elmts)
      {
         max_off_proc_elmts=hypre_AuxParVectorMaxOffProcElmts(aux_vector);
         off_proc_i=hypre_AuxParVectorOffProcI(aux_vector);
         off_proc_data=hypre_AuxParVectorOffProcData(aux_vector);
         hypre_IJVectorAssembleOffProcValsPar(vector, max_off_proc_elmts, 
		current_num_elmts, off_proc_i, off_proc_data);
	 hypre_TFree(hypre_AuxParVectorOffProcI(aux_vector));
	 hypre_TFree(hypre_AuxParVectorOffProcData(aux_vector));
	 hypre_AuxParVectorMaxOffProcElmts(aux_vector) = 0;
	 hypre_AuxParVectorCurrentNumElmts(aux_vector) = 0;
      }
   }

   return hypre_error_flag;
}
                                 
/******************************************************************************
 *
 * hypre_IJVectorGetValuesPar
 *
 * get a potentially noncontiguous set of IJVectorPar components
 *
 *****************************************************************************/
int
hypre_IJVectorGetValuesPar(hypre_IJVector *vector,
                           int             num_values,
                           const HYPRE_BigInt *indices,
                           double         *values      )
{
   int my_id;
   int j, k;
   HYPRE_BigInt i, vec_start, vec_stop;
   double *data;
   int ierr = 0;

   HYPRE_BigInt *IJpartitioning = hypre_IJVectorPartitioning(vector);
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   MPI_Comm comm = hypre_IJVectorComm(vector);
   hypre_Vector *local_vector = hypre_ParVectorLocalVector(par_vector);

/* If no components are to be retrieved, perform no checking and return */
   if (num_values < 1) return 0;

   MPI_Comm_rank(comm, &my_id);

/* If par_vector == NULL or partitioning == NULL or local_vector == NULL 
   let user know of catastrophe and exit */

   if (!par_vector)
   {
      printf("par_vector == NULL -- ");
      printf("hypre_IJVectorGetValuesPar\n");
      printf("**** Vector storage is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }
   if (!IJpartitioning)
   {
      printf("IJpartitioning == NULL -- ");
      printf("hypre_IJVectorGetValuesPar\n");
      printf("**** IJVector partitioning is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }
   if (!local_vector)
   {
      printf("local_vector == NULL -- ");
      printf("hypre_IJVectorGetValuesPar\n");
      printf("**** Vector local data is either unallocated or orphaned ****\n");
      hypre_error_in_arg(1);
   }

#ifdef HYPRE_NO_GLOBAL_PARTITION
   vec_start = IJpartitioning[0];
   vec_stop  = IJpartitioning[1];
#else
   vec_start = IJpartitioning[my_id];
   vec_stop  = IJpartitioning[my_id+1];
#endif

   if (vec_start > vec_stop) 
   {
      printf("vec_start > vec_stop -- ");
      printf("hypre_IJVectorGetValuesPar\n");
      printf("**** This vector partitioning should not occur ****\n");
      hypre_error_in_arg(1);
   }

/* Determine whether indices points to local indices only,
   and if not, let user know of catastrophe and exit.
   If indices == NULL, assume that num_values components are to be
   retrieved from block starting at vec_start */

   if (indices)
   {
      for (j = 0; j < num_values; i++)
      { 	
	ierr += (indices[j] <  vec_start);
        ierr += (indices[j] >= vec_stop);
      }
   }

   if (ierr)
   {
      printf("indices beyond local range -- ");
      printf("hypre_IJVectorGetValuesPar\n");
      printf("**** Indices specified are unusable ****\n");
      hypre_error_in_arg(3);
   }
    
   data = hypre_VectorData(local_vector);

   if (indices)
   {
      for (j = 0; j < num_values; j++)
      {
         k = (int)(indices[j] - vec_start);
         values[j] = data[k];
      }
   }
   else
   {
      for (j = 0; j < num_values; j++)
         values[j] = data[j];
   }

   return hypre_error_flag;
}

/******************************************************************************

 * hypre_IJVectorAssembleOffProcValsPar

 * This is for handling set and get values calls to off-proc. entries -
 * it is called from assemble.  There is an alternate version for
 * when the assumed partition is being used.

 *****************************************************************************/

#ifndef HYPRE_NO_GLOBAL_PARTITION

int
hypre_IJVectorAssembleOffProcValsPar( hypre_IJVector *vector, 
   				      int max_off_proc_elmts,
   				      int current_num_elmts,
   				      HYPRE_BigInt *off_proc_i,
   			     	      double *off_proc_data)
{
   MPI_Comm comm = hypre_IJVectorComm(vector);
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);
   MPI_Request *requests;
   MPI_Status *status;
   int i, j, j2;
   int iii, indx, ip;
   HYPRE_BigInt first_index, row;
   int proc_id, num_procs, my_id;
   int num_sends, num_sends2;
   int num_recvs;
   int num_requests;
   int vec_start, vec_len;
   int *send_procs;
   HYPRE_BigInt *send_i;
   int *send_map_starts;
   int *recv_procs;
   HYPRE_BigInt *recv_i;
   int *recv_vec_starts;
   int *info;
   int *int_buffer;
   int *proc_id_mem;
   HYPRE_BigInt *partitioning;
   int *displs;
   int *recv_buf;
   double *send_data;
   double *recv_data;
   double *data = hypre_VectorData(hypre_ParVectorLocalVector(par_vector));

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm, &my_id);
   partitioning = hypre_IJVectorPartitioning(vector);

   first_index = partitioning[my_id];


   info = hypre_CTAlloc(int,num_procs);  
   proc_id_mem = hypre_CTAlloc(int,current_num_elmts);
   for (i=0; i < current_num_elmts; i++)
   {
      row = off_proc_i[i]; 
      if (row < 0) row = -row-1;
      proc_id = hypre_FindProc(partitioning,row,num_procs);
      proc_id_mem[i] = proc_id; 
      info[proc_id]++;
   }

   /* determine send_procs and amount of data to be sent */   
   num_sends = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         num_sends++;
      }
   }
   num_sends2 = 2*num_sends;
   send_procs =  hypre_CTAlloc(int,num_sends);
   send_map_starts =  hypre_CTAlloc(int,num_sends+1);
   int_buffer =  hypre_CTAlloc(int,num_sends2);
   j = 0;
   j2 = 0;
   send_map_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      if (info[i])
      {
         send_procs[j++] = i;
         send_map_starts[j] = send_map_starts[j-1]+info[i];
         int_buffer[j2++] = i;
	 int_buffer[j2++] = info[i];
      }
   }

   MPI_Allgather(&num_sends2,1,MPI_INT,info,1,MPI_INT,comm);

   displs = hypre_CTAlloc(int, num_procs+1);
   displs[0] = 0;
   for (i=1; i < num_procs+1; i++)
        displs[i] = displs[i-1]+info[i-1];
   recv_buf = hypre_CTAlloc(int, displs[num_procs]);

   MPI_Allgatherv(int_buffer,num_sends2,MPI_INT,recv_buf,info,displs,
			MPI_INT,comm);

   hypre_TFree(int_buffer);
   hypre_TFree(info);

   /* determine recv procs and amount of data to be received */
   num_recvs = 0;
   for (j=0; j < displs[num_procs]; j+=2)
   {
      if (recv_buf[j] == my_id)
	 num_recvs++;
   }

   recv_procs = hypre_CTAlloc(int,num_recvs);
   recv_vec_starts = hypre_CTAlloc(int,num_recvs+1);

   j2 = 0;
   recv_vec_starts[0] = 0;
   for (i=0; i < num_procs; i++)
   {
      for (j=displs[i]; j < displs[i+1]; j+=2)
      {
         if (recv_buf[j] == my_id)
         {
	    recv_procs[j2++] = i;
	    recv_vec_starts[j2] = recv_vec_starts[j2-1]+recv_buf[j+1];
         }
         if (j2 == num_recvs) break;
      }
   }
   hypre_TFree(recv_buf);
   hypre_TFree(displs);

   /* set up data to be sent to send procs */
   /* send_i contains for each send proc 
      indices, send_data contains corresponding values */
      
   send_i = hypre_CTAlloc(HYPRE_BigInt,send_map_starts[num_sends]);
   send_data = hypre_CTAlloc(double,send_map_starts[num_sends]);
   recv_i = hypre_CTAlloc(HYPRE_BigInt,recv_vec_starts[num_recvs]);
   recv_data = hypre_CTAlloc(double,recv_vec_starts[num_recvs]);
    
   for (i=0; i < current_num_elmts; i++)
   {
      proc_id = proc_id_mem[i];
      indx = hypre_BinarySearch(send_procs,proc_id,num_sends);
      iii = send_map_starts[indx];
      send_i[iii] = off_proc_i[i]; 
      send_data[iii] = off_proc_data[i];
      send_map_starts[indx]++;
   }

   hypre_TFree(proc_id_mem);

   for (i=num_sends; i > 0; i--)
   {
      send_map_starts[i] = send_map_starts[i-1];
   }
   send_map_starts[0] = 0;

   num_requests = num_recvs+num_sends;

   requests = hypre_CTAlloc(MPI_Request, num_requests);
   status = hypre_CTAlloc(MPI_Status, num_requests);

   j=0; 
   for (i=0; i < num_recvs; i++)
   {
       vec_start = recv_vec_starts[i];
       vec_len = recv_vec_starts[i+1] - vec_start;
       ip = recv_procs[i];
       MPI_Irecv(&recv_i[vec_start], vec_len, MPI_HYPRE_BIG_INT, ip, 0, comm, 
			&requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
       vec_start = send_map_starts[i];
       vec_len = send_map_starts[i+1] - vec_start;
       ip = send_procs[i];
       MPI_Isend(&send_i[vec_start], vec_len, MPI_HYPRE_BIG_INT, ip, 0, comm, 
			&requests[j++]);
   }
  
   if (num_requests)
   {
      MPI_Waitall(num_requests, requests, status);
   }

   j=0;
   for (i=0; i < num_recvs; i++)
   {
       vec_start = recv_vec_starts[i];
       vec_len = recv_vec_starts[i+1] - vec_start;
       ip = recv_procs[i];
       MPI_Irecv(&recv_data[vec_start], vec_len, MPI_DOUBLE, ip, 0, comm, 
			&requests[j++]);
   }

   for (i=0; i < num_sends; i++)
   {
       vec_start = send_map_starts[i];
       vec_len = send_map_starts[i+1] - vec_start;
       ip = send_procs[i];
       MPI_Isend(&send_data[vec_start], vec_len, MPI_DOUBLE, ip, 0, comm, 
			&requests[j++]);
   }
  
   if (num_requests)
   {
      MPI_Waitall(num_requests, requests, status);
      hypre_TFree(requests);
      hypre_TFree(status);
   }

   hypre_TFree(send_i);
   hypre_TFree(send_data);
   hypre_TFree(send_procs);
   hypre_TFree(send_map_starts);
   hypre_TFree(recv_procs);

   for (i=0; i < recv_vec_starts[num_recvs]; i++)
   {
      row = recv_i[i];
      if (row < 0)
      {
         row = -row-1;
         j = (int)(row - first_index);
         data[j] += recv_data[i];
      }
      else
      {
         j = (int)(row - first_index);
         data[j] = recv_data[i];
      }
   }

   hypre_TFree(recv_vec_starts);
   hypre_TFree(recv_i);
   hypre_TFree(recv_data);

   return hypre_error_flag;
}

#else


/*   assumed partition version */


int
hypre_IJVectorAssembleOffProcValsPar( hypre_IJVector *vector, 
   				      int max_off_proc_elmts,
   				      int current_num_elmts,
   				      HYPRE_BigInt *off_proc_i,
   			     	      double *off_proc_data)
{

   int myid;
   HYPRE_BigInt global_num_rows;
   int i, j, in, k;
   int proc_id, last_proc, prev_id, tmp_id;
   int max_response_size;
   int ex_num_contacts = 0;
   HYPRE_BigInt range_start, range_end;
   int storage;
   int indx;
   int num_ranges, row_count;
   HYPRE_BigInt row;
   int num_recvs;
   int counter;
   HYPRE_BigInt upper_bound;
   int num_real_procs;
   int first_index;
   
   HYPRE_BigInt *big_ex_contact_buf = NULL;
   HYPRE_BigInt *row_list=NULL;
   int *a_proc_id=NULL, *orig_order=NULL;
   int *real_proc_id = NULL, *us_real_proc_id = NULL;
   int *ex_contact_procs = NULL, *ex_contact_vec_starts = NULL;
   int *recv_starts=NULL;
   int *response_buf_starts=NULL;
   HYPRE_BigInt *response_buf = NULL;
   int *num_rows_per_proc = NULL;
   int  tmp_int;
   int  obj_size_bytes, int_size, double_size, big_int_size;

   void *void_contact_buf = NULL;
   void *index_ptr;
   void *recv_data_ptr;

   double tmp_double;
   double *vector_data;
   double value;
   
   hypre_DataExchangeResponse      response_obj1, response_obj2;
   hypre_ProcListElements          send_proc_obj; 

   MPI_Comm comm = hypre_IJVectorComm(vector);
   hypre_ParVector *par_vector = hypre_IJVectorObject(vector);

   hypre_IJAssumedPart   *apart;


   MPI_Comm_rank(comm, &myid);
   
   global_num_rows = hypre_ParVectorGlobalSize(par_vector);
 
   /* verify that we have created the assumed partition */

   if  (hypre_ParVectorAssumedPartition(par_vector) == NULL)
   {
      hypre_ParVectorCreateAssumedPartition(par_vector);
   }
   apart = hypre_ParVectorAssumedPartition(par_vector);



   /* get the assumed processor id for each row */
   a_proc_id = hypre_CTAlloc(int, current_num_elmts);
   orig_order =  hypre_CTAlloc(int, current_num_elmts);
   real_proc_id = hypre_CTAlloc(int, current_num_elmts);
   row_list =   hypre_CTAlloc(HYPRE_BigInt, current_num_elmts);

   if (current_num_elmts > 0)
   {
      for (i=0; i < current_num_elmts; i++)
      {
         row = off_proc_i[i]; 
         if (row < 0) row = -row-1;
         row_list[i] = row;
         hypre_GetAssumedPartitionProcFromRow (comm, row, global_num_rows, &proc_id);
         a_proc_id[i] = proc_id;
         orig_order[i] = i;
      }

      /* now we need to find the actual order of each row  - sort on row -
         this will result in proc ids sorted also...*/
      
      hypre_BigQsortb2i(row_list, a_proc_id, orig_order, 0, current_num_elmts -1);

      /* calculate the number of contacts */
      ex_num_contacts = 1;
      last_proc = a_proc_id[0];
      for (i=1; i < current_num_elmts; i++)
      {
         if (a_proc_id[i] > last_proc)      
         {
            ex_num_contacts++;
            last_proc = a_proc_id[i];
         }
      }
      
   }
   
   /* now we will go through a create a contact list - need to contact
      assumed processors and find out who the actual row owner is - we
      will contact with a range (2 numbers) */




   ex_contact_procs = hypre_CTAlloc(int, ex_num_contacts);
   ex_contact_vec_starts =  hypre_CTAlloc(int, ex_num_contacts+1);
   big_ex_contact_buf =  hypre_CTAlloc(HYPRE_BigInt, ex_num_contacts*2);


   counter = 0;
   range_end = -1;
   for (i=0; i< current_num_elmts; i++) 
   {
      if (row_list[i] > range_end)
      {
         /* assumed proc */
         proc_id = a_proc_id[i];

         /* end of prev. range */
         if (counter > 0)  big_ex_contact_buf[counter*2 - 1] = row_list[i-1];
         
        /*start new range*/
    	 ex_contact_procs[counter] = proc_id;
         ex_contact_vec_starts[counter] = counter*2;
         big_ex_contact_buf[counter*2] =  row_list[i];
         counter++;
         
         hypre_GetAssumedPartitionRowRange(comm, proc_id, global_num_rows, 
                                           &range_start, &range_end); 


      }
   }


  /*finish the starts*/
   ex_contact_vec_starts[counter] =  counter*2;
   /*finish the last range*/
   if (counter > 0)  
      big_ex_contact_buf[counter*2 - 1] = row_list[current_num_elmts - 1];



  /*create response object - can use same fill response as used in the commpkg
     routine */
   response_obj1.fill_response = hypre_RangeFillResponseIJDetermineRecvProcs;
   response_obj1.data1 =  apart; /* this is necessary so we can fill responses*/ 
   response_obj1.data2 = NULL;
   
   max_response_size = 6;  /* 6 means we can fit 3 ranges*/
   
   hypre_DataExchangeList(ex_num_contacts, ex_contact_procs, 
                    big_ex_contact_buf, ex_contact_vec_starts, sizeof(HYPRE_BigInt), 
                     sizeof(HYPRE_BigInt), &response_obj1, max_response_size, 4, 
                     comm, (void**) &response_buf, &response_buf_starts);


   /* now response_buf contains
     a proc_id followed by an upper bound for the range.  */



   hypre_TFree(ex_contact_procs);
   hypre_TFree(big_ex_contact_buf);
   hypre_TFree(ex_contact_vec_starts);

   hypre_TFree(a_proc_id);



   /*how many ranges were returned?*/
   num_ranges = response_buf_starts[ex_num_contacts];   
   num_ranges = num_ranges/2;
   
   
   prev_id = -1;
   j = 0;
   counter = 0;
   num_real_procs = 0;


   /* loop through ranges - create a list of actual processor ids*/
   for (i=0; i<num_ranges; i++)
   {
      upper_bound = response_buf[i*2+1];
      counter = 0;
      tmp_id = (int) response_buf[i*2];
      
      /* loop through row_list entries - counting how many are in the range */
      while (row_list[j] <= upper_bound && j < current_num_elmts)     
      {
         real_proc_id[j] = tmp_id;
         j++;
         counter++;       
      }
      if (counter > 0 && tmp_id != prev_id)        
      {
         num_real_procs++;
      }
      prev_id = tmp_id;
   }


   /* now we have the list of real procesors ids (real_proc_id) - 
      and the number of distinct ones -
      so now we can set up data to be sent - we have int and double data.
      (row number and value) - we will send everything as a void since
       we may not know the rel sizes of ints and doubles*/  
 
   /*first find out how many elements to send per proc - so we can do
     storage */
 
   int_size = sizeof(int);
   double_size = sizeof(double);
   big_int_size = sizeof(HYPRE_BigInt);

   obj_size_bytes = hypre_max(big_int_size, double_size);
    
   ex_contact_procs = hypre_CTAlloc(int, num_real_procs);
   num_rows_per_proc = hypre_CTAlloc(int, num_real_procs);
   
   counter = 0;
   
   if (num_real_procs > 0 )
   {
      ex_contact_procs[0] = real_proc_id[0];
      num_rows_per_proc[0] = 1;
      
      for (i=1; i < current_num_elmts; i++) /* loop through real procs - these are sorted
                                               (row_list is sorted also)*/
      {
         if (real_proc_id[i] == ex_contact_procs[counter]) /* same processor */
         {
            num_rows_per_proc[counter] += 1; /*another row */
         }
         else /* new processor */
         {
            counter++;
            ex_contact_procs[counter] = real_proc_id[i];
            num_rows_per_proc[counter] = 1;
         }
      }
   }
     
   /* calculate total storage and make vec_starts arrays */
   storage = 0;
   ex_contact_vec_starts = hypre_CTAlloc(int, num_real_procs + 1);
   ex_contact_vec_starts[0] = -1;

   for (i=0; i < num_real_procs; i++)
   {
      storage += 1 + 2*  num_rows_per_proc[i];
      ex_contact_vec_starts[i+1] = -storage-1; /* need negative for next loop */
   }      

   void_contact_buf = hypre_MAlloc(storage*obj_size_bytes);
   index_ptr = void_contact_buf; /* step through with this index */

   /* set up data to be sent to send procs */
   /* for each proc, ex_contact_buf_d contains #rows, row #, data, etc. */
      
   /* un-sort real_proc_id  - we want to access data arrays in order */

   us_real_proc_id =  hypre_CTAlloc(int, current_num_elmts);
   for (i=0; i < current_num_elmts; i++)
   {
      us_real_proc_id[orig_order[i]] = real_proc_id[i];
   }
   hypre_TFree(real_proc_id);



   prev_id = -1;
   for (i=0; i < current_num_elmts; i++)
   {
      proc_id = us_real_proc_id[i];
      row = off_proc_i[i]; /*can't use row list[i] - you loose the negative signs that 
                               differentiate add/set values */
      /* find position of this processor */
      indx = hypre_BinarySearch(ex_contact_procs, proc_id, num_real_procs);
      in =  ex_contact_vec_starts[indx];

      index_ptr = (void *) ((char *) void_contact_buf + in*obj_size_bytes);

      if (in < 0) /* first time for this processor - add the number of rows to the buffer */
      {
         in = -in - 1;
         /* re-calc. index_ptr since in_i was negative */
         index_ptr = (void *) ((char *) void_contact_buf + in*obj_size_bytes);

         tmp_int =  num_rows_per_proc[indx];
         memcpy( index_ptr, &tmp_int, int_size);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);

         in++;
      }
      /* add row # */   
      memcpy( index_ptr, &row, big_int_size);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in++;


      /* add value */
      tmp_double = off_proc_data[i];
      memcpy( index_ptr, &tmp_double, double_size);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in++;

      
      /* increment the indexes to keep track of where we are - fix later */
      ex_contact_vec_starts[indx] = in;
      
   }
   
   /* some clean up */
 
   hypre_TFree(response_buf);
   hypre_TFree(response_buf_starts);

   hypre_TFree(us_real_proc_id);
   hypre_TFree(orig_order);
   hypre_TFree(row_list);
   hypre_TFree(num_rows_per_proc);
   

   for (i=num_real_procs; i > 0; i--)
   {
      ex_contact_vec_starts[i] =   ex_contact_vec_starts[i-1];
   }

   ex_contact_vec_starts[0] = 0;

  

  /* now send the data */

  /***********************************/
   /* now get the info in send_proc_obj_d */


   /* the response we expect is just a confirmation*/
   response_buf = NULL;
   response_buf_starts = NULL;

   /*build the response object*/

   /* use the send_proc_obj for the info kept from contacts */
   /*estimate inital storage allocation */

   send_proc_obj.length = 0;
   send_proc_obj.storage_length = num_real_procs + 5;
   send_proc_obj.id = NULL; /* don't care who sent it to us */
   send_proc_obj.vec_starts = hypre_CTAlloc(int, send_proc_obj.storage_length + 1); 
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = storage + 20;
   send_proc_obj.v_elements = hypre_MAlloc(obj_size_bytes*send_proc_obj.element_storage_length);

   response_obj2.fill_response = hypre_FillResponseIJOffProcVals;
   response_obj2.data1 = NULL;
   response_obj2.data2 = &send_proc_obj;

   max_response_size = 0;

   hypre_DataExchangeList(num_real_procs, ex_contact_procs, 
                          void_contact_buf, ex_contact_vec_starts, obj_size_bytes,
                          0, &response_obj2, max_response_size, 5, 
                          comm,  (void **) &response_buf, &response_buf_starts);




   /***********************************/


   hypre_TFree(response_buf);
   hypre_TFree(response_buf_starts);

   hypre_TFree(ex_contact_procs);
   hypre_TFree(void_contact_buf);
   hypre_TFree(ex_contact_vec_starts);


   /* Now we can unpack the send_proc_objects and either set or add to 
      the vector data */

   num_recvs = send_proc_obj.length; 

   /* alias */
   recv_data_ptr = send_proc_obj.v_elements;
   recv_starts = send_proc_obj.vec_starts;
   
   vector_data = hypre_VectorData(hypre_ParVectorLocalVector(par_vector));
   first_index = hypre_ParVectorFirstIndex(par_vector);
   

   for (i=0; i < num_recvs; i++)
   {
    
      indx = recv_starts[i];

      /* get the number of rows for  this recv */
      memcpy( &row_count, recv_data_ptr, int_size);
      recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
      indx++;

      for (j=0; j < row_count; j++) /* for each row: unpack info */
      {

         /* row # */
         memcpy( &row, recv_data_ptr, big_int_size);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         /* value */
         memcpy( &value, recv_data_ptr, double_size);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;


         if (row < 0) /* add */
         {
            row = -row-1;
            k = row - first_index;
            vector_data[k] += value;
         }
         else  /* set */
         {
            k = row - first_index;
            vector_data[k] = value;
         }
      }
   }
   

   hypre_TFree(send_proc_obj.v_elements);
   hypre_TFree(send_proc_obj.vec_starts);

 
   return hypre_error_flag;
}

#endif
