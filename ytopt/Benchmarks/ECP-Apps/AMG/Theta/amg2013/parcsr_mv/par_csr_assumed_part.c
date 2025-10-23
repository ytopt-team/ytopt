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



/*---------------------------------------------------- 
 * Functions for the IJ assumed partition
 * (Some of these were formerly in new_commpkg.c)
 *  AHB 4/06                                            
 *-----------------------------------------------------*/

#include "headers.h"

/* This is used only in the function below */
#define CONTACT(a,b)  (contact_list[(a)*3+(b)])

/*--------------------------------------------------------------------
 * hypre_LocateAssummedPartition
 * Reconcile assumed partition with actual partition.  Essentially
 * each processor ends of with a partition of its assumed partition.
 *--------------------------------------------------------------------*/


int 
hypre_LocateAssummedPartition(MPI_Comm comm, HYPRE_BigInt row_start, HYPRE_BigInt row_end, 
			HYPRE_BigInt global_num_rows, 
                        hypre_IJAssumedPart *part, int myid)
{  

   int       i;

   HYPRE_BigInt       *contact_list;
   int        contact_list_length, contact_list_storage;
   
   HYPRE_BigInt contact_row_start[2], contact_row_end[2], contact_ranges;
   int        owner_start, owner_end;
   HYPRE_BigInt tmp_row_start, tmp_row_end;

   HYPRE_BigInt locate_row_start[2];
   int        locate_ranges, complete;

   int        locate_row_count, rows_found;  
 
   HYPRE_BigInt tmp_range[2];
   HYPRE_BigInt *sortme;   
   int       *si;

   const int  flag1 = 17;  

   MPI_Request  *requests;
   MPI_Status   status0, *statuses;




   /*-----------------------------------------------------------
    *  Contact ranges - 
    *  which rows do I have that others are assumed responsible for?
    *  (at most two ranges - maybe none)
    *-----------------------------------------------------------*/
  

   contact_row_start[0]=0;
   contact_row_end[0]=0;
   contact_row_start[1]=0;
   contact_row_end[1]=0;
   contact_ranges = 0;
 
   if (row_start <= row_end ) { /*must own at least one row*/

      if ( part->row_end < row_start  || row_end < part->row_start  )   
      {  /*no overlap - so all of my rows and only one range*/
         contact_row_start[0] = row_start;
         contact_row_end[0] = row_end;
         contact_ranges++;
      }
      else /* the two regions overlap - so one or two ranges */
      {  
         /* check for contact rows on the low end of the local range */
	 if (row_start < part->row_start) 
	 {
            contact_row_start[0] = row_start;
	    contact_row_end[0] = part->row_start - 1;
	    contact_ranges++;
	 } 
	 if (part->row_end < row_end) /* check the high end */
         {
            if (contact_ranges) /* already found one range */ 
            {
	       contact_row_start[1] = part->row_end +1;
	       contact_row_end[1] = row_end;
	    }
	    else
	    {
	       contact_row_start[0] =  part->row_end +1;
	       contact_row_end[0] = row_end;
	    } 
	    contact_ranges++;
	 }
      }
   }

   /*-----------------------------------------------------------
    *  Contact: find out who is assumed responsible for these 
    *       ranges of contact rows and contact them 
    *
    *-----------------------------------------------------------*/

     
   contact_list_length = 0;
   contact_list_storage = 5; 
   contact_list = hypre_TAlloc(HYPRE_BigInt, contact_list_storage*3); /*each contact needs 3 ints */

   for (i=0; i<contact_ranges; i++)
   {
  
      /*get start and end row owners */
      hypre_GetAssumedPartitionProcFromRow(comm, contact_row_start[i], global_num_rows, 
                                            &owner_start);
      hypre_GetAssumedPartitionProcFromRow(comm, contact_row_end[i], global_num_rows, 
                                             &owner_end);

      if (owner_start == owner_end) /* same processor owns the whole range */
      {

         if (contact_list_length == contact_list_storage)
         {
            /*allocate more space*/
            contact_list_storage += 5;
            contact_list = hypre_TReAlloc(contact_list, HYPRE_BigInt, (contact_list_storage*3));
         }
         CONTACT(contact_list_length, 0) = (HYPRE_BigInt) owner_start;   /*proc #*/
         CONTACT(contact_list_length, 1) = contact_row_start[i];  /* start row */
         CONTACT(contact_list_length, 2) = contact_row_end[i];  /*end row */
         contact_list_length++;
      }
      else
      { 
        complete = 0;
        while (!complete) 
        {
            hypre_GetAssumedPartitionRowRange(comm, owner_start, global_num_rows, 
                                         &tmp_row_start, &tmp_row_end); 
           
            if (tmp_row_end >= contact_row_end[i])
	    {
	       tmp_row_end =  contact_row_end[i];
               complete = 1; 
	    }     
            if (tmp_row_start <  contact_row_start[i])
	    {
	      tmp_row_start =  contact_row_start[i];
            }


            if (contact_list_length == contact_list_storage)
            {
               /*allocate more space*/
               contact_list_storage += 5;
               contact_list = hypre_TReAlloc(contact_list, HYPRE_BigInt, (contact_list_storage*3));
            }


            CONTACT(contact_list_length, 0) = (HYPRE_BigInt) owner_start;   /*proc #*/
            CONTACT(contact_list_length, 1) = tmp_row_start;  /* start row */
            CONTACT(contact_list_length, 2) = tmp_row_end;  /*end row */
            contact_list_length++;
	    owner_start++; /*processors are seqential */
        }
      }
   }

   requests = hypre_CTAlloc(MPI_Request, contact_list_length);
   statuses = hypre_CTAlloc(MPI_Status, contact_list_length);

   /*send out messages */
   for (i=0; i< contact_list_length; i++) 
   {
      MPI_Isend(&CONTACT(i,1) ,2, MPI_HYPRE_BIG_INT, CONTACT(i,0), flag1 , 
                 comm, &requests[i]);
   }

   /*-----------------------------------------------------------
    *  Locate ranges - 
    *  which rows in my assumed range do I not own
    *  (at most two ranges - maybe none)
    *  locate_row_count = total number of rows I must locate
    *-----------------------------------------------------------*/


   locate_row_count = 0;
 
   locate_row_start[0]=0;
   locate_row_start[1]=0;

   locate_ranges = 0;

   if (part->row_end < row_start  || row_end < part->row_start  ) 
   /*no overlap - so all of my assumed rows */ 
   {
      locate_row_start[0] = part->row_start;
      locate_ranges++;
      locate_row_count += part->row_end - part->row_start + 1; 
   }
   else /* the two regions overlap */
   {
      if (part->row_start < row_start) 
      {/* check for locate rows on the low end of the local range */
         locate_row_start[0] = part->row_start;
         locate_ranges++;
         locate_row_count += (row_start-1) - part->row_start + 1;
      } 
      if (row_end < part->row_end) /* check the high end */
      {
         if (locate_ranges) /* already have one range */ 
         {
	    locate_row_start[1] = row_end +1;
	 }
         else
         {
	    locate_row_start[0] = row_end +1;
         } 
         locate_ranges++;
         locate_row_count += part->row_end - (row_end + 1) + 1;
      }
   }


    /*-----------------------------------------------------------
     * Receive messages from other procs telling us where
     * all our  locate rows actually reside 
     *-----------------------------------------------------------*/


    /* we will keep a partition of our assumed partition - list ourselves 
       first.  We will sort later with an additional index.
       In practice, this should only contain a few processors */
 
   /*which part do I own?*/
   tmp_row_start = hypre_max(part->row_start, row_start);
   tmp_row_end = hypre_min(row_end, part->row_end);

   if (tmp_row_start <= tmp_row_end)
   {
      part->proc_list[0] =   myid;
      part->row_start_list[0] = tmp_row_start;
      part->row_end_list[0] = tmp_row_end;
      part->length++;
   }
  
   /* now look for messages that tell us which processor has our locate rows */
   /* these will be blocking receives as we know how many to expect and they should
       be waiting (and we don't want to continue on without them) */

   rows_found = 0;

   while (rows_found != locate_row_count) {
  
      MPI_Recv( tmp_range, 2 , MPI_HYPRE_BIG_INT, MPI_ANY_SOURCE, 
                flag1 , comm, &status0);
     
      if (part->length==part->storage_length)
      {
	part->storage_length+=10;
        part->proc_list = hypre_TReAlloc(part->proc_list, int, part->storage_length);
        part->row_start_list =   hypre_TReAlloc(part->row_start_list, HYPRE_BigInt, part->storage_length);
        part->row_end_list =   hypre_TReAlloc(part->row_end_list, HYPRE_BigInt, part->storage_length);

      }
      part->row_start_list[part->length] = tmp_range[0];
      part->row_end_list[part->length] = tmp_range[1]; 

      part->proc_list[part->length] = status0.MPI_SOURCE;
      rows_found += (int)(tmp_range[1]- tmp_range[0]) + 1;
      
      part->length++;
   } 

   /*In case the partition of the assumed partition is longish, 
     we would like to know the sorted order */
   si= hypre_CTAlloc(int, part->length); 
   sortme = hypre_CTAlloc(HYPRE_BigInt, part->length); 

   for (i=0; i<part->length; i++) 
   {
       si[i] = i;
       sortme[i] = part->row_start_list[i];
   }
   hypre_BigQsortbi( sortme, si, 0, (part->length)-1);
   part->sort_index = si;

  /*free the requests */
   MPI_Waitall(contact_list_length, requests, 
                    statuses);

   
   hypre_TFree(statuses);
   hypre_TFree(requests);
   

   hypre_TFree(sortme);
   hypre_TFree(contact_list);


   return hypre_error_flag;
   

}

/*--------------------------------------------------------------------
 * hypre_ParCSRMatrixCreateAssumedPartition -
 * Each proc gets it own range. Then 
 * each needs to reconcile its actual range with its assumed
 * range - the result is essentila a partition of its assumed range -
 * this is the assumed partition.   
 *--------------------------------------------------------------------*/


int
hypre_ParCSRMatrixCreateAssumedPartition( hypre_ParCSRMatrix *matrix) 
{


   HYPRE_BigInt global_num_cols;
   int myid;
   HYPRE_BigInt  row_start=0, row_end=0, col_start = 0, col_end = 0;

   MPI_Comm   comm;
   
   hypre_IJAssumedPart *apart;

   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(matrix); 
   comm = hypre_ParCSRMatrixComm(matrix);
   
   /* find out my actualy range of rows and columns */
   hypre_ParCSRMatrixGetLocalRange( matrix,
                                    &row_start, &row_end ,
                                    &col_start, &col_end );
   MPI_Comm_rank(comm, &myid );

   /* allocate space */
   apart = hypre_CTAlloc(hypre_IJAssumedPart, 1);

  /* get my assumed partitioning  - we want partitioning of the vector that the
      matrix multiplies - so we use the col start and end */
   hypre_GetAssumedPartitionRowRange( comm, myid, global_num_cols, &(apart->row_start), 
                                             &(apart->row_end));

  /*allocate some space for the partition of the assumed partition */
    apart->length = 0;
    /*room for 10 owners of the assumed partition*/ 
    apart->storage_length = 10; /*need to be >=1 */ 
    apart->proc_list = hypre_TAlloc(int, apart->storage_length);
    apart->row_start_list =   hypre_TAlloc(HYPRE_BigInt, apart->storage_length);
    apart->row_end_list =   hypre_TAlloc(HYPRE_BigInt, apart->storage_length);


    /* now we want to reconcile our actual partition with the assumed partition */
    hypre_LocateAssummedPartition(comm, col_start, col_end, global_num_cols, apart, myid);

    /* this partition will be saved in the matrix data structure until the matrix is destroyed */
    hypre_ParCSRMatrixAssumedPartition(matrix) = apart;
   
    return hypre_error_flag;


}

/*--------------------------------------------------------------------
 * hypre_ParCSRMatrixDestroyAssumedPartition
 *--------------------------------------------------------------------*/
int 
hypre_ParCSRMatrixDestroyAssumedPartition(hypre_ParCSRMatrix *matrix )
{

   hypre_IJAssumedPart *apart;
   
   apart = hypre_ParCSRMatrixAssumedPartition(matrix);
   

   if(apart->storage_length > 0) 
   {      
      hypre_TFree(apart->proc_list);
      hypre_TFree(apart->row_start_list);
      hypre_TFree(apart->row_end_list);
      hypre_TFree(apart->sort_index);
   }

   hypre_TFree(apart);
   
   return hypre_error_flag;

}

/*--------------------------------------------------------------------
 * hypre_GetAssumedPartitionProcFromRow
 * Assumed partition for IJ case. Given a particular row j, return
 * the processor that is assumed to own that row.
 *--------------------------------------------------------------------*/


int
hypre_GetAssumedPartitionProcFromRow( MPI_Comm comm, HYPRE_BigInt row, HYPRE_BigInt global_num_rows, int *proc_id)
{

   int     num_procs;
   HYPRE_BigInt size, extra, temp;
   HYPRE_BigInt switch_row;
   
  
   MPI_Comm_size(comm, &num_procs );
 
   /* j = floor[(row*p/N]  - this overflows*/
   /* *proc_id = (row*num_procs)/global_num_rows;*/
       
   /* this looks a bit odd, but we have to be very careful that
      this function and the next are inverses - and rounding 
      errors make this difficult!!!!! */

   size = global_num_rows /(HYPRE_BigInt)num_procs;
   extra = global_num_rows - size*(HYPRE_BigInt)num_procs;
   switch_row = (size + 1)*extra;
   
   if (row >= switch_row)
   {
      temp = extra + (row - switch_row)/size;
      *proc_id = (int)temp;
   }
   else
   {
      temp = row/(size+1);      
      *proc_id = (int)temp;
   }


   return hypre_error_flag;


}
/*--------------------------------------------------------------------
 * hypre_GetAssumedPartitionRowRange
 * Assumed partition for IJ case. Given a particular processor id, return
 * the assumed range of rows ([row_start, row_end]) for that processor.
 *--------------------------------------------------------------------*/


int
hypre_GetAssumedPartitionRowRange( MPI_Comm comm, int proc_id, HYPRE_BigInt global_num_rows, 
                             HYPRE_BigInt *row_start, HYPRE_BigInt* row_end) 
{

   int    num_procs;
   int    size, extra;
   

   MPI_Comm_size(comm, &num_procs );


  /* this may look non-intuitive, but we have to be very careful that
      this function and the next are inverses - and avoiding overflow and
      rounding errors makes this difficult! */

   size = (int)(global_num_rows /num_procs);
   extra = (int)(global_num_rows - (HYPRE_BigInt)size*(HYPRE_BigInt)num_procs);

   *row_start = (HYPRE_BigInt)size*(HYPRE_BigInt)proc_id;
   *row_start += (HYPRE_BigInt)hypre_min(proc_id, extra);
   

   *row_end =  (HYPRE_BigInt)size*(HYPRE_BigInt)(proc_id+1);
   *row_end += (HYPRE_BigInt)hypre_min(proc_id+1, extra);
   *row_end = *row_end - 1;


   return hypre_error_flag;

}


/*--------------------------------------------------------------------
 * hypre_ParVectorCreateAssumedPartition -

 * Essentially the same as for a matrix!

 * Each proc gets it own range. Then 
 * each needs to reconcile its actual range with its assumed
 * range - the result is essentila a partition of its assumed range -
 * this is the assumed partition.   
 *--------------------------------------------------------------------*/


int
hypre_ParVectorCreateAssumedPartition( hypre_ParVector *vector) 
{


   HYPRE_BigInt global_num;
   int myid;
   HYPRE_BigInt  start=0, end=0;

   MPI_Comm   comm;
   
   hypre_IJAssumedPart *apart;

   global_num = hypre_ParVectorGlobalSize(vector); 
   comm = hypre_ParVectorComm(vector);
   
   /* find out my actualy range of rows */
   start =  hypre_ParVectorFirstIndex(vector);
   end = hypre_ParVectorLastIndex(vector);
   
   MPI_Comm_rank(comm, &myid );

   /* allocate space */
   apart = hypre_CTAlloc(hypre_IJAssumedPart, 1);

  /* get my assumed partitioning  - we want partitioning of the vector that the
      matrix multiplies - so we use the col start and end */
   hypre_GetAssumedPartitionRowRange( comm, myid, global_num, &(apart->row_start), 
                                             &(apart->row_end));

  /*allocate some space for the partition of the assumed partition */
    apart->length = 0;
    /*room for 10 owners of the assumed partition*/ 
    apart->storage_length = 10; /*need to be >=1 */ 
    apart->proc_list = hypre_TAlloc(int, apart->storage_length);
    apart->row_start_list =   hypre_TAlloc(HYPRE_BigInt, apart->storage_length);
    apart->row_end_list =   hypre_TAlloc(HYPRE_BigInt, apart->storage_length);


    /* now we want to reconcile our actual partition with the assumed partition */
    hypre_LocateAssummedPartition(comm, start, end, global_num, apart, myid);

    /* this partition will be saved in the vector data structure until the vector is destroyed */
    hypre_ParVectorAssumedPartition(vector) = apart;
   
    return hypre_error_flag;


}

/*--------------------------------------------------------------------
 * hypre_ParVectorDestroyAssumedPartition
 *--------------------------------------------------------------------*/
int 
hypre_ParVectorDestroyAssumedPartition(hypre_ParVector *vector )
{

   hypre_IJAssumedPart *apart;
   
   apart = hypre_ParVectorAssumedPartition(vector);
   

   if(apart->storage_length > 0) 
   {      
      hypre_TFree(apart->proc_list);
      hypre_TFree(apart->row_start_list);
      hypre_TFree(apart->row_end_list);
      hypre_TFree(apart->sort_index);
   }

   hypre_TFree(apart);
   
   return hypre_error_flag;

}
