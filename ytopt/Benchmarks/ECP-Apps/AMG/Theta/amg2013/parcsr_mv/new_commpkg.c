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
 * Communication package that uses an assumed partition
 *  AHB 6/04                                            
 *-----------------------------------------------------*/

#include "headers.h"

/* some debugging tools*/
#define mydebug 0

/*==========================================================================*/



int PrintCommpkg(hypre_ParCSRMatrix *A, const char *file_name)
{
   
   int num_sends, num_recvs;
   
   int *recv_vec_starts, *recv_procs;
   int *send_map_starts, *send_map_elements, *send_procs;

   int i;
   int my_id;
   
   MPI_Comm comm;
   
   hypre_ParCSRCommPkg *comm_pkg;
   

   char   new_file[80];
   FILE *fp;

   comm_pkg =  hypre_ParCSRMatrixCommPkg(A);
   

   comm =  hypre_ParCSRCommPkgComm(comm_pkg);
   
   num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
   send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   send_map_elements = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   
   MPI_Comm_rank(comm, &my_id);

   sprintf(new_file,"%s.%d",file_name,my_id);

   fp = fopen(new_file, "w");
   fprintf(fp, "num_recvs = %d\n", num_recvs);
   for (i=0; i < num_recvs; i++)
   {
      fprintf(fp, "recv_proc [start, end] = %d [%d, %d] \n", recv_procs[i], recv_vec_starts[i], recv_vec_starts[i+1]-1);
   }

   fprintf(fp, "num_sends = %d\n", num_sends);
   for (i=0; i < num_sends; i++)
   {
      fprintf(fp, "send_proc [start, end] = %d [%d, %d] \n", send_procs[i], send_map_starts[i], send_map_starts[i+1]-1);
   }

   for (i = 0; i< send_map_starts[num_sends]; i++)
   {
      fprintf(fp, "send_map_elements (%d) = %d\n", i, send_map_elements[i]);
   }

   fclose(fp);

   return hypre_error_flag;



}


/*------------------------------------------------------------------
 * hypre_NewCommPkgCreate_core
 *
 * This does the work for  hypre_NewCommPkgCreate - we have to split it 
 * off so that it can also be used for block matrices.
 *--------------------------------------------------------------------------*/

int 
hypre_NewCommPkgCreate_core(
/* input args: */
   MPI_Comm comm, HYPRE_BigInt *col_map_off_d, HYPRE_BigInt first_col_diag,
   HYPRE_BigInt col_start, HYPRE_BigInt col_end, 
   int num_cols_off_d, HYPRE_BigInt global_num_cols,
/* pointers to output args: */
   int *p_num_recvs, int **p_recv_procs, int **p_recv_vec_starts,
   int *p_num_sends, int **p_send_procs, int ** p_send_map_starts,
   int **p_send_map_elements, hypre_IJAssumedPart *apart)

{
   int        num_procs, myid;
   int        j, i;
   HYPRE_BigInt  range_start, range_end, big_size; 

   int        size;
   int        count;  

   int        num_recvs, *recv_procs = NULL, *recv_vec_starts=NULL;
   int        tmp_id, prev_id;

   int        num_sends;

   int        ex_num_contacts, *ex_contact_procs=NULL, *ex_contact_vec_starts=NULL;
   HYPRE_BigInt *ex_contact_buf=NULL;
   int        *send_map_elmts=NULL;
    
   int        num_ranges;
   HYPRE_BigInt upper_bound;
   

   HYPRE_BigInt *response_buf = NULL;
   int        *response_buf_starts=NULL;

   int        max_response_size;
   
   hypre_DataExchangeResponse        response_obj1, response_obj2;
   hypre_ProcListElements            send_proc_obj; 

#if mydebug
   int tmp_int, index;
#endif

   MPI_Comm_size(comm, &num_procs );
   MPI_Comm_rank(comm, &myid );


#if mydebug

    printf("myid = %i, my assumed local range: [%i, %i]\n", myid, 
                         apart->row_start, apart->row_end);

      for (i=0; i<apart.length; i++)
      {
        printf("myid = %d, proc %d owns assumed partition range = [%d, %d]\n", 
                myid, apart->proc_list[i], apart->row_start_list[i], 
	        apart->row_end_list[i]);
      }

      printf("myid = %d, length of apart = %d\n", myid, apart->length);

#endif


   /*-----------------------------------------------------------
    *  Everyone knows where their assumed range is located
    * (because of the assumed partition object (apart).
    *  For the comm. package, each proc must know it's receive
    *  procs (who it will receive data from and how much data) 
    *  and its send procs 
    *  (who it will send data to) and the indices of the elements
    *  to be sent.  This is based on the non-zero
    *  entries in its rows. Each proc should know this from the user. 
    *-----------------------------------------------------------*/
   

   /*------------------------------------------------------------
    *  First, get the receive processors
    *  each par_csr matrix will have a certain number of columns
    *  (num_cols_off_d) given in col_map_offd[] for which it needs
    *  data from another processor. 
    *
    *------------------------------------------------------------*/

    /*calculate the assumed receive processors*/

   /* need to populate num_recvs, *recv_procs, and *recv_vec_starts 
      (correlates to starts in col_map_off_d for recv_procs) for 
      the comm. package*/


   /*create contact information*/

   ex_num_contacts = 0;

   /*estimate the storage needed*/
   if (num_cols_off_d > 0 && (apart->row_end - apart->row_start) > 0  )
   {
      big_size = col_map_off_d[num_cols_off_d-1] - col_map_off_d[0];
   
      size = (int)(big_size/(apart->row_end - apart->row_start)) + 2;
   }
   else
   {
      size = 0;
   }
   

   /*we will contact each with a range of cols that we need*/
   /* it is ok to contact yourself - because then there doesn't
      need to be separate code */

   ex_contact_procs = hypre_CTAlloc(int, size);
   ex_contact_vec_starts =  hypre_CTAlloc(int, size+1);
   ex_contact_buf =  hypre_CTAlloc(HYPRE_BigInt, size*2);

   range_end = -1;
   for (i=0; i< num_cols_off_d; i++) 
   { 
      if (col_map_off_d[i] > range_end)
      {


         hypre_GetAssumedPartitionProcFromRow(comm, col_map_off_d[i], 
                                              global_num_cols, &tmp_id);

         if (ex_num_contacts == size) /*need more space? */ 
         {
           size += 20;
           ex_contact_procs = hypre_TReAlloc(ex_contact_procs, int, size);
           ex_contact_vec_starts = hypre_TReAlloc(ex_contact_vec_starts, int, size+1);
           ex_contact_buf = hypre_TReAlloc(ex_contact_buf, HYPRE_BigInt, size*2);
         }

         /* end of prev. range */
         if (ex_num_contacts > 0)  ex_contact_buf[ex_num_contacts*2 - 1] = col_map_off_d[i-1];
         
        /*start new range*/
    	 ex_contact_procs[ex_num_contacts] = tmp_id;
         ex_contact_vec_starts[ex_num_contacts] = ex_num_contacts*2;
         ex_contact_buf[ex_num_contacts*2] =  col_map_off_d[i];
         
         
         ex_num_contacts++;

         hypre_GetAssumedPartitionRowRange(comm, tmp_id, global_num_cols, 
                                           &range_start, &range_end); 

      }
   }

   /*finish the starts*/
   ex_contact_vec_starts[ex_num_contacts] =  ex_num_contacts*2;
   /*finish the last range*/
   if (ex_num_contacts > 0)  ex_contact_buf[ex_num_contacts*2 - 1] = col_map_off_d[num_cols_off_d-1];


   /*don't allocate space for responses */
    

   /*create response object*/
   response_obj1.fill_response = hypre_RangeFillResponseIJDetermineRecvProcs;
   response_obj1.data1 =  apart; /* this is necessary so we can fill responses*/ 
   response_obj1.data2 = NULL;
   
   max_response_size = 6;  /* 6 means we can fit 3 ranges*/
   
   
   hypre_DataExchangeList(ex_num_contacts, ex_contact_procs, 
                    ex_contact_buf, ex_contact_vec_starts, sizeof(HYPRE_BigInt), 
                     sizeof(HYPRE_BigInt), &response_obj1, max_response_size, 1, 
                     comm, (void**) &response_buf, &response_buf_starts);



   /*now create recv_procs[] and recv_vec_starts[] and num_recvs 
     from the complete data in response_buf - this array contains
     a proc_id followed by an upper bound for the range.  */


   /*initialize */ 
   num_recvs = 0;
   size  = ex_num_contacts+20; /* num of recv procs should be roughly similar size 
                                 to number of contacts  - add a buffer of 20*/
 
   
   recv_procs = hypre_CTAlloc(int, size);
   recv_vec_starts =  hypre_CTAlloc(int, size+1);
   recv_vec_starts[0] = 0;
   
   /*how many ranges were returned?*/
   num_ranges = response_buf_starts[ex_num_contacts];   
   num_ranges = num_ranges/2;
   
   prev_id = -1;
   j = 0;
   count = 0;
   
   /* loop through ranges */
   for (i=0; i<num_ranges; i++)
   {
      upper_bound = response_buf[i*2+1];
      count = 0;
      /* loop through off_d entries - counting how many are in the range */
      while (j < num_cols_off_d && col_map_off_d[j] <= upper_bound)     
      {
         j++;
         count++;       
      }
      if (count > 0)        
      {
         /*add the range if the proc id != myid*/    
         tmp_id = response_buf[i*2];
         if (tmp_id != myid)
         {
            if (tmp_id != prev_id) /*increment the number of recvs */
            {
               /*check size of recv buffers*/
               if (num_recvs == size) 
               {
                  size+=20;
                  recv_procs = hypre_TReAlloc(recv_procs,int, size);
                  recv_vec_starts =  hypre_TReAlloc(recv_vec_starts,int, size+1);
               }
            
               recv_vec_starts[num_recvs+1] = j; /*the new start is at this element*/
               recv_procs[num_recvs] =  tmp_id; /*add the new processor*/
               num_recvs++;

            }
            else
            {
               /*same processor - just change the vec starts*/
               recv_vec_starts[num_recvs] = j; /*the new start is at this element*/
            }
         }
         prev_id = tmp_id;
         
      }
      
   }
 



#if mydebug
      for (i=0; i < num_recvs; i++) 
      {
          printf("myid = %d, recv proc = %d, vec_starts = [%d : %d]\n", 
                  myid, recv_procs[i], recv_vec_starts[i],recv_vec_starts[i+1]-1);
      }
#endif
 

   /*------------------------------------------------------------
    *  determine the send processors
    *  each processor contacts its recv procs to let them
    *  know they are a send processor
    *
    *-------------------------------------------------------------*/

   /* the contact information is the recv_processor infomation - so
      nothing more to do to generate contact info*/

   /* the response we expect is just a confirmation*/
   hypre_TFree(response_buf);
   hypre_TFree(response_buf_starts);
   response_buf = NULL;
   response_buf_starts = NULL;

   /*build the response object*/
   /*estimate for inital storage allocation that we send to as many procs 
     as we recv from + pad by 5*/
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = num_recvs + 5;
   send_proc_obj.id = hypre_CTAlloc(int, send_proc_obj.storage_length);
   send_proc_obj.vec_starts = hypre_CTAlloc(int, send_proc_obj.storage_length + 1); 
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = num_cols_off_d;
   send_proc_obj.elements = hypre_CTAlloc(HYPRE_BigInt, send_proc_obj.element_storage_length);

   response_obj2.fill_response = hypre_FillResponseIJDetermineSendProcs;
   response_obj2.data1 = NULL;
   response_obj2.data2 = &send_proc_obj; /*this is where we keep info from contacts*/
  
   max_response_size = 0;
      


   hypre_DataExchangeList(num_recvs, recv_procs, 
                     col_map_off_d, recv_vec_starts, sizeof(HYPRE_BigInt),
                    sizeof(HYPRE_BigInt), &response_obj2, max_response_size, 2, 
                    comm,  (void **) &response_buf, &response_buf_starts);



   num_sends = send_proc_obj.length; 

   /*send procs are in send_proc_object.id */
   /*send proc starts are in send_proc_obj.vec_starts */

#if mydebug
   printf("myid = %d, num_sends = %d\n", myid, num_sends);   
   for (i=0; i < num_sends; i++) 
   {
      tmp_int = send_proc_obj.vec_starts[i+1] - send_proc_obj.vec_starts[i];
      index = send_proc_obj.vec_starts[i];
      for (j=0; j< tmp_int; j++) 
      {
         printf("myid = %d, send proc = %d, send element = %d\n",myid,  
                send_proc_obj.id[i],send_proc_obj.elements[index+j]); 
      }   
   }
#endif

   /*-----------------------------------------------------------
    *  We need to sort the send procs and send elements (to produce
    *  the same result as with the standard comm package)
    *   11/07/05
    *-----------------------------------------------------------*/
   
   {
      
      int *orig_order;
      int *orig_send_map_starts;
      HYPRE_BigInt *orig_send_elements;
      int  ct, sz, pos;
      
      orig_order = hypre_CTAlloc(int, num_sends);
      orig_send_map_starts = hypre_CTAlloc(int, num_sends+1);
      orig_send_elements = hypre_CTAlloc(HYPRE_BigInt, send_proc_obj.vec_starts[num_sends]);
      
      orig_send_map_starts[0] = 0;
      /* copy send map starts and elements */ 
      for (i=0; i< num_sends; i++)
      {
         orig_order[i] = i;
         orig_send_map_starts[i+1] = send_proc_obj.vec_starts[i+1];
      }
      for (i=0; i< send_proc_obj.vec_starts[num_sends]; i++)
      {
         orig_send_elements[i] = send_proc_obj.elements[i];
      }
      /* sort processor ids - keep track of original order */
      hypre_qsort2i( send_proc_obj.id, orig_order, 0, num_sends-1 );
      
      /* now rearrange vec starts and send elements to correspond to proc ids */ 
      ct = 0;
      for (i=0; i< num_sends; i++)
      {
         pos = orig_order[i];
         sz = orig_send_map_starts[pos + 1] - orig_send_map_starts[pos];
         send_proc_obj.vec_starts[i+1] =  ct + sz;
         for (j = 0; j< sz; j++)
         {
            send_proc_obj.elements[ct +j] = orig_send_elements[orig_send_map_starts[pos]+j];
         }
         ct += sz;
      }
      /* clean up */
      hypre_TFree(orig_order);
      hypre_TFree(orig_send_elements);
      hypre_TFree(orig_send_map_starts);
   }
      

   /*-----------------------------------------------------------
    *  Return output info for setting up the comm package
    *-----------------------------------------------------------*/
   

   *p_num_recvs = num_recvs;
   *p_recv_procs = recv_procs;
   *p_recv_vec_starts = recv_vec_starts;
   *p_num_sends = num_sends;
   *p_send_procs = send_proc_obj.id;
   *p_send_map_starts = send_proc_obj.vec_starts;


   /*send map elements have global index - need local instead*/

   if (num_sends)
   {
      send_map_elmts = hypre_CTAlloc(int,send_proc_obj.vec_starts[num_sends]);
      for (i=0; i<send_proc_obj.vec_starts[num_sends]; i++)
      {   
         send_map_elmts[i] = (int)(send_proc_obj.elements[i]-first_col_diag);
         /*send_proc_obj.elements[i] -= first_col_diag;*/
      }
   }

   *p_send_map_elements = send_map_elmts;

   /*-----------------------------------------------------------
    *  Clean up
    *-----------------------------------------------------------*/

  
   if(ex_contact_procs)      hypre_TFree(ex_contact_procs);
   if(ex_contact_vec_starts) hypre_TFree(ex_contact_vec_starts);
   hypre_TFree(ex_contact_buf);
   

   if(response_buf)        hypre_TFree(response_buf);
   if(response_buf_starts) hypre_TFree(response_buf_starts);

   
   /* don't free send_proc_obj.id,send_proc_obj.vec_starts,send_proc_obj.elements;
      recv_procs, recv_vec_starts.  These are aliased to the comm package and
      will be destroyed there */

   return hypre_error_flag;

}

/*------------------------------------------------------------------
 * hypre_NewCommPkgCreate
 * this is an alternate way of constructing the comm package                                 
 * (compare to hypre_MatvecCommPkgCreate() in par_csr_communication.c
 * that should be more scalable 
 *-------------------------------------------------------------------*/

int 
hypre_NewCommPkgCreate( hypre_ParCSRMatrix *parcsr_A)
{

   HYPRE_BigInt row_start=0, row_end=0, col_start = 0, col_end = 0;
   int        num_recvs, *recv_procs, *recv_vec_starts;

   int        num_sends, *send_procs, *send_map_starts;
   int        *send_map_elements;

   int        num_cols_off_d; 
   HYPRE_BigInt *col_map_off_d; 

   HYPRE_BigInt  first_col_diag;
   HYPRE_BigInt  global_num_cols;


   MPI_Comm   comm;

   hypre_ParCSRCommPkg	 *comm_pkg;
   hypre_IJAssumedPart   *apart;
   
   /*-----------------------------------------------------------
    * get parcsr_A information 
    *----------------------------------------------------------*/

   hypre_ParCSRMatrixGetLocalRange( parcsr_A,
                                    &row_start, &row_end ,
                                    &col_start, &col_end );
   
   col_map_off_d =  hypre_ParCSRMatrixColMapOffd(parcsr_A);
   num_cols_off_d = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(parcsr_A));
   
   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(parcsr_A); 

   comm = hypre_ParCSRMatrixComm(parcsr_A);

   first_col_diag = hypre_ParCSRMatrixFirstColDiag(parcsr_A);


   /* Create the assumed partition */
   if  (hypre_ParCSRMatrixAssumedPartition(parcsr_A) == NULL)
   {
      hypre_ParCSRMatrixCreateAssumedPartition(parcsr_A);
   }

   apart = hypre_ParCSRMatrixAssumedPartition(parcsr_A);
   
   /*-----------------------------------------------------------
    * get commpkg info information 
    *----------------------------------------------------------*/

   hypre_NewCommPkgCreate_core( comm, col_map_off_d, first_col_diag, 
                                col_start, col_end, 
                                num_cols_off_d, global_num_cols,
                                &num_recvs, &recv_procs, &recv_vec_starts,
                                &num_sends, &send_procs, &send_map_starts, 
                                &send_map_elements, apart);
   

   if (!num_recvs)
   {
      hypre_TFree(recv_procs);
      recv_procs = NULL;
   }
   if (!num_sends)
   {
      hypre_TFree(send_procs);
      hypre_TFree(send_map_elements);
      send_procs = NULL;
      send_map_elements = NULL;
   }
   

  /*-----------------------------------------------------------
   * setup commpkg
   *----------------------------------------------------------*/

   comm_pkg = hypre_CTAlloc(hypre_ParCSRCommPkg, 1);

   hypre_ParCSRCommPkgComm(comm_pkg) = comm;
   hypre_ParCSRCommPkgNumRecvs(comm_pkg) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs(comm_pkg) = recv_procs;
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = recv_vec_starts;
   hypre_ParCSRCommPkgNumSends(comm_pkg) = num_sends;
   hypre_ParCSRCommPkgSendProcs(comm_pkg) = send_procs;
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = send_map_starts;
   hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = send_map_elements;
   
   hypre_ParCSRMatrixCommPkg(parcsr_A) = comm_pkg;

   return hypre_error_flag;
      
   
}

/*------------------------------------------------------------------
 *  hypre_NewCommPkgDestroy
 *  Destroy the comm package
 *------------------------------------------------------------------*/


int
hypre_NewCommPkgDestroy(hypre_ParCSRMatrix *parcsr_A)
{


   hypre_ParCSRCommPkg	 *comm_pkg = hypre_ParCSRMatrixCommPkg(parcsr_A);


   /*even if num_sends and num_recvs  = 0, storage may have been allocated */

   if (hypre_ParCSRCommPkgSendProcs(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgSendProcs(comm_pkg));
   } 
   if (hypre_ParCSRCommPkgSendMapElmts(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgSendMapElmts(comm_pkg));
   }
   if (hypre_ParCSRCommPkgSendMapStarts(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgSendMapStarts(comm_pkg));
   }
   if (hypre_ParCSRCommPkgRecvProcs(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgRecvProcs(comm_pkg));
   }
   if (hypre_ParCSRCommPkgRecvVecStarts(comm_pkg))
   {
      hypre_TFree(hypre_ParCSRCommPkgRecvVecStarts(comm_pkg));
   }

   hypre_TFree(comm_pkg);
   hypre_ParCSRMatrixCommPkg(parcsr_A) = NULL;  /*this gets freed again in destroy 
                                                  parscr since there are two comm 
                                                  packages now*/  

   return hypre_error_flag;
}




/*--------------------------------------------------------------------
 * hypre_RangeFillResponseIJDetermineRecvProcs
 * Fill response function for determining the recv. processors
 * data exchange
 *--------------------------------------------------------------------*/

int
hypre_RangeFillResponseIJDetermineRecvProcs(void *p_recv_contact_buf,
                                      int contact_size, int contact_proc, void *ro, 
                                      MPI_Comm comm, void **p_send_response_buf, 
                                      int *response_message_size)
{
   int    myid, tmp_id;
   HYPRE_BigInt row_end;
   int    j;
   int    index, size;
   HYPRE_BigInt    row_val;
   
   HYPRE_BigInt   *send_response_buf = (HYPRE_BigInt *) *p_send_response_buf;
   HYPRE_BigInt   *recv_contact_buf = (HYPRE_BigInt * ) p_recv_contact_buf;


   hypre_DataExchangeResponse  *response_obj = ro; 
   hypre_IJAssumedPart               *part = response_obj->data1;
   
   int overhead = response_obj->send_response_overhead;

   /*-------------------------------------------------------------------
    * we are getting a range of off_d entries - need to see if we own them
    * or how many ranges to send back  - send back
    * with format [proc_id end_row  proc_id #end_row  proc_id #end_row etc...].
    *----------------------------------------------------------------------*/ 

   MPI_Comm_rank(comm, &myid );


   /* populate send_response_buf */
      
   index = 0; /*count entries in send_response_buf*/
   
   j = 0; /*marks which partition of the assumed partition we are in */
   row_val = recv_contact_buf[0]; /*beginning of range*/
   row_end = part->row_end_list[part->sort_index[j]];
   tmp_id = part->proc_list[part->sort_index[j]];

   /*check storage in send_buf for adding the ranges */
   size = 2*(part->length);
         
   if ( response_obj->send_response_storage  < size  )
   {

      response_obj->send_response_storage =  hypre_max(size, 20); 
      send_response_buf = hypre_TReAlloc( send_response_buf, HYPRE_BigInt, 
                                         response_obj->send_response_storage + overhead );
      *p_send_response_buf = send_response_buf;    /* needed when using ReAlloc */
   }


   while (row_val > row_end) /*which partition to start in */
   {
      j++;
      row_end = part->row_end_list[part->sort_index[j]];   
      tmp_id = part->proc_list[part->sort_index[j]];
   }

   /*add this range*/
   send_response_buf[index++] = (HYPRE_BigInt)tmp_id;
   send_response_buf[index++] = row_end; 

   j++; /*increase j to look in next partition */
      
    
   /*any more?  - now compare with end of range value*/
   row_val = recv_contact_buf[1]; /*end of range*/
   while ( j < part->length && row_val > row_end  )
   {
      row_end = part->row_end_list[part->sort_index[j]];  
      tmp_id = part->proc_list[part->sort_index[j]];

      send_response_buf[index++] = (HYPRE_BigInt)tmp_id;
      send_response_buf[index++] = row_end; 

      j++;
      
   }


   *response_message_size = index;
   *p_send_response_buf = send_response_buf;

   return hypre_error_flag;

}



/*--------------------------------------------------------------------
 * hypre_FillResponseIJDetermineSendProcs
 * Fill response function for determining the send processors
 * data exchange
 *--------------------------------------------------------------------*/

int
hypre_FillResponseIJDetermineSendProcs(void *p_recv_contact_buf, 
                                 int contact_size, int contact_proc, void *ro, 
                                 MPI_Comm comm, void **p_send_response_buf, 
                                 int *response_message_size )
{
   int    myid;
   int    i, index, count, elength;

   HYPRE_BigInt    *recv_contact_buf = (HYPRE_BigInt * ) p_recv_contact_buf;

   hypre_DataExchangeResponse  *response_obj = ro;  

   hypre_ProcListElements      *send_proc_obj = response_obj->data2;   


   MPI_Comm_rank(comm, &myid );


   /*check to see if we need to allocate more space in send_proc_obj for ids*/
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length +=20; /*add space for 20 more processors*/
      send_proc_obj->id = hypre_TReAlloc(send_proc_obj->id,int, 
					 send_proc_obj->storage_length);
      send_proc_obj->vec_starts = hypre_TReAlloc(send_proc_obj->vec_starts,int, 
                                  send_proc_obj->storage_length + 1);
   }
  
   /*initialize*/ 
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /*this is the number of elements*/

   /*send proc*/ 
   send_proc_obj->id[count] = contact_proc; 

   /*do we need more storage for the elements?*/
     if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = hypre_max(contact_size, 50);   
      elength += index;
      send_proc_obj->elements = hypre_TReAlloc(send_proc_obj->elements, 
					       HYPRE_BigInt, elength);
      send_proc_obj->element_storage_length = elength; 
   }
   /*populate send_proc_obj*/
   for (i=0; i< contact_size; i++) 
   { 
      send_proc_obj->elements[index++] = recv_contact_buf[i];
   }
   send_proc_obj->vec_starts[count+1] = index;
   send_proc_obj->length++;
   

  /*output - no message to return (confirmation) */
   *response_message_size = 0; 
  
   return hypre_error_flag;

}

