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

/* see exchange_data.README for additional information */
/* AHB 6/04 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"


/*---------------------------------------------------
 * hypre_CreateBinaryTree()
 * its children and parent processor ids)
 *----------------------------------------------------*/

int hypre_CreateBinaryTree(int myid, int num_procs, hypre_BinaryTree *tree)
{

   int  i, proc, size=0;
   int  ierr = 0;
   int  *tmp_child_id;
   int  num=0, parent = 0;

   /* initialize*/
   proc = myid;
   


   /*how many children can a processor have?*/
   for (i = 1; i < num_procs; i *= 2)
   {  
     size++;
   }

   /* allocate space */
      tmp_child_id = hypre_TAlloc(int, size);

   /* find children and parent */
   for (i = 1; i < num_procs; i *= 2)
   {
     if ( (proc % 2) == 0)
     {
        if( (myid + i) < num_procs )
        {
	   tmp_child_id[num] = myid + i;
           num++;
         }
	proc /= 2;
     }
     else
     {
        parent = myid - i;
        break;
     }

   }
   
   hypre_BinaryTreeParentId(tree) = parent;
   hypre_BinaryTreeNumChild(tree) = num;
   hypre_BinaryTreeChildIds(tree) = tmp_child_id;

   return(ierr);

}

/*---------------------------------------------------
 * hypre_DestroyBinaryTree()
 * Destroy storage created by createBinaryTree
 *----------------------------------------------------*/
int hypre_DestroyBinaryTree(hypre_BinaryTree *tree) 
{

  int ierr = 0;

  hypre_TFree(hypre_BinaryTreeChildIds(tree));
 
  return ierr;
}


/*---------------------------------------------------
 * hypre_DataExchangeList()
 * This function is for sending a list of messages ("contacts" to
 * a list of processors.  The receiving processors
 * do not know how many messages they are getting. The
 * sending process expects a "response" (either a confirmation or 
 * some sort of data back from the receiving processor).
 *  
 *----------------------------------------------------*/



/* should change to where the buffers for sending and receiving are voids
   instead of ints - then cast accordingly */

int hypre_DataExchangeList(int num_contacts, 
		     int *contact_proc_list, void *contact_send_buf, 
		     int *contact_send_buf_starts, int contact_obj_size, 
                     int response_obj_size,
		     hypre_DataExchangeResponse *response_obj, int max_response_size, 
                     int rnum, MPI_Comm comm,  void **p_response_recv_buf, 
                     int **p_response_recv_buf_starts)
{  


   /*-------------------------------------------
    *  parameters: 
    *   
    *    num_contacts              = how many procs to contact
    *    contact_proc_list         = list of processors to contact 
    *    contact_send_buf          = array of data to send
    *    contact_send_buf_starts   = index for contact_send_buf corresponding to 
    *                                contact_proc_list
    *    contact_obj_size          = sizeof() one obj in contact list  
   
    *    response_obj_size          = sizeof() one obj in response_recv_buf 
    *    response_obj              = this will give us the function we need to 
    *                                fill the reponse as well as 
    *                                any data we might need to accomplish that 
    *    max_response_size         = max size of a single response expected (do NOT
    *                                need to be an absolute upper bound)  
    *    rnum                      = two consequentive exchanges should have different 
    *                                rnums. Alternate rnum = 1 
    *                                and rnum=2  - these flags will be even (so odd
    *                                numbered tags could be used in calling code) 
    *    p_response_recv_buf       = where to receive the reponses - will be allocated 
    *                                in this function  
    *    p_response_recv_buf_starts  = index of p_response_buf corresponding to 
    *                                contact_buf_list - will be allocated here  

    *-------------------------------------------*/

   int  num_procs, myid;
   int  ierr = 0;
   int  i;
   int  terminate, responses_complete;
   int  children_complete;
   int  contact_flag;
   int  proc;
   int  contact_size;

   int  size, post_size, copy_size;
   int  total_size, count;      

   void *start_ptr = NULL, *index_ptr=NULL;
   int  *int_ptr=NULL;  

   void *response_recv_buf = NULL;
   void *send_response_buf = NULL;

   int  *response_recv_buf_starts = NULL;
   void *initial_recv_buf = NULL;  

   void *recv_contact_buf = NULL;
   int  recv_contact_buf_size = 0;
   
   int  response_message_size = 0;

   int  overhead;
 
   int  max_response_size_bytes;
   
   int  max_response_total_bytes;
   
   void **post_array = NULL;  /*this must be set to null or realloc will crash */
   int  post_array_storage = 0;
   int  post_array_size = 0;
   int   num_post_recvs =0;
   
   void **contact_ptrs = NULL, **response_ptrs=NULL, **post_ptrs=NULL;
   


   hypre_BinaryTree tree;

   MPI_Request *response_requests, *contact_requests;
   MPI_Status  *response_statuses, *contact_statuses;
   
   MPI_Request  *post_send_requests = NULL, *post_recv_requests = NULL;
   MPI_Status   *post_send_statuses = NULL, *post_recv_statuses = NULL;

   MPI_Request *term_requests, term_request1, request_parent;
   MPI_Status  *term_statuses, term_status1, status_parent;
   MPI_Status  status, fill_status;
  

   const int contact_tag = 1000*rnum;
   const int response_tag = 1002*rnum;
   const int term_tag =  1004*rnum;  
   const int post_tag = 1006*rnum;
   

   MPI_Comm_size(comm, &num_procs );
   MPI_Comm_rank(comm, &myid );


   /* ---------initializations ----------------*/ 



   /*if the response_obj_size or contact_obj_size is 0, set to sizeof(int) */
   if (!response_obj_size) response_obj_size = sizeof(int);
   if (!contact_obj_size) contact_obj_size = sizeof(int);

   max_response_size_bytes = max_response_size*response_obj_size;
 
                   
   /* pre-allocate the max space for responding to contacts */
   overhead = ceil((double) sizeof(int)/response_obj_size); /*for appending an integer*/
   
   max_response_total_bytes = (max_response_size+overhead)*response_obj_size;

  

   response_obj->send_response_overhead = overhead;
   response_obj->send_response_storage = max_response_size;

   send_response_buf = hypre_MAlloc(max_response_total_bytes);
      
   /*allocate space for inital recv array for the responses - give each processor
     size max_response_size */

   initial_recv_buf = hypre_MAlloc(max_response_total_bytes*num_contacts);
   response_recv_buf_starts =   hypre_CTAlloc(int, num_contacts+1);  


   contact_ptrs = hypre_TAlloc( void *, num_contacts);
   response_ptrs = hypre_TAlloc(void *, num_contacts);
   

   /*-------------SEND CONTACTS AND POST RECVS FOR RESPONSES---*/

   for (i=0; i<= num_contacts; i++)
   {
      response_recv_buf_starts[i] = i*(max_response_size+overhead);
   }
   
   /* Send "contact" messages to the list of processors and 
      pre-post receives to wait for their response*/

   responses_complete = 1;
   if (num_contacts > 0 )
   {
      responses_complete = 0;
      response_requests = hypre_CTAlloc(MPI_Request, num_contacts);
      response_statuses = hypre_CTAlloc(MPI_Status, num_contacts);
      contact_requests = hypre_CTAlloc(MPI_Request, num_contacts);
      contact_statuses = hypre_CTAlloc(MPI_Status, num_contacts);

      /* post receives - could be confirmation or data*/
      /* the size to post is max_response_total_bytes*/     
      
      for (i=0; i< num_contacts; i++) 
      {
         /* response_ptrs[i] =  initial_recv_buf + i*max_response_total_bytes ; */
         response_ptrs[i] =  (void *)((char *) initial_recv_buf + i*max_response_total_bytes) ; 

         MPI_Irecv(response_ptrs[i], max_response_total_bytes, MPI_BYTE, contact_proc_list[i], 
                   response_tag, comm, &response_requests[i]);

      }


      /* send out contact messages */
      start_ptr = contact_send_buf;
      for (i=0; i< num_contacts; i++) 
      {
         contact_ptrs[i] = start_ptr; 
         size =  contact_send_buf_starts[i+1] - contact_send_buf_starts[i]  ; 
         MPI_Isend(contact_ptrs[i], size*contact_obj_size, MPI_BYTE, contact_proc_list[i], 
                   contact_tag, comm, &contact_requests[i]); 
         /*  start_ptr += (size*contact_obj_size); */        
         start_ptr = (void *) ((char *) start_ptr  + (size*contact_obj_size)); 


      }
   }


   /*------------BINARY TREE-----------------------*/


     /*initialize for the termination check sweep */
   children_complete = 1;/*indicates whether we have recv. term messages 
                           from our children*/
 
   if (num_procs > 1) { 
      ierr = hypre_CreateBinaryTree(myid, num_procs, &tree);

      /* we will get a message from all of our children when they 
        have received responses for all of their contacts.  
        So post receives now */

   
      term_requests = hypre_CTAlloc(MPI_Request, tree.num_child); 
      term_statuses = hypre_CTAlloc(MPI_Status, tree.num_child);

      for (i=0; i< tree.num_child; i++) 
      {
	 MPI_Irecv(NULL, 0, MPI_INT, tree.child_id[i], term_tag, comm, 
		   &term_requests[i]);
      }

      terminate = 0;
 
      children_complete = 0;
      
   }
   else if (num_procs ==1 && num_contacts > 0 ) /* added 11/08 */
   {
      terminate = 0;
   }
   

   /*---------PROBE LOOP-----------------------------------------*/

  /*Look for incoming contact messages - don't know how many I will get!*/
  
   while (!terminate)
   {

      /* did I receive any contact messages? */
      MPI_Iprobe(MPI_ANY_SOURCE, contact_tag, comm, &contact_flag, &status);

      while (contact_flag)
      {
         /* received contacts - from who and what do we do ?*/
         proc = status.MPI_SOURCE;
         MPI_Get_count(&status, MPI_BYTE, &contact_size);

         contact_size = contact_size/contact_obj_size;
         

         /*---------------FILL RESPONSE ------------------------*/

         /*first receive the contact buffer - then call a function
           to determine how to populate the send buffer for the reponse*/

         /* do we have enough space to recv it? */
         if(contact_size > recv_contact_buf_size) 
         {
            recv_contact_buf = hypre_ReAlloc(recv_contact_buf, 
                                              contact_obj_size*contact_size);
            recv_contact_buf_size = contact_size;
         } 


         /* this must be blocking - can't fill recv without the buffer*/         
         MPI_Recv(recv_contact_buf, contact_size*contact_obj_size, MPI_BYTE, proc, 
            contact_tag, comm, &fill_status);



         response_obj->fill_response(recv_contact_buf, contact_size, proc, response_obj, 
                                     comm, &send_response_buf, &response_message_size );


        /* we need to append the size of the send obj */
        /* first we copy out any part that may be needed to send later so we don't overwrite */
        post_size = response_message_size - max_response_size; 
        if (post_size > 0) /*we will need to send the extra information later */   
        {
           /*printf("myid = %d, post_size = %d\n", myid, post_size);*/

           if (post_array_size == post_array_storage)
             
           {
              /* allocate room for more posts  - add 20*/
              post_array_storage += 20;       
              post_array = hypre_TReAlloc(post_array, void *, post_array_storage); 
              post_send_requests = hypre_TReAlloc(post_send_requests, MPI_Request, post_array_storage);
           }
           /* allocate space for the data this post only*/
           /* this should not happen often (unless a poor max_size has been chosen)
             - so we will allocate space for the data as needed */
           size = post_size*response_obj_size;
           post_array[post_array_size] =  hypre_MAlloc(size); 
           /* index_ptr =  send_response_buf + max_response_size_bytes */;
           index_ptr =  (void *) ((char *) send_response_buf + max_response_size_bytes);
               
           memcpy(post_array[post_array_size], index_ptr, size);
    
        /*now post any part of the message that is too long with a non-blocking send and
          a different tag */

           MPI_Isend(post_array[post_array_size], size, MPI_BYTE, proc, post_tag, 
                     comm, &post_send_requests[post_array_size]); 

           post_array_size++;
        }

       /*now append the size information into the overhead storage */
       /* index_ptr =  send_response_buf + max_response_size_bytes; */
        index_ptr =  (void *) ((char *) send_response_buf + max_response_size_bytes);

        memcpy(index_ptr, &response_message_size, response_obj_size);

         
        /*send the block of data that includes the overhead */        
        /* this is a blocking send - the recv has already been posted */
        MPI_Send(send_response_buf, max_response_total_bytes, 
                 MPI_BYTE, proc, response_tag, comm); 

    
        /*--------------------------------------------------------------*/

      
	/* look for any more contact messages*/
        MPI_Iprobe(MPI_ANY_SOURCE, contact_tag, comm, &contact_flag, &status);

      }
    
      /* no more contact messages waiting - either 
         (1) check to see if we have received all of our response messages
         (2) partitcipate in the termination sweep (check for messages from children) 
         (3) participate in termination sweep (check for message from parent)*/
    
      if (!responses_complete) 
      {
	ierr = MPI_Testall(num_contacts, response_requests, &responses_complete, 
                    response_statuses);
        if (responses_complete && num_procs == 1) terminate = 1; /*added 11/08 */

      }
      else if(!children_complete) /* have all of our children received all of their 
                                     response messages?*/
      {

         
         ierr = MPI_Testall(tree.num_child, term_requests, &children_complete, 
			    term_statuses);

        

         /* if we have gotten term messages from all of our children, send a term 
            message to our parent.  Then post a receive to hear back from parent */ 
	 if (children_complete & (myid > 0)) /*root does not have a parent*/ 
         {

       
           MPI_Isend(NULL, 0, MPI_INT, tree.parent_id, term_tag, comm, &request_parent);
            
           MPI_Irecv(NULL, 0, MPI_INT, tree.parent_id, term_tag, comm, 
		     &term_request1);
         }
      }
      else /*have we gotten a term message from our parent? */
      {
	if (myid == 0) /* root doesn't have a parent */
        {
	  terminate = 1;
        } 
        else  
        {
	   MPI_Test(&term_request1, &terminate, &term_status1);
        }  
         if(terminate) /*tell children to terminate */
	 {
            if (myid > 0 ) MPI_Wait(&request_parent, &status_parent);
            
	    for (i=0; i< tree.num_child; i++)
	    {  /*a blocking send  - recv has been posted already*/
	       MPI_Send(NULL, 0, MPI_INT, tree.child_id[i], term_tag, comm);
      
	    }
	 }
      } 
  }

   /* end of (!terminate) loop */


   /* ----some clean up before post-processing ----*/
   if (recv_contact_buf_size > 0)
   {
      hypre_TFree(recv_contact_buf);
   }
  
   hypre_Free(send_response_buf);
   hypre_TFree(contact_ptrs);
   hypre_TFree(response_ptrs);
     



   /*-----------------POST PROCESSING------------------------------*/

   /* more data to receive? */
   /* move to recv buffer and update response_recv_buf_starts */

   total_size = 0;  /*total number of items in response buffer */
   num_post_recvs = 0; /*num of post processing recvs to post */
   start_ptr = initial_recv_buf;
   response_recv_buf_starts[0] = 0; /*already allocated above */

  /*an extra loop to determine sizes.  This is better than reallocating
     the array that will be used in posting the irecvs */ 
   for (i=0; i< num_contacts; i++)
   {
      /* int_ptr = (int *) (start_ptr + max_response_size_bytes);  */ /*the overhead int*/
      int_ptr = (int *) ( (char *) start_ptr + max_response_size_bytes); /*the overhead int*/


      response_message_size =  *int_ptr;
      response_recv_buf_starts[i+1] = response_recv_buf_starts[i] + response_message_size;
      total_size +=  response_message_size;
      if (max_response_size < response_message_size) num_post_recvs++;
      /* start_ptr += max_response_total_bytes; */
      start_ptr =  (void *) ((char *) start_ptr + max_response_total_bytes);
   }
      
   post_recv_requests = hypre_TAlloc(MPI_Request, num_post_recvs);
   post_recv_statuses = hypre_TAlloc(MPI_Status, num_post_recvs);
   post_ptrs = hypre_TAlloc(void *, num_post_recvs);

   /*second loop to post any recvs and set up recv_response_buf */
   response_recv_buf = hypre_MAlloc(total_size*response_obj_size);
   index_ptr = response_recv_buf;
   start_ptr = initial_recv_buf;
   count = 0;
   
   for (i=0; i< num_contacts; i++)
   {
      
      response_message_size =  response_recv_buf_starts[i+1] -  response_recv_buf_starts[i];
      copy_size = hypre_min(response_message_size, max_response_size);
      
      memcpy(index_ptr, start_ptr, copy_size*response_obj_size);   
      /* index_ptr += copy_size*response_obj_size; */  
      index_ptr = (void *) ((char *) index_ptr + copy_size*response_obj_size);  

      if (max_response_size < response_message_size)
      {
         size = (response_message_size - max_response_size)*response_obj_size;
         post_ptrs[count] = index_ptr;
         MPI_Irecv(post_ptrs[count], size, MPI_BYTE, contact_proc_list[i]     ,
                   post_tag,  comm, &post_recv_requests[count]);
         count++;
         /* index_ptr+=size;*/        
          index_ptr=  (void *) ((char *) index_ptr + size);  
      }
        
      /* start_ptr += max_response_total_bytes; */
      start_ptr = (void *) ((char *) start_ptr + max_response_total_bytes);

   }

   post_send_statuses = hypre_TAlloc(MPI_Status, post_array_size);


   /*--------------CLEAN UP------------------- */
  
   hypre_Free(initial_recv_buf);
      

   if (num_contacts > 0 ) 
   {
      /*these should be done */
      ierr = MPI_Waitall(num_contacts, contact_requests, 
                         contact_statuses);
     

      hypre_TFree(response_requests);
      hypre_TFree(response_statuses);
      hypre_TFree(contact_requests);
      hypre_TFree(contact_statuses);
      

   }

   /* clean up from the post processing - the arrays, requests, etc. */

   if (num_post_recvs)
   {
      ierr = MPI_Waitall(num_post_recvs, post_recv_requests, 
                         post_recv_statuses);  
      hypre_TFree(post_recv_requests);
      hypre_TFree(post_recv_statuses);
      hypre_TFree(post_ptrs);
   }
   

   if (post_array_size)
   {
      ierr = MPI_Waitall(post_array_size, post_send_requests, 
                      post_send_statuses);

      hypre_TFree(post_send_requests);
      hypre_TFree(post_send_statuses);
 
      for (i=0; i< post_array_size; i++)
      {
         hypre_Free(post_array[i]);
      }
      hypre_TFree(post_array);
   
   }
   

   if (num_procs > 1)
   {
      hypre_TFree(term_requests);
      hypre_TFree(term_statuses);
 
      ierr = hypre_DestroyBinaryTree(&tree);

   }

   /* output  */
   *p_response_recv_buf = response_recv_buf;
   *p_response_recv_buf_starts = response_recv_buf_starts;


   return(ierr);



}
