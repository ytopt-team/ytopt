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

#ifndef hypre_EXCHANGE_DATA_HEADER
#define hypre_EXCHANGE_DATA_HEADER

#define hypre_BinaryTreeParentId(tree)      (tree->parent_id)
#define hypre_BinaryTreeNumChild(tree)      (tree->num_child)
#define hypre_BinaryTreeChildIds(tree)      (tree->child_id)
#define hypre_BinaryTreeChildId(tree, i)    (tree->child_id[i])


typedef struct
{
   int                   parent_id;
   int                   num_child;
   int		        *child_id;
} hypre_BinaryTree;



/* In the fill_response() function the user needs to set the recv__buf
   and the response_message_size.  Memory of size send_response_storage has been
   alllocated for the send_buf (in exchange_data) - if more is needed, then
   realloc and adjust
   the send_response_storage.  The realloc amount should be storage+overhead. 
   If the response is an empty "confirmation" message, then set
   response_message_size =0 (and do not modify the send_buf) */


typedef struct
{
   int    (*fill_response)(void* recv_buf, int contact_size, 
                           int contact_proc, void* response_obj, 
                           MPI_Comm comm, void** response_buf, 
                           int* response_message_size);
   int     send_response_overhead; /*set by exchange data */
   int     send_response_storage;  /*storage allocated for send_response_buf*/
   void    *data1;                 /*data fields user may want to access in fill_response */
   void    *data2;
   
} hypre_DataExchangeResponse;


int hypre_CreateBinaryTree(int, int, hypre_BinaryTree*);
int hypre_DestroyBinaryTree(hypre_BinaryTree*);


int hypre_DataExchangeList(int num_contacts, 
		     int *contact_proc_list, void *contact_send_buf, 
		     int *contact_send_buf_starts, int contact_obj_size, 
                     int response_obj_size,
		     hypre_DataExchangeResponse *response_obj, int max_response_size, 
                     int rnum, MPI_Comm comm,  void **p_response_recv_buf, 
                     int **p_response_recv_buf_starts);


#endif /* end of header */
