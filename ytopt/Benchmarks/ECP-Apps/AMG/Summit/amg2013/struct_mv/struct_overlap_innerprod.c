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
 * Structured inner product routine for overlapped grids. Computes the
 * inner product of two vectors over an overlapped grid.
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_StructOverlapInnerProd
 *--------------------------------------------------------------------------*/

#ifdef HYPRE_USE_PTHREADS
double          *local_result_ref[hypre_MAX_THREADS];
#endif

double           final_innerprod_result;


double
hypre_StructOverlapInnerProd( hypre_StructVector *x,
                              hypre_StructVector *y )
{
   double               local_result, overlap_result;
   double               process_result;
                   
   hypre_Box           *x_data_box;
   hypre_Box           *y_data_box;

   hypre_BoxArray      *overlap_boxes;
                   
   int                  xi;
   int                  yi;
                   
   double              *xp;
   double              *yp;
                   
   hypre_BoxArray      *boxes;
   hypre_Box           *boxi, *boxj, intersect_box;
   hypre_BoxNeighbors  *neighbors= hypre_StructGridNeighbors(hypre_StructVectorGrid(y));
   int                 *neighbors_procs= hypre_BoxNeighborsProcs(neighbors);
   hypre_BoxArray      *selected_nboxes;
   hypre_BoxArray      *tmp_box_array, *tmp2_box_array;

   hypre_Index          loop_size;
   hypre_IndexRef       start;
   hypre_Index          unit_stride;
                   
   int                  i, j;
   int                  myid;
   int                  boxarray_size;
   int                  loopi, loopj, loopk;
#ifdef HYPRE_USE_PTHREADS
   int                  threadid = hypre_GetThreadID();
#endif

   
   local_result = 0.0;
   process_result = 0.0;
   hypre_SetIndex(unit_stride, 1, 1, 1);

   MPI_Comm_rank(hypre_StructVectorComm(y), &myid);

   /*-----------------------------------------------------------------------
    * Determine the overlapped boxes on this local processor.
    *-----------------------------------------------------------------------*/
   boxes        = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   boxarray_size= hypre_BoxArraySize(boxes);

   /*-----------------------------------------------------------------------
    * To compute the inner product over this local processor, given a box, 
    * the inner product between x & y is computed over the whole box and
    * over any overlapping between this box and overlap_boxes. The latter
    * result is subtracted from the former. Overlapping between more than
    * two boxes are handled.
    *-----------------------------------------------------------------------*/
   hypre_ForBoxI(i, boxes)
   {
      boxi  = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(boxi);

      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      hypre_BoxGetSize(boxi, loop_size);

#ifdef HYPRE_USE_PTHREADS
   local_result_ref[threadid] = &local_result;
#endif

      hypre_BoxLoop2Begin(loop_size,
                          x_data_box, start, unit_stride, xi,
                          y_data_box, start, unit_stride, yi);

      hypre_BoxLoop2For(loopi, loopj, loopk, xi, yi)
      {
          local_result += xp[xi] * yp[yi];
      }
      hypre_BoxLoop2End(xi, yi);

      /*--------------------------------------------------------------------
       * intersect all boxes from (i+1) to boxarray_size 
       *--------------------------------------------------------------------*/
      overlap_boxes= hypre_BoxArrayCreate(0);
      for (j= (i+1); j< boxarray_size; j++)
      {
         boxj= hypre_BoxArrayBox(boxes, j);
         hypre_IntersectBoxes(boxi, boxj, &intersect_box);

         if (hypre_BoxVolume(&intersect_box))
         {
             hypre_AppendBox(&intersect_box, overlap_boxes);
         }
      }

      if (hypre_BoxArraySize(overlap_boxes) > 1)
      {
         hypre_UnionBoxes(overlap_boxes);
      }

      /*--------------------------------------------------------------------
       * compute inner product over overlap  
       *--------------------------------------------------------------------*/
      if (hypre_BoxArraySize(overlap_boxes))
      {
         overlap_result= 0.0;
         hypre_ForBoxI(j, overlap_boxes)
         {
            boxj  = hypre_BoxArrayBox(overlap_boxes, j);
            start = hypre_BoxIMin(boxj);

            hypre_BoxGetSize(boxj, loop_size);
            hypre_BoxLoop2Begin(loop_size,
                                x_data_box, start, unit_stride, xi,
                                y_data_box, start, unit_stride, yi);

            hypre_BoxLoop2For(loopi, loopj, loopk, xi, yi)
            {
               overlap_result += xp[xi] * yp[yi];
            }
            hypre_BoxLoop2End(xi, yi);
         }
         local_result-= overlap_result;
      }
      hypre_BoxArrayDestroy(overlap_boxes);
   }

   /*-----------------------------------------------------------------------
    * Determine the across processor overlap. The inner product is computed
    * and subtracted on processors that share the overlap except the one
    * with the lowest processor id. Therefore, on this processor, we need 
    * to subtract only overlaps with boxes on processors with id < myid.
    *-----------------------------------------------------------------------*/
   boxes= hypre_BoxNeighborsBoxes(neighbors);
   selected_nboxes= hypre_BoxArrayCreate(0);

   /* extract only boxes on processors with ids < myid. */
   hypre_ForBoxI(i, boxes)
   {
      if (neighbors_procs[i] < myid)
      {
         hypre_AppendBox(hypre_BoxArrayBox(boxes, i), selected_nboxes);
      }
   }

   boxes= hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   overlap_boxes= hypre_BoxArrayCreate(0);
   hypre_ForBoxI(i, boxes)
   {
      boxi= hypre_BoxArrayBox(boxes, i);
      hypre_ForBoxI(j, selected_nboxes)
      {
         boxj= hypre_BoxArrayBox(selected_nboxes, j);
   
         hypre_IntersectBoxes(boxi, boxj, &intersect_box);
         if (hypre_BoxVolume(&intersect_box))
         {
             hypre_AppendBox(&intersect_box, overlap_boxes);
         }
      }
   }
   hypre_BoxArrayDestroy(selected_nboxes);

   /*-----------------------------------------------------------------------
    * Union the overlap_boxes and then begin to compute and subtract chunks 
    * and norms.
    *-----------------------------------------------------------------------*/
   if (hypre_BoxArraySize(overlap_boxes) > 1)
   {
      hypre_UnionBoxes(overlap_boxes);
   }

   if (hypre_BoxArraySize(overlap_boxes))
   {
      overlap_result= 0.0;
      hypre_ForBoxI(i, boxes)
      {
         boxi  = hypre_BoxArrayBox(boxes, i);

         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
         xp = hypre_StructVectorBoxData(x, i);
         yp = hypre_StructVectorBoxData(y, i);

         tmp_box_array= hypre_BoxArrayCreate(0);
         hypre_ForBoxI(j, overlap_boxes)
         {
            boxj= hypre_BoxArrayBox(overlap_boxes, j);

            hypre_IntersectBoxes(boxi, boxj, &intersect_box);
            if (hypre_BoxVolume(&intersect_box))
            {
               start= hypre_BoxIMin(&intersect_box);
               hypre_BoxGetSize(&intersect_box, loop_size);

               hypre_BoxLoop2Begin(loop_size,
                                   x_data_box, start, unit_stride, xi,
                                   y_data_box, start, unit_stride, yi);

               hypre_BoxLoop2For(loopi, loopj, loopk, xi, yi)
               {
                  overlap_result += xp[xi] * yp[yi];
               }
               hypre_BoxLoop2End(xi, yi);

               hypre_AppendBox(&intersect_box, tmp_box_array);

            }  /* if (hypre_BoxVolume(&intersect_box)) */
         }     /* hypre_ForBoxI(j, overlap_boxes) */

         /*-------------------------------------------------------------------------
          * Subtract the intersection boxes so that norm on higher degree overlaps
          * on this processor are computed only once.
          *-------------------------------------------------------------------------*/
         tmp2_box_array= hypre_BoxArrayCreate(0);
         hypre_SubtractBoxArrays(overlap_boxes, tmp_box_array, tmp2_box_array);
         hypre_BoxArrayDestroy(tmp_box_array);
         hypre_BoxArrayDestroy(tmp2_box_array);

      }  /* hypre_ForBoxI(i, boxes) */

      local_result-= overlap_result;
   }  /* if (hypre_BoxArraySize(overlap_boxes)) */

   hypre_BoxArrayDestroy(overlap_boxes);

#ifdef HYPRE_USE_PTHREADS
   if (threadid != hypre_NumThreads)
   {
      for (i = 0; i < hypre_NumThreads; i++)
         process_result += *local_result_ref[i];
   }
   else
      process_result = *local_result_ref[threadid];
#else
   process_result = local_result;
#endif


   MPI_Allreduce(&process_result, &final_innerprod_result, 1,
                 MPI_DOUBLE, MPI_SUM, hypre_StructVectorComm(x));


#ifdef HYPRE_USE_PTHREADS
   if (threadid == 0 || threadid == hypre_NumThreads)
#endif
   hypre_IncFLOPCount(2*hypre_StructVectorGlobalSize(x));

   return final_innerprod_result;
}
