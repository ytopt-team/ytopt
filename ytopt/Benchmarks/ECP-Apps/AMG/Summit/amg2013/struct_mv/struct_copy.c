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
 * Structured copy routine
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * hypre_StructCopy
 *--------------------------------------------------------------------------*/

int
hypre_StructCopy( hypre_StructVector *x,
                  hypre_StructVector *y     )
{
   int ierr = 0;

   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;
                   
   int              xi;
   int              yi;
                   
   double          *xp;
   double          *yp;
                   
   hypre_BoxArray  *boxes;
   hypre_Box       *box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;
                   
   int              i;
   int              loopi, loopj, loopk;

   hypre_SetIndex(unit_stride, 1, 1, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
      {
         box   = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

         xp = hypre_StructVectorBoxData(x, i);
         yp = hypre_StructVectorBoxData(y, i);

         hypre_BoxGetSize(box, loop_size);

         hypre_BoxLoop2Begin(loop_size,
                             x_data_box, start, unit_stride, xi,
                             y_data_box, start, unit_stride, yi);

	 hypre_BoxLoop2For(loopi, loopj, loopk, xi, yi)
            {
               yp[yi] = xp[xi];
            }
         hypre_BoxLoop2End(xi, yi);
      }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructPartialCopy: copy only the components on a subset of the grid.
 * A BoxArrayArray of boxes are needed- for each box of x, only an array
 * of subboxes (i.e., a boxarray for each box of x) are copied.
 *--------------------------------------------------------------------------*/

int
hypre_StructPartialCopy( hypre_StructVector  *x,
                         hypre_StructVector  *y,    
                         hypre_BoxArrayArray *array_boxes )
{
   int ierr = 0;

   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;

   int              xi;
   int              yi;

   double          *xp;
   double          *yp;

   hypre_BoxArray  *boxes;
   hypre_Box       *box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;

   int              i, j ;
   int              loopi, loopj, loopk;

   hypre_SetIndex(unit_stride, 1, 1, 1);

   hypre_ForBoxArrayI(i, array_boxes)
      {
         boxes = hypre_BoxArrayArrayBoxArray(array_boxes, i);

         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

         xp = hypre_StructVectorBoxData(x, i);
         yp = hypre_StructVectorBoxData(y, i);

         /* array of sub_boxes of box_i of the vector */
         hypre_ForBoxI(j, boxes)
         {
            box = hypre_BoxArrayBox(boxes, j);

            start = hypre_BoxIMin(box);
            hypre_BoxGetSize(box, loop_size);

            hypre_BoxLoop2Begin(loop_size,
                                x_data_box, start, unit_stride, xi,
                                y_data_box, start, unit_stride, yi);

            hypre_BoxLoop2For(loopi, loopj, loopk, xi, yi)
              {
                  yp[yi] = xp[xi];
              }
            hypre_BoxLoop2End(xi, yi);
         }
      }

   return ierr;
}
