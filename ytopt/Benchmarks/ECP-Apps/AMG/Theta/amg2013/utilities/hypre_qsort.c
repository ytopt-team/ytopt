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

#include "HYPRE.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void swap( int *v,
           int  i,
           int  j )
{
   int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void swap2(int     *v,
           double  *w,
           int      i,
           int      j )
{
   int temp;
   double temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_swap2i(int  *v,
                  int  *w,
                  int  i,
                  int  j )
{
   int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/


/* AB 11/04 */

void hypre_swap3i(int  *v,
                  int  *w,
                  int  *z,
                  int  i,
                  int  j )
{
   int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
   temp = z[i];
   z[i] = z[j];
   z[j] = temp;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void qsort0( int *v,
             int  left,
             int  right )
{
   int i, last;

   if (left >= right)
      return;
   swap( v, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (v[i] < v[left])
      {
         swap(v, ++last, i);
      }
   swap(v, left, last);
   qsort0(v, left, last-1);
   qsort0(v, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void qsort1( int *v,
	     double *w,
             int  left,
             int  right )
{
   int i, last;

   if (left >= right)
      return;
   swap2( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (v[i] < v[left])
      {
         swap2(v, w, ++last, i);
      }
   swap2(v, w, left, last);
   qsort1(v, w, left, last-1);
   qsort1(v, w, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_qsort2i( int *v,
                    int *w,
                    int  left,
                    int  right )
{
   int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_swap2i( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_swap2i(v, w, ++last, i);
      }
   }
   hypre_swap2i(v, w, left, last);
   hypre_qsort2i(v, w, left, last-1);
   hypre_qsort2i(v, w, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*   sort on w (double), move v (AB 11/04) */


void hypre_qsort2( int *v,
	     double *w,
             int  left,
             int  right )
{
   int i, last;

   if (left >= right)
      return;
   swap2( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (w[i] < w[left])
      {
         swap2(v, w, ++last, i);
      }
   swap2(v, w, left, last);
   hypre_qsort2(v, w, left, last-1);
   hypre_qsort2(v, w, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* sort on v, move w and z (AB 11/04) */

void hypre_qsort3i( int *v,
                    int *w,
                    int *z,
                    int  left,
                    int  right )
{
   int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_swap3i( v, w, z, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_swap3i(v, w, z, ++last, i);
      }
   }
   hypre_swap3i(v, w, z, left, last);
   hypre_qsort3i(v, w, z, left, last-1);
   hypre_qsort3i(v, w, z, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigSwapbi(HYPRE_BigInt  *v,
                  int  *w,
                  int  i,
                  int  j )
{
   HYPRE_BigInt big_temp;
   int temp;

   big_temp = v[i];
   v[i] = v[j];
   v[j] = big_temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigQsortbi( HYPRE_BigInt *v,
                    int *w,
                    int  left,
                    int  right )
{
   int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_BigSwapbi( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_BigSwapbi(v, w, ++last, i);
      }
   }
   hypre_BigSwapbi(v, w, left, last);
   hypre_BigQsortbi(v, w, left, last-1);
   hypre_BigQsortbi(v, w, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigSwapLoc(HYPRE_BigInt  *v,
                  int  *w,
                  int  i,
                  int  j )
{
   HYPRE_BigInt big_temp;

   big_temp = v[i];
   v[i] = v[j];
   v[j] = big_temp;
   w[i] = j;
   w[j] = i;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigQsortbLoc( HYPRE_BigInt *v,
                    int *w,
                    int  left,
                    int  right )
{
   int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_BigSwapLoc( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_BigSwapLoc(v, w, ++last, i);
      }
   }
   hypre_BigSwapLoc(v, w, left, last);
   hypre_BigQsortbLoc(v, w, left, last-1);
   hypre_BigQsortbLoc(v, w, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/


void hypre_BigSwapb2i(HYPRE_BigInt  *v,
                     int  *w,
                     int  *z,
                     int  i,
                     int  j )
{
   HYPRE_BigInt big_temp;
   int temp;

   big_temp = v[i];
   v[i] = v[j];
   v[j] = big_temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
   temp = z[i];
   z[i] = z[j];
   z[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigQsortb2i( HYPRE_BigInt *v,
                       int *w,
                       int *z,
                       int  left,
                       int  right )
{
   int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_BigSwapb2i( v, w, z, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_BigSwapb2i(v, w, z, ++last, i);
      }
   }
   hypre_BigSwapb2i(v, w, z, left, last);
   hypre_BigQsortb2i(v, w, z, left, last-1);
   hypre_BigQsortb2i(v, w, z, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
void hypre_BigSwap( HYPRE_BigInt *v,
           int  i,
           int  j )
{
   HYPRE_BigInt temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigQsort0( HYPRE_BigInt *v,
             int  left,
             int  right )
{
   int i, last;

   if (left >= right)
      return;
   hypre_BigSwap( v, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (v[i] < v[left])
      {
         hypre_BigSwap(v, ++last, i);
      }
   hypre_BigSwap(v, left, last);
   hypre_BigQsort0(v, left, last-1);
   hypre_BigQsort0(v, last+1, right);
}

