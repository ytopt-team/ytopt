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
 * hypre_GeneratePartitioning:
 * generates load balanced partitioning of a 1-d array
 *--------------------------------------------------------------------------*/
/* for multivectors, length should be the (global) length of a single vector.
 Thus each of the vectors of the multivector will get the same data distribution. */

int
hypre_GeneratePartitioning(HYPRE_BigInt length, int num_procs, HYPRE_BigInt **part_ptr)
{
   int ierr = 0;
   HYPRE_BigInt *part;
   HYPRE_BigInt size;
   int i, rest;


   part = hypre_CTAlloc(HYPRE_BigInt, num_procs+1);
   size = length / (HYPRE_BigInt)num_procs;
   rest = (int)(length - size*(HYPRE_BigInt)num_procs);
   part[0] = 0;
   for (i=0; i < num_procs; i++)
   {
	part[i+1] = part[i]+size;
	if (i < rest) part[i+1]++;
   }


   *part_ptr = part;
   return ierr;
}


/* This function differs from the above in that it only returns
   the portion of the partition belonging to the individual process - 
   to do this it requires the processor id as well AHB 6/05*/

int
hypre_GenerateLocalPartitioning(HYPRE_BigInt length, int num_procs, int myid, HYPRE_BigInt **part_ptr)
{


   int ierr = 0;
   HYPRE_BigInt *part;
   HYPRE_BigInt size;
   int rest;

   part = hypre_CTAlloc(HYPRE_BigInt, 2);
   size = length /(HYPRE_BigInt)num_procs;
   rest = (int)(length - size*(HYPRE_BigInt)num_procs);

   /* first row I own */
   part[0] = size*(HYPRE_BigInt)myid;
   part[0] += (HYPRE_BigInt)hypre_min(myid, rest);
   
   /* last row I own */
   part[1] =  size*(HYPRE_BigInt)(myid+1);
   part[1] += (HYPRE_BigInt)hypre_min(myid+1, rest);
   part[1] = part[1] - 1;

   /* add 1 to last row since this is for "starts" vector */
   part[1] = part[1] + 1;
   

   *part_ptr = part;
   return ierr;
}
