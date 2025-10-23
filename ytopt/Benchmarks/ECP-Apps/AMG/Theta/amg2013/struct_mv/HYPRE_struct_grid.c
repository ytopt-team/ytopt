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
 * HYPRE_StructGrid interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructGridCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_StructGridCreate( MPI_Comm          comm,
                        int               dim,
                        HYPRE_StructGrid *grid )
{
   int ierr;

   ierr = hypre_StructGridCreate(comm, dim, grid);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridDestroy
 *--------------------------------------------------------------------------*/

int
HYPRE_StructGridDestroy( HYPRE_StructGrid grid )
{
   return ( hypre_StructGridDestroy(grid) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridSetExtents
 *--------------------------------------------------------------------------*/

int
HYPRE_StructGridSetExtents( HYPRE_StructGrid  grid,
                            int              *ilower,
                            int              *iupper )
{
   hypre_Index  new_ilower;
   hypre_Index  new_iupper;

   int          d;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim((hypre_StructGrid *) grid); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }

   return ( hypre_StructGridSetExtents(grid, new_ilower, new_iupper) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructGridPeriodicity
 *--------------------------------------------------------------------------*/

int
HYPRE_StructGridSetPeriodic( HYPRE_StructGrid  grid,
                             int              *periodic )
{
   hypre_Index  new_periodic;

   int          d;

   hypre_ClearIndex(new_periodic);
   for (d = 0; d < hypre_StructGridDim(grid); d++)
   {
      hypre_IndexD(new_periodic, d) = periodic[d];
   }

   return ( hypre_StructGridSetPeriodic(grid, new_periodic) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridAssemble
 *--------------------------------------------------------------------------*/

int
HYPRE_StructGridAssemble( HYPRE_StructGrid grid )
{
#ifdef HYPRE_NO_GLOBAL_PARTITION
   return ( hypre_StructGridAssembleWithAP(grid) );
#else
   return ( hypre_StructGridAssemble(grid) );
#endif
}

/*---------------------------------------------------------------------------
 * GEC0902
 * HYPRE_StructGridSetNumGhost
 * to set the numghost array inside the struct_grid_struct using an internal
 * function. This is just a wrapper.
 *--------------------------------------------------------------------------*/
int
HYPRE_StructGridSetNumGhost( HYPRE_StructGrid grid, int *num_ghost )
{
  return ( hypre_StructGridSetNumGhost(grid, num_ghost) );
}
