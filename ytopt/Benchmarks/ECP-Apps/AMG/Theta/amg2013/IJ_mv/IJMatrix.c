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
 * hypre_IJMatrix interface
 *
 *****************************************************************************/

#include "./IJ_mv.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * hypre_IJMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the row partitioning 

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

int
hypre_IJMatrixGetRowPartitioning( HYPRE_IJMatrix matrix ,
				  HYPRE_BigInt    **row_partitioning )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- hypre_IJMatrixGetRowPartitioning\n");
      exit(1);
   }

   if ( hypre_IJMatrixRowPartitioning(ijmatrix))
      *row_partitioning = hypre_IJMatrixRowPartitioning(ijmatrix);
   else
      return -1;

   return -99;
}
/*--------------------------------------------------------------------------
 * hypre_IJMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the column partitioning

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

int
hypre_IJMatrixGetColPartitioning( HYPRE_IJMatrix matrix ,
				  HYPRE_BigInt    **col_partitioning )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- hypre_IJMatrixGetColPartitioning\n");
      exit(1);
   }

   if ( hypre_IJMatrixColPartitioning(ijmatrix))
      *col_partitioning = hypre_IJMatrixColPartitioning(ijmatrix);
   else
      return -1;

   return -99;
}
/*--------------------------------------------------------------------------
 * hypre_IJMatrixSetObject
 *--------------------------------------------------------------------------*/

int 
hypre_IJMatrixSetObject( HYPRE_IJMatrix  matrix, 
                         void           *object )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (hypre_IJMatrixObject(ijmatrix) != NULL)
   {
      printf("Referencing a new IJMatrix object can orphan an old -- ");
      printf("hypre_IJMatrixSetObject\n");
      exit(1);
   }

   hypre_IJMatrixObject(ijmatrix) = object;

   return 0;
}
