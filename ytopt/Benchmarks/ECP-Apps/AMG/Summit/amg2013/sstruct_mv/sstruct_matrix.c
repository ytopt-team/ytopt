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
 * Member functions for hypre_SStructPMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*==========================================================================
 * SStructPMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixRef
 *--------------------------------------------------------------------------*/

int
hypre_SStructPMatrixRef( hypre_SStructPMatrix  *matrix,
                         hypre_SStructPMatrix **matrix_ref )
{
   hypre_SStructPMatrixRefCount(matrix) ++;
   *matrix_ref = matrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixCreate
 *--------------------------------------------------------------------------*/

int
hypre_SStructPMatrixCreate( MPI_Comm               comm,
                            hypre_SStructPGrid    *pgrid,
                            hypre_SStructStencil **stencils,
                            hypre_SStructPMatrix **pmatrix_ptr )
{
   hypre_SStructPMatrix  *pmatrix;
   int                    nvars;
   int                  **smaps;
   hypre_StructStencil ***sstencils;
   hypre_StructMatrix  ***smatrices;
   int                  **symmetric;

   hypre_StructStencil   *sstencil;
   int                   *vars;
   hypre_Index           *sstencil_shape;
   int                    sstencil_size;
   int                    new_dim;
   int                   *new_sizes;
   hypre_Index          **new_shapes;
   int                    size;
   hypre_StructGrid      *sgrid;

   int                    vi, vj;
   int                    i, j, k;

   pmatrix = hypre_TAlloc(hypre_SStructPMatrix, 1);

   hypre_SStructPMatrixComm(pmatrix)     = comm;
   hypre_SStructPMatrixPGrid(pmatrix)    = pgrid;
   hypre_SStructPMatrixStencils(pmatrix) = stencils;
   nvars = hypre_SStructPGridNVars(pgrid);
   hypre_SStructPMatrixNVars(pmatrix) = nvars;

   /* create sstencils */
   smaps     = hypre_TAlloc(int *, nvars);
   sstencils = hypre_TAlloc(hypre_StructStencil **, nvars);
   new_sizes  = hypre_TAlloc(int, nvars);
   new_shapes = hypre_TAlloc(hypre_Index *, nvars);
   size = 0;
   for (vi = 0; vi < nvars; vi++)
   {
      sstencils[vi] = hypre_TAlloc(hypre_StructStencil *, nvars);
      for (vj = 0; vj < nvars; vj++)
      {
         sstencils[vi][vj] = NULL;
         new_sizes[vj] = 0;
      }

      sstencil       = hypre_SStructStencilSStencil(stencils[vi]);
      vars           = hypre_SStructStencilVars(stencils[vi]);
      sstencil_shape = hypre_StructStencilShape(sstencil);
      sstencil_size  = hypre_StructStencilSize(sstencil);

      smaps[vi] = hypre_TAlloc(int, sstencil_size);
      for (i = 0; i < sstencil_size; i++)
      {
         j = vars[i];
         new_sizes[j]++;
      }
      for (vj = 0; vj < nvars; vj++)
      {
         if (new_sizes[vj])
         {
            new_shapes[vj] = hypre_TAlloc(hypre_Index, new_sizes[vj]);
            new_sizes[vj] = 0;
         }
      }
      for (i = 0; i < sstencil_size; i++)
      {
         j = vars[i];
         k = new_sizes[j];
         hypre_CopyIndex(sstencil_shape[i], new_shapes[j][k]);
         smaps[vi][i] = k;
         new_sizes[j]++;
      }
      new_dim = hypre_StructStencilDim(sstencil);
      for (vj = 0; vj < nvars; vj++)
      {
         if (new_sizes[vj])
         {
            sstencils[vi][vj] = hypre_StructStencilCreate(new_dim,
                                                          new_sizes[vj],
                                                          new_shapes[vj]);
         }
         size = hypre_max(size, new_sizes[vj]);
      }
   }
   hypre_SStructPMatrixSMaps(pmatrix)     = smaps;
   hypre_SStructPMatrixSStencils(pmatrix) = sstencils;
   hypre_TFree(new_sizes);
   hypre_TFree(new_shapes);

   /* create smatrices */
   smatrices = hypre_TAlloc(hypre_StructMatrix **, nvars);
   for (vi = 0; vi < nvars; vi++)
   {
      smatrices[vi] = hypre_TAlloc(hypre_StructMatrix *, nvars);
      for (vj = 0; vj < nvars; vj++)
      {
         smatrices[vi][vj] = NULL;
         if (sstencils[vi][vj] != NULL)
         {
            sgrid = hypre_SStructPGridSGrid(pgrid, vi);
            smatrices[vi][vj] =
               hypre_StructMatrixCreate(comm, sgrid, sstencils[vi][vj]);
         }
      }
   }
   hypre_SStructPMatrixSMatrices(pmatrix) = smatrices;

   /* create symmetric */
   symmetric = hypre_TAlloc(int *, nvars);
   for (vi = 0; vi < nvars; vi++)
   {
      symmetric[vi] = hypre_TAlloc(int, nvars);
      for (vj = 0; vj < nvars; vj++)
      {
         symmetric[vi][vj] = 0;
      }
   }
   hypre_SStructPMatrixSymmetric(pmatrix) = symmetric;

   hypre_SStructPMatrixSEntriesSize(pmatrix) = size;
   hypre_SStructPMatrixSEntries(pmatrix) = hypre_TAlloc(int, size);

   hypre_SStructPMatrixRefCount(pmatrix)   = 1;

   *pmatrix_ptr = pmatrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPMatrixDestroy( hypre_SStructPMatrix *pmatrix )
{
   hypre_SStructStencil  **stencils;
   int                     nvars;
   int                   **smaps;
   hypre_StructStencil  ***sstencils;
   hypre_StructMatrix   ***smatrices;
   int                   **symmetric;
   int                     vi, vj;

   if (pmatrix)
   {
      hypre_SStructPMatrixRefCount(pmatrix) --;
      if (hypre_SStructPMatrixRefCount(pmatrix) == 0)
      {
         stencils  = hypre_SStructPMatrixStencils(pmatrix);
         nvars     = hypre_SStructPMatrixNVars(pmatrix);
         smaps     = hypre_SStructPMatrixSMaps(pmatrix);
         sstencils = hypre_SStructPMatrixSStencils(pmatrix);
         smatrices = hypre_SStructPMatrixSMatrices(pmatrix);
         symmetric = hypre_SStructPMatrixSymmetric(pmatrix);
         for (vi = 0; vi < nvars; vi++)
         {
            HYPRE_SStructStencilDestroy(stencils[vi]);
            hypre_TFree(smaps[vi]);
            for (vj = 0; vj < nvars; vj++)
            {
               hypre_StructStencilDestroy(sstencils[vi][vj]);
               hypre_StructMatrixDestroy(smatrices[vi][vj]);
            }
            hypre_TFree(sstencils[vi]);
            hypre_TFree(smatrices[vi]);
            hypre_TFree(symmetric[vi]);
         }
         hypre_TFree(stencils);
         hypre_TFree(smaps);
         hypre_TFree(sstencils);
         hypre_TFree(smatrices);
         hypre_TFree(symmetric);
         hypre_TFree(hypre_SStructPMatrixSEntries(pmatrix));
         hypre_TFree(pmatrix);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixInitialize
 *--------------------------------------------------------------------------*/
int 
hypre_SStructPMatrixInitialize( hypre_SStructPMatrix *pmatrix )
{
   int                    nvars       = hypre_SStructPMatrixNVars(pmatrix);
   int                  **symmetric   = hypre_SStructPMatrixSymmetric(pmatrix);
   hypre_SStructPGrid    *pgrid       = hypre_SStructPMatrixPGrid(pmatrix);
   HYPRE_SStructVariable *vartypes    = hypre_SStructPGridVarTypes(pgrid);
   int                    ndim        = hypre_SStructPGridNDim(pgrid);

   int                    num_ghost[6]= {1, 1, 1, 1, 1, 1};
   hypre_StructMatrix    *smatrix;
   hypre_StructGrid      *sgrid;

   hypre_Index            varoffset;
   int                    vi, vj, d;

   for (vi = 0; vi < nvars; vi++)
   {
     /* use variable vi add_numghost */
      sgrid= hypre_SStructPGridSGrid(pgrid, vi);
      hypre_SStructVariableGetOffset(vartypes[vi], ndim, varoffset);

      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            HYPRE_StructMatrixSetSymmetric(smatrix, symmetric[vi][vj]);
            hypre_StructMatrixSetNumGhost(smatrix, num_ghost);

            for (d = 0; d < 3; d++)
            {
               hypre_StructMatrixAddNumGhost(smatrix)[2*d]=
                                       hypre_IndexD(varoffset, d);
               hypre_StructMatrixAddNumGhost(smatrix)[2*d+1]=
                                       hypre_IndexD(varoffset, d);
            }

            hypre_StructMatrixInitialize(smatrix);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixSetValues
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPMatrixSetValues( hypre_SStructPMatrix *pmatrix,
                               hypre_Index           index,
                               int                   var,
                               int                   nentries,
                               int                  *entries,
                               double               *values,
                               int                   add_to )
{
   hypre_SStructStencil *stencil = hypre_SStructPMatrixStencil(pmatrix, var);
   int                  *smap    = hypre_SStructPMatrixSMap(pmatrix, var);
   int                  *vars    = hypre_SStructStencilVars(stencil);
   hypre_StructMatrix   *smatrix;
   int                  *sentries;
   int                   i;

   smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entries[0]]);

   sentries = hypre_SStructPMatrixSEntries(pmatrix);
   for (i = 0; i < nentries; i++)
   {
      sentries[i] = smap[entries[i]];
   }

   hypre_StructMatrixSetValues(smatrix, index, nentries, sentries, values, add_to);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPMatrixSetBoxValues( hypre_SStructPMatrix *pmatrix,
                                  hypre_Index           ilower,
                                  hypre_Index           iupper,
                                  int                   var,
                                  int                   nentries,
                                  int                  *entries,
                                  double               *values,
                                  int                   add_to )
{
   hypre_SStructStencil *stencil = hypre_SStructPMatrixStencil(pmatrix, var);
   int                  *smap    = hypre_SStructPMatrixSMap(pmatrix, var);
   int                  *vars    = hypre_SStructStencilVars(stencil);
   hypre_StructMatrix   *smatrix;
   hypre_Box            *box;
   int                  *sentries;
   int                   i;

   smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entries[0]]);

   box = hypre_BoxCreate();
   hypre_CopyIndex(ilower, hypre_BoxIMin(box));
   hypre_CopyIndex(iupper, hypre_BoxIMax(box));

   sentries = hypre_SStructPMatrixSEntries(pmatrix);
   for (i = 0; i < nentries; i++)
   {
      sentries[i] = smap[entries[i]];
   }

   hypre_StructMatrixSetBoxValues(smatrix, box, nentries, sentries, values, add_to);

   hypre_BoxDestroy(box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPMatrixAssemble( hypre_SStructPMatrix *pmatrix )
{
   int                 nvars = hypre_SStructPMatrixNVars(pmatrix);
   hypre_StructMatrix *smatrix;
   int                 vi, vj;

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            hypre_StructMatrixAssemble(smatrix);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
int
hypre_SStructPMatrixSetSymmetric( hypre_SStructPMatrix *pmatrix,
                                  int                   var,
                                  int                   to_var,
                                  int                   symmetric )
{
   int **pmsymmetric = hypre_SStructPMatrixSymmetric(pmatrix);

   int vstart = var;
   int vsize  = 1;
   int tstart = to_var;
   int tsize  = 1;
   int v, t;

   if (var == -1)
   {
      vstart = 0;
      vsize  = hypre_SStructPMatrixNVars(pmatrix);
   }
   if (to_var == -1)
   {
      tstart = 0;
      tsize  = hypre_SStructPMatrixNVars(pmatrix);
   }

   for (v = vstart; v < vsize; v++)
   {
      for (t = tstart; t < tsize; t++)
      {
         pmsymmetric[v][t] = symmetric;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixPrint
 *--------------------------------------------------------------------------*/

int
hypre_SStructPMatrixPrint( const char           *filename,
                           hypre_SStructPMatrix *pmatrix,
                           int                   all )
{
   int                 nvars = hypre_SStructPMatrixNVars(pmatrix);
   hypre_StructMatrix *smatrix;
   int                 vi, vj;
   char                new_filename[255];

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            sprintf(new_filename, "%s.%02d.%02d", filename, vi, vj);
            hypre_StructMatrixPrint(new_filename, smatrix, all);
         }
      }
   }

   return hypre_error_flag;
}

/*==========================================================================
 * SStructUMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructUMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_SStructUMatrixInitialize( hypre_SStructMatrix *matrix )
{
   HYPRE_IJMatrix          ijmatrix   = hypre_SStructMatrixIJMatrix(matrix);
   hypre_SStructGraph     *graph      = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid      *grid       = hypre_SStructGraphGrid(graph);
   int                     nparts     = hypre_SStructGraphNParts(graph);
   hypre_SStructPGrid    **pgrids     = hypre_SStructGraphPGrids(graph);
   hypre_SStructStencil ***stencils   = hypre_SStructGraphStencils(graph);
   int                     nUventries = hypre_SStructGraphNUVEntries(graph);
   int                    *iUventries = hypre_SStructGraphIUVEntries(graph);
   hypre_SStructUVEntry  **Uventries  = hypre_SStructGraphUVEntries(graph);
   int                   **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   hypre_StructGrid       *sgrid;
   hypre_SStructStencil   *stencil;
   int                    *split;
   int                     nvars;
   int                     nrows, nnzs ;
   int                     part, var, entry, i, j, k,m,b;
   int                    *row_sizes;
   int                     max_row_size;

   int                    matrix_type = hypre_SStructMatrixObjectType(matrix);

   hypre_Box              *gridbox;
   hypre_Box              *loopbox;
   hypre_Box              *ghostbox;
   hypre_BoxArray         *boxes;
   int                    *num_ghost;


   HYPRE_IJMatrixSetObjectType(ijmatrix, HYPRE_PARCSR);

   /* GEC1002 the ghlocalsize is used to set the number of rows   */
 
   if (matrix_type == HYPRE_PARCSR)
   {
     nrows = hypre_SStructGridLocalSize(grid);
   }
   if (matrix_type == HYPRE_SSTRUCT || matrix_type == HYPRE_STRUCT)
   {
     nrows = hypre_SStructGridGhlocalSize(grid) ;
   }

   /* set row sizes */
   m = 0;
   row_sizes = hypre_CTAlloc(int, nrows);
   max_row_size = 0;
   for (part = 0; part < nparts; part++)
   {
      nvars = hypre_SStructPGridNVars(pgrids[part]);
      for (var = 0; var < nvars; var++)
      {
         sgrid   = hypre_SStructPGridSGrid(pgrids[part], var);
              
         stencil = stencils[part][var];
         split   = hypre_SStructMatrixSplit(matrix, part, var);
         nnzs = 0;
         for (entry = 0; entry < hypre_SStructStencilSize(stencil); entry++)
         {
            if (split[entry] == -1)
            {
               nnzs++;
            }
         }
#if 0
         /* TODO: For now, assume stencil is full/complete */
         if (hypre_SStructMatrixSymmetric(matrix))
         {
            nnzs = 2*nnzs - 1;
         }
#endif

	 /**************/

         boxes = hypre_StructGridBoxes(sgrid) ;
         num_ghost = hypre_StructGridNumGhost(sgrid);
         for (b = 0; b < hypre_BoxArraySize(boxes); b++)
	 {
            gridbox = hypre_BoxArrayBox(boxes, b);
            ghostbox = hypre_BoxCreate();
            loopbox  = hypre_BoxCreate();
            hypre_CopyBox(gridbox,ghostbox);
	    hypre_BoxExpand(ghostbox,num_ghost);

            if (matrix_type == HYPRE_SSTRUCT || matrix_type == HYPRE_STRUCT)
	    {
               hypre_CopyBox(ghostbox,loopbox);
            }
            if (matrix_type == HYPRE_PARCSR)
	    {
	       hypre_CopyBox(gridbox,loopbox);
            }

            for (k = hypre_BoxIMinZ(loopbox); k <= hypre_BoxIMaxZ(loopbox); k++)
            {
              for (j = hypre_BoxIMinY(loopbox); j <= hypre_BoxIMaxY(loopbox); j++)
              {
                for (i = hypre_BoxIMinX(loopbox); i <= hypre_BoxIMaxX(loopbox); i++)
                {
		    if (   ( ( i>=hypre_BoxIMinX(gridbox) )
		        &&   ( j>=hypre_BoxIMinY(gridbox) ) )
		        &&   ( k>=hypre_BoxIMinZ(gridbox) ) )
		    {
                      if (  ( ( i<=hypre_BoxIMaxX(gridbox) )
                           && ( j<=hypre_BoxIMaxY(gridbox) ) )
                           && ( k<=hypre_BoxIMaxZ(gridbox) ) )
                      {
                          row_sizes[m] = nnzs;
                          max_row_size = hypre_max(max_row_size, row_sizes[m]);
                      }
                    }
                   m++;  
                }
              }
            }
            hypre_BoxDestroy(ghostbox); 
            hypre_BoxDestroy(loopbox);
         }


         if (nvneighbors[part][var])
         {
            max_row_size = hypre_max(max_row_size,
                                     hypre_SStructStencilSize(stencil));
         }


        /*********************/
      }
   }

   /* GEC0902 essentially for each UVentry we figure out how many extra columns
    * we need to add to the rowsizes                                   */

   for (entry = 0; entry < nUventries; entry++)
   {
         i = iUventries[entry];
         row_sizes[i] += hypre_SStructUVEntryNUEntries(Uventries[i]);
         max_row_size = hypre_max(max_row_size, row_sizes[i]);
   }

   /* ZTODO: Update row_sizes based on neighbor off-part couplings */
   HYPRE_IJMatrixSetRowSizes (ijmatrix, (const int *) row_sizes);

   hypre_TFree(row_sizes);
   hypre_SStructMatrixTmpColCoords(matrix) =
      hypre_CTAlloc(HYPRE_BigInt, max_row_size);
   hypre_SStructMatrixTmpCoeffs(matrix) =
      hypre_CTAlloc(double, max_row_size);

   /* GEC1002 at this point the processor has the partitioning (creation of ij) */

   HYPRE_IJMatrixInitialize(ijmatrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructUMatrixSetValues
 *
 * (add_to > 0): add-to values
 * (add_to = 0): set values
 * (add_to < 0): get values
 *--------------------------------------------------------------------------*/

int 
hypre_SStructUMatrixSetValues( hypre_SStructMatrix *matrix,
                               int                  part,
                               hypre_Index          index,
                               int                  var,
                               int                  nentries,
                               int                 *entries,
                               double              *values,
                               int                  add_to )
{
   HYPRE_IJMatrix        ijmatrix = hypre_SStructMatrixIJMatrix(matrix);
   hypre_SStructGraph   *graph   = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid    *grid    = hypre_SStructGraphGrid(graph);
   hypre_SStructStencil *stencil = hypre_SStructGraphStencil(graph, part, var);
   int                  *vars    = hypre_SStructStencilVars(stencil);
   hypre_Index          *shape   = hypre_SStructStencilShape(stencil);
   int                   size    = hypre_SStructStencilSize(stencil);
   hypre_IndexRef        offset;
   hypre_Index           to_index;
   hypre_SStructUVEntry *Uventry;
   hypre_BoxMapEntry    *map_entry;
   hypre_SStructMapInfo *entry_info;
   HYPRE_BigInt          row_coord;
   HYPRE_BigInt         *col_coords;
   int                   ncoeffs;
   double               *coeffs;
   int                   i, entry;
   int                   proc, myproc;
   /* GEC1002 the matrix type */
   int                   matrix_type = hypre_SStructMatrixObjectType(matrix);

   hypre_SStructGridFindMapEntry(grid, part, index, var, &map_entry);
   if (map_entry == NULL)
   {
      hypre_error_in_arg(1);
      hypre_error_in_arg(2);
      hypre_error_in_arg(3);
      /* RDF: This printing shouldn't be on by default */
      printf("Warning: Attempt to set coeffs for point not in grid\n");
      printf("hypre_SStructUMatrixSetValues call aborted for grid point\n");
      printf("    part=%d, var=%d, index=(%d, %d, %d)\n", part, var,
             hypre_IndexD(index,0),
             hypre_IndexD(index,1),
             hypre_IndexD(index,2) );
      return hypre_error_flag;
   }
   else
   {
      hypre_BoxMapEntryGetInfo(map_entry, (void **) &entry_info);
   }

   /* Only Set values if I am the owner process; off-process AddTo and Get
    * values are done by IJ */
   if (!add_to)
   {
      hypre_SStructMapEntryGetProcess(map_entry, &proc);
      MPI_Comm_rank(hypre_SStructGridComm(grid), &myproc);
      if (proc != myproc)
      {
         return hypre_error_flag;
      }
   }

   /* GEC1002 get the rank using the function with the type=matrixtype*/
   hypre_SStructMapEntryGetGlobalRank(map_entry, index, &row_coord, matrix_type);

  
   col_coords = hypre_SStructMatrixTmpColCoords(matrix);
   coeffs     = hypre_SStructMatrixTmpCoeffs(matrix);
   ncoeffs = 0;
   for (i = 0; i < nentries; i++)
   {
      entry = entries[i];

      if (entry < size)
      {
         /* stencil entries */
         offset = shape[entry];
         hypre_IndexX(to_index) = hypre_IndexX(index) + hypre_IndexX(offset);
         hypre_IndexY(to_index) = hypre_IndexY(index) + hypre_IndexY(offset);
         hypre_IndexZ(to_index) = hypre_IndexZ(index) + hypre_IndexZ(offset);
         
         hypre_SStructGridFindMapEntry(grid, part, to_index, vars[entry],
                                       &map_entry);
         
         if (map_entry != NULL)
        {

	    
	     hypre_SStructMapEntryGetGlobalRank(map_entry, to_index,
                                              &col_coords[ncoeffs],matrix_type);
	    

           coeffs[ncoeffs] = values[i];
           ncoeffs++;
        }
      }
      else
      {
         /* non-stencil entries */
         entry -= size;
         hypre_SStructGraphFindUVEntry(graph, part, index, var, &Uventry);
        
	 col_coords[ncoeffs] = hypre_SStructUVEntryRank(Uventry, entry);   
         coeffs[ncoeffs] = values[i];
         ncoeffs++;
      }
   }

   if (add_to > 0)
   {
      HYPRE_IJMatrixAddToValues(ijmatrix, 1, &ncoeffs, &row_coord,
                                (const HYPRE_BigInt *) col_coords,
                                (const double *) coeffs);
   }
   else if (add_to > -1)
   {
      HYPRE_IJMatrixSetValues(ijmatrix, 1, &ncoeffs, &row_coord,
                              (const HYPRE_BigInt *) col_coords,
                              (const double *) coeffs);
   }
   else
   {
      HYPRE_IJMatrixGetValues(ijmatrix, 1, &ncoeffs, &row_coord,
                              col_coords, values);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Note: Entries must all be of type stencil or non-stencil, but not both.
 *
 * (add_to > 0): add-to values
 * (add_to = 0): set values
 * (add_to < 0): get values
 *--------------------------------------------------------------------------*/

int 
hypre_SStructUMatrixSetBoxValues( hypre_SStructMatrix *matrix,
                                  int                  part,
                                  hypre_Index          ilower,
                                  hypre_Index          iupper,
                                  int                  var,
                                  int                  nentries,
                                  int                 *entries,
                                  double              *values,
                                  int                  add_to )
{
   HYPRE_IJMatrix        ijmatrix = hypre_SStructMatrixIJMatrix(matrix);
   hypre_SStructGraph   *graph   = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid    *grid    = hypre_SStructGraphGrid(graph);
   hypre_SStructStencil *stencil = hypre_SStructGraphStencil(graph, part, var);
   int                  *vars    = hypre_SStructStencilVars(stencil);
   hypre_Index          *shape   = hypre_SStructStencilShape(stencil);
   int                   size    = hypre_SStructStencilSize(stencil);
   hypre_IndexRef        offset;
   hypre_BoxMap         *map;
   hypre_BoxMapEntry   **map_entries;
   int                   nmap_entries;
   hypre_BoxMapEntry   **map_to_entries;
   int                   nmap_to_entries;
   int                   nrows;
   int                  *ncols;
   HYPRE_BigInt         *rows;
   HYPRE_BigInt         *cols;
   double               *ijvalues;
   hypre_Box            *box;
   hypre_Box            *to_box;
   hypre_Box            *map_box;
   hypre_Box            *int_box;
   hypre_Index           index;
   hypre_Index           rs, cs;
   int                   sy, sz;
   HYPRE_BigInt          row_base, col_base;
   int                   val_base;
   int                   e, entry, ii, jj, i, j, k;
   int                   proc, myproc;
  /* GEC1002 the matrix type */
   int                   matrix_type = hypre_SStructMatrixObjectType(matrix);

   box = hypre_BoxCreate();

   /*------------------------------------------
    * all stencil entries
    *------------------------------------------*/

   if (entries[0] < size)
   {
      to_box  = hypre_BoxCreate();
      map_box = hypre_BoxCreate();
      int_box = hypre_BoxCreate();

      hypre_CopyIndex(ilower, hypre_BoxIMin(box));
      hypre_CopyIndex(iupper, hypre_BoxIMax(box));
      /* ZTODO: check that this change fixes multiple-entry problem */
      nrows    = hypre_BoxVolume(box)*nentries;
      ncols    = hypre_CTAlloc(int, nrows);
      for (i = 0; i < nrows; i++)
      {
         ncols[i] = 1;
      }
      rows     = hypre_CTAlloc(HYPRE_BigInt, nrows);
      cols     = hypre_CTAlloc(HYPRE_BigInt, nrows);
      ijvalues = hypre_CTAlloc(double, nrows);

      sy = (hypre_IndexX(iupper) - hypre_IndexX(ilower) + 1);
      sz = (hypre_IndexY(iupper) - hypre_IndexY(ilower) + 1) * sy;

      map = hypre_SStructGridMap(grid, part, var);
      hypre_BoxMapIntersect(map, ilower, iupper, &map_entries, &nmap_entries);
         
      for (ii = 0; ii < nmap_entries; ii++)
      {
         /* Only Set values if I am the owner process; off-process AddTo and Get
          * values are done by IJ */
         if (!add_to)
         {
            hypre_SStructMapEntryGetProcess(map_entries[ii], &proc);
            MPI_Comm_rank(hypre_SStructGridComm(grid), &myproc);
            if (proc != myproc)
            {
               continue;
            }
         }

         /* GEC1002 introducing the strides based on the type of the matrix  */
         hypre_SStructMapEntryGetStrides(map_entries[ii], rs, matrix_type);

         hypre_CopyIndex(ilower, hypre_BoxIMin(box));
         hypre_CopyIndex(iupper, hypre_BoxIMax(box));
         hypre_BoxMapEntryGetExtents(map_entries[ii],
                                     hypre_BoxIMin(map_box),
                                     hypre_BoxIMax(map_box));
         hypre_IntersectBoxes(box, map_box, int_box);
         hypre_CopyBox(int_box, box);

         nrows = 0;
         for (e = 0; e < nentries; e++)
         {
            entry = entries[e];

            hypre_CopyBox(box, to_box);

            offset = shape[entry];
            hypre_BoxIMinX(to_box) += hypre_IndexX(offset);
            hypre_BoxIMinY(to_box) += hypre_IndexY(offset);
            hypre_BoxIMinZ(to_box) += hypre_IndexZ(offset);
            hypre_BoxIMaxX(to_box) += hypre_IndexX(offset);
            hypre_BoxIMaxY(to_box) += hypre_IndexY(offset);
            hypre_BoxIMaxZ(to_box) += hypre_IndexZ(offset);

            map = hypre_SStructGridMap(grid, part, vars[entry]);
            hypre_BoxMapIntersect(map, hypre_BoxIMin(to_box),
                                  hypre_BoxIMax(to_box),
                                  &map_to_entries, &nmap_to_entries );
         
            for (jj = 0; jj < nmap_to_entries; jj++)
            {

             /* GEC1002 introducing the strides based on the type of the matrix  */
  
               hypre_SStructMapEntryGetStrides(map_to_entries[jj], cs, matrix_type);

               hypre_BoxMapEntryGetExtents(map_to_entries[jj],
                                           hypre_BoxIMin(map_box),
                                           hypre_BoxIMax(map_box));
               hypre_IntersectBoxes(to_box, map_box, int_box);

               hypre_CopyIndex(hypre_BoxIMin(int_box), index);

                /* GEC1002 introducing the rank based on the type of the matrix  */

               hypre_SStructMapEntryGetGlobalRank(map_to_entries[jj],
                                                  index, &col_base,matrix_type);

               hypre_IndexX(index) -= hypre_IndexX(offset);
               hypre_IndexY(index) -= hypre_IndexY(offset);
               hypre_IndexZ(index) -= hypre_IndexZ(offset);

                /* GEC1002 introducing the rank based on the type of the matrix  */

               hypre_SStructMapEntryGetGlobalRank(map_entries[ii],
                                                  index, &row_base,matrix_type);

               hypre_IndexX(index) -= hypre_IndexX(ilower);
               hypre_IndexY(index) -= hypre_IndexY(ilower);
               hypre_IndexZ(index) -= hypre_IndexZ(ilower);
               val_base = e + (hypre_IndexX(index) +
                               hypre_IndexY(index)*sy +
                               hypre_IndexZ(index)*sz) * nentries;

               for (k = 0; k < hypre_BoxSizeZ(int_box); k++)
               {
                  for (j = 0; j < hypre_BoxSizeY(int_box); j++)
                  {
                     for (i = 0; i < hypre_BoxSizeX(int_box); i++)
                     {
                        rows[nrows] = row_base + (HYPRE_BigInt)(i*rs[0] + j*rs[1] + k*rs[2]);
                        cols[nrows] = col_base + (HYPRE_BigInt)(i*cs[0] + j*cs[1] + k*cs[2]);
                        ijvalues[nrows] =
                           values[val_base + (i + j*sy + k*sz)*nentries];
                        nrows++;
                     }
                  }
               }
            }

            hypre_TFree(map_to_entries);
         }

         /*------------------------------------------
          * set IJ values one stencil entry at a time
          *------------------------------------------*/

         if (add_to > 0)
         {
            HYPRE_IJMatrixAddToValues(ijmatrix, nrows, ncols,
                                      (const HYPRE_BigInt *) rows,
                                      (const HYPRE_BigInt *) cols,
                                      (const double *) ijvalues);
         }
         else if (add_to > -1)
         {
            HYPRE_IJMatrixSetValues(ijmatrix, nrows, ncols,
                                    (const HYPRE_BigInt *) rows,
                                    (const HYPRE_BigInt *) cols,
                                    (const double *) ijvalues);
         }
         else
         {
            HYPRE_IJMatrixGetValues(ijmatrix, nrows, ncols, rows, cols, values);
         }
      }

      hypre_TFree(map_entries);

      hypre_TFree(ncols);
      hypre_TFree(rows);
      hypre_TFree(cols);
      hypre_TFree(ijvalues);

      hypre_BoxDestroy(to_box);
      hypre_BoxDestroy(map_box);
      hypre_BoxDestroy(int_box);
    }

   /*------------------------------------------
    * non-stencil entries
    *------------------------------------------*/

   else
   {
      hypre_CopyIndex(ilower, hypre_BoxIMin(box));
      hypre_CopyIndex(iupper, hypre_BoxIMax(box));

      for (k = hypre_BoxIMinZ(box); k <= hypre_BoxIMaxZ(box); k++)
      {
         for (j = hypre_BoxIMinY(box); j <= hypre_BoxIMaxY(box); j++)
         {
            for (i = hypre_BoxIMinX(box); i <= hypre_BoxIMaxX(box); i++)
            {
               hypre_SetIndex(index, i, j, k);
               hypre_SStructUMatrixSetValues(matrix, part, index, var,
                                             nentries, entries, values, add_to);
               values += nentries;
            }
         }
      }
   }

   hypre_BoxDestroy(box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructUMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_SStructUMatrixAssemble( hypre_SStructMatrix *matrix )
{
   HYPRE_IJMatrix ijmatrix = hypre_SStructMatrixIJMatrix(matrix);

   HYPRE_IJMatrixAssemble(ijmatrix);
   HYPRE_IJMatrixGetObject(ijmatrix,
                           (void **) &hypre_SStructMatrixParCSRMatrix(matrix));

   return hypre_error_flag;
}

/*==========================================================================
 * SStructMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixRef
 *--------------------------------------------------------------------------*/

int
hypre_SStructMatrixRef( hypre_SStructMatrix  *matrix,
                        hypre_SStructMatrix **matrix_ref )
{
   hypre_SStructMatrixRefCount(matrix) ++;
   *matrix_ref = matrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixSplitEntries
 *--------------------------------------------------------------------------*/

int
hypre_SStructMatrixSplitEntries( hypre_SStructMatrix *matrix,
                                 int                  part,
                                 int                  var,
                                 int                  nentries,
                                 int                 *entries,
                                 int                 *nSentries_ptr,
                                 int                **Sentries_ptr,
                                 int                 *nUentries_ptr,
                                 int                **Uentries_ptr )
{
   hypre_SStructGraph   *graph   = hypre_SStructMatrixGraph(matrix);
   int                  *split   = hypre_SStructMatrixSplit(matrix, part, var);
   hypre_SStructStencil *stencil = hypre_SStructGraphStencil(graph, part, var);
   int                   entry;
   int                   i;

   int                   nSentries = 0;
   int                  *Sentries  = hypre_SStructMatrixSEntries(matrix);
   int                   nUentries = 0;
   int                  *Uentries  = hypre_SStructMatrixUEntries(matrix);

   for (i = 0; i < nentries; i++)
   {
      entry = entries[i];
      if (entry < hypre_SStructStencilSize(stencil))
      {
         /* stencil entries */
         if (split[entry] > -1)
         {
            Sentries[nSentries] = split[entry];
            nSentries++;
         }
         else
         {
            Uentries[nUentries] = entry;
            nUentries++;
         }
      }
      else
      {
         /* non-stencil entries */
         Uentries[nUentries] = entry;
         nUentries++;
      }
   }

   *nSentries_ptr = nSentries;
   *Sentries_ptr  = Sentries;
   *nUentries_ptr = nUentries;
   *Uentries_ptr  = Uentries;

   return hypre_error_flag;
}

