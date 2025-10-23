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
 * HYPRE_SStructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixCreate( MPI_Comm              comm,
                           HYPRE_SStructGraph    graph,
                           HYPRE_SStructMatrix  *matrix_ptr )
{
   int ierr = 0;

   /* GEC1202 grid not needed  */
   /*  hypre_SStructGrid      *grid     = hypre_SStructGraphGrid(graph); */
   hypre_SStructStencil ***stencils = hypre_SStructGraphStencils(graph);

   hypre_SStructMatrix    *matrix;
   int                  ***splits;
   int                     nparts;
   hypre_SStructPMatrix  **pmatrices;
   int                  ***symmetric;

   hypre_SStructPGrid     *pgrid;
   int                     nvars;

   int                     stencil_size;
   int                    *stencil_vars;
   int                     pstencil_size;

   HYPRE_SStructVariable   vitype, vjtype;
   int                     part, vi, vj, i;
   int                     size;

   matrix = hypre_TAlloc(hypre_SStructMatrix, 1);

   hypre_SStructMatrixComm(matrix)  = comm;
   hypre_SStructMatrixNDim(matrix)  = hypre_SStructGraphNDim(graph);
   hypre_SStructGraphRef(graph, &hypre_SStructMatrixGraph(matrix));

   /* compute S/U-matrix split */
   nparts = hypre_SStructGraphNParts(graph);
   hypre_SStructMatrixNParts(matrix) = nparts;
   splits = hypre_TAlloc(int **, nparts);
   hypre_SStructMatrixSplits(matrix) = splits;
   pmatrices = hypre_TAlloc(hypre_SStructPMatrix *, nparts);
   hypre_SStructMatrixPMatrices(matrix) = pmatrices;
   symmetric = hypre_TAlloc(int **, nparts);
   hypre_SStructMatrixSymmetric(matrix) = symmetric;
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGraphPGrid(graph, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      splits[part] = hypre_TAlloc(int *, nvars);
      symmetric[part] = hypre_TAlloc(int *, nvars);
      for (vi = 0; vi < nvars; vi++)
      {
         stencil_size  = hypre_SStructStencilSize(stencils[part][vi]);
         stencil_vars  = hypre_SStructStencilVars(stencils[part][vi]);
         pstencil_size = 0;
         splits[part][vi] = hypre_TAlloc(int, stencil_size);
         symmetric[part][vi] = hypre_TAlloc(int, nvars);
         for (i = 0; i < stencil_size; i++)
         {
            vj = stencil_vars[i];
            vitype = hypre_SStructPGridVarType(pgrid, vi);
            vjtype = hypre_SStructPGridVarType(pgrid, vj);
            if (vjtype == vitype)
            {
               splits[part][vi][i] = pstencil_size;
               pstencil_size++;
            }
            else
            {
               splits[part][vi][i] = -1;
            }
         }
         for (vj = 0; vj < nvars; vj++)
         {
            symmetric[part][vi][vj] = 0;
         }
      }
   }

   /* GEC0902 move the IJ creation to the initialization phase  
    * ilower = hypre_SStructGridGhstartRank(grid);
    * iupper = ilower + hypre_SStructGridGhlocalSize(grid) - 1; 
    * HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper,
    *                    &hypre_SStructMatrixIJMatrix(matrix)); */
   
   hypre_SStructMatrixIJMatrix(matrix)     = NULL;
   hypre_SStructMatrixParCSRMatrix(matrix) = NULL;

   size = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGraphPGrid(graph, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      for (vi = 0; vi < nvars; vi++)
      {
         size = hypre_max(size, hypre_SStructStencilSize(stencils[part][vi]));
      }
   }
   hypre_SStructMatrixEntriesSize(matrix) = size;
   hypre_SStructMatrixSEntries(matrix) = hypre_TAlloc(int, size);
   hypre_SStructMatrixUEntries(matrix) = hypre_TAlloc(int, size);
   hypre_SStructMatrixTmpColCoords(matrix) = NULL;
   hypre_SStructMatrixTmpCoeffs(matrix)    = NULL;

   hypre_SStructMatrixNSSymmetric(matrix) = 0;
   hypre_SStructMatrixGlobalSize(matrix)  = 0;
   hypre_SStructMatrixRefCount(matrix)    = 1;
  
   /* GEC0902 setting the default of the object_type to HYPRE_SSTRUCT */ 

   hypre_SStructMatrixObjectType(matrix) = HYPRE_SSTRUCT;

   *matrix_ptr = matrix;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructMatrixDestroy( HYPRE_SStructMatrix matrix )
{
   int ierr = 0;

   hypre_SStructGraph     *graph;
   int                  ***splits;
   int                     nparts;
   hypre_SStructPMatrix  **pmatrices;
   int                  ***symmetric;
   hypre_SStructPGrid     *pgrid;
   int                     nvars;
   int                     part, var;

   if (matrix)
   {
      hypre_SStructMatrixRefCount(matrix) --;
      if (hypre_SStructMatrixRefCount(matrix) == 0)
      {
         graph        = hypre_SStructMatrixGraph(matrix);
         splits       = hypre_SStructMatrixSplits(matrix);
         nparts       = hypre_SStructMatrixNParts(matrix);
         pmatrices    = hypre_SStructMatrixPMatrices(matrix);
         symmetric    = hypre_SStructMatrixSymmetric(matrix);
         for (part = 0; part < nparts; part++)
         {
            pgrid = hypre_SStructGraphPGrid(graph, part);
            nvars = hypre_SStructPGridNVars(pgrid);
            for (var = 0; var < nvars; var++)
            {
               hypre_TFree(splits[part][var]);
               hypre_TFree(symmetric[part][var]);
            }
            hypre_TFree(splits[part]);
            hypre_TFree(symmetric[part]);
            hypre_SStructPMatrixDestroy(pmatrices[part]);
         }
         HYPRE_SStructGraphDestroy(graph);
         hypre_TFree(splits);
         hypre_TFree(pmatrices);
         hypre_TFree(symmetric);
         HYPRE_IJMatrixDestroy(hypre_SStructMatrixIJMatrix(matrix));
         hypre_TFree(hypre_SStructMatrixSEntries(matrix));
         hypre_TFree(hypre_SStructMatrixUEntries(matrix));
         hypre_TFree(hypre_SStructMatrixTmpColCoords(matrix));
         hypre_TFree(hypre_SStructMatrixTmpCoeffs(matrix));
         hypre_TFree(matrix);
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixInitialize( HYPRE_SStructMatrix matrix )
{
   int ierr = 0;

   int                     nparts    = hypre_SStructMatrixNParts(matrix);
   hypre_SStructGraph     *graph     = hypre_SStructMatrixGraph(matrix);
   hypre_SStructPMatrix  **pmatrices = hypre_SStructMatrixPMatrices(matrix);
   int                  ***symmetric = hypre_SStructMatrixSymmetric(matrix);
   hypre_SStructStencil ***stencils  = hypre_SStructGraphStencils(graph);
   int                    *split;

   MPI_Comm                pcomm;
   hypre_SStructPGrid     *pgrid;
   hypre_SStructStencil  **pstencils;
   int                     nvars;

   int                     stencil_size;
   hypre_Index            *stencil_shape;
   int                    *stencil_vars;
   int                     pstencil_ndim;
   int                     pstencil_size;

   int                     part, var, i;

   /* GEC0902 addition of variables for ilower and iupper   */
   MPI_Comm                 comm;
   hypre_SStructGrid       *grid;
   HYPRE_BigInt             ilower, iupper;
   int                   matrix_type = hypre_SStructMatrixObjectType(matrix);

   /* S-matrix */
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGraphPGrid(graph, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      pstencils = hypre_TAlloc(hypre_SStructStencil *, nvars);
      for (var = 0; var < nvars; var++)
      {
         split = hypre_SStructMatrixSplit(matrix, part, var);
         stencil_size  = hypre_SStructStencilSize(stencils[part][var]);
         stencil_shape = hypre_SStructStencilShape(stencils[part][var]);
         stencil_vars  = hypre_SStructStencilVars(stencils[part][var]);
         pstencil_ndim = hypre_SStructStencilNDim(stencils[part][var]);
         pstencil_size = 0;
         for (i = 0; i < stencil_size; i++)
         {
            if (split[i] > -1)
            {
               pstencil_size++;
            }
         }
         HYPRE_SStructStencilCreate(pstencil_ndim, pstencil_size,
                                    &pstencils[var]);
         for (i = 0; i < stencil_size; i++)
         {
            if (split[i] > -1)
            {
               HYPRE_SStructStencilSetEntry(pstencils[var], split[i],
                                            stencil_shape[i],
                                            stencil_vars[i]);
            }
         }
      }
      pcomm = hypre_SStructPGridComm(pgrid);
      hypre_SStructPMatrixCreate(pcomm, pgrid, pstencils, &pmatrices[part]);
      for (var = 0; var < nvars; var++)
      {
         for (i = 0; i < nvars; i++)
         {
            hypre_SStructPMatrixSetSymmetric(pmatrices[part], var, i,
                                             symmetric[part][var][i]);
         }
      }
      hypre_SStructPMatrixInitialize(pmatrices[part]);
   }

      /* U-matrix */

   /* GEC0902  knowing the kind of matrix we can create the IJMATRIX with the 
    *  the right dimension (HYPRE_PARCSR without ghosts) */

   grid  = hypre_SStructGraphGrid(graph); 
   comm  =  hypre_SStructMatrixComm(matrix); 

   if(matrix_type == HYPRE_PARCSR)
   {
     ilower = hypre_SStructGridStartRank(grid);
     iupper = ilower + (HYPRE_BigInt)(hypre_SStructGridLocalSize(grid) - 1);
   }
   
    if(matrix_type == HYPRE_SSTRUCT || matrix_type == HYPRE_STRUCT)
   {
     ilower = hypre_SStructGridGhstartRank(grid);
     iupper = ilower + (HYPRE_BigInt)(hypre_SStructGridGhlocalSize(grid) - 1);
   }
    
     HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper,
                        &hypre_SStructMatrixIJMatrix(matrix)); 


   /* U-matrix */
   hypre_SStructUMatrixInitialize(matrix);

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixSetValues( HYPRE_SStructMatrix  matrix,
                              int                  part,
                              int                 *index,
                              int                  var,
                              int                  nentries,
                              int                 *entries,
                              double              *values )
{
   int ierr = 0;
   int                   ndim  = hypre_SStructMatrixNDim(matrix);
   int                  *Sentries;
   int                  *Uentries;
   int                   nSentries;
   int                   nUentries;
   hypre_SStructPMatrix *pmatrix;
   hypre_Index           cindex;

   hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   hypre_CopyToCleanIndex(index, ndim, cindex);

   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      hypre_SStructPMatrixSetValues(pmatrix, cindex, var,
                                    nSentries, Sentries, values, 0);
   }
   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetValues(matrix, part, cindex, var,
                                    nUentries, Uentries, values, 0);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixSetBoxValues( HYPRE_SStructMatrix  matrix,
                                 int                  part,
                                 int                 *ilower,
                                 int                 *iupper,
                                 int                  var,
                                 int                  nentries,
                                 int                 *entries,
                                 double              *values )
{
   int ierr = 0;
   int                   ndim  = hypre_SStructMatrixNDim(matrix);
   int                  *Sentries;
   int                  *Uentries;
   int                   nSentries;
   int                   nUentries;
   hypre_SStructPMatrix *pmatrix;
   hypre_Index           cilower;
   hypre_Index           ciupper;

   hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      hypre_SStructPMatrixSetBoxValues(pmatrix, cilower, ciupper, var,
                                       nSentries, Sentries, values, 0);
   }
   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetBoxValues(matrix, part, cilower, ciupper, var,
                                       nUentries, Uentries, values, 0);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
int
HYPRE_SStructMatrixGetValues( HYPRE_SStructMatrix  matrix,
                              int                  part,
                              int                 *index,
                              int                  var,
                              int                  nentries,
                              int                 *entries,
                              double              *values )
{
   int ierr = 0;
   int                   ndim  = hypre_SStructMatrixNDim(matrix);
   int                  *Sentries;
   int                  *Uentries;
   int                   nSentries;
   int                   nUentries;
   hypre_SStructPMatrix *pmatrix;
   hypre_Index           cindex;

   hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   hypre_CopyToCleanIndex(index, ndim, cindex);

   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      hypre_SStructPMatrixSetValues(pmatrix, cindex, var,
                                    nSentries, Sentries, values, -1);
   }
   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetValues(matrix, part, cindex, var,
                                    nUentries, Uentries, values, -1);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixGetBoxValues( HYPRE_SStructMatrix  matrix,
                                 int                  part,
                                 int                 *ilower,
                                 int                 *iupper,
                                 int                  var,
                                 int                  nentries,
                                 int                 *entries,
                                 double              *values )
{
   int ierr = 0;
   int                   ndim  = hypre_SStructMatrixNDim(matrix);
   int                  *Sentries;
   int                  *Uentries;
   int                   nSentries;
   int                   nUentries;
   hypre_SStructPMatrix *pmatrix;
   hypre_Index           cilower;
   hypre_Index           ciupper;
   int                   action;

   hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   action= -2;   /* action < -1: get values */
   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      hypre_SStructPMatrixSetBoxValues(pmatrix, cilower, ciupper, var,
                                       nSentries, Sentries, values, action);
   }
   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetBoxValues(matrix, part, cilower, ciupper, var,
                                       nUentries, Uentries, values, action);
   }

   return ierr;
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructMatrixAddToValues( HYPRE_SStructMatrix  matrix,
                                int                  part,
                                int                 *index,
                                int                  var,
                                int                  nentries,
                                int                 *entries,
                                double              *values )
{
   int ierr = 0;
   int                   ndim  = hypre_SStructMatrixNDim(matrix);
   int                  *Sentries;
   int                  *Uentries;
   int                   nSentries;
   int                   nUentries;
   hypre_SStructPMatrix *pmatrix;
   hypre_Index           cindex;

   hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   hypre_CopyToCleanIndex(index, ndim, cindex);

   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      hypre_SStructPMatrixSetValues(pmatrix, cindex, var,
                                    nSentries, Sentries, values, 1);
   }
   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetValues(matrix, part, cindex, var,
                                    nUentries, Uentries, values, 1);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructMatrixAddToBoxValues( HYPRE_SStructMatrix  matrix,
                                   int                  part,
                                   int                 *ilower,
                                   int                 *iupper,
                                   int                  var,
                                   int                  nentries,
                                   int                 *entries,
                                   double              *values )
{
   int ierr = 0;
   int                   ndim  = hypre_SStructMatrixNDim(matrix);
   int                  *Sentries;
   int                  *Uentries;
   int                   nSentries;
   int                   nUentries;
   hypre_SStructPMatrix *pmatrix;
   hypre_Index           cilower;
   hypre_Index           ciupper;

   hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      hypre_SStructPMatrixSetBoxValues(pmatrix, cilower, ciupper, var,
                                       nSentries, Sentries, values, 1);
   }
   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetBoxValues(matrix, part, cilower, ciupper, var,
                                       nUentries, Uentries, values, 1);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructMatrixAssemble( HYPRE_SStructMatrix matrix )
{
   int ierr = 0;

   hypre_SStructGraph      *graph       = hypre_SStructMatrixGraph(matrix);
   int                      nparts      = hypre_SStructMatrixNParts(matrix);
   hypre_SStructPMatrix   **pmatrices   = hypre_SStructMatrixPMatrices(matrix);
   hypre_SStructGrid       *grid        = hypre_SStructGraphGrid(graph);
   int                    **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   hypre_SStructNeighbor ***vneighbors  = hypre_SStructGridVNeighbors(grid);

   hypre_SStructPMatrix    *pmatrix;
   hypre_SStructStencil    *stencil;
   hypre_Index             *shape;
   int                     *smap;
   int                     *vars;
   hypre_StructMatrix      *smatrix;
   hypre_StructGrid        *sgrid;
   hypre_SStructNeighbor   *vneighbor;

   hypre_Box               *box, *sbox, *ibox;
   hypre_IndexRef           offset;

   int                     *entries;
   int                     *Sentries;
   int                     *Uentries;
   int                      nSentries;
   int                      nUentries;

   double                  *values = NULL;

   int                      nvars, nentries;
   int                      part, var, entry, sentry, b, sb;

   /*------------------------------------------------------
    * Move off-part couplings (described by neighbor info)
    * from S-matrix structure into U-matrix structure.
    *------------------------------------------------------*/

   box  = hypre_BoxCreate();
   ibox = hypre_BoxCreate();

   nentries = hypre_SStructMatrixEntriesSize(matrix);
   entries  = hypre_TAlloc(int, nentries);
   for (entry = 0; entry < nentries; entry++)
   {
      entries[entry] = entry;
   }

   for (part = 0; part < nparts; part++)
   {
      pmatrix  = pmatrices[part];

      nvars = hypre_SStructPMatrixNVars(pmatrix);
      for (var = 0; var < nvars; var++)
      {
         stencil  = hypre_SStructPMatrixStencil(pmatrix, var);
         smap     = hypre_SStructPMatrixSMap(pmatrix, var);
         shape    = hypre_SStructStencilShape(stencil);
         vars     = hypre_SStructStencilVars(stencil);
         nentries = hypre_SStructStencilSize(stencil);

         hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                         &nSentries, &Sentries,
                                         &nUentries, &Uentries);

         for (entry = 0; entry < nSentries; entry++)
         {
            smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var,
                                                  vars[entries[entry]]);
            sentry = smap[entries[entry]];

            /* Shift/intersect neighbor box and move values */
            for (b = 0; b < nvneighbors[part][var]; b++)
            {
               vneighbor = &vneighbors[part][var][b];
               hypre_CopyBox(hypre_SStructNeighborBox(vneighbor), box);

               /* shift box by stencil offset */
               offset = shape[entry];
               hypre_BoxIMinX(box) -= hypre_IndexX(offset);
               hypre_BoxIMinY(box) -= hypre_IndexY(offset);
               hypre_BoxIMinZ(box) -= hypre_IndexZ(offset);
               hypre_BoxIMaxX(box) -= hypre_IndexX(offset);
               hypre_BoxIMaxY(box) -= hypre_IndexY(offset);
               hypre_BoxIMaxZ(box) -= hypre_IndexZ(offset);

               sgrid = hypre_StructMatrixGrid(smatrix);
               hypre_ForStructGridBoxI(sb, sgrid)
                  {
                     sbox = hypre_StructGridBox(sgrid, sb);
                     hypre_IntersectBoxes(box, sbox, ibox);

                     if (hypre_BoxVolume(ibox))
                     {
                        values = hypre_TReAlloc(values, double,
                                                hypre_BoxVolume(ibox));

                        /* move matrix values from S-matrix to U-matrix */
                        hypre_StructMatrixSetBoxValues(smatrix, ibox,
                                                       1, &sentry, values, -1);

                        hypre_SStructUMatrixSetBoxValues(matrix, part,
                                                         hypre_BoxIMin(ibox),
                                                         hypre_BoxIMax(ibox),
                                                         var, 1, &entry,
                                                         values, 1);
                     }
                  }
            }
         }
      }
   }

   hypre_TFree(entries);
   hypre_TFree(values);
   hypre_BoxDestroy(box);
   hypre_BoxDestroy(ibox);

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPMatrixAssemble(pmatrices[part]);
   }

   /* U-matrix */
   hypre_SStructUMatrixAssemble(matrix);

   return ierr;
}

/*--------------------------------------------------------------------------
 * NOTE: Should set things up so that this information can be passed
 * immediately to the PMatrix.  Unfortunately, the PMatrix is
 * currently not created until the SStructMatrix is initialized.
 *--------------------------------------------------------------------------*/
 
int
HYPRE_SStructMatrixSetSymmetric( HYPRE_SStructMatrix matrix,
                                 int                 part,
                                 int                 var,
                                 int                 to_var,
                                 int                 symmetric )
{
   int ierr = 0;

   int                ***msymmetric = hypre_SStructMatrixSymmetric(matrix);
   hypre_SStructGraph   *graph      = hypre_SStructMatrixGraph(matrix);
   hypre_SStructPGrid   *pgrid;

   int pstart = part;
   int psize  = 1;
   int vstart = var;
   int vsize  = 1;
   int tstart = to_var;
   int tsize  = 1;
   int p, v, t;

   if (part == -1)
   {
      pstart = 0;
      psize  = hypre_SStructMatrixNParts(matrix);
   }

   for (p = pstart; p < psize; p++)
   {
      pgrid = hypre_SStructGraphPGrid(graph, p);
      if (var == -1)
      {
         vstart = 0;
         vsize  = hypre_SStructPGridNVars(pgrid);
      }
      if (to_var == -1)
      {
         tstart = 0;
         tsize  = hypre_SStructPGridNVars(pgrid);
      }

      for (v = vstart; v < vsize; v++)
      {
         for (t = tstart; t < tsize; t++)
         {
            msymmetric[p][v][t] = symmetric;
         }
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
int
HYPRE_SStructMatrixSetNSSymmetric( HYPRE_SStructMatrix matrix,
                                   int                 symmetric )
{
   int ierr = 0;

   hypre_SStructMatrixNSSymmetric(matrix) = symmetric;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixSetObjectType( HYPRE_SStructMatrix  matrix,
                                  int                  type )
{
   int ierr = 0;

   hypre_SStructGraph     *graph    = hypre_SStructMatrixGraph(matrix);
   int                  ***splits   = hypre_SStructMatrixSplits(matrix);
   int                     nparts   = hypre_SStructMatrixNParts(matrix);
   hypre_SStructStencil ***stencils = hypre_SStructGraphStencils(graph);

   hypre_SStructPGrid     *pgrid;
   int                     nvars;
   int                     stencil_size;
   int                     part, var, i;

   hypre_SStructMatrixObjectType(matrix) = type ;   

   /* RDF: This and all other modifications to 'split' really belong
    * in the Initialize routine */
   if (type != HYPRE_SSTRUCT && type != HYPRE_STRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGraphPGrid(graph, part);
         nvars = hypre_SStructPGridNVars(pgrid);
         for (var = 0; var < nvars; var++)
         {
            stencil_size = hypre_SStructStencilSize(stencils[part][var]);
            for (i = 0; i < stencil_size; i++)
            {
               splits[part][var][i] = -1;
            }
         }
      }
   }
   
   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixGetObject( HYPRE_SStructMatrix   matrix,
                              void                **object )
{
   int ierr = 0;

   int            type     = hypre_SStructMatrixObjectType(matrix);
   HYPRE_IJMatrix ijmatrix = hypre_SStructMatrixIJMatrix(matrix);
   hypre_SStructPMatrix *pA;
   hypre_StructMatrix   *sA;
   int                   part, var;
 

   if (type == HYPRE_PARCSR)
   {
       HYPRE_IJMatrixGetObject(ijmatrix, object);
   }

   else if (type == HYPRE_SSTRUCT)
   {
      *object= matrix;
   }

   else if (type == HYPRE_STRUCT)
   {
      /* only one part & one variable */
      part= 0;
      pA  = hypre_SStructMatrixPMatrix(matrix, part);
      var = 0;
      sA  = hypre_SStructPMatrixSMatrix(pA, var, var);
     *object= sA;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixGetObject2( HYPRE_SStructMatrix   matrix,
                               void                **object )
{
   int ierr = 0;

   int            type     = hypre_SStructMatrixObjectType(matrix);
   HYPRE_IJMatrix ijmatrix = hypre_SStructMatrixIJMatrix(matrix);
   hypre_SStructPMatrix *pA;
   hypre_StructMatrix   *sA;
   int                   part, var;
 

   if (type == HYPRE_PARCSR)
   {
      /* only difference from ..GetObject: here returns an IJMatrix, not a ParCSRMatrix */
      *object = ijmatrix;
   }

   else if (type == HYPRE_SSTRUCT)
   {
      *object= matrix;
   }

   else if (type == HYPRE_STRUCT)
   {
      /* only one part & one variable */
      part= 0;
      pA  = hypre_SStructMatrixPMatrix(matrix, part);
      var = 0;
      sA  = hypre_SStructPMatrixSMatrix(pA, var, var);
     *object= sA;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixPrint( const char          *filename,
                          HYPRE_SStructMatrix  matrix,
                          int                  all )
{
   int ierr = 0;
   int  nparts = hypre_SStructMatrixNParts(matrix);
   int  part;
   char new_filename[255];

   for (part = 0; part < nparts; part++)
   {
      sprintf(new_filename, "%s.%02d", filename, part);
      hypre_SStructPMatrixPrint(new_filename,
                                hypre_SStructMatrixPMatrix(matrix, part),
                                all);
   }

   /* U-matrix */
   sprintf(new_filename, "%s.UMatrix", filename);
   HYPRE_IJMatrixPrint(hypre_SStructMatrixIJMatrix(matrix), new_filename);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixMatvec
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixMatvec( double alpha,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector x,
                           double beta,
                           HYPRE_SStructVector y     )
{
   return ( hypre_SStructMatvec( alpha, (hypre_SStructMatrix *) A,
                                 (hypre_SStructVector *) x, beta,
                                 (hypre_SStructVector *) y) );
}

