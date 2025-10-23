#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_sstruct_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include "sstruct_mv.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * Data structures
 *--------------------------------------------------------------------------*/

char infile_default[50] = "sstruct.in.MG.FD";

typedef int Index[3];

/*------------------------------------------------------------
 * ProblemIndex:
 *
 * The index has extra information stored in entries 3-8 that
 * determine how the index gets "mapped" to finer index spaces.
 *
 * NOTE: For implementation convenience, the index is "pre-shifted"
 * according to the values in entries 6,7,8.  The following discussion
 * describes how "un-shifted" indexes are mapped, because that is a
 * more natural way to think about this mapping problem, and because
 * that is the convention used in the input file for this code.  The
 * reason that pre-shifting is convenient is because it makes the true
 * value of the index on the unrefined index space readily available
 * in entries 0-2, hence, all operations on that unrefined space are
 * straightforward.  Also, the only time that the extra mapping
 * information is needed is when an index is mapped to a new refined
 * index space, allowing us to isolate the mapping details to the
 * routine MapProblemIndex.  The only other effected routine is
 * SScanProblemIndex, which takes the user input and pre-shifts it.
 *
 * - Entries 3,4,5 have values of either 0 or 1 that indicate
 *   whether to map an index "to the left" or "to the right".
 *   Here is a 1D diagram:
 *
 *    --  |     *     |    unrefined index space
 *   |
 *    --> | * | . | * |    refined index space (factor = 3)
 *          0       1
 *
 *   The '*' index on the unrefined index space gets mapped to one of
 *   the '*' indexes on the refined space based on the value (0 or 1)
 *   of the relevent entry (3,4, or 5).  The actual mapping formula is
 *   as follows (with refinement factor, r):
 *
 *   mapped_index[i] = r*index[i] + (r-1)*index[i+3]
 *
 * - Entries 6,7,8 contain "shift" information.  The shift is
 *   simply added to the mapped index just described.  So, the
 *   complete mapping formula is as follows:
 *
 *   mapped_index[i] = r*index[i] + (r-1)*index[i+3] + index[i+6]
 *
 *------------------------------------------------------------*/

typedef int ProblemIndex[9];

typedef struct
{
   /* for GridSetExtents */
   int                    nboxes;
   ProblemIndex          *ilowers;
   ProblemIndex          *iuppers;
   int                   *boxsizes;
   int                    max_boxsize;

   /* for GridSetVariables */
   int                    nvars;
   HYPRE_SStructVariable *vartypes;

   /* for GridAddVariables */
   int                    add_nvars;
   ProblemIndex          *add_indexes;
   HYPRE_SStructVariable *add_vartypes;

   /* for GridSetNeighborBox */
   int                    glue_nboxes;
   ProblemIndex          *glue_ilowers;
   ProblemIndex          *glue_iuppers;
   int                   *glue_nbor_parts;
   ProblemIndex          *glue_nbor_ilowers;
   ProblemIndex          *glue_nbor_iuppers;
   Index                 *glue_index_maps;
   int                   *glue_primaries;

   /* for GraphSetStencil */
   int                   *stencil_num;

   /* for GraphAddEntries */
   int                    graph_nboxes;
   ProblemIndex          *graph_ilowers;
   ProblemIndex          *graph_iuppers;
   Index                 *graph_strides;
   int                   *graph_vars;
   int                   *graph_to_parts;
   ProblemIndex          *graph_to_ilowers;
   ProblemIndex          *graph_to_iuppers;
   Index                 *graph_to_strides;
   int                   *graph_to_vars;
   Index                 *graph_index_maps;
   Index                 *graph_index_signs;
   int                   *graph_entries;
   double                *graph_values;
   int                   *graph_boxsizes;

   /* MatrixSetValues */
   int                    matset_nboxes;
   ProblemIndex          *matset_ilowers;
   ProblemIndex          *matset_iuppers;
   Index                 *matset_strides;
   int                   *matset_vars;
   int                   *matset_entries;
   double                *matset_values;

   /* MatrixAddToValues */
   int                    matadd_nboxes;
   ProblemIndex          *matadd_ilowers;
   ProblemIndex          *matadd_iuppers;
   int                   *matadd_vars;
   int                   *matadd_nentries;
   int                  **matadd_entries;
   double               **matadd_values;

   Index                  periodic;

} ProblemPartData;
 
typedef struct
{
   int              ndim;
   int              nparts;
   ProblemPartData *pdata;
   int              max_boxsize;

   int              nstencils;
   int             *stencil_sizes;
   Index          **stencil_offsets;
   int            **stencil_vars;
   double         **stencil_values;

   int              symmetric_num;
   int             *symmetric_parts;
   int             *symmetric_vars;
   int             *symmetric_to_vars;
   int             *symmetric_booleans;

   int              ns_symmetric;

   int              npools;
   int             *pools;   /* array of size nparts */
   int              ndists;  /* number of (pool) distributions */
   int		   *dist_npools;
   int		  **dist_pools;
} ProblemData;

/*--------------------------------------------------------------------------
 * definitions for IJ examples, can be replaced later when sstruct interface
 * completely scalable 
 *--------------------------------------------------------------------------*/

/*3D Laplace on a cube*/
int BuildParLaplacian (int argc , char *argv [], double *system_size_ptr, HYPRE_ParCSRMatrix *A_ptr , HYPRE_ParVector *rhs_ptr , HYPRE_ParVector *x_ptr);
/* Laplace type problem, with 27pt stencil on a cube */
int BuildParLaplacian27pt (int argc , char *argv [], double *system_size_ptr, HYPRE_ParCSRMatrix *A_ptr , HYPRE_ParVector *rhs_ptr , HYPRE_ParVector *x_ptr);
/* PDE with jumps on a cube */
int BuildParVarDifConv (int argc , char *argv [], double *system_size_ptr, HYPRE_ParCSRMatrix *A_ptr , HYPRE_ParVector *rhs_ptr , HYPRE_ParVector *x_ptr);
/* 2D PDE with rotated anisotropy */
int BuildParRotate7pt (int argc , char *argv [], double *system_size_ptr , HYPRE_ParCSRMatrix *A_ptr , HYPRE_ParVector *rhs_ptr , HYPRE_ParVector *x_ptr);
/* 2D PDE with 9pt stencil */
int BuildParLaplacian9pt (int argc , char *argv [], double *system_size_ptr , HYPRE_ParCSRMatrix *A_ptr , HYPRE_ParVector *rhs_ptr , HYPRE_ParVector *x_ptr);
 

/*--------------------------------------------------------------------------
 * Compute new box based on variable type
 *--------------------------------------------------------------------------*/

int
GetVariableBox( Index  cell_ilower,
                Index  cell_iupper,
                int    int_vartype,
                Index  var_ilower,
                Index  var_iupper )
{
   int ierr = 0;
   HYPRE_SStructVariable  vartype = (HYPRE_SStructVariable) int_vartype;

   var_ilower[0] = cell_ilower[0];
   var_ilower[1] = cell_ilower[1];
   var_ilower[2] = cell_ilower[2];
   var_iupper[0] = cell_iupper[0];
   var_iupper[1] = cell_iupper[1];
   var_iupper[2] = cell_iupper[2];

   switch(vartype)
   {
      case HYPRE_SSTRUCT_VARIABLE_CELL:
      var_ilower[0] -= 0; var_ilower[1] -= 0; var_ilower[2] -= 0;
      break;
      case HYPRE_SSTRUCT_VARIABLE_NODE:
      var_ilower[0] -= 1; var_ilower[1] -= 1; var_ilower[2] -= 1;
      break;
      case HYPRE_SSTRUCT_VARIABLE_XFACE:
      var_ilower[0] -= 1; var_ilower[1] -= 0; var_ilower[2] -= 0;
      break;
      case HYPRE_SSTRUCT_VARIABLE_YFACE:
      var_ilower[0] -= 0; var_ilower[1] -= 1; var_ilower[2] -= 0;
      break;
      case HYPRE_SSTRUCT_VARIABLE_ZFACE:
      var_ilower[0] -= 0; var_ilower[1] -= 0; var_ilower[2] -= 1;
      break;
      case HYPRE_SSTRUCT_VARIABLE_XEDGE:
      var_ilower[0] -= 0; var_ilower[1] -= 1; var_ilower[2] -= 1;
      break;
      case HYPRE_SSTRUCT_VARIABLE_YEDGE:
      var_ilower[0] -= 1; var_ilower[1] -= 0; var_ilower[2] -= 1;
      break;
      case HYPRE_SSTRUCT_VARIABLE_ZEDGE:
      var_ilower[0] -= 1; var_ilower[1] -= 1; var_ilower[2] -= 0;
      break;
      case HYPRE_SSTRUCT_VARIABLE_UNDEFINED:
      break;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * Read routines
 *--------------------------------------------------------------------------*/

int
SScanIntArray( char  *sdata_ptr,
               char **sdata_ptr_ptr,
               int    size,
               int   *array )
{
   int i;

   sdata_ptr += strspn(sdata_ptr, " \t\n[");
   for (i = 0; i < size; i++)
   {
      array[i] = strtol(sdata_ptr, &sdata_ptr, 10);
   }
   sdata_ptr += strcspn(sdata_ptr, "]") + 1;

   *sdata_ptr_ptr = sdata_ptr;
   return 0;
}

int
SScanDblArray( char   *sdata_ptr,
               char  **sdata_ptr_ptr,
               int     size,
               double *array )
{
   int i;
                                                                                                                           
   sdata_ptr += strspn(sdata_ptr, " \t\n[");
   for (i = 0; i < size; i++)
   {
      array[i] = strtod(sdata_ptr, &sdata_ptr);
   }
   sdata_ptr += strcspn(sdata_ptr, "]") + 1;
                                                                                                                           
   *sdata_ptr_ptr = sdata_ptr;
   return 0;
}
                                                                                                                           
int
SScanProblemIndex( char          *sdata_ptr,
                   char         **sdata_ptr_ptr,
                   int            ndim,
                   ProblemIndex   index )
{
   int  i;
   char sign[3];

   /* initialize index array */
   for (i = 0; i < 9; i++)
   {
      index[i]   = 0;
   }

   sdata_ptr += strspn(sdata_ptr, " \t\n(");
   switch (ndim)
   {
      case 1:
      sscanf(sdata_ptr, "%d%c",
             &index[0], &sign[0]);
      break;

      case 2:
      sscanf(sdata_ptr, "%d%c%d%c",
             &index[0], &sign[0], &index[1], &sign[1]);
      break;

      case 3:
      sscanf(sdata_ptr, "%d%c%d%c%d%c",
             &index[0], &sign[0], &index[1], &sign[1], &index[2], &sign[2]);
      break;
   }
   sdata_ptr += strcspn(sdata_ptr, ":)");
   if ( *sdata_ptr == ':' )
   {
      /* read in optional shift */
      sdata_ptr += 1;
      switch (ndim)
      {
         case 1:
            sscanf(sdata_ptr, "%d", &index[6]);
            break;
            
         case 2:
            sscanf(sdata_ptr, "%d%d", &index[6], &index[7]);
            break;
            
         case 3:
            sscanf(sdata_ptr, "%d%d%d", &index[6], &index[7], &index[8]);
            break;
      }
      /* pre-shift the index */
      for (i = 0; i < ndim; i++)
      {
         index[i] += index[i+6];
      }
   }
   sdata_ptr += strcspn(sdata_ptr, ")") + 1;

   for (i = 0; i < ndim; i++)
   {
      if (sign[i] == '+')
      {
         index[i+3] = 1;
      }
   }

   *sdata_ptr_ptr = sdata_ptr;
   return 0;
}

int
ReadData( char         *filename,
          ProblemData  *data_ptr )
{
   ProblemData        data;
   ProblemPartData    pdata;

   int                myid;
   FILE              *file;

   char              *sdata = NULL;
   char              *sdata_line;
   char              *sdata_ptr;
   int                sdata_size;
   int                size;
   int                memchunk = 10000;
   int                maxline  = 250;

   char               key[250];

   int                part, var, s, entry, i, il, iu;

   /*-----------------------------------------------------------
    * Read data file from process 0, then broadcast
    *-----------------------------------------------------------*/
 
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   if (myid == 0)
   {
      if ((file = fopen(filename, "r")) == NULL)
      {
         printf("Error: can't open input file %s\n", filename);
         exit(1);
      }

      /* allocate initial space, and read first input line */
      sdata_size = 0;
      sdata = hypre_TAlloc(char, memchunk);
      sdata_line = fgets(sdata, maxline, file);

      s= 0;
      while (sdata_line != NULL)
      {
         sdata_size += strlen(sdata_line) + 1;

         /* allocate more space, if necessary */
         if ((sdata_size + maxline) > s)
         {
            sdata = hypre_TReAlloc(sdata, char, (sdata_size + memchunk));
            s= sdata_size + memchunk;
         }
         
         /* read the next input line */
         sdata_line = fgets((sdata + sdata_size), maxline, file);
      }
   }

   /* broadcast the data size */
   MPI_Bcast(&sdata_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

   /* broadcast the data */
   sdata = hypre_TReAlloc(sdata, char, sdata_size);
   MPI_Bcast(sdata, sdata_size, MPI_CHAR, 0, MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Parse the data and fill ProblemData structure
    *-----------------------------------------------------------*/

   data.max_boxsize = 0;
   data.symmetric_num = 0;
   data.symmetric_parts    = NULL;
   data.symmetric_vars     = NULL;
   data.symmetric_to_vars  = NULL;
   data.symmetric_booleans = NULL;
   data.ns_symmetric = 0;
   data.ndists = 0;
   data.dist_npools = NULL;
   data.dist_pools  = NULL;

   sdata_line = sdata;
   while (sdata_line < (sdata + sdata_size))
   {
      sdata_ptr = sdata_line;
      
      if ( ( sscanf(sdata_ptr, "%s", key) > 0 ) && ( sdata_ptr[0] != '#' ) )
      {
         sdata_ptr += strcspn(sdata_ptr, " \t\n");

         if ( strcmp(key, "GridCreate:") == 0 )
         {
            data.ndim = strtol(sdata_ptr, &sdata_ptr, 10);
            data.nparts = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pdata = hypre_CTAlloc(ProblemPartData, data.nparts);
         }
         else if ( strcmp(key, "GridSetExtents:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.nboxes % 10) == 0)
            {
               size = pdata.nboxes + 10;
               pdata.ilowers =
                  hypre_TReAlloc(pdata.ilowers, ProblemIndex, size);
               pdata.iuppers =
                  hypre_TReAlloc(pdata.iuppers, ProblemIndex, size);
               pdata.boxsizes =
                  hypre_TReAlloc(pdata.boxsizes, int, size);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.ilowers[pdata.nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.iuppers[pdata.nboxes]);
            /* check use of +- in GridSetExtents */
            il = 1;
            iu = 1;
            for (i = 0; i < data.ndim; i++)
            {
               il *= pdata.ilowers[pdata.nboxes][i+3];
               iu *= pdata.iuppers[pdata.nboxes][i+3];
            }
            if ( (il != 0) || (iu != 1) )
            {
               printf("Error: Invalid use of `+-' in GridSetExtents\n");
               exit(1);
            }
            pdata.boxsizes[pdata.nboxes] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.boxsizes[pdata.nboxes] *=
                  (pdata.iuppers[pdata.nboxes][i] -
                   pdata.ilowers[pdata.nboxes][i] + 2);
            }
            pdata.max_boxsize =
               hypre_max(pdata.max_boxsize, pdata.boxsizes[pdata.nboxes]);
            pdata.nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridSetVariables:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            pdata.nvars = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.vartypes = hypre_CTAlloc(HYPRE_SStructVariable, pdata.nvars);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          pdata.nvars, (int *) pdata.vartypes);
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridAddVariables:") == 0 )
         {
            /* TODO */
            printf("GridAddVariables not yet implemented!\n");
            exit(1);
         }
         else if ( strcmp(key, "GridSetNeighborBox:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.glue_nboxes % 10) == 0)
            {
               size = pdata.glue_nboxes + 10;
               pdata.glue_ilowers =
                  hypre_TReAlloc(pdata.glue_ilowers, ProblemIndex, size);
               pdata.glue_iuppers =
                  hypre_TReAlloc(pdata.glue_iuppers, ProblemIndex, size);
               pdata.glue_nbor_parts =
                  hypre_TReAlloc(pdata.glue_nbor_parts, int, size);
               pdata.glue_nbor_ilowers =
                  hypre_TReAlloc(pdata.glue_nbor_ilowers, ProblemIndex, size);
               pdata.glue_nbor_iuppers =
                  hypre_TReAlloc(pdata.glue_nbor_iuppers, ProblemIndex, size);
               pdata.glue_index_maps =
                  hypre_TReAlloc(pdata.glue_index_maps, Index, size);
               pdata.glue_primaries =
                  hypre_TReAlloc(pdata.glue_primaries, int, size);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_ilowers[pdata.glue_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_iuppers[pdata.glue_nboxes]);
            pdata.glue_nbor_parts[pdata.glue_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_nbor_ilowers[pdata.glue_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.glue_nbor_iuppers[pdata.glue_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.glue_index_maps[pdata.glue_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.glue_index_maps[pdata.glue_nboxes][i] = i;
            }
            sdata_ptr += strcspn(sdata_ptr, ":\t\n");
            if ( *sdata_ptr == ':' )
            {
               /* read in optional primary indicator */
               sdata_ptr += 1;
               pdata.glue_primaries[pdata.glue_nboxes] =
                  strtol(sdata_ptr, &sdata_ptr, 10);
            }
            else
            {
               pdata.glue_primaries[pdata.glue_nboxes] = -1;
               sdata_ptr -= 1;
            }
            pdata.glue_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GridSetPeriodic:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim, pdata.periodic);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.periodic[i] = 0;
            }
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "StencilCreate:") == 0 )
         {
            data.nstencils = strtol(sdata_ptr, &sdata_ptr, 10);
            data.stencil_sizes   = hypre_CTAlloc(int, data.nstencils);
            data.stencil_offsets = hypre_CTAlloc(Index *, data.nstencils);
            data.stencil_vars    = hypre_CTAlloc(int *, data.nstencils);
            data.stencil_values  = hypre_CTAlloc(double *, data.nstencils);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          data.nstencils, data.stencil_sizes);
            for (s = 0; s < data.nstencils; s++)
            {
               data.stencil_offsets[s] =
                  hypre_CTAlloc(Index, data.stencil_sizes[s]);
               data.stencil_vars[s] =
                  hypre_CTAlloc(int, data.stencil_sizes[s]);
               data.stencil_values[s] =
                  hypre_CTAlloc(double, data.stencil_sizes[s]);
            }
         }
         else if ( strcmp(key, "StencilSetEntry:") == 0 )
         {
            s = strtol(sdata_ptr, &sdata_ptr, 10);
            entry = strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr,
                          data.ndim, data.stencil_offsets[s][entry]);
            for (i = data.ndim; i < 3; i++)
            {
               data.stencil_offsets[s][entry][i] = 0;
            }
            data.stencil_vars[s][entry] = strtol(sdata_ptr, &sdata_ptr, 10);
            data.stencil_values[s][entry] = strtod(sdata_ptr, &sdata_ptr);
         }
         else if ( strcmp(key, "GraphSetStencil:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            var = strtol(sdata_ptr, &sdata_ptr, 10);
            s = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if (pdata.stencil_num == NULL)
            {
               pdata.stencil_num = hypre_CTAlloc(int, pdata.nvars);
            }
            pdata.stencil_num[var] = s;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "GraphAddEntries:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.graph_nboxes % 10) == 0)
            {
               size = pdata.graph_nboxes + 10;
               pdata.graph_ilowers =
                  hypre_TReAlloc(pdata.graph_ilowers, ProblemIndex, size);
               pdata.graph_iuppers =
                  hypre_TReAlloc(pdata.graph_iuppers, ProblemIndex, size);
               pdata.graph_strides =
                  hypre_TReAlloc(pdata.graph_strides, Index, size);
               pdata.graph_vars =
                  hypre_TReAlloc(pdata.graph_vars, int, size);
               pdata.graph_to_parts =
                  hypre_TReAlloc(pdata.graph_to_parts, int, size);
               pdata.graph_to_ilowers =
                  hypre_TReAlloc(pdata.graph_to_ilowers, ProblemIndex, size);
               pdata.graph_to_iuppers =
                  hypre_TReAlloc(pdata.graph_to_iuppers, ProblemIndex, size);
               pdata.graph_to_strides =
                  hypre_TReAlloc(pdata.graph_to_strides, Index, size);
               pdata.graph_to_vars =
                  hypre_TReAlloc(pdata.graph_to_vars, int, size);
               pdata.graph_index_maps =
                  hypre_TReAlloc(pdata.graph_index_maps, Index, size);
               pdata.graph_index_signs =
                  hypre_TReAlloc(pdata.graph_index_signs, Index, size);
               pdata.graph_entries =
                  hypre_TReAlloc(pdata.graph_entries, int, size);
               pdata.graph_values =
                  hypre_TReAlloc(pdata.graph_values, double, size);
               pdata.graph_boxsizes =
                  hypre_TReAlloc(pdata.graph_boxsizes, int, size);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_ilowers[pdata.graph_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_iuppers[pdata.graph_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_strides[pdata.graph_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_strides[pdata.graph_nboxes][i] = 1;
            }
            pdata.graph_vars[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.graph_to_parts[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_to_ilowers[pdata.graph_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.graph_to_iuppers[pdata.graph_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_to_strides[pdata.graph_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_to_strides[pdata.graph_nboxes][i] = 1;
            }
            pdata.graph_to_vars[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.graph_index_maps[pdata.graph_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.graph_index_maps[pdata.graph_nboxes][i] = i;
            }
            for (i = 0; i < 3; i++)
            {
               pdata.graph_index_signs[pdata.graph_nboxes][i] = 1;
               if ( pdata.graph_to_iuppers[pdata.graph_nboxes][i] <
                    pdata.graph_to_ilowers[pdata.graph_nboxes][i] )
               {
                  pdata.graph_index_signs[pdata.graph_nboxes][i] = -1;
               }
            }
            pdata.graph_entries[pdata.graph_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.graph_values[pdata.graph_nboxes] =
               strtod(sdata_ptr, &sdata_ptr);
            pdata.graph_boxsizes[pdata.graph_nboxes] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.graph_boxsizes[pdata.graph_nboxes] *=
                  (pdata.graph_iuppers[pdata.graph_nboxes][i] -
                   pdata.graph_ilowers[pdata.graph_nboxes][i] + 1);
            }
            pdata.graph_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "MatrixSetSymmetric:") == 0 )
         {
            if ((data.symmetric_num % 10) == 0)
            {
               size = data.symmetric_num + 10;
               data.symmetric_parts =
                  hypre_TReAlloc(data.symmetric_parts, int, size);
               data.symmetric_vars =
                  hypre_TReAlloc(data.symmetric_vars, int, size);
               data.symmetric_to_vars =
                  hypre_TReAlloc(data.symmetric_to_vars, int, size);
               data.symmetric_booleans =
                  hypre_TReAlloc(data.symmetric_booleans, int, size);
            }
            data.symmetric_parts[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_vars[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_to_vars[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_booleans[data.symmetric_num] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            data.symmetric_num++;
         }
         else if ( strcmp(key, "MatrixSetNSSymmetric:") == 0 )
         {
            data.ns_symmetric = strtol(sdata_ptr, &sdata_ptr, 10);
         }
         else if ( strcmp(key, "MatrixSetValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.matset_nboxes % 10) == 0)
            {
               size = pdata.matset_nboxes + 10;
               pdata.matset_ilowers =
                  hypre_TReAlloc(pdata.matset_ilowers, ProblemIndex, size);
               pdata.matset_iuppers =
                  hypre_TReAlloc(pdata.matset_iuppers, ProblemIndex, size);
               pdata.matset_strides =
                  hypre_TReAlloc(pdata.matset_strides, Index, size);
               pdata.matset_vars =
                  hypre_TReAlloc(pdata.matset_vars, int, size);
               pdata.matset_entries =
                  hypre_TReAlloc(pdata.matset_entries, int, size);
               pdata.matset_values =
                  hypre_TReAlloc(pdata.matset_values, double, size);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matset_ilowers[pdata.matset_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
                              pdata.matset_iuppers[pdata.matset_nboxes]);
            SScanIntArray(sdata_ptr, &sdata_ptr, data.ndim,
                          pdata.matset_strides[pdata.matset_nboxes]);
            for (i = data.ndim; i < 3; i++)
            {
               pdata.matset_strides[pdata.matset_nboxes][i] = 1;
            }
            pdata.matset_vars[pdata.matset_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matset_entries[pdata.matset_nboxes] =
               strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matset_values[pdata.matset_nboxes] =
               strtod(sdata_ptr, &sdata_ptr);
            pdata.matset_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "MatrixAddToValues:") == 0 )
         {
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            pdata = data.pdata[part];
            if ((pdata.matadd_nboxes% 10) == 0)
            {
               size = pdata.matadd_nboxes+10;
               pdata.matadd_ilowers=
                  hypre_TReAlloc(pdata.matadd_ilowers, ProblemIndex, size);
               pdata.matadd_iuppers=
                  hypre_TReAlloc(pdata.matadd_iuppers, ProblemIndex, size);
               pdata.matadd_vars=
                  hypre_TReAlloc(pdata.matadd_vars, int, size);
               pdata.matadd_nentries=
                  hypre_TReAlloc(pdata.matadd_nentries, int, size);
               pdata.matadd_entries=
                  hypre_TReAlloc(pdata.matadd_entries, int *, size);
               pdata.matadd_values=
                  hypre_TReAlloc(pdata.matadd_values, double *, size);
            }
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
               pdata.matadd_ilowers[pdata.matadd_nboxes]);
            SScanProblemIndex(sdata_ptr, &sdata_ptr, data.ndim,
               pdata.matadd_iuppers[pdata.matadd_nboxes]);
            pdata.matadd_vars[pdata.matadd_nboxes]=
                strtol(sdata_ptr, &sdata_ptr, 10);
            i= strtol(sdata_ptr, &sdata_ptr, 10);
            pdata.matadd_nentries[pdata.matadd_nboxes]= i;
            pdata.matadd_entries[pdata.matadd_nboxes] =
               hypre_TAlloc(int, i);
            SScanIntArray(sdata_ptr, &sdata_ptr, i,
              (int*) pdata.matadd_entries[pdata.matadd_nboxes]);
            pdata.matadd_values[pdata.matadd_nboxes] =
               hypre_TAlloc(double, i);
            SScanDblArray(sdata_ptr, &sdata_ptr, i,
              (double *) pdata.matadd_values[pdata.matadd_nboxes]);
            pdata.matadd_nboxes++;
            data.pdata[part] = pdata;
         }
         else if ( strcmp(key, "ProcessPoolCreate:") == 0 )
         {
            data.ndists++;
            data.dist_npools= hypre_TReAlloc(data.dist_npools, int, data.ndists);
            data.dist_pools= hypre_TReAlloc(data.dist_pools, int *, data.ndists);
            data.dist_npools[data.ndists-1] = strtol(sdata_ptr, &sdata_ptr, 10);
            data.dist_pools[data.ndists-1] = hypre_CTAlloc(int, data.nparts);
#if 0
            data.npools = strtol(sdata_ptr, &sdata_ptr, 10);
            data.pools = hypre_CTAlloc(int, data.nparts);
#endif
         }
         else if ( strcmp(key, "ProcessPoolSetPart:") == 0 )
         {
            i = strtol(sdata_ptr, &sdata_ptr, 10);
            part = strtol(sdata_ptr, &sdata_ptr, 10);
            data.dist_pools[data.ndists-1][part] = i;
            /*data.pools[part] = i; */
         }
      }

      sdata_line += strlen(sdata_line) + 1;
   }

   data.max_boxsize = 0;
   for (part = 0; part < data.nparts; part++)
   {
      data.max_boxsize =
         hypre_max(data.max_boxsize, data.pdata[part].max_boxsize);
   }

   hypre_TFree(sdata);

   *data_ptr = data; 
   return 0;
}
 
/*--------------------------------------------------------------------------
 * Distribute routines
 *--------------------------------------------------------------------------*/

int
MapProblemIndex( ProblemIndex index,
                 Index        m )
{
   /* un-shift the index */
   index[0] -= index[6];
   index[1] -= index[7];
   index[2] -= index[8];
   /* map the index */
   index[0] = m[0]*index[0] + (m[0]-1)*index[3];
   index[1] = m[1]*index[1] + (m[1]-1)*index[4];
   index[2] = m[2]*index[2] + (m[2]-1)*index[5];
   /* pre-shift the new mapped index */
   index[0] += index[6];
   index[1] += index[7];
   index[2] += index[8];

   return 0;
}

int
IntersectBoxes( ProblemIndex ilower1,
                ProblemIndex iupper1,
                ProblemIndex ilower2,
                ProblemIndex iupper2,
                ProblemIndex int_ilower,
                ProblemIndex int_iupper )
{
   int d, size;

   size = 1;
   for (d = 0; d < 3; d++)
   {
      int_ilower[d] = hypre_max(ilower1[d], ilower2[d]);
      int_iupper[d] = hypre_min(iupper1[d], iupper2[d]);
      size *= hypre_max(0, (int_iupper[d] - int_ilower[d] + 1));
   }

   return size;
}

int
DistributeData( ProblemData   global_data,
                int	      pooldist,
                Index        *refine,
                Index        *distribute,
                Index        *block,
                int           num_procs,
                int           myid,
                ProblemData  *data_ptr )
{
   ProblemData      data = global_data;
   ProblemPartData  pdata;
   int             *pool_procs;
   int              np, pid;
   int              pool, part, box, b, p, q, r, i, d;
   int              dmap, sign, size;
   Index            m, mmap, n;
   ProblemIndex     ilower, iupper, int_ilower, int_iupper;

   /* set default pool distribution */
   data.npools = data.dist_npools[pooldist];
   data.pools  = data.dist_pools[pooldist];

   /* determine first process number in each pool */
   pool_procs = hypre_CTAlloc(int, (data.npools+1));
   for (part = 0; part < data.nparts; part++)
   {
      pool = data.pools[part] + 1;
      np = distribute[part][0] * distribute[part][1] * distribute[part][2];
      pool_procs[pool] = hypre_max(pool_procs[pool], np);

   }
   pool_procs[0] = 0;
   for (pool = 1; pool < (data.npools + 1); pool++)
   {
      pool_procs[pool] = pool_procs[pool - 1] + pool_procs[pool];
   }

   /* check number of processes */
   if (pool_procs[data.npools] != num_procs)
   {
      printf("Error: Invalid number of processes or process topology\n");
      exit(1);
   }

   /* modify part data */
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      pool  = data.pools[part];
      np  = distribute[part][0] * distribute[part][1] * distribute[part][2];
      pid = myid - pool_procs[pool];

      if ( (pid < 0) || (pid >= np) )
      {
         /* none of this part data lives on this process */
         pdata.nboxes = 0;
         pdata.glue_nboxes = 0;
         pdata.graph_nboxes = 0;
         pdata.matset_nboxes = 0;
         pdata.matadd_nboxes = 0;
      }
      else
      {
         /* refine boxes */
         m[0] = refine[part][0];
         m[1] = refine[part][1];
         m[2] = refine[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               MapProblemIndex(pdata.ilowers[box], m);
               MapProblemIndex(pdata.iuppers[box], m);
            }

            for (box = 0; box < pdata.graph_nboxes; box++)
            {
               MapProblemIndex(pdata.graph_ilowers[box], m);
               MapProblemIndex(pdata.graph_iuppers[box], m);
               mmap[0] = m[pdata.graph_index_maps[box][0]];
               mmap[1] = m[pdata.graph_index_maps[box][1]];
               mmap[2] = m[pdata.graph_index_maps[box][2]];
               MapProblemIndex(pdata.graph_to_ilowers[box], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[box], mmap);
            }
            for (box = 0; box < pdata.matset_nboxes; box++)
            {
               MapProblemIndex(pdata.matset_ilowers[box], m);
               MapProblemIndex(pdata.matset_iuppers[box], m);
            }
            for (box = 0; box < pdata.matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.matadd_ilowers[box], m);
               MapProblemIndex(pdata.matadd_iuppers[box], m);
            }
         }

         /* refine and distribute boxes */
         m[0] = distribute[part][0];
         m[1] = distribute[part][1];
         m[2] = distribute[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            p = pid % m[0];
            q = ((pid - p) / m[0]) % m[1];
            r = (pid - p - q*m[0]) / (m[0]*m[1]);

            for (box = 0; box < pdata.nboxes; box++)
            {
               n[0] = pdata.iuppers[box][0] - pdata.ilowers[box][0] + 1;
               n[1] = pdata.iuppers[box][1] - pdata.ilowers[box][1] + 1;
               n[2] = pdata.iuppers[box][2] - pdata.ilowers[box][2] + 1;

               MapProblemIndex(pdata.ilowers[box], m);
               MapProblemIndex(pdata.iuppers[box], m);
               pdata.iuppers[box][0] = pdata.ilowers[box][0] + n[0] - 1;
               pdata.iuppers[box][1] = pdata.ilowers[box][1] + n[1] - 1;
               pdata.iuppers[box][2] = pdata.ilowers[box][2] + n[2] - 1;

               pdata.ilowers[box][0] = pdata.ilowers[box][0] + p*n[0];
               pdata.ilowers[box][1] = pdata.ilowers[box][1] + q*n[1];
               pdata.ilowers[box][2] = pdata.ilowers[box][2] + r*n[2];
               pdata.iuppers[box][0] = pdata.iuppers[box][0] + p*n[0];
               pdata.iuppers[box][1] = pdata.iuppers[box][1] + q*n[1];
               pdata.iuppers[box][2] = pdata.iuppers[box][2] + r*n[2];
            }

            i = 0;
            for (box = 0; box < pdata.graph_nboxes; box++)
            {
               MapProblemIndex(pdata.graph_ilowers[box], m);
               MapProblemIndex(pdata.graph_iuppers[box], m);
               mmap[0] = m[pdata.graph_index_maps[box][0]];
               mmap[1] = m[pdata.graph_index_maps[box][1]];
               mmap[2] = m[pdata.graph_index_maps[box][2]];
               MapProblemIndex(pdata.graph_to_ilowers[box], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[box], mmap);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[b], pdata.iuppers[b],
                                 pdata.vartypes[pdata.graph_vars[box]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.graph_ilowers[box],
                                        pdata.graph_iuppers[box],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        dmap = pdata.graph_index_maps[box][d];
                        sign = pdata.graph_index_signs[box][d];
                        pdata.graph_to_ilowers[i][dmap] =
                           pdata.graph_to_ilowers[box][dmap] +
                           sign * pdata.graph_to_strides[box][d] *
                           ((int_ilower[d] - pdata.graph_ilowers[box][d]) /
                            pdata.graph_strides[box][d]);
                        pdata.graph_to_iuppers[i][dmap] =
                           pdata.graph_to_iuppers[box][dmap] +
                           sign * pdata.graph_to_strides[box][d] *
                           ((int_iupper[d] - pdata.graph_iuppers[box][d]) /
                            pdata.graph_strides[box][d]);
                        pdata.graph_ilowers[i][d] = int_ilower[d];
                        pdata.graph_iuppers[i][d] = int_iupper[d];
                        pdata.graph_strides[i][d] =
                           pdata.graph_strides[box][d];
                        pdata.graph_to_strides[i][d] =
                           pdata.graph_to_strides[box][d];
                        pdata.graph_index_maps[i][d]  = dmap;
                        pdata.graph_index_signs[i][d] = sign;
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.graph_ilowers[i][d] =
                           pdata.graph_ilowers[box][d];
                        pdata.graph_iuppers[i][d] =
                           pdata.graph_iuppers[box][d];
                        pdata.graph_to_ilowers[i][d] =
                           pdata.graph_to_ilowers[box][d];
                        pdata.graph_to_iuppers[i][d] =
                           pdata.graph_to_iuppers[box][d];
                     }
                     pdata.graph_vars[i]     = pdata.graph_vars[box];
                     pdata.graph_to_parts[i] = pdata.graph_to_parts[box];
                     pdata.graph_to_vars[i]  = pdata.graph_to_vars[box];
                     pdata.graph_entries[i]  = pdata.graph_entries[box];
                     pdata.graph_values[i]   = pdata.graph_values[box];
                     i++;
                     break;
                  }
               }
            }
            pdata.graph_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.matset_nboxes; box++)
            {
               MapProblemIndex(pdata.matset_ilowers[box], m);
               MapProblemIndex(pdata.matset_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[b], pdata.iuppers[b],
                                 pdata.vartypes[pdata.matset_vars[box]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.matset_ilowers[box],
                                        pdata.matset_iuppers[box],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.matset_ilowers[i][d] = int_ilower[d];
                        pdata.matset_iuppers[i][d] = int_iupper[d];
                        pdata.matset_strides[i][d] =
                           pdata.matset_strides[box][d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.matset_ilowers[i][d] =
                           pdata.matset_ilowers[box][d];
                        pdata.matset_iuppers[i][d] =
                           pdata.matset_iuppers[box][d];
                     }
                     pdata.matset_vars[i]     = pdata.matset_vars[box];
                     pdata.matset_entries[i]  = pdata.matset_entries[box];
                     pdata.matset_values[i]   = pdata.matset_values[box];
                     i++;
                     break;
                  }
               }
            }
            pdata.matset_nboxes = i;

            i = 0;
            for (box = 0; box < pdata.matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.matadd_ilowers[box], m);
               MapProblemIndex(pdata.matadd_iuppers[box], m);

               for (b = 0; b < pdata.nboxes; b++)
               {
                  /* first convert the box extents based on vartype */
                  GetVariableBox(pdata.ilowers[b], pdata.iuppers[b],
                                 pdata.vartypes[pdata.matadd_vars[box]],
                                 ilower, iupper);
                  size = IntersectBoxes(pdata.matadd_ilowers[box],
                                        pdata.matadd_iuppers[box],
                                        ilower, iupper,
                                        int_ilower, int_iupper);
                  if (size > 0)
                  {
                     /* if there is an intersection, it is the only one */
                     for (d = 0; d < 3; d++)
                     {
                        pdata.matadd_ilowers[i][d] = int_ilower[d];
                        pdata.matadd_iuppers[i][d] = int_iupper[d];
                     }
                     for (d = 3; d < 9; d++)
                     {
                        pdata.matadd_ilowers[i][d] =
                           pdata.matadd_ilowers[box][d];
                        pdata.matadd_iuppers[i][d] =
                           pdata.matadd_iuppers[box][d];
                     }
                     pdata.matadd_vars[i]     = pdata.matadd_vars[box];
                     pdata.matadd_entries[i]  = pdata.matadd_entries[box];
                     pdata.matadd_values[i]   = pdata.matadd_values[box];
                     i++;
                     break;
                  }
               }
            }
            pdata.matadd_nboxes = i;
         }

         /* refine and block boxes */
         m[0] = block[part][0];
         m[1] = block[part][1];
         m[2] = block[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            pdata.ilowers = hypre_TReAlloc(pdata.ilowers, ProblemIndex,
                                           m[0]*m[1]*m[2]*pdata.nboxes);
            pdata.iuppers = hypre_TReAlloc(pdata.iuppers, ProblemIndex,
                                           m[0]*m[1]*m[2]*pdata.nboxes);
            pdata.boxsizes = hypre_TReAlloc(pdata.boxsizes, int,
                                            m[0]*m[1]*m[2]*pdata.nboxes);
            for (box = 0; box < pdata.nboxes; box++)
            {
               n[0] = pdata.iuppers[box][0] - pdata.ilowers[box][0] + 1;
               n[1] = pdata.iuppers[box][1] - pdata.ilowers[box][1] + 1;
               n[2] = pdata.iuppers[box][2] - pdata.ilowers[box][2] + 1;

               MapProblemIndex(pdata.ilowers[box], m);

               MapProblemIndex(pdata.iuppers[box], m);
               pdata.iuppers[box][0] = pdata.ilowers[box][0] + n[0] - 1;
               pdata.iuppers[box][1] = pdata.ilowers[box][1] + n[1] - 1;
               pdata.iuppers[box][2] = pdata.ilowers[box][2] + n[2] - 1;

               i = box;
               for (r = 0; r < m[2]; r++)
               {
                  for (q = 0; q < m[1]; q++)
                  {
                     for (p = 0; p < m[0]; p++)
                     {
                        pdata.ilowers[i][0] = pdata.ilowers[box][0] + p*n[0];
                        pdata.ilowers[i][1] = pdata.ilowers[box][1] + q*n[1];
                        pdata.ilowers[i][2] = pdata.ilowers[box][2] + r*n[2];
                        pdata.iuppers[i][0] = pdata.iuppers[box][0] + p*n[0];
                        pdata.iuppers[i][1] = pdata.iuppers[box][1] + q*n[1];
                        pdata.iuppers[i][2] = pdata.iuppers[box][2] + r*n[2];
                        for (d = 3; d < 9; d++)
                        {
                           pdata.ilowers[i][d] = pdata.ilowers[box][d];
                           pdata.iuppers[i][d] = pdata.iuppers[box][d];
                        }
                        i += pdata.nboxes;
                     }
                  }
               }
            }
            pdata.nboxes *= m[0]*m[1]*m[2];

            for (box = 0; box < pdata.graph_nboxes; box++)
            {
               MapProblemIndex(pdata.graph_ilowers[box], m);
               MapProblemIndex(pdata.graph_iuppers[box], m);
               mmap[0] = m[pdata.graph_index_maps[box][0]];
               mmap[1] = m[pdata.graph_index_maps[box][1]];
               mmap[2] = m[pdata.graph_index_maps[box][2]];
               MapProblemIndex(pdata.graph_to_ilowers[box], mmap);
               MapProblemIndex(pdata.graph_to_iuppers[box], mmap);
            }
            for (box = 0; box < pdata.matset_nboxes; box++)
            {
               MapProblemIndex(pdata.matset_ilowers[box], m);
               MapProblemIndex(pdata.matset_iuppers[box], m);
            }
            for (box = 0; box < pdata.matadd_nboxes; box++)
            {
               MapProblemIndex(pdata.matadd_ilowers[box], m);
               MapProblemIndex(pdata.matadd_iuppers[box], m);
            }
         }

         /* map remaining ilowers & iuppers */
         m[0] = refine[part][0] * block[part][0] * distribute[part][0];
         m[1] = refine[part][1] * block[part][1] * distribute[part][1];
         m[2] = refine[part][2] * block[part][2] * distribute[part][2];
         if ( (m[0] * m[1] * m[2]) > 1)
         {
            for (box = 0; box < pdata.glue_nboxes; box++)
            {
               MapProblemIndex(pdata.glue_ilowers[box], m);
               MapProblemIndex(pdata.glue_iuppers[box], m);
               mmap[0] = m[pdata.glue_index_maps[box][0]];
               mmap[1] = m[pdata.glue_index_maps[box][1]];
               mmap[2] = m[pdata.glue_index_maps[box][2]];
               MapProblemIndex(pdata.glue_nbor_ilowers[box], mmap);
               MapProblemIndex(pdata.glue_nbor_iuppers[box], mmap);
            }
         }

         /* compute box sizes, etc. */
         pdata.max_boxsize = 0;
         for (box = 0; box < pdata.nboxes; box++)
         {
            pdata.boxsizes[box] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.boxsizes[box] *=
                  (pdata.iuppers[box][i] - pdata.ilowers[box][i] + 2);
            }
            pdata.max_boxsize =
               hypre_max(pdata.max_boxsize, pdata.boxsizes[box]);
         }
         for (box = 0; box < pdata.graph_nboxes; box++)
         {
            pdata.graph_boxsizes[box] = 1;
            for (i = 0; i < 3; i++)
            {
               pdata.graph_boxsizes[box] *=
                  (pdata.graph_iuppers[box][i] -
                   pdata.graph_ilowers[box][i] + 1);
            }
         }
         for (box = 0; box < pdata.matset_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size*= (pdata.matset_iuppers[box][i] -
                       pdata.matset_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            size = 1;
            for (i = 0; i < 3; i++)
            {
               size*= (pdata.matadd_iuppers[box][i] -
                       pdata.matadd_ilowers[box][i] + 1);
            }
            pdata.max_boxsize = hypre_max(pdata.max_boxsize, size);
         }
      }

      if (pdata.nboxes == 0)
      {
         hypre_TFree(pdata.ilowers);
         hypre_TFree(pdata.iuppers);
         hypre_TFree(pdata.boxsizes);
         pdata.max_boxsize = 0;
      }

      if (pdata.glue_nboxes == 0)
      {
         hypre_TFree(pdata.glue_ilowers);
         hypre_TFree(pdata.glue_iuppers);
         hypre_TFree(pdata.glue_nbor_parts);
         hypre_TFree(pdata.glue_nbor_ilowers);
         hypre_TFree(pdata.glue_nbor_iuppers);
         hypre_TFree(pdata.glue_index_maps);
         hypre_TFree(pdata.glue_primaries);
      }

      if (pdata.graph_nboxes == 0)
      {
         hypre_TFree(pdata.graph_ilowers);
         hypre_TFree(pdata.graph_iuppers);
         hypre_TFree(pdata.graph_strides);
         hypre_TFree(pdata.graph_vars);
         hypre_TFree(pdata.graph_to_parts);
         hypre_TFree(pdata.graph_to_ilowers);
         hypre_TFree(pdata.graph_to_iuppers);
         hypre_TFree(pdata.graph_to_strides);
         hypre_TFree(pdata.graph_to_vars);
         hypre_TFree(pdata.graph_index_maps);
         hypre_TFree(pdata.graph_index_signs);
         hypre_TFree(pdata.graph_entries);
         hypre_TFree(pdata.graph_values);
         hypre_TFree(pdata.graph_boxsizes);
      }

      if (pdata.matset_nboxes == 0)
      {
         hypre_TFree(pdata.matset_ilowers);
         hypre_TFree(pdata.matset_iuppers);
         hypre_TFree(pdata.matset_strides);
         hypre_TFree(pdata.matset_vars);
         hypre_TFree(pdata.matset_entries);
         hypre_TFree(pdata.matset_values);
      }

      if (pdata.matadd_nboxes == 0)
      {
         hypre_TFree(pdata.matadd_ilowers);
         hypre_TFree(pdata.matadd_iuppers);
         hypre_TFree(pdata.matadd_vars);
         hypre_TFree(pdata.matadd_nentries);
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            hypre_TFree(pdata.matadd_entries[box]);
            hypre_TFree(pdata.matadd_values[box]);
         }
         hypre_TFree(pdata.matadd_entries);
         hypre_TFree(pdata.matadd_values);
      }

      data.pdata[part] = pdata;
   }

   data.max_boxsize = 0;
   for (part = 0; part < data.nparts; part++)
   {
      data.max_boxsize =
         hypre_max(data.max_boxsize, data.pdata[part].max_boxsize);
   }

   hypre_TFree(pool_procs);

   *data_ptr = data; 
   return 0;
}

/*--------------------------------------------------------------------------
 * Destroy data
 *--------------------------------------------------------------------------*/

int
DestroyData( ProblemData   data )
{
   ProblemPartData  pdata;
   int              part, box, s;

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];

      if (pdata.nboxes > 0)
      {
         hypre_TFree(pdata.ilowers);
         hypre_TFree(pdata.iuppers);
         hypre_TFree(pdata.boxsizes);
      }

      if (pdata.nvars > 0)
      {
         hypre_TFree(pdata.vartypes);
      }

      if (pdata.add_nvars > 0)
      {
         hypre_TFree(pdata.add_indexes);
         hypre_TFree(pdata.add_vartypes);
      }

      if (pdata.glue_nboxes > 0)
      {
         hypre_TFree(pdata.glue_ilowers);
         hypre_TFree(pdata.glue_iuppers);
         hypre_TFree(pdata.glue_nbor_parts);
         hypre_TFree(pdata.glue_nbor_ilowers);
         hypre_TFree(pdata.glue_nbor_iuppers);
         hypre_TFree(pdata.glue_index_maps);
         hypre_TFree(pdata.glue_primaries);
      }

      if (pdata.nvars > 0)
      {
         hypre_TFree(pdata.stencil_num);
      }

      if (pdata.graph_nboxes > 0)
      {
         hypre_TFree(pdata.graph_ilowers);
         hypre_TFree(pdata.graph_iuppers);
         hypre_TFree(pdata.graph_strides);
         hypre_TFree(pdata.graph_vars);
         hypre_TFree(pdata.graph_to_parts);
         hypre_TFree(pdata.graph_to_ilowers);
         hypre_TFree(pdata.graph_to_iuppers);
         hypre_TFree(pdata.graph_to_strides);
         hypre_TFree(pdata.graph_to_vars);
         hypre_TFree(pdata.graph_index_maps);
         hypre_TFree(pdata.graph_index_signs);
         hypre_TFree(pdata.graph_entries);
         hypre_TFree(pdata.graph_values);
         hypre_TFree(pdata.graph_boxsizes);
      }

      if (pdata.matset_nboxes > 0)
      {
         hypre_TFree(pdata.matset_ilowers);
         hypre_TFree(pdata.matset_iuppers);
         hypre_TFree(pdata.matset_strides);
         hypre_TFree(pdata.matset_vars);
         hypre_TFree(pdata.matset_entries);
         hypre_TFree(pdata.matset_values);
      }

      if (pdata.matadd_nboxes > 0)
      {
         hypre_TFree(pdata.matadd_ilowers);
         hypre_TFree(pdata.matadd_iuppers);
         hypre_TFree(pdata.matadd_vars);
         hypre_TFree(pdata.matadd_nentries);
         for (box = 0; box < pdata.matadd_nboxes; box++)
         {
            hypre_TFree(pdata.matadd_entries[box]);
            hypre_TFree(pdata.matadd_values[box]);
         }
         hypre_TFree(pdata.matadd_entries);
         hypre_TFree(pdata.matadd_values);
      }
   }
   hypre_TFree(data.pdata);

   for (s = 0; s < data.nstencils; s++)
   {
      hypre_TFree(data.stencil_offsets[s]);
      hypre_TFree(data.stencil_vars[s]);
      hypre_TFree(data.stencil_values[s]);
   }
   hypre_TFree(data.stencil_sizes);
   hypre_TFree(data.stencil_offsets);
   hypre_TFree(data.stencil_vars);
   hypre_TFree(data.stencil_values);

   if (data.symmetric_num > 0)
   {
      hypre_TFree(data.symmetric_parts);
      hypre_TFree(data.symmetric_vars);
      hypre_TFree(data.symmetric_to_vars);
      hypre_TFree(data.symmetric_booleans);
   }

   hypre_TFree(data.pools);

   return 0;
}

/*--------------------------------------------------------------------------
 * Routine to load cosine function
 *--------------------------------------------------------------------------*/

int
SetCosineVector(   double  scale,
                   Index   ilower,
                   Index   iupper,
                   double *values)
{
   int          i, j, k;
   int          count = 0;

   for (k = ilower[2]; k <= iupper[2]; k++)
   {
      for (j = ilower[1]; j <= iupper[1]; j++)
      {
         for (i = ilower[0]; i <= iupper[0]; i++)
         {
            values[count] = scale * cos((i+j+k)/10.0);
            count++;
         }
      }
   }

   return(0);
}

/*--------------------------------------------------------------------------
 * Print usage info
 *--------------------------------------------------------------------------*/

int
PrintUsage( char *progname,
            int   myid )
{
   if ( myid == 0 )
   {
      printf("\n");
      printf("Usage: %s [<options>]\n", progname);
      printf("\n");
      printf("  -in <filename> : input file (default is `%s').\n", infile_default);
      printf("      NOTE: -in must come become before parameters that modify the problem (-P, etc.).\n");
      printf("\n");
      printf("  -P <Px> <Py> <Pz>   : define processor topology\n\n");
      printf("  -pooldist <p>       : pool distribution to use\n");
      printf("  -r <rx> <ry> <rz>   : refine part(s) for default problem\n");
      printf("  -b <bx> <by> <bz>   : refine and block part(s) for default problem\n\n");
      printf("  -n <nx> <ny> <nz>   : define size per processor for problems on cube\n");
      printf("  -c <cx> <cy> <cz>   : define anisotropies for Laplace problem\n\n");
      printf("  -laplace            : 3D Laplace problem on a cube\n");
      printf("  -27pt               : Problem with 27-point stencil on a cube\n");
      printf("  -9pt                : build 9pt 2D laplacian problem\n");
      printf("  -jumps              : PDE with jumps on a cube\n\n");
      printf("  -solver <ID>        : solver ID (default = 0)\n");
      printf("                        0 - PCG with AMG (low complexity) precond\n");
      printf("                        1 - PCG with diagonal scaling\n");
      printf("                        2 - GMRES(10) with AMG (low complexity) precond\n");
      printf("                        3 - GMRES(10) with diagonal scaling\n\n");
      printf("  -printstats         : print out detailed info on AMG preconditioner\n\n");
      printf("  -printsystem        : print out the system\n\n");
      printf("  -rhsfromcosine      : solution is cosine function (default), can be used for\n");
      printf("                        default problem only\n");
      printf("  -rhsone             : rhs is vector with unit components \n");
      printf("\n");
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Test driver for semi-structured matrix interface
 *--------------------------------------------------------------------------*/
 
int
main( int   argc,
      char *argv[] )
{
   char                 *infile;
   ProblemData           global_data;
   ProblemData           data;
   ProblemPartData       pdata;
   int                   nparts;
   int                   pooldist;
   int                  *parts;
   Index                *refine;
   Index                *distribute;
   Index                *block;
   int                   object_type;
   int                   solver_id = 0 ;
   int                   print_system = 0;
   int                   print_level = 0;
   int                   cosine, struct_cosine;
   double                scale;
                        
   HYPRE_SStructGrid     grid;
   HYPRE_SStructStencil *stencils;
   HYPRE_SStructGraph    graph;
   HYPRE_SStructMatrix   A;
   HYPRE_SStructVector   b;
   HYPRE_SStructVector   x;

   HYPRE_ParCSRMatrix    par_A;
   HYPRE_ParVector       par_b;
   HYPRE_ParVector       par_x;
   HYPRE_Solver          par_solver;
   HYPRE_Solver          par_precond;


   Index                 ilower, iupper;
   Index                 index, to_index;
   double               *values;

   int                   num_iterations;
   double                final_res_norm;
                         
   int                   num_procs, myid;
   int                   time_index;
   double                wall_time;

   int                   sym;

   int                   arg_index, part, var, box, s, entry, i, j, k, size;
   int                   build_matrix_type;
   int                   build_rhs_type;
   
   double                system_size;
   int		        *P_xyz, *r_xyz;
    
   /*int    max_levels = 25;
   int    num_sweeps = 1;
   int    ns_coarse = 1;
   int    verbose = 0; */
   double  tol = 1.e-6;
   int    maxit_prec = 100;
   int    maxit_sol = 500;
   /*int coarse_threshold = 9;
   int seq_threshold = 0;
   

   int     mc_err;
   
   int    cheby_order = 2;
   double cheby_eig_ratio = .3; */
   
   int wall = 0;
   

   int pde_index, mesh_index, discr_index;
   int sref, pref, vis, spm;

  

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   

   hypre_InitMemoryDebug(myid);

   /* afem defaults */
   mesh_index = pde_index =discr_index = 0;
   sref = pref = vis = 0;
   spm = 4; /* This used to be 2 - changed to be a "more stable" option */
   

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
                                                                                               
   arg_index = 1;
   build_matrix_type = 1;
   build_rhs_type = -1;
   
   while ( (arg_index < argc))
   {
      if ( strcmp(argv[arg_index], "-laplace") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
      }
      
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 3;
      }
      else if ( strcmp(argv[arg_index], "-jumps") == 0 )
      {
         arg_index++;
         build_matrix_type      = 4;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
	 solver_id = atoi(argv[arg_index++]);
	 if (solver_id < 0 || solver_id > 3) 
	 {
	    printf("Wrong solver id ! Code will exit!!\n");
	    return(1);
         }
      }
      else if ( strcmp(argv[arg_index], "-printsystem") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-printstats") == 0 )
      {
         arg_index++;
         print_level = 1;
      }
      else
         arg_index++;
   }
   
   if (build_matrix_type > 1 && build_matrix_type < 8)
   {
      time_index = hypre_InitializeTiming("Setup matrix and rhs");
      hypre_BeginTiming(time_index);

      if ( build_matrix_type == 2 )
         BuildParLaplacian(argc, argv, &system_size, &par_A, &par_b, &par_x);
      else if ( build_matrix_type == 3 )
         BuildParLaplacian27pt(argc, argv, &system_size, &par_A, &par_b, &par_x);
      else if ( build_matrix_type == 4 )
         BuildParVarDifConv(argc, argv, &system_size, &par_A, &par_b, &par_x);
      hypre_EndTiming(time_index);
      hypre_PrintTiming("Setup matrix and rhs", &wall_time, MPI_COMM_WORLD);
      hypre_FinalizeTiming(time_index);
      hypre_ClearTiming();
      fflush(NULL);

      if (print_system)
      {
         HYPRE_ParCSRMatrixPrintIJ(par_A, 0, 0, "parcsr.out.A");
         HYPRE_ParVectorPrintIJ(par_b, 0, "parcsr.out.b");
         HYPRE_ParVectorPrintIJ(par_x, 0, "parcsr.out.x0");
      }
   }
   else 
   {
   /*-----------------------------------------------------------
    * Read input file
    *-----------------------------------------------------------*/

   arg_index = 1;

   /* parse command line for input file name */
   infile = infile_default;
   if (argc > 1)
   {
      if ( strcmp(argv[arg_index], "-in") == 0 )
      {
         arg_index++;
         infile = argv[arg_index++];
      }
   }

   ReadData(infile, &global_data);

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   sym   = 1;

   nparts = global_data.nparts;
   pooldist = 0;

   parts      = hypre_TAlloc(int, nparts);
   refine     = hypre_TAlloc(Index, nparts);
   distribute = hypre_TAlloc(Index, nparts);
   block      = hypre_TAlloc(Index, nparts);
   P_xyz      = hypre_TAlloc(int, 3);
   r_xyz      = hypre_TAlloc(int, 3);
   for (part = 0; part < nparts; part++)
   {
      parts[part] = part;
      for (j = 0; j < 3; j++)
      {
         refine[part][j]     = 1;
         distribute[part][j] = 1;
         block[part][j]      = 1;
      }
   }

   print_system = 0;
   cosine = 1;
   struct_cosine = 0;
   system_size = 512.;
   for (j = 0; j < 3; j++)
   {
      r_xyz[j] = 1;
      P_xyz[j] = 1;
   }
   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   while (arg_index < argc)
   {
      /*if ( strcmp(argv[arg_index], "-pt") == 0 )
      {
         arg_index++;
         nparts = 0;
         while ( strncmp(argv[arg_index], "-", 1) != 0 )
         {
            parts[nparts++] = atoi(argv[arg_index++]);
         }
      }
      else*/ if ( strcmp(argv[arg_index], "-pooldist") == 0 )
      {
         arg_index++;
         pooldist = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-r") == 0 )
      {
         arg_index++;
         for (j = 0; j < 3; j++)
            r_xyz[j] = atoi(argv[arg_index++]);
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            for (j = 0; j < 3; j++)
               refine[part][j] = r_xyz[j];
         }
         system_size *= (double)r_xyz[0]*(double)r_xyz[1]*(double)r_xyz[2];
         hypre_TFree(r_xyz);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         for (j = 0; j < 3; j++)
            P_xyz[j] = atoi(argv[arg_index++]);
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            for (j = 0; j < 3; j++)
               distribute[part][j] = P_xyz[j];
         }
         system_size *= (double)P_xyz[0]*(double)P_xyz[1]*(double)P_xyz[2];
         hypre_TFree(P_xyz);
      }
      else if ( strcmp(argv[arg_index], "-b") == 0 )
      {
         arg_index++;
         for (i = 0; i < nparts; i++)
         {
            part = parts[i];
            k = arg_index;
            for (j = 0; j < 3; j++)
            {
               block[part][j] = atoi(argv[k++]);
            }
         }
         arg_index += 3;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
	 solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rhsone") == 0 )
      {
         arg_index++;
         cosine = 0;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromcosine") == 0 )
      {
         arg_index++;
         cosine = 1;
         struct_cosine = 1;
      }
      else if ( strcmp(argv[arg_index], "-printsystem") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         PrintUsage(argv[0], myid);
         exit(1);
         break;
      }
      else
         arg_index++;
   }

   /*-----------------------------------------------------------
    * Distribute data
    *-----------------------------------------------------------*/

   DistributeData(global_data, pooldist, refine, distribute, block,
                  num_procs, myid, &data);

   /*-----------------------------------------------------------
    * Synchronize so that timings make sense
    *-----------------------------------------------------------*/

   MPI_Barrier(MPI_COMM_WORLD);

   /*-----------------------------------------------------------
    * Set up the grid
    *-----------------------------------------------------------*/

   time_index = hypre_InitializeTiming("SStruct Interface");
   hypre_BeginTiming(time_index);

   HYPRE_SStructGridCreate(MPI_COMM_WORLD, data.ndim, data.nparts, &grid);
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      for (box = 0; box < pdata.nboxes; box++)
      {
         HYPRE_SStructGridSetExtents(grid, part,
                                     pdata.ilowers[box], pdata.iuppers[box]);
      }

      HYPRE_SStructGridSetVariables(grid, part, pdata.nvars, pdata.vartypes);

      /* GridAddVariabes */

      /* GridSetNeighborBox */
      for (box = 0; box < pdata.glue_nboxes; box++)
      {
#if 1 /* will add primary to the interface soon */
         HYPRE_SStructGridSetNeighborBox(grid, part,
                                         pdata.glue_ilowers[box],
                                         pdata.glue_iuppers[box],
                                         pdata.glue_nbor_parts[box],
                                         pdata.glue_nbor_ilowers[box],
                                         pdata.glue_nbor_iuppers[box],
                                         pdata.glue_index_maps[box]);
#else
         HYPRE_SStructGridSetNeighborBoxZ(grid, part,
                                          pdata.glue_ilowers[box],
                                          pdata.glue_iuppers[box],
                                          pdata.glue_nbor_parts[box],
                                          pdata.glue_nbor_ilowers[box],
                                          pdata.glue_nbor_iuppers[box],
                                          pdata.glue_index_maps[box],
                                          pdata.glue_primaries[box]);
#endif
      }

      HYPRE_SStructGridSetPeriodic(grid, part, pdata.periodic);
   }
   HYPRE_SStructGridAssemble(grid);

   /*-----------------------------------------------------------
    * Set up the stencils
    *-----------------------------------------------------------*/

   stencils = hypre_CTAlloc(HYPRE_SStructStencil, data.nstencils);
   for (s = 0; s < data.nstencils; s++)
   {
      HYPRE_SStructStencilCreate(data.ndim, data.stencil_sizes[s],
                                 &stencils[s]);
      for (entry = 0; entry < data.stencil_sizes[s]; entry++)
      {
         HYPRE_SStructStencilSetEntry(stencils[s], entry,
                                      data.stencil_offsets[s][entry],
                                      data.stencil_vars[s][entry]);
      }
   }

   /*-----------------------------------------------------------
    * Set object type
    *-----------------------------------------------------------*/

   object_type = HYPRE_PARCSR;  

   /*-----------------------------------------------------------
    * Set up the graph
    *-----------------------------------------------------------*/

   HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);

   /* HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
   if ( object_type != HYPRE_SSTRUCT )
   {
       HYPRE_SStructGraphSetObjectType(graph, object_type);
   }

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];

      /* set stencils */
      for (var = 0; var < pdata.nvars; var++)
      {
         HYPRE_SStructGraphSetStencil(graph, part, var,
                                      stencils[pdata.stencil_num[var]]);
      }

      /* add entries */
      for (box = 0; box < pdata.graph_nboxes; box++)
      {
         for (index[2] = pdata.graph_ilowers[box][2];
              index[2] <= pdata.graph_iuppers[box][2];
              index[2] += pdata.graph_strides[box][2])
         {
            for (index[1] = pdata.graph_ilowers[box][1];
                 index[1] <= pdata.graph_iuppers[box][1];
                 index[1] += pdata.graph_strides[box][1])
            {
               for (index[0] = pdata.graph_ilowers[box][0];
                    index[0] <= pdata.graph_iuppers[box][0];
                    index[0] += pdata.graph_strides[box][0])
               {
                  for (i = 0; i < 3; i++)
                  {
                     j = pdata.graph_index_maps[box][i];
                     k = index[i] - pdata.graph_ilowers[box][i];
                     k /= pdata.graph_strides[box][i];
                     k *= pdata.graph_index_signs[box][i];
                     to_index[j] = pdata.graph_to_ilowers[box][j] +
                        k * pdata.graph_to_strides[box][j];
                  }
                  HYPRE_SStructGraphAddEntries(graph, part, index,
                                               pdata.graph_vars[box],
                                               pdata.graph_to_parts[box],
                                               to_index,
                                               pdata.graph_to_vars[box]);
               }
            }
         }
      }
   }

   HYPRE_SStructGraphAssemble(graph);

   /*-----------------------------------------------------------
    * Set up the matrix
    *-----------------------------------------------------------*/

   values = hypre_TAlloc(double, data.max_boxsize);

   HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);

   /* TODO HYPRE_SStructMatrixSetSymmetric(A, 1); */
   for (i = 0; i < data.symmetric_num; i++)
   {
      HYPRE_SStructMatrixSetSymmetric(A, data.symmetric_parts[i],
                                      data.symmetric_vars[i],
                                      data.symmetric_to_vars[i],
                                      data.symmetric_booleans[i]);
   }
   HYPRE_SStructMatrixSetNSSymmetric(A, data.ns_symmetric);

   /* HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
   if ( object_type != HYPRE_SSTRUCT )
   {
       HYPRE_SStructMatrixSetObjectType(A, object_type);
   }

   HYPRE_SStructMatrixInitialize(A);

   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];

      /* set stencil values */
      for (var = 0; var < pdata.nvars; var++)
      {
         s = pdata.stencil_num[var];
         for (i = 0; i < data.stencil_sizes[s]; i++)
         {
            for (j = 0; j < pdata.max_boxsize; j++)
            {
               values[j] = data.stencil_values[s][i];
            }
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                              pdata.vartypes[var], ilower, iupper);
               HYPRE_SStructMatrixSetBoxValues(A, part, ilower, iupper,
                                               var, 1, &i, values);
            }
         }
      }

      /* set non-stencil entries */
      for (box = 0; box < pdata.graph_nboxes; box++)
      {
         /*
          * RDF NOTE: Add a separate interface routine for setting non-stencil
          * entries.  It would be more efficient to set boundary values a box
          * at a time, but AMR may require striding, and some codes may already
          * have a natural values array to pass in, but can't because it uses
          * ghost values.
          *
          * Example new interface routine:
          *   SetNSBoxValues(matrix, part, ilower, iupper, stride, entry
          *                  values_ilower, values_iupper, values);
          */

/* since we have already tested SetBoxValues above, use SetValues here */
#if 0
         for (j = 0; j < pdata.graph_boxsizes[box]; j++)
         {
            values[j] = pdata.graph_values[box];
         }
         HYPRE_SStructMatrixSetBoxValues(A, part,
                                         pdata.graph_ilowers[box],
                                         pdata.graph_iuppers[box],
                                         pdata.graph_vars[box],
                                         1, &pdata.graph_entries[box],
                                         values);
#else
         for (index[2] = pdata.graph_ilowers[box][2];
              index[2] <= pdata.graph_iuppers[box][2];
              index[2] += pdata.graph_strides[box][2])
         {
            for (index[1] = pdata.graph_ilowers[box][1];
                 index[1] <= pdata.graph_iuppers[box][1];
                 index[1] += pdata.graph_strides[box][1])
            {
               for (index[0] = pdata.graph_ilowers[box][0];
                    index[0] <= pdata.graph_iuppers[box][0];
                    index[0] += pdata.graph_strides[box][0])
               {
                  HYPRE_SStructMatrixSetValues(A, part, index,
                                               pdata.graph_vars[box],
                                               1, &pdata.graph_entries[box],
                                               &pdata.graph_values[box]);
               }
            }
         }
#endif
      }

      /* reset some matrix values */
      for (box = 0; box < pdata.matset_nboxes; box++)
      {
         size= 1;
         for (j = 0; j < 3; j++)
         {
            size*= (pdata.matset_iuppers[box][j] -
                    pdata.matset_ilowers[box][j] + 1);
         }
         for (j = 0; j < size; j++)
         {
            values[j] = pdata.matset_values[box];
         }
         HYPRE_SStructMatrixSetBoxValues(A, part,
                                         pdata.matset_ilowers[box],
                                         pdata.matset_iuppers[box],
                                         pdata.matset_vars[box],
                                         1, &pdata.matset_entries[box],
                                         values);
      }

      /* add to some matrix values */
      for (box = 0; box < pdata.matadd_nboxes; box++)
      {
         size = 1;
         for (j = 0; j < 3; j++)
         {
            size*= (pdata.matadd_iuppers[box][j] -
                    pdata.matadd_ilowers[box][j] + 1);
         }

         for (entry = 0; entry < pdata.matadd_nentries[box]; entry++)
         {
            for (j = 0; j < size; j++)
            {
               values[j] = pdata.matadd_values[box][entry];
            }
          
            HYPRE_SStructMatrixAddToBoxValues(A, part, 
                                              pdata.matadd_ilowers[box],
                                              pdata.matadd_iuppers[box],
                                              pdata.matadd_vars[box],
                                              1, &pdata.matadd_entries[box][entry],
                                              values);
         }
      }
   }

   HYPRE_SStructMatrixAssemble(A);

   /*-----------------------------------------------------------
    * Set up the linear system
    *-----------------------------------------------------------*/

   HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);

   /* HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
   if ( object_type != HYPRE_SSTRUCT )
   {
       HYPRE_SStructVectorSetObjectType(b, object_type);
   }

   HYPRE_SStructVectorInitialize(b);
   for (j = 0; j < data.max_boxsize; j++)
   {
      values[j] = 1.0;
   }
   for (part = 0; part < data.nparts; part++)
   {
      pdata = data.pdata[part];
      for (var = 0; var < pdata.nvars; var++)
      {
         for (box = 0; box < pdata.nboxes; box++)
         {
            GetVariableBox(pdata.ilowers[box], pdata.iuppers[box],
                           pdata.vartypes[var], ilower, iupper);
            HYPRE_SStructVectorSetBoxValues(b, part, ilower, iupper,
                                            var, values);
         }
      }
   }
   HYPRE_SStructVectorAssemble(b);

   HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);

   /* HYPRE_SSTRUCT is the default, so we don't have to call SetObjectType */
   if ( object_type != HYPRE_SSTRUCT )
   {
       HYPRE_SStructVectorSetObjectType(x, object_type);
   }

   HYPRE_SStructVectorInitialize(x);
   /*-----------------------------------------------------------
    * If requested, reset linear system so that it has
    * exact solution:
    *
    *   u(part,var,i,j,k) = (part+1)*(var+1)*cosine[(i+j+k)/10]
    * 
    *-----------------------------------------------------------*/

   if (cosine)
   {
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            scale = (part+1.0)*(var+1.0);
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                              ilower, iupper);
               SetCosineVector(scale, ilower, iupper, values);
               HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper,
                                               var, values);
            }
         }
      }
   }
   HYPRE_SStructVectorAssemble(x);

   hypre_EndTiming(time_index);
   hypre_PrintTiming("SStruct Interface", &wall_time, MPI_COMM_WORLD);
   hypre_FinalizeTiming(time_index);
   hypre_ClearTiming();
   fflush(NULL);

   /*-----------------------------------------------------------
    * If requested, reset linear system so that it has
    * exact solution:
    *
    *   u(part,var,i,j,k) = (part+1)*(var+1)*cosine[(i+j+k)/10]
    * 
    *-----------------------------------------------------------*/

   if (cosine)
   {
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            scale = (part+1.0)*(var+1.0);
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                              ilower, iupper);
               SetCosineVector(scale, ilower, iupper, values);
               HYPRE_SStructVectorSetBoxValues(x, part, ilower, iupper,
                                               var, values);
            }
         }
      }
   }

   /*-----------------------------------------------------------
    * Get the objects out
    * NOTE: This should go after the cosine part, but for the bug
    *-----------------------------------------------------------*/

   HYPRE_SStructMatrixGetObject(A, (void **) &par_A);
   HYPRE_SStructVectorGetObject(b, (void **) &par_b);
   HYPRE_SStructVectorGetObject(x, (void **) &par_x);

   /*-----------------------------------------------------------
    * Finish resetting the linear system
    *-----------------------------------------------------------*/

   if (cosine)
   {
      /* This if/else is due to a bug in SStructMatvec */
      if (object_type != HYPRE_PARCSR)
      {
         HYPRE_SStructVectorAssemble(x);
         /* Apply A to cosine vector to yield righthand side */
         hypre_SStructMatvec(1.0, A, x, 0.0, b);
         /* Reset initial guess to zero */
         hypre_SStructMatvec(0.0, A, b, 0.0, x);
      }
      else
      {
         /* Apply A to cosine vector to yield righthand side */
         HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 0.0, par_b );
         /* Reset initial guess to zero */
         HYPRE_ParCSRMatrixMatvec(0.0, par_A, par_b, 0.0, par_x );
      }
   }

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/
   /*hypre_SStructMatvec(1.0, A, x, 2.0, b);
   HYPRE_ParCSRMatrixMatvec(1.0, par_A, par_x, 2.0, par_b );*/
                                                                                                               
   if (print_system)
   {
      HYPRE_SStructVectorGather(b);
      HYPRE_SStructVectorGather(x);
      HYPRE_SStructMatrixPrint("sstruct.out.A",  A, 0);
      HYPRE_SStructVectorPrint("sstruct.out.b",  b, 0);
      HYPRE_SStructVectorPrint("sstruct.out.x0", x, 0);
   }

   /*-----------------------------------------------------------
    * Debugging code
    *-----------------------------------------------------------*/

#if DEBUG
   {
      FILE *file;
      char  filename[255];
                       
      /* result is 1's on the interior of the grid */
      hypre_SStructMatvec(1.0, A, b, 0.0, x);
      HYPRE_SStructVectorPrint("sstruct.out.matvec", x, 0);

      /* result is all 1's */
      hypre_SStructCopy(b, x);
      HYPRE_SStructVectorPrint("sstruct.out.copy", x, 0);

      /* result is all 2's */
      hypre_SStructScale(2.0, x);
      HYPRE_SStructVectorPrint("sstruct.out.scale", x, 0);

      /* result is all 0's */
      hypre_SStructAxpy(-2.0, b, x);
      HYPRE_SStructVectorPrint("sstruct.out.axpy", x, 0);

      /* result is 1's with 0's on some boundaries */
      hypre_SStructCopy(b, x);
      sprintf(filename, "sstruct.out.gatherpre.%05d", myid);
      file = fopen(filename, "w");
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                              ilower, iupper);
               HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               fprintf(file, "\nPart %d, var %d, box %d:\n", part, var, box);
               for (i = 0; i < pdata.boxsizes[box]; i++)
               {
                  fprintf(file, "%e\n", values[i]);
               }
            }
         }
      }
      fclose(file);

      /* result is all 1's */
      HYPRE_SStructVectorGather(x);
      sprintf(filename, "sstruct.out.gatherpost.%05d", myid);
      file = fopen(filename, "w");
      for (part = 0; part < data.nparts; part++)
      {
         pdata = data.pdata[part];
         for (var = 0; var < pdata.nvars; var++)
         {
            for (box = 0; box < pdata.nboxes; box++)
            {
               GetVariableBox(pdata.ilowers[box], pdata.iuppers[box], var,
                              ilower, iupper);
               HYPRE_SStructVectorGetBoxValues(x, part, ilower, iupper,
                                               var, values);
               fprintf(file, "\nPart %d, var %d, box %d:\n", part, var, box);
               for (i = 0; i < pdata.boxsizes[box]; i++)
               {
                  fprintf(file, "%e\n", values[i]);
               }
            }
         }
      }

      /* re-initializes x to 0 */
      hypre_SStructAxpy(-1.0, b, x);
   }
#endif

   hypre_TFree(values);
   }

      /*-----------------------------------------------------------
       * Solve the system using ParCSR version of PCG
       *-----------------------------------------------------------*/
      
      if ((solver_id > -1) && (solver_id < 2))
      {
         time_index = hypre_InitializeTiming("PCG Setup");
         hypre_BeginTiming(time_index);
         
         HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &par_solver);
         HYPRE_PCGSetTol( par_solver, tol );
         HYPRE_PCGSetTwoNorm( par_solver, 1 );
         HYPRE_PCGSetRelChange( par_solver, 0 );
         HYPRE_PCGSetPrintLevel( par_solver, 0 );
         
         if (solver_id == 0)
         {
            /* use BoomerAMG as preconditioner */
            HYPRE_PCGSetMaxIter( par_solver, maxit_prec);
            HYPRE_BoomerAMGCreate(&par_precond); 
            HYPRE_BoomerAMGSetCoarsenType(par_precond, 10);
            HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
            HYPRE_BoomerAMGSetAggNumLevels(par_precond, 1);
            HYPRE_BoomerAMGSetInterpType(par_precond, 6);
            HYPRE_BoomerAMGSetPMaxElmts(par_precond, 4);
            HYPRE_BoomerAMGSetTol(par_precond, 0.0);
            HYPRE_BoomerAMGSetRelaxType(par_precond, 8);
            HYPRE_BoomerAMGSetCycleRelaxType(par_precond, 8, 3);
            HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, 1, 3);
            HYPRE_BoomerAMGSetRelaxOrder(par_precond, 0);
            HYPRE_BoomerAMGSetPrintLevel(par_precond, print_level);
            HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
            HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
            HYPRE_PCGSetPrecond( par_solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                 (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                 par_precond );
         }
         else if (solver_id == 1)
         {
            /* use diagonal scaling as preconditioner */
            HYPRE_PCGSetMaxIter( par_solver, maxit_sol);
            par_precond = NULL;
            HYPRE_PCGSetPrecond(  par_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                  par_precond );
         }

         HYPRE_PCGSetup( par_solver, (HYPRE_Matrix) par_A,
                         (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );
         
         hypre_EndTiming(time_index);

         hypre_PrintTiming("Setup phase times", &wall_time, MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
         fflush(NULL);
         
         if (myid == 0)
	    printf ("\nSystem Size / Setup Phase Time: %e\n\n", (system_size/ wall_time));
 
         time_index = hypre_InitializeTiming("PCG Solve");
         hypre_BeginTiming(time_index);
         
         HYPRE_PCGSolve( par_solver, (HYPRE_Matrix) par_A,
                         (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );
         
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", &wall_time, MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
         fflush(NULL);
         
         HYPRE_PCGGetNumIterations( par_solver, &num_iterations );
         HYPRE_PCGGetFinalRelativeResidualNorm( par_solver, &final_res_norm );
         HYPRE_ParCSRPCGDestroy(par_solver);
         
         if (solver_id == 0)
         {
            HYPRE_BoomerAMGDestroy(par_precond);
         }
      }
      /*-----------------------------------------------------------
       * Solve the system using GMRES
       *-----------------------------------------------------------*/
      
      /*-----------------------------------------------------------
       * Solve the system using ParCSR version of GMRES
       *-----------------------------------------------------------*/
      
      if ((solver_id > 1) && (solver_id < 4))
      {
         time_index = hypre_InitializeTiming("GMRES Setup");
         hypre_BeginTiming(time_index);
         
         HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &par_solver);
         HYPRE_GMRESSetKDim(par_solver, 10);
         HYPRE_GMRESSetTol(par_solver, tol);
         HYPRE_GMRESSetPrintLevel(par_solver, 0);
         HYPRE_GMRESSetLogging(par_solver, 1);

         if (solver_id == 2)
         {
            /* use BoomerAMG as preconditioner */
            HYPRE_GMRESSetMaxIter(par_solver, maxit_prec);
            HYPRE_BoomerAMGCreate(&par_precond); 
            HYPRE_BoomerAMGSetCoarsenType(par_precond, 10);
            HYPRE_BoomerAMGSetStrongThreshold(par_precond, 0.25);
            HYPRE_BoomerAMGSetAggNumLevels(par_precond, 1);
            HYPRE_BoomerAMGSetInterpType(par_precond, 6);
            HYPRE_BoomerAMGSetPMaxElmts(par_precond, 4);
            HYPRE_BoomerAMGSetTol(par_precond, 0.0);
            HYPRE_BoomerAMGSetRelaxOrder(par_precond, 0);
            HYPRE_BoomerAMGSetRelaxType(par_precond, 8);
            HYPRE_BoomerAMGSetCycleRelaxType(par_precond, 8, 3);
            HYPRE_BoomerAMGSetCycleNumSweeps(par_precond, 1, 3);
            HYPRE_BoomerAMGSetPrintLevel(par_precond, print_level);
            HYPRE_BoomerAMGSetPrintFileName(par_precond, "sstruct.out.log");
            HYPRE_BoomerAMGSetMaxIter(par_precond, 1);
            HYPRE_GMRESSetPrecond( par_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                   (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                   par_precond );
         }
         else if (solver_id == 3)
         {
            /* use diagonal scaling as preconditioner */
            HYPRE_GMRESSetMaxIter(par_solver, maxit_sol);
            par_precond = NULL;
            HYPRE_GMRESSetPrecond(  par_solver,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                    par_precond );
         }
         
         HYPRE_GMRESSetup( par_solver, (HYPRE_Matrix) par_A,
                           (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);
         
         hypre_EndTiming(time_index);


         hypre_PrintTiming("Setup phase times", &wall_time, MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
         fflush(NULL);

         if (myid == 0)
	    printf ("\nSystem Size / Setup Phase Time: %e\n\n", (system_size/ wall_time));
         
         time_index = hypre_InitializeTiming("GMRES Solve");
         hypre_BeginTiming(time_index);
         
         HYPRE_GMRESSolve( par_solver, (HYPRE_Matrix) par_A,
                           (HYPRE_Vector) par_b, (HYPRE_Vector) par_x);
         
         hypre_EndTiming(time_index);
         hypre_PrintTiming("Solve phase times", &wall_time, MPI_COMM_WORLD);
         hypre_FinalizeTiming(time_index);
         hypre_ClearTiming();
         fflush(NULL);
         
         HYPRE_GMRESGetNumIterations( par_solver, &num_iterations);
         HYPRE_GMRESGetFinalRelativeResidualNorm( par_solver,
                                                  &final_res_norm);
         HYPRE_ParCSRGMRESDestroy(par_solver);
         
         if (solver_id == 2)
         {
            HYPRE_BoomerAMGDestroy(par_precond);
         }
      }
      
      /*-----------------------------------------------------------
       * Gather the solution vector
       *-----------------------------------------------------------*/
      
      if (build_matrix_type == 1)
      {
         HYPRE_SStructVectorGather(x);
         
         /*-----------------------------------------------------------
          * Print the solution and other info
          *-----------------------------------------------------------*/
         
         if (print_system)
         {
            HYPRE_SStructVectorPrint("sstruct.out.x", x, 0);
         }
      }
     
      if (myid == 0 )
      {
         printf("\n");
         printf("AMG2013 Benchmark version 1.0\n");
         printf("Iterations = %d\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf ("\nSystem Size * Iterations / Solve Phase Time: %e\n",
                 (system_size*(double)num_iterations/ wall_time));
         if (wall)
            printf ("\nSolve Time: %e\n", wall_time);
         printf("\n");
      }
      
   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/
   if (build_matrix_type == 1)
   {
      HYPRE_SStructGridDestroy(grid);
      for (s = 0; s < data.nstencils; s++)
      {
         HYPRE_SStructStencilDestroy(stencils[s]);
      }
      hypre_TFree(stencils);
      HYPRE_SStructGraphDestroy(graph);
      HYPRE_SStructMatrixDestroy(A);
      HYPRE_SStructVectorDestroy(b);
      HYPRE_SStructVectorDestroy(x);
   
   
      DestroyData(data);
   
      hypre_TFree(parts);
      hypre_TFree(refine);
      hypre_TFree(distribute);
      hypre_TFree(block);
   }
   else
   {
      HYPRE_ParCSRMatrixDestroy(par_A);
      HYPRE_ParVectorDestroy(par_b);
      HYPRE_ParVectorDestroy(par_x); 
   }

   hypre_FinalizeMemoryDebug();


   /* Finalize MPI */
   MPI_Finalize();


   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParLaplacian( int                  argc,
                   char                *argv[],
                   double              *system_size_ptr,
                   HYPRE_ParCSRMatrix  *A_ptr,
                   HYPRE_ParVector  *rhs_ptr,
                   HYPRE_ParVector  *x_ptr)
{
   int                 nx, ny, nz;
   HYPRE_BigInt        g_nx, g_ny, g_nz;
   int                 P, Q, R;
   double              cx, cy, cz;

   HYPRE_ParCSRMatrix  A;
   HYPRE_ParVector rhs;
   HYPRE_ParVector x;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;

   int                 arg_index;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
   g_nx = 0;
   g_ny = 0;
   g_nz = 0;

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   g_nx = nx*P;
   g_ny = ny*Q;
   g_nz = nz*R;

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  3D 7-point Laplace problem on a cube\n");
#ifdef HYPRE_LONG_LONG
      printf("  (nx_global, ny_global, nz_global) = (%lld, %lld, %lld)\n", g_nx, g_ny, g_nz);
#else
      printf("  (nx_global, ny_global, nz_global) = (%d, %d, %d)\n", g_nx, g_ny, g_nz);
#endif
      printf("  (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      printf("  (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(double, 4);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0*cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0*cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0*cz;
   }

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian(MPI_COMM_WORLD, 
		g_nx, g_ny, g_nz, P, Q, R, p, q, r, values, &rhs, &x);

   hypre_TFree(values);

   *system_size_ptr = (double)g_nx*(double)g_ny*(double)g_nz; 
   *A_ptr = A;
   *rhs_ptr = rhs;
   *x_ptr = x;

   return (0);
}

/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

int
BuildParLaplacian27pt( int                  argc,
                       char                *argv[],
                       double              *system_size_ptr,
                       HYPRE_ParCSRMatrix  *A_ptr,
                       HYPRE_ParVector  *rhs_ptr,
                       HYPRE_ParVector  *x_ptr)
{
   int                 nx, ny, nz;
   HYPRE_BigInt        g_nx, g_ny, g_nz;
   int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;
   HYPRE_ParVector  rhs;
   HYPRE_ParVector  x;

   int                 num_procs, myid;
   int                 p, q, r;
   double             *values;
   int                 arg_index;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
   g_nx = 0;
   g_ny = 0;
   g_nz = 0;

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      } 
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   g_nx = nx*P;
   g_ny = ny*Q;
   g_nz = nz*R;

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
 
   if (myid == 0)
   {
      printf("  Laplace type problem with a 27-point stencil \n");
#ifdef HYPRE_LONG_LONG
      printf("  (nx_global, ny_global, nz_global) = (%lld, %lld, %lld)\n", g_nx, g_ny, g_nz);
#else
      printf("  (nx_global, ny_global, nz_global) = (%d, %d, %d)\n", g_nx, g_ny, g_nz);
#endif
      printf("  (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix 
    *-----------------------------------------------------------*/
 
   values = hypre_CTAlloc(double, 2);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
	values[0] = 8.0;
   if (nx*ny == 1 || nx*nz == 1 || ny*nz == 1)
	values[0] = 2.0;
   values[1] = -1.;

   A = (HYPRE_ParCSRMatrix) GenerateLaplacian27pt(MPI_COMM_WORLD,
                               g_nx, g_ny, g_nz, P, Q, R, p, q, r, values, &rhs, &x);

   hypre_TFree(values);

   /* *system_size_ptr = (double)P*(double)Q*(double)R*
      (double)nx*(double)ny*(double)nz;  */
   *system_size_ptr = (double)g_nx*(double)g_ny*(double)g_nz; 
   *rhs_ptr = rhs;
   *x_ptr = x;
   *A_ptr = A;

   return (0);
}

int
BuildParVarDifConv( int                  argc,
                    char                *argv[],
                    double              *system_size_ptr,
                    HYPRE_ParCSRMatrix  *A_ptr    ,
                    HYPRE_ParVector  *rhs_ptr,
                    HYPRE_ParVector  *x_ptr)
{
   int                 nx, ny, nz;
   HYPRE_BigInt        g_nx, g_ny, g_nz;
   int                 P, Q, R;

   HYPRE_ParCSRMatrix  A;
   HYPRE_ParVector  rhs;
   HYPRE_ParVector  x;

   int                 num_procs, myid;
   int                 p, q, r;
   int                 arg_index = 0;
   double              eps = 1;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
   g_nx = 0;
   g_ny = 0;
   g_nz = 0;

   nx = 10;
   ny = 10;
   nz = 10;
   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   g_nx = nx*P;
   g_ny = ny*Q;
   g_nz = nz*R;

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P*Q*R) != num_procs)
   {
      printf("Error: Invalid number of processors or processor topology\n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      printf("  elliptic PDE on a cube with jumps\n");
#ifdef HYPRE_LONG_LONG
      printf("  (nx_global, ny_global, nz_global) = (%lld, %lld, %lld)\n", g_nx, g_ny, g_nz);
#else
      printf("  (nx_global, ny_global, nz_global) = (%d, %d, %d)\n", g_nx, g_ny, g_nz);
#endif
      printf("  (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
   }
   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p)/P) % Q;
   r = ( myid - p - P*q)/( P*Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   A = (HYPRE_ParCSRMatrix) GenerateVarDifConv(MPI_COMM_WORLD,
                               g_nx, g_ny, g_nz, P, Q, R, p, q, r, eps, &rhs, &x);

   /* *system_size_ptr = (double)P*(double)Q*(double)R*
      (double)nx*(double)ny*(double)nz; */
   *system_size_ptr = (double)g_nx*(double)g_ny*(double)g_nz; 
   *A_ptr = A;
   *rhs_ptr = rhs;
   *x_ptr = x;

   return (0);
}
