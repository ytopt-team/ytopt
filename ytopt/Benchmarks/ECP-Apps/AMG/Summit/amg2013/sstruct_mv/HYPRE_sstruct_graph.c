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
 * HYPRE_SStructGraph interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGraphCreate( MPI_Comm             comm,
                          HYPRE_SStructGrid    grid,
                          HYPRE_SStructGraph  *graph_ptr )
{
   int  ierr = 0;

   hypre_SStructGraph     *graph;
   int                     nparts;
   hypre_SStructStencil ***stencils;
   hypre_SStructPGrid    **pgrids;
   int                     nvars;
   int                     part, var;

   graph = hypre_TAlloc(hypre_SStructGraph, 1);

   hypre_SStructGraphComm(graph) = comm;
   hypre_SStructGraphNDim(graph) = hypre_SStructGridNDim(grid);
   hypre_SStructGridRef(grid, &hypre_SStructGraphGrid(graph));
   nparts = hypre_SStructGridNParts(grid);
   hypre_SStructGraphNParts(graph) = nparts;
   pgrids = hypre_SStructGridPGrids(grid);
   hypre_SStructGraphPGrids(graph) = pgrids;
   stencils = hypre_TAlloc(hypre_SStructStencil **, nparts);
   for (part = 0; part < nparts; part++)
   {
      nvars = hypre_SStructPGridNVars(pgrids[part]);
      stencils[part] = hypre_TAlloc(hypre_SStructStencil *, nvars);
      for (var = 0; var < nvars; var++)
      {
         stencils[part][var] = NULL;
      }
   }
   hypre_SStructGraphStencils(graph) = stencils;

   hypre_SStructGraphNUVEntries(graph)  = 0;
   hypre_SStructGraphAUVEntries(graph)  = 0;
   hypre_SStructGraphIUVEntries(graph)  = NULL;

   hypre_SStructGraphUVEntries(graph)   = NULL;
   hypre_SStructGraphTotUEntries(graph) = 0;
   hypre_SStructGraphRefCount(graph)    = 1;
   hypre_SStructGraphObjectType(graph) = HYPRE_SSTRUCT;

   *graph_ptr = graph;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphDestroy
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGraphDestroy( HYPRE_SStructGraph graph )
{
   int  ierr = 0;

   int                     nparts;
   hypre_SStructPGrid    **pgrids;
   hypre_SStructStencil ***stencils;
   int                     nUventries;
   int                    *iUventries;
 
   hypre_SStructUVEntry  **Uventries;
   hypre_SStructUVEntry   *Uventry;
   int                     nvars;
   int                     part, var, i;

   if (graph)
   {
      hypre_SStructGraphRefCount(graph) --;
      if (hypre_SStructGraphRefCount(graph) == 0)
      {
         nparts   = hypre_SStructGraphNParts(graph);
         pgrids   = hypre_SStructGraphPGrids(graph);
         stencils = hypre_SStructGraphStencils(graph);
         nUventries = hypre_SStructGraphNUVEntries(graph);
         iUventries = hypre_SStructGraphIUVEntries(graph);

         Uventries  = hypre_SStructGraphUVEntries(graph);
         for (part = 0; part < nparts; part++)
         {
            nvars = hypre_SStructPGridNVars(pgrids[part]);
            for (var = 0; var < nvars; var++)
            {
               HYPRE_SStructStencilDestroy(stencils[part][var]);
            }
            hypre_TFree(stencils[part]);
         }
         HYPRE_SStructGridDestroy(hypre_SStructGraphGrid(graph));
         hypre_TFree(stencils);
         for (i = 0; i < nUventries; i++)
         {
            Uventry = Uventries[iUventries[i]];
            if (Uventry)
            {
               hypre_TFree(hypre_SStructUVEntryUEntries(Uventry));
               hypre_TFree(Uventry);
            }
            Uventries[iUventries[i]] = NULL;
         }
         hypre_TFree(iUventries);
         hypre_TFree(Uventries);
         hypre_TFree(graph);
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphSetStencil
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGraphSetStencil( HYPRE_SStructGraph   graph,
                              int                  part,
                              int                  var,
                              HYPRE_SStructStencil stencil )
{
   int  ierr = 0;

   hypre_SStructStencilRef(stencil,
                           &hypre_SStructGraphStencil(graph, part, var));

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphAddEntries-
 *   THIS IS FOR A NON-OVERLAPPING GRID GRAPH.
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGraphAddEntries( HYPRE_SStructGraph   graph,
                              int                  part,
                              int                 *index,
                              int                  var,
                              int                  to_part,
                              int                 *to_index,
                              int                  to_var )
{
   int  ierr = 0;

   hypre_SStructGrid     *grid       = hypre_SStructGraphGrid(graph);
   int                    ndim       = hypre_SStructGridNDim(grid);
   int                    nUventries = hypre_SStructGraphNUVEntries(graph);
   int                    aUventries = hypre_SStructGraphAUVEntries(graph);
   int                   *iUventries = hypre_SStructGraphIUVEntries(graph);
 
   int                    type       = hypre_SStructGraphObjectType(graph);
                              
   hypre_SStructUVEntry **Uventries  = hypre_SStructGraphUVEntries(graph);
   hypre_SStructUVEntry  *Uventry;
   int                    nUentries;
   hypre_SStructUEntry   *Uentries;

   hypre_BoxMapEntry     *map_entry;
   hypre_Index            cindex;
   HYPRE_BigInt           big_rank, startrank;
   int                    rank, i;
   int                    box, to_box, to_proc;

   if (!nUventries)
   {
      /* allocate space for non-stencil entries GEC1102
       * the size equal to the ghost local size of grid */
      aUventries = hypre_SStructGridGhlocalSize(grid);
      iUventries = hypre_TAlloc(int, aUventries);

      Uventries = hypre_CTAlloc(hypre_SStructUVEntry *, aUventries);
      hypre_SStructGraphAUVEntries(graph) = aUventries;
      hypre_SStructGraphIUVEntries(graph) = iUventries;

      hypre_SStructGraphUVEntries(graph)  = Uventries;
   }
   else if (nUventries >= aUventries)
   {
      /* need more space in iUventries array */
      aUventries += 1000;
      iUventries = hypre_TReAlloc(iUventries, int, aUventries);
      hypre_SStructGraphAUVEntries(graph) = aUventries;
      hypre_SStructGraphIUVEntries(graph) = iUventries;

   }

   /* compute location (rank) for Uventry */
   hypre_CopyToCleanIndex(index, ndim, cindex);
   hypre_SStructGridFindMapEntry(grid, part, cindex, var, &map_entry);

    /* GEC0203 getting the rank */ 
   hypre_SStructMapEntryGetGlobalRank(map_entry, cindex, &big_rank, type);

   /* GEC 0902 filling up the iUventries with local ghrank
    * since HYPRE_SSTRUCT is chosen */

   if (type == HYPRE_SSTRUCT || type == HYPRE_STRUCT) 
   { 
     startrank = hypre_SStructGridGhstartRank(grid);
   }
   if (type == HYPRE_PARCSR)
   {
     startrank = hypre_SStructGridStartRank(grid);
   }
    
   rank = (int)(big_rank-startrank);

   iUventries[nUventries] = rank;

   if (Uventries[rank] == NULL)
   {
      Uventry = hypre_TAlloc(hypre_SStructUVEntry, 1);
      hypre_SStructUVEntryPart(Uventry) = part;
      hypre_CopyToCleanIndex(index, ndim, hypre_SStructUVEntryIndex(Uventry));
      hypre_SStructUVEntryVar(Uventry) = var;
      hypre_SStructMapEntryGetBox(map_entry, &box);
      hypre_SStructUVEntryBox(Uventry)= box;
      nUentries = 1;
      Uentries = hypre_TAlloc(hypre_SStructUEntry, nUentries);
   }
   else
   {
      Uventry = Uventries[rank];
      nUentries = hypre_SStructUVEntryNUEntries(Uventry) + 1;
      Uentries = hypre_SStructUVEntryUEntries(Uventry);
      Uentries = hypre_TReAlloc(Uentries, hypre_SStructUEntry, nUentries);
   }
   hypre_SStructUVEntryNUEntries(Uventry) = nUentries;
   hypre_SStructUVEntryUEntries(Uventry)  = Uentries;

   i = nUentries - 1;
   hypre_SStructUVEntryToPart(Uventry, i) = to_part;
   hypre_CopyToCleanIndex(to_index, ndim,
                          hypre_SStructUVEntryToIndex(Uventry, i));
   hypre_SStructUVEntryToVar(Uventry, i) = to_var;

   hypre_CopyToCleanIndex(to_index, ndim, cindex);
   hypre_SStructGridFindMapEntry(grid, to_part, cindex, to_var, &map_entry);
   hypre_SStructMapEntryGetBox(map_entry, &to_box);
   hypre_SStructUVEntryToBox(Uventry, i)= to_box;
   hypre_SStructMapEntryGetProcess(map_entry, &to_proc);
   hypre_SStructUVEntryToProc(Uventry, i)= to_proc;

   Uventries[rank] = Uventry; /* GEC1102 where rank labels Uventries */

   hypre_SStructGraphNUVEntries(graph) ++;
   hypre_SStructGraphUVEntries(graph) = Uventries;
   hypre_SStructGraphTotUEntries(graph) ++;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGraphAssemble
 *
 * This routine mainly computes the column numbers for the non-stencil
 * graph entries (i.e., those created by GraphAddEntries calls).  The
 * routine will compute as many of these numbers on-process, but if
 * the information needed to compute a column is not stored locally,
 * it will be computed off-process instead.
 *
 * To do this, column info is first requested from other processes
 * (tag=1 communications).  While waiting for this info, requests from
 * other processes are filled (tag=2).  Simultaneously, a fanin/fanout
 * procedure (tag=0) is invoked to determine when to stop: each
 * process participates in the send portion of the fanin once it has
 * received all of its requested column data and once it has completed
 * its receive portion of the fanin; each process then participates in
 * the fanout.
 *
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructGraphAssemble( HYPRE_SStructGraph graph )
{
   int ierr = 0;

   MPI_Comm               comm        = hypre_SStructGraphComm(graph);
   hypre_SStructGrid     *grid        = hypre_SStructGraphGrid(graph);
   int                    nUventries  = hypre_SStructGraphNUVEntries(graph);
   int                   *iUventries  = hypre_SStructGraphIUVEntries(graph);
   hypre_SStructUVEntry **Uventries   = hypre_SStructGraphUVEntries(graph);
   int                    totUentries = hypre_SStructGraphTotUEntries(graph);

   int                    type        = hypre_SStructGraphObjectType(graph);


   hypre_SStructUVEntry  *Uventry;
   hypre_SStructUEntry   *Uentry;
   int                    nUentries;
   int                    to_part;
   hypre_IndexRef         to_index;
   int                    to_var;
   int                    to_box;
   int                    to_proc;
   int                    proc;
   HYPRE_BigInt           rank;
   hypre_BoxMapEntry     *map_entry;

   /* type 0 communications used to determine completion (NULL messages) */
   MPI_Request           *t0requests;
   MPI_Status            *t0statuses;
   int                   *t0hiprocs;
   int                    t0nhi;
   int                    t0loproc;

   /* type 1 communications used to get column data from other processes */
   MPI_Request           *t1requests, tmprequest;
   MPI_Status            *t1statuses;
   int                  **t1sendbufs;
   int                  **t1recvbufs;
   int                   *t1bufsizes;
   int                   *t1bufprocs;
   hypre_SStructUEntry ***t1Uentries;
   int                    t1totsize;
   int                    t1ncomms;
   int                    t1complete;
   struct UentryLink
   {
      hypre_SStructUEntry *uentry;
      struct UentryLink   *next;
   };
   struct UentryLink     *t1links;
   struct UentryLink     *t1linksptr;
   struct UentryLink    **t1lists;

   /* type 2 communications used to send column data to other processes */
   MPI_Status             t2status;
   int                   *t2commbuf;
   int                    t2bufsize, t2maxbufsize;
   int                    t2flag;

   int                    nprocs, myproc;
   int                    fanin_complete, complete;
   int                    i, j;

   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &myproc);

   /*---------------------------------------------------------
    * Sort the iUventries array and eliminate duplicates.
    *---------------------------------------------------------*/

   if (nUventries > 1)
   {
      qsort0(iUventries, 0, nUventries - 1);

      j = 1;
      for (i = 1; i < nUventries; i++)
      {
         if (iUventries[i] > iUventries[i-1])
         {
            iUventries[j] = iUventries[i];
            j++;
         }
      }
      nUventries = j;
      hypre_SStructGraphNUVEntries(graph) = nUventries;
   }
   /*---------------------------------------------------------
    * Compute non-stencil column numbers (if possible), and
    * start building requests for needed off-process info.
    *---------------------------------------------------------*/

   t1totsize = 0;
   t1ncomms = 0;
   for (i = 0; i < nUventries; i++)
   {
      Uventry = Uventries[iUventries[i]];
      nUentries = hypre_SStructUVEntryNUEntries(Uventry);
      for (j = 0; j < nUentries; j++)
      {
         to_part  = hypre_SStructUVEntryToPart(Uventry, j);
         to_index = hypre_SStructUVEntryToIndex(Uventry, j);
         to_var   = hypre_SStructUVEntryToVar(Uventry, j);
         /*
           hypre_SStructGridFindMapEntry(grid, to_part, to_index, to_var,
                                       &map_entry);
          */

         /*---------------------------------------------------------
          * used in future? The to_box corresponds to the first
          * map_entry on the map_entry link list.
          *---------------------------------------------------------*/
         to_box   = hypre_SStructUVEntryToBox(Uventry, j);
         to_proc  = hypre_SStructUVEntryToProc(Uventry, j);
         hypre_SStructGridBoxProcFindMapEntry(grid, to_part, to_var, to_box,
                                              to_proc, &map_entry);
         if (map_entry != NULL)
         {
            /* compute ranks locally */

           hypre_SStructMapEntryGetGlobalRank(map_entry, to_index, &rank, type);          
           hypre_SStructUVEntryRank(Uventry, j) = rank;
         
         }
         else
         {
	   printf("off process computation\n");

            /* compute ranks off-process: start building type 1 requests */
            hypre_SStructMapEntryGetProcess(map_entry, &proc);

            /* initialize some things */
            if (t1totsize == 0)
            {
               t1links    = hypre_TAlloc(struct UentryLink, totUentries);
               t1linksptr = t1links;
               t1lists    = hypre_CTAlloc(struct UentryLink *, nprocs);
               t1bufprocs = hypre_TAlloc(int, 10);
            }

            /* add a new process to the requests list */
            if (t1lists[proc] == NULL)
            {
               if (t1ncomms%10 == 0)
               {
                  t1bufprocs = hypre_TReAlloc(t1bufprocs, int, (t1ncomms+10));
               }
               t1bufprocs[t1ncomms] = proc;
               t1ncomms++;
            }

            /* set up the new link */
            (t1linksptr -> uentry) = hypre_SStructUVEntryUEntry(Uventry, j);
            (t1linksptr -> next)   = t1lists[proc];

            /* insert link into the appropriate list */
            t1lists[proc] = t1linksptr;
            t1linksptr++;

            t1totsize++;
         }
      }
   }

   /* set up remaining type 1 request info */
 
   if (t1ncomms > 0)
   {
      t1bufsizes    = hypre_TAlloc(int, t1ncomms);
      t1Uentries    = hypre_TAlloc(hypre_SStructUEntry **, t1ncomms);
      t1Uentries[0] = hypre_TAlloc(hypre_SStructUEntry *, t1totsize);
      j = 0;
      for (i = 0; i < t1ncomms; i++)
      {
         proc = t1bufprocs[i];
         t1linksptr = t1lists[proc];
         t1bufsizes[i] = 0;
         while (t1linksptr != NULL)
         {
            t1Uentries[0][j++] = (t1linksptr -> uentry);
            t1linksptr = (t1linksptr -> next);
            t1bufsizes[i]++;
         }
      }
      hypre_TFree(t1links);
      hypre_TFree(t1lists);

      t1sendbufs    = hypre_TAlloc(int *, t1ncomms);
      t1sendbufs[0] = hypre_TAlloc(int, t1totsize*6);
      t1recvbufs    = hypre_TAlloc(int *, t1ncomms);
      t1recvbufs[0] = hypre_TAlloc(int, t1totsize);

      for (j = 0; j < t1totsize; j++)
      {
         Uentry = t1Uentries[0][j];
         to_part  = hypre_SStructUEntryToPart(Uentry);
         to_index = hypre_SStructUEntryToIndex(Uentry);
         to_var   = hypre_SStructUEntryToVar(Uentry);
         to_box   = hypre_SStructUEntryToBox(Uentry);
         t1sendbufs[0][6*j  ] = to_part;
         t1sendbufs[0][6*j+1] = hypre_IndexD(to_index, 0);
         t1sendbufs[0][6*j+2] = hypre_IndexD(to_index, 1);
         t1sendbufs[0][6*j+3] = hypre_IndexD(to_index, 2);
         t1sendbufs[0][6*j+4] = to_var;
         t1sendbufs[0][6*j+5] = to_box;
      }

      /* GEC1002 commenting this out to replace it by something else  
       * since I think that a 6 is missing in sending buffer 
       * 
       * for (i = 1; i < t1ncomms; i++)
       * {
       *  t1sendbufs[i] = t1sendbufs[i-1] + t1bufsizes[i-1];
       *  t1recvbufs[i] = t1sendbufs[i-1] + t1bufsizes[i-1];
       *  t1Uentries[i] = t1Uentries[i-1] + t1bufsizes[i-1];
       *	  }   */

      for (i = 1; i < t1ncomms; i++)
      {
         t1sendbufs[i] = t1sendbufs[i-1] + 6*t1bufsizes[i-1];
         t1recvbufs[i] = t1sendbufs[i-1] + t1bufsizes[i-1];
         t1Uentries[i] = t1Uentries[i-1] + t1bufsizes[i-1];
      }  
   }

   /*---------------------------------------------------------
    * Exchange column data with other processes
    *---------------------------------------------------------*/

   t1complete = 1;
   if (t1ncomms > 0)
   {
      t1requests = hypre_CTAlloc(MPI_Request, t1ncomms);
      t1statuses = hypre_CTAlloc(MPI_Status, t1ncomms);
      
      /* post type 1 requests (note: tag=1 on send; tag=2 on receive) */
      for (i = 0; i < t1ncomms; i++)
      {

         MPI_Irecv(t1recvbufs[i], t1bufsizes[i], MPI_INT, t1bufprocs[i],
                   2, comm, &t1requests[i]);
      }
      for (i = 0; i < t1ncomms; i++)
      {
         /* Note: No need to check below that these have completed */
         MPI_Isend(t1sendbufs[i], t1bufsizes[i]*6, MPI_INT, t1bufprocs[i],
                   1, comm, &tmprequest);
      }

      t1complete = 0;
   }

   complete = 1;
   if (nprocs > 1)
   {
      /* compute type 0 communication info */
      for (i = 1, t0nhi = 0; i < nprocs; i *= 2, t0nhi++);
      t0hiprocs = hypre_TAlloc(int, t0nhi);
      t0nhi = 0;
      proc = myproc;
      for (i = 1; i < nprocs; i *= 2)
      {
         if ((proc % 2) == 0)
         {
            /* added if test to avoid t0hi ranks exceeding nprocs */
            if ((myproc + i) < nprocs) 
            {
               t0hiprocs[t0nhi] = myproc + i;
               t0nhi++;
            }
            proc /= 2;
         }
         else
         {
            t0loproc = myproc - i;
            break;
         }
      }

      t0requests = hypre_CTAlloc(MPI_Request, (t0nhi + 1));
      t0statuses = hypre_CTAlloc(MPI_Status, (t0nhi + 1));
      
      /* post type 0 fanin receives */
      for (i = 0; i < t0nhi; i++)
      {
         MPI_Irecv(NULL, 0, MPI_INT, t0hiprocs[i], 0, comm, &t0requests[i]);
      }

      t2commbuf = NULL;
      t2maxbufsize = 0;

      fanin_complete = 0;
      complete = 0;
   }

   while (!complete)
   {
      /* fill as many column data requests as possible */
      /* (note: tag=1 on receive; tag=2 on send) */
      MPI_Iprobe(MPI_ANY_SOURCE, 1, comm, &t2flag, &t2status);
      while (t2flag)
      {
         proc = t2status.MPI_SOURCE;
         
         MPI_Get_count(&t2status, MPI_INT, &t2bufsize);
         if (t2bufsize > t2maxbufsize)
         {
            t2commbuf = hypre_TReAlloc(t2commbuf, int, t2bufsize);
            t2maxbufsize = t2bufsize;
         }
         MPI_Recv(t2commbuf, t2bufsize, MPI_INT, proc, 1, comm, &t2status);

         t2bufsize /= 6;
         for (j = 0; j < t2bufsize; j++)
         {
            to_part                   = t2commbuf[6*j];
            hypre_IndexD(to_index, 0) = t2commbuf[6*j+1];
            hypre_IndexD(to_index, 1) = t2commbuf[6*j+2];
            hypre_IndexD(to_index, 2) = t2commbuf[6*j+3];
            to_var                    = t2commbuf[6*j+4];
            to_box                    = t2commbuf[6*j+5]; /* future use? */
            hypre_SStructGridFindMapEntry(grid, to_part, to_index, to_var,
                                          &map_entry);

            hypre_SStructMapEntryGetGlobalRank(map_entry, to_index, &rank, type);
       
            t2commbuf[j] = rank;     
         }
	  MPI_Send(t2commbuf, t2bufsize, MPI_INT, proc, 2, comm);

         MPI_Iprobe(MPI_ANY_SOURCE, 1, comm, &t2flag, &t2status);
      }

      /* attempt to complete type 0 and type 1 communications */
      if (!t1complete)
      {
         MPI_Testall(t1ncomms, t1requests, &t1complete, t1statuses);
      }
      else if (!fanin_complete)
      {
         MPI_Testall(t0nhi, t0requests, &fanin_complete, t0statuses);
         if (fanin_complete)
         {
            if (myproc > 0)
            {
               /* post type 0 fanin send */
               MPI_Send(NULL, 0, MPI_INT, t0loproc, 0, comm);

               /* post type 0 fanout receive */
               MPI_Irecv(NULL, 0, MPI_INT, t0loproc, 0, comm, t0requests);
            }
         }
      }
      else
      {
         MPI_Test(t0requests, &complete, t0statuses);
         if (complete)
         {
            /* post type 0 fanout sends */
            for (i = 0; i < t0nhi; i++)
            {
               MPI_Send(NULL, 0, MPI_INT, t0hiprocs[i], 0, comm);
            }
         }
      }
   }

   /* copy column data computed off-process into local data structure */
   if (t1ncomms > 0)
   {
      for (j = 0; j < t1totsize; j++)
      {
         Uentry = t1Uentries[0][j];
         hypre_SStructUEntryRank(Uentry) = t1recvbufs[0][j];       
      }
   }

   if (nprocs > 1)
   {
      hypre_TFree(t0requests);
      hypre_TFree(t0statuses);
      hypre_TFree(t0hiprocs);

      hypre_TFree(t2commbuf);
   }

   if (t1ncomms > 0)
   {
      hypre_TFree(t1requests);
      hypre_TFree(t1statuses);
      hypre_TFree(t1sendbufs[0]);
      hypre_TFree(t1sendbufs);
      hypre_TFree(t1recvbufs[0]);
      hypre_TFree(t1recvbufs);
      hypre_TFree(t1bufsizes);
      hypre_TFree(t1bufprocs);
      hypre_TFree(t1Uentries[0]);
      hypre_TFree(t1Uentries);
   }

   return ierr;
}
/*****************************************************************
 *
 *
 *****************************************************************/
int HYPRE_SStructGraphSetObjectType(HYPRE_SStructGraph  graph,
                                    int                 type)
{
  int ierr = 0;
  hypre_SStructGraphObjectType(graph) = type;
  return ierr;
}
