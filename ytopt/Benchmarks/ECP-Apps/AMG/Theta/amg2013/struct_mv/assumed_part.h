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
 * Header info for the struct assumed partition
 *
 *****************************************************************************/

#ifndef hypre_ASSUMED_PART_HEADER
#define hypre_ASSUMED_PART_HEADER


/* to prevent overflow */

#define hypre_doubleBoxVolume(box) \
   ((double) hypre_BoxSizeX(box) * (double) hypre_BoxSizeY(box) * (double) hypre_BoxSizeZ(box))


typedef struct 
{
   /* the entries will be the same for all procs */  
   hypre_BoxArray      *regions;
   int                 num_regions;      
   int                 *proc_partitions;
   hypre_Index         *divisions;
   /* these entries are specific to each proc */
   hypre_BoxArray      *my_partition;
   hypre_BoxArray      *my_partition_boxes;
   int                 *my_partition_proc_ids;
   int                 *my_partition_boxnums;
   int                 my_partition_ids_size;   
   int                 my_partition_ids_alloc;
   int                 my_partition_num_distinct_procs;
    
} hypre_StructAssumedPart;


/*Accessor macros */

#define hypre_StructAssumedPartRegions(apart) ((apart)->regions) 
#define hypre_StructAssumedPartNumRegions(apart) ((apart)->num_regions) 
#define hypre_StructAssumedPartDivisions(apart) ((apart)->divisions) 
#define hypre_StructAssumedPartDivision(apart, i) ((apart)->divisions[i]) 
#define hypre_StructAssumedPartProcPartitions(apart) ((apart)->proc_partitions) 
#define hypre_StructAssumedPartProcPartition(apart, i) ((apart)->proc_partitions[i]) 
#define hypre_StructAssumedPartMyPartition(apart) ((apart)->my_partition)
#define hypre_StructAssumedPartMyPartitionBoxes(apart) ((apart)->my_partition_boxes)
#define hypre_StructAssumedPartMyPartitionProcIds(apart) ((apart)->my_partition_proc_ids)
#define hypre_StructAssumedPartMyPartitionIdsSize(apart) ((apart)->my_partition_ids_size)
#define hypre_StructAssumedPartMyPartitionIdsAlloc(apart) ((apart)->my_partition_ids_alloc)
#define hypre_StructAssumedPartMyPartitionNumDistinctProcs(apart) ((apart)->my_partition_num_distinct_procs)
#define hypre_StructAssumedPartMyPartitionBoxnums(apart) ((apart)->my_partition_boxnums)



#endif
