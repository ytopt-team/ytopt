
#ifndef hypre_PARCSR_ASSUMED_PART
#define  hypre_PARCSR_ASSUMED_PART

typedef struct
{
   int                   length;
   HYPRE_BigInt          row_start;
   HYPRE_BigInt          row_end;
   int                   storage_length;
   int                   *proc_list;
   HYPRE_BigInt	         *row_start_list;
   HYPRE_BigInt          *row_end_list;  
  int                    *sort_index;
} hypre_IJAssumedPart;




#endif /* hypre_PARCSR_ASSUMED_PART */

