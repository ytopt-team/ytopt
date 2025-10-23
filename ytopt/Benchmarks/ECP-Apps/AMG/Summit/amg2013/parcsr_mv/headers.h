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




#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "HYPRE.h"
#include "parcsr_mv.h"


#ifdef HYPRE_USE_MCSUP
  #include <numa.h>
  #include "mcsup.h"
#endif
