#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.4 $
#EHEADER**********************************************************************


.SUFFIXES:
.SUFFIXES: .c .o
                                                                                               
srcdir = .
                                                                                               
include ../Makefile.include
                                                                                               
CFLAGS = \
 -I..\
 -I../utilities\
 -I../struct_mv\
 -I../sstruct_mv\
 -I../IJ_mv\
 -I../seq_mv\
 -I../parcsr_mv\
 -I../parcsr_ls\
 -I../krylov\
 ${INCLUDE_CFLAGS}\
 -DHYPRE_TIMING
                                                                                               
LFLAGS =\
 -L.\
 -L../parcsr_ls\
 -L../parcsr_mv\
 -L../IJ_mv\
 -L../seq_mv\
 -L../sstruct_mv\
 -L../struct_mv\
 -L../krylov\
 -L../utilities\
 -lparcsr_ls\
 -lparcsr_mv\
 -lseq_mv\
 -lsstruct_mv\
 -lIJ_mv\
 -lHYPRE_struct_mv\
 -lkrylov\
 -lHYPRE_utilities\
 ${INCLUDE_LFLAGS} 

##################################################################
# Targets
##################################################################
                                                                                                                                         
all: amg2013
                                                                                                                                         
clean:
	@rm -f *.o amg2013
                                                                                                                                         
veryclean: clean
	@rm -f amg2013
                                                                                                                                         
##################################################################
# Rules
##################################################################
                                                                                                                                         
amg2013: amg2013.o
	@echo  "Linking" $@ "... "
	${CC} -o amg2013 amg2013.o ${LFLAGS}
                                                                                                                        
                 
##################################################################
# Generic rules
##################################################################
                                                                                                                                         
.c.o:
	${CC} -o $@ -c ${CFLAGS} $<
                                                               
