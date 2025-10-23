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
.SUFFIXES: .c .f .o
                                                                                               
srcdir = .
                                                                                               
include ../Makefile.include
                                                                                               
CFLAGS =\
 -I..\
 ${INCLUDE_CFLAGS}\
 -DHYPRE_TIMING
                                                                                               
RANLIB= ranlib


##################################################################
# Targets
##################################################################

all: libHYPRE_utilities.a

clean:
	@rm -f *.o 

veryclean: clean
	@rm -f libHYPRE_*.a

##################################################################
# Utilities rules
##################################################################

UTILITIES_HEADERS =\
 amg_linklist.h\
 exchange_data.h\
 general.h\
 hypre_memory.h\
 hypre_smp_forloop.h\
 HYPRE_utilities.h\
 mpistubs.h\
 threading.h\
 thread_mpistubs.h\
 timing.h\
 umalloc_local.h\
 utilities.h

UTILITIES_FILES =\
 amg_linklist.c\
 binsearch.c\
 exchange_data.c\
 hypre_memory.c\
 hypre_qsort.c\
 memory_dmalloc.c\
 mpistubs.c\
 qsplit.c\
 random.c\
 threading.c\
 thread_mpistubs.c\
 timer.c\
 timing.c\
 umalloc_local.c\
 hypre_error.c

UTILITIES_OBJS = ${UTILITIES_FILES:.c=.o}

 
libHYPRE_utilities.a: ${UTILITIES_OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${UTILITIES_OBJS}
	${RANLIB} $@

${UTILITIES_OBJS}: ${UTILITIES_HEADERS}

timer.o: timer.c
	${CC} -D_POSIX_SOURCE -o $@ -c ${CFLAGS} $<

##################################################################
# Generic rules
##################################################################

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<
