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
 -I../utilities\
 -I../struct_mv\
 -I../seq_mv\
 -I../parcsr_mv\
 -I../IJ_mv\
 ${INCLUDE_CFLAGS}
                                                                                               
RANLIB = ranlib


HEADERS =\
 box_map.h\
 headers.h\
 HYPRE_sstruct_mv.h\
 sstruct_graph.h\
 sstruct_grid.h\
 sstruct_matrix.h\
 sstruct_mv.h\
 sstruct_stencil.h\
 sstruct_vector.h

FILES =\
 box_map.c\
 HYPRE_sstruct_graph.c\
 HYPRE_sstruct_grid.c\
 HYPRE_sstruct_matrix.c\
 HYPRE_sstruct_stencil.c\
 HYPRE_sstruct_vector.c\
 sstruct_axpy.c\
 sstruct_copy.c\
 sstruct_graph.c\
 sstruct_grid.c\
 sstruct_innerprod.c\
 sstruct_matrix.c\
 sstruct_matvec.c\
 sstruct_overlap_innerprod.c\
 sstruct_scale.c\
 sstruct_stencil.c\
 sstruct_vector.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libsstruct_mv.a

clean:
	@rm -f *.o *.dxx

veryclean: clean
	@rm -f lib*.a

##################################################################
# Rules
##################################################################

libsstruct_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################
                                                                                               
.c.o:
	${CC} -o $@ -c ${CFLAGS} $<
