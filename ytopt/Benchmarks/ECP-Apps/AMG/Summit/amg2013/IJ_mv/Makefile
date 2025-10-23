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
 -I../parcsr_ls\
 ${INCLUDE_CFLAGS}
 
RANLIB = ranlib

HEADERS =\
 aux_parcsr_matrix.h\
 aux_par_vector.h\
 headers.h\
 HYPRE_IJ_mv.h\
 IJ_matrix.h\
 IJ_mv.h\
 IJ_vector.h

FILES =\
 aux_parcsr_matrix.c\
 aux_par_vector.c\
 HYPRE_IJMatrix.c\
 HYPRE_IJVector.c\
 IJMatrix.c\
 IJMatrix_parcsr.c\
 IJVector_parcsr.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libIJ_mv.a

clean:
	@rm -f *.o *.dxx
 
veryclean: clean
	@rm -f lib*.a
 
##################################################################
# Rules
##################################################################

libIJ_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################
 
.c.o:
	${CC} -o $@ -c ${CFLAGS} $<
