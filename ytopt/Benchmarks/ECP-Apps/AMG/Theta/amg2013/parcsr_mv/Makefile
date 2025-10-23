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
 -I../seq_mv\
 ${INCLUDE_CFLAGS}
                                                                                               
RANLIB = ranlib
                                                                                               
HEADERS =\
 headers.h\
 HYPRE_parcsr_mv.h\
 new_commpkg.h\
 par_csr_communication.h\
 par_csr_matrix.h\
 parcsr_mv.h\
 par_vector.h

FILES =\
 HYPRE_parcsr_matrix.c\
 HYPRE_parcsr_vector.c\
 new_commpkg.c\
 par_csr_assumed_part.c\
 par_csr_communication.c\
 par_csr_matop.c\
 par_csr_matrix.c\
 par_csr_matop_marked.c\
 par_csr_matvec.c\
 par_vector.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libparcsr_mv.a
 
 
clean:
	@rm -f *.o *.dxx

veryclean: clean
	@rm -f lib*.a

##################################################################
# Rules
##################################################################

libparcsr_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################
                                                                                               
.c.o:
	${CC} -o $@ -c ${CFLAGS} $<
