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
 ${INCLUDE_CFLAGS}
 
RANLIB = ranlib

HEADERS =\
 csr_matrix.h\
 headers.h\
 HYPRE_seq_mv.h\
 seq_mv.h\
 vector.h

FILES =\
 big_csr_matrix.c\
 csr_matop.c\
 csr_matrix.c\
 csr_matvec.c\
 genpart.c\
 HYPRE_csr_matrix.c\
 HYPRE_vector.c\
 vector.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################
 
all: libseq_mv.a

clean:
	@rm -f *.o *.dxx
 
veryclean: clean
	@rm -f lib*.a
 

##################################################################
# Rules
##################################################################

libseq_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################
 
.c.o:
	${CC} -o $@ -c ${CFLAGS} $<

