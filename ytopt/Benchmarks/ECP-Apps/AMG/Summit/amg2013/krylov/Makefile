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
 
include ../Makefile.include

CFLAGS = \
 -I..\
 -I../utilities\
 ${INCLUDE_CFLAGS}

RANLIB = ranlib

HEADERS =\
 all_krylov.h\
 gmres.h\
 krylov.h\
 pcg.h\
 HYPRE_MatvecFunctions.h

FILES =\
 gmres.c\
 HYPRE_gmres.c\
 HYPRE_pcg.c\
 pcg.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libkrylov.a

clean:
	rm -f *.o *.dxx

veryclean: clean
	rm -f lib*.a


##################################################################
# Rules
##################################################################

libkrylov.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################
 
.c.o:
	${CC} -o $@ -c ${CFLAGS} $<

