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



srcdir = .

HYPRE_DIRS =\
 utilities\
 krylov\
 IJ_mv\
 parcsr_ls\
 parcsr_mv\
 seq_mv\
 struct_mv\
 sstruct_mv\
 test

all:
	@ \
	for i in ${HYPRE_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i; make); \
	    echo ""; \
	  fi; \
	done

clean:
	@ \
	for i in ${HYPRE_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Cleaning $$i ..."; \
	    (cd $$i; make clean); \
	  fi; \
	done

veryclean:
	@ \
	for i in ${HYPRE_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Very-cleaning $$i ..."; \
	    (cd $$i; make veryclean); \
	  fi; \
	done

tags:
	find . -name "*.c" -or -name "*.C" -or -name "*.h" -or -name "*.c??" -or -name "*.h??" -or -name "*.f" | etags -
