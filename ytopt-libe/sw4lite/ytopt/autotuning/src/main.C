//  SW4 LICENSE
// # ----------------------------------------------------------------------
// # SW4 - Seismic Waves, 4th order
// # ----------------------------------------------------------------------
// # Copyright (c) 2013, Lawrence Livermore National Security, LLC. 
// # Produced at the Lawrence Livermore National Laboratory. 
// # 
// # Written by:
// # N. Anders Petersson (petersson1@llnl.gov)
// # Bjorn Sjogreen      (sjogreen2@llnl.gov)
// # 
// # LLNL-CODE-643337 
// # 
// # All rights reserved. 
// # 
// # This file is part of SW4, Version: 1.0
// # 
// # Please also read LICENCE.txt, which contains "Our Notice and GNU General Public License"
// # 
// # This program is free software; you can redistribute it and/or modify
// # it under the terms of the GNU General Public License (as published by
// # the Free Software Foundation) version 2, dated June 1991. 
// # 
// # This program is distributed in the hope that it will be useful, but
// # WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// # conditions of the GNU General Public License for more details. 
// # 
// # You should have received a copy of the GNU General Public License
// # along with this program; if not, write to the Free Software
// # Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA 
#include <mpi.h>
#include <string>
#include <iostream>
using namespace std;
#include "EW.h"

int main( int argc, char** argv )
{
   int myRank;
   double  time_start, time_end;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

   //MPI_Barrier(MPI_COMM_WORLD);
   time_start = MPI_Wtime();

   string filename;
   if( argc <= 1 )
   {
      if( myRank == 0 )
      {
	 cout  << "ERROR: ****No input file specified!" << endl;
	 for (int i = 0; i < argc; ++i)
	    cout << "Argv[" << i << "] = " << argv[i] << endl;
      }
      //MPI_Finalize();
      //return 1;
   }
   else
   {
      filename = argv[1];
      EW simulation(filename);
      //MPI_Finalize();
      //return 0;
   }

   //MPI_Barrier(MPI_COMM_WORLD);
   time_end = MPI_Wtime();
   if(myRank == 0) cout <<  " Runtime(seconds): " << time_end - time_start << endl;

   MPI_Finalize();

   return 0;
}
