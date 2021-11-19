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
#ifndef SW4_CHECKPOINT_H
#define SW4_CHECKPOINT_H

#include <string>
#include <sstream>
#include <vector>

#include "sw4.h"
//#include "boundaryConditionTypes.h"
#include "Sarray.h"
#include "Parallel_IO.h"

class EW;

class CheckPoint
{
public:
   static CheckPoint* nil; // nil pointer
   CheckPoint( EW * a_ew,
	       float_sw4 time, 
	       float_sw4 timeInterval, 
	       int cycle, 
	       int cycleInterval,
	       string fname,
	       size_t bufsize=10000000 );
   CheckPoint( EW * a_ew, string fname, size_t bufsize=10000000 );
   ~CheckPoint();
   void write_checkpoint( float_sw4 a_time, int a_cycle, std::vector<Sarray>& a_Um,
			  std::vector<Sarray>& a_U );
   void read_checkpoint( float_sw4& a_time, int& a_cycle, std::vector<Sarray>& a_Um,
			 std::vector<Sarray>& a_U );
   void setup_sizes();
   bool timeToWrite( float_sw4 time, int cycle, float_sw4 dt );

protected:
   void define_pio( );
   void setSteps( int a_steps );
   void compute_file_suffix( int cycle, std::stringstream & fileSuffix );

   std::string mFilePrefix;
   std::string mRestartFile;
   float_sw4 mTime;
   float_sw4 mTimeInterval;
   bool m_time_done;
   float_sw4 mNextTime;
   float_sw4 mStartTime;

   int mWritingCycle;
   int mCycleInterval;
   bool m_winallocated;
   size_t m_bufsize;

private:
   CheckPoint(); // make it impossible to call default constructor
   CheckPoint(const CheckPoint &cp ); // hide copy constructor 
   int mPreceedZeros; // number of digits for unique time step in file names
   bool m_double;
   EW* mEW;
   Parallel_IO** m_parallel_io;
   std::vector<int*> mWindow; // Local in processor start + end indices for (i,j,k) for each grid level
   std::vector<int*> mGlobalDims; // Global start + end indices for (i,j,k) for each grid level
   std::vector<bool> m_ihavearray;
};

#endif
