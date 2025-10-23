// -*-c++-*-
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
#ifndef TIMESERIES_H
#define TIMESERIES_H

#include <vector>
#include <string>

#include "sw4.h"

class EW;
class Sarray;
class Filter;

using namespace std;

class TimeSeries{

public:

// support for derived quantities of the time derivative are not yet implemented
  enum receiverMode{Displacement, Div, Curl, Strains, Velocity, DisplacementGradient /*, DivVelo, CurlVelo, StrainsVelo */ };

TimeSeries( EW* a_ew, std::string fileName, std::string staName, receiverMode mode, bool sacFormat, bool usgsFormat, 
	    float_sw4 x, float_sw4 y, float_sw4 z, bool topoDepth, int writeEvery, bool xyzcomponent=true );
~TimeSeries();

void allocateRecordingArrays( int numberOfTimeSteps, float_sw4 startTime, float_sw4 timeStep );
  
void recordData(vector<float_sw4> & u );
void recordData( float_sw4* u );
int urec_size();

void writeFile( string suffix="" );

void readFile( EW* ew, bool ignore_utc );

   //float_sw4 **getRecordingArray(){ return mRecordedSol; }

   //int getNsteps() const {return mLastTimeStep+1;}

bool myPoint(){ return m_myPoint; }

receiverMode getMode(){ return m_mode; }

float_sw4 getX() const {return mX;}
float_sw4 getY() const {return mY;}
float_sw4 getZ() const {return mZ;}

   //float_sw4 arrival_time( float_sw4 lod );

TimeSeries* copy( EW* a_ew, string filename, bool addname=false );

   //float_sw4 misfit( TimeSeries& observed, TimeSeries* diff, float_sw4& dshift, float_sw4& ddshift, float_sw4& dd1shift );
   //float_sw4 misfit2( TimeSeries& observed );

   //void interpolate( TimeSeries& intpfrom );

   //void use_as_forcing( int n, std::vector<Sarray>& f, std::vector<float_sw4> & h, float_sw4 dt,
   //		     Sarray& Jac, bool topography_exists );

   //float_sw4 product( TimeSeries& ts ) const;
   //float_sw4 product_wgh( TimeSeries& ts ) const;

   //void set_utc_to_simulation_utc();
   //void filter_data( Filter* filter_ptr );

void print_timeinfo() const;
   //void set_window( float_sw4 winl, float_sw4 winr );
   //void exclude_component( bool usex, bool usey, bool usez );
   //void readSACfiles( EW* ew, const char* sac1, const char* sac2, const char* sac3, bool ignore_utc );
   //void set_shift( float_sw4 shift );
   //float_sw4 get_shift() const;
   //void add_shift( float_sw4 shift );
std::string getStationName(){return m_staName;}

   //void set_scalefactor( float_sw4 value );
   //bool get_compute_scalefactor() const;
   //float_sw4 get_scalefactor() const;

// for simplicity, make the grid point location public
int m_i0;
int m_j0;
int m_k0;
int m_grid0;

private:   
TimeSeries();
void write_usgs_format( string a_fileName);
void write_sac_format( int npts, char *ofile, float *y, float btime, float dt, char *var,
		       float cmpinc, float cmpaz);
bool find_topo_zcoord( float_sw4 X, float_sw4 Y, float_sw4& Ztopo );
bool computeNearestGridPointZ( float_sw4 X, float_sw4 Y, float_sw4 Z,
			       int& grid0, int& k0 );
   
   //float_sw4 utc_distance( int utc1[7], int utc2[7] );
   //void dayinc( int date[7] );
int lastofmonth( int year, int month );
   //int utccompare( int utc1[7], int utc2[7] );
   //int leap_second_correction( int utc1[7], int utc2[7] );

   //void readSACheader( const char* fname, float_sw4& dt, float_sw4& t0, float_sw4& lat,
   //		    float_sw4& lon, float_sw4& cmpaz, float_sw4& cmpinc, int utc[7], int& npts);
   //void readSACdata( const char* fname, int npts, float_sw4* u );		    
   //void convertjday( int jday, int year, int& day, int& month );   
   //void getwgh( float_sw4 ai, float_sw4 wgh[6], float_sw4 dwgh[6], float_sw4 ddwgh[6] );
   //void getwgh5( float_sw4 ai, float_sw4 wgh[6], float_sw4 dwgh[6], float_sw4 ddwgh[6] );

receiverMode m_mode;
int m_nComp;

bool m_myPoint; // set to true if this processor writes to the arrays

std::string m_fileName, m_staName;

float_sw4 mX, mY, mZ, mGPX, mGPY, mGPZ; // original and actual location
float_sw4 m_zTopo;

bool m_zRelativeToTopography; // location is given relative to topography

int mWriteEvery;

bool m_usgsFormat, m_sacFormat;
string m_path;

// start time, shift, and time step 
float_sw4 m_t0, m_shift, m_dt;

// size of recording arrays
int mAllocatedSize;

// index of last recorded element
int mLastTimeStep;

// recording arrays
float_sw4** mRecordedSol;
float**  mRecordedFloats;

// ignore this station if it is above the topography or outside the computational domain
//bool mIgnore;

// sac header data
int mEventYear, mEventMonth, mEventDay, mEventHour, mEventMinute;
float_sw4 mEventSecond, m_rec_lat, m_rec_lon, m_rec_gp_lat, m_rec_gp_lon;
float_sw4 m_epi_lat, m_epi_lon, m_epi_depth, m_epi_time_offset, m_x_azimuth;

// sac ascii or binary?
bool mBinaryMode;

// UTC time for start of seismogram, 
//     m_t0  =  m_utc - 'utc reference time',
//           where 'utc reference time' corresponds to simulation time zero.
//     the time series values are thus given by simulation times t_k = m_t0 + m_shift + k*m_dt, k=0,1,..,mLastTimeStep
int m_utc[7];

// Variables for rotating the output displacement or velocity components when Nort-East-UP is 
// selected (m_xyzcomponent=false) instead of Cartesian components (m_xyzcomponent=true).
bool m_xyzcomponent;
float_sw4 m_calpha, m_salpha, m_thxnrm, m_thynrm;

   //bool m_compute_scalefactor;
   //float_sw4 m_scalefactor;

//  float_sw4 m_dthi, m_velocities;
// float_sw4 m_dmx, m_dmy, m_dmz, m_d0x, m_d0y, m_d0z;
// float_sw4 m_dmxy, m_dmxz, m_dmyz, m_d0xy, m_d0xz, m_d0yz;

// Window for optimization, m_winL, m_winR given relative simulation time zero.
//   float_sw4 m_winL, m_winR;
//   bool   m_use_win, m_use_x, m_use_y, m_use_z;

// quiet mode?
//   bool mQuietMode;

// pointer to EW object
   EW * m_ew;

};


#endif
