#!/usr/bin/env perl
  
#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday);

foreach $filename (@ARGV) {
 #  print "Start to preprocess ", $filename, "...\n";
   $ssum = 0.0;
   $nmax = 2;
   @nn = (1..$nmax);
   for(@nn) {
    $retval = gettimeofday( );
    system("$filename");
    $tt = gettimeofday( );
    $ttotal = $tt - $retval;
    $ssum = $ssum + $ttotal;
   }
   $avg = $ssum / $nmax;
   printf("%.3f\n", $avg);
 #  print "End to preprocess ", $avg, "...\n";
}

