#!/usr/bin/env perl -w

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday); 

$A_FILE = "tmpoutfile.txt";
foreach $filename (@ARGV) {
    #print "Start to preprocess ", $filename, "...\n";
    $retval = gettimeofday( ); 
    system("python '$filename' > tmpoutfile.txt");
    $tt = gettimeofday( );
    $ttotal = $tt - $retval;
    printf("%.4f", $ttotal);
}
