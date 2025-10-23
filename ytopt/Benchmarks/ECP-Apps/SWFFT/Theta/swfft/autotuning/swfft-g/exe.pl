#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday);

$A_FILE = "gm.report";
my $i = 0;
my $j = 0;
my $avg = 0;
foreach $filename (@ARGV) {
 #  print "Start to preprocess ", $filename, "...\n";
    system("geopmlaunch aprun -n 4096 -N 1 --geopm-ctl=pthread --geopm-report gm.report -- $filename 2 4096 > tmpoutfile.txt 2>&1");
    open (TEMFILE, '<', $A_FILE);
    while (<TEMFILE>) {
        $line = $_;
        chomp ($line);
        if ($line =~ /Application Totals/) {
	    if ($i == 0) {
                $i = 1;
	    } else {
		    $i = 0;
	    }
	    $j ++;
        }
        if ($i == 1) {
          if ($line =~ /package-energy/) {
                ($v1, $v2) = split(': ', $line);
		chomp ($v2);
		$avg += $v2;
          }
          if ($line =~ /dram-energy/) {
                ($v3, $v4) = split(': ', $line);
		chomp ($v4);
		$avg += $v4;
		$i = 0;
          }
        }
   }
   if ($j != 0) {
        printf("%.3f\n", $avg/$j);
   } else {
        printf("Wrong input file!\n");
   }
   close(TEMFILE);
}

