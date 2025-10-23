#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday);

$A_FILE = "gm.report";
my $i = 0;
foreach $filename (@ARGV) {
 #  print "Start to preprocess ", $filename, "...\n";
    system("geopmlaunch aprun -n 8 -N 1 --geopm-preload --geopm-ctl=pthread --geopm-report gm.report -- $filename >/dev/null 2>&1");
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
        }
        if ($i == 1) {
          if ($line =~ /package-energy/) {
                ($v1, $v2) = split(': ', $line);
		 chomp ($v2);
          }
          if ($line =~ /dram-energy/) {
                ($v3, $v4) = split(': ', $line);
		chomp ($v4);
		$i = 0;
          }
        }
   }
   printf("%.3f\n", $v2+$v4);
   close(TEMFILE);
   #system("unlink  gm.report");
}

