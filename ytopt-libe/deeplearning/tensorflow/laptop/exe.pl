#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday); 

$A_FILE = "tmpoutfile.txt";
foreach $filename (@ARGV) {
 #  print "Start to preprocess ", $filename, "...\n";
    system("python $filename > tmpoutfile.txt");
    open (TEMFILE, '<', $A_FILE);
    while (<TEMFILE>) {
        $line = $_;
        chomp ($line);

        if ($line =~ /Test accuracy/) {
                ($v1, $v2) = split(': ', $line);
   		if ($v2) {
			printf("%.3f", 1/$v2)
   		}
        }
   }
   close(TEMFILE);
   system("unlink  tmpoutfile.txt");
}
