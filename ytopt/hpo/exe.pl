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

        if ($line =~ /Loss:/) {
                ($v1, $v2) = split('Loss:', $line);
		chomp ($v2);
        }
   }
   if ($v2 == 0 ) {
        printf("0");
   } else {
        printf("%.4f", $v2);
   }
   close(TEMFILE);
}
