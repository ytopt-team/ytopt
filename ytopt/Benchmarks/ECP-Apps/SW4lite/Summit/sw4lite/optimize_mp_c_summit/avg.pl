#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# processgm.pl: process gm.report to get the average enery
#
use Time::HiRes qw(gettimeofday);

my $i = 0;
my $j = 0;
my $avg = 0;
foreach $filename (@ARGV) {
 #  print "Start to preprocess ", $filename, "...\n";
    open (TEMFILE, '<', $filename);
    while (<TEMFILE>) {
        $line = $_;
        chomp ($line);
        if ($line =~ /Runtime/) {
                ($v1, $v2) = split(': ', $line);
		chomp ($v2);
                ($v3, $v4) = split('seconds', $v2);
		chomp ($v3);
		$avg += $v3;
        }
   }
   close(TEMFILE);
   $j ++;
}
   if ($j != 0) {
   	printf("%.3f\n", $avg/$j);
   } else {
   	printf("Wrong input file!\n");
   }

