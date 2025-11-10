#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday); 

$A_FILE = "tmpoutfile.txt";
foreach $filename (@ARGV) {
 #  print "Start to preprocess ", $filename, "...\n";
    system("srun -N 1 -n 1 --ntasks-per-gpu=1 --gpus-per-node=1 -c 10 --cpu-bind=sockets sh $filename > tmpoutfile.txt 2>&1");
    my $v3 = 0;
    open (TEMFILE, '<', $A_FILE);
    while (<TEMFILE>) {
        $line = $_;
        chomp ($line);

        if ($line =~ /Calculation Rate \(inactive\)/) {
                ($v1, $v2) = split('=', $line);
                ($v3, $v4) = split(' ', $v2);
		printf("%.3f", -1 * $v3);
        }
   }
   if ($v3 == 0 ) {
	printf("-1");
   }
   close(TEMFILE);
}
