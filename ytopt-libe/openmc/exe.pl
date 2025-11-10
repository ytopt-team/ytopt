#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday); 

$A_FILE = "tmpoutfile.txt";
foreach $filename (@ARGV) {
 #  print "Start to preprocess ", $filename, "...\n";
    system("srun -N 1 -n 2 --ntasks-per-gpu=2 --gpus-per-node=1 -c 32 --cpu-bind=threads sh $filename > tmpoutfile.txt 2>&1");
    my $v3 = 0;
    open (TEMFILE, '<', $A_FILE);
    while (<TEMFILE>) {
        $line = $_;
        chomp ($line);

        if ($line =~ /Total time elapsed/) {
                ($v1, $v2) = split('=', $line);
                ($v3, $v4) = split(' ', $v2);
		printf("%.3f", $v3);
        }
   }
   if ($v3 == 0 ) {
	printf("500");
   }
   close(TEMFILE);
}
