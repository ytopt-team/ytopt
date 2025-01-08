#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday); 

$A_FILE = "tmpfile.txt";
my $acc = -1;
foreach $filename (@ARGV) {
 #  print "Start to preprocess ", $filename, "...\n";
    system("python $filename > tmpfile.txt");
    open (TEMFILE, '<', $A_FILE);
    while (<TEMFILE>) {
        $line = $_;
        chomp ($line);

        if ($line =~ /Accuracy:/) {
                ($v1,$v2,$v3,$v4,$v5,$v6, $v7,$v8) = split(' ', $line);
		($v9,$v10) = split('/', $v7);
		$acc = $v10 / $v9;
#		print $acc;
        }
   }
   close(TEMFILE);
   if ($acc) {
	printf("%.3f", $acc)
   }
 #  system("unlink  tmpfile.txt");
}
