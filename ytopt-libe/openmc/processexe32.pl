#!/usr/bin/perl
#Change this path!
#Author: Xingfu Wu
# MCS, ANL
# processexe.pl: process the file exe.pl to change tne proper number of nodes
#

$A_FILE = "tmpfile.txt";

$filename1 =  $ARGV[0];
    #print "Start to process ", $filename, "...\n";
    $fname = ">" . $A_FILE;
    $i = 0;
    open(OUTFILE, $fname);
    open (TEMFILE, $filename1);
    while (<TEMFILE>) {
        $line = $_;
        chomp ($line);

        if ($line =~ /system/) {
                print OUTFILE "    system(\"srun -N 32 -n ", $ARGV[2]*8*32, " --ntasks-per-gpu=", $ARGV[2], " --gpus-per-node=8 --gpu-bind=closest -c ", $ARGV[1], " --cpu-bind=", $ARGV[3]," sh \$filename > tmpoutfile.txt 2>&1\");", "\n";
	} else {
                print OUTFILE $line, "\n";
        }
    }
   close(TEMFILE);
   close(OUTFILE);
   system("mv $A_FILE $filename1");
   system("chmod +x $filename1");
#exit main
exit 0;
