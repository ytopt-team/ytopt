#!/usr/bin/perl 
#Change this path!
#Author: Xingfu Wu
# MCS, ANL
# normal.pl: process the file into a normailzed data 
#

$A_FILE = "tmppower.txt";

my @tgpu;
foreach $filename (@ARGV) {
    my @pgpu;
    $nsample = 0;
    $gpu = 0;
    $avg = 0;
  
    #print "Start to process ", $filename, "...\n";
    $fname = "> " . $A_FILE;
    $i = 0;
    open(OUTFILE, $fname);
    open (TEMFILE, $filename);
    while (<TEMFILE>) {
	$line = $_;
	chomp ($line);

	if (($line =~ /gpu/) || ($line =~ /Idx/)) {
        } else {
		($v1, $v2, $v3, $v4, $v5) = split(' ', $line);
#		print OUTFILE $i, "\t", $v2, "\t", $v3, "\t", $v5, "\n";
#		print OUTFILE $v2, "\n";
		chomp($v2);
        	push(@pgpu, $v2);
#		print OUTFILE $i, "\t", $v2, "\n";
		$i++;
        }
   }
   close(TEMFILE);
 #calculate the average for power of node, cpu and mem
   $nsample = $#pgpu + 1;
   $gpu += $_ for @pgpu;
   $avg = $gpu / $nsample; 
   chomp($avg);
   push(@tgpu, $avg);
   #   print OUTFILE $i, "\t", $nsample, "\t", $avg, "\n";

#system("mv $A_FILE $filename");
#print "End to process ", $filename, "...\n";
}
$ns = $#tgpu + 1;
$agpu += $_ for @tgpu;
$tavg = $agpu / $ns;
print OUTFILE "Average power: ", $tavg, "\n";
close(OUTFILE);
#exit main
exit 0;
