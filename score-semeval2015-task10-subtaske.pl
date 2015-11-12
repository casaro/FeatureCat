#!/usr/bin/perl

################################################################################################################################
#  Author: Svetlana Kiritchenko
#  Information and Communications Technologies / Technologies de l'information et des communications
#  National Research Council Canada /Conseil national de recherches Canada
#  
#  Description: checks format/scores SemEval-2015 task 10, subtask E
#
#  Usage: score-semeval2015-task10-subtaskE.pl <file-predictions> <file-gold>
#  
#  The script checks the format of <file-predictions>.
#  The format of <file-predictions> should be the same as the format of the trial data file "subtaskE_trial_data.txt".
# 
#  If <file-gold> is provided, the scripts evaluates 
#  the predictions against the gold ratings and 
#  outputs the following statistics:
#  1) Kendall rank correlation coefficient
#  2) Spearman rank correlation coefficient 
#
#  To run the script, you need the package Statistics::RankCorrelation to be installed.
#
#
#  Last modified: July 10, 2014
#
#################################################################################################################################

use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

use Statistics::RankCorrelation;
use List::Util 'min';

die "Usage: score-semeval2015-task10-subtaskE.pl <file-predictions> <file-gold (optional)>\n" if(@ARGV < 1);


# read the input file with predicted scores
my $input_file =  $ARGV[0];
my %pred_scores = ();
read_file($input_file, \%pred_scores);

exit() if(@ARGV == 1);


# read the file with gold scores
my $gold_file =  $ARGV[1];
my %gold_scores = ();
read_file($gold_file, \%gold_scores);


# compare the predicted and gold ratings
my @gold_ratings = (), my @pred_ratings = ();
my $term;

# if a score is missing for a term in the file with predicted scores,
# assign the default score = minimal score - 1
my $default_score = min (values %pred_scores) - 1;

foreach $term (keys %gold_scores) {
	if(!defined $pred_scores{$term}) {
		print "ERROR: '$term' was not found in $input_file; assigning the default score of $default_score\n";
		push(@pred_ratings, $default_score);
	} else {
		push(@pred_ratings, $pred_scores{$term});
	}
	
	push(@gold_ratings, $gold_scores{$term});
}


# if all of the value in @pred_ratings are the same, set the correlation to zero
my $same  = 1;
for(my $i = 1; $i < @pred_ratings; $i++) {
  if ($pred_ratings[$i] != $pred_ratings[$i-1]) { 
    $same = 0;
    last;
  }
}

if ($same == 1) {
  print "\nAll values in $input_file are the same\n";
  print "Kendall rank correlation coefficient: 0\n";
  print "Spearman rank correlation coefficient: 0\n";
  exit();
}


my $cor = Statistics::RankCorrelation->new(\@gold_ratings, \@pred_ratings);

my $tau = $cor->kendall;
printf "\nKendall rank correlation coefficient: %.5f\n", $tau;

my $rho = $cor->spearman;
printf "Spearman rank correlation coefficient: %.5f\n", $rho;



# Read an input file
# Format: <term><tab><score>
sub read_file {
	my($file, $scores) = @_;
	my($term, $score, $count_lines);
	
	open INPUT, '<:encoding(UTF-8)', $file or die "ERROR: Cannot open the input file $file\n";
	while(<INPUT>) {
		s/^[ \t]+//;
		s/[ \t\n\r]+$//;
		
		die "ERROR: Wrong format in line: $_. Expected format: <term><tab><real-valued score>\n" if (!/^([^\t]+)\t(\-?\d+(\.\d+)?)$/);
		
		$term = $1; $score = $2;
		
		$term =~ s/\s+$//; 
		$scores->{$term} = $score;
		$count_lines++;
		
	}
	close INPUT;
	
	print STDERR "$file: the format is ok; $count_lines lines.\n";
}




