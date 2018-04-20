from TsvHandler import TsvHandler
from IntervalTree import IntervalTree
import numpy
from os import path
import sys

candidate_data_directory = "/Users/saureous/data/candidate_frequencies/GIAB/WG/"
candidate_data_filename = "candidate_frequencies_chr1-18_full.npz"
candidate_data_path = path.join(candidate_data_directory, candidate_data_filename)
regions_bed_path = "/Users/saureous/data/GIAB/NA12878_GRCh37_confident.bed"

tsv_handler = TsvHandler(regions_bed_path)

# collect intervals from BED in illumina PG standards and convert to intervals that make sense: 0-based, closed
bed_intervals_by_chromosome = tsv_handler.get_bed_intervals_by_chromosome(universal_offset=-1, start_offset=1)

interval_trees_by_chromosome = dict()

for chromosome in bed_intervals_by_chromosome:
    intervals = bed_intervals_by_chromosome[chromosome]

    interval_tree = IntervalTree(intervals)
    interval_trees_by_chromosome[chromosome] = interval_tree

print("chromosomes: ", bed_intervals_by_chromosome.keys())

data = numpy.load(candidate_data_path)['a']

coordinates = data[:, :2]

n = coordinates.shape[0]
confident_indexes = list()
for i in range(n):
    coord = coordinates[i]
    chromosome_number, position = map(int,coord)
    chromosome_name = str(chromosome_number)

    interval = [position, position]

    if i % 10000 == 0:
        completed_percentage = float(i)/n*100
        sys.stdout.write('\r')
        sys.stdout.write("%d%% completed" % completed_percentage)

    if interval in interval_trees_by_chromosome[chromosome_name]:
        confident_indexes.append(i)

print('\n')

confident_data = data[confident_indexes]

print("Original size:\t", n)
print("Subset size:\t", len(confident_indexes))

output_path = path.join(candidate_data_directory, "candidate_frequencies_chr1-18_confident.npz")

numpy.savez_compressed(output_path, a=confident_data)

