from TsvHandler import TsvHandler
from IntervalTree import IntervalTree
from FileManager import FileManager
import numpy
import os
from os import path
import sys
from tqdm import tqdm


def generate_interval_tree_from_bed_file(regions_bed_path):
    tsv_handler = TsvHandler(regions_bed_path)

    # collect intervals from BED in illumina PG standards and convert to intervals that make sense: 0-based, closed
    bed_intervals_by_chromosome = tsv_handler.get_bed_intervals_by_chromosome(universal_offset=-1, start_offset=1)

    interval_trees_by_chromosome = dict()

    for chromosome in bed_intervals_by_chromosome:
        intervals = bed_intervals_by_chromosome[chromosome]

        interval_tree = IntervalTree(intervals)
        interval_trees_by_chromosome[chromosome] = interval_tree

    print("chromosomes: ", bed_intervals_by_chromosome.keys())

    return interval_trees_by_chromosome


def subset_npz_file(candidate_data_directory, candidate_data_filename, output_directory, interval_trees_by_chromosome):
    data = numpy.load(path.join(candidate_data_directory, candidate_data_filename))['a']

    if data.size > 0:
        coordinates = data[:, :2]

        n = coordinates.shape[0]
        confident_indexes = list()
        for i in range(n):
            coord = coordinates[i]
            chromosome_number, position = map(int, coord)
            chromosome_name = str(chromosome_number)

            interval = [position, position]

            # if i % 10000 == 0:
            #     completed_percentage = float(i) / n * 100
            #     sys.stdout.write('\r')
            #     sys.stdout.write("%d%% completed" % completed_percentage)

            if interval in interval_trees_by_chromosome[chromosome_name]:
                confident_indexes.append(i)

        confident_data = data[confident_indexes]

        # print("Original size:\t", n)
        # print("Subset size:\t", len(confident_indexes))

        if not path.exists(output_directory):
            os.mkdir(output_directory)

        output_path = path.join(output_directory, "confident_" + candidate_data_filename)

        numpy.savez_compressed(output_path, a=confident_data)


def subset_all_files_in_directory(parent_directory, interval_trees_by_chromosome, output_directory):
    file_paths = FileManager.get_all_filepaths_by_type(parent_directory_path=parent_directory, file_extension=".npz")

    for f,file_path in tqdm(enumerate(file_paths)):
        directory, filename = path.split(file_path)

        if filename.startswith("confident"):
            print(directory, filename)

        subset_npz_file(candidate_data_directory=directory,
                        candidate_data_filename=filename,
                        interval_trees_by_chromosome=interval_trees_by_chromosome,
                        output_directory=output_directory)

        if f % 100 == 0:
            completed_percentage = float(f) / len(file_paths) * 100
            sys.stdout.write('\r')
            sys.stdout.write("%d%% completed" % completed_percentage)


if __name__ == "__main__":
    candidate_data_directory = "/home/ryan/data/GIAB/filter_model_training_data/vision/WG/0_threshold/confident/run_05032018_163935"
    output_directory = "/home/ryan/data/GIAB/filter_model_training_data/vision/WG/0_threshold/run_05022018_181731/confident"
    # candidate_data_filename = "candidate_frequencies_WG_full.npz"
    # candidate_data_path = path.join(candidate_data_directory, candidate_data_filename)
    regions_bed_path = "/home/ryan/data/GIAB/NA12878_GRCh37_confident.bed"

    interval_trees_by_chromosome = generate_interval_tree_from_bed_file(regions_bed_path)

    # subset_npz_file(candidate_data_directory, candidate_data_filename, interval_trees_by_chromosome)

    subset_all_files_in_directory(parent_directory=candidate_data_directory,
                                  interval_trees_by_chromosome=interval_trees_by_chromosome,
                                  output_directory=output_directory)

