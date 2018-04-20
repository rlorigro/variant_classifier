from FileManager import FileManager
import os
import numpy
from concatenate import concatenate_directory_of_npz_files

master_directory = "/Users/saureous/data/candidate_frequencies/GIAB/WG/regional"

dir_names = list(map(str, range(1,19)))

all_data = list()
for dir_name in dir_names:
    dir_path = os.path.join(master_directory, dir_name)

    data = concatenate_directory_of_npz_files(dir_path)

    all_data.append(data)

all_data_concatenated = numpy.concatenate(all_data)

print(all_data_concatenated.shape)

output_dir_path = "/Users/saureous/data/candidate_frequencies/GIAB/WG/"
output_file_path = os.path.join(output_dir_path, "candidate_frequencies_Chr1-18_full.npz")

numpy.savez_compressed(output_file_path, a=all_data_concatenated)

