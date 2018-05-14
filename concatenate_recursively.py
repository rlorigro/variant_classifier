from FileManager import FileManager
import os
import numpy
from concatenate import concatenate_directory_of_npz_files
import h5py

master_directory = "/home/ryan/data/GIAB/filter_model_training_data/WG/0_threshold"

dir_names = FileManager.get_subdirectories(master_directory)

all_data = list()
for dir_name in dir_names:
    dir_path = os.path.join(master_directory, dir_name)

    data = concatenate_directory_of_npz_files(dir_path)

    all_data.append(data)

all_data_concatenated = numpy.concatenate(all_data, axis=0)

print(all_data_concatenated.shape)
print(all_data_concatenated.size)

height = all_data_concatenated.shape[0]
width = all_data_concatenated.shape[1]

all_data_concatenated = all_data_concatenated.squeeze()

print(all_data_concatenated.shape)

output_dir_path = "/home/ryan/data/GIAB/filter_model_training_data/WG/0_threshold/"
output_file_path = os.path.join(output_dir_path, "candidate_frequencies_WG_full_0_percent_0_absolute.hdf5")

h5_file = h5py.File(output_file_path, 'w')
h5_dataset = h5_file.create_dataset("dataset", data=all_data_concatenated, chunks=(10000,width))

# numpy.savez_compressed(output_file_path, a=all_data_concatenated)

