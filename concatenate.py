from FileManager import FileManager
import os
import numpy

'''
Read a directory of NPZ files, concatenate their arrays, and save as a single array
'''

def concatenate_directory_of_npz_files(path):
    file_manager = FileManager()

    file_paths = file_manager.get_file_paths_from_directory(directory_path=path)

    training_sets = list()

    shapes = set()
    for path in file_paths:
        if path.endswith(".npz"):
            array = numpy.load(path)['a']

            # ignore empty arrays
            if len(array[0]) > 0:
                training_sets.append(numpy.load(path)['a'])
            else:
                print("omitted because empty: ", path)

            shape = '-'.join(list(map(str, array.shape)))
            shapes.add(shape)

    training_data = numpy.concatenate(training_sets, axis=0)

    return training_data


# training_data_dir_path = "/Users/saureous/data/candidate_frequencies/GIAB/chr1/regional/"
#
# output_dir_path = "/Users/saureous/data/candidate_frequencies/GIAB/chr1/"
# output_file_path = os.path.join(output_dir_path, "candidate_frequencies_chr1_full.npz")
#
# training_data = concatenate_directory_of_npz_files(path=training_data_dir_path)
#
# numpy.savez_compressed(output_file_path, a=training_data)

