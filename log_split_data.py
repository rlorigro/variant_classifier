from FileManager import FileManager
from TsvWriter import TsvWriter
import numpy
from tqdm import tqdm

parent_directory_path = "/home/ryan/data/GIAB/filter_model/chr1_19__0_all_1_coverage"
file_extension = ".npz"

file_paths = FileManager.get_all_filepaths_by_type(parent_directory_path=parent_directory_path,
                                                   file_extension=file_extension)

log_header = ["file_path", "length"]

log_writer = TsvWriter(output_directory=parent_directory_path,
                       header=log_header,
                       filename_prefix="dataset_log.tsv")

for path in tqdm(file_paths):
    data = numpy.load(path)['a']
    length = data.shape[1]

    if data.size > 0:
        log_writer.append_row([path,length])

