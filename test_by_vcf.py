from VcfHandler import VCFFileProcessor
from FileManager import FileManager
from IntervalTree import IntervalTree
import numpy


SNP, IN, DEL = 0, 1, 2

parent_directory_path = "/home/ryan/data/GIAB/filter_model_training_data/WG/0_threshold/split"
vcf_path = "/home/ryan/data/GIAB/NA12878_GRCh37.vcf.gz"
file_extension = ".npz"

file_paths = FileManager.get_all_filepaths_by_type(parent_directory_path=parent_directory_path,
                                                   file_extension=file_extension)

def get_positional_support(coordinates, frequency_data):
    length = coordinates.shape[0]

    positional_support = dict()
    for i in range(length):
        start_position = int(coordinates[i][1])

        # find the number of bins for each frequency vector, for each type of allele
        frequency_data_length = frequency_data.shape[1]
        n_alleles_per_type = int(frequency_data_length/3)

        snp_freq = frequency_data[i,:n_alleles_per_type]
        in_freq = frequency_data[i,n_alleles_per_type:n_alleles_per_type*2]
        del_freq = frequency_data[i,n_alleles_per_type*2:]

        # print(snp_freq, in_freq, del_freq)
        # print(snp_freq.shape, in_freq.shape, del_freq.shape)

        snp_supported = (numpy.sum(snp_freq) > 0)
        in_supported = (numpy.sum(in_freq) > 0)
        del_supported = (numpy.sum(del_freq) > 0)

        positional_support[start_position] = [snp_supported, in_supported, del_supported]

    return positional_support


def validate_positional_variants(positional_variants, positional_support):
    false_negatives = dict()

    for position in positional_variants:
        records = positional_variants[position]

        if position in positional_support:
            support = positional_support[position]
        else:
            support = [False, False, False]

        for type_index in [SNP, IN, DEL]:
            if len(records[type_index]) > 0:
                if not support[type_index]:
                    pass
                    print("NOT SUPPORTED:", position, type_index, records[type_index], support)
                    for record in records:
                        for variant in record:
                            print(variant)

        break


def main():
    for path in file_paths:
        data = numpy.load(path)['a']
        length = data.shape[0]

        x = data[:, 6:-1]
        y = data[:, -1:]
        coordinates = data[:,0:2]
        frequency_data = x[:,:-3]

        # print(x[0,:])
        # print()
        # print(frequency_data[0,:])

        chromosome_number = int(coordinates[0][0])
        start_position = int(coordinates[0][1])
        stop_position = int(coordinates[-1][1])

        chromosome_name = str(chromosome_number)

        print(chromosome_number, start_position, stop_position)

        positional_support = get_positional_support(coordinates, frequency_data)

        vcf_handler = VCFFileProcessor(vcf_path)
        vcf_handler.populate_dictionary(contig=chromosome_name,
                                        start_pos=start_position,
                                        end_pos=stop_position,
                                        hom_filter=True)

        positional_variants = vcf_handler.get_variant_dictionary()
        vcf_positions = sorted(positional_variants.keys())

        print(vcf_positions)

        validate_positional_variants(positional_variants, positional_support)


def get_vcf_intervals_by_allele_type(positional_variants):
    for position in positional_variants:
        snp_records, in_records, del_records = positional_variants[position]


        # print(positional_variants[position])


main()