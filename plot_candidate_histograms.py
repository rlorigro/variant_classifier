import numpy
from matplotlib import pyplot
import os

numpy.set_printoptions(suppress=True, threshold=numpy.nan, linewidth=200, formatter={'float_kind':'{:0.3f}'.format})


MISMATCH, INSERT, DELETE = 0,1,2
HOM, HET, HOM_ALT = 0,1,2

N_TOP_ALLELES = 1

GT_NAMES = ["hom_ref", "het", "hom_alt"]
TYPE_NAMES = ["mismatch", "insert", "delete"]

OUTLIER_THRESHOLDS_UPPER = [0.4, 0.8, 1.0]
OUTLIER_THRESHOLDS_LOWER = [0.0, 0.2, 0.4]

FIND_OUTLIERS = True

FILTER_BY_DEPTH = False
FILTER_BY_BASE_QUALITY = False
FILTER_BY_VCF_QUALITY_FIELD = False

DEPTH_THRESHOLD = 30
BASE_QUALITY_THRESHOLD = 50

def read_data(data_path):
    """
    Get freguency and label vectors for subsets of candidate sites based whether they are an insert or not
    :param data_path: path to NPZ file containing all vectors
    :return:
    """
    # shape = (1482008,26,1)
    data = numpy.load(data_path)['a']
    data = data.reshape(data.shape[0],data.shape[1])

    if FILTER_BY_VCF_QUALITY_FIELD:
        mask_vector = (data[:,5] == 1).squeeze()
        data = data[mask_vector,:]

    if FILTER_BY_DEPTH:
        mask_vector = (data[:,-2] > float(DEPTH_THRESHOLD)/1000).squeeze()

        # print(len(depth_mask_vector))
        # print(depth_mask_vector[1:25])
        # print(data[1:25,2])

        data = data[mask_vector, :]

    if FILTER_BY_BASE_QUALITY:
        mask_vector = (data[:,-3] > float(BASE_QUALITY_THRESHOLD)/1000).squeeze()

        print(len(mask_vector))
        print(mask_vector[1:25])
        print(data[1:25,-3])

        data = data[mask_vector, :]

    freq_vectors = data[:,6:-4] # starts at 6 for raw data, 5 for output from testing

    mismatch_freq_vectors = freq_vectors[:,:8]
    insert_freq_vectors = freq_vectors[:,8:16]
    delete_freq_vectors = freq_vectors[:,16:]

    print(mismatch_freq_vectors.shape, insert_freq_vectors.shape, delete_freq_vectors.shape)
    print(freq_vectors[1,:])
    print(mismatch_freq_vectors[1,:])
    print(insert_freq_vectors[1,:])
    print(delete_freq_vectors[1,:])

    mismatch_labels = data[:,2]
    insert_labels = data[:,3]
    delete_labels = data[:,4]

    coordinates = data[:,:2]
    depth = data[:,-2]

    # print(data.shape)
    # print(data[8000:8500,:])

    # print(insert_labels[8000:8500])

    n_total = data.shape[0]  # data set entries

    freq_vectors_by_type = [mismatch_freq_vectors, insert_freq_vectors, delete_freq_vectors]
    labels_by_type = [mismatch_labels, insert_labels, delete_labels]

    return coordinates, freq_vectors_by_type, labels_by_type, depth


def get_outlier_sites(candidate_frequency, candidate_coordinates, candidate_depths, upper_threshold, lower_threshold):
    mask_vector_lower_threshold = (candidate_frequency < lower_threshold).squeeze()
    mask_vector_upper_threshold = (candidate_frequency > upper_threshold).squeeze()

    data = [candidate_coordinates, candidate_frequency, candidate_depths]
    data = numpy.concatenate(data, axis=1)

    lower_outlier_frequencies = data[mask_vector_lower_threshold, :]
    upper_outlier_frequencies = data[mask_vector_upper_threshold, :]

    return lower_outlier_frequencies, upper_outlier_frequencies


def histograms_from_data(freq_vectors, labels, coordinates, depth, multiplier=1.0):
    """
    Generate histogram data corresponding to groups of frequency vectors that are inserts/non-inserts or hom/het/hom_alt
    :param freq_vectors: an iterable of 2 numpy 2d arrays: insert and non-insert, containing allele frequencies
    :param labels: an iterable of 2 numpy 2d arrays: insert and non-insert, containing allele labels
    :return:
    """
    types = [MISMATCH, INSERT, DELETE]
    genotypes = [HOM, HET, HOM_ALT]

    histograms = [[list() for genotype in genotypes] for type in types]  # 0 = non_insert, 1 = insert,
    # outliers = [[list() for genotype in genotypes] for type in types]  # 0 = non_insert, 1 = insert,
    # qualities = [[list() for genotype in genotypes] for type in types]

    bins = numpy.arange(0, 1, step=0.02)

    for type in types:
        # get data pertaining to individual mutation classes only (SNP/IN/DEL)
        vectors_subset = freq_vectors[type]
        labels_subset = labels[type]

        for gt in genotypes:
            mask_vector = (labels_subset == gt).squeeze()

            # get the data pertaining to individual genotypes only (hom/het/hom_alt)
            vectors_subset_gt = vectors_subset[mask_vector,:]
            coordinates_subset_gt = coordinates[mask_vector,:]
            depth_subset_gt = depth.reshape(depth.shape[0],1)[mask_vector,:]    # keep 1D array as 2D with dim_2 = 1

            print(vectors_subset_gt.shape)
            print(coordinates_subset_gt.shape)
            print(depth_subset_gt.shape)

            for i in range(N_TOP_ALLELES):

                frequencies, bins = numpy.histogram(vectors_subset_gt[:,i], bins=bins)

                if FIND_OUTLIERS:
                    # collect a 1D vector for frequencies
                    top_allele_frequency = vectors_subset_gt[:, i].reshape(vectors_subset_gt.shape[0], 1)
                    threshold_upper = OUTLIER_THRESHOLDS_UPPER[gt]
                    threshold_lower = OUTLIER_THRESHOLDS_LOWER[gt]

                    lower_outliers, upper_outliers = get_outlier_sites(candidate_frequency=top_allele_frequency,
                                                                       candidate_coordinates=coordinates_subset_gt,
                                                                       candidate_depths=depth_subset_gt,
                                                                       lower_threshold=threshold_lower,
                                                                       upper_threshold=threshold_upper)

                    print()
                    print(TYPE_NAMES[type], GT_NAMES[gt], "greater than: ", threshold_upper)
                    print(upper_outliers[:, :])
                    print()
                    print(TYPE_NAMES[type], GT_NAMES[gt], "less than: ", threshold_lower)
                    print(lower_outliers[:, :])

                frequencies = numpy.log10(frequencies)
                frequencies *= multiplier

                histograms[type][gt].append([frequencies, bins])

    return histograms


def plot_histograms(histograms, output_dir, filename):
    """
    Plot a grid of histograms corresponding to each of the subtypes of candidates: insert/non-insert, hom/het/hom_alt
    :param histograms:
    :return:
    """
    types = [MISMATCH, INSERT, DELETE]
    genotypes = [HOM, HET, HOM_ALT]

    # colors = [(1.0, 0.05, 0.0),
    #           (1.0, 0.47, 0.0),
    #           (1.0, 0.68, 0.0)]

    colors = [(0.0, 0.60, 0.53),
              (0.96, 0.62, 0.0),
              (0.94, 0.0, 0.7)]

    fig, axes = pyplot.subplots(len(types), len(genotypes), sharex=True, sharey=True)

    for type in types:
        # print(type)
        for gt in genotypes:
            # print(gt)
            for i in range(N_TOP_ALLELES):
                frequencies, bins = histograms[type][gt][i]

                center = (bins[:-1]+bins[1:])/2

                # print(bins)
                # print(frequencies)

                axes[type][gt].bar(center, frequencies, align="center", width=1.0/50, color=colors[i], alpha=0.60)
                axes[type][gt].set_xlim(0, 1.0)
                axes[type][gt].set_ylim(0, 6)

                if gt == 0 and type == 0:
                    axes[type][gt].set_ylabel("Mismatch Freq. (Log10)")
                if gt == 0 and type == 1:
                    axes[type][gt].set_ylabel("Insert Freq. (Log10)")
                if gt == 0 and type == 2:
                    axes[type][gt].set_ylabel("Delete Freq. (Log10)")
                if gt == 1 and type == 2:
                    axes[type][gt].set_xlabel("Allele frequency")
                if type == 0:
                    if gt == HOM:
                        axes[type][gt].set_title("Homozygous")
                    if gt == HET:
                        axes[type][gt].set_title("Heterozygous")
                    if gt == HOM_ALT:
                        axes[type][gt].set_title("Homozygous Alternate")

                            # axes[type][gt].set_yscale('log')

    # pyplot.show()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_path = os.path.join(output_dir, filename)

    fig = pyplot.gcf()
    # fig.set_size_inches(10, 8)
    # fig.savefig(output_path, dpi=150, transparent=True)

    pyplot.show()


# data_path = "/Users/saureous/data/candidate_frequencies/chr1/quality/candidate_frequencies_chr1_full.npz"
# data_path = "/Users/saureous/data/candidate_frequencies/GIAB/chr1/candidate_frequencies_chr1_full.npz"
data_path = "/Users/saureous/data/candidate_frequencies/GIAB/chr1/candidate_frequencies_chr1_confident.npz"

output_dir_figure = "histograms/"
filename = "candidate_frequency_distributions_30.png"

coordinates, vectors, labels, depth = read_data(data_path=data_path)
histograms = histograms_from_data(coordinates=coordinates, freq_vectors=vectors, labels=labels, depth=depth)
plot_histograms(histograms=histograms, output_dir=output_dir_figure, filename=filename)

