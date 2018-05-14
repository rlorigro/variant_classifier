import numpy
from matplotlib import pyplot

data_path = "/home/ryan/code/variant_classifier/output/WG_GIAB_chr_1-19_0-thresholds_1-coverage_2018-5-10-13-35-10-3-130/truth_vs_prediction.npz"

data = numpy.load(data_path)['a']


truth_labels = data[:,0]
predict_logits = data[:,1]
metadata = data[:,2:]

print(truth_labels[5:10])
print(predict_logits[5:10])

print()

sensitivities = list()
precisions = list()
threshold_labels = list()


range_size = 1000
for i in range(range_size-1):
    threshold = (float(1)/range_size)*(i+1)

    print()
    print(threshold)

    predict_labels = numpy.copy(predict_logits)

    zero_mask = (predict_labels < threshold)
    one_mask = (predict_labels > threshold)

    predict_labels[zero_mask] = 0
    predict_labels[one_mask] = 1

    # non_equivalency_mask = (predict_labels != truth_labels)
    true_positive_mask = (truth_labels == 1)
    true_negative_mask = (truth_labels == 0)

    n_true_positive = numpy.count_nonzero(predict_labels[true_positive_mask])
    n_false_positive = numpy.count_nonzero(predict_labels[true_negative_mask])
    n_true_negative = numpy.count_nonzero(true_negative_mask) - n_false_positive
    n_false_negative = numpy.count_nonzero(true_positive_mask) - n_true_positive

    print("TP: ", n_true_positive)
    print("FP: ", n_false_positive)
    print("TN: ", n_true_negative)
    print("FN: ", n_false_negative)

    if n_true_positive == 0:
        n_true_positive = 1e-15

    sensitivity = n_true_positive / (n_true_positive + n_false_negative)
    precision = n_true_positive / (n_true_positive + n_false_positive)

    print(sensitivity)
    print(precision)

    sensitivities.append(sensitivity)
    precisions.append(precision)

    if int(threshold*1000) % int(0.1*1000) < 1:
        threshold_labels.append([sensitivity,precision,round(threshold,1)])

    # print(zero_mask[5:10])
    # print(one_mask[5:10])
    # print(predict_labels[5:10])

print("precision")
print(precisions)
print("sensitivity")
print(sensitivities)

ax = pyplot.axes()
pyplot.gcf().set_size_inches(10,8)
ax.plot(sensitivities,precisions)
ax.set_ylabel("Precision")
ax.set_xlabel("Sensitivity")

# ax.set_xlim(0,1.0)
# ax.set_ylim(0,1.0)

for label in threshold_labels:
    x,y,t = label
    ax.text(x,y,"T="+str(t),ha="right",va="top")

pyplot.show()

