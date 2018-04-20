import datetime
import sys
from os import path
import numpy
import torch
import torch.nn as nn
import random
from matplotlib import pyplot
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F


# results indices:
PERCENT_MATCHES, MATCHES, CONFUSION_VECTORS, N_FALSE_NEG, N_FALSE_POS, N_TRUE_NEG, N_TRUE_POS = 0,1,2,3,4,5,6


class ShallowLinear(nn.Module):
    '''
    A simple model to take candidate site data vectors and predict their status as a variant or non-variant
    '''
    def __init__(self):
        super(ShallowLinear, self).__init__()

        self.layer_sizes = [27, 64, 1]
        D_in, H1, D_out = self.layer_sizes

        print(D_in, H1, D_out)

        self.linear1 = nn.Linear(D_in, H1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(H1, D_out)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # print(x.size())

        x = self.linear2(x)
        x = self.sigmoid1(x)
        # print(x.size())

        return x


class Linear(nn.Module):
    '''
    A simple model to take candidate site data vectors and predict their status as a variant or non-variant
    '''
    def __init__(self):
        super(Linear, self).__init__()

        self.layer_sizes = [27, 1]
        D_in, D_out = self.layer_sizes

        print(D_in, D_out)

        self.linear1 = nn.Linear(D_in, D_out)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        # print(x.size())

        return x


class CandidateAlleleDataset(Dataset):
    def __init__(self, data):
        # check if path exists
        print(data.shape)

        x = data[:,6:-1]
        y = data[:,-1:]
        metadata = data[:,:5]
        # data_vcf_filter = data[,5:6]

        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor     # for MSE Loss or BCE loss
        # y_dtype = torch.LongTensor      # for CE Loss

        self.length = x.shape[0]

        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)

        self.metadata = metadata

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.metadata[index]

    def __len__(self):
        return self.length


def plot_loss(losses, show=True):
    ax = pyplot.axes()
    ax.set_xlabel("Batch (n=%d)")
    ax.set_ylabel("Loss")
    x_loss = list(range(len(losses)))
    pyplot.plot(x_loss, losses)

    if show:
        pyplot.show()

    pyplot.close()


def save_full_output(y, y_predict_logits, metadata, output_directory):
    data = numpy.concatenate((y, y_predict_logits, metadata), axis=1)

    print(data.shape)

    filename = "truth_vs_prediction.npz"
    write_dataset(output_dir=output_directory,
                  filename=filename,
                  data=data)


def write_dataset(output_dir, filename, data):
    """
    Create a npz training set of all labeled candidate sites found in the region
    :return:
    """
    #filename = "candidate_frequencies_confusion.npz"

    numpy.savez_compressed(path.join(output_dir, filename), a=data)


def train_batch(model, x, y, optimizer, loss_fn):
    # Run forward calculation
    y_predict = model.forward(x)

    # Compute loss.
    loss = loss_fn(y_predict, y)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    return loss.data[0]


def train(model, loader, optimizer, loss_fn, epochs=5, cutoff=None):
    losses = list()

    batch_index = 0
    i = 0
    for e in range(epochs):
        for x, y, metadata in loader:
            x = Variable(x)
            y = Variable(y)

            loss = train_batch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)

            batch_index += 1

            if i % 100 == 0:
                print(i, loss)

            i += 1

            if cutoff is not None and i > cutoff:
                break
        if cutoff is not None and i > cutoff:
            break

        print("Epoch: ", e+1)
        print("Batches: ", batch_index)

    return losses


def test_batch(model, x, y):
    # run forward calculation
    y_predict = model.forward(x)

    return y, y_predict


def test(model, loader):
    metadata_vectors = list()
    x_vectors = list()
    y_vectors = list()
    y_predict_vectors = list()

    batch_index = 0
    for x, y, metadata in loader:
        x = Variable(x)
        y = Variable(y)

        y, y_predict = test_batch(model=model, x=x, y=y)

        metadata_vectors.append(metadata)
        x_vectors.append(x.data.numpy())
        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

    y_predict_matrix = numpy.concatenate(y_predict_vectors)
    y_matrix = numpy.concatenate(y_vectors)
    x_matrix = numpy.concatenate(x_vectors)
    metadata = numpy.concatenate(metadata_vectors)

    return y_predict_matrix, y_matrix, x_matrix, metadata


# def grid_search(data_loader_train, data_loader_test):
#     a = numpy.arange(2,5.5,step=0.25)
#     parameter_range = [10**(-x) for x in a]
#
#     print("range: ", parameter_range)
#
#     accuracies = list()
#     losses = list()
#
#     for learning_rate in parameter_range:
#         for weight_decay in parameter_range:
#             shallow_model = ShallowLinear()
#
#             loss = train(model=shallow_model, loader=data_loader_train, learning_rate=learning_rate, weight_decay=weight_decay)
#             accuracy = test(model=shallow_model, loader=data_loader_test)
#
#             losses.append([learning_rate, weight_decay, loss])
#             accuracies.append([learning_rate, weight_decay, accuracy])
#
#             print('\t'.join(map(str, [learning_rate, weight_decay, accuracy])))
#             sys.stdout.flush()


def assess_prediction(y, y_predict):
    print(y[:1, :9])
    print(y_predict[:1, :9])

    y_argmax = numpy.argmax(y, axis=1)
    y_predict_argmax = numpy.argmax(y_predict, axis=1)

    print(y_argmax[:1])
    print(y_predict_argmax[:1])

    confusion = (y_argmax != y_predict_argmax).squeeze()
    n_confusion = sum(confusion)

    accuracy = 1-(n_confusion/y.shape[0])

    print("Accuracy: ", accuracy)

    pyplot.hist(y_predict_argmax, bins=list(range(46)))
    pyplot.show()
    pyplot.hist(y_argmax, bins=list(range(46)))
    pyplot.show()


def calculate_testing_stats(y_matrix, y_predict_matrix, x_matrix, metadata_matrix, output_directory, filter_false_positive=True, save_confusion_data=True):
    # convert labels to flattened 1d vector
    truth_labels = numpy.squeeze(y_matrix)
    predict_labels = numpy.squeeze(y_predict_matrix).round()  #assume 0.5 threshold

    # print(predict_labels[0:4])
    # print(predict_labels.round()[0:4])
    # print(truth_labels.shape, predict_labels.shape, x_matrix.shape)

    # get boolean mask vectors for entries that have correct vs incorrect predictions
    equivalency_mask_vector = (predict_labels == truth_labels)
    non_equivalency_mask_vector = numpy.invert(equivalency_mask_vector)

    # get vectors that had incorrect classifications
    confusion_vectors = x_matrix[non_equivalency_mask_vector,:]
    confusion_labels = truth_labels[non_equivalency_mask_vector]

    # get vectors that had correct classifications
    non_confusion_vectors = x_matrix[equivalency_mask_vector,:]
    non_confusion_labels = truth_labels[equivalency_mask_vector]

    # get boolean mask vectors for labels that are 1 or 0
    positive_mask_confusion = (confusion_labels == 1)
    negative_mask_confusion = (confusion_labels == 0)

    positive_mask_nonconfusion = (non_confusion_labels == 1)
    negative_mask_nonconfusion = (non_confusion_labels == 0)

    # get false positives and false negatives
    false_negative = confusion_labels[positive_mask_confusion]
    false_positive = confusion_labels[negative_mask_confusion]
    true_negative = non_confusion_labels[negative_mask_nonconfusion]
    true_positive = non_confusion_labels[positive_mask_nonconfusion]

    if save_confusion_data:
        confusion_metadata = metadata_matrix[non_equivalency_mask_vector,:]
        confusion_labels = confusion_labels.reshape((confusion_labels.shape[0],1))
        # print(confusion_metadata.shape, confusion_vectors.shape, confusion_labels.shape)
        confusion_data = numpy.concatenate((confusion_metadata, confusion_vectors, confusion_labels), axis=1)

        # print(confusion_data[1,:])

        filename = "candidate_frequencies_confusion.npz"
        write_dataset(output_dir=output_directory,
                      filename=filename,
                      data=confusion_data)

    n_false_negative = len(false_negative)
    n_false_positive = len(false_positive)
    n_true_negative = len(true_negative)
    n_true_positive = len(true_positive)

    length = len(confusion_labels)
    confusion_labels = confusion_labels.reshape((length, 1))

    # print(confusion_vectors.shape, confusion_labels.shape)

    labeled_confusion_vectors = numpy.concatenate((confusion_vectors, confusion_labels), axis=1)

    if filter_false_positive:
        labeled_confusion_vectors = labeled_confusion_vectors[numpy.squeeze(positive_mask_confusion),:]

    total_matches = numpy.sum(equivalency_mask_vector)
    percent_matches = float(total_matches)/len(truth_labels)

    return percent_matches, total_matches, labeled_confusion_vectors, n_false_negative, n_false_positive, n_true_negative, n_true_positive


def run(dataset_train, dataset_test):
    # Batch size is the number of training examples used to calculate each iteration's gradient
    batch_size_train = 128

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=False)

    # Define the hyperparameters
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Instantiate model
    model = Linear()

    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define the loss function
    loss_fn = nn.BCELoss()      # mean squared error

    # Train and get the resulting loss per iteration
    loss_per_iteration = train(model=model, loader=data_loader_train, optimizer=optimizer, loss_fn=loss_fn, epochs=1, cutoff=None)

    # Test and get the resulting predicted y values
    y_predict_matrix, y_matrix, x_matrix, metadata = test(model=model, loader=data_loader_test)

    parameters = list()

    parameters.append(learning_rate)
    parameters.append(weight_decay)
    parameters.append(weight_decay)

    return loss_per_iteration, y_predict_matrix, y_matrix, x_matrix, metadata, learning_rate, weight_decay, optimizer, loss_fn, model


def main():
    data_directory = "/Users/saureous/data/candidate_frequencies/GIAB/WG/"
    data_filename = "candidate_frequencies_chr1-18_confident.npz"
    data_path = path.join(data_directory, data_filename)

    data = numpy.load(data_path)['a']
    data = data.reshape((data.shape[0], data.shape[1]))
    print(data.shape)

    training_set_relative_size = 0.7

    n_total = data.shape[0]  # data set entries
    n_train = int(round(n_total*training_set_relative_size))  # training set entries

    all_indices = list(range(0, n_total))
    random.shuffle(all_indices)

    indices_train = all_indices[:n_train]
    indices_test = all_indices[n_train:]

    data_train = data[indices_train,:]
    data_test = data[indices_test,:]

    dataset_train = CandidateAlleleDataset(data=data_train)
    dataset_test = CandidateAlleleDataset(data=data_test)

    print("Train set size: ", dataset_train.length)
    print("Test set size: ", dataset_test.length)

    # grid_search(data_loader_train=data_loader_train, data_loader_test=data_loader_test)
    loss_per_iteration, \
    y_predict_matrix, \
    y_matrix, \
    x_matrix, \
    metadata_matrix, \
    learning_rate, \
    weight_decay, \
    optimizer, \
    loss_fn, \
    model = run(dataset_train=dataset_train,
                dataset_test=dataset_test)

    stats = calculate_testing_stats(y_matrix=y_matrix,
                                    y_predict_matrix=y_predict_matrix,
                                    x_matrix=x_matrix,
                                    metadata_matrix=metadata_matrix,
                                    output_directory=data_directory)

    total_false_negative = stats[N_FALSE_NEG]
    total_true_negative = stats[N_TRUE_NEG]
    total_false_positive = stats[N_FALSE_POS]
    total_true_positive = stats[N_TRUE_POS]

    print("learning rate: ", learning_rate)
    print("weight decay: ", weight_decay)
    print("optimizer: ", optimizer)
    print("loss function: ", loss_fn)
    print("model: ", model)

    print("Total sensitivity\t", float(total_true_positive)/(total_true_positive+total_false_negative))
    print("false negatives\t", total_false_negative)
    print("false positives\t", total_false_positive)
    print("true negatives\t", total_true_negative)
    print("true positives\t", total_true_positive)

    # save_full_output(output_logits_batches=all_predict_logits,
    #                  truth_labels_batches=all_truth_labels,
    #                  output_directory=output_directory,
    #                  metadata=all_metadata)


main()
