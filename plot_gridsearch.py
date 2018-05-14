from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from IterativeAverage import IterativeAverage
import csv
import math
from pandas import DataFrame
import pandas
import sys


def harmonic_mean(a):
    n = len(a)
    a_inverse = [1.0/value for value in a]
    harmonic_mean = n/(sum(a_inverse))

    return harmonic_mean


def generate_harmonic_mean_column(gridsearch_dataframe, column_name="f1 score"):
    gridsearch_dataframe[column_name] = gridsearch_dataframe.apply(
        lambda row: harmonic_mean([row["precision"],row["sensitivity"]]), axis=1)

    return gridsearch_dataframe


def get_grouped_column_data(dataframe, group_names, target_data_name):
    '''
    Take a dataframe representing a function of 2 variables and multiple outputs that depend on those variables,
    with some amount of randomness. Group outputs of the same 2 parameters, find their average and standard deviation,
    and return x,y,z,z_std_error lists, with z/z_std_error corresponding to the <target_data_name> dependent variable.

    also returns z min and z max 3-tuples.

    :param dataframe: A dataframe with 2 parameters and some number of dependent variables
    :param group_names: the parameter names which determine the value of the dependent variable
    :param target_data_name: the name of the dependent variable of interest
    :return: x, y, z, z_std_error
    '''
    grouped_results = dataframe.groupby(group_names)
    average_results = grouped_results.mean()
    stdev_results = grouped_results.std()

    min_z_value = sys.maxsize
    max_z_value = 0
    coord_z_max, coord_z_min = None, None
    x,y,z = list(), list(), list()
    z_std_error = list()

    for average_data, std_data in zip(average_results.iterrows(), stdev_results.iterrows()):
        avg_index, avg_row = average_data
        std_index, std_row = std_data

        x_parameter, y_parameter = avg_row.name
        x_parameter = math.log10(x_parameter)
        y_parameter = math.log10(y_parameter)

        average = avg_row[target_data_name]
        std_error = std_row[target_data_name]

        x.append(x_parameter)
        y.append(y_parameter)
        z.append(average)
        z_std_error.append(std_error)

        if average > max_z_value:
            max_z_value = average
            coord_z_max = (x_parameter, y_parameter, average)

        if average < min_z_value:
            min_z_value = average
            coord_z_min = (x_parameter, y_parameter, average)

    return x, y, z, z_std_error, coord_z_max, coord_z_min


def plot_surface_with_error_bars(x, y, z, z_std_error, coord_z_max, coord_z_min, z_label="value", animate=False):
    fig = pyplot.figure()
    fig.set_size_inches(10, 8)

    ax = Axes3D(fig)
    ax.set_xlabel("Learning Rate (log10)")
    ax.set_ylabel("Weight Decay (log10)")
    ax.set_zlabel(z_label)
    ax.set_title(z_label)
    ax.set_zlim([coord_z_min[2]*0.95, coord_z_max[2]*1.05])

    for i in range(len(x)):
        x_i = x[i]
        y_i = y[i]
        avg = z[i]
        std_deviation = z_std_error[i]

        lower_bound = avg - 2*std_deviation
        upper_bound = avg + 2*std_deviation

        ax.plot([x_i,x_i], [y_i,y_i], [lower_bound,upper_bound], marker="_", color='k', alpha=0.5, zorder=2)

    points = ax.scatter(x, y, z, linewidth=0, color='k', zorder=sys.maxsize)
    surface = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.7, zorder=1)

    ax.text(text=". Max",
            x=coord_z_max[0],
            y=coord_z_max[1],
            z=coord_z_max[2],
            color=[0.05, 0.6, 0.05],
            s=100,
            zorder=sys.maxsize)

    ax.text(text=". Min",
            x=coord_z_min[0],
            y=coord_z_min[1],
            z=coord_z_min[2],
            color=[0.05, 0.6, 0.05],
            s=100,
            zorder=sys.maxsize)

    if not animate:
        # display static plot
        ax.view_init(30, 75)
        pyplot.show()

    else:
        # display animate rotating plot
        for angle in range(0, 360):
            ax.set_title(str(angle))
            ax.view_init(30, angle)
            pyplot.draw()
            pyplot.pause(.001)

    return fig, ax


tsv_file_path = "/home/ryan/code/variant_classifier/output/gridsearch_results/gridsearch_results_2018-4-20-17-56-43-4-110"

gridsearch_results = pandas.read_csv(tsv_file_path, sep='\t')

gridsearch_results = generate_harmonic_mean_column(gridsearch_results)

target_data_name = "false negative"

x, y, z, z_std_error, coord_z_max, coord_z_min = get_grouped_column_data(dataframe=gridsearch_results,
                                                                         group_names=["learning rate", "weight decay"],
                                                                         target_data_name=target_data_name)

print("max (log10):\t", coord_z_max)
print("min (log10):\t", coord_z_min)
print("max:\t", (10**coord_z_max[0], 10**coord_z_max[1], coord_z_max[2]))
print("min:\t", (10**coord_z_min[0], 10**coord_z_min[1], coord_z_min[2]))

plot_surface_with_error_bars(x=x,
                             y=y,
                             z=z,
                             z_std_error=z_std_error,
                             coord_z_max=coord_z_max,
                             coord_z_min=coord_z_min,
                             z_label=target_data_name)
