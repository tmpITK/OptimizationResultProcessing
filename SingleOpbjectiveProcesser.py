from __future__ import print_function
import re
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import abc
import numpy as np
import xml.etree.ElementTree as ET

"""
NOTE: file formats used:
	-ind_file: 	generation, population size, (fitnes results), [resulting parameters]
	-final_archive_file: (fitnes results)
"""


class FileDoesNotExist(Exception):
    def __init__(self, missing_file, directory):
        message = "%s is not in %s" % (missing_file, directory)
        super(FileDoesNotExist, self).__init__(message)


class MetaOptimizationSettings:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_optimization_settings(self):
        raise NotImplementedError


class OptimizationSettings(MetaOptimizationSettings):
    '''
    Class for getting and storing the needed settings of the optimization process.
    '''

    def __init__(self, xml_file, directory):
        self.LAST_ELEMENT_INDEX = -1
	self.EXTENSION_INDEX = -4

        self.directory = directory

        try:
            self.xml = ET.parse(self.directory + xml_file)
        except IOError:
            raise FileDoesNotExist(xml_file, self.directory)

        self.get_optimization_settings()
        self.number_of_objectives = len(self.features)

    def get_optimization_settings(self):
        root = self.xml.getroot()

        for child in root:
            if child.tag == "evo_strat":
                self.algorithm_name = child.text
            if child.tag == "model_path":
                self.model_name = child.text.split('/')[self.LAST_ELEMENT_INDEX]
		self.model_name = self.model_name[:self.EXTENSION_INDEX] #remove .hoc
            if child.tag == "max_evaluation":
                self.number_of_generations = int(float(child.text))
            if child.tag == "pop_size":
                self.population_size = int(float(child.text))
            if child.tag == "num_params":
                self.number_of_parameters = int(child.text)
            if child.tag == "feats":
                self.features = child.text.split(", ")
            if child.tag == "weights":
                self.weights = map(self._float_or_int, child.text.strip().lstrip("[").rstrip("]").split(","))

    @staticmethod
    def _float_or_int(value):
        try:
            integer = int(value)
            return integer
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return unicode(value.strip("u").strip('\''))

    def getParams(self):
        print(self.__dict__.keys())


class RawSingleObjectiveResult(OptimizationSettings):

    def __init__(self, xml_file="_settings.xml", directory='', stat_file='stat_file.txt'):
        OptimizationSettings.__init__(self, xml_file, directory)
        self.stat_file = self.directory + stat_file
        self.statistics = []

        self.parse_stat_file()

    def parse_stat_file(self):
        with open(self.stat_file, 'rb') as f:
            for generation in iter(f):
                self.fill_in_statistics_list(self.split_and_convert_values(generation))

    def split_and_convert_values(self, generation):
        return [float(value) for value in generation.split(", ")]

    def fill_in_statistics_list(self, generation):
        current_statistics = self.get_wanted_statistics(generation)
        if self.algorithm_name == 'PSO':
            if len(self.statistics)==0:
                self.statistics.append(current_statistics)
            else:
                smoothed_statistics = self.smooth_statistics(current_statistics)
                self.statistics.append(smoothed_statistics)
        else:
            self.statistics.append(current_statistics)


    def get_wanted_statistics(self, generation):
        """
        Reminder: Inspyred stat file format: generation index, pop size, max, min, mean, median, std
        """
        MAX_INDEX = 2
        MIN_INDEX = 3
        MEDIAN_INDEX = 5

        return [generation[MAX_INDEX], generation[MIN_INDEX], generation[MEDIAN_INDEX]]

    def smooth_statistics(self, current_statistics):
        smoothed_statistics = []
        for stat_index, stat_component in enumerate(current_statistics):
            current_best = min([stat[stat_index] for stat in self.statistics])
            if stat_component < current_best:
                smoothed_statistics.append(stat_component)
            else:
                smoothed_statistics.append(current_best)
        return smoothed_statistics

class GeneralPlotter(object):
    def __init__(self, algorithm_name, model_name, directory, features=''):
        self.algorithm_name = algorithm_name
        self.model_name = model_name
        self.directory = directory
        self.features = features

    def create_generation_plot(self, statistics, title=''):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot([row[0] for row in statistics], 'r.-', label="max", linewidth=1.5)
        ax.plot([row[1] for row in statistics], 'r--', label="min", linewidth=1.5)
        ax.plot([row[2] for row in statistics], 'r', label="median", linewidth=1.5)

        ax.set_xlim(0, len(statistics))

        fig.suptitle('{0}{1} on {2}'.format(title, self.algorithm_name, self.model_name))
        plt.xlabel('generations')
        plt.ylabel('score value')
        plt.yscale('log')

        plt.legend(loc='best', fontsize=14, ncol=1)
        plt.savefig(self.directory + '{0}{1}_on_{2}'.format(title, self.algorithm_name, self.model_name), format='png')
        plt.close()

    def create_min_plot_of_all_runs(self, all_minimums_of_all_runs):
        number_of_runs = len(all_minimums_of_all_runs)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for run in range(number_of_runs):
            ax.plot(all_minimums_of_all_runs[run], label=str(run))
        ax.set_xlim(0, len(all_minimums_of_all_runs[0]))

        plt.xlabel('generations')
        plt.ylabel('score value')
        plt.legend(loc='best')

        plt.savefig(
            self.directory + '{0}_runs_of_{1}_on_{2}'.format(number_of_runs, self.algorithm_name, self.model_name),
            format='png')
        plt.close()

    @staticmethod
    def tune_limit(values):
        return (max(values) - min(values)) / 100


def get_directories(directory_base_name):
    regex = re.compile(directory_base_name + '_.')
    all_elements_in_cwd = [element for element in os.listdir(cwd) if os.path.isdir(element)]

    return [directory + '/' for directory in all_elements_in_cwd if re.match(regex, directory)]


def fill_statistics_for_all_runs(all_minimums_of_all_runs):
    all_statistics_of_all_runs = []
    for i in range(len(all_minimums_of_all_runs[0])):
        current_column = [column[i] for column in all_minimums_of_all_runs]
        all_statistics_of_all_runs.append(calculate_statistics(current_column))
    return all_statistics_of_all_runs


def calculate_statistics(data):
    maximum = max(data)
    minimum = min(data)
    median = np.median(data)
    return [maximum, minimum, median]


def write_separate_statistics_to_separate_files(all_statistics_of_all_runs, cwd):
    STAT_TYPES = ["max", "min", "median"]

    for index, stat_type in enumerate(STAT_TYPES):
        with open("{0}{1}.txt".format(results_directory, stat_type), "wb") as f:
            for stats in all_statistics_of_all_runs:
                f.write('%s\n' % stats[index])


def write_statistics_to_file(directory, statistics, population_size):
    with open(directory + "all_stat_file.txt", "wb") as f:
        for index, generation in enumerate(statistics):
            f.write('%d, %d, %s\n' % (index, population_size, ", ".join(map(str, generation))))


def create_directory_for_results(cwd):
    results_dir = cwd + '/results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir + '/'


if __name__ == '__main__':
    start_time = time.time()
    INDEX_OF_MINIMUM = 1
    cwd = os.getcwd()
    results_directory = create_directory_for_results(cwd)

    # Parameters for every run (only initialized the variables)
    algorithm_name = ''
    population_size = 0
    model_name = ''

    # This must be give to the script by hand: what is the base directory name of the results
    base_directory = 'hh_pas_surrogate'
    directories = get_directories(base_directory)

    all_minimums_of_all_runs = []
    # NSGAII on HODGKIN-HUXLEY
    for instance_index, directory in enumerate(directories):
        print(directory)
        single_objective_result = RawSingleObjectiveResult(directory=directory)
        all_minimums_of_all_runs.append([row[INDEX_OF_MINIMUM] for row in single_objective_result.statistics])

        if instance_index == 0:
            algorithm_name = single_objective_result.algorithm_name
            population_size = single_objective_result.population_size
            model_name = single_objective_result.model_name

    all_statistics_of_all_runs = fill_statistics_for_all_runs(all_minimums_of_all_runs)
    write_separate_statistics_to_separate_files(all_statistics_of_all_runs, results_directory)
    write_statistics_to_file((results_directory), all_statistics_of_all_runs, population_size)

    plotter = GeneralPlotter(algorithm_name, model_name, directory=(results_directory))
    plotter.create_min_plot_of_all_runs(all_minimums_of_all_runs)
    plotter.create_generation_plot(all_statistics_of_all_runs, title="Statistics_of_every_run_of_")

    print("--- %s seconds ---" % (time.time() - start_time))
