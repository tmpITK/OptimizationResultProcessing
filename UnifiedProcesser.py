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

    def __init__(self, xml_file="_settings.xml", directory=''):
        self.LAST_ELEMENT_INDEX = -1

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


class RawMultiObjectiveOptimizationResult(OptimizationSettings):
    '''
    Class for getting and storing the result of the optimization procces.
    '''

    def __init__(self, xml_file="_settings.xml", directory='', ind_file="ind_file.txt"):
        OptimizationSettings.__init__(self, xml_file, directory)
        self.ind_file = self.directory + ind_file
        self.generations = []
        self.parse_individual_file()

    def parse_individual_file(self):
        with open(self.ind_file, 'rb') as f:
            current_generation = []
            for individual in iter(f):
                current_generation.append(self.split_values_of_individuals(individual)[2:])
                if self.is_end_of_generation(len(current_generation)):
                    self.save_generation(current_generation)
                    current_generation = []

    @staticmethod
    def remove_unwanted_characters(element):
        remove_these_chars = ['(', ')', '[', ']', ',']
        for char in remove_these_chars:
            if char in element:
                element = element.replace(char, '')
        return element

    def split_values_of_individuals(self, new_individual):
        return [float(self.remove_unwanted_characters(value)) for value in new_individual.split()]

    def is_end_of_generation(self, length_of_generation):
        return length_of_generation == self.population_size

    def save_generation(self, new_generation):
        self.generations.append(new_generation)

    def print_untouched_generations(self):
        self.print_given_generations(self.generations)

    @staticmethod
    def print_given_generations(generations):
        for generation in generations:
            print(*generation, sep='\n')


class WeightedMooResult(RawMultiObjectiveOptimizationResult):
    """
    Class for calculating the weighted sum of the objective functions' fitness values
    and sorting the generations by this sum to get the individual with the lowest
    fitness score.
    """

    def __init__(self, xml_file="_settings.xml", directory='', ind_file='ind_file.txt'):
        RawMultiObjectiveOptimizationResult.__init__(self, xml_file, directory, ind_file)

        self.INDEX_OF_WEIGHTED_SUM = 0
        self.OFFSET = 0  # ind_file format! -> generation and individual index
        self.NEW_OFFSET = 1  # we insert the weighted sum!
        self.plotter = GeneralPlotter(self.algorithm_name, self.model_name, self.directory)

        self.sorted_generations = []
        self.statistics = []

        self.insert_weighted_sums()
        self.sort_individuals_by_weighted_sum()
        self.write_sorted_individuals_to_file()
        self.fill_statistics_list()
        self.write_statistics_to_file(self.directory, self.statistics, self.population_size)
        self.plot_statistics()

    def insert_weighted_sums(self):
        for generation in self.generations:
            index_of_original_generation = self.generations.index(generation)

            for individual in generation:
                index_of_original_individual = generation.index(individual)
                weighted_sum = self.calculate_weighted_sum(self.get_objectives_of_individual(individual))
                individual.insert(self.INDEX_OF_WEIGHTED_SUM, weighted_sum)
                generation[index_of_original_individual] = individual

            self.generations[index_of_original_generation] = generation

    def get_objectives_of_individual(self, individual):
        objectives = individual[:self.number_of_objectives]
        return objectives

    def calculate_weighted_sum(self, objectives):
        return sum([obj * weight for obj, weight in zip(objectives, self.weights)])

    def sort_individuals_by_weighted_sum(self):
        for generation in self.generations:
            self.sorted_generations.append(sorted(generation, key=lambda x: x[self.INDEX_OF_WEIGHTED_SUM]))

    def fill_statistics_list(self):
        for generation in self.sorted_generations:
            current_generation = [row[self.INDEX_OF_WEIGHTED_SUM] for row in generation]
            self.statistics.append(self.calculate_statistics(current_generation))

    @staticmethod
    def calculate_statistics(data):
        maximum = max(data)
        minimum = min(data)
        median = np.median(data)
        return [maximum, minimum, median]

    @staticmethod
    def write_statistics_to_file(directory, statistics, population_size):
        with open(directory + "sorted_stat_file.txt", "wb") as f:
            for index, generation in enumerate(statistics):
                f.write('%d, %d, %s\n' % (index, population_size, ", ".join(map(str, generation))))

    def write_sorted_individuals_to_file(self):
        with open(self.directory + "sorted_ind_file.txt", "wb") as f:
            for gen_i, generation in enumerate(self.sorted_generations):
                for ind_i, individual in enumerate(generation):
                    f.write("%s\n" % self.format_individual_for_writing(individual, gen_i, ind_i))

    def format_individual_for_writing(self, individual, gen_i, ind_i):
        individual = [str(value) for value in individual]

        indexes = [str(gen_i), str(ind_i)]
        weighted_sum = individual[self.INDEX_OF_WEIGHTED_SUM]
        objectives = individual[self.NEW_OFFSET:self.NEW_OFFSET + self.number_of_objectives]
        parameters = individual[self.NEW_OFFSET + self.number_of_objectives:]

        return "{0}, {1}, ({2}), [{3}]".format(", ".join(indexes), ", ".join(weighted_sum), ", ".join(objectives),
                                               ", ".join(parameters))

    def plot_statistics(self):
        self.plotter.create_generation_plot(self.statistics)


class NormalMooResult(RawMultiObjectiveOptimizationResult):
    """
    Class for getting and storing the archive of multi objective optimization results
    and plotting the Pareto Front.
    """

    def __init__(self, xml_file="_settings.xml", directory='', ind_file="ind_file.txt", final_archive_file="final_archive.txt"):
        RawMultiObjectiveOptimizationResult.__init__(self, xml_file, directory, ind_file)

        self.OFFSET = 1
        self.LAST_ELEMENT_INDEX = -1

        self.plotter = GeneralPlotter(self.algorithm_name, self.model_name, self.directory, self.features)
        self.final_archive_file = self.directory + final_archive_file
        self.final_archive = []
        self.final_generation = []
        self.final_generation_objectives = []

        self.separate_final_generation()
        self.separate_final_generation_objectives()
        self.parse_final_archive_file()
        if self.number_of_objectives in range(1, 4):
            self.plot_pareto_front()

    def separate_final_generation(self):
        self.final_generation = self.generations[self.LAST_ELEMENT_INDEX]

    def separate_final_generation_objectives(self):
        self.final_generation_objectives = [individual[self.OFFSET:self.OFFSET + self.number_of_objectives] for
                                            individual in self.final_generation]

    def parse_final_archive_file(self):
        with open(self.final_archive_file, 'rb') as arc_file:
            for individual in iter(arc_file):
                self.save_archived_individual_objectives(self.split_values_of_individuals(individual))

    def save_archived_individual_objectives(self, archived_individual):
        self.final_archive.append(archived_individual)

    def plot_pareto_front(self):
        self.plotter.create_pareto_plot(self.final_archive)

        if self.algorithm_name == "PAES":
            plotter.create_pareto_plot(self.final_generation_objectives, "Final Generation")


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
        plt.savefig(self.directory + '{0}{1}_on_{2}'.format(title, self.algorithm_name, self.model_name), format='pdf')
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
            format='pdf')
        plt.close()

    def create_pareto_plot(self, best_individuals, title="Pareto Front"):
        number_of_objectives = len(self.features)
        OBJECTIVE_NUMBER = range(number_of_objectives)

        x = []
        y = []
        for individual in best_individuals:
            x.append(individual[OBJECTIVE_NUMBER[0]])
            y.append(individual[OBJECTIVE_NUMBER[1]])

        if number_of_objectives > 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            z = [row[OBJECTIVE_NUMBER[2]] for row in best_individuals]
            tuner_z = self.tune_limit(z)
            ax.scatter(x, y, z, color='b')
            ax.set_zlim(min(z) - tuner_z, max(z) + tuner_z)
            ax.set_zlabel(self.features[OBJECTIVE_NUMBER[2]])
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(x, y, color='b')

        tuner_x = self.tune_limit(x)
        tuner_y = self.tune_limit(y)
        ax.set_xlim(min(x) - tuner_x, max(x) + tuner_x)
        ax.set_ylim(min(y) - tuner_y, max(y) + tuner_y)
        fig.suptitle('{0}_of_{1}_on_{2}'.format(title, self.algorithm_name, self.model_name))
        ax.autoscale_view(True, True, True)

        ax.set_xlabel(self.features[OBJECTIVE_NUMBER[0]])
        ax.set_ylabel(self.features[OBJECTIVE_NUMBER[1]])
        plt.savefig(self.directory + '{0}_of_{1}_on_{2}'.format(title, self.algorithm_name, self.model_name),
                    format='pdf')
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


def calculate_statistics(current_column):
    return WeightedMooResult.calculate_statistics(current_column)


def write_separate_statistics_to_separate_files(all_statistics_of_all_runs, cwd):
    STAT_TYPES = ["max", "min", "median"]

    for index, stat_type in enumerate(STAT_TYPES):
        with open("{0}{1}.txt".format(results_directory, stat_type), "wb") as f:
            for stats in all_statistics_of_all_runs:
                f.write('%s\n' % stats[index])


def write_statistics_to_file(directory, statistics, population_size):
    return WeightedMooResult.write_statistics_to_file(directory, statistics, population_size)


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

    # Parameters for every run
    algorithm_name = ''
    population_size = 0
    model_name = ''

    # This must be give to the script by hand: what is the base directory name of the results
    base_directory = 'hh_pas_surrogate'
    directories = get_directories(base_directory)

    #
    all_minimums_of_all_runs = []
    # NSGAII on HODGKIN-HUXLEY
    for instance_index, directory in enumerate(directories):
        sorted_result = WeightedMooResult(directory=directory)
        all_minimums_of_all_runs.append([row[INDEX_OF_MINIMUM] for row in sorted_result.statistics])
        multi_objective_result = NormalMooResult(directory=directory)

        if instance_index == 0:
            algorithm_name = multi_objective_result.algorithm_name
            population_size = multi_objective_result.population_size
            model_name = multi_objective_result.model_name

    all_statistics_of_all_runs = fill_statistics_for_all_runs(all_minimums_of_all_runs)
    write_separate_statistics_to_separate_files(all_statistics_of_all_runs, results_directory)
    write_statistics_to_file((results_directory), all_statistics_of_all_runs, population_size)

    plotter = GeneralPlotter(algorithm_name, model_name, directory=(results_directory))
    plotter.create_min_plot_of_all_runs(all_minimums_of_all_runs)
    plotter.create_generation_plot(all_statistics_of_all_runs, title="Statistics of every run of ")

    print("--- %s seconds ---" % (time.time() - start_time))
