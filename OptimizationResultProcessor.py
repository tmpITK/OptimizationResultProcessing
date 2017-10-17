from __future__ import print_function
import os
import re
import abc
import time
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import PlotOptimizationResult

"""
NOTE: file formats used:
	-ind_file: 	generation, population size, (fitnes results), [resulting parameters]
	-final_archive_file: (fitnes results)
"""


class FileDoesNotExist(Exception):
	def __init__(self, missing_file, directory):
		message = "%s is not in %s" % (missing_file, directory)
		super(FileDoesNotExist, self).__init__(message)


class IncorrectNumberOfObjectives(Exception):
	def __init__(self):
		message = "Number of objectives must be either 2 or 3"
		super(IncorrectNumberOfObjectives, self).__init__(message)


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

	def get_optimization_settings(self):
		root = self.xml.getroot()

		for child in root:
			if child.tag == "evo_strat":
				algorithm_name = child.text
			if child.tag == "model_path":
				model_name = child.text.split('/')[self.LAST_ELEMENT_INDEX]
			if child.tag == "max_evaluation":
				number_of_generations = int(float(child.text))
			if child.tag =="pop_size":
				population_size = int(float(child.text))
			if child.tag == "num_params":
				number_of_parameters = int(child.text)
			if child.tag == "feats":
				features = child.text.split(", ")
			if child.tag == "weights":
				weights =  map(self._float_or_int,child.text.strip().lstrip("[").rstrip("]").split(","))

		return [algorithm_name, model_name, number_of_generations, population_size, number_of_parameters,
					features, weights]

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

class RawOptimizationResult(object):
	'''
	Class for getting and storing the result of the optimization procces.
	'''
	def __init__(self, op_settings, ind_file="ind_file.txt"):
		self.directory = op_settings.directory
		self.algorithm_name, self.model_name, self.number_of_generations, self.population_size, \
									self.number_of_parameters, self.features, self.weights = op_settings.get_optimization_settings()
		self.number_of_objectives = len(self.features)
		self.ind_file = self.directory + ind_file
		self.generations = []

		self.parse_individual_file()

	def parse_individual_file(self):
		with open(self.ind_file, 'rb') as f:
			current_generation = []
			for individual in iter(f):
				current_generation.append(self.split_individual_values(individual))
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

	def split_individual_values(self, new_individual):
		return [float(self.remove_unwanted_characters(value)) for value in new_individual.split()]

	def is_end_of_generation(self, length_of_generation):
	    return length_of_generation == self.population_size

	def save_generation(self, new_generation):
		self.generations.append(new_generation)

	def print_untouched_generations(self):
		self.print_given_generations(self.generations)

	@staticmethod
	def print_given_generations(generations):
		for  generation in generations:
			print(*generation, sep='\n')


class WeightedMooResult(RawOptimizationResult):

	"""
	Class for calculating the weighted sum of the objective functions' fitness values
	and sorting the generations by this sum to get the individual with the lowest
	fitness score.
	"""

	def __init__(self, op_settings, ind_file='ind_file.txt'):
		RawOptimizationResult.__init__(self, op_settings, ind_file)

		self.INDEX_OF_WEIGHTED_SUM = 2
		self.OFFSET = 2 	#ind_file format! -> generation and individual index
		self.NEW_OFFSET = self.OFFSET+1 #we insert the weighted sum!

		self.sorted_generations = []
		self.statistics = []

		self.insert_weighted_sums()
		self.sort_individuals_by_weighted_sum()
		self.write_sorted_individuals()
		self.fill_statistics_list()
		self.write_statistics(self.directory, self.statistics, self.population_size)
		self.plot_statistics()

	def insert_weighted_sums(self):
		for generation in self.generations:
			index_of_original_generation = self.generations.index(generation)

			for individual in generation:
				index_of_original_individual = generation.index(individual)
				weighted_sum = self.calculate_weighted_sum(self.get_individual_objectives(individual))
				individual.insert(self.OFFSET, weighted_sum)
				generation[index_of_original_individual] = individual

			self.generations[index_of_original_generation] = generation

	def get_individual_objectives(self, individual):
		objectives = individual[self.OFFSET:self.number_of_objectives+self.OFFSET]
		return objectives

	def calculate_weighted_sum(self, objectives):
		return sum([obj*weight for obj, weight in zip(objectives, self.weights)])

	def sort_individuals_by_weighted_sum(self):
		for generation in self.generations:
			self.sorted_generations.append(sorted(generation,key=lambda x: x[self.INDEX_OF_WEIGHTED_SUM]))

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
	def write_statistics(directory, statistics, population_size):
		with open(directory + "sorted_stat_file.txt", "wb") as f:
			for index, generation in enumerate(statistics):
				f.write('%d, %d, %s\n' % (index, population_size, ", ".join(map(str, generation))))

	def write_sorted_individuals(self):
		with open(self.directory + "sorted_ind_file.txt", "wb") as f:
			for generation in self.sorted_generations:
				for individual in generation:
					f.write("%s\n" % self.format_individual_for_writing(individual))

	def format_individual_for_writing(self,individual):
		individual = [str(value) for value in individual]

		indexes = individual[:self.OFFSET]
		weighted_sum = individual[self.OFFSET:self.NEW_OFFSET]
		objectives = individual[self.NEW_OFFSET:self.NEW_OFFSET+self.number_of_objectives]
		parameters = individual[self.NEW_OFFSET+self.number_of_objectives:]

		return  "{0}, {1}, ({2}), [{3}]".format(", ".join(indexes), ", ".join(weighted_sum), ", ".join(objectives), ", ".join(parameters))

	def plot_statistics(self):
		plotter = PlotOptimizationResult.GeneralPlotter(self.algorithm_name, self.model_name, self.directory)
		plotter.create_generation_plot(self.statistics)


class NormalMooResult(RawOptimizationResult):

	"""
	Class for getting and storing the archive of multi objective optimization results
	and plotting the Pareto Front.
	"""

	def __init__(self, op_settings, ind_file="ind_file.txt", final_archive_file="final_archive.txt"):
		RawOptimizationResult.__init__(self, op_settings, ind_file)

		self.OFFSET = 2
		self.LAST_ELEMENT_INDEX = -1

		if self.number_of_objectives not in range(1,4):
			raise IncorrectNumberOfObjectives()

		self.final_archive_file = self.directory + final_archive_file
		self.final_archive = []
		self.final_generation = []
		self.final_generation_objectives = []

		self.separate_final_generation()
		self.separate_final_generation_objectives()
		self.parse_final_archive_file()
		self.plot_pareto_front()

	def separate_final_generation(self):
		self.final_generation = self.generations[self.LAST_ELEMENT_INDEX]

	def separate_final_generation_objectives(self):
		self.final_generation_objectives = [individual[self.OFFSET:self.OFFSET + self.number_of_objectives] for individual in self.final_generation]

	def parse_final_archive_file(self):
		with open(self.final_archive_file, 'rb') as arc_file:
			for individual in iter(arc_file):
				self.save_archived_individual_objectives(self.split_individual_values(individual))

	def save_archived_individual_objectives(self, archived_individual):
		self.final_archive.append(archived_individual)

	def plot_pareto_front(self):
		plotter = PlotOptimizationResult.GeneralPlotter(self.algorithm_name, self.model_name, self.directory, self.features)
		plotter.create_pareto_plot(self.final_archive)

		if self.algorithm_name == "PAES":
			plotter.create_pareto_plot(self.final_generation_objectives, "Final Generation")


def get_directories(directory_base_name):
	CHILD_DIR_INDEX = 0
	LAST_ELEMENT_INDEX = -1

	regex = re.compile(directory_base_name+'_.')
	return [x[CHILD_DIR_INDEX]+'/' for x in os.walk(cwd) if re.match(regex, x[CHILD_DIR_INDEX].split('/')[LAST_ELEMENT_INDEX])]


def fill_statistics_for_all_runs(all_minimums_of_all_runs):
	all_statistics_of_all_runs = []
	for i in range(len(all_minimums_of_all_runs[0])):
			current_column = [column[i] for column in all_minimums_of_all_runs]
			all_statistics_of_all_runs.append(calculate_statistics(current_column))
	return all_statistics_of_all_runs


def calculate_statistics(current_column):
	return WeightedMooResult.calculate_statistics(current_column)


def write_separate_statistics(all_statistics_of_all_runs, cwd):
	STAT_TYPES = ["max", "min", "median"]
	results_directory = create_directory_for_statistics(cwd)

	for index, stat_type in enumerate(STAT_TYPES):
		with open("{0}{1}.txt".format(results_directory, stat_type), "wb") as f:
			for stats in all_statistics_of_all_runs:
				f.write('%s\n' % stats[index])


def write_statistics(directory, statistics, population_size):
	return WeightedMooResult.write_statistics(directory, statistics, population_size)


def create_directory_for_statistics(cwd):
	results_dir = cwd + '/results'
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	return results_dir + '/'


if __name__ == '__main__':
	start_time = time.time()
	INDEX_OF_MINIMUM = 1
	cwd = os.getcwd()

	#Parameters for every run
	algorithm_name = ''
	population_size = 0
	model_name = ''

	#This must be give to the script by hand: what is the base directory name of the results
	hh_base_directory = 'hh_pas_surrogate'
	hh_directories = get_directories(hh_base_directory)

	vclamp_base_directory = 'VClamp_surrogate'
	vclamp_directories = get_directories(vclamp_base_directory)

	#
	all_minimums_of_all_runs = []
	#NSGAII on HODGKIN-HUXLEY
	for instance_index, directory in enumerate(hh_directories):
		op_settings = OptimizationSettings(directory=directory)
		sorted_result = WeightedMooResult(op_settings)
		all_minimums_of_all_runs.append([row[INDEX_OF_MINIMUM] for row in sorted_result.statistics])
		multi_objective_result = NormalMooResult(op_settings)

		if instance_index == 0:
			algorithm_name = multi_objective_result.algorithm_name
			population_size = multi_objective_result.population_size
			model_name = multi_objective_result.model_name

	all_statistics_of_all_runs = fill_statistics_for_all_runs(all_minimums_of_all_runs)
	write_separate_statistics(all_statistics_of_all_runs, cwd)
	write_statistics((cwd+'/'), all_statistics_of_all_runs, population_size)

	plotter = PlotOptimizationResult.GeneralPlotter(algorithm_name, model_name, directory=(cwd+'/'))
	plotter.create_generation_plot(all_statistics_of_all_runs, title="Statistics of every run of ")


	for directory in vclamp_directories:
		paes_clamp_op_settings = OptimizationSettings(directory=directory)
		paes_clamp_sorted_result = WeightedMooResult(paes_clamp_op_settings)
		paes_clamp_multi_objective_result = NormalMooResult(paes_clamp_op_settings)

	'''
	nsga_hh_op_settings = OptimizationSettings("hh_settings.xml")
	nsga_hh_sorted_result = WeightedMooResult(nsga_hh_op_settings, "hh_ind_file.txt")
	nsga_hh_multi_objective_result = NormalMooResult(nsga_hh_op_settings, "hh_ind_file.txt", "hh_final_archive.txt")
	plt.show()
	'''

	print("--- %s seconds ---" % (time.time() - start_time))
