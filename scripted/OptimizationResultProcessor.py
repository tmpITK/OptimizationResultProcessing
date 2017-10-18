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
				current_generation.append(self.split_values_of_individuals(individual))
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
				individual.insert(self.OFFSET, weighted_sum)
				generation[index_of_original_individual] = individual

			self.generations[index_of_original_generation] = generation

	def get_objectives_of_individual(self, individual):
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
	def write_statistics_to_file(directory, statistics, population_size):
		with open(directory + "sorted_stat_file.txt", "wb") as f:
			for index, generation in enumerate(statistics):
				f.write('%d, %d, %s\n' % (index, population_size, ", ".join(map(str, generation))))

	def write_sorted_individuals_to_file(self):
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

		self.final_archive_file = self.directory + final_archive_file
		self.final_archive = []
		self.final_generation = []
		self.final_generation_objectives = []

		self.separate_final_generation()
		self.separate_final_generation_objectives()
		self.parse_final_archive_file()
		if self.number_of_objectives in range(1,4):
			self.plot_pareto_front()

	def separate_final_generation(self):
		self.final_generation = self.generations[self.LAST_ELEMENT_INDEX]

	def separate_final_generation_objectives(self):
		self.final_generation_objectives = [individual[self.OFFSET:self.OFFSET + self.number_of_objectives] for individual in self.final_generation]

	def parse_final_archive_file(self):
		with open(self.final_archive_file, 'rb') as arc_file:
			for individual in iter(arc_file):
				self.save_archived_individual_objectives(self.split_values_of_individuals(individual))

	def save_archived_individual_objectives(self, archived_individual):
		self.final_archive.append(archived_individual)

	def plot_pareto_front(self):
		plotter = PlotOptimizationResult.GeneralPlotter(self.algorithm_name, self.model_name, self.directory, self.features)
		plotter.create_pareto_plot(self.final_archive)

		if self.algorithm_name == "PAES":
			plotter.create_pareto_plot(self.final_generation_objectives, "Final Generation")
