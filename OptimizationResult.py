from __future__ import print_function
import os
import re
import abc
import time
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
NOTE: file formats used:
	-ind_file: 	generation, population size, (fitnes results), [resulting parameters]
	-final_archive_file: (fitnes results)
"""
class MetaOptimizationSettings:
	__metaclass__ = abc.ABCMeta
	@abc.abstractmethod
	def getOptimizationSettings(self):
		raise NotImplementedError

class OptimizationSettings(MetaOptimizationSettings):
	'''
	Class for getting and storing the needed settings of the optimization process.
	'''

	def __init__(self, xml_file="_settings.xml", directory=''):
		self.LAST_ELEMENT_INDEX = -1

		self.directory = directory
		self.xml_file = self.directory + xml_file

	def getOptimizationSettings(self):
		xml = ET.parse(self.xml_file)
		root = xml.getroot()

		for child in root:
			if child.tag == "evo_strat":
				self.algorithm_name = child.text
			if child.tag == "model_path":
				self.model_name = child.text.split('/')[self.LAST_ELEMENT_INDEX]
			if child.tag == "boundaries":
				self.boundaries =  map(lambda x:map(self._float_or_int,x.strip().split(", ")), child.text.strip()[2:len(child.text.strip())-2].split("], ["))
			if child.tag == "max_evaluation":
				self.number_of_generations = int(float(child.text))
			if child.tag =="pop_size":
				self.population_size = int(float(child.text))
			if child.tag == "num_params":
				self.number_of_parameters = int(child.text)
			if child.tag == "feats":
				self.features = child.text.split(", ")
			if child.tag == "weights":
				self.weights =  map(self._float_or_int,child.text.strip().lstrip("[").rstrip("]").split(","))
				self.number_of_objectives = len(self.weights)
		return [self.directory, self.algorithm_name, self.model_name, self.boundaries, self.number_of_generations, self.population_size, self.number_of_parameters,
					self.features, self.weights, self.number_of_objectives]

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

class RawOptimizationResult():
	'''
	Class for getting and storing the result of the optimization procces.
	'''
	def __init__(self, opSettings, ind_file="ind_file.txt"):
		self.directory, self.algorithm_name, self.model_name, self.boundaries, self.number_of_generations, self.population_size, \
									self.number_of_parameters, self.features, self.weights, self.number_of_objectives = opSettings.getOptimizationSettings()
		self.ind_file = self.directory + ind_file
		self.generations = []

		self.parseIndividualFile()

	def parseIndividualFile(self):
		with open(self.ind_file, 'rb') as f:
			current_generation = []
			for line in iter(f):
				current_generation.append([float(self.removeUnwantedChars(element)) for element in line.split()])
				if len(current_generation) == self.population_size:
					self.saveGeneration(current_generation)
					current_generation = []

	@staticmethod
	def removeUnwantedChars(element):
		remove_these_chars = ['(', ')', '[', ']', ',']
		for char in remove_these_chars:
			if char in element:
				element = element.replace(char, '')
		return element

	def saveGeneration(self, new_generation):
		self.generations.append(new_generation)

	def printRawIndividualResults(self):
		self.printGivenIndividuals(self.generations)

	@staticmethod
	def printGivenIndividuals(generations):
		for  generation in generations:
			print(*generation, sep='\n')

class SortedMOOResult(RawOptimizationResult):

	"""
	Class for calculating the weighted sum of the objective functions' fitness values
	and sorting the generations by this sum to get the individual with the lowest
	fitness score.
	"""

	def __init__(self, opSettings, ind_file='ind_file.txt'):
		RawOptimizationResult.__init__(self, opSettings, ind_file)

		self.INDEX_OF_WEIGHTED_SUM = 2
		self.OFFSET = 2 	#ind_file format!
		self.NEW_OFFSET = 3 #we inserted the weighted sum!

		self.sorted_generations = []
		self.statistics = []

		self.insertWeightedSums()
		self.sortIndividualsByWeightedSum()
		self.writeSortedIndividuals()
		self.calculateStatistics()
		self.writeStatistics()
		self.plotStatistics()

	def insertWeightedSums(self):
		for generation in self.generations:
			index_of_original_generation = self.generations.index(generation)

			for individual in generation:
				index_of_original_individual = generation.index(individual)
				individual = individual[:self.OFFSET] + [self.calculateWeightedSum(individual[self.OFFSET:self.number_of_objectives+self.OFFSET])] +  individual[self.OFFSET:]
				generation[index_of_original_individual] = individual

			self.generations[index_of_original_generation] = generation

	def calculateWeightedSum(self, objectives):
		return sum([obj*weight for obj, weight in zip(objectives, self.weights)])

	def sortIndividualsByWeightedSum(self):
		for generation in self.generations:
			self.sorted_generations.append(sorted(generation,key=lambda x: x[self.INDEX_OF_WEIGHTED_SUM]))

	def prepareIndividualForWriting(self,individual):
		individual = [str(elem) for elem in individual]
		return individual[:3] + ['(' + ", ".join(individual[self.NEW_OFFSET:self.NEW_OFFSET+self.number_of_objectives]) +
	 									')'] + ['[' + ", ".join(individual[self.NEW_OFFSET+self.number_of_objectives:]) + ']']
	def writeStatistics(self):
		with open(self.directory + "sorted_stat_file.txt", "wb") as f:
			for index, generation in enumerate(self.statistics):
				f.write('%d, %d, %s\n' % (index, self.population_size, ",".join(map(str, generation))))

	def calculateStatistics(self):
		for generation in self.sorted_generations:
			maximum_of_generation = max([row[self.INDEX_OF_WEIGHTED_SUM] for row in generation])
			minimum_of_generation = min([row[self.INDEX_OF_WEIGHTED_SUM] for row in generation])
			median_of_generation = np.median([row[self.INDEX_OF_WEIGHTED_SUM] for row in generation])
			self.statistics.append([maximum_of_generation, minimum_of_generation, median_of_generation])

	def writeSortedIndividuals(self):
		with open(self.directory + "sorted_ind_file.txt", "wb") as f:
			for generation in self.sorted_generations:
				for individual in generation:
					f.write("%s\n" % ", ".join(self.prepareIndividualForWriting(individual)))

	def plotStatistics(self):
		fig = plt.figure()

		plt.plot([row[0] for row in self.statistics], 'r.-', label = "max", linewidth=1.5)
		plt.plot([row[1] for row in self.statistics], 'r--', label = "min", linewidth=1.5)
		plt.plot([row[2] for row in self.statistics], 'r', label = "median", linewidth=1.5)

		fig.suptitle('{0} on {1}'.format(self.algorithm_name, self.model_name))
		plt.xlabel('generations')
		plt.ylabel('score value')
		plt.yscale('log')

		plt.legend(fontsize=14, ncol=1)
		plt.savefig(self.directory + '{0} on {1}'.format(self.algorithm_name, self.model_name), format='pdf')

class TrueMOOResult(RawOptimizationResult):

	"""
	Class for getting and storing the archive of multi objective optimization results
	and plotting the Pareto Front.
	"""

	def __init__(self, opSettings, ind_file="ind_file.txt", final_archive_file="final_archive.txt"):
		RawOptimizationResult.__init__(self, opSettings, ind_file)

		self.OFFSET = 2
		self.LAST_ELEMENT_INDEX = -1
		self.OBJECTIVE_NUMBER = range(self.number_of_objectives)

		self.final_archive_file = self.directory + final_archive_file
		self.final_archive = []
		self.final_generation = []
		self.final_generation_objectives = []

		self.separateFinalGeneration()
		self.separateFinalGenerationObjectives()
		self.parseFinalArchiveFile()
		self.plotParetoFront(self.final_archive, "Pareto Front")

		if(self.algorithm_name == "PAES"):
			self.plotParetoFront(self.final_generation_objectives, "Final Generation")
			
	def separateFinalGeneration(self):
		self.final_generation = self.generations[self.LAST_ELEMENT_INDEX]

	def separateFinalGenerationObjectives(self):
		self.final_generation_objectives = [individual[self.OFFSET:self.OFFSET + self.number_of_objectives] for individual in self.final_generation]

	def parseFinalArchiveFile(self):
		with open(self.final_archive_file, 'rb') as arc_file:
			for line in iter(arc_file):
				self.saveArchivedIndividualObjectives([float(self.removeUnwantedChars(element)) for element in line.split()])

	def saveArchivedIndividualObjectives(self, archived_individual):
		self.final_archive.append(archived_individual)

	def plotParetoFront(self, best_individuals, title):

		x = []
		y = []
		for individual in best_individuals:
			x.append(individual[self.OBJECTIVE_NUMBER[0]])
			y.append(individual[self.OBJECTIVE_NUMBER[1]])
		if self.number_of_objectives > 2:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			z = [row[self.OBJECTIVE_NUMBER[2]] for row in best_individuals]
			ax.scatter(x, y, z, color='b')
			ax.set_ylim(min(z), max(z))
			ax.set_zlabel(self.features[self.OBJECTIVE_NUMBER[2]])
		else:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.scatter(x, y, color='b')
			ax.set_xlim(min(x), max(x))
			ax.set_ylim(min(y), max(y))
		fig.suptitle('{0} of {1} on {2}'.format(title,  self.algorithm_name, self.model_name))
		ax.autoscale_view(True,True,True)

		ax.set_xlabel(self.features[self.OBJECTIVE_NUMBER[0]])
		ax.set_ylabel(self.features[self.OBJECTIVE_NUMBER[1]])
		plt.savefig(self.directory + '{0} of {1} on {2}'.format(title, self.algorithm_name, self.model_name), format='pdf')

if __name__ == '__main__':
	CHILD_DIR_INDEX = 0
	LAST_ELEMENT_INDEX = -1
	start_time = time.time()

	cwd = os.getcwd()
	hh_regex = re.compile('hh_pas_surrogate_.')
	hh_directories = [x[CHILD_DIR_INDEX]+'/' for x in os.walk(cwd) if re.match(hh_regex, x[CHILD_DIR_INDEX].split('/')[LAST_ELEMENT_INDEX])]

	vclamp_regex = re.compile('VClamp_surrogate_.')
	vclamp_directories = [x[CHILD_DIR_INDEX]+'/' for x in os.walk(cwd) if re.match(vclamp_regex, x[CHILD_DIR_INDEX].split('/')[LAST_ELEMENT_INDEX])]

	for directory in hh_directories:
		nsga_hh_opSettings = OptimizationSettings(directory=directory)
		nsga_hh_sorted_result = SortedMOOResult(nsga_hh_opSettings)
		nsga_hh_multi_objective_result = TrueMOOResult(nsga_hh_opSettings)
		plt.show()

	for directory in vclamp_directories:
		paes_clamp_opSettings = OptimizationSettings(directory=directory)
		paes_clamp_sorted_result = SortedMOOResult(paes_clamp_opSettings)
		paes_clamp_multi_objective_result = TrueMOOResult(paes_clamp_opSettings)
		plt.show()

	nsga_hh_opSettings = OptimizationSettings("hh_settings.xml")
	nsga_hh_sorted_result = SortedMOOResult(nsga_hh_opSettings, "hh_ind_file.txt")
	nsga_hh_multi_objective_result = TrueMOOResult(nsga_hh_opSettings, "hh_ind_file.txt", "hh_final_archive.txt")
	plt.show()

	print("--- %s seconds ---" % (time.time() - start_time))
