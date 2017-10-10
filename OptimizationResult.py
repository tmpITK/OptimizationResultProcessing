from __future__ import print_function
import os
import abc
import time
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



cwd = os.getcwd()
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

	def __init__(self, xml_file):
		self.xml_file = xml_file

	def getOptimizationSettings(self):
		xml = ET.parse(self.xml_file)
		root = xml.getroot()

		for child in root:
			if child.tag == "evo_strat":
				self.evolutionary_strategy = child.text
			if child.tag == "boundaries":
				self.boundaries =  map(lambda x:map(self._float_or_int,x.strip().split(", ")), child.text.strip()[2:len(child.text.strip())-2].split("], ["))
			if child.tag == "max_evaluation":
				self.maximum_number_of_evaluations = int(float(child.text))
			if child.tag =="pop_size":
				self.population_size = int(float(child.text))
			if child.tag == "num_params":
				self.number_of_parameters = int(child.text)
			if child.tag == "feats":
				self.features = child.text.split(", ")
			if child.tag == "weights":
				self.weights =  map(self._float_or_int,child.text.strip().lstrip("[").rstrip("]").split(","))
				self.number_of_objectives = len(self.weights)
		return [self.evolutionary_strategy, self.boundaries, self.maximum_number_of_evaluations, self.population_size, self.number_of_parameters,
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
	def __init__(self, opSettings, ind_file):
		self.evolutionary_strategy, self.boundaries, self.maximum_number_of_evaluations, self.population_size, self.number_of_parameters, self.features, self.weights, self.number_of_objectives = opSettings.getOptimizationSettings()
		self.ind_file = ind_file
		self.generations = []
		self.parseIndividualFile()

	@staticmethod
	def removeUnwantedChars(element):
		remove_these_chars = ['(', ')', '[', ']', ',']
		for char in remove_these_chars:
			if char in element:
				element = element.replace(char, '')
		return element

	def parseIndividualFile(self):
		with open(self.ind_file, 'rb') as f:
			current_generation = []
			for line in iter(f):
				current_generation.append([float(self.removeUnwantedChars(element)) for element in line.split()])
				if len(current_generation) == self.population_size:
					self.generations.append(current_generation)
					current_generation = []

	def getRawIndividualResults(self):
		self.printIndividualResults(self.generations)

	@staticmethod
	def printIndividualResults(generations):
		for index, generation in enumerate(generations):
			print(*generation, sep='\n')

class SortedMOOResult(RawOptimizationResult):

	"""
	Class for calculating the weighted sum of the objective functions' fitness values
	and sorting the generations by this sum to get the individual with the lowest
	fitness score.
	"""

	def __init__(self, opSettings, ind_file):
		RawOptimizationResult.__init__(self, opSettings, ind_file)

		self.INDEX_OF_WEIGHTED_SUM = 2

		self.sorted_generations = []
		self.statistics = []

		self.insertWeightedSums()
		self.sortIndividualsByWeightedSum()
		self.calculateStatistics()
		self.writeStatistics()
		self.plotStatistics()
		self.writeSortedIndividuals()

	def insertWeightedSums(self):
		for generation in self.generations:
			index_of_original_generation = self.generations.index(generation)

			for individual in generation:
				indexOfOriginalIndividual = generation.index(individual)
				individual = individual[:2] + [self.calculateWeightedSum(individual[2:self.number_of_objectives+2])] +  individual[2:]
				generation[indexOfOriginalIndividual] = individual

			self.generations[index_of_original_generation] = generation

	def calculateWeightedSum(self, objectives):
		return sum([obj*weight for obj, weight in zip(objectives, self.weights)])

	def sortIndividualsByWeightedSum(self):
		for generation in self.generations:
			self.sorted_generations.append(sorted(generation,key=lambda x: x[self.INDEX_OF_WEIGHTED_SUM]))

	def calculateStatistics(self):
		for generation in self.sorted_generations:
			maximum_of_generation = max([row[self.INDEX_OF_WEIGHTED_SUM] for row in generation])
			minimum_of_generation = min([row[self.INDEX_OF_WEIGHTED_SUM] for row in generation])
			median_of_generation = np.median([row[self.INDEX_OF_WEIGHTED_SUM] for row in generation])
			self.statistics.append([maximum_of_generation, minimum_of_generation, median_of_generation])

	def plotStatistics(self):
		fig = plt.figure()

		plt.plot([row[0] for row in self.statistics], 'r.-', label = "max", linewidth=1.5)
		plt.plot([row[1] for row in self.statistics], 'r--', label = "min", linewidth=1.5)
		plt.plot([row[2] for row in self.statistics], 'r', label = "median", linewidth=1.5)

		fig.suptitle('{0} on {1}'.format(self.evolutionary_strategy, "Voltage Clamp"))
		plt.xlabel('generations')
		plt.ylabel('score value')
		plt.yscale('log')

		plt.legend(fontsize=14, ncol=1)	#ncol= number of columns of the labels
		plt.savefig('{0} on {1}'.format(self.evolutionary_strategy, "Voltage Clamp"), format='pdf')
		plt.show()
		plt.close()

	def writeStatistics(self):
		with open("sorted_stat_file.txt", "wb") as f:
			for index, generation in enumerate(self.statistics):
				f.write('%d, %d, %s\n' % (index, self.population_size, ",".join(map(str, generation))))

	def prepareIndividualForWriting(self,individual):
		individual = [str(elem) for elem in individual]
		return individual[:3] + ['(' + ", ".join(individual[3:3+self.number_of_objectives]) + ')'] + ['[' + ", ".join(individual[3+self.number_of_objectives:]) + ']']

	def writeSortedIndividuals(self):
		with open("sorted_ind_file.txt", "wb") as f:
			for generation in self.sorted_generations:
				for individual in generation:
					f.write("%s\n" % ", ".join(self.prepareIndividualForWriting(individual)))

class TrueMOOResult(RawOptimizationResult):

	"""
	Class for getting and storing the archive of multi objective optimization results
	and plotting the Pareto Front.
	"""

	def __init__(self, opSettings, ind_file, final_archive_file):
		RawOptimizationResult.__init__(self, opSettings, ind_file)

		self.NUMBER_OF_OBJECTIVE = range(self.number_of_objectives)

		self.final_archive_file = final_archive_file
		self.final_archive = []

		self.parseFinalArchiveFile()
		self.plotParetoFront()

	def parseFinalArchiveFile(self):
		with open(self.final_archive_file, 'rb') as arc_file:
			for line in iter(arc_file):
				self.final_archive.append([float(self.removeUnwantedChars(element)) for element in line.split()])

	def plotParetoFront(self):

		x = []
		y = []
		for individual in self.final_archive:
			x.append(individual[self.NUMBER_OF_OBJECTIVE[0]])
			y.append(individual[self.NUMBER_OF_OBJECTIVE[1]])
		if self.number_of_objectives > 2:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			z = [row[self.NUMBER_OF_OBJECTIVE[2]] for row in self.final_archive]
			ax.scatter(x, y, z, color='b')
			ax.set_zlabel(self.features[self.NUMBER_OF_OBJECTIVE[2]])
		else:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.scatter(x, y, color='b')
		fig.suptitle('Pareto Front of {0} by {1}'.format("Voltage Clamp", self.evolutionary_strategy))
		ax.autoscale_view(True,True,True)

		ax.set_xlabel(self.features[self.NUMBER_OF_OBJECTIVE[0]])
		ax.set_ylabel(self.features[self.NUMBER_OF_OBJECTIVE[1]])
		plt.savefig('Pareto Front of {0} by {1}'.format("Voltage Clamp", self.evolutionary_strategy), format='pdf')
		plt.show()

if __name__ == '__main__':
	start_time = time.time()

	opSettings = OptimizationSettings("hh_settings.xml")
	sorted_by_weighted_sum_multi_objective_result = SortedMOOResult(opSettings, "hh_ind_file.txt")
	true_multi_objective_result = TrueMOOResult(opSettings, "hh_ind_file.txt", "hh_final_archive.txt")

	print("--- %s seconds ---" % (time.time() - start_time))
