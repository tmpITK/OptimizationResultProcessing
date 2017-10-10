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
				self.algorithm_name = child.text
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
		return [self.algorithm_name, self.boundaries, self.number_of_generations, self.population_size, self.number_of_parameters,
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
		self.algorithm_name, self.boundaries, self.number_of_generations, self.population_size, self.number_of_parameters, self.features, self.weights, self.number_of_objectives = opSettings.getOptimizationSettings()
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

	def printRawIndividualResults(self):
		self.printGivenIndividuals(self.generations)

	def printFinalGeneration(self):
		self.printGivenIndividuals([self.final_generation])

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

		fig.suptitle('{0} on {1}'.format(self.algorithm_name, "Voltage Clamp"))
		plt.xlabel('generations')
		plt.ylabel('score value')
		plt.yscale('log')

		plt.legend(fontsize=14, ncol=1)	#ncol= number of columns of the labels
		plt.savefig('{0} on {1}'.format(self.algorithm_name, "Voltage Clamp"), format='pdf')
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
		self.final_generation = []
		self.final_generation_objectives = []

		self.getFinalGeneration()
		self.getFinalGenerationObjectives()
		self.parseFinalArchiveFile()
		self.plotParetoFront(self.final_archive, "Pareto Front")
		if(self.algorithm_name == "PAES"):
			self.plotParetoFront(self.final_generation_objectives, "Final Generation")

	def parseFinalArchiveFile(self):
		with open(self.final_archive_file, 'rb') as arc_file:
			for line in iter(arc_file):
				self.final_archive.append([float(self.removeUnwantedChars(element)) for element in line.split()])

	def getFinalGeneration(self):
		self.final_generation = self.generations[-1]

	def getFinalGenerationObjectives(self):
		self.final_generation_objectives = [individual[2:5] for individual in self.final_generation]

	def plotParetoFront(self, best_individuals, title):

		x = []
		y = []
		for individual in best_individuals:
			x.append(individual[self.NUMBER_OF_OBJECTIVE[0]])
			y.append(individual[self.NUMBER_OF_OBJECTIVE[1]])
		if self.number_of_objectives > 2:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			z = [row[self.NUMBER_OF_OBJECTIVE[2]] for row in best_individuals]
			ax.scatter(x, y, z, color='b')
			ax.set_ylim(min(z), max(z))
			ax.set_zlabel(self.features[self.NUMBER_OF_OBJECTIVE[2]])
		else:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.scatter(x, y, color='b')
			ax.set_xlim(min(x), max(x))
			ax.set_ylim(min(y), max(y))
		fig.suptitle('{0} of {1} on {2}'.format(title,  self.algorithm_name, "Voltage Clamp"))
		ax.autoscale_view(True,True,True)

		ax.set_xlabel(self.features[self.NUMBER_OF_OBJECTIVE[0]])
		ax.set_ylabel(self.features[self.NUMBER_OF_OBJECTIVE[1]])
		plt.savefig('{0} of {1} on {2}'.format(title, self.algorithm_name, "Voltage Clamp"), format='pdf')
		plt.show()

if __name__ == '__main__':
	start_time = time.time()

	opSettings = OptimizationSettings("paes_settings.xml")
	sorted_result = SortedMOOResult(opSettings, "paes_ind_file.txt")
	multi_objective_result = TrueMOOResult(opSettings, "paes_ind_file.txt", "paes_final_archive.txt")

	print("--- %s seconds ---" % (time.time() - start_time))
