import os
import re
import operator
import itertools
import numpy as np
from operator import mul
from operator import itemgetter
import xml.etree.ElementTree as ET
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


cwd = os.getcwd()


def _float_or_int(val):
            try:
                a=int(val)
                return a
            except ValueError:
                try:
                    return float(val)
                except ValueError:
                    return unicode(val.strip("u").strip('\''))


def parseSettings(xml_file="_settings.xml"):

    xml = ET.parse(xml_file)
    root = xml.getroot()

    for child in root:
        if child.tag == "evo_strat":
            evo_strat = child.text
        if child.tag == "boundaries":
            boundaries =  map(lambda x:map(_float_or_int,x.strip().split(", ")), child.text.strip()[2:len(child.text.strip())-2].split("], ["))
        if child.tag == "max_evaluation":
            max_eval = 1
	if child.tag == "resolution":
	    resolution = map(_float_or_int,child.text.strip().lstrip("[").rstrip("]").split(","))
	    pop_size = reduce(mul, resolution, 1)
        if child.tag == "num_params":
            num_param = int(child.text)
        if child.tag=="feats":
            feats =child.text.strip().split(", ")
    return boundaries, max_eval, resolution, pop_size, num_param, evo_strat, feats


def parseIndividuals(ind_file='ind_file.txt'):
    REMOVE_THESE_CHARS = ['[', ' [', ']', ',']
    NEEDED_COLS = [1] + range(3, num_param+3)

    individuals = []
    with open(ind_file, 'rb') as f:

	for individual in iter(f):
	    individuals.append([float(remove_unwanted_characters(value, REMOVE_THESE_CHARS)) for i, value in enumerate(individual.split()) if i in NEEDED_COLS])

    individuals	= np.array(individuals)

    OFFSET = 1
    for i in range(OFFSET,len(individuals[0])):
        for j in range(len(individuals)):
            individuals[j][i] = renormalize(individuals[j][i], i)

    return individuals


def remove_unwanted_characters(element, removeTheseChars):
    for char in removeTheseChars:
        if char in element:
            element = element.replace(char, '')
    return element


def renormalize(individual, i):
    UPPER_BOUND_INDEX = 1
    LOWER_BOUND_INDEX = 0
    OFFSET = 1
    normalizedIndividual = individual*(boundaries[UPPER_BOUND_INDEX][i-OFFSET]-boundaries[LOWER_BOUND_INDEX][i-OFFSET])+boundaries[LOWER_BOUND_INDEX][i-OFFSET]
    return normalizedIndividual


def parseFitnesComponents(fitnesCompsFile="fitness_components.txt"):
	REMOVE_THESE_CHARS = ['[', ']', ', ', ',']
	NEEDED_COLS = range(len(feats) + 1)

	print(NEEDED_COLS)
	fitnesComps = []
	with open(fitnesCompsFile, 'rb') as f:
		for fitnesComponent in iter(f):
			fitnesComps.append([float(remove_unwanted_characters(value, REMOVE_THESE_CHARS)) for i, value in enumerate(fitnesComponent.split()) if i in NEEDED_COLS])

	fitnesComps	= np.array(fitnesComps)
	fitnesComps = sorted(fitnesComps,key=itemgetter(len(feats))) #sort by fitnes

	return fitnesComps


def mergeIndividualsAndFitnes(individuals, fitComps):

	individualsAndFitnes =[]
	for i in range(len(individuals)):
		temp=[]
		temp.extend(individuals[i])
		temp.extend(fitComps[i])
		individualsAndFitnes.append(temp)

	return individualsAndFitnes


def createPlanes(lim, index):

	fitnesIndex = index + num_param + 1 #pop_index + parameters are in front of the fitnesses

	points = np.ndarray(shape = (len(everyIndivs), num_param+1))

	for i in range(len(everyIndivs)):
	   for j in range(num_param):
		    points[i][j] = everyIndivs[i][j+1]     #parameters start from 1
	   points[i][-1] = everyIndivs[i][fitnesIndex] #fitnes goes to last place in points

	for i in range(0,num_param):
		plane = np.ndarray(shape = (len(everyIndivs), num_param))
		plane = [row[:i].tolist() + row[i+1:].tolist() for row in points if str(row[i]) == paramVals[i]]

        combinations = list(itertools.combinations(range(0,num_param-1),2)) #for getting the combinations of the possible parameters for the 3 dimensional projection

        neededLabels = labels[:i] + labels[i+1:]
        neededParamVals = paramVals[:i] + paramVals[i+1:]
        for j in range(len(combinations)):
            neededCols = [column(plane, combinations[j][0]), column(plane, combinations[j][1]), column(plane, num_param-1)]
            createScatterPlot(neededCols, neededLabels ,lim[i], neededParamVals, feats[index])


def column(matrix, i):
    return [row[i] for row in matrix]


def createScatterPlot(coordinates, labels, lim, exactParameterValues, suptitle):
    X_COORD = 0
    Y_COORD = 1
    Z_COORD = 2

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim3d(lim)
    st = fig.suptitle(suptitle)
    for i in range(len(coordinates[X_COORD])):
	if([str(coord) for j,  coord in enumerate(column(coordinates,i)) if checkIfParameterCoordinate(j, len(coordinates)-1)] == exactParameterValues):
            ax.scatter(coordinates[X_COORD][i], coordinates[Y_COORD][i], coordinates[Z_COORD][i], c = "r")
            del coordinates[X_COORD][i], coordinates[Y_COORD][i], coordinates[Z_COORD][i]
            break

    ax.scatter(coordinates[X_COORD], coordinates[Y_COORD], coordinates[Z_COORD])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel('fitnes')
    plt.show()


def checkIfParameterCoordinate(coordinateIndex, fitnesIndex):
    return coordinateIndex != fitnesIndex

if __name__ == '__main__':

    paramVals = ["0.12", "0.036", "0.0003"]
    labels = ['gnabar', 'gkbar', 'gl']
    boundaries, max_eval, resolution, pop_size, num_param, evo_strat, feats = parseSettings()
    individuals = parseIndividuals()
    fitnesComps = parseFitnesComponents()
    lims = [[[0,0.35] ,[0,0.2] ,[0,0.35]],[[0,0.05],[0,0.02],[0,0.05]],[[0,2],[0,2],[0,2]],[[0,2],[0,2],[0,2]],[[0,2],[0,2],[0,2]],[[0,0.1],[0,0.1],[0,0.12]],[[0,2],[0,2],[0,2]],[[0,0.01],[0,0.01],[0,0.01]],[[0,1.8],[0,1.8],[0,1.8]],[[0,1.1],[0,1.1],[0,1.1]]]

    everyIndivs = mergeIndividualsAndFitnes(individuals,  fitnesComps)
    feats.append("weighted_sum")

    createPlanes(lims[-1], len(feats)-1)
    for i in range(len(feats)):
        createPlanes(lims[i], i)

