from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *
import itertools
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import os
import xml.etree.ElementTree as ET
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

cwd = os.getcwd()


def _float_or_int(val):
    try:
        a = int(val)
        return a
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return unicode(val.strip("u").strip('\''))


def parse_settings(xml_file):
    xml = ET.parse(xml_file)
    root = xml.getroot()

    for child in root:
        if child.tag == "evo_strat":
            evo_strat = child.text
        if child.tag == "boundaries":
            boundaries = map(lambda x: map(_float_or_int, x.strip().split(", ")),
                             child.text.strip()[2:len(child.text.strip()) - 2].split("], ["))
        if child.tag == "max_evaluation":
            max_eval = int(float(child.text))
        if child.tag == "pop_size":
            pop_size = int(float(child.text))
        if child.tag == "num_params":
            num_param = int(child.text)
    return boundaries, max_eval, pop_size, num_param, evo_strat


def parse_individuals(ind_file):
    START_INDEX_OF_PARAMETERS = 3
    generations = []
    with open(ind_file) as f:
        current_generation = []
        for individual in iter(f):
            individual = split_values_of_individuals(individual)[START_INDEX_OF_PARAMETERS:START_INDEX_OF_PARAMETERS+num_param]
            renormalized_individual = renormalize(individual)
            current_generation.append(renormalized_individual)
            if  is_end_of_generation(len(current_generation)):
                generations.append(current_generation)
                current_generation = []
    return generations

def remove_unwanted_characters(element):
    remove_these_chars = ['(', ')', '[', ']', ',']
    for char in remove_these_chars:
        if char in element:
            element = element.replace(char, '')
    return element

def split_values_of_individuals(new_individual):
    return [float(remove_unwanted_characters(value)) for value in new_individual.split()]

def is_end_of_generation(length_of_generation):
    return length_of_generation ==  population_size

def renormalize(parameters):
    MAX_INDEX = 1
    MIN_INDEX = 0
    for i, parameter in enumerate(parameters):
        parameters[i] = parameter * (boundaries[MAX_INDEX][i] - boundaries[MIN_INDEX][i]) + boundaries[MIN_INDEX][i]
    return parameters


def update(gen):

    plt.clf()
    st = fig.suptitle("{0} {1} {2}".format(evo_strat, model_name, str(gen)))
    points = inds_gen[gen]
    num_cols = len(boundaries[0])-1
    combinations = list(itertools.combinations(range(0, num_param), 2))

    for i, v in enumerate(xrange(num_plots)):
        if proj[i] == '3d':

            exec ("ax{0} = fig.add_subplot(2,num_cols,v+1,projection = proj[i])".format(v))
            for j in range(len(points)):
                exec ("ax{0}.scatter(points[j][0], points[j][1], points[j][2])".format(v))
        else:
            exec ("ax{0} = fig.add_subplot(2,num_cols,v+1)".format(v))
            for j in range(1,len(points)):
                exec ("ax{0}.scatter(points[j][combinations[i-1][0]], points[j][combinations[i-1][1]], c=colors[i])".format(v))
                if j == len(points)-1:
                    exec ("ax{0}.scatter(points[0][combinations[i-1][0]], points[0][combinations[i-1][1]], c=colors[-1])".format(v))
    set_legend()

def init():
    gen = 0
    st.set_text("{0} {1} {2}".format(evo_strat, model_name, str(gen)))
    points = inds_gen[gen]
    num_cols = len(boundaries[0])-1
    combinations = list(itertools.combinations(range(0, num_param), 2))

    for i, v in enumerate(xrange(num_plots)):
        if proj[i] == '3d':
            exec ("ax{0} = fig.add_subplot(2,num_cols,v+1,projection = proj[i])".format(v))
            for j in range(len(points)):
                exec ("ax{0}.scatter(points[j][0], points[j][1], points[j][2])".format(v))
        else:
            exec ("ax{0} = fig.add_subplot(2,num_cols,v+1)".format(v))
            for j in range(len(points)):
                exec ("ax{0}.scatter(points[j][combinations[i-1][0]], points[j][combinations[i-1][1]], c=colors[i])".format(v))


def set_legend():
    combinations = list(itertools.combinations(range(0, num_param), 2))

    for i, v in enumerate(xrange(num_plots)):
        exec ("ax{0} = fig.get_axes()[{0}]".format(v))
        if proj[i] == '3d':
            exec("threeD_axes(ax{0})".format(v))
        else:
            exec("twoD_axes(ax{0}, combinations[{0}-1])".format(v))


def threeD_axes(ax):

    ax.scatter(exact_point[0], exact_point[1], exact_point[2], c='r')
    ax.set_xlim3d([boundaries[0][0], boundaries[1][0]])
    ax.set_xlabel(labels[0])

    ax.set_ylim3d([boundaries[0][1], boundaries[1][1]])
    ax.set_ylabel(labels[1])

    ax.set_zlim3d([boundaries[0][2], boundaries[1][2]])
    ax.set_zlabel(labels[2])

def twoD_axes(ax,combination):

    ax.scatter(exact_point[combination[0]], exact_point[combination[1]], c='r')
    ax.set_xlim([boundaries[0][combination[0]], boundaries[1][combination[0]]])
    ax.set_xlabel(labels[combination[0]])

    ax.set_ylim([boundaries[0][combination[1]], boundaries[1][combination[1]]])
    ax.set_ylabel(labels[combination[1]])

if __name__ == '__main__':
    boundaries, max_eval, population_size, num_param, evo_strat = parse_settings("_settings.xml")
    inds_gen = parse_individuals('ind_file.txt')

    exact_point = [0.01,2, 0.3, 3]
    labels = ['weight','delay', 'tau_rise', 'tau_decay']
    model_name = "CLAMP"

    fig = plt.figure(figsize=(12, 8))
    st = fig.suptitle("")
    num_plots = 2*(len(boundaries[0])-1)

    proj = [None, None, None, None, None, None, None, None]
    colors = ['C0', 'C1', 'C2', 'C5', 'C6', 'C8', 'C9', 'C4', 'C7', 'C3']

    anim = animation.FuncAnimation(fig, update, frames=len(inds_gen), init_func=init(), interval=300, repeat=False)

    #plt.show()
    anim.save('PSO_CLAMP.html')
