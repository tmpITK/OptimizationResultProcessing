from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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


def parseSettings(xml_file):
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


def parseIndividuals(ind_file):
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
#def save_generation(new_generation):
    #generations.append(new_generation)



def update(gen):
    st.set_text(evo_strat + " on HH " + str(gen))
    updateAxes()

    points = inds_gen[gen]

    for i in range(len(points)):
        ax1.scatter(points[i][0], points[i][1], points[i][2])
        ax2.scatter(points[i][0], points[i][1], c="k")
        ax3.scatter(points[i][1], points[i][2], c="y")
        ax4.scatter(points[i][0], points[i][2], c="g")

def init():
    st.set_text(evo_strat + " on HH " + str(0))

    points = inds_gen[0]

    for i in range(len(points)):
        ax1.scatter(points[i][0], points[i][1], points[i][2])
        ax2.scatter(points[i][0], points[i][1], c="k")
        ax3.scatter(points[i][1], points[i][2], c="y", animated=True)
        ax4.scatter(points[i][0], points[i][2], c="g", animated=True)


def updateAxes():
    ax1.cla()
    ax1.scatter(0.12, 0.036, 0.0003, c='r')
    ax1.set_xlim3d([boundaries[0][0], boundaries[1][0]])
    ax1.set_xlabel('gnabar_hh')

    ax1.set_ylim3d([boundaries[0][1], boundaries[1][1]])
    ax1.set_ylabel('gkbar_hh')

    ax1.set_zlim3d([boundaries[0][2], boundaries[1][2]])
    ax1.set_zlabel('gl_hh')

    ax2.cla()
    ax2.scatter(0.12, 0.036, c='r')
    ax2.set_xlim([boundaries[0][0] - 0.02, boundaries[1][0]])
    ax2.set_xlabel('gnabar_hh')

    ax2.set_ylim([boundaries[0][1], boundaries[1][1]])
    ax2.set_ylabel('gkbar_hh')

    ax3.cla()
    ax3.scatter(0.036, 0.0003, c='r')
    ax3.set_xlim([boundaries[0][1], boundaries[1][1]])
    ax3.set_xlabel('gkbar_hh')

    ax3.set_ylim([boundaries[0][2], boundaries[1][2]])
    ax3.set_ylabel('gl_hh')

    ax4.cla()
    ax4.scatter(0.12, 0.0003, c='r')
    ax4.set_xlim([boundaries[0][0], boundaries[1][0]])
    ax4.set_xlabel('gnabar_hh')

    ax4.set_ylim([boundaries[0][2], boundaries[1][2]])
    ax4.set_ylabel('gl_hh')





boundaries, max_eval, population_size, num_param, evo_strat = parseSettings("_settings.xml")
inds_gen = parseIndividuals('ind_file.txt')
fig = plt.figure(figsize=(12, 8))
st = fig.suptitle(evo_strat + " on HH")
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

anim = animation.FuncAnimation(fig, update, frames=len(inds_gen),  interval=300, repeat=False)

# plt.show()
anim.save('PSO_HH.html')
