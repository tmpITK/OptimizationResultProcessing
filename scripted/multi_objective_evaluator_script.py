import re
import os
import time
import PlotOptimizationResult as POR
import OptimizationResultProcessor as ORP

def get_directories(directory_base_name):
	CHILD_DIR_INDEX = 0
	LAST_ELEMENT_INDEX = -1

	regex = re.compile(directory_base_name+'_.')
	return [x[CHILD_DIR_INDEX]+'/' for x in os.walk(cwd) if re.match(regex, x[CHILD_DIR_INDEX].split('/')[LAST_ELEMENT_INDEX])]


def fill_statistics_for_all_runs(all_minimums_of_all_runs):
	all_statistics_of_all_runs = []
	for i in range(len(all_minimums_of_all_runs[0])):
			current_column = [column[i] for column in all_minimums_of_all_runs]
			all_statistics_of_all_runs.append(ORP.WeightedMooResult.calculate_statistics(current_column))
	return all_statistics_of_all_runs

if __name__ == '__main__':
	start_time = time.time()
	INDEX_OF_MINIMUM = 1
	cwd = os.getcwd()

	#Parameters for every run
	algorithm_name = ''
	population_size = 0
	model_name = ''

	#This must be give to the script by hand: what is the base directory name of the results
	base_directory = 'hh_pas_surrogate'
	directories = get_directories(base_directory)

	#
	all_minimums_of_all_runs = []
	#NSGAII on HODGKIN-HUXLEY
	for instance_index, directory in enumerate(directories):
		op_settings = ORP.OptimizationSettings(directory=directory)
		sorted_result = ORP.WeightedMooResult(op_settings)
		all_minimums_of_all_runs.append([row[INDEX_OF_MINIMUM] for row in sorted_result.statistics])
		multi_objective_result = ORP.NormalMooResult(op_settings)

		if instance_index == 0:
			algorithm_name = multi_objective_result.algorithm_name
			population_size = multi_objective_result.population_size
			model_name = multi_objective_result.model_name

	all_statistics_of_all_runs = fill_statistics_for_all_runs(all_minimums_of_all_runs)
	plotter = POR.GeneralPlotter(algorithm_name, model_name, directory=(cwd+'/'))
	plotter.create_generation_plot(all_statistics_of_all_runs, title="Statistics of every run of ")
	ORP.WeightedMooResult.write_statistics((cwd+'/'), all_statistics_of_all_runs, population_size)

	print("--- %s seconds ---" % (time.time() - start_time))
