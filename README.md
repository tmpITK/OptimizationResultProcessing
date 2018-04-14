# OptimizationResultProcessing

Needed a script to process results of optimization. Primarily for multi objective algorithms.

Developed with inspyred algos.

Made for personal use and learning mainly. It probably has many questionable parts, because I'm
exploring Python's capabilities.

How to use (scripted):

Put the three files in the main directory where the result directories are.

The only user input needed is the base directory in which the results are found.

The directories must contain:

				-an xml file named "_settings.xml" with the appropriate datas.
				-an "ind_file.txt" with inspyred type individual file format.
				-a "final_archive.txt" with the objectives of the final individuals / row

The program will create the following files in the separate directories of the Optimization results:

	-a "sorted_stat_file.txt" with the statistics based on the order of the
		 individuals in a generation by the weighted sum of the objectives.
	-a "sorted_ind_file.txt"
	-a Pareto Front plot
	-a generational plot of the minimums, maximums and medians of the generations

In the main directory:

	-a results directory:	-"max.txt", "min.txt", "median.txt" with the max, min, median of all the runs alltogether
	-a generational plot based on these aggregated statistics

Island was created for the processing of Island model type results. Not in its final form currently.

fitnes_grid_plot plots the energy surface of the Hodgkin-Huxley model component by component (additional files will be uploaded)

fitnes_grid_plot_summ plots the weighted sum of the chosen components. 

The latter two will be fused into one in the future.

MOVIE

Scripts for visualizing the progress of an INSPYRED algorithm in the parameter space, using the ind_file.txt and _settings.xml. 

    Must be given:  -axes labels
                    -in case of a test model the exact parameter combination (currently only works like that)
                    -projection type 2 or 3D. Generally I made 3D possible if there are only 3 parameters. Having 2 and 3D projections in every case will result in way too many subplots.
                    -the path of the aformentioned files
