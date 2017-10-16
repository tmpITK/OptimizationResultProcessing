import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GeneralPlotter(object):
    def __init__(self, algorithm_name, model_name, directory, features=''):
        self.algorithm_name = algorithm_name
        self.model_name = model_name
        self.directory = directory
        self.features = features

    def create_generation_plot(self, statistics, title=''):
        fig = plt.figure()

        plt.plot([row[0] for row in statistics], 'r.-', label="max", linewidth=1.5)
        plt.plot([row[1] for row in statistics], 'r--', label="min", linewidth=1.5)
        plt.plot([row[2] for row in statistics], 'r', label="median", linewidth=1.5)

        fig.suptitle('{0}{1} on {2}'.format(title, self.algorithm_name, self.model_name))
        plt.xlabel('generations')
        plt.ylabel('score value')
        plt.yscale('log')

        plt.legend(fontsize=14, ncol=1)
        plt.savefig(self.directory + '{0}{1} on {2}'.format(title, self.algorithm_name, self.model_name), format='pdf')
        plt.close()


    def create_pareto_plot(self, best_individuals, title="Pareto Front"):
        number_of_objectives = len(self.features)
        OBJECTIVE_NUMBER = range(number_of_objectives)

        x = []
        y = []
        for individual in best_individuals:
            x.append(individual[OBJECTIVE_NUMBER[0]])
            y.append(individual[OBJECTIVE_NUMBER[1]])

        if number_of_objectives > 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            z = [row[OBJECTIVE_NUMBER[2]] for row in best_individuals]
            tuner_z = self.tune_limit(z)
            ax.scatter(x, y, z, color='b')
            ax.set_zlim(min(z) - tuner_z, max(z) + tuner_z)
            ax.set_zlabel(self.features[OBJECTIVE_NUMBER[2]])
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(x, y, color='b')

        tuner_x = self.tune_limit(x)
        tuner_y = self.tune_limit(y)
        ax.set_xlim(min(x) - tuner_x, max(x) + tuner_x)
        ax.set_ylim(min(y) - tuner_y, max(y) + tuner_y)
        fig.suptitle('{0} of {1} on {2}'.format(title, self.algorithm_name, self.model_name))
        ax.autoscale_view(True, True, True)

        ax.set_xlabel(self.features[OBJECTIVE_NUMBER[0]])
        ax.set_ylabel(self.features[OBJECTIVE_NUMBER[1]])
        plt.savefig(self.directory + '{0} of {1} on {2}'.format(title, self.algorithm_name, self.model_name),format='pdf')
        plt.close()

    @staticmethod
    def tune_limit(values):
        return (max(values) - min(values)) / 100
