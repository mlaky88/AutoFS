from niapy.problems import Problem
from niapy.task import Task, OptimizationType
import numpy as np
from niaautofs.utils import calculate_dimension_of_the_problem, float_to_category, float_to_num, threshold
#from niaautofs.filter_methods import mutual_information
import copy


class FsProblem(Problem):
    def __init__(self, dataset, evaluation_metrics, evaluation_metrics_weights):
        super().__init__(len(dataset.features)-1, 0, 1)
        self.dataset = dataset
        self.evaluation_metrics = evaluation_metrics
        self.evaluation_metrics_weights = evaluation_metrics_weights        

    def _evaluate(self, x):
        r"""Evaluate the feature selection problem."""

        # x is a vector of presence/absence of features
        # x = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        #First filter then wrapper.
        print(type(x))
        print(x)        
        feature_count = np.sum(x>0.5)
        #mutual_information(self.dataset)

        pass


class AutoFsProblem(Problem):
    def __init__(self,data,inner_algorithms,evaluation_metrics,hyperparameters,optimize_evaluation_metrics_weights):
        
        dimension = calculate_dimension_of_the_problem(hyperparameters, evaluation_metrics, optimize_evaluation_metrics_weights)
        super().__init__(dimension, 0,1)

        self.data = data
        self.inner_algorithms = inner_algorithms
        self.evaluation_metrics = evaluation_metrics
        self.hyperparameters = hyperparameters
        self.optimize_evaluation_metrics_weights = optimize_evaluation_metrics_weights

        self.best_fitness = -np.inf
        self.best_solution = None

    def get_best_solution(self):
        return self.best_solution

    def _evaluate(self, x):
        r"""Evaluate the feature selection problem."""

        inner_algorithm = self.inner_algorithms[float_to_category(
            self.inner_algorithms, x[0])]
        
        pos_x = 1

        hyperparameters = float_to_num(self.hyperparameters, x[pos_x:pos_x + len(self.hyperparameters)])

        pos_x += len(self.hyperparameters)

        selected_metrics_idxs, selected_metrics = threshold(self.evaluation_metrics, x[pos_x:pos_x + len(self.evaluation_metrics)])

        if not selected_metrics:
            return -np.inf
        
        pos_x += len(self.evaluation_metrics)

        metrics_weights = [1] * len(selected_metrics)
        if self.optimize_evaluation_metrics_weights:            
            metrics_weights = x[pos_x:pos_x + len(selected_metrics)]
            metrics_weights = [metrics_weights[i] for i in selected_metrics_idxs]

        inner_algorithm.population_size = hyperparameters[0]

        print(inner_algorithm.Name[1])
        print(hyperparameters)
        print(selected_metrics)
        print(metrics_weights)
        
        fsproblem = FsProblem(
            dataset=self.data,
            evaluation_metrics=selected_metrics,
            evaluation_metrics_weights=metrics_weights
        )
        
        task = Task(problem=fsproblem, max_evals=hyperparameters[1], optimization_type=OptimizationType.MAXIMIZATION)

        _, fitness = inner_algorithm.run(task)

        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = copy.deepcopy(x)

        return fitness