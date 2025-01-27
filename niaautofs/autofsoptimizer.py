from niapy.util.factory import get_algorithm
from niapy.task import Task, OptimizationType
from niaautofs.autofsproblem import AutoFsProblem

__all__ = ['AutoFsOptimizer']


class AutoFsOptimizer:
    def __init__(self, **kwargs):
        self.data = None
        self.inner_algorithms = None
        self.evaluation_metrics = None
        self.hyperparameters = None
        self.log_output_file = None
        self.log_verbose = False

        self.set_parameters(**kwargs)    

    def set_parameters(self, data, inner_algorithms, inner_wrapper_classifiers, evaluation_metrics,hyperparameters, log_output_file, log_verbose=False):
        self.data = data
        self.inner_algorithms = inner_algorithms
        self.inner_wrapper_classifiers = inner_wrapper_classifiers
        self.evaluation_metrics = evaluation_metrics
        self.hyperparameters = hyperparameters
        self.log_output_file = log_output_file
        self.log_verbose = log_verbose

    def get_data(self):
        return self.data
    
    def get_algorithms(self):
        return self.algorithms
    
    def get_evaluation_metrics(self):
        return self.evaluation_metrics
    
    def get_hyperparameters(self):
        return self.hyperparameters
    
    def run(self, outer_algorithm, population_size, max_evals, seed, optimize_evaluation_metrics_weights=False):

        algorithm = get_algorithm(outer_algorithm,population_size=population_size,seed=seed)

        problem = AutoFsProblem(
            data=self.data, 
            inner_algorithms=self.inner_algorithms, 
            evaluation_metrics=self.evaluation_metrics, 
            hyperparameters=self.hyperparameters, 
            optimize_evaluation_metrics_weights=optimize_evaluation_metrics_weights
            )

        task = Task(
            problem=problem, 
            max_evals=max_evals, 
            optimization_type=OptimizationType.MAXIMIZATION
            )

        algorithm.run(task)

        return problem.get_best_solution()