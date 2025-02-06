from niapy.util.factory import get_algorithm
from niapy.task import Task, OptimizationType
from niaautofs.autofsproblem import AutoFsProblem

__all__ = ['AutoFsOptimizer']


class AutoFsOptimizer:
    def __init__(self, **kwargs):
        self.data = None
        self.inner_algorithms = None
        self.pipeline_evaluation_algorithm = None
        self.hyperparameters = None
        self.log_output_file = None
        self.log_verbose = False

        self.set_parameters(**kwargs)    

    def set_parameters(self, data, inner_algorithms, inner_filter_methods, pipeline_evaluation_algorithm,hyperparameters, log_output_file, log_verbose=False):
        self.data = data
        self.inner_algorithms = inner_algorithms
        self.inner_filter_methods = inner_filter_methods
        self.pipeline_evaluation_algorithm = pipeline_evaluation_algorithm
        self.hyperparameters = hyperparameters
        self.log_output_file = log_output_file
        self.log_verbose = log_verbose

    def get_data(self):
        return self.data
    
    def get_algorithms(self):
        return self.inner_algorithms
    
    def get_inner_filter_methods(self):
        return self.inner_filter_methods
    
    def get_hyperparameters(self):
        return self.hyperparameters
    
    def run(self, outer_algorithm, population_size, max_evals, seed, dataset_name, optimize_evaluation_metrics_weights=False):

        algorithm = get_algorithm(outer_algorithm,population_size=population_size,seed=seed)

        problem = AutoFsProblem(
            data=self.data,
            dataset_name=dataset_name,
            inner_algorithms=self.inner_algorithms, 
            filter_methods = self.inner_filter_methods,
            pipeline_evaluation_algorithm=self.pipeline_evaluation_algorithm, 
            hyperparameters=self.hyperparameters, 
            optimize_evaluation_metrics_weights=optimize_evaluation_metrics_weights
            )

        task = Task(
            problem=problem, 
            max_evals=max_evals, 
            optimization_type=OptimizationType.MAXIMIZATION,
            enable_logging=True
            )

        best_pipeline, fitness = algorithm.run(task)
        print("Best solution:", best_pipeline, fitness)
        return problem.get_best_solution()