from niapy.problems import Problem
from niapy.task import Task, OptimizationType
import numpy as np
from niaautofs.utils import calculate_dimension_of_the_problem, float_to_category, float_to_num, threshold_filter_methods
import copy
from niaautofs.filter_methods import FilterMethods
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from niaautofs.dataset import Dataset


class FsProblem(Problem):
    def __init__(self, dataset, evaluation_metrics, evaluation_metrics_weights,dataset_name):
        super().__init__(len(dataset.features)-1, 0, 1)
        self.dataset = dataset
        self.evaluation_metrics = evaluation_metrics
        self.evaluation_metrics_weights = evaluation_metrics_weights
        self.fs = FilterMethods(dataset,dataset_name)

    def _evaluate(self, x):
        r"""Evaluate the feature selection problem."""

        fitness = 0
        for filter_method, weight in zip(self.evaluation_metrics,self.evaluation_metrics_weights):
            filter_name = filter_method["name"]
            if filter_name == "mi":
                fitness += self.fs.mi(x) * weight
            elif filter_name == "mrmr":
                fitness += self.fs.mrmr(x) * weight
            elif filter_name == "ncfs":
                fitness += self.fs.ncfs(x) * weight
            elif filter_name == "relevance_penalty":
                fitness += self.fs.relevance(x) * weight
            else:
                print("Filter method not found")
                return -np.inf
            
        return fitness / sum(self.evaluation_metrics_weights)

class AutoFsProblem(Problem):
    def __init__(self,data,dataset_name, inner_algorithms, filter_methods, pipeline_evaluation_algorithm,hyperparameters,optimize_evaluation_metrics_weights):

        self.num_vals = sum([len(filter_method.keys()) for filter_method in filter_methods])
        dimension = calculate_dimension_of_the_problem(hyperparameters, filter_methods,pipeline_evaluation_algorithm, optimize_evaluation_metrics_weights)
        print("Dimension:",dimension)
        lower = np.zeros(dimension)
        lower[1+len(hyperparameters)+self.num_vals:-1] = 0.01
        upper = np.ones(dimension)
        upper[-1] = 0.99

        super().__init__(dimension, upper=upper, lower=lower)
        

        self.data = data
        self.dataset_name = dataset_name
        self.inner_algorithms = inner_algorithms
        self.filter_methods = filter_methods
        self.pipeline_evaluation_algorithm = pipeline_evaluation_algorithm
        self.hyperparameters = hyperparameters
        self.optimize_evaluation_metrics_weights = optimize_evaluation_metrics_weights

        self.best_solution = None
        self.best_fitness = -np.inf

        train_data, test_data, y_train, y_test = train_test_split(self.data.transactions.drop(columns="class"),self.data.transactions["class"] ,test_size=0.3, random_state=42, stratify=self.data.transactions["class"])
        train_data.insert(0,"class",y_train)
        test_data.insert(0,"class",y_test)
        self.train_data = Dataset(train_data)
        self.test_data = Dataset(test_data)

    def get_best_solution(self):
        return self.best_solution
    
    def calculate_fitness(self, x, alpha, N=3): #Now still using whole dataset
        knn = KNeighborsClassifier(n_neighbors=N, n_jobs=-1)

        X = self.train_data.transactions.drop(columns="class")
        X = X.drop(columns=X.columns[x < 0.5])
        y = self.train_data.transactions["class"]

        knn.fit(X, y)
        X_test = self.test_data.transactions.drop(columns="class")
        X_test = X_test.drop(columns=X_test.columns[x < 0.5])
        y_test = self.test_data.transactions["class"]
        y_pred = knn.predict(X_test)
        
        accuracy = accuracy_score(y_test.values, y_pred)

        num_selected = sum([1 if xi >= 0.5 else 0 for xi in x])  

        print("fit(x) = acc({:.4f}) * {:.4f} - ({:.4f}) * ({} / {})".format(accuracy,alpha,1-alpha,num_selected,len(x)))
        return accuracy * alpha - (1-alpha) * (num_selected / len(x))

    def _evaluate(self, x):
        r"""Evaluate the feature selection problem."""

        inner_algorithm = self.inner_algorithms[float_to_category(
            self.inner_algorithms, x[0])]
        
        pos_x = 1

        hyperparameters = float_to_num(self.hyperparameters, x[pos_x:pos_x + len(self.hyperparameters)])

        pos_x += len(self.hyperparameters)

        selected_metrics_idxs, selected_metrics = threshold_filter_methods(self.filter_methods, x[pos_x:pos_x + self.num_vals])

        if not selected_metrics:
            return -np.inf
        
        pos_x += self.num_vals

        metrics_weights = [1] * len(selected_metrics)
        if self.optimize_evaluation_metrics_weights:            
            metrics_weights = x[pos_x:pos_x + len(self.filter_methods)]
            metrics_weights = [metrics_weights[i] for i in selected_metrics_idxs]

        if sum(metrics_weights) == 0:
            return -np.inf
        
        alpha = x[-1]

        inner_algorithm.population_size = hyperparameters[0]
                
        fsproblem = FsProblem(
            dataset=self.train_data,
            evaluation_metrics=selected_metrics,
            evaluation_metrics_weights=metrics_weights,
            dataset_name=self.dataset_name
        )
        
        task = Task(problem=fsproblem, max_evals=hyperparameters[1], optimization_type=OptimizationType.MAXIMIZATION) #check

        best_solution, inner_fitness = inner_algorithm.run(task)
        
        if sum(best_solution[best_solution>0.5]) == 0:
            return -np.inf
        
        fitness = self.calculate_fitness(best_solution, alpha)
        if fitness > self.best_fitness:
            self.best_solution = best_solution
            self.best_fitness = fitness

            print("\nNew best fitness: {} \n".format(fitness))
            print("Algorithm: {}".format(inner_algorithm.Name[1]))
            print("Hyperparams: {}".format(hyperparameters))
            print("Metrics:")
            for i, metric in enumerate(selected_metrics):
                for key in metric.keys():
                    if key == "name":
                        print("\t{}({:.3f}): ".format(metric[key],metrics_weights[i]),end="")
                    else:
                        print("{}={:.4f} ".format(key, metric[key]),end="")     
                print()

        return fitness