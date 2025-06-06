class FsPipeline:
    def __init__(self, algorithm, algorithm_parameters, hyperparameters, filter_methods, filter_method_weights,selected_features,fitness,alpha):
        self.algorithm = algorithm
        self.algorithm_parameters = algorithm_parameters
        self.hyperparameters = hyperparameters
        self.filter_methods = filter_methods
        self.filter_method_weights = filter_method_weights
        self.selected_features = selected_features
        self.fitness = fitness
        self.alpha = alpha

    def __str__(self):
        return f"\nAlgorithm: {self.algorithm} \nParameters: {self.algorithm_parameters} \
            \nHyperparameters: {self.hyperparameters} \nFilter Methods: {self.filter_methods} \
                \nFilter Method Weights: {self.filter_method_weights} \nSelected Features: {sum([1 if x >=0.5 else 0 for x in self.selected_features.tolist()])}/{len(self.selected_features)} \
                 \nFitness: {self.fitness} \nAlpha: {self.alpha}"

    