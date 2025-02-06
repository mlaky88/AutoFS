import argparse, time

from niaautofs.dataset import Dataset
from niaautofs.autofsoptimizer import AutoFsOptimizer
from niapy.algorithms.basic import ParticleSwarmOptimization, DifferentialEvolution, GeneticAlgorithm
from niapy.algorithms.basic.ga import uniform_crossover, uniform_mutation
from niapy.algorithms.modified import LpsrSuccessHistoryAdaptiveDifferentialEvolution, SelfAdaptiveDifferentialEvolution

import niaautofs.filter_methods

def parse_cli():
    cli_parser = argparse.ArgumentParser(description="Run the AutoFS framework.")
    cli_parser.add_argument("--dataset", type=str, default="Abalone", help="Dataset name. Dataset must be in the datasets folder.")
    cli_parser.add_argument("--algorithm", type=str, default="ParticleSwarmAlgorithm", help="Algorithm to use for optimization of the pipelines.")
    cli_parser.add_argument("--popsize", type=int, default=30, help="Population size.")
    cli_parser.add_argument("--maxfes", type=int, default=500, help="Maximum number of pipeline evaluations.")
    cli_parser.add_argument("--ow", dest="ow", default=False, action="store_true", help="Optimize evaluation metric weights.")   
    cli_parser.add_argument("--use_alpha", dest="use_alpha", default=False, action="store_true", help="Use alpha in subset evaluation")
    cli_parser.add_argument("--seed", type=int, default=37, help="Random seed.")
    cli_parser.add_argument("--run", type=int, default=1, help="Run number")
    cli_parser.add_argument("--folder", type=str, help="Folder name for the output files (.ppln).")

    return cli_parser.parse_args()

if __name__ == '__main__':

    cli = parse_cli()
    
    # dataset must have a columns named "class" for classification
    data = Dataset("datasets/{}.csv".format(cli.dataset))
    #print(data)
    #print(data.transactions)

    import numpy as np
    x = [np.array([1, 0, 1, 0, 1, 0, 1, 0]),np.array([1, 0, 1, 0, 1, 0, 1, 1]),np.array([1, 0, 1, 0, 1, 0, 1, 0])]
    fm = niaautofs.filter_methods.FilterMethods(data,"Abalone")

    for i in x:
        print(fm.mi(i))
        print(fm.ncfs(i))



    #exit(1)
    

    # define which preprocessing methods to use
    #preprocessing = ["min_max_scaling", "squash_cosine", "z_score_normalization", "remove_highly_correlated_features", "discretization_kmeans"]

    # define evolutionary/swarm intelligence algorithms for inner optimization
    algorithms = [ParticleSwarmOptimization(min_velocity=-4, max_velocity=4,seed=cli.seed),
                    DifferentialEvolution(crossover_probability=0.9, differential_weight=0.5,seed=cli.seed),
                    GeneticAlgorithm(crossover=uniform_crossover, mutation=uniform_mutation, crossover_rate=0.9, mutation_rate=0.1,seed=cli.seed), 
                    LpsrSuccessHistoryAdaptiveDifferentialEvolution(seed=cli.seed),
                    SelfAdaptiveDifferentialEvolution(seed=cli.seed)]

    # define inner algorithms hyperparameters and their min/max values
    hyperparameter1 = {
        "parameter": "NP",
        "min": 10,
        "max": 30
    }

    hyperparameter2 = {
        "parameter": "N_FES",
        "min": 2000,
        "max": 5000
    }
    # create array of hyperparameters
    hyperparameters = [hyperparameter1, hyperparameter2]

    # evaluation criteria

    filter_method1 = {  # chromosome encoding : presence, val for all params
        "name": "mi",
        "beta": {"min": 0, "max": 1}
    } #2
    filter_method2 = {
        "name": "mrmr",
        "alpha": {"min": 0.75, "max": 1}, # < 45 features otherwise [1.5,2]
        "beta": {"min": 0, "max": 1}
    } #3
    filter_method3 = {
        "name": "ncfs",
        "beta": {"min": 0.05, "max": 0.5}
    } #2
    filter_method4 = {
        "name": "relevance_penalty",
        "alpha": {"min": 0.05, "max": 0.5}
    } #2

    inner_filter_methods = [filter_method1, filter_method2, filter_method3, filter_method4]

    pipeline_evaluation_algorithm = "5NN"

    start_run = time.time()
    AutoFsOptimizer(
        data=data,
        inner_algorithms=algorithms,
        inner_filter_methods=inner_filter_methods,
        pipeline_evaluation_algorithm=pipeline_evaluation_algorithm,
        hyperparameters=hyperparameters,
        log_output_file="{}.fs".format(cli.folder),
        log_verbose=True
    ).run(cli.algorithm, cli.popsize, cli.maxfes, cli.seed, cli.dataset,cli.ow)

    end_run = time.time()

    print("Run time: {:.4f} seconds".format(end_run - start_run))    