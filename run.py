import numpy as np
import pandas as pd
import argparse, time

from niaarm.dataset import Dataset
from niaautofs.autofsoptimizer import AutoFsOptimizer
from niapy.algorithms.basic import ParticleSwarmOptimization, DifferentialEvolution, GeneticAlgorithm
from niapy.algorithms.basic.ga import uniform_crossover, uniform_mutation
from niapy.algorithms.modified import ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution, LpsrSuccessHistoryAdaptiveDifferentialEvolution, SelfAdaptiveDifferentialEvolution




def parse_cli():
    cli_parser = argparse.ArgumentParser(description="Run the AutoFS framework.")
    cli_parser.add_argument("--dataset", type=str, default="Abalone", help="Dataset name. Dataset must be in the datasets folder.")
    cli_parser.add_argument("--algorithm", type=str, default="ParticleSwarmAlgorithm", help="Algorithm to use for optimization of the pipelines.")
    cli_parser.add_argument("--popsize", type=int, default=30, help="Population size.")
    cli_parser.add_argument("--maxfes", type=int, default=500, help="Maximum number of pipeline evaluations.")
    cli_parser.add_argument("--ow", dest="ow", default=False, action="store_true", help="Optimize evaluation metric weights.")    cli_parser.add_argument("--sf", dest="sf", default=False, action="store_true", help="Use surrogate fitness.")
    cli_parser.add_argument("--seed", type=int, default=37, help="Random seed.")
    cli_parser.add_argument("--run", type=int, default=1, help="Run number")
    cli_parser.add_argument("--folder", type=str, help="Folder name for the output files (.ppln).")

    return cli_parser.parse_args()

if __name__ == '__main__':

    cli = parse_cli()
    
    data = Dataset("datasets/{}.csv".format(cli.dataset))

    # define which preprocessing methods to use
    preprocessing = ["min_max_scaling", "squash_cosine", "z_score_normalization", "remove_highly_correlated_features", "discretization_kmeans"]

    # define evolutionary/swarm intelligence algorithms for inner optimization
    algorithms = [ParticleSwarmOptimization(min_velocity=-4, max_velocity=4,seed=cli.seed),
                    DifferentialEvolution(crossover_probability=0.9, differential_weight=0.5,seed=cli.seed),
                    GeneticAlgorithm(crossover=uniform_crossover, mutation=uniform_mutation, crossover_rate=0.9, mutation_rate=0.1,seed=cli.seed), 
                    ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution(seed=cli.seed),
                    LpsrSuccessHistoryAdaptiveDifferentialEvolution(seed=cli.seed),
                    SelfAdaptiveDifferentialEvolution(seed=cli.seed)]

    # define hyperparameters and their min/max values
    hyperparameter1 = {
        "parameter": "NP",
        "min": 10,
        "max": 30
    }

    hyperparameter2 = {
        "parameter": "N_FES",
        "min": 2000,
        "max": 10000
    }
    # create array
    hyperparameters = [hyperparameter1, hyperparameter2]

    # evaluation criteria
    evaluation_metrics = [
        "feature_count",
        "accuracy_on_validation",
        "fisher_score",
        "mutual_information",
        "??",
        "relieff"]  

    start_run = time.time()

    end_run = time.time()
    print("Run time: {:.4f} seconds".format(end_run - start_run))    