"""Module for generating counterfactual explanations using genetic algorithms.

This module implements a counterfactual explanation system based on genetic
algorithms for machine learning models. The goal is to find the minimal
modifications needed in an instance to change the model's prediction.

Classes:
    CounterfactualExplanation: Main class that implements the genetic algorithm
        to generate counterfactual explanations.

Typical usage example:

    from counterfactual_explanations import CounterfactualExplanation
    from counterfactual_explanations.structs import CEConfig
    
    config = CEConfig(pop_size=50, n_generations=100)
    ce = CounterfactualExplanation(model, config)
    ce.explain(input_data, instance_idx, ground_truth, format_instance)
"""

import copy
import logging
import random
from pathlib import Path

import numpy as np

from common.tools import get_optimal_precision_recall
from common.views import plot_timeseries_differences
from counterfactual_explanations.individual import Individual
from counterfactual_explanations.structs import CEConfig
from instances.format_instance import FormatInstance
from model.model import Model

logger = logging.getLogger(__name__)

class CounterfactualExplanation:
    """Generates counterfactual explanations using genetic algorithms.

    This class implements a genetic algorithm to find counterfactual explanations
    for machine learning models. It evolves a population of candidate solutions
    to find minimal modifications that change the model's prediction.

    Attributes:
        _model: The machine learning model to explain.
        _ce_type: Type of counterfactual explanation method.
        _n_slices: Number of slices for time series data.
        _sampling_rate: Sampling rate for data processing.
        _max_range: Maximum range for mutations.
        _pop_size: Size of the genetic algorithm population.
        _n_generations: Maximum number of generations to evolve.
        _min_fitness: Minimum fitness threshold for early stopping.
        _prob_crossover: Probability of crossover operation.
        _prob_mutation: Probability of mutation operation.
        _prob_change_piece: Probability of changing a data piece.
        _tournament_size: Size of tournament for parent selection.
        _label_idx: Index of the target label/class.
        _max_retries: Maximum number of retries for operations.
    """

    def __init__(self, model: Model, config: CEConfig) -> None:
        """Initializes the CounterfactualExplanation with model and configuration.

        Args:
            model: The machine learning model to generate explanations for.
            config: Configuration object containing all hyperparameters for
                the genetic algorithm and counterfactual generation process.
        """
        # Store model reference
        self._model = model
        
        # Genetic algorithm configuration
        self._ce_type = config.ce_type
        self._n_slices = config.n_slices
        self._sampling_rate = config.sampling_rate
        self._max_range = config.max_range
        self._pop_size = config.pop_size
        self._n_generations = config.n_generations
        self._min_fitness = config.min_fitness
        
        # Genetic operators probabilities
        self._prob_crossover = config.prob_crossover
        self._prob_mutation = config.prob_mutation
        self._prob_change_piece = config.prob_change_piece
        
        # Selection and retry parameters
        self._tournament_size = config.tournament_size
        self._label_idx = config.label_idx
        self._max_retries = config.max_retries

    def explain(
            self,
            input_model: np.ndarray,
            instance_idx: int,
            ground_truth: np.ndarray,
            format_instance: FormatInstance,
            variate_labels: list[str] | None,
            class_labels: list[str] | None,
            probs: np.ndarray | None = None
        ) -> None:
        """Generates a counterfactual explanation for a specific instance.

        This method runs the complete counterfactual explanation pipeline:
        1. Formats the target instance
        2. Calculates the optimal threshold for classification
        3. Determines the prediction goal (opposite of current prediction)
        4. Initializes and evolves the population using genetic algorithm
        5. Generates and saves a visualization of the result

        Args:
            input_model: Array containing all input instances for the model.
            instance_idx: Index of the specific instance to explain.
            ground_truth: True labels for the input instances.
            format_instance: Formatter object to preprocess instances.
            variate_labels: Optional labels for variables in time series data.
            class_labels: Optional labels for classes/categories.

        Returns:
            None: The method saves a plot visualization as a side effect.

        Note:
            The method modifies internal state and generates visualization files
            in the current working directory or specified output path.
        """
        # Store instance formatter and format the target instance
        self._format_instance = format_instance
        self._instance = self._format_instance.format(input_model[instance_idx])
        self._probs = None if probs is None else self._normalize_array(probs[self._label_idx][0])
        
        # Calculate optimal classification threshold
        self._threshold = self._get_threshold(input_model, ground_truth)

        # Get current prediction and set opposite as goal
        instance_pred = self._model.predict(np.array([[input_model[instance_idx]]]))
        original_pred = 1 if instance_pred[0][self._label_idx] > self._threshold else 0
        self._goal = 1 - original_pred  # Flip the prediction

        # Run the genetic algorithm
        self._init_pop()
        if len(self.population) == 0:
            print("No valid individuals could be created. Try increasing max_retries or adjusting parameters.")  # noqa: E501
            return None
        self._calculate_fitness()
        cs_instance, cs_fitness = self._run()
        
        # Generate visualization
        if class_labels is None:
            plot_name = Path(
                f"counterfactual_explanation_{self._pop_size}_{self._n_generations}_{self._ce_type}.png"
            )
        else:
            plot_name = Path(
                f"counterfactual_explanation_{self._pop_size}_{self._n_generations}_{self._ce_type}_{class_labels[self._label_idx]}.png"
            )
        
        plot_timeseries_differences(
            self._instance[0], cs_instance[0], plot_name, variate_labels
        )
        
        print("FITNESS:", cs_fitness)
        return cs_fitness

    def _get_threshold(self, input_model: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculates the optimal classification threshold for the model.

        Uses precision-recall optimization to determine the best threshold
        for binary classification decisions.

        Args:
            input_model: Input data for threshold calculation.
            ground_truth: True labels corresponding to input_model.

        Returns:
            float: Optimal threshold value for the specified label index.
        """
        y_pred = self._model.predict(input_model)
        _, _, _, thresholds = get_optimal_precision_recall(ground_truth, y_pred)
        return thresholds[self._label_idx]

    def _init_pop(self) -> None:
        """Initializes the genetic algorithm population.

        Creates an initial population of individuals by applying random mutations
        to copies of the original instance. Each individual represents a potential
        counterfactual explanation candidate.

        The method attempts to create `_pop_size` valid individuals, with up to
        `_max_retries` attempts per individual to generate a successful mutation.

        Raises:
            No explicit exceptions, but logs warnings if population size is
            smaller than expected due to failed mutations.

        Note:
            The population is stored in `self.population` and each individual
            contains a mutated version of the original instance.
        """
        self.population = []

        # Attempt to create the specified number of individuals
        for _ in range(self._pop_size):
            # Create individual with deep copy to avoid reference issues
            individual = Individual(copy.deepcopy(self._instance))

            # Try multiple times to generate a valid mutation
            aux_prob_change_piece = self._prob_change_piece
            for _ in range(self._max_retries):
                success = individual.mutation(
                    self._instance,
                    self._model,
                    self._ce_type,
                    self._n_slices,
                    self._sampling_rate,
                    self._max_range,
                    aux_prob_change_piece,
                    self._label_idx,
                    self._max_retries,
                    self._threshold,
                    self._goal,
                    self._format_instance,
                    self._probs
                )
                if success:
                    self.population.append(individual)
                    logger.info(f"Added individual to population: {len(self.population)}")
                    break  # Move to next individual
                
                # Increase probabilities slightly for next attempt
                aux_prob_change_piece = round(aux_prob_change_piece * 1.05, 3)
                logger.info(f"Probabilities updated: {aux_prob_change_piece}")
        
        logger.info(f"Initialized population with {len(self.population)} individuals")

    def _calculate_fitness(self) -> None:
        """Calculates fitness values for all individuals in the population.
        
        Fitness is calculated based on two main components:
        1. Number of differences between individual.value and original instance
        2. Sum of absolute differences between the arrays
        
        The fitness function encourages solutions that are both sparse
        (fewer changes) and minimal (smaller magnitude changes).
        
        Lower fitness values indicate better solutions, as the goal is to
        minimize both the number and magnitude of changes needed.
        
        Note:
            This method modifies the `fitness` attribute of each individual
            in the population. The fitness calculation uses L1 norm for
            measuring differences.
        """
        for individual in self.population:
            # Create binary mask showing positions where values differ
            differences_mask = (individual.value != self._instance).astype(int)
            
            # Count total number of positions that were modified
            num_differences = np.sum(differences_mask)
            
            # Calculate sum of absolute differences (L1 norm)
            absolute_differences = np.abs(individual.value - self._instance)
            sum_absolute_differences = np.sum(absolute_differences)
            
            # Combine both metrics - can be adjusted based on requirements
            # Lower values indicate better fitness (fewer and smaller changes)
            individual.fitness = num_differences + sum_absolute_differences
            

    def _select_parents(self) -> tuple:
        """Selects two parents using tournament selection.

        Tournament selection works by randomly sampling a subset of individuals
        from the population and selecting the one with the best (lowest) fitness.
        This process is repeated twice to get two parents.

        Returns:
            tuple: A tuple containing two parent tuples. Each parent tuple
                contains (individual_value, fitness_score).

        Note:
            Uses deep copy to prevent unintended modifications to the original
            individuals when creating offspring.
        """
        # Select first parent through tournament selection
        tournament = random.sample(self.population, self._tournament_size)
        best = min(tournament, key=lambda x: x.fitness)
        parent1 = (copy.deepcopy(best.value), best.fitness)

        # Select second parent through tournament selection
        tournament = random.sample(self.population, self._tournament_size)
        best = min(tournament, key=lambda x: x.fitness)
        parent2 = (copy.deepcopy(best.value), best.fitness)

        return (
            parent1,
            parent2,
        )

    def _find_best(self) -> Individual:
        """Finds and returns the individual with the best (lowest) fitness.

        Returns:
            Individual: The individual with the minimum fitness value in the
                current population.

        Note:
            Lower fitness values indicate better solutions in this implementation.
        """
        return min(self.population, key=lambda x: x.fitness)

    def _iterate(self) -> None:
        """Performs one generation of the genetic algorithm.

        This method implements the core genetic algorithm loop:
        1. Preserves the best individual (elitism)
        2. Generates new offspring through selection, crossover, and mutation
        3. Replaces the old population with the new generation
        4. Recalculates fitness for all individuals

        The process uses:
        - Tournament selection for choosing parents
        - Probabilistic crossover and mutation operations
        - Elitism to preserve the best solution across generations

        Note:
            This method modifies the population in-place and should be called
            iteratively until convergence or maximum generations are reached.
        """
        new_population = []
        
        # Elitism: preserve the best individual from current generation
        curr_best = self._find_best()
        new_population.append(curr_best)

        # Generate offspring to fill the rest of the population
        for _ in range(self._pop_size):
            # Select parents using tournament selection
            parent1, parent2 = self._select_parents()

            # Create offspring from selected parents
            child1 = Individual(parent1[0], parent1[1])
            child2 = Individual(parent2[0], parent2[1])

            # Apply crossover with original instance with specified probability
            if random.random() < 0.3:  # noqa: S311
                for _ in range(self._max_retries):
                    success = child1.crossover_original(
                        self._instance,
                        self._model,
                        self._prob_change_piece,
                        self._label_idx,
                        self._max_retries,
                        self._threshold,
                        self._goal,
                        self._format_instance
                    )
                    if success:
                        logger.info("Crossover with original successful")
                        break  # Crossover successful, proceed

            if random.random() < 0.3:  # noqa: S311
                for _ in range(self._max_retries):
                    success = child2.crossover_original(
                        self._instance,
                        self._model,
                        self._prob_change_piece,
                        self._label_idx,
                        self._max_retries,
                        self._threshold,
                        self._goal,
                        self._format_instance
                    )
                    if success:
                        logger.info("Crossover with original successful")
                        break  # Crossover successful, proceed

            # Apply crossover with specified probability
            if random.random() < self._prob_crossover:  # noqa: S311
                for _ in range(self._max_retries):
                    success = child1.crossover(
                        child2,
                        self._model,
                        self._prob_change_piece,
                        self._label_idx,
                        self._max_retries,
                        self._threshold,
                        self._goal,
                        self._format_instance
                    )
                    if success:
                        logger.info("Crossover successful")
                        break  # Crossover successful, proceed
            
            # Apply mutation to first child with specified probability
            if random.random() < self._prob_mutation:  # noqa: S311
                aux_prob_change_piece = self._prob_change_piece
                for _ in range(self._max_retries):
                    success = child1.mutation(
                        self._instance,
                        self._model,
                        self._ce_type,
                        self._n_slices,
                        self._sampling_rate,
                        self._max_range,
                        aux_prob_change_piece,
                        self._label_idx,
                        self._max_retries,
                        self._threshold,
                        self._goal,
                        self._format_instance,
                        self._probs
                    )
                    if success:
                        logger.info("Mutation successful")
                        break  # Mutation successful, proceed

                    # Increase probabilities slightly for next attempt
                    aux_prob_change_piece = round(aux_prob_change_piece * 1.05, 3)
                    logger.info(f"Probabilities updated: {aux_prob_change_piece}")
            
            # Apply mutation to second child with specified probability
            if random.random() < self._prob_mutation:  # noqa: S311
                aux_prob_change_piece = self._prob_change_piece
                for _ in range(self._max_retries):
                    success = child2.mutation(
                        self._instance,
                        self._model,
                        self._ce_type,
                        self._n_slices,
                        self._sampling_rate,
                        self._max_range,
                        aux_prob_change_piece,
                        self._label_idx,
                        self._max_retries,
                        self._threshold,
                        self._goal,
                        self._format_instance,
                        self._probs
                    )
                    if success:
                        logger.info("Mutation successful")
                        break  # Mutation successful, proceed

                    # Increase probabilities slightly for next attempt
                    aux_prob_change_piece = round(aux_prob_change_piece * 1.05, 3)
                    logger.info(f"Probabilities updated: {aux_prob_change_piece}")

            # Add offspring to new population
            new_population.append(child1)
            new_population.append(child2)

        # Replace old population with new generation
        self.population = new_population
        
        # Recalculate fitness for all individuals in the new population
        self._calculate_fitness()

    def _run(self) -> np.ndarray:
        """Executes the complete genetic algorithm to find counterfactual explanations.

        Runs the genetic algorithm for the specified number of generations or
        until the minimum fitness threshold is reached. Each generation involves
        selection, crossover, mutation, and fitness evaluation.

        The algorithm uses elitism to preserve the best solution and applies
        probabilistic genetic operators to evolve the population toward better
        counterfactual explanations.

        Returns:
            np.ndarray: The best counterfactual instance found, represented as
                the value array of the individual with the lowest fitness.

        Note:
            Early stopping occurs if `_min_fitness` is specified and achieved.
            The method logs progress information for each generation.
        """
        # Evolution loop: run for specified number of generations
        for generation in range(self._n_generations):
            logger.info(f"Generation {generation + 1}")
            if len(self.population) == 1:
                best = self.population[0]
                logger.info(f"Only one individual left with fitness {best.fitness}")
                break

            # Perform one generation of genetic algorithm
            self._iterate()
            
            # Check for early stopping condition
            best = self._find_best()
            if self._min_fitness is not None and self._min_fitness >= best.fitness:
                logger.info(
                    f"Early stopping: fitness {best.fitness} reached minimum {self._min_fitness}"
                )
                break
            
            # Decay probabilities over generations
            if self._probs is not None:
                self._probs = self._probs * 0.95
            self._prob_change_piece = (
                round(self._prob_change_piece * 0.95, 3) 
                if self._prob_change_piece > 0.1
                else 0.1
            )
            logger.info(f"Probabilities decayed: {self._prob_change_piece}")
        
        # Return the best solution found
        best = self._find_best()
        print(f"Best fitness: {best.fitness}")
        return best.value, best.fitness

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalizes a NumPy array to the range [0, 1] using Min-Max Normalization.
        """
        if arr.ndim > 1:
            norm_arr = []
            for i in range(arr.shape[0]):
                arr_min = np.min(abs(arr[i]))
                arr_max = np.max(abs(arr[i]))

                if arr_max == arr_min:  # Handle cases where all values are the same
                    norm_arr.append(np.zeros_like(arr[i], dtype=float))

                norm_arr.append((abs(arr[i]) - arr_min) / (arr_max - arr_min))
        else:
            arr_min = np.min(abs(arr))
            arr_max = np.max(abs(arr))

            if arr_max == arr_min:  # Handle cases where all values are the same
                return np.zeros_like(arr, dtype=float)

            norm_arr = (abs(arr) - arr_min) / (arr_max - arr_min)
        return np.array(norm_arr)