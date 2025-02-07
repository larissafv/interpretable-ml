import numpy as np
import random

class Individuo:
    def __init__(self, value, changes, pred, fitness=None):
        self.value = value
        self.changes = changes
        self.pred = pred
        self.fitness = fitness

    def crossover(self, other, n_leads, interval, model, threshold, label_idx) -> None:
        leads = np.random.choice(12, n_leads)
        for lead in leads:
            idx = np.random.choice(4096, 1)
            if idx < 4095 - interval:
                start_idx = idx[0]
            else:
                start_idx = 4095 - interval
            for i in range(interval):
                self.value[start_idx+i][lead], other.value[start_idx+i][lead] = other.value[start_idx+i][lead], self.value[start_idx+i][lead]
                self.changes[start_idx+i][lead], other.changes[start_idx+i][lead] = other.changes[start_idx+i][lead], self.changes[start_idx+i][lead]

        preds = model(np.array([self.value, other.value]), training=False)
        mask = preds > threshold
        y_pred = np.zeros_like(preds)
        y_pred[mask] = 1
        self.pred = y_pred[0][label_idx]
        other.pred = y_pred[1][label_idx]

        return None


    def mutation(self, n_leads, n_points, max_rate, model, threshold, label_idx) -> None:
        leads = np.random.choice(12, n_leads)
        for lead in leads:
            idxs = np.random.choice(4096, n_points, replace = False)
            for idx in idxs:
                rate = random.uniform(0, max_rate)
                direction = np.random.choice([-1, 1], 1)[0]
                self.value[idx][lead] += rate * direction
                self.changes[idx][lead] = rate
        preds = model(np.array([self.value]), training=False)
        mask = preds > threshold
        y_pred = np.zeros_like(preds)
        y_pred[mask] = 1
        self.pred = y_pred[0][label_idx]
        return None

class GP:
    def __init__(
        self,
        pop_size,
        max_rate,
        generations,
        prob_crossover,
        prob_mutation,
        tournament_size,
        ecg,
        original_pred,
        n_points,
        n_leads,
        interval,
        model,
        threshold,
        file_name,
        label_idx
    ):
        self.population = []
        self.pop_size = pop_size
        self.max_rate = max_rate
        self.generations = generations
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.tournament_size = tournament_size
        self.ecg = ecg
        self.original_pred = original_pred
        self.n_points = n_points
        self.n_leads = n_leads
        self.interval = interval
        self.model = model
        self.threshold = threshold
        self.file_name = file_name
        self.label_idx = label_idx

        self.init_pop()
        self.calculate_fitness()

    def init_pop(self) -> None:
        self.population = [Individuo(self.ecg.copy(), np.zeros_like(self.ecg), self.original_pred) for _ in range(self.pop_size)]
        for individual in self.population:
            individual.mutation(self.n_leads, np.random.choice(2048, 1)[0], self.max_rate, self.model, self.threshold, self.label_idx)

    def calculate_fitness(self) -> None:
        for individual in self.population:
            if self.original_pred - individual.pred == 0:
                pred_penalty = 5000
            else:
                pred_penalty = 0
            nonzero = np.nonzero(individual.changes)[0].shape[0]
            aux_nonzero = np.nonzero(individual.changes)[0].shape[0]
            nonzero_array = np.nonzero(individual.changes)
            for i in range(aux_nonzero):
                diff = abs(individual.value[nonzero_array[0][i]][nonzero_array[1][i]] - self.ecg[nonzero_array[0][i]][nonzero_array[1][i]])
                if diff == 0.0:
                    nonzero -= 1
                individual.changes[nonzero_array[0][i]][nonzero_array[1][i]] = diff
            individual.fitness = pred_penalty + np.sum(individual.changes) + nonzero

    def select_parents(self):
        tournament = random.sample(self.population, self.tournament_size)
        best = min(tournament, key=lambda x: x.fitness)
        parent1 = (best.value.copy(), best.changes.copy(), best.pred, best.fitness)

        tournament = random.sample(self.population, self.tournament_size)
        best = min(tournament, key=lambda x: x.fitness)
        parent2 = (best.value.copy(), best.changes.copy(), best.pred, best.fitness)

        return (parent1, parent2,)


    def find_best(self) -> Individuo:
        return min(self.population, key=lambda x: x.fitness)

    def iterate(self) -> None:
        new_population = []
        curr_best = self.find_best()
        new_population.append(curr_best)

        while len(new_population) < self.pop_size:
            parent1, parent2 = self.select_parents()

            child1 = Individuo(parent1[0], parent1[1], parent1[2], parent1[3])
            child2 = Individuo(parent2[0], parent2[1], parent2[2], parent2[3])

            if random.random() < self.prob_crossover:
                child1.crossover(child2, self.n_leads, self.interval, self.model, self.threshold, self.label_idx)
            if random.random() < self.prob_mutation:
                child1.mutation(self.n_leads, self.n_points, self.max_rate, self.model, self.threshold, self.label_idx)
            if random.random() < self.prob_mutation:
                child2.mutation(self.n_leads, self.n_points, self.max_rate, self.model, self.threshold, self.label_idx)

            new_population.append(child1)
            new_population.append(child2)

        self.population = new_population
        self.calculate_fitness()

    def run(self) -> None:
        for generation in range(self.generations):
            print(f"Generation {generation + 1}")
            self.iterate()
            best = self.find_best()
            if best.fitness == 5.0:
                break
        best = self.find_best()
        return best.value