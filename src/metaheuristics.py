"""
Módulo C: Metaheurísticas
Recocido Simulado y Algoritmo Genético sobre el orden de entregas.
"""

from __future__ import annotations
import random
import math
import time
import statistics
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from copy import deepcopy

from scenario import Scenario, Coord
from pathfinding import find_path_through_deliveries


@dataclass
class MetaResult:
    best_order: List[Coord]
    best_cost: float
    best_path: List[Coord]
    evaluations: int
    time_seconds: float
    history: List[float] = field(default_factory=list)
    algorithm: str = ""
    final_temp: float = 0.0
    generations: int = 0
    mean_cost: float = 0.0
    std_cost: float = 0.0

    def __str__(self):
        return (
            f"[{self.algorithm}] best={self.best_cost:.2f} | "
            f"evals={self.evaluations} | t={self.time_seconds:.2f}s"
        )


def evaluate_order(
    scenario: Scenario,
    order: List[Coord],
    competitor_strategy: Optional[str] = None,
) -> Tuple[float, List[Coord]]:
    res = find_path_through_deliveries(
        scenario, order, algorithm="astar", competitor_strategy=competitor_strategy
    )
    if not res.success:
        return float("inf"), []
    return res.cost, res.path


def random_order(deliveries: List[Coord], seed: Optional[int] = None) -> List[Coord]:
    order = deliveries[:]
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(order)
    else:
        random.shuffle(order)
    return order


# -------------------- Simulated Annealing --------------------

def simulated_annealing(
    scenario: Scenario,
    initial_temp: float = 100.0,
    final_temp: float = 0.1,
    cooling_rate: float = 0.95,
    iterations_per_temp: int = 50,
    competitor_strategy: Optional[str] = None,
    seed: Optional[int] = None,
) -> MetaResult:
    """
    Recocido simulado con aceptación probabilística de Metropolis
    y enfriamiento geométrico.
    """
    if seed is not None:
        random.seed(seed)

    current = random_order(scenario.deliveries, seed)
    current_cost, current_path = evaluate_order(scenario, current, competitor_strategy)
    best = current[:]
    best_cost = current_cost
    best_path = current_path[:]
    history = [current_cost]
    evaluations = 1
    t0 = time.perf_counter()

    temp = initial_temp
    while temp > final_temp:
        for _ in range(iterations_per_temp):
            # Vecino: swap de dos posiciones
            i, j = random.sample(range(len(current)), 2)
            neighbor = current[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            neigh_cost, neigh_path = evaluate_order(scenario, neighbor, competitor_strategy)
            evaluations += 1

            delta = neigh_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = neighbor
                current_cost = neigh_cost
                current_path = neigh_path
                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost
                    best_path = current_path[:]

            history.append(best_cost)

        temp *= cooling_rate

    elapsed = time.perf_counter() - t0
    return MetaResult(
        best_order=best,
        best_cost=best_cost,
        best_path=best_path,
        evaluations=evaluations,
        time_seconds=elapsed,
        history=history,
        algorithm="SimulatedAnnealing",
        final_temp=temp,
    )


# -------------------- Genetic Algorithm --------------------

def tournament_selection(
    population: List[List[Coord]],
    fitness: List[float],
    k: int = 3,
) -> List[Coord]:
    indices = random.sample(range(len(population)), k)
    best_idx = min(indices, key=lambda i: fitness[i])
    return population[best_idx][:]


def order_crossover(parent1: List[Coord], parent2: List[Coord]) -> Tuple[List[Coord], List[Coord]]:
    """Order Crossover (OX) clásico para permutaciones."""
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))

    def ox(p1, p2):
        child = [None] * n
        child[a:b+1] = p1[a:b+1]
        fill = [x for x in p2 if x not in child]
        idx = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child

    return ox(parent1, parent2), ox(parent2, parent1)


def mutate_swap(order: List[Coord], mutation_rate: float) -> List[Coord]:
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(order)), 2)
        order = order[:]
        order[i], order[j] = order[j], order[i]
    return order


def genetic_algorithm(
    scenario: Scenario,
    population_size: int = 40,
    generations: int = 80,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.15,
    tournament_size: int = 3,
    elitism: int = 2,
    competitor_strategy: Optional[str] = None,
    seed: Optional[int] = None,
) -> MetaResult:
    """
    Algoritmo genético con representación de permutación,
    selección por torneo, Order Crossover y mutación por intercambio.
    """
    if seed is not None:
        random.seed(seed)

    # Población inicial
    population = [random_order(scenario.deliveries) for _ in range(population_size)]
    fitness = []
    paths = []
    for ind in population:
        c, p = evaluate_order(scenario, ind, competitor_strategy)
        fitness.append(c)
        paths.append(p)

    evaluations = population_size
    best_idx = min(range(population_size), key=lambda i: fitness[i])
    best_order = population[best_idx][:]
    best_cost = fitness[best_idx]
    best_path = paths[best_idx][:]
    history = [best_cost]
    t0 = time.perf_counter()

    for gen in range(generations):
        new_pop = []
        new_fit = []
        new_paths = []

        # Elitismo
        elite_indices = sorted(range(population_size), key=lambda i: fitness[i])[:elitism]
        for idx in elite_indices:
            new_pop.append(population[idx][:])
            new_fit.append(fitness[idx])
            new_paths.append(paths[idx][:])

        # Generar el resto
        while len(new_pop) < population_size:
            p1 = tournament_selection(population, fitness, tournament_size)
            p2 = tournament_selection(population, fitness, tournament_size)

            if random.random() < crossover_rate:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            c1 = mutate_swap(c1, mutation_rate)
            c2 = mutate_swap(c2, mutation_rate)

            for child in (c1, c2):
                if len(new_pop) >= population_size:
                    break
                cost, path = evaluate_order(scenario, child, competitor_strategy)
                evaluations += 1
                new_pop.append(child)
                new_fit.append(cost)
                new_paths.append(path)

        population = new_pop
        fitness = new_fit
        paths = new_paths

        gen_best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
        if fitness[gen_best_idx] < best_cost:
            best_cost = fitness[gen_best_idx]
            best_order = population[gen_best_idx][:]
            best_path = paths[gen_best_idx][:]
        history.append(best_cost)

    elapsed = time.perf_counter() - t0
    return MetaResult(
        best_order=best_order,
        best_cost=best_cost,
        best_path=best_path,
        evaluations=evaluations,
        time_seconds=elapsed,
        history=history,
        algorithm="GeneticAlgorithm",
        generations=generations,
    )


def run_multiple(
    method: str,
    scenario: Scenario,
    num_runs: int = 20,
    params: Optional[Dict] = None,
    seeds: Optional[List[int]] = None,
    competitor_strategy: Optional[str] = None,
) -> Tuple[MetaResult, List[MetaResult]]:
    """Ejecuta varias veces y calcula estadísticas."""
    params = params or {}
    results = []
    for i in range(num_runs):
        seed = seeds[i] if seeds and i < len(seeds) else random.randint(0, 10**9)
        if method == "sa":
            res = simulated_annealing(
                scenario,
                initial_temp=params.get("initial_temp", 100.0),
                final_temp=params.get("final_temp", 0.1),
                cooling_rate=params.get("cooling_rate", 0.95),
                iterations_per_temp=params.get("iterations_per_temp", 50),
                competitor_strategy=competitor_strategy,
                seed=seed,
            )
        else:
            res = genetic_algorithm(
                scenario,
                population_size=params.get("population_size", 40),
                generations=params.get("generations", 80),
                crossover_rate=params.get("crossover_rate", 0.8),
                mutation_rate=params.get("mutation_rate", 0.15),
                tournament_size=params.get("tournament_size", 3),
                elitism=params.get("elitism", 2),
                competitor_strategy=competitor_strategy,
                seed=seed,
            )
        results.append(res)

    costs = [r.best_cost for r in results]
    best = min(results, key=lambda r: r.best_cost)
    best.mean_cost = statistics.mean(costs)
    best.std_cost = statistics.stdev(costs) if len(costs) > 1 else 0.0
    return best, results


if __name__ == "__main__":
    from scenario import load_scenario
    sc = load_scenario("../config")
    print(sc.summary())

    print("\n--- Simulated Annealing (3 runs) ---")
    best_sa, _ = run_multiple("sa", sc, num_runs=3, params={"iterations_per_temp": 20})
    print(best_sa)
    print(f"Mean={best_sa.mean_cost:.2f} Std={best_sa.std_cost:.2f}")

    print("\n--- Genetic Algorithm (3 runs) ---")
    best_ga, _ = run_multiple("ga", sc, num_runs=3, params={"generations": 30, "population_size": 20})
    print(best_ga)
    print(f"Mean={best_ga.mean_cost:.2f} Std={best_ga.std_cost:.2f}")
