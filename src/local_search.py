"""
Módulo B: Búsqueda local
Hill Climbing sobre el orden de visita de las entregas.
"""

from __future__ import annotations
import random
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from copy import deepcopy

from scenario import Scenario, Coord
from pathfinding import find_path_through_deliveries, SearchResult


@dataclass
class HCResult:
    best_order: List[Coord]
    best_cost: float
    best_path: List[Coord]
    iterations: int
    nodes_evaluated: int
    time_seconds: float
    history: List[float] = field(default_factory=list)
    local_optima_reached: bool = False
    plateau_detected: bool = False
    initial_cost: float = float("inf")
    improvements: int = 0

    def __str__(self):
        return (
            f"HillClimbing | best={self.best_cost:.2f} | "
            f"iters={self.iterations} | evals={self.nodes_evaluated} | "
            f"mejoras={self.improvements} | plateau={self.plateau_detected}"
        )


def evaluate_order(
    scenario: Scenario,
    order: List[Coord],
    competitor_strategy: Optional[str] = None,
) -> Tuple[float, List[Coord]]:
    """Evalúa un orden de entregas devolviendo (costo, path)."""
    res = find_path_through_deliveries(
        scenario,
        order,
        algorithm="astar",
        competitor_strategy=competitor_strategy,
    )
    if not res.success:
        return float("inf"), []
    return res.cost, res.path


def swap_neighborhood(order: List[Coord]) -> List[List[Coord]]:
    """Genera vecinos por intercambio de dos posiciones."""
    neighbors = []
    n = len(order)
    for i in range(n):
        for j in range(i + 1, n):
            new_order = order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            neighbors.append(new_order)
    return neighbors


def hill_climbing(
    scenario: Scenario,
    initial_order: Optional[List[Coord]] = None,
    max_iterations: int = 200,
    competitor_strategy: Optional[str] = None,
    seed: Optional[int] = None,
) -> HCResult:
    """
    Hill Climbing con vecindad de intercambios.
    Acepta solo mejoras estrictas (greedy).
    """
    if seed is not None:
        random.seed(seed)

    if initial_order is None:
        order = scenario.deliveries[:]
        random.shuffle(order)
    else:
        order = initial_order[:]

    current_cost, current_path = evaluate_order(scenario, order, competitor_strategy)
    best_order = order[:]
    best_cost = current_cost
    best_path = current_path[:]
    history = [current_cost]
    improvements = 0
    plateau = False
    local_optima = False
    nodes_evaluated = 1
    t0 = time.perf_counter()

    for it in range(max_iterations):
        neighbors = swap_neighborhood(order)
        random.shuffle(neighbors)  # para explorar distinto orden

        improved = False
        best_neighbor_cost = current_cost
        best_neighbor = order
        best_neighbor_path = current_path

        for neigh in neighbors:
            cost, path = evaluate_order(scenario, neigh, competitor_strategy)
            nodes_evaluated += 1
            if cost < best_neighbor_cost:
                best_neighbor_cost = cost
                best_neighbor = neigh
                best_neighbor_path = path
                improved = True

        if improved and best_neighbor_cost < current_cost:
            order = best_neighbor
            current_cost = best_neighbor_cost
            current_path = best_neighbor_path
            improvements += 1
            if current_cost < best_cost:
                best_cost = current_cost
                best_order = order[:]
                best_path = current_path[:]
        else:
            # No hay mejora estricta -> óptimo local o meseta
            local_optima = True
            # Detectar meseta: si todos los vecinos tienen el mismo costo
            costs = []
            for neigh in neighbors[: min(10, len(neighbors))]:
                c, _ = evaluate_order(scenario, neigh, competitor_strategy)
                costs.append(c)
            if costs and all(abs(c - current_cost) < 1e-6 for c in costs):
                plateau = True
            break

        history.append(current_cost)

    elapsed = time.perf_counter() - t0
    return HCResult(
        best_order=best_order,
        best_cost=best_cost,
        best_path=best_path,
        iterations=it + 1 if 'it' in dir() else 0,
        nodes_evaluated=nodes_evaluated,
        time_seconds=elapsed,
        history=history,
        local_optima_reached=local_optima,
        plateau_detected=plateau,
        initial_cost=history[0] if history else float("inf"),
        improvements=improvements,
    )


def multi_start_hill_climbing(
    scenario: Scenario,
    num_starts: int = 15,
    max_iterations: int = 200,
    competitor_strategy: Optional[str] = None,
    seeds: Optional[List[int]] = None,
) -> Tuple[HCResult, List[HCResult]]:
    """Ejecuta Hill Climbing desde múltiples puntos iniciales."""
    results = []
    best_overall: Optional[HCResult] = None

    for i in range(num_starts):
        seed = seeds[i] if seeds and i < len(seeds) else None
        # Generar orden inicial aleatorio
        order = scenario.deliveries[:]
        if seed is not None:
            random.seed(seed)
        random.shuffle(order)

        res = hill_climbing(
            scenario,
            initial_order=order,
            max_iterations=max_iterations,
            competitor_strategy=competitor_strategy,
            seed=seed,
        )
        results.append(res)
        if best_overall is None or res.best_cost < best_overall.best_cost:
            best_overall = res

    return best_overall, results  # type: ignore


if __name__ == "__main__":
    from scenario import load_scenario
    sc = load_scenario("../config")
    print(sc.summary())
    best, all_res = multi_start_hill_climbing(sc, num_starts=5, max_iterations=50)
    print("Mejor global:", best)
    print("Costos de las corridas:", [r.best_cost for r in all_res])
