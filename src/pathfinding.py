"""
Módulo A: Búsqueda heurística
Implementa Greedy Best-First Search y A*.
"""

from __future__ import annotations
import heapq
import time
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field

from scenario import Scenario, Coord


@dataclass
class SearchResult:
    path: List[Coord]
    cost: float
    nodes_expanded: int
    time_seconds: float
    success: bool
    algorithm: str
    heuristic_name: str
    g_values: Dict[Coord, float] = field(default_factory=dict)
    f_values: Dict[Coord, float] = field(default_factory=dict)

    def __str__(self):
        status = "OK" if self.success else "FAIL"
        return (
            f"[{self.algorithm}] {status} | costo={self.cost:.2f} | "
            f"nodos={self.nodes_expanded} | tiempo={self.time_seconds*1000:.1f}ms | "
            f"largo={len(self.path)}"
        )


def manhattan(a: Coord, b: Coord) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Coord, b: Coord) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
}


def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def greedy_best_first(
    scenario: Scenario,
    start: Optional[Coord] = None,
    goal: Optional[Coord] = None,
    heuristic_name: str = "manhattan",
    competitor_strategy: Optional[str] = None,
) -> SearchResult:
    """Greedy Best-First Search: prioriza solo h(n)."""
    start = start or scenario.start
    goal = goal or scenario.goal
    h_func = HEURISTICS.get(heuristic_name, manhattan)

    t0 = time.perf_counter()
    open_set = []  # (h, counter, node)
    counter = 0
    heapq.heappush(open_set, (h_func(start, goal), counter, start))
    came_from: Dict[Coord, Coord] = {}
    visited = set()
    nodes_expanded = 0
    g_values: Dict[Coord, float] = {start: 0.0}

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            cost = scenario.path_cost(path, competitor_strategy)
            elapsed = time.perf_counter() - t0
            return SearchResult(
                path=path,
                cost=cost,
                nodes_expanded=nodes_expanded,
                time_seconds=elapsed,
                success=True,
                algorithm="Greedy Best-First",
                heuristic_name=heuristic_name,
                g_values=g_values,
            )

        for neigh in scenario.neighbors(current):
            if neigh in visited:
                continue
            # Solo usamos h(n)
            h = h_func(neigh, goal)
            counter += 1
            heapq.heappush(open_set, (h, counter, neigh))
            if neigh not in came_from:
                came_from[neigh] = current
                g_values[neigh] = g_values.get(current, 0) + scenario.move_cost(
                    current, neigh, competitor_strategy
                )

    elapsed = time.perf_counter() - t0
    return SearchResult(
        path=[],
        cost=float("inf"),
        nodes_expanded=nodes_expanded,
        time_seconds=elapsed,
        success=False,
        algorithm="Greedy Best-First",
        heuristic_name=heuristic_name,
    )


def astar(
    scenario: Scenario,
    start: Optional[Coord] = None,
    goal: Optional[Coord] = None,
    heuristic_name: str = "manhattan",
    competitor_strategy: Optional[str] = None,
) -> SearchResult:
    """A* Search: f(n) = g(n) + h(n)."""
    start = start or scenario.start
    goal = goal or scenario.goal
    h_func = HEURISTICS.get(heuristic_name, manhattan)

    t0 = time.perf_counter()
    open_set = []  # (f, counter, node)
    counter = 0
    g_score: Dict[Coord, float] = {start: 0.0}
    f_score: Dict[Coord, float] = {start: h_func(start, goal)}
    heapq.heappush(open_set, (f_score[start], counter, start))
    came_from: Dict[Coord, Coord] = {}
    closed = set()
    nodes_expanded = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            cost = g_score[current]
            elapsed = time.perf_counter() - t0
            return SearchResult(
                path=path,
                cost=cost,
                nodes_expanded=nodes_expanded,
                time_seconds=elapsed,
                success=True,
                algorithm="A*",
                heuristic_name=heuristic_name,
                g_values=g_score,
                f_values=f_score,
            )

        for neigh in scenario.neighbors(current):
            if neigh in closed:
                continue
            tentative_g = g_score[current] + scenario.move_cost(
                current, neigh, competitor_strategy
            )
            if tentative_g >= g_score.get(neigh, float("inf")):
                continue
            came_from[neigh] = current
            g_score[neigh] = tentative_g
            f = tentative_g + h_func(neigh, goal)
            f_score[neigh] = f
            counter += 1
            heapq.heappush(open_set, (f, counter, neigh))

    elapsed = time.perf_counter() - t0
    return SearchResult(
        path=[],
        cost=float("inf"),
        nodes_expanded=nodes_expanded,
        time_seconds=elapsed,
        success=False,
        algorithm="A*",
        heuristic_name=heuristic_name,
    )


def find_path_through_deliveries(
    scenario: Scenario,
    order: List[Coord],
    algorithm: str = "astar",
    heuristic_name: str = "manhattan",
    competitor_strategy: Optional[str] = None,
) -> SearchResult:
    """
    Encuentra una ruta completa: start -> d1 -> d2 -> ... -> goal
    concatenando segmentos.
    """
    points = [scenario.start] + list(order) + [scenario.goal]
    full_path: List[Coord] = []
    total_cost = 0.0
    total_nodes = 0
    total_time = 0.0
    search_fn = astar if algorithm == "astar" else greedy_best_first

    for i in range(len(points) - 1):
        seg = search_fn(
            scenario,
            start=points[i],
            goal=points[i + 1],
            heuristic_name=heuristic_name,
            competitor_strategy=competitor_strategy,
        )
        total_nodes += seg.nodes_expanded
        total_time += seg.time_seconds
        if not seg.success:
            return SearchResult(
                path=[],
                cost=float("inf"),
                nodes_expanded=total_nodes,
                time_seconds=total_time,
                success=False,
                algorithm=f"{algorithm}+segments",
                heuristic_name=heuristic_name,
            )
        # Evitar duplicar el punto de conexión
        if full_path:
            full_path.extend(seg.path[1:])
        else:
            full_path.extend(seg.path)
        total_cost += seg.cost

    return SearchResult(
        path=full_path,
        cost=total_cost,
        nodes_expanded=total_nodes,
        time_seconds=total_time,
        success=True,
        algorithm=f"{algorithm}+segments",
        heuristic_name=heuristic_name,
    )


if __name__ == "__main__":
    from scenario import load_scenario
    sc = load_scenario("../config")
    print(sc.summary())
    r1 = greedy_best_first(sc)
    print(r1)
    r2 = astar(sc)
    print(r2)
    print("Camino A* (primeros 15):", r2.path[:15], "...")
