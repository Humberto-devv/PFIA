"""Pruebas unitarias del escenario y pathfinding básico."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from scenario import Scenario, load_scenario
from pathfinding import greedy_best_first, astar, find_path_through_deliveries


@pytest.fixture
def scenario():
    return load_scenario(Path(__file__).parent.parent / "config")


def test_scenario_loads(scenario):
    assert scenario.grid_size == 12
    assert len(scenario.deliveries) >= 3
    assert scenario.start == (0, 0)
    assert scenario.goal == (11, 11)


def test_start_valid(scenario):
    assert scenario.is_valid(scenario.start)
    assert scenario.is_valid(scenario.goal)


def test_obstacles_invalid(scenario):
    for obs in list(scenario.obstacles)[:3]:
        assert not scenario.is_valid(obs)


def test_neighbors(scenario):
    neigh = scenario.neighbors(scenario.start)
    assert len(neigh) > 0
    for n in neigh:
        assert scenario.is_valid(n)


def test_astar_finds_path(scenario):
    res = astar(scenario)
    assert res.success
    assert res.path[0] == scenario.start
    assert res.path[-1] == scenario.goal
    assert res.cost > 0
    assert res.nodes_expanded > 0


def test_greedy_finds_path(scenario):
    res = greedy_best_first(scenario)
    assert res.success
    assert res.path[0] == scenario.start
    assert res.path[-1] == scenario.goal


def test_path_through_deliveries(scenario):
    res = find_path_through_deliveries(scenario, scenario.deliveries, algorithm="astar")
    assert res.success
    # Debe visitar todas las entregas
    path_set = set(res.path)
    for d in scenario.deliveries:
        assert d in path_set


def test_invalid_map_raises():
    with pytest.raises(Exception):
        Scenario(
            grid_size=5,
            start=(0, 0),
            goal=(4, 4),
            deliveries=[(1, 1)],  # menos de 3
            obstacles=set(),
            terrain_multipliers={},
            fuel_capacity=50,
            fuel_per_move=1,
            max_payload=2,
            delivery_fuel_bonus=0,
            competitor_cells=set(),
            competitor_cost_increase=1,
            competitor_availability_prob=0.5,
            competitor_strategies=["a"],
            allowed_moves=[(0, 1), (1, 0)],
            seed=1,
        )


def test_competitor_affects_cost(scenario):
    scenario.set_competitor_active(True)
    # Tomar una celda del competidor si existe
    if scenario.competitor_cells:
        cell = next(iter(scenario.competitor_cells))
        # Buscar un vecino válido para calcular costo de entrada
        for n in scenario.neighbors(cell):
            cost_with = scenario.move_cost(n, cell, competitor_strategy="aggressive")
            scenario.set_competitor_active(False)
            cost_without = scenario.move_cost(n, cell)
            scenario.set_competitor_active(True)
            assert cost_with >= cost_without
            break
