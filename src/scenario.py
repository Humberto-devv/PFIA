"""
Módulo de escenario logístico compartido.
Representa la cuadrícula 12x12, costos, recursos y competidor.
"""

from __future__ import annotations
import json
import csv
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import copy


Coord = Tuple[int, int]


@dataclass
class Scenario:
    grid_size: int
    start: Coord
    goal: Coord
    deliveries: List[Coord]
    obstacles: Set[Coord]
    terrain_multipliers: Dict[Coord, float]
    fuel_capacity: float
    fuel_per_move: float
    max_payload: int
    delivery_fuel_bonus: float
    competitor_cells: Set[Coord]
    competitor_cost_increase: float
    competitor_availability_prob: float
    competitor_strategies: List[str]
    allowed_moves: List[Coord]
    seed: int
    competitor_active: bool = True  # controlado por el competidor

    def __post_init__(self):
        self._validate()

    def _validate(self):
        n = self.grid_size
        for name, pos in [("start", self.start), ("goal", self.goal)]:
            if not (0 <= pos[0] < n and 0 <= pos[1] < n):
                raise ValueError(f"{name} fuera de la cuadrícula: {pos}")
        for d in self.deliveries:
            if not (0 <= d[0] < n and 0 <= d[1] < n):
                raise ValueError(f"Entrega fuera de la cuadrícula: {d}")
            if d in self.obstacles:
                raise ValueError(f"Entrega sobre obstáculo: {d}")
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("Inicio o meta sobre obstáculo")
        if len(self.deliveries) < 3:
            raise ValueError("Se requieren al menos 3 puntos de entrega")

    @classmethod
    def from_json(cls, path: str | Path) -> "Scenario":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        terrain = {}
        for zone in data.get("terrain_costs", {}).get("high_cost_zones", []):
            mult = zone["multiplier"]
            for cell in zone["cells"]:
                terrain[tuple(cell)] = mult

        default_cost = data.get("terrain_costs", {}).get("default", 1.0)
        # Guardamos solo los multiplicadores especiales; el default se aplica después

        moves = data.get("moves", {}).get("allowed", [[0, 1], [1, 0], [0, -1], [-1, 0]])
        moves = [tuple(m) for m in moves]

        rc = data.get("resource_constraints", {})
        comp = data.get("competitor", {})

        return cls(
            grid_size=data["grid_size"],
            start=tuple(data["start"]),
            goal=tuple(data["goal"]),
            deliveries=[tuple(d) for d in data["deliveries"]],
            obstacles={tuple(o) for o in data.get("obstacles", [])},
            terrain_multipliers=terrain,
            fuel_capacity=rc.get("fuel_capacity", 100.0),
            fuel_per_move=rc.get("fuel_per_move", 1.0),
            max_payload=rc.get("max_payload", 3),
            delivery_fuel_bonus=rc.get("delivery_fuel_bonus", 0.0),
            competitor_cells={tuple(c) for c in comp.get("affected_cells", [])},
            competitor_cost_increase=comp.get("cost_increase", 2.0),
            competitor_availability_prob=comp.get("availability_prob", 0.5),
            competitor_strategies=comp.get("strategies", ["aggressive", "defensive", "balanced"]),
            allowed_moves=moves,
            seed=data.get("seed", 42),
        )

    def is_valid(self, pos: Coord) -> bool:
        x, y = pos
        return (
            0 <= x < self.grid_size
            and 0 <= y < self.grid_size
            and pos not in self.obstacles
        )

    def neighbors(self, pos: Coord) -> List[Coord]:
        res = []
        for dx, dy in self.allowed_moves:
            nxt = (pos[0] + dx, pos[1] + dy)
            if self.is_valid(nxt):
                res.append(nxt)
        return res

    def move_cost(self, from_pos: Coord, to_pos: Coord, competitor_strategy: Optional[str] = None) -> float:
        """Costo de moverse de from_pos a to_pos, considerando terreno y competidor."""
        if not self.is_valid(to_pos):
            return float("inf")
        base = self.terrain_multipliers.get(to_pos, 1.0)
        cost = base * self.fuel_per_move

        # Efecto del competidor
        if self.competitor_active and to_pos in self.competitor_cells:
            # La estrategia del competidor modula el incremento
            factor = 1.0
            if competitor_strategy == "aggressive":
                factor = 1.5
            elif competitor_strategy == "defensive":
                factor = 0.7
            elif competitor_strategy == "balanced":
                factor = 1.0
            cost += self.competitor_cost_increase * factor
        return cost

    def path_cost(self, path: List[Coord], competitor_strategy: Optional[str] = None) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            c = self.move_cost(path[i], path[i + 1], competitor_strategy)
            if c == float("inf"):
                return float("inf")
            total += c
        return total

    def fuel_consumed(self, path: List[Coord], competitor_strategy: Optional[str] = None) -> float:
        return self.path_cost(path, competitor_strategy)

    def is_feasible_path(self, path: List[Coord], competitor_strategy: Optional[str] = None) -> bool:
        if not path or path[0] != self.start:
            return False
        fuel = self.fuel_capacity
        for i in range(len(path) - 1):
            cost = self.move_cost(path[i], path[i + 1], competitor_strategy)
            if cost == float("inf"):
                return False
            fuel -= cost
            if fuel < 0:
                return False
            # bonus al entregar
            if path[i + 1] in self.deliveries:
                fuel = min(self.fuel_capacity, fuel - self.delivery_fuel_bonus)
        return True

    def set_competitor_active(self, active: bool):
        self.competitor_active = active

    def clone(self) -> "Scenario":
        return copy.deepcopy(self)

    def summary(self) -> str:
        return (
            f"Grid {self.grid_size}x{self.grid_size} | "
            f"Start={self.start} Goal={self.goal} | "
            f"Deliveries={len(self.deliveries)} | "
            f"Obstacles={len(self.obstacles)} | "
            f"Seed={self.seed}"
        )


def load_scenario(config_dir: str | Path = "config") -> Scenario:
    path = Path(config_dir) / "map.json"
    return Scenario.from_json(path)


if __name__ == "__main__":
    sc = load_scenario("../config")
    print(sc.summary())
    print("Vecinos de start:", sc.neighbors(sc.start))
    print("Costo ejemplo:", sc.move_cost(sc.start, (0, 1)))
