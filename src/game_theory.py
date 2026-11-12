"""
Módulo D: Teoría de Juegos
Juego de dos jugadores (nosotros vs competidor) sobre estrategias de ruta/recursos.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import itertools
import numpy as np

from scenario import Scenario, Coord
from pathfinding import find_path_through_deliveries


# Estrategias del jugador (nosotros)
PLAYER_STRATEGIES = [
    "ruta_rapida",       # prioriza A* directo, ignora un poco al competidor
    "ruta_segura",       # evita zonas del competidor
    "ruta_equilibrada",  # balance costo/riesgo
]

# Estrategias del competidor (ya definidas en el escenario)
# "aggressive", "defensive", "balanced"


@dataclass
class GameResult:
    payoff_matrix: np.ndarray
    row_labels: List[str]
    col_labels: List[str]
    pure_nash: List[Tuple[int, int]]
    dominated_rows: List[int]
    dominated_cols: List[int]
    best_responses_row: Dict[int, List[int]]
    best_responses_col: Dict[int, List[int]]
    interpretation: str

    def __str__(self):
        s = "Matriz de pagos (filas=Nosotros, columnas=Competidor):\n"
        s += f"{'':20}" + "".join(f"{c:>14}" for c in self.col_labels) + "\n"
        for i, row in enumerate(self.payoff_matrix):
            s += f"{self.row_labels[i]:20}" + "".join(f"{v:14.2f}" for v in row) + "\n"
        s += f"\nEquilibrios de Nash puros: {self.pure_nash}\n"
        s += f"Filas dominadas: {self.dominated_rows}\n"
        s += f"Columnas dominadas: {self.dominated_cols}\n"
        return s


def compute_payoff(
    scenario: Scenario,
    player_strategy: str,
    competitor_strategy: str,
    delivery_order: Optional[List[Coord]] = None,
) -> float:
    """
    Calcula el pago (negativo del costo) para una combinación de estrategias.
    Un pago más alto es mejor para nosotros.
    """
    # Ajustar el escenario según estrategias
    sc = scenario.clone()

    # Estrategia del competidor afecta el costo de sus celdas
    sc.competitor_active = True

    # Orden de entregas (si no se da, usamos el orden original)
    order = delivery_order or sc.deliveries[:]

    # Ajuste según nuestra estrategia
    if player_strategy == "ruta_segura":
        # Intentamos penalizar más las celdas del competidor para evitarlas
        # (simulado aumentando artificialmente el costo en evaluación)
        original_increase = sc.competitor_cost_increase
        sc.competitor_cost_increase = original_increase * 2.5  # más aversión al riesgo
    elif player_strategy == "ruta_rapida":
        sc.competitor_cost_increase *= 0.4  # menos peso al competidor
    # "ruta_equilibrada" usa el valor por defecto

    res = find_path_through_deliveries(
        sc, order, algorithm="astar", competitor_strategy=competitor_strategy
    )
    if not res.success:
        return -10000.0  # penalización fuerte

    # El pago es -costo (queremos maximizar el pago = minimizar costo)
    # Añadimos un pequeño bonus/malus según alineación de estrategias
    bonus = 0.0
    if player_strategy == "ruta_segura" and competitor_strategy == "aggressive":
        bonus = 5.0  # nos preparamos bien
    elif player_strategy == "ruta_rapida" and competitor_strategy == "defensive":
        bonus = 3.0
    elif player_strategy == "ruta_equilibrada":
        bonus = 1.5

    return -res.cost + bonus


def build_payoff_matrix(
    scenario: Scenario,
    delivery_order: Optional[List[Coord]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    row_labels = PLAYER_STRATEGIES
    col_labels = scenario.competitor_strategies

    matrix = np.zeros((len(row_labels), len(col_labels)))
    for i, p_strat in enumerate(row_labels):
        for j, c_strat in enumerate(col_labels):
            matrix[i, j] = compute_payoff(scenario, p_strat, c_strat, delivery_order)
    return matrix, row_labels, col_labels


def find_dominated_strategies(matrix: np.ndarray) -> Tuple[List[int], List[int]]:
    """Encuentra estrategias estrictamente dominadas."""
    n_rows, n_cols = matrix.shape
    dominated_rows = []
    dominated_cols = []

    # Filas dominadas (para el maximizador)
    for i in range(n_rows):
        for k in range(n_rows):
            if i == k:
                continue
            if np.all(matrix[k, :] > matrix[i, :]):
                dominated_rows.append(i)
                break

    # Columnas dominadas (para el minimizador, recordando que él quiere minimizar nuestro pago)
    # Es decir, el competidor prefiere columnas que den pagos más bajos a nosotros.
    for j in range(n_cols):
        for m in range(n_cols):
            if j == m:
                continue
            # m domina a j si en todas las filas matrix[:, m] < matrix[:, j]
            if np.all(matrix[:, m] < matrix[:, j]):
                dominated_cols.append(j)
                break

    return dominated_rows, dominated_cols


def best_responses(matrix: np.ndarray) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Mejores respuestas para cada estrategia del oponente."""
    n_rows, n_cols = matrix.shape
    br_row = {}  # para cada columna j, las filas i que son BR
    br_col = {}  # para cada fila i, las columnas j que son BR (minimizan)

    for j in range(n_cols):
        max_val = np.max(matrix[:, j])
        br_row[j] = [i for i in range(n_rows) if abs(matrix[i, j] - max_val) < 1e-6]

    for i in range(n_rows):
        min_val = np.min(matrix[i, :])
        br_col[i] = [j for j in range(n_cols) if abs(matrix[i, j] - min_val) < 1e-6]

    return br_row, br_col


def find_pure_nash(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Encuentra equilibrios de Nash en estrategias puras."""
    br_row, br_col = best_responses(matrix)
    nash = []
    n_rows, n_cols = matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            if i in br_row.get(j, []) and j in br_col.get(i, []):
                nash.append((i, j))
    return nash


def analyze_game(
    scenario: Scenario,
    delivery_order: Optional[List[Coord]] = None,
) -> GameResult:
    matrix, row_labels, col_labels = build_payoff_matrix(scenario, delivery_order)
    dominated_rows, dominated_cols = find_dominated_strategies(matrix)
    br_row, br_col = best_responses(matrix)
    pure_nash = find_pure_nash(matrix)

    # Interpretación logística
    interp = []
    if pure_nash:
        for i, j in pure_nash:
            interp.append(
                f"Equilibrio ({row_labels[i]}, {col_labels[j]}): "
                f"pago={matrix[i,j]:.2f}. "
                f"Recomendación: adoptar estrategia '{row_labels[i]}' cuando el competidor juega '{col_labels[j]}'."
            )
    else:
        interp.append("No existen equilibrios de Nash en estrategias puras. Se recomienda considerar estrategias mixtas o robustas.")

    if dominated_rows:
        names = [row_labels[i] for i in dominated_rows]
        interp.append(f"Estrategias dominadas (evitar): {names}")

    interpretation = " ".join(interp)

    return GameResult(
        payoff_matrix=matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        pure_nash=pure_nash,
        dominated_rows=dominated_rows,
        dominated_cols=dominated_cols,
        best_responses_row=br_row,
        best_responses_col=br_col,
        interpretation=interpretation,
    )


if __name__ == "__main__":
    from scenario import load_scenario
    sc = load_scenario("../config")
    print(sc.summary())
    result = analyze_game(sc)
    print(result)
    print("\nInterpretación:")
    print(result.interpretation)
