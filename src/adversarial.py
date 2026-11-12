"""
Módulo E: Minimax y poda alfa-beta
Subproblema adversario por turnos: negociación/asignación de recursos
(acceso prioritario a celdas del competidor o reducción de costos).
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import time
import copy

from scenario import Scenario, Coord


@dataclass
class GameState:
    """Estado del juego de negociación de recursos."""
    turn: int                    # 0 = nosotros (max), 1 = competidor (min)
    our_resource: float          # recurso que tenemos (presupuesto de negociación)
    rival_resource: float
    shared_pool: float           # recurso compartido en disputa
    our_advantage: float         # ventaja acumulada (afecta costo final)
    rival_advantage: float
    depth: int = 0
    last_action: Optional[str] = None
    terminal: bool = False

    def is_terminal(self, max_depth: int) -> bool:
        return self.depth >= max_depth or self.shared_pool <= 0 or self.terminal

    def utility(self) -> float:
        """Función de utilidad desde nuestra perspectiva."""
        # Queremos maximizar nuestra ventaja relativa y el recurso restante
        return (self.our_advantage - self.rival_advantage) + 0.3 * self.our_resource - 0.1 * self.rival_resource


ACTIONS = [
    "claim_high",    # reclamar mucho del pool (agresivo)
    "claim_medium",  # reclamar cantidad media
    "claim_low",     # reclamar poco (conservador)
    "invest",        # invertir recurso propio para ganar ventaja
    "block",         # intentar bloquear al rival
]


def apply_action(state: GameState, action: str, is_maximizer: bool) -> GameState:
    """Aplica una acción y devuelve el nuevo estado."""
    new_state = copy.deepcopy(state)
    new_state.depth += 1
    new_state.last_action = action
    new_state.turn = 1 - state.turn

    pool = new_state.shared_pool

    if action == "claim_high":
        take = min(pool * 0.5, pool)
        if is_maximizer:
            new_state.our_resource += take
            new_state.our_advantage += 2.0
        else:
            new_state.rival_resource += take
            new_state.rival_advantage += 2.0
        new_state.shared_pool -= take

    elif action == "claim_medium":
        take = min(pool * 0.3, pool)
        if is_maximizer:
            new_state.our_resource += take
            new_state.our_advantage += 1.2
        else:
            new_state.rival_resource += take
            new_state.rival_advantage += 1.2
        new_state.shared_pool -= take

    elif action == "claim_low":
        take = min(pool * 0.15, pool)
        if is_maximizer:
            new_state.our_resource += take
            new_state.our_advantage += 0.5
        else:
            new_state.rival_resource += take
            new_state.rival_advantage += 0.5
        new_state.shared_pool -= take

    elif action == "invest":
        cost = 5.0
        if is_maximizer and new_state.our_resource >= cost:
            new_state.our_resource -= cost
            new_state.our_advantage += 3.5
        elif not is_maximizer and new_state.rival_resource >= cost:
            new_state.rival_resource -= cost
            new_state.rival_advantage += 3.5

    elif action == "block":
        # Reduce la ventaja del oponente
        if is_maximizer:
            new_state.rival_advantage = max(0, new_state.rival_advantage - 1.5)
            new_state.our_resource -= 2.0
        else:
            new_state.our_advantage = max(0, new_state.our_advantage - 1.5)
            new_state.rival_resource -= 2.0

    # Asegurar no negativos
    new_state.our_resource = max(0, new_state.our_resource)
    new_state.rival_resource = max(0, new_state.rival_resource)
    new_state.shared_pool = max(0, new_state.shared_pool)

    if new_state.shared_pool <= 0.5:
        new_state.terminal = True

    return new_state


@dataclass
class MinimaxResult:
    value: float
    best_action: Optional[str]
    nodes_visited: int
    nodes_pruned: int
    time_seconds: float
    algorithm: str
    principal_variation: List[str] = field(default_factory=list)

    def __str__(self):
        return (
            f"[{self.algorithm}] valor={self.value:.3f} | "
            f"acción={self.best_action} | nodos={self.nodes_visited} | "
            f"podados={self.nodes_pruned} | t={self.time_seconds*1000:.1f}ms"
        )


def minimax(
    state: GameState,
    depth: int,
    maximizing: bool,
    stats: Dict[str, int],
    path: List[str],
) -> Tuple[float, Optional[str], List[str]]:
    stats["visited"] += 1

    if state.is_terminal(depth) or depth == 0:
        return state.utility(), None, path[:]

    best_action = None
    best_path = path[:]

    if maximizing:
        max_eval = float("-inf")
        for action in ACTIONS:
            child = apply_action(state, action, is_maximizer=True)
            eval_score, _, child_path = minimax(
                child, depth - 1, False, stats, path + [action]
            )
            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action
                best_path = child_path
        return max_eval, best_action, best_path
    else:
        min_eval = float("inf")
        for action in ACTIONS:
            child = apply_action(state, action, is_maximizer=False)
            eval_score, _, child_path = minimax(
                child, depth - 1, True, stats, path + [action]
            )
            if eval_score < min_eval:
                min_eval = eval_score
                best_action = action
                best_path = child_path
        return min_eval, best_action, best_path


def alphabeta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    stats: Dict[str, int],
    path: List[str],
) -> Tuple[float, Optional[str], List[str]]:
    stats["visited"] += 1

    if state.is_terminal(depth) or depth == 0:
        return state.utility(), None, path[:]

    best_action = None
    best_path = path[:]

    if maximizing:
        max_eval = float("-inf")
        for action in ACTIONS:
            child = apply_action(state, action, is_maximizer=True)
            eval_score, _, child_path = alphabeta(
                child, depth - 1, alpha, beta, False, stats, path + [action]
            )
            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action
                best_path = child_path
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                stats["pruned"] += 1
                break
        return max_eval, best_action, best_path
    else:
        min_eval = float("inf")
        for action in ACTIONS:
            child = apply_action(state, action, is_maximizer=False)
            eval_score, _, child_path = alphabeta(
                child, depth - 1, alpha, beta, True, stats, path + [action]
            )
            if eval_score < min_eval:
                min_eval = eval_score
                best_action = action
                best_path = child_path
            beta = min(beta, eval_score)
            if beta <= alpha:
                stats["pruned"] += 1
                break
        return min_eval, best_action, best_path


def run_minimax(
    initial_state: Optional[GameState] = None,
    max_depth: int = 4,
    use_alpha_beta: bool = False,
) -> MinimaxResult:
    if initial_state is None:
        initial_state = GameState(
            turn=0,
            our_resource=30.0,
            rival_resource=30.0,
            shared_pool=40.0,
            our_advantage=0.0,
            rival_advantage=0.0,
            depth=0,
        )

    stats = {"visited": 0, "pruned": 0}
    t0 = time.perf_counter()

    if use_alpha_beta:
        value, action, pv = alphabeta(
            initial_state, max_depth, float("-inf"), float("inf"),
            True, stats, []
        )
        algo = "Minimax+AlphaBeta"
    else:
        value, action, pv = minimax(
            initial_state, max_depth, True, stats, []
        )
        algo = "Minimax"

    elapsed = time.perf_counter() - t0
    return MinimaxResult(
        value=value,
        best_action=action,
        nodes_visited=stats["visited"],
        nodes_pruned=stats["pruned"],
        time_seconds=elapsed,
        algorithm=algo,
        principal_variation=pv,
    )


def compare_minimax_alphabeta(max_depth: int = 4) -> Dict[str, Any]:
    """Verifica igualdad de valor y compara nodos."""
    state = GameState(
        turn=0,
        our_resource=30.0,
        rival_resource=30.0,
        shared_pool=40.0,
        our_advantage=0.0,
        rival_advantage=0.0,
    )

    res_mm = run_minimax(state, max_depth=max_depth, use_alpha_beta=False)
    res_ab = run_minimax(state, max_depth=max_depth, use_alpha_beta=True)

    equal_value = abs(res_mm.value - res_ab.value) < 1e-6
    return {
        "minimax": res_mm,
        "alphabeta": res_ab,
        "values_equal": equal_value,
        "nodes_saved": res_mm.nodes_visited - res_ab.nodes_visited,
        "pruning_ratio": res_ab.nodes_pruned / max(1, res_ab.nodes_visited),
    }


if __name__ == "__main__":
    print("=== Comparación Minimax vs Alpha-Beta ===")
    for d in [3, 4]:
        print(f"\nProfundidad {d}:")
        cmp = compare_minimax_alphabeta(max_depth=d)
        print(cmp["minimax"])
        print(cmp["alphabeta"])
        print(f"Valores iguales: {cmp['values_equal']}")
        print(f"Nodos ahorrados: {cmp['nodes_saved']}")
        print(f"Ratio de poda: {cmp['pruning_ratio']:.2%}")
        print(f"PV: {cmp['alphabeta'].principal_variation}")
