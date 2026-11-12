"""
Sistema inteligente de planificación, optimización y toma de decisiones adversarias.
Punto de entrada principal e integración de todos los módulos.
"""

from __future__ import annotations
import json
import csv
import time
import random
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent))

from scenario import Scenario, load_scenario
from pathfinding import greedy_best_first, astar, find_path_through_deliveries, SearchResult
from local_search import multi_start_hill_climbing, HCResult
from metaheuristics import run_multiple, MetaResult
from game_theory import analyze_game, GameResult
from adversarial import compare_minimax_alphabeta, run_minimax


ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_params() -> Dict:
    with open(CONFIG_DIR / "params.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Guardado: {path}")


def experiment_heuristic_search(scenario: Scenario, params: Dict) -> List[Dict]:
    print("\n" + "="*60)
    print("A. BÚSQUEDA HEURÍSTICA (Greedy Best-First vs A*)")
    print("="*60)

    results = []
    # Tres escenarios: sin competidor, con competidor balanced, con aggressive
    scenarios_cfg = [
        ("sin_competidor", None, False),
        ("competidor_balanced", "balanced", True),
        ("competidor_aggressive", "aggressive", True),
    ]

    for name, strat, active in scenarios_cfg:
        sc = scenario.clone()
        sc.set_competitor_active(active)
        print(f"\n--- Escenario: {name} ---")

        for algo_name, algo_fn in [("Greedy", greedy_best_first), ("A*", astar)]:
            for h_name in ["manhattan", "euclidean"]:
                res = algo_fn(sc, heuristic_name=h_name, competitor_strategy=strat)
                print(f"  {res}")
                results.append({
                    "escenario": name,
                    "algoritmo": res.algorithm,
                    "heuristica": h_name,
                    "exito": res.success,
                    "costo": res.cost if res.success else None,
                    "nodos_expandidos": res.nodes_expanded,
                    "tiempo_ms": res.time_seconds * 1000,
                    "largo_ruta": len(res.path),
                    "competidor_strategy": strat or "none",
                })

        # También ruta completa a través de entregas (orden original)
        for algo in ["astar", "greedy"]:
            res = find_path_through_deliveries(
                sc, sc.deliveries, algorithm=algo, competitor_strategy=strat
            )
            print(f"  Ruta completa ({algo}): {res}")
            results.append({
                "escenario": name,
                "algoritmo": f"{algo}+entregas",
                "heuristica": "manhattan",
                "exito": res.success,
                "costo": res.cost if res.success else None,
                "nodos_expandidos": res.nodes_expanded,
                "tiempo_ms": res.time_seconds * 1000,
                "largo_ruta": len(res.path),
                "competidor_strategy": strat or "none",
            })

    save_csv(RESULTS_DIR / "heuristic_search.csv", results)
    return results


def experiment_local_search(scenario: Scenario, params: Dict) -> Dict:
    print("\n" + "="*60)
    print("B. BÚSQUEDA LOCAL (Hill Climbing multi-start)")
    print("="*60)

    hc_params = params.get("hill_climbing", {})
    num_starts = hc_params.get("num_initial_solutions", 15)
    max_iter = hc_params.get("max_iterations", 200)
    seeds = params.get("experiments", {}).get("random_seeds", list(range(num_starts)))

    best, all_res = multi_start_hill_climbing(
        scenario,
        num_starts=num_starts,
        max_iterations=max_iter,
        seeds=seeds[:num_starts],
    )

    print(f"Mejor resultado global: {best}")
    costs = [r.best_cost for r in all_res]
    print(f"Costos de {len(all_res)} arranques: min={min(costs):.2f} "
          f"mean={statistics.mean(costs):.2f} std={statistics.stdev(costs) if len(costs)>1 else 0:.2f}")
    plateaus = sum(1 for r in all_res if r.plateau_detected)
    local_opt = sum(1 for r in all_res if r.local_optima_reached)
    print(f"Óptimos locales detectados: {local_opt}/{len(all_res)}")
    print(f"Mesetas detectadas: {plateaus}/{len(all_res)}")

    rows = []
    for i, r in enumerate(all_res):
        rows.append({
            "run": i,
            "best_cost": r.best_cost,
            "initial_cost": r.initial_cost,
            "improvements": r.improvements,
            "iterations": r.iterations,
            "evaluations": r.nodes_evaluated,
            "time_s": r.time_seconds,
            "local_optima": r.local_optima_reached,
            "plateau": r.plateau_detected,
            "order": str(r.best_order),
        })
    save_csv(RESULTS_DIR / "hill_climbing.csv", rows)

    return {"best": best, "all": all_res, "stats": {
        "min": min(costs), "mean": statistics.mean(costs),
        "std": statistics.stdev(costs) if len(costs) > 1 else 0.0,
        "local_optima": local_opt, "plateaus": plateaus,
    }}


def experiment_metaheuristics(scenario: Scenario, params: Dict) -> Dict:
    print("\n" + "="*60)
    print("C. METAHEURÍSTICAS (SA y GA)")
    print("="*60)

    seeds = params.get("experiments", {}).get("random_seeds", list(range(30)))
    sa_params = params.get("simulated_annealing", {})
    ga_params = params.get("genetic_algorithm", {})
    num_runs = max(sa_params.get("num_runs", 20), ga_params.get("num_runs", 20))

    print(f"\n--- Simulated Annealing ({num_runs} ejecuciones) ---")
    best_sa, all_sa = run_multiple(
        "sa", scenario, num_runs=num_runs, params=sa_params, seeds=seeds[:num_runs]
    )
    print(best_sa)
    print(f"  Mean={best_sa.mean_cost:.2f}  Std={best_sa.std_cost:.2f}")

    print(f"\n--- Genetic Algorithm ({num_runs} ejecuciones) ---")
    best_ga, all_ga = run_multiple(
        "ga", scenario, num_runs=num_runs, params=ga_params, seeds=seeds[:num_runs]
    )
    print(best_ga)
    print(f"  Mean={best_ga.mean_cost:.2f}  Std={best_ga.std_cost:.2f}")

    rows = []
    for method, results in [("SA", all_sa), ("GA", all_ga)]:
        for i, r in enumerate(results):
            rows.append({
                "method": method,
                "run": i,
                "best_cost": r.best_cost,
                "evaluations": r.evaluations,
                "time_s": r.time_seconds,
                "order": str(r.best_order),
            })
    save_csv(RESULTS_DIR / "metaheuristics.csv", rows)

    # Resumen comparativo
    summary = {
        "SA": {
            "best": best_sa.best_cost,
            "mean": best_sa.mean_cost,
            "std": best_sa.std_cost,
            "avg_evals": statistics.mean([r.evaluations for r in all_sa]),
            "avg_time": statistics.mean([r.time_seconds for r in all_sa]),
        },
        "GA": {
            "best": best_ga.best_cost,
            "mean": best_ga.mean_cost,
            "std": best_ga.std_cost,
            "avg_evals": statistics.mean([r.evaluations for r in all_ga]),
            "avg_time": statistics.mean([r.time_seconds for r in all_ga]),
        },
    }
    print("\nResumen comparativo SA vs GA:")
    for m, s in summary.items():
        print(f"  {m}: best={s['best']:.2f} mean={s['mean']:.2f}±{s['std']:.2f} "
              f"evals={s['avg_evals']:.0f} time={s['avg_time']:.2f}s")

    return {"sa": (best_sa, all_sa), "ga": (best_ga, all_ga), "summary": summary}


def experiment_game_theory(scenario: Scenario, best_order: Optional[List] = None) -> GameResult:
    print("\n" + "="*60)
    print("D. TEORÍA DE JUEGOS")
    print("="*60)

    result = analyze_game(scenario, delivery_order=best_order)
    print(result)
    print("\nInterpretación logística:")
    print(result.interpretation)

    # Guardar matriz
    rows = []
    for i, rlabel in enumerate(result.row_labels):
        row = {"estrategia_jugador": rlabel}
        for j, clabel in enumerate(result.col_labels):
            row[clabel] = result.payoff_matrix[i, j]
        rows.append(row)
    save_csv(RESULTS_DIR / "game_theory_matrix.csv", rows)

    return result


def experiment_adversarial(params: Dict) -> Dict:
    print("\n" + "="*60)
    print("E. MINIMAX Y PODA ALFA-BETA")
    print("="*60)

    max_depth = params.get("minimax", {}).get("max_depth", 4)
    comparison = compare_minimax_alphabeta(max_depth=max_depth)

    print(comparison["minimax"])
    print(comparison["alphabeta"])
    print(f"Valores iguales: {comparison['values_equal']}")
    print(f"Nodos ahorrados por poda: {comparison['nodes_saved']}")
    print(f"Ratio de poda: {comparison['pruning_ratio']:.2%}")
    print(f"Variación principal: {comparison['alphabeta'].principal_variation}")

    rows = [
        {
            "algoritmo": "Minimax",
            "profundidad": max_depth,
            "valor": comparison["minimax"].value,
            "nodos_visitados": comparison["minimax"].nodes_visited,
            "nodos_podados": 0,
            "tiempo_ms": comparison["minimax"].time_seconds * 1000,
            "mejor_accion": comparison["minimax"].best_action,
        },
        {
            "algoritmo": "AlphaBeta",
            "profundidad": max_depth,
            "valor": comparison["alphabeta"].value,
            "nodos_visitados": comparison["alphabeta"].nodes_visited,
            "nodos_podados": comparison["alphabeta"].nodes_pruned,
            "tiempo_ms": comparison["alphabeta"].time_seconds * 1000,
            "mejor_accion": comparison["alphabeta"].best_action,
        },
    ]
    save_csv(RESULTS_DIR / "minimax_comparison.csv", rows)
    return comparison


def integrate_and_plan(
    scenario: Scenario,
    best_order: List,
    game_result: GameResult,
    adversarial_action: str,
) -> Dict:
    """
    Integra los resultados de optimización y análisis estratégico
    para producir un plan final.
    """
    print("\n" + "="*60)
    print("INTEGRACIÓN Y PLAN FINAL")
    print("="*60)

    # Elegir estrategia de juego según Nash o mejor respuesta robusta
    if game_result.pure_nash:
        i, j = game_result.pure_nash[0]
        chosen_player_strat = game_result.row_labels[i]
        expected_rival_strat = game_result.col_labels[j]
    else:
        # Elegir la estrategia con mejor pago mínimo (maximin)
        matrix = game_result.payoff_matrix
        row_mins = matrix.min(axis=1)
        best_i = int(row_mins.argmax())
        chosen_player_strat = game_result.row_labels[best_i]
        expected_rival_strat = "balanced"

    print(f"Estrategia elegida (juego): {chosen_player_strat}")
    print(f"Estrategia esperada del rival: {expected_rival_strat}")
    print(f"Acción de negociación (minimax): {adversarial_action}")

    # Calcular ruta final con el mejor orden y la estrategia elegida
    sc = scenario.clone()
    sc.set_competitor_active(True)
    final_path_res = find_path_through_deliveries(
        sc, best_order, algorithm="astar", competitor_strategy=expected_rival_strat
    )

    plan = {
        "timestamp": datetime.now().isoformat(),
        "order_entregas": [list(c) for c in best_order],
        "estrategia_jugador": chosen_player_strat,
        "estrategia_rival_esperada": expected_rival_strat,
        "accion_negociacion": adversarial_action,
        "costo_estimado": final_path_res.cost if final_path_res.success else None,
        "ruta_valida": final_path_res.success,
        "largo_ruta": len(final_path_res.path),
        "nodos_expandidos_ruta": final_path_res.nodes_expanded,
        "fuel_capacity": scenario.fuel_capacity,
        "interpretacion_juego": game_result.interpretation,
    }

    print(f"\nPlan final:")
    print(f"  Orden de entregas: {best_order}")
    print(f"  Costo estimado: {plan['costo_estimado']}")
    print(f"  Ruta válida: {plan['ruta_valida']}")
    print(f"  Largo de ruta: {plan['largo_ruta']}")

    with open(RESULTS_DIR / "plan_final.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"  → Guardado: {RESULTS_DIR / 'plan_final.json'}")

    return plan


def main():
    print("="*60)
    print("SISTEMA INTELIGENTE DE PLANIFICACIÓN LOGÍSTICA")
    print("Búsqueda heurística + local + metaheurísticas + juegos + minimax")
    print("="*60)

    # Cargar escenario y parámetros (reproducibilidad)
    scenario = load_scenario(CONFIG_DIR)
    params = load_params()
    random.seed(scenario.seed)

    print(f"\nEscenario cargado: {scenario.summary()}")
    print(f"Entregas: {scenario.deliveries}")
    print(f"Obstáculos: {len(scenario.obstacles)}")
    print(f"Semilla global: {scenario.seed}")

    # A. Búsqueda heurística
    heur_results = experiment_heuristic_search(scenario, params)

    # B. Búsqueda local
    hc_data = experiment_local_search(scenario, params)
    best_order_hc = hc_data["best"].best_order

    # C. Metaheurísticas
    meta_data = experiment_metaheuristics(scenario, params)
    # Tomar el mejor orden global entre SA y GA
    best_sa = meta_data["sa"][0]
    best_ga = meta_data["ga"][0]
    if best_sa.best_cost <= best_ga.best_cost:
        best_order = best_sa.best_order
        best_meta_cost = best_sa.best_cost
        best_meta_name = "SA"
    else:
        best_order = best_ga.best_order
        best_meta_cost = best_ga.best_cost
        best_meta_name = "GA"
    print(f"\nMejor orden global ({best_meta_name}): costo={best_meta_cost:.2f}")
    print(f"  Orden: {best_order}")

    # D. Teoría de juegos
    game_res = experiment_game_theory(scenario, best_order=best_order)

    # E. Minimax / Alpha-Beta
    adv_data = experiment_adversarial(params)
    best_action = adv_data["alphabeta"].best_action

    # Integración
    plan = integrate_and_plan(scenario, best_order, game_res, best_action)

    print("\n" + "="*60)
    print("EJECUCIÓN COMPLETADA")
    print(f"Resultados en: {RESULTS_DIR}")
    print("="*60)

    return {
        "scenario": scenario,
        "heuristic": heur_results,
        "hill_climbing": hc_data,
        "metaheuristics": meta_data,
        "game_theory": game_res,
        "adversarial": adv_data,
        "plan": plan,
    }


if __name__ == "__main__":
    main()
