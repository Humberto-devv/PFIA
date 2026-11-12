# Sistema Inteligente de Planificación, Optimización y Toma de Decisiones Adversarias

Proyecto de segundo parcial – Modalidad individual.  
Desarrollo en Python.

## Descripción

Sistema que integra:

- **A.** Búsqueda heurística (Greedy Best-First y A*)
- **B.** Búsqueda local (Hill Climbing multi-start)
- **C.** Metaheurísticas (Recocido Simulado y Algoritmo Genético)
- **D.** Teoría de Juegos (matriz de pagos, dominancia, Nash puros)
- **E.** Minimax y poda Alfa-Beta (negociación de recursos)

Todo sobre un escenario logístico compartido de cuadrícula 12×12 configurable por JSON.

## Estructura

```
logistics_ai/
├── config/
│   ├── map.json          # Escenario (grid, entregas, obstáculos, costos, competidor)
│   └── params.json       # Parámetros de algoritmos y semillas
├── src/
│   ├── scenario.py       # Representación compartida del mapa y recursos
│   ├── pathfinding.py    # Greedy + A*
│   ├── local_search.py   # Hill Climbing
│   ├── metaheuristics.py # SA + GA
│   ├── game_theory.py    # Juego 2 jugadores
│   ├── adversarial.py    # Minimax / Alpha-Beta
│   └── main.py           # Integración y experimentos
├── tests/
│   └── test_scenario.py
├── results/              # CSV y JSON generados
├── figures/
├── data/
├── requirements.txt
└── README.md
```

## Instalación

```bash
cd logistics_ai
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

Desde la raíz del proyecto:

```bash
# Ejecución completa de todos los módulos + experimentos
cd src
python main.py
```

O módulo por módulo:

```bash
cd src
python scenario.py
python pathfinding.py
python local_search.py
python metaheuristics.py
python game_theory.py
python adversarial.py
```

## Pruebas

```bash
cd logistics_ai
pytest tests/ -v
```

## Configuración del escenario

En el archivo `config/map.json` se establece:

- Tamaño de la cuadrícula, inicio, meta
- Puntos de entrega (≥3)
- Obstáculos
- Zonas de alto costo
- Restricciones de combustible/payload
- Celdas y estrategias del competidor
- Semilla de reproducibilidad

Los parámetros de los algoritmos se controlan en `config/params.json`.

## Reproducibilidad

- Semilla global en `map.json` (`seed`)
- Lista de semillas de experimentos en `params.json` → `experiments.random_seeds`
- Todos los resultados se guardan en `results/` con los datos de cada corrida

## Notas de diseño

- **Heurística**: Manhattan (admisible en grid 4-conectada con costos ≥1). Se compara también Euclidean.
- **Hill Climbing**: vecindad por intercambio de dos entregas; multi-start para mitigar óptimos locales.
- **SA**: aceptación Metropolis + enfriamiento geométrico documentado.
- **GA**: representación de permutación, Order Crossover (OX), mutación swap, elitismo y torneo.
- **Juego**: 3 estrategias por jugador; se calcula matriz de pagos a partir del costo de la ruta bajo cada combinación.
- **Minimax**: subproblema de negociación de un recurso compartido por turnos; se verifica igualdad de valor entre minimax puro y alfa-beta.

## Limitaciones conocidas

- La evaluación de un orden de entregas requiere varias llamadas a A* (una por segmento), por lo que las metaheurísticas son costosas computacionalmente.
- El modelo del competidor es simplificado (incremento de costo en celdas fijas).
- No se implementan estrategias mixtas de Nash (solo puras).
- La profundidad de minimax está limitada (por defecto 4) por explosión combinatoria.
