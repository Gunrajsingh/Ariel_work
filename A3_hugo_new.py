"""
Assignment 3 — Robot Olympics (monolith with evolutionary loop, evolving both body and controller via nested EAs)

This file runs a hierarchical EA loop:
- Outer EA: evolves robot BODY genotype as before, using tournament selection, SBX, block mutation, elitism, immigrants, and adaptive mutation.
- Inner EA (sub-loop): for each candidate body, runs a full EA to evolve the neural controller for that body to maximize forward displacement.

Fitness for each body is set as the best fitness found by its optimal controller.

The body and controller evolutionary operators match Assignment 2 & 3 templates and are inherited from those files.

Outputs per run:
- Best fitness curve .png
- Per-generation best snapshot images
- JSONs of best body genotype and decoded graph
- Optional path overlay image

Known limitations:
- This assignment performs nested EAs and can be very computationally expensive.
- All controller evolution settings are local to each body evaluation.

"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco as mj
import numpy as np
import numpy.typing as npt
from deap import base, creator, tools
from rich.console import Console

# =========================
# Project-scoped imports (Ariel)
# =========================
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker

console = Console()

# =========================
# Global settings
# =========================
SCRIPT_NAME = "A3_body_controller_EA"
CWD = Path(".").resolve()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# Outer EA (body) params
POP_SIZE = 60
N_GEN = 20
TOURNSIZE = 3
CXPB = 0.9
MUTPB = 0.25
SBX_ETA = 20.0
IMM_FRAC = 0.10
STAGNATION_STEPS = (5, 10)
MUTPB_BOOSTS = (1.8, 2.5)

# Sim + environment
SIM_DURATION = 15
RATE_LIMIT_DU = 0.015
HINGE_LIMIT = math.pi / 2
WARMUP_STEPS = 240
SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]

# Body encoding
BODY_L = 64
NUM_OF_MODULES = 30

# Plot outputs
EA_FITNESS_PNG = DATA / "ea_best_over_generations.png"

# =========================
# Genotype and operators for BODY
# =========================
def init_body_genotype(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = rng.random(n).astype(np.float32)
    c = rng.random(n).astype(np.float32)
    r = rng.random(n).astype(np.float32)
    return (t, c, r)

def _sbx_pair(a: np.ndarray, b: np.ndarray, eta: float, low=0.0, high=1.0, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng
    u = rng.random(a.shape, dtype=np.float32)
    beta = np.empty_like(a, dtype=np.float32)
    mask = u <= 0.5
    beta[mask] = (2.0 * u[mask]) ** (1.0 / (eta + 1.0))
    beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (eta + 1.0))
    c1 = 0.5 * ((a + b) - beta * (b - a))
    c2 = 0.5 * ((a + b) + beta * (b - a))
    return np.clip(c1, low, high), np.clip(c2, low, high)

def sbx_body(
    g1: tuple[np.ndarray, np.ndarray, np.ndarray],
    g2: tuple[np.ndarray, np.ndarray, np.ndarray],
    eta: float = SBX_ETA,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    a1, b1 = _sbx_pair(g1[0], g2[0], eta, rng=rng)
    a2, b2 = _sbx_pair(g1[1], g2[1], eta, rng=rng)
    a3, b3 = _sbx_pair(g1[2], g2[2], eta, rng=rng)
    return (a1, a2, a3), (b1, b2, b3)

def block_mutation(
    g: tuple[np.ndarray, np.ndarray, np.ndarray],
    indpb: float = 1.0 / BODY_L,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    t, c, r = g
    mask_t = rng.random(t.shape) < indpb
    mask_c = rng.random(c.shape) < indpb
    mask_r = rng.random(r.shape) < indpb
    t = t.copy()
    c = c.copy()
    r = r.copy()
    t[mask_t] = rng.random(np.count_nonzero(mask_t)).astype(np.float32)
    c[mask_c] = rng.random(np.count_nonzero(mask_c)).astype(np.float32)
    r[mask_r] = rng.random(np.count_nonzero(mask_r)).astype(np.float32)
    return (t, c, r)

# =========================
# Decoding to body graph and MuJoCo spec
# =========================
@dataclass
class BuiltBody:
    nde: NeuralDevelopmentalEncoding
    decoded_graph: Any
    mjspec: Any

def build_body(geno: tuple[np.ndarray, np.ndarray, np.ndarray], nde_modules: int, rng: np.random.Generator) -> BuiltBody:
    t, c, r = geno
    t = t.astype(np.float32)
    c = c.astype(np.float32)
    r = r.astype(np.float32)
    nde = NeuralDevelopmentalEncoding(number_of_modules=nde_modules)
    nde.t = t
    nde.c = c
    nde.r = r
    nde.n_modules = nde_modules
    p_mats = nde.forward([t, c, r])
    decoder = HighProbabilityDecoder(nde_modules)
    graph = decoder.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
    spec = construct_mjspec_from_graph(graph)
    return BuiltBody(nde=nde, decoded_graph=graph, mjspec=spec)

def save_body_artifacts(run_dir: Path, built: BuiltBody, tag: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    gpath = run_dir / f"{tag}_decoded_graph.json"
    save_graph_as_json(built.decoded_graph, str(gpath))
    nde_json = {
        "t": built.nde.t.tolist(),
        "c": built.nde.c.tolist(),
        "r": built.nde.r.tolist(),
        "n_modules": built.nde.n_modules,
    }
    with open(run_dir / f"{tag}_nde.json", "w") as f:
        json.dump(nde_json, f, indent=2)

def _choose_track_body(model: mj.MjModel) -> str:
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "robot-core")
    if bid != -1:
        return "robot-core"
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, 1 if model.nbody > 1 else 0)

def _topdown_camera() -> mj.MjvCamera:
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat = [2.5, 0.0, 0.0]
    cam.distance = 10.0
    cam.azimuth = 0
    cam.elevation = -90
    return cam

def render_snapshot(world, save_path: str | None = None):
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    single_frame_renderer(
        world.spec,
        data,
        width=640,
        height=480,
        camera=_topdown_camera(),
        save=True,
        save_path=save_path,
    )

# =========================
# Controller EA (from A2)
# =========================

def theta_size(inp: int, hidden: int, out: int) -> int:
    # Returns total number of flattened MLP weights and biases
    return inp * hidden + hidden + hidden * hidden + hidden + hidden * out + out

def unpack_theta(theta: np.ndarray, inp: int, hidden: int, out: int):
    # Unpacks flattened weights for a 2-hidden-layer MLP
    i = 0
    W1 = theta[i:i + inp * hidden].reshape(inp, hidden)
    i += inp * hidden
    b1 = theta[i:i + hidden]
    i += hidden
    W2 = theta[i:i + hidden * hidden].reshape(hidden, hidden)
    i += hidden * hidden
    b2 = theta[i:i + hidden]
    i += hidden
    W3 = theta[i:i + hidden * out].reshape(hidden, out)
    i += hidden * out
    b3 = theta[i:i + out]
    return (W1, b1, W2, b2, W3, b3)

def mlp_forward(x: np.ndarray, params) -> np.ndarray:
    W1, b1, W2, b2, W3, b3 = params
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    y = np.tanh(h2 @ W3 + b3)
    return y

def run_episode_with_controller_theta(theta, model, data, steps=1000, rate_limit_du=0.015, hinge_limit=math.pi/2, warmup_steps=240):
    # One episode for a candidate controller on the given model/data. Returns fitness.
    try:
        track_body = _choose_track_body(model)
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind=track_body)
        tracker.setup(spec=model, data=data)
    except Exception:
        tracker = None
    start_xy = np.array(data.body(track_body).xpos[:2], dtype=float).copy() if tracker else np.zeros(2, dtype=float)
    params = unpack_theta(theta, inp=len(data.qpos) + len(data.qvel) + 3, hidden=8, out=model.nu)
    u_apply = np.zeros(model.nu, dtype=float)
    for step in range(steps):
        tsec = data.time
        tf = np.array([tsec, math.sin(2 * math.pi * tsec), math.cos(2 * math.pi * tsec)], dtype=float)
        x = np.concatenate([data.qpos, data.qvel, tf])
        u_cmd = mlp_forward(x, params)
        ramp = min(1.0, step / max(1, warmup_steps))
        u_cmd = ramp * u_cmd
        du = np.clip(u_cmd - u_apply, -rate_limit_du, rate_limit_du)
        u_apply = np.clip(u_apply + du, -hinge_limit, hinge_limit)
        data.ctrl = u_apply
        mj.mj_step(model, data)
        if tracker:
            tracker.log(data)
    hist = tracker.history.get("xpos", [[]]) if tracker else None
    if hist is None or len(hist) < 2:
        end_xy = np.array(data.body(track_body).xpos[:2], dtype=float)
        start3 = [float(start_xy[0]), float(start_xy[1]), 0.0]
        end3 = [float(end_xy[0]), float(end_xy[1]), 0.0]
        hist = [start3, end3]
    disp = float(hist[-1][0] - hist[0][0])
    return disp

def controller_ea_for_body(model, data, pop_size=40, n_gen=10, hidden=8, steps=1000):
    # Evolve controllers for this body+model, return best fitness achieved
    INP = len(data.qpos) + len(data.qvel) + 3
    OUT = model.nu
    GENOME_SIZE = theta_size(INP, hidden, OUT)
    WLOW, WHIGH = -2.0, 2.0

    try:
        creator.ControllerFitnessMax
    except AttributeError:
        creator.create("ControllerFitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.ControllerIndividual
    except AttributeError:
        creator.create("ControllerIndividual", list, fitness=creator.ControllerFitnessMax)

    toolbox = base.Toolbox()

    def init_controller_ind():
        theta = np.random.normal(0.0, 0.1, size=GENOME_SIZE).astype(float)
        theta = np.clip(theta, WLOW, WHIGH)
        return creator.ControllerIndividual(theta.tolist())

    def mate(ind1, ind2):
        a1 = np.array(ind1, dtype=float)
        a2 = np.array(ind2, dtype=float)
        eta = 20.0
        low, up = WLOW, WHIGH
        child1, child2 = tools.cxSimulatedBinaryBounded(a1, a2, eta, low, up)
        ind1[:] = child1.tolist()
        ind2[:] = child2.tolist()
        if hasattr(ind1.fitness, "values"):
            del ind1.fitness.values
        if hasattr(ind2.fitness, "values"):
            del ind2.fitness.values
        return ind1, ind2

    def mutate(ind):
        tools.mutPolynomialBounded(ind, low=WLOW, up=WHIGH, eta=20.0, indpb=1.0/GENOME_SIZE)
        if hasattr(ind.fitness, "values"):
            del ind.fitness.values
        return (ind,)

    def evaluate(ind):
        theta = np.array(ind, dtype=float)
        model_local = mj.MjModel(model)  # Fresh copy -- ensure identical environment
        data_local = mj.MjData(model_local)
        mj.mj_resetData(model_local, data_local)
        mj.mj_forward(model_local, data_local)
        try:
            fit = run_episode_with_controller_theta(theta, model_local, data_local, steps=steps)
        except Exception:
            fit = -1e9
        return (fit,)

    toolbox.register("individual", init_controller_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=pop_size)
    ELITE_K = max(1, pop_size // 20)
    hof = tools.HallOfFame(ELITE_K)
    mutation_boosts = [1.8, 2.5]
    stagnation_steps = [3, 6]
    best_per_gen = [None]
    best_fit = -np.inf
    no_improve = 0

    for gen in range(n_gen):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9:
                toolbox.mate(c1, c2)
        adapt_mutpb = 0.2
        if no_improve >= stagnation_steps[0]:
            adapt_mutpb = min(1.0, 0.2 * mutation_boosts[0])
        if no_improve >= stagnation_steps[1]:
            adapt_mutpb = min(1.0, 0.2 * mutation_boosts[1])
        for m in offspring:
            if random.random() < adapt_mutpb:
                toolbox.mutate(m)
        n_imm = max(0, int(0.10 * len(offspring)))
        for _ in range(n_imm):
            offspring[random.randrange(len(offspring))] = toolbox.individual()
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        hof.update(offspring)
        elites = list(map(toolbox.clone, hof.items))
        offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        pop = elites + offspring[: max(0, pop_size - len(elites))]
        best = tools.selBest(pop, 1)[0]
        best_fit_val = best.fitness.values[0]
        if best_fit_val > best_fit + 1e-12:
            best_fit = best_fit_val
            no_improve = 0
        else:
            no_improve += 1
        best_per_gen.append(best_fit_val)
    return best_fit

# =========================
# DEAP scaffolding (body EA, outer loop)
# =========================
try:
    creator.FitnessMax
except AttributeError:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
try:
    creator.BodyIndividual
except AttributeError:
    creator.create("BodyIndividual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def init_individual():
    geno = init_body_genotype(RNG, BODY_L)
    return creator.BodyIndividual([geno])

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def mate(ind1, ind2):
    g1 = ind1[0]
    g2 = ind2[0]
    c1, c2 = sbx_body(g1, g2, eta=SBX_ETA, rng=RNG)
    ind1[0] = c1
    ind2[0] = c2
    if hasattr(ind1.fitness, "values"):
        del ind1.fitness.values
    if hasattr(ind2.fitness, "values"):
        del ind2.fitness.values
    return ind1, ind2

def mutate(ind):
    g = ind[0]
    ind[0] = block_mutation(g, indpb=1.0 / BODY_L, rng=RNG)
    if hasattr(ind.fitness, "values"):
        del ind.fitness.values
    return (ind,)

toolbox.register("mate", mate)
toolbox.register("mutate", mutate)

def evaluate_individual(ind):
    # For each body candidate, run controller EA and use best controller fitness as body fitness
    geno = ind[0]
    try:
        built = build_body(geno, nde_modules=NUM_OF_MODULES, rng=RNG)
        world = OlympicArena()
        world.spawn(built.mjspec.spec, position=SPAWN_POS)
        model = world.spec.compile()
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        # Controller EA sub-loop
        controller_best_fitness = controller_ea_for_body(model, data, pop_size=40, n_gen=10, hidden=8, steps=1000)
    except Exception as e:
        console.log(f"[Eval] Exception (body+controller EA): {e}. Assigning poor fitness.")
        controller_best_fitness = -1e9
    return (controller_best_fitness,)

toolbox.register("evaluate", evaluate_individual)

def _hof_similar(a, b) -> bool:
    try:
        g1 = a[0]
        g2 = b[0]
        v1 = np.concatenate([np.ravel(g1[0]), np.ravel(g1[1]), np.ravel(g1[2])])
        v2 = np.concatenate([np.ravel(g2[0]), np.ravel(g2[1]), np.ravel(g2[2])])
        return bool(np.allclose(v1, v2, atol=1e-12, rtol=1e-12))
    except Exception:
        try:
            return bool(len(a) == len(b))
        except Exception:
            return False

toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

# =========================
# EA runner
# =========================
def run_ea():
    random.seed(SEED)
    np.random.seed(SEED)
    pop = toolbox.population(n=POP_SIZE)
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)
    ELITE_K = max(1, POP_SIZE // 20)
    hof = tools.HallOfFame(ELITE_K, similar=_hof_similar)
    hof.update(pop)
    best_per_gen = [tools.selBest(pop, 1)[0].fitness.values[0]]
    no_improve = 0
    best_so_far = best_per_gen[-1]
    # ----- Render/init snapshot of gen 0 best -----
    best0 = tools.selBest(pop, 1)[0]
    built0 = build_body(best0[0], nde_modules=NUM_OF_MODULES, rng=RNG)
    save_body_artifacts(DATA, built0, tag="gen_000_best")
    try:
        world0 = OlympicArena()
        world0.spawn(built0.mjspec.spec, position=SPAWN_POS)
        render_snapshot(world0, save_path=str(DATA / "gen_000_best.png"))
    except Exception as e:
        console.log(f"[Render init] {e}")
    t_wall = time.time()
    for gen in range(1, N_GEN + 1):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)
        # Adaptive mutation
        adapt_mutpb = MUTPB
        if no_improve >= STAGNATION_STEPS[0]:
            adapt_mutpb = min(1.0, MUTPB * MUTPB_BOOSTS[0])
        if no_improve >= STAGNATION_STEPS[1]:
            adapt_mutpb = min(1.0, MUTPB * MUTPB_BOOSTS[1])
        for m in offspring:
            if random.random() < adapt_mutpb:
                toolbox.mutate(m)
        # Random immigrants
        n_imm = max(0, int(IMM_FRAC * len(offspring)))
        for _ in range(n_imm):
            offspring[random.randrange(len(offspring))] = toolbox.individual()
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        # Elitism
        hof.update(offspring)
        elites = list(map(toolbox.clone, hof.items))
        offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        pop = elites + offspring[: max(0, POP_SIZE - len(elites))]
        # Logging
        best = tools.selBest(pop, 1)[0]
        best_per_gen.append(best.fitness.values[0])
        if best.fitness.values[0] > best_so_far + 1e-12:
            best_so_far = best.fitness.values[0]
            no_improve = 0
        else:
            no_improve += 1
        dt_wall = time.time() - t_wall
        console.log(
            f"[EA] Gen {gen:3d} | best = {best.fitness.values[0]:.4f} | "
            f"no_improve={no_improve:2d} | t={dt_wall:.1f}s"
        )
        # Save artifacts + per-gen snapshot
        built = build_body(best[0], nde_modules=NUM_OF_MODULES, rng=RNG)
        tag = f"gen_{gen:03d}_best"
        save_body_artifacts(DATA, built, tag=tag)
        try:
            world_g = OlympicArena()
            world_g.spawn(built.mjspec.spec, position=SPAWN_POS)
            render_snapshot(world_g, save_path=str(DATA / f"{tag}.png"))
        except Exception as e:
            console.log(f"[Render gen {gen}] {e}")
    # Plot curve
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(best_per_gen, marker="o")
        plt.xlabel("Generation")
        plt.ylabel("Best fitness (x-displacement)")
        plt.title(f"EA best over {N_GEN} generations (pop={POP_SIZE})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(EA_FITNESS_PNG, dpi=150)
        plt.close()
        console.log(f"[EA] Saved curve to {EA_FITNESS_PNG}")
    except Exception as e:
        console.log(f"[Plot] {e}")

    return tools.selBest(pop, 1)[0], best_per_gen

# =========================
# Main
# =========================
def main():
    best, curve = run_ea()
    print(f"[EA] Done. Best fitness = {best.fitness.values[0]:.4f}")

if __name__ == "__main__":
    main()
