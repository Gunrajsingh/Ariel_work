"""
Assignment 3 — Robot Olympics (monolith with evolutionary loop)

This file extends your A3 template with a full EA loop that evolves the BODY ONLY
for now. It uses the same EA scaffolding and operator styles as your A2 code:
- Tournament selection
- SBX crossover (here applied to the three [0,1] body vectors)
- Polynomial-bounded mutation analogue: here a block-reset mutation on [0,1]
- Elitism via Hall-of-Fame
- Random immigrants
- Adaptive mutation when stagnating

Controller remains the simple NN stub in this file. Swap it for your A2 MLP
genome later without changing the EA loop.

Outputs per run:
- Best fitness curve .png
- Per-generation best snapshot images
- JSONs of best body genotype and decoded graph
- Optional path overlay image

Known limitations:
- This A3 focuses on body evolution. Controller is a placeholder.
- Rendering is off by default; turn on video if needed.

"""

from __future__ import annotations

import csv
import json
import math
import os
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
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# =========================
# Type checking
# =========================
try:
    from typing import TypedDict
except ImportError:  # py<3.8 fallback
    TypedDict = dict  # type: ignore

console = Console()

# =========================
# Global settings
# =========================
SCRIPT_NAME = "A3_fixingbasic"
CWD = Path(".").resolve()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# EA parameters
POP_SIZE = 60
N_GEN = 20
TOURNSIZE = 3
CXPB = 0.9
MUTPB = 0.25
SBX_ETA = 20.0
IMM_FRAC = 0.10
STAGNATION_STEPS = (5, 10)  # boosts at 5, 10
MUTPB_BOOSTS = (1.8, 2.5)

# Sim + environment
SIM_DURATION = 15  # seconds
RATE_LIMIT_DU = 0.025
HINGE_LIMIT = math.pi / 2
WARMUP_STEPS = 120  # ~0.5 s at 240 Hz; simple_runner uses default dt
HARD_TIMEOUT = 15  # wall-clock safety
SPAWN_POS = [-0.8, 0, np.float64(18.68007496458845)]  # left as-is per user note
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

def _sbx_pair(a: np.ndarray, b: np.ndarray, eta: float, low=0.0, high=1.0, rng: np.random.Generator | None=None):
    rng = rng or np.random.default_rng
    u = rng.random(a.shape, dtype=np.float32)
    beta = np.empty_like(a, dtype=np.float32)
    mask = u <= 0.5
    beta[mask] = (2.0 * u[mask]) ** (1.0 / (eta + 1.0))
    beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (eta + 1.0))
    c1 = 0.5 * ((a + b) - beta * (b - a))
    c2 = 0.5 * ((a + b) + beta * (b - a))
    return np.clip(c1, low, high), np.clip(c2, low, high)

def sbx_body(g1: tuple[np.ndarray, np.ndarray, np.ndarray],
             g2: tuple[np.ndarray, np.ndarray, np.ndarray],
             eta: float = SBX_ETA,
             rng: np.random.Generator | None=None) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray],
                                                            tuple[np.ndarray, np.ndarray, np.ndarray]]:
    a1, b1 = _sbx_pair(g1[0], g2[0], eta, rng=rng)
    a2, b2 = _sbx_pair(g1[1], g2[1], eta, rng=rng)
    a3, b3 = _sbx_pair(g1[2], g2[2], eta, rng=rng)
    return (a1, a2, a3), (b1, b2, b3)

def block_mutation(g: tuple[np.ndarray, np.ndarray, np.ndarray],
                   indpb: float = 1.0 / BODY_L,
                   rng: np.random.Generator | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    t, c, r = g
    # reset each gene with prob indpb
    mask_t = rng.random(t.shape) < indpb
    mask_c = rng.random(c.shape) < indpb
    mask_r = rng.random(r.shape) < indpb
    t = t.copy(); c = c.copy(); r = r.copy()
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
    # --- NDE API per template: construct with module count, then forward([t,c,r]) ---
    t = t.astype(np.float32); c = c.astype(np.float32); r = r.astype(np.float32)
    nde = NeuralDevelopmentalEncoding(number_of_modules=nde_modules)
    # compatibility for downstream saving
    nde.t = t; nde.c = c; nde.r = r; nde.n_modules = nde_modules
    p_mats = nde.forward([t, c, r])
    decoder = HighProbabilityDecoder(nde_modules)
    graph = decoder.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
    spec = construct_mjspec_from_graph(graph)
    return BuiltBody(nde=nde, decoded_graph=graph, mjspec=spec)

def save_body_artifacts(run_dir: Path, built: BuiltBody, tag: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save decoded graph JSON
    gpath = run_dir / f"{tag}_decoded_graph.json"
    save_graph_as_json(built.decoded_graph, str(gpath))
    # Save NDE as JSON
    nde_json = {
        "t": built.nde.t.tolist(),
        "c": built.nde.c.tolist(),
        "r": built.nde.r.tolist(),
        "n_modules": built.nde.n_modules,
    }
    with open(run_dir / f"{tag}_nde.json", "w") as f:
        json.dump(nde_json, f, indent=2)

# =========================
# Controller (stub NN)
# =========================
HINGE_LIMIT = math.pi / 2
RATE_LIMIT_DU = 0.025
WARMUP_STEPS = 120  # ~0.5 s at 240 Hz; simple_runner uses default dt

def nn_controller(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
    """Simple 3-layer random-weight NN per step. Placeholder."""
    inp = len(data.qpos)
    hid = 8
    out = model.nu
    w1 = RNG.normal(0.0, 0.5, size=(inp, hid))
    w2 = RNG.normal(0.0, 0.5, size=(hid, hid))
    w3 = RNG.normal(0.0, 0.5, size=(hid, out))
    x  = data.qpos
    h1 = np.tanh(x @ w1)
    h2 = np.tanh(h1 @ w2)
    y  = np.tanh(h2 @ w3)
    return y * np.pi

# =========================
# Evaluation harness (patched)
# =========================
def run_episode_for_genotype(body_geno, duration: int = SIM_DURATION, mode: Literal["simple","no_render"]="simple"):
    """Build body, run in OlympicArena with stub controller, return (fitness, history_xyz, graph)."""
    mj.set_mjcb_control(None)
    world = OlympicArena()

    built = build_body(body_geno, nde_modules=NUM_OF_MODULES, rng=RNG)

    # Spawn with the robot's MjSpec, compile with world.spec.compile() per working template
    world.spawn(built.mjspec.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # tracker for the "core" geom; matches your template
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    # ---- Fix 1+2: per-episode fixed weights and action safety (warm-up, rate limit, clamp) ----
    def _episode_controller():
        inp = len(data.qpos)
        hid = 8
        out = model.nu
        w1 = RNG.normal(0.0, 0.5, size=(inp, hid))
        w2 = RNG.normal(0.0, 0.5, size=(hid, hid))
        w3 = RNG.normal(0.0, 0.5, size=(hid, out))
        u_apply = np.zeros(out, dtype=float)
        step = 0
        def _cb(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
            nonlocal u_apply, step, w1, w2, w3
            x = d.qpos
            h1 = np.tanh(x @ w1)
            h2 = np.tanh(h1 @ w2)
            y  = np.tanh(h2 @ w3) * math.pi  # raw command in radians
            y  = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            # warm-up ramp
            ramp = 1.0 if WARMUP_STEPS <= 0 else min(1.0, step / max(1, WARMUP_STEPS))
            u_cmd = ramp * y
            # rate limit
            du = np.clip(u_cmd - u_apply, -RATE_LIMIT_DU, RATE_LIMIT_DU)
            u_apply = u_apply + du
            # actuator range clamp
            u_apply = np.clip(u_apply, -HINGE_LIMIT, HINGE_LIMIT)
            step += 1
            return u_apply.astype(np.float64, copy=False)
        return _cb

    episode_cb = _episode_controller()
    ctrl = Controller(controller_callback_function=episode_cb, tracker=tracker)

    # Bind tracker
    if ctrl.tracker is not None:
        ctrl.tracker.setup(world.spec, data)

    # Set callback
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, *args, **kwargs))

    # Run
    if mode == "simple":
        simple_runner(model, data, duration=duration)
    else:
        # No visual; same as simple
        simple_runner(model, data, duration=duration)

    # --- PATCH 2: fixed NumPy truth-value ambiguity ---
    hist = tracker.history.get("xpos", [[]])
    if (
        hist is None
        or len(hist) == 0
        or not isinstance(hist, (list, tuple))
        or not isinstance(hist[0], (list, tuple, np.ndarray))
    ):
        hist = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # simple forward metric: x displacement
    disp = float(hist[-1][0] - hist[0][0])
    return disp, hist, built.decoded_graph

# =========================
# Rendering helpers
# =========================
def render_snapshot(world, data, save_path: str | None = None):
    img = single_frame_renderer(world.spec, data, width=640, height=480)
    if save_path:
        with open(save_path, "wb") as f:
            f.write(img)
    return img

def show_xpos_history(history_xyz: list[list[float]], save_path: str | None = None) -> None:
    """Overlay XY path on a background frame of the OlympicArena."""
    # Background frame
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # World background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    img = single_frame_renderer(world.spec, data, width=640, height=480)
    # Simple placeholder: we only save the background image
    if save_path:
        with open(save_path, "wb") as f:
            f.write(img)

# =========================
# DEAP scaffolding
# =========================
try:
    creator.FitnessMax
except AttributeError:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
try:
    creator.BodyIndividual
except AttributeError:
    # store the three numpy arrays inside a list-like Individual
    creator.create("BodyIndividual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def init_individual():
    geno = init_body_genotype(RNG, BODY_L)
    # Wrap inside a DEAP list; element 0 is the body genotype tuple
    return creator.BodyIndividual([geno])

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Crossover over the tuple of arrays
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

# Mutation: block reset on [0,1]
def mutate(ind):
    g = ind[0]
    ind[0] = block_mutation(g, indpb=1.0 / BODY_L, rng=RNG)
    if hasattr(ind.fitness, "values"):
        del ind.fitness.values
    return (ind,)

toolbox.register("mate", mate)
toolbox.register("mutate", mutate)

# Evaluation
def evaluate_individual(ind):
    geno = ind[0]
    try:
        fit, hist, graph = run_episode_for_genotype(geno, duration=SIM_DURATION, mode="simple")
    except Exception as e:
        console.log(f"[Eval] Exception: {e}. Assigning poor fitness.")
        fit, hist, graph = -1e9, [[0,0,0]], None
    return (fit,)

toolbox.register("evaluate", evaluate_individual)

# ---- Fix 3: HallOfFame similarity must return a boolean ----
def _hof_similar(a, b) -> bool:
    # Individuals store the genotype tuple at index 0: (t, c, r) arrays in [0,1].
    try:
        g1 = a[0]; g2 = b[0]
        v1 = np.concatenate([np.ravel(g1[0]), np.ravel(g1[1]), np.ravel(g1[2])])
        v2 = np.concatenate([np.ravel(g2[0]), np.ravel(g2[1]), np.ravel(g2[2])])
        return bool(np.allclose(v1, v2, atol=1e-12, rtol=1e-12))
    except Exception:
        # Fallback: compare lengths to avoid ambiguous truth values
        try:
            return bool(len(a) == len(b))
        except Exception:
            return False

# Selection
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

# =========================
# EA runner
# =========================
def run_ea():
    random.seed(SEED)
    np.random.seed(SEED)

    # Initial population
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate initial
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    # Hall of Fame (elitism)
    ELITE_K = max(1, POP_SIZE // 20)
    hof = tools.HallOfFame(ELITE_K, similar=_hof_similar)
    hof.update(pop)

    best_per_gen = [tools.selBest(pop, 1)[0].fitness.values[0]]

    no_improve = 0
    best_so_far = best_per_gen[-1]

    # Save initial best artifacts
    best0 = tools.selBest(pop, 1)[0]
    built0 = build_body(best0[0], nde_modules=NUM_OF_MODULES, rng=RNG)
    save_body_artifacts(DATA, built0, tag="gen_000_best")
    try:
        # Snapshot
        world0 = OlympicArena()
        world0.spawn(built0.mjspec.spec, spawn_position=SPAWN_POS)
        m0 = world0.spec.compile()
        d0 = mj.MjData(m0)
        render_snapshot(world0, d0, save_path=str(DATA / "gen_000_best.png"))
    except Exception as e:
        console.log(f"[Render init] {e}")

    t_wall = time.time()
    for gen in range(1, N_GEN + 1):
        # Selection and clone
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

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
            idx = random.randrange(len(offspring))
            offspring[idx] = toolbox.individual()

        # Evaluate
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Update HOF and elitism
        hof.update(offspring)
        elites = list(map(toolbox.clone, hof.items))

        # Survivor selection: elitism + best of offspring
        offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        pop = elites + offspring[: max(0, POP_SIZE - len(elites))]

        # Track
        best = tools.selBest(pop, 1)[0]
        best_per_gen.append(best.fitness.values[0])

        if best.fitness.values[0] > best_so_far + 1e-12:
            best_so_far = best.fitness.values[0]
            no_improve = 0
        else:
            no_improve += 1

        dt_wall = time.time() - t_wall
        console.log(
            f"[EA] Gen {gen:3d} | best = {best.fitness.values[0]:.4f} | no_improve={no_improve:2d} | t={dt_wall:.1f}s"
        )

        # Save per-gen best artifacts
        built = build_body(best[0], nde_modules=NUM_OF_MODULES, rng=RNG)
        save_body_artifacts(DATA, built, tag=f"gen_{gen:03d}_best")

        # Snapshot
        try:
            world_g = OlympicArena()
            world_g.spawn(built.mjspec.spec, spawn_position=SPAWN_POS)
            mg = world_g.spec.compile()
            dg = mj.MjData(mg)
            render_snapshot(world_g, dg, save_path=str(DATA / f"gen_{gen:03d}_best.png"))
        except Exception as e:
            console.log(f"[Render gen {gen}] {e}")

    # Save curve
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
