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
STAGNATION_STEPS = (5, 10)
MUTPB_BOOSTS = (1.8, 2.5)

# Sim + environment
SIM_DURATION = 15
RATE_LIMIT_DU = 0.015
HINGE_LIMIT = math.pi / 2
WARMUP_STEPS = 240
HARD_TIMEOUT = 15
SPAWN_POS = [-0.8, 0, 0.1]  # keep as-is
TARGET_POSITION = [5, 0, 0.5]

# Body encoding
BODY_L = 64
NUM_OF_MODULES = 30

# Plot outputs
EA_FITNESS_PNG = DATA / "ea_best_over_generations.png"

# =========================
# A2 brain loader (learned controller)
# =========================
CONTROLLER_JSON_PATH = "controller_data"
_A2_BRAIN_CACHE = None


def _solve_inp_len(theta_len: int, hidden: int, out_dim: int) -> int:
    num = theta_len - out_dim
    inp_float = num / float(hidden) - hidden - out_dim - 2
    inp = int(round(inp_float))
    if inp < 1:
        raise ValueError(f"Bad dims: theta_len={theta_len}, hidden={hidden}, out={out_dim}")
    return inp


def _unpack_theta(theta: np.ndarray, inp: int, hidden: int, out_dim: int):
    i = 0
    W1 = theta[i:i + inp * hidden].reshape(inp, hidden)
    i += inp * hidden
    b1 = theta[i:i + hidden]
    i += hidden
    W2 = theta[i:i + hidden * hidden].reshape(hidden, hidden)
    i += hidden * hidden
    b2 = theta[i:i + hidden]
    i += hidden
    W3 = theta[i:i + hidden * out_dim].reshape(hidden, out_dim)
    i += hidden * out_dim
    b3 = theta[i:i + out_dim]
    return (W1, b1, W2, b2, W3, b3)


def _mlp_forward(x: np.ndarray, params) -> np.ndarray:
    W1, b1, W2, b2, W3, b3 = params
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    y = np.tanh(h2 @ W3 + b3)
    return y


def _load_a2_brain(path: str = CONTROLLER_JSON_PATH):
    global _A2_BRAIN_CACHE
    if _A2_BRAIN_CACHE is not None:
        return _A2_BRAIN_CACHE
    try:
        with open(path, "r") as f:
            blob = json.load(f)
        theta = np.asarray(blob["theta"], dtype=float)
        hidden = int(blob["hidden"])
        out_saved = int(blob["nu"])
        inp_saved = _solve_inp_len(len(theta), hidden, out_saved)
        params = _unpack_theta(theta, inp_saved, hidden, out_saved)
        _A2_BRAIN_CACHE = {
            "params": params,
            "inp_saved": inp_saved,
            "hidden": hidden,
            "out_saved": out_saved,
        }
        console.log(
            f"[Brain] Loaded A2 MLP: inp={inp_saved}, hidden={hidden}, out={out_saved}, theta_len={len(theta)}"
        )
    except Exception as e:
        console.log(f"[Brain] Could not load controller_data ({e}); falling back to random stub.")
        _A2_BRAIN_CACHE = None
    return _A2_BRAIN_CACHE

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

# =========================
# Controller (stub NN)
# =========================
def nn_controller(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
    inp = len(data.qpos)
    hid = 8
    out = model.nu
    w1 = RNG.normal(0.0, 0.5, size=(inp, hid))
    w2 = RNG.normal(0.0, 0.5, size=(hid, hid))
    w3 = RNG.normal(0.0, 0.5, size=(hid, out))
    x = data.qpos
    h1 = np.tanh(x @ w1)
    h2 = np.tanh(h1 @ w2)
    y = np.tanh(h2 @ w3)
    return y * np.pi

# =========================
# Evaluation harness using simple_runner
# =========================
def _choose_track_body(model: mj.MjModel) -> str:
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "robot-core")
    if bid != -1:
        return "robot-core"
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, 1 if model.nbody > 1 else 0)


def run_episode_for_genotype(body_geno, duration: int = SIM_DURATION, mode: Literal["simple", "no_render"] = "simple"):
    """
    Build body, reset data, set controller, run headless via simple_runner.
    Returns (fitness = x-displacement, history_xyz, graph).
    """
    mj.set_mjcb_control(None)

    world = OlympicArena()
    built = build_body(body_geno, nde_modules=NUM_OF_MODULES, rng=RNG)
    world.spawn(built.mjspec.spec, position=SPAWN_POS)

    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset then forward to make derived fields consistent
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    if model.nu == 0 or model.nv == 0 or model.nbody < 2:
        return -1e6, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], built.decoded_graph

    track_body = _choose_track_body(model)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind=track_body)

    # Record absolute start pose for robust fallback
    start_xy = np.array(data.body(track_body).xpos[:2], dtype=float).copy()

    def _episode_controller():
        step = 0
        brain = _load_a2_brain()
        out = model.nu
        ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=float).reshape(out, 2)
        low = ctrlrange[:, 0].copy()
        high = ctrlrange[:, 1].copy()
        limited = np.array(model.actuator_ctrllimited, dtype=bool).reshape(-1)
        bad = (~limited) | ~np.isfinite(low) | ~np.isfinite(high) | (high <= low)
        low[bad] = -1.0
        high[bad] = 1.0
        center = 0.5 * (low + high)
        halfspan = 0.5 * (high - low)
        u_apply = center.copy()
        base_rate = 0.02 * (high - low)
        rate = np.minimum(base_rate, RATE_LIMIT_DU)

        def _cb(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
            nonlocal u_apply, step, brain
            t_now = d.time
            x_pq = np.concatenate([d.qpos, d.qvel]).astype(float, copy=False)
            if brain is not None:
                inp_saved = brain["inp_saved"]
                out_saved = brain["out_saved"]
                params = brain["params"]
                x_in = np.zeros(inp_saved, dtype=float)
                x_in[-3:] = [t_now, math.sin(2 * math.pi * t_now), math.cos(2 * math.pi * t_now)]
                k = min(x_pq.size, inp_saved - 3)
                if k > 0:
                    x_in[:k] = x_pq[:k]
                y = _mlp_forward(x_in, params)
                if out_saved == out:
                    y_out = y
                elif out_saved > out:
                    y_out = y[:out]
                else:
                    y_out = np.resize(y, out)
            else:
                hid = 8
                w1 = RNG.normal(0.0, 0.1, size=(x_pq.size + 3, hid))
                w2 = RNG.normal(0.0, 0.1, size=(hid, hid))
                w3 = RNG.normal(0.0, 0.1, size=(hid, out))
                x_in = np.concatenate(
                    [x_pq, [t_now, math.sin(2 * math.pi * t_now), math.cos(2 * math.pi * t_now)]]
                )
                h1 = np.tanh(x_in @ w1)
                h2 = np.tanh(h1 @ w2)
                y_out = np.tanh(h2 @ w3)

            y_out = np.nan_to_num(y_out, nan=0.0, posinf=0.0, neginf=0.0)
            u_target = center + halfspan * y_out
            ramp = 1.0 if WARMUP_STEPS <= 0 else min(1.0, step / max(1, WARMUP_STEPS))
            u_cmd = center + ramp * (u_target - center)
            du = np.clip(u_cmd - u_apply, -rate, rate)
            u_apply = np.clip(u_apply + du, low, high)
            step += 1
            return u_apply.astype(np.float64, copy=False)

        return _cb

    episode_cb = _episode_controller()
    ctrl = Controller(controller_callback_function=episode_cb, tracker=tracker)

    if ctrl.tracker is not None:
        ctrl.tracker.setup(world.spec, data)

    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Run headless
    simple_runner(model, data, duration=duration)

    # Prefer tracker history. If absent, fall back to absolute pose delta.
    hist = tracker.history.get("xpos", [[]])
    if (
        hist is None
        or len(hist) < 2
        or not isinstance(hist, (list, tuple))
        or not isinstance(hist[0], (list, tuple, np.ndarray))
    ):
        end_xy = np.array(data.body(track_body).xpos[:2], dtype=float)
        start3 = [float(start_xy[0]), float(start_xy[1]), 0.0]
        end3 = [float(end_xy[0]), float(end_xy[1]), 0.0]
        hist = [start3, end3]

    disp = float(hist[-1][0] - hist[0][0])
    return disp, hist, built.decoded_graph

def _topdown_camera() -> mj.MjvCamera:
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat = [2.5, 0.0, 0.0]
    cam.distance = 10.0
    cam.azimuth = 0
    cam.elevation = -90
    return cam

# =========================
# Rendering helpers
# =========================
def render_snapshot(world, save_path: str | None = None):
    """Compile a fresh model and render one frame from a top-down camera."""
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
# DEAP scaffolding
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
    geno = ind[0]
    try:
        fit, hist, graph = run_episode_for_genotype(geno, duration=SIM_DURATION, mode="simple")
    except Exception as e:
        console.log(f"[Eval] Exception: {e}. Assigning poor fitness.")
        fit, hist, graph = -1e9, [[0, 0, 0]], None
    return (fit,)


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

    # Evaluate initial population
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
        # helper compiles model & renders internally
        render_snapshot(world0, save_path=str(DATA / "gen_000_best.png"))
    except Exception as e:
        console.log(f"[Render init] {e}")

    t_wall = time.time()

    for gen in range(1, N_GEN + 1):
        # Selection + cloning
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

        # Evaluate
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
            # helper compiles model & renders internally
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
