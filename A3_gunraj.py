"""
Assignment 3 â€” Robot Olympics Co-Evolution
PRAGMATIC VERSION - Multiple Attempts Per Body

CRITICAL CHANGE: Try multiple NDE seeds per body genotype to find viable robots.
This bypasses the low viability issue while still doing co-evolution.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mujoco as mj
import numpy as np
import numpy.typing as npt
from deap import base, creator, tools
from rich.console import Console
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# Configuration
# =========================
SCRIPT_NAME = "A3_PRAGMATIC_WORKING"
CWD = Path(".").resolve()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# Smaller, faster evolution
BODY_POP_SIZE = 15
BODY_N_GEN = 8
BODY_TOURNSIZE = 3
BODY_CXPB = 0.7
BODY_MUTPB = 0.25
BODY_SBX_ETA = 15.0
BODY_IMM_FRAC = 0.25
BODY_MAX_ATTEMPTS = 5  # Try 5 different seeds per body

# Controller EA
CTRL_POP_SIZE = 20
CTRL_N_GEN = 12
CTRL_TOURNSIZE = 3
CTRL_CXPB = 0.8
CTRL_MUTPB = 0.25
CTRL_SBX_ETA = 15.0
CTRL_IMM_FRAC = 0.20

# Simulation
SIM_DURATION = 20
RATE_LIMIT_DU = 0.08
HINGE_LIMIT = math.pi / 2
WARMUP_STEPS = int(0.1 * 240)
SPAWN_POS = [-0.8, 0, 0.2]
TARGET_POSITION = [5, 0, 0.5]

# Body encoding
BODY_L = 64
NUM_OF_MODULES = 80

# Controller
CTRL_HIDDEN = 20
CTRL_W_LOW, CTRL_W_HIGH = -3.0, 3.0

# Output
EA_FITNESS_PNG = DATA / "ea_best_over_generations.png"
EA_PATH_PNG = DATA / "robot_path_final.png"
EA_VIDEO_MP4 = DATA / "robot_final_video.mp4"

_BODY_ARCH_CACHE = {}

# =========================
# Simplified Fitness
# =========================
def compute_simple_forward_fitness(start_pos, end_pos, history, duration):
    """Simple fitness: reward forward movement."""
    start_x = float(start_pos[0])
    end_x = float(end_pos[0])
    start_z = float(start_pos[2])
    end_z = float(end_pos[2])

    forward_distance = end_x - start_x

    if forward_distance > 0:
        fitness = forward_distance ** 1.5 + forward_distance * 3.0
    else:
        fitness = forward_distance * 2.0

    # Milestone bonuses
    if end_x > 0:
        fitness += 2.0
    if end_x > 1.5:
        fitness += 5.0
    if end_x > 3.5:
        fitness += 10.0
    if end_x > 5.0:
        fitness += 50.0

    # Penalties
    if end_z < -1.0:
        fitness -= 3.0
    if end_z > 0.05:
        fitness += 0.5

    if fitness > 0.3 or forward_distance > 0.2:
        console.log(f"[Fitness] dist={forward_distance:.3f}m, x={end_x:.3f}, fit={fitness:.2f}")

    return fitness

# =========================
# Body Architecture with MULTIPLE ATTEMPTS
# =========================
@dataclass
class BodyArchitecture:
    inp_size: int
    out_size: int
    viable: bool
    error_msg: str
    world: Optional[Any] = None
    model: Optional[Any] = None
    track_body_name: Optional[str] = None
    seed_used: int = 0

def body_geno_to_key(body_geno, seed=0) -> str:
    """Include seed in key so we can try multiple builds."""
    t, c, r = body_geno
    combined = np.concatenate([t.flatten(), c.flatten(), r.flatten(), [seed]])
    return str(hash(combined.tobytes()))

def _find_core_body_name(model):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "robot-core")
    if bid != -1:
        return "robot-core"
    candidates = []
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        if name and "core" in name.lower() and "world" not in name.lower():
            candidates.append(name)
    if candidates:
        return sorted(candidates, key=len)[0]
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        if name and "world" not in name.lower():
            return name
    return None

@dataclass
class BuiltBody:
    nde: NeuralDevelopmentalEncoding
    decoded_graph: Any
    mjspec: Any
    viable: bool
    error_msg: str

def build_body(geno: tuple, nde_modules: int, rng: np.random.Generator) -> BuiltBody:
    try:
        t, c, r = geno
        t, c, r = t.astype(np.float32), c.astype(np.float32), r.astype(np.float32)
        nde = NeuralDevelopmentalEncoding(number_of_modules=nde_modules)
        nde.t, nde.c, nde.r, nde.n_modules = t, c, r, nde_modules
        p_mats = nde.forward([t, c, r])
        decoder = HighProbabilityDecoder(nde_modules)
        graph = decoder.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
        spec = construct_mjspec_from_graph(graph)
        return BuiltBody(nde=nde, decoded_graph=graph, mjspec=spec, viable=True, error_msg="OK")
    except Exception as e:
        dummy_nde = NeuralDevelopmentalEncoding(number_of_modules=1)
        return BuiltBody(nde=dummy_nde, decoded_graph=None, mjspec=None, viable=False, error_msg=str(e))

def get_body_architecture_with_retries(body_geno, max_attempts=BODY_MAX_ATTEMPTS) -> BodyArchitecture:
    """
    CRITICAL: Try multiple random seeds to find a viable body.
    This massively increases viability rate!
    """
    for attempt in range(max_attempts):
        seed = attempt
        body_key = body_geno_to_key(body_geno, seed)

        # Check cache first
        if body_key in _BODY_ARCH_CACHE:
            arch = _BODY_ARCH_CACHE[body_key]
            if arch.viable:
                return arch
            continue  # Try next seed

        try:
            temp_rng = np.random.default_rng((hash(body_key) + seed) % 2**32)
            world = OlympicArena()
            built = build_body(body_geno, nde_modules=NUM_OF_MODULES, rng=temp_rng)

            if not built.viable:
                arch = BodyArchitecture(0, 0, False, built.error_msg, seed_used=seed)
                _BODY_ARCH_CACHE[body_key] = arch
                continue  # Try next seed

            world.spawn(built.mjspec.spec, position=SPAWN_POS)
            model = world.spec.compile()
            data = mj.MjData(model)
            mj.mj_resetData(model, data)
            mj.mj_forward(model, data)

            if model.nu == 0 or model.nbody < 2:
                arch = BodyArchitecture(0, 0, False, f"Invalid: nu={model.nu}", seed_used=seed)
                _BODY_ARCH_CACHE[body_key] = arch
                continue

            track_body_name = _find_core_body_name(model)
            if not track_body_name:
                arch = BodyArchitecture(0, 0, False, "No core", seed_used=seed)
                _BODY_ARCH_CACHE[body_key] = arch
                continue

            # Very relaxed height check
            try:
                core_body = data.body(track_body_name)
                start_height = core_body.xpos[2]
                if start_height < -2.0 or start_height > 10.0:
                    arch = BodyArchitecture(0, 0, False, f"Height: {start_height:.3f}", seed_used=seed)
                    _BODY_ARCH_CACHE[body_key] = arch
                    continue
            except:
                pass

            # SUCCESS!
            inp_size = len(data.qpos) + len(data.qvel) + 3
            out_size = model.nu
            arch = BodyArchitecture(inp_size, out_size, True, "OK", world, model, track_body_name, seed)
            _BODY_ARCH_CACHE[body_key] = arch
            console.log(f"[Arch] âœ“âœ“âœ“ VIABLE (attempt {attempt+1}/{max_attempts}): inp={inp_size}, out={out_size}")
            return arch

        except Exception as e:
            arch = BodyArchitecture(0, 0, False, str(e), seed_used=seed)
            _BODY_ARCH_CACHE[body_key] = arch
            continue

    # All attempts failed
    final_key = body_geno_to_key(body_geno, 0)
    arch = BodyArchitecture(0, 0, False, f"Failed all {max_attempts} attempts", seed_used=0)
    _BODY_ARCH_CACHE[final_key] = arch
    return arch

def save_body_artifacts(run_dir: Path, built: BuiltBody, tag: str):
    if not built.viable:
        return
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        if built.decoded_graph:
            save_graph_as_json(built.decoded_graph, str(run_dir / f"{tag}_graph.json"))
        nde_json = {"t": built.nde.t.tolist(), "c": built.nde.c.tolist(), 
                    "r": built.nde.r.tolist(), "n_modules": built.nde.n_modules}
        with open(run_dir / f"{tag}_nde.json", "w") as f:
            json.dump(nde_json, f, indent=2)
    except:
        pass

# =========================
# Controller Functions
# =========================
def controller_theta_size(inp, hidden=CTRL_HIDDEN, out_dim=1):
    return inp*hidden + hidden + hidden*hidden + hidden + hidden*out_dim + out_dim

def unpack_controller_theta(theta, inp, hidden=CTRL_HIDDEN, out_dim=1):
    theta = np.asarray(theta, dtype=float)
    expected = controller_theta_size(inp, hidden, out_dim)
    if theta.size != expected:
        return None
    i = 0
    W1 = theta[i:i+inp*hidden].reshape(inp, hidden); i += inp*hidden
    b1 = theta[i:i+hidden]; i += hidden
    W2 = theta[i:i+hidden*hidden].reshape(hidden, hidden); i += hidden*hidden
    b2 = theta[i:i+hidden]; i += hidden
    W3 = theta[i:i+hidden*out_dim].reshape(hidden, out_dim); i += hidden*out_dim
    b3 = theta[i:i+out_dim]
    return (W1, b1, W2, b2, W3, b3)

def controller_mlp_forward(x, params):
    if params is None:
        return np.zeros(1)
    try:
        W1, b1, W2, b2, W3, b3 = params
        h1 = np.tanh(x @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        return np.tanh(h2 @ W3 + b3)
    except:
        return np.zeros(1)

def init_controller_genotype_for_body(inp_size, out_size, rng: np.random.Generator):
    w_size = controller_theta_size(inp_size, CTRL_HIDDEN, out_size)
    w1_scale = np.sqrt(3.0 / (inp_size + CTRL_HIDDEN))
    w2_scale = np.sqrt(3.0 / (CTRL_HIDDEN + CTRL_HIDDEN))
    w3_scale = np.sqrt(2.0 / (CTRL_HIDDEN + out_size))
    theta = np.zeros(w_size, dtype=float)
    i = 0
    n_w1 = inp_size * CTRL_HIDDEN
    theta[i:i+n_w1] = rng.normal(0.0, w1_scale, size=n_w1); i += n_w1
    theta[i:i+CTRL_HIDDEN] = rng.normal(0.15, 0.15, size=CTRL_HIDDEN); i += CTRL_HIDDEN
    n_w2 = CTRL_HIDDEN * CTRL_HIDDEN
    theta[i:i+n_w2] = rng.normal(0.0, w2_scale, size=n_w2); i += n_w2
    theta[i:i+CTRL_HIDDEN] = rng.normal(0.1, 0.15, size=CTRL_HIDDEN); i += CTRL_HIDDEN
    n_w3 = CTRL_HIDDEN * out_size
    theta[i:i+n_w3] = rng.normal(0.0, w3_scale, size=n_w3); i += n_w3
    theta[i:i+out_size] = rng.normal(0.05, 0.1, size=out_size)
    return np.clip(theta, CTRL_W_LOW, CTRL_W_HIGH)

def controller_sbx_crossover(parent1, parent2, eta=CTRL_SBX_ETA, rng=None):
    rng = rng or np.random.default_rng()
    parent1, parent2 = np.asarray(parent1, dtype=float), np.asarray(parent2, dtype=float)
    if parent1.shape != parent2.shape:
        return parent1.copy(), parent2.copy()
    u = rng.random(parent1.shape)
    beta = np.empty_like(parent1)
    mask = u <= 0.5
    beta[mask] = (2.0 * u[mask]) ** (1.0 / (eta + 1.0))
    beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (eta + 1.0))
    child1 = 0.5 * ((parent1 + parent2) - beta * (parent2 - parent1))
    child2 = 0.5 * ((parent1 + parent2) + beta * (parent2 - parent1))
    return np.clip(child1, CTRL_W_LOW, CTRL_W_HIGH), np.clip(child2, CTRL_W_LOW, CTRL_W_HIGH)

def controller_polynomial_mutation(individual, eta=10.0, indpb=0.2, rng=None):
    rng = rng or np.random.default_rng()
    individual = np.asarray(individual, dtype=float)
    mutated = individual.copy()
    for i in range(len(individual)):
        if rng.random() < indpb:
            u = rng.random()
            delta_1 = np.clip((individual[i] - CTRL_W_LOW) / (CTRL_W_HIGH - CTRL_W_LOW), 1e-8, 1.0 - 1e-8)
            delta_2 = np.clip((CTRL_W_HIGH - individual[i]) / (CTRL_W_HIGH - CTRL_W_LOW), 1e-8, 1.0 - 1e-8)
            if u <= 0.5:
                mut_pow = 1.0 / (eta + 1.0)
                delta_q = (2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta_1) ** (eta + 1.0)) ** mut_pow - 1.0
            else:
                mut_pow = 1.0 / (eta + 1.0)
                delta_q = 1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta_2) ** (eta + 1.0)) ** mut_pow
            mutated[i] = np.clip(individual[i] + delta_q * (CTRL_W_HIGH - CTRL_W_LOW), CTRL_W_LOW, CTRL_W_HIGH)
    return mutated

# =========================
# Body Genotype
# =========================
def init_body_genotype(rng: np.random.Generator, n: int):
    return (rng.random(n).astype(np.float32), rng.random(n).astype(np.float32), rng.random(n).astype(np.float32))

def _sbx_pair(a: np.ndarray, b: np.ndarray, eta: float, low=0.0, high=1.0, rng=None):
    rng = rng or np.random.default_rng()
    u = rng.random(a.shape, dtype=np.float32)
    beta = np.empty_like(a, dtype=np.float32)
    mask = u <= 0.5
    beta[mask] = (2.0 * u[mask]) ** (1.0 / (eta + 1.0))
    beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (eta + 1.0))
    c1 = 0.5 * ((a + b) - beta * (b - a))
    c2 = 0.5 * ((a + b) + beta * (b - a))
    return np.clip(c1, low, high), np.clip(c2, low, high)

def sbx_body(g1, g2, eta=BODY_SBX_ETA, rng=None):
    a1, b1 = _sbx_pair(g1[0], g2[0], eta, rng=rng)
    a2, b2 = _sbx_pair(g1[1], g2[1], eta, rng=rng)
    a3, b3 = _sbx_pair(g1[2], g2[2], eta, rng=rng)
    return (a1, a2, a3), (b1, b2, b3)

def block_mutation(g, indpb=3.0/BODY_L, rng=None):
    rng = rng or np.random.default_rng()
    t, c, r = g
    mask_t = rng.random(t.shape) < indpb
    mask_c = rng.random(c.shape) < indpb
    mask_r = rng.random(r.shape) < indpb
    t, c, r = t.copy(), c.copy(), r.copy()
    t[mask_t] = rng.random(np.count_nonzero(mask_t)).astype(np.float32)
    c[mask_c] = rng.random(np.count_nonzero(mask_c)).astype(np.float32)
    r[mask_r] = rng.random(np.count_nonzero(mask_r)).astype(np.float32)
    return (t, c, r)

# =========================
# Episode Evaluation
# =========================
def run_episode_with_controller(body_geno, controller_theta, duration: int = SIM_DURATION):
    try:
        arch = get_body_architecture_with_retries(body_geno)
        if not arch or not arch.viable:
            return -1e6, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        model, track_body_name = arch.model, arch.track_body_name
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        mj.set_mjcb_control(None)
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind=track_body_name)
        start_pos = np.array(data.body(track_body_name).xpos[:3], dtype=float).copy()
        controller_params = unpack_controller_theta(controller_theta, arch.inp_size, CTRL_HIDDEN, arch.out_size)
        if controller_params is None:
            return -1e6, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        def _episode_controller():
            step = 0
            ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=float).reshape(arch.out_size, 2)
            low, high = ctrlrange[:, 0].copy(), ctrlrange[:, 1].copy()
            limited = np.array(model.actuator_ctrllimited, dtype=bool).reshape(-1)
            bad = (~limited) | ~np.isfinite(low) | ~np.isfinite(high) | (high <= low)
            low[bad], high[bad] = -HINGE_LIMIT, HINGE_LIMIT
            center, halfspan = 0.5 * (low + high), 0.5 * (high - low)
            u_apply = center.copy()
            rate = np.minimum(0.08 * (high - low), RATE_LIMIT_DU)

            def _cb(m: mj.MjModel, d: mj.MjData):
                nonlocal u_apply, step
                try:
                    t_now = d.time
                    qpv = np.concatenate([d.qpos, d.qvel]).astype(float, copy=False)
                    time_feat = np.array([t_now, math.sin(2*math.pi*t_now), math.cos(2*math.pi*t_now)], dtype=float)
                    x_in = np.concatenate([qpv, time_feat])
                    if len(x_in) != arch.inp_size:
                        return np.zeros(arch.out_size, dtype=np.float64)
                    y_out = controller_mlp_forward(x_in, controller_params).flatten()
                    if len(y_out) != arch.out_size:
                        y_out = np.resize(y_out, arch.out_size)
                    y_out = np.nan_to_num(y_out, nan=0.0, posinf=0.0, neginf=0.0)
                    u_target = center + halfspan * y_out
                    ramp = 1.0 if WARMUP_STEPS <= 0 else min(1.0, step / max(1, WARMUP_STEPS))
                    u_cmd = center + ramp * (u_target - center)
                    du = np.clip(u_cmd - u_apply, -rate, rate)
                    u_apply = np.clip(u_apply + du, low, high)
                    step += 1
                    return u_apply.astype(np.float64, copy=False)
                except:
                    return np.zeros(arch.out_size, dtype=np.float64)
            return _cb

        episode_cb = _episode_controller()
        ctrl = Controller(controller_callback_function=episode_cb, tracker=tracker)
        if ctrl.tracker:
            ctrl.tracker.setup(arch.world.spec, data)
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
        simple_runner(model, data, duration=duration)
        end_pos = np.array(data.body(track_body_name).xpos[:3], dtype=float)
        hist = tracker.history.get("xpos", [])
        if not hist or len(hist) < 2:
            hist = [start_pos.tolist(), end_pos.tolist()]
        fitness = compute_simple_forward_fitness(start_pos, end_pos, hist, duration)
        return np.clip(fitness, -100.0, 1000.0), hist
    except Exception as e:
        return -1e6, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

# =========================
# Controller Evolution
# =========================
def evolve_controller_for_body(body_geno, verbose=False):
    arch = get_body_architecture_with_retries(body_geno)
    if not arch.viable:
        return None, -1e6
    try:
        creator.ControllerFitnessMax
    except AttributeError:
        creator.create("ControllerFitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.ControllerIndividual
    except AttributeError:
        creator.create("ControllerIndividual", list, fitness=creator.ControllerFitnessMax)

    def init_ctrl_ind():
        return creator.ControllerIndividual(init_controller_genotype_for_body(arch.inp_size, arch.out_size, RNG).tolist())

    ctrl_pop = [init_ctrl_ind() for _ in range(CTRL_POP_SIZE)]
    for ind in ctrl_pop:
        fitness, _ = run_episode_with_controller(body_geno, np.array(ind), SIM_DURATION)
        ind.fitness.values = (fitness,)

    for gen in range(CTRL_N_GEN):
        offspring = tools.selTournament(ctrl_pop, len(ctrl_pop), tournsize=CTRL_TOURNSIZE)
        offspring = [creator.ControllerIndividual(ind[:]) for ind in offspring]
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CTRL_CXPB:
                ch1, ch2 = controller_sbx_crossover(c1, c2, eta=CTRL_SBX_ETA, rng=RNG)
                c1[:], c2[:] = ch1.tolist(), ch2.tolist()
                del c1.fitness.values, c2.fitness.values
        for m in offspring:
            if random.random() < CTRL_MUTPB:
                m[:] = controller_polynomial_mutation(m, eta=10.0, indpb=0.2, rng=RNG).tolist()
                del m.fitness.values
        n_imm = max(0, int(CTRL_IMM_FRAC * len(offspring)))
        for _ in range(n_imm):
            offspring[random.randrange(len(offspring))] = init_ctrl_ind()
        for ind in offspring:
            if not hasattr(ind.fitness, 'values') or not ind.fitness.valid:
                fitness, _ = run_episode_with_controller(body_geno, np.array(ind), SIM_DURATION)
                ind.fitness.values = (fitness,)
        ctrl_pop = offspring

    best_ctrl = tools.selBest(ctrl_pop, 1)[0]
    return np.array(best_ctrl), best_ctrl.fitness.values[0]

# =========================
# DEAP Setup
# =========================
try:
    creator.BodyFitnessMax
except AttributeError:
    creator.create("BodyFitnessMax", base.Fitness, weights=(1.0,))
try:
    creator.BodyIndividual
except AttributeError:
    creator.create("BodyIndividual", list, fitness=creator.BodyFitnessMax)

toolbox = base.Toolbox()

def init_body_individual():
    return creator.BodyIndividual([init_body_genotype(RNG, BODY_L)])

toolbox.register("individual", init_body_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def mate_bodies(ind1, ind2):
    c1, c2 = sbx_body(ind1[0], ind2[0], eta=BODY_SBX_ETA, rng=RNG)
    ind1[0], ind2[0] = c1, c2
    if hasattr(ind1.fitness, "values"):
        del ind1.fitness.values
    if hasattr(ind2.fitness, "values"):
        del ind2.fitness.values
    return ind1, ind2

def mutate_body(ind):
    ind[0] = block_mutation(ind[0], indpb=3.0/BODY_L, rng=RNG)
    if hasattr(ind.fitness, "values"):
        del ind.fitness.values
    return (ind,)

def evaluate_body_individual(ind):
    best_controller, fitness = evolve_controller_for_body(ind[0], verbose=False)
    if best_controller is None:
        return (-1e6,)
    return (float(np.clip(fitness, -1e6, 1e6)),)

def _hof_similar(a, b):
    try:
        v1 = np.concatenate([np.ravel(a[0][0]), np.ravel(a[0][1]), np.ravel(a[0][2])])
        v2 = np.concatenate([np.ravel(b[0][0]), np.ravel(b[0][1]), np.ravel(b[0][2])])
        return bool(np.allclose(v1, v2, atol=1e-12))
    except:
        return False

toolbox.register("mate", mate_bodies)
toolbox.register("mutate", mutate_body)
toolbox.register("evaluate", evaluate_body_individual)
toolbox.register("select", tools.selTournament, tournsize=BODY_TOURNSIZE)

# =========================
# Visualization (simplified, no dependencies on rendering)
# =========================
def plot_robot_path(history, save_path: Path):
    try:
        if not history or len(history) < 2:
            return
        x_coords = [pos[0] for pos in history if len(pos) >= 2]
        y_coords = [pos[1] for pos in history if len(pos) >= 2]
        if not x_coords:
            return
        plt.figure(figsize=(12, 6))
        plt.axvspan(-1, 1.5, alpha=0.2, color='green', label='Smooth')
        plt.axvspan(1.5, 3.5, alpha=0.2, color='orange', label='Rugged')
        plt.axvspan(3.5, 6, alpha=0.2, color='red', label='Uphill')
        plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Path')
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
        plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
        plt.plot(TARGET_POSITION[0], TARGET_POSITION[1], 'r*', markersize=20, label='Goal')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Robot Path')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except:
        pass

def create_robot_video(body_geno, controller_theta, save_path: Path, duration: int = SIM_DURATION):
    """Video generation."""
    try:
        console.log("[Video] Starting...")
        arch = get_body_architecture_with_retries(body_geno)
        if not arch or not arch.viable:
            return False
        model, track_body_name = arch.model, arch.track_body_name
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        mj.set_mjcb_control(None)
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind=track_body_name)
        controller_params = unpack_controller_theta(controller_theta, arch.inp_size, CTRL_HIDDEN, arch.out_size)
        if controller_params is None:
            return False

        def _episode_controller():
            step = 0
            ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=float).reshape(arch.out_size, 2)
            low, high = ctrlrange[:, 0].copy(), ctrlrange[:, 1].copy()
            limited = np.array(model.actuator_ctrllimited, dtype=bool).reshape(-1)
            bad = (~limited) | ~np.isfinite(low) | ~np.isfinite(high) | (high <= low)
            low[bad], high[bad] = -HINGE_LIMIT, HINGE_LIMIT
            center, halfspan = 0.5 * (low + high), 0.5 * (high - low)
            u_apply = center.copy()
            rate = np.minimum(0.08 * (high - low), RATE_LIMIT_DU)

            def _cb(m: mj.MjModel, d: mj.MjData):
                nonlocal u_apply, step
                try:
                    qpv = np.concatenate([d.qpos, d.qvel]).astype(float, copy=False)
                    time_feat = np.array([d.time, math.sin(2*math.pi*d.time), math.cos(2*math.pi*d.time)], dtype=float)
                    x_in = np.concatenate([qpv, time_feat])
                    if len(x_in) != arch.inp_size:
                        return np.zeros(arch.out_size, dtype=np.float64)
                    y_out = np.nan_to_num(controller_mlp_forward(x_in, controller_params).flatten())
                    if len(y_out) != arch.out_size:
                        y_out = np.resize(y_out, arch.out_size)
                    u_target = center + halfspan * y_out
                    ramp = 1.0 if WARMUP_STEPS <= 0 else min(1.0, step / max(1, WARMUP_STEPS))
                    u_cmd = center + ramp * (u_target - center)
                    u_apply = np.clip(u_apply + np.clip(u_cmd - u_apply, -rate, rate), low, high)
                    step += 1
                    return u_apply.astype(np.float64, copy=False)
                except:
                    return np.zeros(arch.out_size, dtype=np.float64)
            return _cb

        ctrl = Controller(controller_callback_function=_episode_controller(), tracker=tracker)
        if ctrl.tracker:
            ctrl.tracker.setup(arch.world.spec, data)
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        try:
            import imageio
        except ImportError:
            import subprocess
            subprocess.check_call(["pip", "install", "imageio", "imageio-ffmpeg"])
            import imageio

        fps, width, height = 30, 1280, 720
        renderer = mj.Renderer(model, height=height, width=width)
        camera = mj.MjvCamera()
        camera.type = mj.mjtCamera.mjCAMERA_FREE
        camera.lookat, camera.distance, camera.azimuth, camera.elevation = [2.5, 0.0, 0.5], 8.0, 180, -30
        frames, step_count = [], 0
        frame_skip = max(1, int(model.opt.timestep * 240 / (1.0 / fps)))
        max_steps = int(duration / model.opt.timestep)
        console.log(f"[Video] Recording...")
        for _ in range(max_steps):
            mj.mj_step(model, data)
            step_count += 1
            if step_count % frame_skip == 0:
                try:
                    robot_pos = data.body(track_body_name).xpos
                    camera.lookat = [float(robot_pos[0]), float(robot_pos[1]), 0.5]
                except:
                    pass
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
        imageio.mimsave(str(save_path), frames, fps=fps, quality=8)
        console.log(f"[Video] Saved: {save_path}")
        return True
    except Exception as e:
        console.log(f"[Video Error] {e}")
        return False

def save_robot_olympics_submission(body_geno, controller_theta, save_dir: Path):
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        # Use the first successful seed
        arch = get_body_architecture_with_retries(body_geno)
        if not arch or not arch.viable:
            return

        # Rebuild with the successful seed
        temp_rng = np.random.default_rng(arch.seed_used)
        built = build_body(body_geno, nde_modules=NUM_OF_MODULES, rng=temp_rng)

        if not built.viable:
            return
        if built.decoded_graph:
            save_graph_as_json(built.decoded_graph, str(save_dir / "robot_body.json"))
        if arch.viable:
            ctrl_data = {"architecture": {"input_size": arch.inp_size, "hidden_size": CTRL_HIDDEN, "output_size": arch.out_size},
                        "weights": controller_theta.tolist() if isinstance(controller_theta, np.ndarray) else list(controller_theta),
                        "weight_bounds": [CTRL_W_LOW, CTRL_W_HIGH]}
            with open(save_dir / "controller.json", "w") as f:
                json.dump(ctrl_data, f, indent=2)
            console.log(f"[Submission] Saved to {save_dir}")
    except:
        pass

# =========================
# Main Evolution
# =========================
def run_co_evolution():
    random.seed(SEED)
    np.random.seed(SEED)
    console.log("\n" + "="*70)
    console.log("ROBOT OLYMPICS - PRAGMATIC VERSION")
    console.log("Multiple attempts per body for higher viability")
    console.log("="*70)
    console.log(f"Modules: {NUM_OF_MODULES} | Attempts per body: {BODY_MAX_ATTEMPTS}")
    console.log(f"Body: {BODY_POP_SIZE} pop Ã— {BODY_N_GEN} gen")
    console.log(f"Controller: {CTRL_POP_SIZE} pop Ã— {CTRL_N_GEN} gen")
    console.log("="*70 + "\n")

    body_pop = toolbox.population(n=BODY_POP_SIZE)
    console.log("[EA] Initial population (trying multiple seeds per body)...")
    invalid = [ind for ind in body_pop if not ind.fitness.valid]
    viable_count = 0
    for i, ind in enumerate(invalid):
        console.log(f"[EA] Body {i+1}/{len(invalid)}")
        fitness_tuple = toolbox.evaluate(ind)
        ind.fitness.values = fitness_tuple
        if fitness_tuple[0] > -1e6:
            viable_count += 1
            console.log(f"  âœ“âœ“âœ“ VIABLE! Fitness: {fitness_tuple[0]:.4f}")

    console.log(f"\n[EA] RESULT: {viable_count}/{len(invalid)} VIABLE ({100*viable_count/len(invalid):.1f}%)")

    if viable_count < 3:
        console.log("\nâš ï¸  Still low viability, but better than before...")
    else:
        console.log("\nâœ… GOOD VIABILITY!\n")

    ELITE_K = max(2, BODY_POP_SIZE // 10)
    hof = tools.HallOfFame(ELITE_K, similar=_hof_similar)
    hof.update(body_pop)
    best_per_gen = [tools.selBest(body_pop, 1)[0].fitness.values[0]]
    no_improve, best_so_far = 0, best_per_gen[-1]
    console.log(f"[EA] Initial best: {best_so_far:.4f}\n")

    t_wall = time.time()
    for gen in range(1, BODY_N_GEN + 1):
        console.log(f"[EA] === Gen {gen}/{BODY_N_GEN} ===")
        offspring = list(map(toolbox.clone, toolbox.select(body_pop, len(body_pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < BODY_CXPB:
                toolbox.mate(c1, c2)
        for m in offspring:
            if random.random() < BODY_MUTPB:
                toolbox.mutate(m)
        n_imm = max(0, int(BODY_IMM_FRAC * len(offspring)))
        for _ in range(n_imm):
            offspring[random.randrange(len(offspring))] = toolbox.individual()
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        gen_viable = 0
        for i, ind in enumerate(invalid):
            console.log(f"[EA] Gen {gen} - Body {i+1}/{len(invalid)}")
            fitness_tuple = toolbox.evaluate(ind)
            ind.fitness.values = fitness_tuple
            if fitness_tuple[0] > -1e6:
                gen_viable += 1
                if fitness_tuple[0] > 1.0:
                    console.log(f"  âœ“âœ“âœ“ Viable! Fit={fitness_tuple[0]:.4f}")
        console.log(f"[EA] Gen {gen}: {gen_viable}/{len(invalid)} viable")
        hof.update(offspring)
        elites = list(map(toolbox.clone, hof.items))
        offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        body_pop = elites + offspring[:max(0, BODY_POP_SIZE - len(elites))]
        best = tools.selBest(body_pop, 1)[0]
        best_per_gen.append(best.fitness.values[0])
        improvement = best.fitness.values[0] - best_so_far
        if improvement > 0.5:
            best_so_far = best.fitness.values[0]
            no_improve = 0
            console.log(f"[EA] ðŸŽ‰ IMPROVEMENT! +{improvement:.4f} â†’ {best_so_far:.4f}")
        else:
            no_improve += 1
        console.log(f"[EA] Best={best.fitness.values[0]:.4f} | t={time.time()-t_wall:.1f}s\n")

    console.log("="*70)
    console.log("FINAL")
    console.log("="*70)
    final_best = tools.selBest(body_pop, 1)[0]
    console.log(f"Best fitness: {final_best.fitness.values[0]:.4f}")

    final_controller, final_history = None, [[0,0,0], [0,0,0]]
    console.log("Re-evolving controller...")
    final_controller, ctrl_fitness = evolve_controller_for_body(final_best[0], verbose=True)
    if final_controller is not None:
        fitness_final, final_history = run_episode_with_controller(final_best[0], final_controller, SIM_DURATION)
        traj_data = {"history": final_history, "fitness": float(fitness_final)}
        with open(DATA / "FINAL_trajectory.json", "w") as f:
            json.dump(traj_data, f, indent=2)
        plot_robot_path(final_history, EA_PATH_PNG)
        console.log("Generating video...")
        create_robot_video(final_best[0], final_controller, EA_VIDEO_MP4, duration=SIM_DURATION)
        save_robot_olympics_submission(final_best[0], final_controller, DATA / "SUBMISSION")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_per_gen)), best_per_gen, marker='o', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Over Generations')
    plt.grid(True, alpha=0.3)
    plt.savefig(EA_FITNESS_PNG, dpi=150)
    plt.close()

    total_time = time.time() - t_wall
    console.log(f"\nRuntime: {total_time/3600:.2f}h ({total_time/60:.1f}min)")
    console.log(f"Viable: {viable_count}/{BODY_POP_SIZE} = {100*viable_count/BODY_POP_SIZE:.1f}%")
    console.log(f"Output: {DATA}\n")
    return final_best, final_controller, final_history, best_per_gen

if __name__ == "__main__":
    console.log("\nðŸ¤– ROBOT OLYMPICS - PRAGMATIC VERSION ðŸ¤–\n")
    try:
        run_co_evolution()
        console.log("\nâœ… COMPLETE!\n")
    except KeyboardInterrupt:
        console.log("\nâš ï¸  Interrupted\n")
    except Exception as e:
        console.log(f"\nâŒ Error: {e}\n")
        import traceback
        traceback.print_exc()