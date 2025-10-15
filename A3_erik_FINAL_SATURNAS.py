from __future__ import annotations

# A3_erik_PROPER_COEVOLUTION.py
# PROPER CO-EVOLUTION WITH ELITISM
#
# KEY FIXES (v2):
# 1. ELITISM: Both body and controller EAs preserve best individuals
# 2. CONTROLLER CACHING: Never retrain (use cache forever for consistency)
# 3. LONGER TRAINING: 50×50 = 2500 evaluations per body
# 4. HIGHER VIABILITY: fitness > 6.0 (≈1.0m minimum movement)
# 5. REDUCED MUTATION: 0.3 instead of 0.5 (less destructive)
# 6. REDUCED IMMIGRATION: 10% instead of 25% (less disruption)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mujoco as mj
import numpy as np
from deap import base, creator, tools
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments import OlympicArena

console = Console()

# =========================
# PROPER CO-EVOLUTION PARAMETERS
# =========================
SCRIPT_NAME = "A3_erik_proper_coevo"
CWD = Path(".").resolve()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)
N_ATTEMPTS = 5
# Body EA parameters - SMALLER population, better quality
BODY_POP_SIZE = 10  #25         # Slightly reduced for faster iterations
BODY_N_GEN = 10              # More generations
BODY_TOURNSIZE = 3
BODY_CXPB = 0.8
BODY_MUTPB = 0.25             # REDUCED from 0.5 - less destructive mutations
BODY_SBX_ETA = 10.0
BODY_ELITE_K = 5             # Increased from 3 - preserve more good solutions

# Controller EA parameters - Research-backed configuration
CTRL_POP_SIZE = 5   #25        # Research minimum: 40-50
CTRL_N_GEN = 8    # 20          # Total: 50×50 = 2,500 evals per body
CTRL_TOURNSIZE = 3
CTRL_CXPB = 0.4              # REDUCED from 0.8 (crossover destructive for weights)
CTRL_MUTPB = 0.15             # INCREASED from 0.2 (mutation primary operator)
CTRL_SBX_ETA = 10.0
CTRL_ELITE_K = 3             # Preserve best controllers

# Controller caching policy - Controllers represent learned behavior
RETRAIN_EVERY_N_GEN = 999999 # Cache forever: Once trained, controller is the learned behavior
VIABILITY_THRESHOLD = 2.0    # REDUCED from 4.0: More lenient for initial diversity (≈0.46m)

# Sim settings
SIM_DURATION = 15.0
WARMUP_STEPS = 50
STALL_WINDOW_SEC = 2.5
STALL_EPS = 0.02  # 20mm movement required to avoid early termination
RATE_LIMIT_FRAC = 0.08

SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]
TRACK_LENGTH = float(TARGET_POSITION[0] - SPAWN_POS[0])

# Body encoding
BODY_L = 64
NUM_OF_MODULES = 30

# Controller encoding
CTRL_HIDDEN = 16
CTRL_W_LOW, CTRL_W_HIGH = -3.0, 3.0

# Outputs
EA_FITNESS_PNG = DATA / "ea_best_over_generations.png"
PROBE_STEPS = 600

VERBOSE = False
_CTRL_UNLIM_SCALE = np.pi / 2

# =========================
# Types & caching with GENERATION TRACKING
# =========================
@dataclass
class BuiltBody:
    nde: NeuralDevelopmentalEncoding
    decoded_graph: Any
    mjspec: Any
    viable: bool
    error_msg: str

@dataclass
class BodyArchitecture:
    inp_size: int
    out_size: int
    viable: bool
    error_msg: str
    world: Optional[OlympicArena] = None
    model: Optional[mj.MjModel] = None
    track_body_name: Optional[str] = None

@dataclass
class ControllerCache:
    theta: np.ndarray
    fitness: float
    trained_at_generation: int  # NEW: Track when this was trained

_BODY_ARCH_CACHE: dict[str, BodyArchitecture] = {}
_BEST_CTRL_CACHE: dict[str, ControllerCache] = {}  # Now stores ControllerCache objects

def body_geno_to_key(geno) -> str:
    t, c, r = geno
    return (t.tobytes() + c.tobytes() + r.tobytes()).hex()

# =========================
# NDE body building
# =========================
def build_body(geno: tuple[np.ndarray, np.ndarray, np.ndarray], nde_modules: int, rng: np.random.Generator) -> BuiltBody:
    try:
        t, c, r = geno
        t = np.asarray(t, dtype=np.float32)
        c = np.asarray(c, dtype=np.float32)
        r = np.asarray(r, dtype=np.float32)
        nde = NeuralDevelopmentalEncoding(number_of_modules=nde_modules)
        nde.t, nde.c, nde.r = t, c, r
        nde.n_modules = nde_modules
        p_mats = nde.forward([t, c, r])
        decoder = HighProbabilityDecoder(nde_modules)
        graph = decoder.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
        spec = construct_mjspec_from_graph(graph)
        return BuiltBody(nde=nde, decoded_graph=graph, mjspec=spec, viable=True, error_msg="OK")
    except Exception as e:
        return BuiltBody(nde=None, decoded_graph=None, mjspec=None, viable=False, error_msg=str(e))

def save_body_artifacts(run_dir: Path, built: BuiltBody, tag: str):
    if not built.viable:
        return
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

def _find_core_body_name(model: mj.MjModel) -> Optional[str]:
    try:
        jids = model.actuator_trnid[:, 0]
        for jid in jids:
            jid = int(jid)
            if 0 <= jid < model.njnt:
                bid = int(model.jnt_bodyid[jid])
                name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
                if name:
                    return name
    except Exception:
        pass
    for nm in ("robot-core", "robot_core", "core"):
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, nm)
        if bid != -1:
            return nm
    if model.nbody >= 2:
        return mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, 1)
    return None

def get_body_architecture(body_geno_key: str, body_geno) -> BodyArchitecture:
    if body_geno_key in _BODY_ARCH_CACHE:
        return _BODY_ARCH_CACHE[body_geno_key]
    try:
        world = OlympicArena()
        built = build_body(body_geno, nde_modules=NUM_OF_MODULES, rng=RNG)
        if not built.viable:
            arch = BodyArchitecture(0, 0, False, built.error_msg)
            _BODY_ARCH_CACHE[body_geno_key] = arch
            return arch
        world.spawn(built.mjspec.spec, position=SPAWN_POS)
        model = world.spec.compile()
        data = mj.MjData(model)
        mj.mj_resetData(model, data); mj.mj_forward(model, data)
        if model.nu <= 0 or model.nv <= 0:
            arch = BodyArchitecture(0, 0, False, f"Invalid model: nu={model.nu}, nv={model.nv}")
            _BODY_ARCH_CACHE[body_geno_key] = arch
            return arch

        track_body_name = _find_core_body_name(model)
        if not track_body_name:
            arch = BodyArchitecture(0, 0, False, "No trackable body")
            _BODY_ARCH_CACHE[body_geno_key] = arch
            return arch

        inp = len(data.qpos) + len(data.qvel) + 3
        out = model.nu

        arch = BodyArchitecture(inp, out, True, "OK", world, model, track_body_name)
        _BODY_ARCH_CACHE[body_geno_key] = arch
        return arch
    except Exception as e:
        arch = BodyArchitecture(0, 0, False, str(e))
        _BODY_ARCH_CACHE[body_geno_key] = arch
        return arch

# =========================
# Controller (MLP)
# =========================
def controller_theta_size(inp: int, hidden: int, out_dim: int) -> int:
    return inp*hidden + hidden + hidden*hidden + hidden + hidden*out_dim + out_dim

def unpack_controller_theta(theta, inp, hidden, out_dim):
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

def mlp_forward(x, params):
    W1, b1, W2, b2, W3, b3 = params
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    y  = h2 @ W3 + b3
    return y

# =========================
# EXPONENTIAL FITNESS
# =========================
def _estimate_ttf(x_hist: list[float], t_hist: list[float], L: float) -> float:
    for i in range(1, len(x_hist)):
        if x_hist[i-1] < L <= x_hist[i]:
            dx = x_hist[i] - x_hist[i-1]
            dt = t_hist[i] - t_hist[i-1]
            if dx > 0:
                a = (L - x_hist[i-1]) / dx
                return t_hist[i-1] + a * dt
    return t_hist[-1]

def compute_race_fitness_exponential(x_hist: list[float], t_hist: list[float], track_length: float) -> tuple[float, dict]:
    if len(x_hist) < 2:
        return -1e6, dict(finished=0, dist=0.0, speed=0.0, ttf=None)

    dist = max(0.0, float(x_hist[-1] - x_hist[0]))
    dur = max(1e-6, float(t_hist[-1] - t_hist[0]))
    avg_speed = dist / dur

    finished = dist >= (track_length - 0.05)

    if finished:
        ttf = _estimate_ttf(x_hist, t_hist, track_length)
        finish_bonus = 1_000_000.0
        speed_score = (track_length / max(1e-6, ttf)) * 100.0
        fitness = finish_bonus + speed_score
        return fitness, dict(finished=1, dist=track_length, speed=track_length/ttf, ttf=ttf)
    else:
        base_fitness = (dist ** 1.4) * 6.0
        speed_bonus = avg_speed * 0.2
        fitness = base_fitness + speed_bonus
        return fitness, dict(finished=0, dist=dist, speed=avg_speed, ttf=None)

def scale_to_ctrlrange(model: mj.MjModel, u_raw: np.ndarray) -> np.ndarray:
    u_raw = np.clip(u_raw, -1.0, 1.0)
    cr = model.actuator_ctrlrange
    limited = getattr(model, "actuator_ctrllimited", None)
    if limited is None:
        limited = np.zeros(model.nu, dtype=int)
    span = cr[:, 1] - cr[:, 0]
    center = 0.5 * (cr[:, 0] + cr[:, 1])
    amp   = np.where(limited.astype(bool), 0.5 * span, _CTRL_UNLIM_SCALE)
    bias  = np.where(limited.astype(bool), center, 0.0)
    return bias + amp * u_raw

def apply_warmup_and_rate_limit(model: mj.MjModel, u_target: np.ndarray, u_prev: np.ndarray,
                                step_idx: int, warmup_steps: int = WARMUP_STEPS,
                                rate_frac: float = RATE_LIMIT_FRAC) -> np.ndarray:
    cr = model.actuator_ctrlrange
    span = (cr[:, 1] - cr[:, 0])
    du_lim = rate_frac * np.where(span > 0.0, span, _CTRL_UNLIM_SCALE)
    ramp = 1.0 if step_idx >= warmup_steps else float(step_idx) / max(1, warmup_steps)
    u_ramped = u_prev + ramp * (u_target - u_prev)
    du = np.clip(u_ramped - u_prev, -du_lim, du_lim)
    return u_prev + du

# =========================
# Episode runners & helpers
# =========================
def _body_x(model: mj.MjModel, data: mj.MjData, name: Optional[str]) -> float:
    try:
        if name:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                return float(data.xpos[bid, 0])
        if model.nbody >= 2:
            return float(data.xpos[1, 0])
    except Exception:
        pass
    return 0.0

def _body_xy(model: mj.MjModel, data: mj.MjData, name: Optional[str]) -> tuple[float, float]:
    """Return (x, y) of the tracked body; falls back to body 1; (0,0) on failure."""
    try:
        if name:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                return float(data.xpos[bid, 0]), float(data.xpos[bid, 1])
        if model.nbody >= 2:
            return float(data.xpos[1, 0]), float(data.xpos[1, 1])
    except Exception:
        pass
    return 0.0, 0.0

def run_episode_with_controller(body_arch: BodyArchitecture, theta: np.ndarray, steps: int = 800) -> tuple[float, list[list[float]]]:
    """
    Runs one episode and returns:
      - fitness (float)
      - full absolute coordinate history: [[x, y, t], ...] (time included as 3rd value)
    """
    if not body_arch or not body_arch.viable:
        return -1e6, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    model = body_arch.model
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    mj.set_mjcb_control(None)

    INP = body_arch.inp_size
    OUT = body_arch.out_size
    params = unpack_controller_theta(theta, INP, CTRL_HIDDEN, OUT)
    if params is None:
        return -1e6, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # Histories
    x_hist_rel: list[float] = []
    t_hist: list[float] = []
    path_history: list[list[float]] = []  # [[x_abs, y_abs, t], ...]

    u_prev = np.zeros(OUT, dtype=float)
    start_x, start_y = _body_xy(model, data, body_arch.track_body_name)

    dt = float(model.opt.timestep)
    max_steps = min(steps, int(SIM_DURATION / max(1e-6, dt)))
    last_progress_check_t = 0.0

    for step_idx in range(max_steps):
        tsec = float(data.time)
        tf = np.array([tsec, math.sin(2 * math.pi * tsec), math.cos(2 * math.pi * tsec)], dtype=float)
        obs = np.concatenate([np.array(data.qpos, dtype=float), np.array(data.qvel, dtype=float), tf])

        u_nn = mlp_forward(obs, params)
        u_raw = np.tanh(u_nn)
        u_cmd = scale_to_ctrlrange(model, u_raw)
        u_apply = apply_warmup_and_rate_limit(model, u_cmd, u_prev, step_idx)
        data.ctrl[:] = u_apply
        u_prev = u_apply

        mj.mj_step(model, data)

        # Absolute positions for plotting; relative x for fitness/stall checks
        x_abs, y_abs = _body_xy(model, data, body_arch.track_body_name)
        x_hist_rel.append(x_abs - start_x)
        t_hist.append(float(data.time))
        path_history.append([float(x_abs), float(y_abs), float(data.time)])

        # Early stop on reaching goal
        if x_hist_rel[-1] >= (TRACK_LENGTH - 1e-2):
            break

        # Stall detection
        if t_hist[-1] - last_progress_check_t >= STALL_WINDOW_SEC:
            t_goal = t_hist[-1] - STALL_WINDOW_SEC
            j = 0
            while j < len(t_hist) - 1 and t_hist[j + 1] < t_goal:
                j += 1
            last_progress_check_t = t_hist[-1]  # Update checkpoint BEFORE break
            if (x_hist_rel[-1] - x_hist_rel[j]) < STALL_EPS:
                break

    fitness, _ = compute_race_fitness_exponential(x_hist_rel, t_hist, TRACK_LENGTH)

    # Return full absolute coordinate history (with time included as third element)
    return float(fitness), path_history

# =========================
# CONTROLLER EA WITH RE-TRAINING SUPPORT
# =========================
def controller_sbx_crossover(parent1, parent2, eta=CTRL_SBX_ETA, rng=None):
    rng = rng or np.random.default_rng()
    a = np.asarray(parent1, dtype=float)
    b = np.asarray(parent2, dtype=float)
    if a.shape != b.shape:
        return a.copy(), b.copy()
    u = rng.random(a.shape)
    beta = np.empty_like(a)
    mask = u <= 0.5
    beta[mask] = (2.0 * u[mask]) ** (1.0 / (eta + 1.0))
    beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (eta + 1.0))
    c1 = 0.5 * ((a + b) - beta * (b - a))
    c2 = 0.5 * ((a + b) + beta * (b - a))
    return np.clip(c1, CTRL_W_LOW, CTRL_W_HIGH), np.clip(c2, CTRL_W_LOW, CTRL_W_HIGH)

def controller_polynomial_mutation(individual, eta=12.0, indpb=0.15, rng=None):
    rng = rng or np.random.default_rng()
    x = np.asarray(individual, dtype=float).copy()
    for i in range(x.size):
        if rng.random() < indpb:
            u = rng.random()
            delta1 = (x[i] - CTRL_W_LOW) / (CTRL_W_HIGH - CTRL_W_LOW)
            delta2 = (CTRL_W_HIGH - x[i]) / (CTRL_W_HIGH - CTRL_W_LOW)
            delta1 = np.clip(delta1, 1e-8, 1.0 - 1e-8)
            delta2 = np.clip(delta2, 1e-8, 1.0 - 1e-8)
            if u <= 0.5:
                delta_q = (2*u + (1 - 2*u) * (1 - delta1) ** (eta + 1)) ** (1/(eta+1)) - 1
            else:
                delta_q = 1 - (2*(1-u) + 2*(u-0.5) * (1 - delta2) ** (eta + 1)) ** (1/(eta+1))
            x[i] = np.clip(x[i] + delta_q * (CTRL_W_HIGH - CTRL_W_LOW), CTRL_W_LOW, CTRL_W_HIGH)
    return x

def init_controller_genotype_for_body(inp_size, out_size, rng: np.random.Generator):
    w_size = controller_theta_size(inp_size, CTRL_HIDDEN, out_size)
    theta = rng.normal(0.0, 0.5, size=w_size).astype(float)
    return np.clip(theta, CTRL_W_LOW, CTRL_W_HIGH)

def evolve_controller_for_body(body_geno, current_generation: int, verbose=False):
    """
    Controller EA with LONG training (50×50 = 2500 evals)

    KEY CHANGES:
    - ELITISM: Preserves top 3 controllers each generation
    - CACHING: Never retrains (RETRAIN_EVERY_N_GEN = 999999)
    - Uses cached result if available
    """
    key = body_geno_to_key(body_geno)
    arch = get_body_architecture(key, body_geno)
    if not arch.viable:
        if verbose:
            console.log(f"[CtrlEA] Body not viable: {arch.error_msg}")
        return None, -1e6

    # Check if we have a cached controller that's still fresh
    if key in _BEST_CTRL_CACHE:
        cached = _BEST_CTRL_CACHE[key]
        age = current_generation - cached.trained_at_generation
        if age < RETRAIN_EVERY_N_GEN:
            if verbose:
                console.log(f"[CtrlEA] Using cached controller (age={age}gen, fit={cached.fitness:.1f})")
            return cached.theta, cached.fitness

    # Need to train (either no cache, or cache is stale)
    if verbose:
        if key in _BEST_CTRL_CACHE:
            console.log(f"[CtrlEA] Re-training stale controller (age={current_generation - _BEST_CTRL_CACHE[key].trained_at_generation}gen)")
        else:
            console.log(f"[CtrlEA] Training new controller ({CTRL_POP_SIZE}×{CTRL_N_GEN}={CTRL_POP_SIZE*CTRL_N_GEN} evals)")

    INP, OUT = arch.inp_size, arch.out_size
    theta_size = controller_theta_size(INP, CTRL_HIDDEN, OUT)

    # Initialize controller population
    pop = [init_controller_genotype_for_body(INP, OUT, RNG) for _ in range(CTRL_POP_SIZE)]

    # Evaluate initial population
    fitness_vals = []
    for theta in pop:
        fit, _ = run_episode_with_controller(arch, theta, steps=PROBE_STEPS)
        fitness_vals.append(fit)

    best_idx = int(np.argmax(fitness_vals))
    best_theta = pop[best_idx].copy()
    best_fit = fitness_vals[best_idx]

    # Evolution loop
    for gen in range(CTRL_N_GEN):
        # Selection
        selected_indices = []
        for _ in range(CTRL_POP_SIZE):
            contestants = RNG.choice(CTRL_POP_SIZE, size=CTRL_TOURNSIZE, replace=False)
            winner = contestants[np.argmax([fitness_vals[i] for i in contestants])]
            selected_indices.append(winner)
        offspring = [pop[i].copy() for i in selected_indices]

        # Crossover
        for i in range(0, CTRL_POP_SIZE - 1, 2):
            if RNG.random() < CTRL_CXPB:
                offspring[i], offspring[i+1] = controller_sbx_crossover(offspring[i], offspring[i+1], rng=RNG)

        # Mutation
        for i in range(CTRL_POP_SIZE):
            if RNG.random() < CTRL_MUTPB:
                offspring[i] = controller_polynomial_mutation(offspring[i], rng=RNG)

        # Evaluate offspring
        new_fitness_vals = []
        for theta in offspring:
            fit, _ = run_episode_with_controller(arch, theta, steps=PROBE_STEPS)
            new_fitness_vals.append(fit)

        # ELITISM: Replace worst offspring with elites (not first 3!)
        elite_indices = np.argsort(fitness_vals)[-CTRL_ELITE_K:]
        elite_controllers = [pop[i].copy() for i in elite_indices]
        elite_fits = [fitness_vals[i] for i in elite_indices]

        worst_indices = np.argsort(new_fitness_vals)[:CTRL_ELITE_K]
        for i, worst_idx in enumerate(worst_indices):
            offspring[worst_idx] = elite_controllers[i]
            new_fitness_vals[worst_idx] = elite_fits[i]

        # Update population
        pop = offspring
        fitness_vals = new_fitness_vals

        # Track best
        gen_best_idx = int(np.argmax(fitness_vals))
        if fitness_vals[gen_best_idx] > best_fit:
            best_theta = pop[gen_best_idx].copy()
            best_fit = fitness_vals[gen_best_idx]

    # Cache with generation timestamp
    _BEST_CTRL_CACHE[key] = ControllerCache(
        theta=best_theta,
        fitness=best_fit,
        trained_at_generation=current_generation
    )

    return best_theta, best_fit

# =========================
# Distance probing
# =========================
def probe_best_metrics(body_arch: BodyArchitecture, theta: np.ndarray) -> tuple[float, float]:
    if not body_arch.viable:
        return 0.0, TRACK_LENGTH

    fit, path_data = run_episode_with_controller(body_arch, theta, steps=PROBE_STEPS)
    if len(path_data) >= 2:
        dist = abs(path_data[-1][0] - path_data[0][0])
    else:
        dist = 0.0

    remaining = max(0.0, TRACK_LENGTH - dist)
    return dist, remaining

# =========================
# BODY EA EVALUATION
# =========================
def evaluate_body_genotype(body_geno, current_generation: int):
    """
    Evaluate a body by:
    1. Building the morphology
    2. Training/retrieving controller (with re-training if stale)
    3. Returning fitness
    """
    theta, fit = evolve_controller_for_body(body_geno, current_generation, verbose=VERBOSE)
    if theta is None:
        return (-1e6,)

    # Quick viability check
    if fit < VIABILITY_THRESHOLD:
        return (-1e6,)

    return (fit,)

# =========================
# BODY EA OPERATORS
# =========================
def body_sbx_crossover(parent1, parent2, eta=BODY_SBX_ETA, rng=None):
    rng = rng or np.random.default_rng()
    t1, c1, r1 = parent1
    t2, c2, r2 = parent2

    def sbx_1d(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        u = rng.random(a.shape)
        beta = np.empty_like(a)
        mask = u <= 0.5
        beta[mask] = (2.0 * u[mask]) ** (1.0 / (eta + 1.0))
        beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (eta + 1.0))
        c1 = 0.5 * ((a + b) - beta * (b - a))
        c2 = 0.5 * ((a + b) + beta * (b - a))
        return np.clip(c1, 0.0, 1.0), np.clip(c2, 0.0, 1.0)

    t_o1, t_o2 = sbx_1d(t1, t2)
    c_o1, c_o2 = sbx_1d(c1, c2)
    r_o1, r_o2 = sbx_1d(r1, r2)

    return (t_o1, c_o1, r_o1), (t_o2, c_o2, r_o2)

def body_polynomial_mutation(individual, eta=12.0, indpb=1.0/BODY_L, rng=None):
    rng = rng or np.random.default_rng()
    t, c, r = individual

    def mutate_1d(x):
        x = np.asarray(x, dtype=float).copy()
        for i in range(x.size):
            if rng.random() < indpb:
                u = rng.random()
                delta1 = x[i]
                delta2 = 1.0 - x[i]
                delta1 = np.clip(delta1, 1e-8, 1.0 - 1e-8)
                delta2 = np.clip(delta2, 1e-8, 1.0 - 1e-8)
                if u <= 0.5:
                    delta_q = (2*u + (1 - 2*u) * (1 - delta1) ** (eta + 1)) ** (1/(eta+1)) - 1
                else:
                    delta_q = 1 - (2*(1-u) + 2*(u-0.5) * (1 - delta2) ** (eta + 1)) ** (1/(eta+1))
                x[i] = np.clip(x[i] + delta_q, 0.0, 1.0)
        return x

    return (mutate_1d(t), mutate_1d(c), mutate_1d(r))

def init_body_genotype(rng: np.random.Generator):
    t = rng.random(BODY_L).astype(np.float32)
    c = rng.random(BODY_L).astype(np.float32)
    r = rng.random(BODY_L).astype(np.float32)
    return (t, c, r)

# =========================
# DEAP SETUP
# =========================
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", lambda: creator.Individual([init_body_genotype(RNG)]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# NOTE: evaluate needs current_generation parameter - we'll handle this in the main loop
toolbox.register("select", tools.selTournament, tournsize=BODY_TOURNSIZE)
toolbox.register("mate", body_sbx_crossover, rng=RNG)
toolbox.register("mutate", body_polynomial_mutation, rng=RNG)

def _hof_similar(ind1, ind2) -> bool:
    t1, c1, r1 = ind1[0]
    t2, c2, r2 = ind2[0]
    return (np.allclose(t1, t2, atol=1e-6) and
            np.allclose(c1, c2, atol=1e-6) and
            np.allclose(r1, r2, atol=1e-6))

# =========================
# MAIN EA LOOP
# =========================
def run_proper_coevolution_ea():
    console.log(f"[PROPER CO-EVOLUTION WITH ELITISM] Starting...")
    console.log(f"  Body pop: {BODY_POP_SIZE}, Controller: {CTRL_POP_SIZE}×{CTRL_N_GEN}={CTRL_POP_SIZE*CTRL_N_GEN} evals")
    console.log(f"  Body elitism: {BODY_ELITE_K}, Controller elitism: {CTRL_ELITE_K}")
    console.log(f"  Viability threshold: {VIABILITY_THRESHOLD:.1f} fitness (≈{(VIABILITY_THRESHOLD/6.0)**(1/1.4):.2f}m)")

    # Initialize population
    pop = toolbox.population(n=BODY_POP_SIZE)

    # Evaluate Gen 0
    console.log(f"\n[Gen 0] Evaluating {len(pop)} bodies...")
    t_start = time.time()
    for idx, ind in enumerate(pop):
        if (idx + 1) % 5 == 0:
            console.log(f"  Progress: {idx+1}/{len(pop)}")
        ind.fitness.values = evaluate_body_genotype(ind[0], current_generation=0)
    t_elapsed = time.time() - t_start
    console.log(f"[Gen 0] Complete in {t_elapsed/60:.1f} min")

    # Hall of fame
    hof = tools.HallOfFame(BODY_ELITE_K, similar=_hof_similar)
    hof.update(pop)

    best_per_gen = [max(ind.fitness.values[0] for ind in pop)]
    dist_per_gen = []

    gen0_fits = [ind.fitness.values[0] for ind in pop]
    mean_per_gen = [float(np.mean(gen0_fits))]
    std_per_gen  = [float(np.std(gen0_fits))]
    zbest_per_gen = [float((best_per_gen[0] - mean_per_gen[0]) / std_per_gen[0]) if std_per_gen[0] > 0 else 0.0]


    # Initial telemetry
    best0 = max(pop, key=lambda x: x.fitness.values[0])
    dist0 = 0.0
    try:
        built0 = build_body(best0[0], nde_modules=NUM_OF_MODULES, rng=RNG)
        save_body_artifacts(DATA, built0, tag="gen_000_best")

        best_key = body_geno_to_key(best0[0])
        best_arch = _BODY_ARCH_CACHE.get(best_key)
        if best_arch and best_arch.viable:
            cached = _BEST_CTRL_CACHE.get(best_key)
            if cached:
                dist0, _ = probe_best_metrics(best_arch, cached.theta)
    except Exception:
        pass

    dist_per_gen.append(dist0)
    progress = (dist_per_gen[0] / TRACK_LENGTH) * 100
    console.log(f"[Gen 0] best_fit={best_per_gen[0]:.1f}, dist={dist_per_gen[0]:.2f}m ({progress:.1f}%)\n")

    best_so_far = best_per_gen[0]
    no_improve = 0

    for gen in range(1, BODY_N_GEN + 1):
        console.log(f"[Gen {gen}] Starting...")
        t_gen_start = time.time()

        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if RNG.random() < BODY_CXPB:
                new_g1, new_g2 = toolbox.mate(c1[0], c2[0])
                c1[0] = new_g1
                c2[0] = new_g2
                del c1.fitness.values
                del c2.fitness.values

        # Mutation
        adapt_mutpb = BODY_MUTPB
        if no_improve >= 3:
            adapt_mutpb = min(0.6, BODY_MUTPB * 1.5)
        if no_improve >= 5:
            adapt_mutpb = min(0.8, BODY_MUTPB * 2.0)

        for ind in offspring:
            if RNG.random() < adapt_mutpb:
                ind[0] = toolbox.mutate(ind[0])
                del ind.fitness.values

        # Immigration if stagnating
        if no_improve >= 3:
            n_immigrants = max(2, int(0.10 * BODY_POP_SIZE))  # REDUCED from 25% to 10%
            console.log(f"  [Diversity] Adding {n_immigrants} immigrants (stagnation={no_improve}, mutpb={adapt_mutpb:.2f})")
            for _ in range(n_immigrants):
                offspring.append(toolbox.individual())

        # Evaluate invalids
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        console.log(f"  Evaluating {len(invalids)} new bodies...")
        for idx, ind in enumerate(invalids):
            if (idx + 1) % 5 == 0:
                console.log(f"    Progress: {idx+1}/{len(invalids)}")
            ind.fitness.values = evaluate_body_genotype(ind[0], current_generation=gen)

        # ELITISM: Replace population while keeping best individuals
        combined = pop + offspring
        # Filter out invalid fitness values (but be more lenient to avoid empty population)
        valid_combined = [ind for ind in combined if ind.fitness.valid and ind.fitness.values[0] > -5e5]

        # If we have enough valid individuals, use them
        if len(valid_combined) >= BODY_POP_SIZE:
            # Sort by fitness (best first)
            valid_combined.sort(key=lambda x: x.fitness.values[0], reverse=True)
            # Keep top BODY_POP_SIZE individuals
            pop[:] = valid_combined[:BODY_POP_SIZE]
        elif len(valid_combined) > 0:
            # If we have some valid individuals but not enough, keep them all
            valid_combined.sort(key=lambda x: x.fitness.values[0], reverse=True)
            pop[:] = valid_combined
            console.log(f"  [WARNING] Only {len(valid_combined)} viable bodies found, continuing with reduced population")
        else:
            # Emergency: no valid individuals, keep old population
            console.log(f"  [WARNING] No viable offspring, keeping old population")
            # pop stays the same

        # Update hall of fame
        if len(pop) > 0:
            hof.update(pop)

        # Track best
        if len(pop) > 0:
            gen_best_fit = max(ind.fitness.values[0] for ind in pop)
        else:
            console.log(f"  [ERROR] Population is empty! Breaking...")
            break
        best_per_gen.append(gen_best_fit)

        gen_fits = [ind.fitness.values[0] for ind in pop]
        g_mean = float(np.mean(gen_fits)) if gen_fits else float("nan")
        g_std  = float(np.std(gen_fits))  if gen_fits else float("nan")
        g_z    = float((gen_best_fit - g_mean) / g_std) if (gen_fits and g_std > 0) else 0.0
        mean_per_gen.append(g_mean)
        std_per_gen.append(g_std)
        zbest_per_gen.append(g_z)

        if gen_best_fit > best_so_far + 0.1:
            console.log(f"  [NEW BEST] fit={gen_best_fit:.1f} 🎯")
            best_so_far = gen_best_fit
            no_improve = 0
        else:
            no_improve += 1

        # Distance telemetry
        gen_best = max(pop, key=lambda x: x.fitness.values[0])
        try:
            built = build_body(gen_best[0], nde_modules=NUM_OF_MODULES, rng=RNG)
            save_body_artifacts(DATA, built, tag=f"gen_{gen:03d}_best")

            best_key = body_geno_to_key(gen_best[0])
            best_arch = _BODY_ARCH_CACHE.get(best_key)
            if best_arch and best_arch.viable:
                cached = _BEST_CTRL_CACHE.get(best_key)
                if cached:
                    dist, _ = probe_best_metrics(best_arch, cached.theta)
                    dist_per_gen.append(dist)
                else:
                    dist_per_gen.append(0.0)
            else:
                dist_per_gen.append(0.0)
        except Exception:
            dist_per_gen.append(0.0)

        progress = (dist_per_gen[-1] / TRACK_LENGTH) * 100
        t_gen_elapsed = (time.time() - t_gen_start) / 60
        console.log(f"[Gen {gen}] best_fit={gen_best_fit:.1f}, dist={dist_per_gen[-1]:.2f}m ({progress:.1f}%), no_improve={no_improve}, t={t_gen_elapsed:.1f}min\n")

    # Return best
    final_best = hof[0]
    return final_best, best_per_gen, dist_per_gen, mean_per_gen, std_per_gen, zbest_per_gen

def plot_robot_path(history, save_path: Path):
    try:
        if not history or len(history) < 2:
            return
        # history may contain [x,y] or [x,y,t] — we only use x,y
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
# =========================
# BASELINE: Random initialization + selection only (no learning)
# =========================
def evolve_controller_random_for_body(body_geno, verbose=False):
    """
    Baseline controller "training":
      - NO crossover, NO mutation, NO reuse of previous gens
      - For each of CTRL_N_GEN generations, sample CTRL_POP_SIZE random controllers
      - Evaluate all and keep the global best across generations
    """
    key = body_geno_to_key(body_geno)
    arch = get_body_architecture(key, body_geno)
    if not arch.viable:
        if verbose:
            console.log(f"[BaselineCtrl] Body not viable: {arch.error_msg}")
        return None, -1e6

    INP, OUT = arch.inp_size, arch.out_size

    best_theta = None
    best_fit = -1e6

    for g in range(CTRL_N_GEN):
        # Fresh random population every generation
        for _ in range(CTRL_POP_SIZE):
            theta = init_controller_genotype_for_body(INP, OUT, RNG)
            fit, _ = run_episode_with_controller(arch, theta, steps=PROBE_STEPS)
            if fit > best_fit:
                best_fit = fit
                best_theta = theta

    return best_theta, float(best_fit)


def evaluate_body_genotype_baseline(body_geno):
    """
    Baseline body evaluation:
      - For a single body, run the baseline controller search (random only)
      - Apply same viability threshold for fairness
    """
    theta, fit = evolve_controller_random_for_body(body_geno, verbose=False)
    if theta is None or fit < VIABILITY_THRESHOLD:
        return (-1e6,)
    return (fit,)


def run_baseline_random_selection():
    """
    Baseline co-evolution:
      - Same number of generations as the main EA
      - NO crossover, NO mutation, NO inheritance
      - Each generation is a fresh random body population
      - For each body, controllers are found via random search (above)
      - We record best-per-generation and distance telemetry like the main run
    """
    console.log(f"[BASELINE] Starting... (no learning: random init + selection only)")
    console.log(f"  Body pop: {BODY_POP_SIZE}, Controller random search: {CTRL_POP_SIZE}×{CTRL_N_GEN} evals/gen")

    best_per_gen = []
    dist_per_gen = []
    mean_per_gen = []
    std_per_gen  = []
    zbest_per_gen = []    

    best_overall_ind = None
    best_overall_fit = -1e6

    # Include gen 0 + BODY_N_GEN like the main EA (Gen 0 evaluation then 1..N)
    for gen in range(0, BODY_N_GEN + 1):
        console.log(f"[Baseline Gen {gen}] Evaluating {BODY_POP_SIZE} random bodies...")
        t_start = time.time()

        # Fresh random body population
        pop = [creator.Individual([init_body_genotype(RNG)]) for _ in range(BODY_POP_SIZE)]

        for idx, ind in enumerate(pop):
            if (idx + 1) % 5 == 0:
                console.log(f"  Progress: {idx+1}/{len(pop)}")
            ind.fitness.values = evaluate_body_genotype_baseline(ind[0])

        t_elapsed = time.time() - t_start
        console.log(f"[Baseline Gen {gen}] Done in {t_elapsed/60:.1f} min")

        # Pick the generation's best
        gen_best = max(pop, key=lambda x: x.fitness.values[0])
        gen_best_fit = gen_best.fitness.values[0]
        best_per_gen.append(gen_best_fit)
        
        gen_fits = [ind.fitness.values[0] for ind in pop]
        g_mean = float(np.mean(gen_fits)) if gen_fits else float("nan")
        g_std  = float(np.std(gen_fits))  if gen_fits else float("nan")
        g_z    = float((gen_best_fit - g_mean) / g_std) if (gen_fits and g_std > 0) else 0.0
        mean_per_gen.append(g_mean)
        std_per_gen.append(g_std)
        zbest_per_gen.append(g_z)        

        # Update global best
        if gen_best_fit > best_overall_fit:
            best_overall_fit = gen_best_fit
            best_overall_ind = gen_best

        # Distance telemetry + optional artifacts
        try:
            built = build_body(gen_best[0], nde_modules=NUM_OF_MODULES, rng=RNG)
            if built.viable:
                save_body_artifacts(DATA, built, tag=f"baseline_gen_{gen:03d}_best")
                key = body_geno_to_key(gen_best[0])
                arch = get_body_architecture(key, gen_best[0])
                # Use baseline controller sampling again to get a theta for distance probing
                theta, _bf = evolve_controller_random_for_body(gen_best[0], verbose=False)
                if theta is not None and arch and arch.viable:
                    dist, _ = probe_best_metrics(arch, theta)
                else:
                    dist = 0.0
            else:
                dist = 0.0
        except Exception:
            dist = 0.0

        dist_per_gen.append(dist)
        progress = (dist / TRACK_LENGTH) * 100.0
        console.log(f"[Baseline Gen {gen}] best_fit={gen_best_fit:.1f}, dist={dist:.2f}m ({progress:.1f}%)\\n")

    # Return the baseline "winner" and curves
    return best_overall_ind, best_per_gen, dist_per_gen, mean_per_gen, std_per_gen, zbest_per_gen


def main():
    console.log("\n" + "="*70)
    console.log("[PROPER CO-EVOLUTION WITH ELITISM - Robot Olympics]")
    console.log("="*70)
    console.log("METHODOLOGY:")
    console.log("  1. ELITISM: Preserve best 5 bodies, best 3 controllers")
    console.log("  2. LONG TRAINING: 50×50 = 2500 evaluations per body")
    console.log("  3. HIGHER VIABILITY: fitness > 6.0 (≈1.0m minimum)")
    console.log("  4. NO RETRAINING: Controllers cached forever")
    console.log("  5. REDUCED MUTATION: 0.3 (more conservative)")
    console.log("="*70 + "\n")

    # # === BASELINE RUN ===
    console.log('\n' + '='*70)
    console.log('[BASELINE RUN] Random init + selection only (no learning)')
    console.log('='*70)
    base_best, base_fit_curve, base_dist_curve, base_mean_curve, base_std_curve, base_z_curve = run_baseline_random_selection()
    
    fit_runs  = []
    dist_runs = []
    mean_runs = []
    std_runs  = []
    z_runs    = []
    best_overall = None
    best_overall_fit = -1e6
    
    for attempt in range(N_ATTEMPTS):
        console.log(f"[EA] Attempt {attempt+1}/{N_ATTEMPTS}")
        best, fit_curve, dist_curve, mean_curve, std_curve, z_curve = run_proper_coevolution_ea()
        
        fit_runs.append(np.asarray(fit_curve, dtype=float))
        dist_runs.append(np.asarray(dist_curve, dtype=float))
        mean_runs.append(np.asarray(mean_curve, dtype=float))
        std_runs.append(np.asarray(std_curve, dtype=float))
        z_runs.append(np.asarray(z_curve, dtype=float))

        # track the single best individual across attempts
        if best is not None:
            try:
                f = float(best.fitness.values[0])
            except Exception:
                f = -1e6
            if f > best_overall_fit:
                best_overall_fit = f
                best_overall = best
                
    # ==== Per-generation averages across the 5 runs ====
    avg_fit_curve  = np.mean(np.vstack(fit_runs),  axis=0)
    avg_dist_curve = np.mean(np.vstack(dist_runs), axis=0)
    avg_mean_curve = np.mean(np.vstack(mean_runs), axis=0)
    avg_std_curve  = np.mean(np.vstack(std_runs),  axis=0)
    avg_z_curve    = np.mean(np.vstack(z_runs),    axis=0)

    console.rule("[bold green]Averages over 5 runs")
    console.log(f"[AVG] Best fitness per gen (len={len(avg_fit_curve)}): {np.array2string(avg_fit_curve, precision=2)}")
    console.log(f"[AVG] Best distance per gen (len={len(avg_dist_curve)}): {np.array2string(avg_dist_curve, precision=2)}")

    results = dict(
        baseline=dict(
            best=base_best,
            fit_curve=base_fit_curve,
            dist_curve=base_dist_curve,
            mean_curve=base_mean_curve,
            std_curve=base_std_curve,
            z_curve=base_z_curve,
        ),
        runs=dict(
            fit_runs=[r.tolist() for r in fit_runs],
            dist_runs=[r.tolist() for r in dist_runs],
            mean_runs=[r.tolist() for r in mean_runs],
            std_runs=[r.tolist() for r in std_runs],
            z_runs=[r.tolist() for r in z_runs],
        ),
        averages=dict(
            fit_curve=avg_fit_curve.tolist(),
            dist_curve=avg_dist_curve.tolist(),
            mean_curve=avg_mean_curve.tolist(),
            std_curve=avg_std_curve.tolist(),
            z_curve=avg_z_curve.tolist(),
        ),
        best_overall_fit=float(best_overall_fit),
    )

    try:
        # --- Prepare arrays ---
        # Stack learning runs' distance curves: shape (N_ATTEMPTS, G)
        dist_stack = np.vstack(dist_runs)  # each element already np.array
        # Mean/Std across attempts per generation
        dist_mean = np.mean(dist_stack, axis=0)
        dist_std  = np.std(dist_stack, axis=0)

        # --- Plot 1: absolute distance comparison (learning avg vs baseline) ---
        plt.figure(figsize=(10, 5))
        plt.plot(avg_dist_curve, label="Learning (avg of 5)", linewidth=2)
        plt.plot(base_dist_curve, label="Baseline (random)", linewidth=2, linestyle="--")
        plt.xlabel("Generation")
        plt.ylabel("Best distance (m)")
        plt.title("Best Distance per Generation: Learning vs Baseline")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out1 = DATA / "distance_vs_baseline.png"
        plt.savefig(out1, dpi=150, bbox_inches="tight")
        plt.close()
        console.log(f"[Saved] {out1}")

        # --- Plot 2: Z-scores of distance per generation ---
        # Choose the single best learning run by final distance
        best_run_idx = int(np.argmax([runs[-1] for runs in dist_runs]))
        best_run_dist = dist_runs[best_run_idx]

        # Avoid division by zero
        eps = 1e-9
        z_best_run = (best_run_dist - dist_mean) / np.maximum(dist_std, eps)
        # Compare baseline to learning distribution as well
        # (Truncate or pad baseline to match length, though lengths should match)
        L = min(len(base_dist_curve), len(dist_mean))
        base_d_aligned = np.asarray(base_dist_curve[:L], dtype=float)
        mean_aligned   = dist_mean[:L]
        std_aligned    = np.maximum(dist_std[:L], eps)
        z_baseline     = (base_d_aligned - mean_aligned) / std_aligned

        plt.figure(figsize=(10, 5))
        plt.plot(z_best_run[:L], label="Learning (best single run) Z", linewidth=2)
        plt.plot(z_baseline, label="Baseline Z (vs learning dist.)", linewidth=2, linestyle="--")
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("Generation")
        plt.ylabel("Z-score of distance")
        plt.title("Z-scores of Best Distance per Generation")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out2 = DATA / "distance_zscores.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()
        console.log(f"[Saved] {out2}")
    except Exception as e:
        console.log(f"[Plot Error] Could not generate distance plots: {e}")



    # console.log("\n" + "="*70)
    # console.log("[FINAL RESULTS]")
    # console.log("\n" + "="*70)
    # console.log("[BASELINE RESULTS]")
    # if base_fit_curve:
    #     console.log(f"  Baseline best fitness: {base_fit_curve[-1]:.1f}")
    # if base_dist_curve:
    #     console.log(f"  Baseline best distance: {base_dist_curve[-1]:.2f}m / {TRACK_LENGTH:.2f}m ({100*base_dist_curve[-1]/TRACK_LENGTH:.1f}%)")
    # console.log("="*70)

    # Save baseline best body & plot its path
    try:
        built_b = build_body(base_best[0], nde_modules=NUM_OF_MODULES, rng=RNG)
        save_body_artifacts(DATA, built_b, tag="BASELINE_FINAL_BEST")
        key_b = body_geno_to_key(base_best[0])
        arch_b = get_body_architecture(key_b, base_best[0])
        theta_b, _ = evolve_controller_random_for_body(base_best[0], verbose=False)
        if theta_b is not None and arch_b and arch_b.viable:
            fit_b, hist_b = run_episode_with_controller(arch_b, theta_b, steps=PROBE_STEPS)
            plot_robot_path(hist_b, DATA / "baseline_final_best_path.png")
            with open(DATA / "BASELINE_FINAL_BEST_path.json", "w") as f:
                json.dump({"history": hist_b}, f, indent=2)
            console.log(f"[Saved] Baseline best path plot to: {DATA / 'baseline_final_best_path.png'}")
            console.log(f"[Saved] Baseline best path coordinates to: {DATA / 'BASELINE_FINAL_BEST_path.json'}")
    except Exception as e:
        console.log(f"[Error] Could not save baseline artifacts: {e}")
    console.log(f"  Best fitness: {fit_curve[-1]:.1f}")
    console.log(f"  Best distance: {dist_curve[-1]:.2f}m / {TRACK_LENGTH:.2f}m ({100*dist_curve[-1]/TRACK_LENGTH:.1f}%)")
    console.log("="*70)

    # Save best body
    try:
        built = build_body(best_overall[0], nde_modules=NUM_OF_MODULES, rng=RNG)
        save_body_artifacts(DATA, built, tag="FINAL_BEST")
        console.log(f"\n[Saved] Final best to: {DATA}/FINAL_BEST_*.json")
    except Exception as e:
        console.log(f"[Error] Could not save final best: {e}")

    # NEW: run episode for the best and save/plot full coordinate history
    try:
        key = body_geno_to_key(best_overall[0])
        arch = get_body_architecture(key, best_overall[0])
        theta, _ = evolve_controller_for_body(best_overall[0], current_generation=BODY_N_GEN, verbose=False)
        if theta is not None and arch and arch.viable:
            fit, history = run_episode_with_controller(arch, theta, steps=PROBE_STEPS)
            path_png = DATA / "final_best_path.png"
            plot_robot_path(history, path_png)
            with open(DATA / "FINAL_BEST_path.json", "w") as f:
                json.dump({"history": history}, f, indent=2)
            console.log(f"[Saved] Best path plot to: {path_png}")
            console.log(f"[Saved] Best path coordinates to: {DATA / 'FINAL_BEST_path.json'}")
        else:
            console.log("[Warn] Could not obtain controller or architecture for plotting.")
    except Exception as e:
        console.log(f"[Error] Could not plot best path: {e}")

if __name__ == "__main__":
    main()
