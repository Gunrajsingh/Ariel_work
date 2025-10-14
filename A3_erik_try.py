# A3_erik_race_FIXED.py
# Fixed version with stronger fitness, better tracking, and more exploration

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
import copy

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
from ariel.utils.renderers import single_frame_renderer

console = Console()

# =========================
# Global settings - RETUNED
# =========================
SCRIPT_NAME = "A3_co_evolution_fixed"
CWD = Path(".").resolve()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# EA parameters - Body
BODY_POP_SIZE = 20            # INCREASED from 16
BODY_N_GEN = 25               # INCREASED from 20
BODY_TOURNSIZE = 3
BODY_CXPB = 0.6               # REDUCED from 0.7 (less exploitation)
BODY_MUTPB = 0.5              # INCREASED from 0.4
BODY_SBX_ETA = 15.0           # REDUCED from 20 (more variation)
BODY_IMM_FRAC = 0.20          # INCREASED from 0.15

# Controller EA
CTRL_POP_SIZE = 12
CTRL_N_GEN = 6
CTRL_TOURNSIZE = 3
CTRL_CXPB = 0.8
CTRL_MUTPB = 0.3              # INCREASED from 0.25
CTRL_SBX_ETA = 15.0
CTRL_IMM_FRAC = 0.15

# Sim + environment
SIM_DURATION = 15.0
WARMUP_STEPS = 40             # REDUCED further from 50
STALL_WINDOW_SEC = 2.5        # INCREASED from 2.0
STALL_EPS = 0.003             # TIGHTENED from 0.005
RATE_LIMIT_FRAC = 0.10        # INCREASED from 0.08

SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]
TRACK_LENGTH = float(TARGET_POSITION[0] - SPAWN_POS[0])

# Body encoding
BODY_L = 64
NUM_OF_MODULES = 30

# Controller encoding
CTRL_HIDDEN = 16
CTRL_W_LOW, CTRL_W_HIGH = -3.5, 3.5  # INCREASED from -3.0, 3.0

# Outputs
EA_FITNESS_PNG = DATA / "ea_best_over_generations.png"

# Race-aware telemetry
PROBE_STEPS = 700             # INCREASED from 600

# Logging
VERBOSE = False
LOG_ACTUATORS = False
LOG_EPISODES = 0

DEBUG_EPISODES_TO_PRINT = LOG_EPISODES
_CTRL_UNLIM_SCALE = 3.14159

# =========================
# Types & helpers
# =========================
@dataclass
class BuiltBody:
    nde: NeuralDevelopmentalEncoding
    decoded_graph: Any
    mjspec: Any
    viable: bool
    error_msg: str


def init_body_genotype(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = rng.random(n).astype(np.float32)
    c = rng.random(n).astype(np.float32)
    r = rng.random(n).astype(np.float32)
    return (t, c, r)


def _sbx_pair(a: np.ndarray, b: np.ndarray, eta: float, low=0.0, high=1.0, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    u = rng.random(a.shape, dtype=np.float32)
    beta = np.empty_like(a, dtype=np.float32)
    mask = u <= 0.5
    beta[mask] = (2.0 * u[mask]) ** (1.0 / (eta + 1.0))
    beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (eta + 1.0))
    c1 = 0.5 * ((a + b) - beta * (b - a))
    c2 = 0.5 * ((a + b) + beta * (b - a))
    return np.clip(c1, low, high), np.clip(c2, low, high)


def sbx_body(g1, g2, eta=BODY_SBX_ETA, rng: np.random.Generator | None = None):
    a1, b1 = _sbx_pair(g1[0], g2[0], eta, rng=rng)
    a2, b2 = _sbx_pair(g1[1], g2[1], eta, rng=rng)
    a3, b3 = _sbx_pair(g1[2], g2[2], eta, rng=rng)
    return (a1, a2, a3), (b1, b2, b3)


def block_mutation_aggressive(g, indpb: float = 8.0 / BODY_L, rng: np.random.Generator | None = None):
    """AGGRESSIVE: Change ~8 genes per mutation"""
    rng = rng or np.random.default_rng()
    t, c, r = g
    mask_t = rng.random(t.shape) < indpb
    mask_c = rng.random(c.shape) < indpb
    mask_r = rng.random(r.shape) < indpb
    t = t.copy(); c = c.copy(); r = r.copy()
    t[mask_t] = rng.random(np.count_nonzero(mask_t)).astype(np.float32)
    c[mask_c] = rng.random(np.count_nonzero(mask_c)).astype(np.float32)
    r[mask_r] = rng.random(np.count_nonzero(mask_r)).astype(np.float32)
    return (t, c, r)


def build_body(geno: tuple[np.ndarray, np.ndarray, np.ndarray], nde_modules: int, rng: np.random.Generator) -> BuiltBody:
    try:
        t, c, r = geno
        nde = NeuralDevelopmentalEncoding(number_of_modules=nde_modules)
        nde.t = t.astype(np.float32); nde.c = c.astype(np.float32); nde.r = r.astype(np.float32)
        nde.n_modules = nde_modules
        p_mats = nde.forward([nde.t, nde.c, nde.r])
        decoder = HighProbabilityDecoder(nde_modules)
        graph = decoder.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
        spec = construct_mjspec_from_graph(graph)
        return BuiltBody(nde=nde, decoded_graph=graph, mjspec=spec, viable=True, error_msg="OK")
    except Exception as e:
        return BuiltBody(NeuralDevelopmentalEncoding(1), None, None, False, str(e))


def save_body_artifacts(run_dir: Path, built: BuiltBody, tag: str):
    if not built.viable:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if built.decoded_graph:
        gpath = run_dir / f"{tag}_decoded_graph.json"
        save_graph_as_json(built.decoded_graph, str(gpath))
    nde_json = {
        "t": built.nde.t.tolist(),
        "c": built.nde.c.tolist(),
        "r": built.nde.r.tolist(),
        "n_modules": built.nde.n_modules,
        "viable": built.viable,
        "error_msg": built.error_msg,
    }
    with open(run_dir / f"{tag}_nde.json", "w") as f:
        json.dump(nde_json, f, indent=2)

# =========================
# Body architecture & caches
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

_BODY_ARCH_CACHE: dict[str, BodyArchitecture] = {}
_BEST_CTRL_CACHE: dict[str, tuple[np.ndarray, float]] = {}


def body_geno_to_key(body_geno) -> str:
    t, c, r = body_geno
    combined = np.concatenate([t.flatten(), c.flatten(), r.flatten()])
    return str(hash(combined.tobytes()))


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
# STRONGER FITNESS FUNCTION
# =========================
def _estimate_ttf(x_hist: list[float], t_hist: list[float], L: float) -> float:
    for i in range(1, len(x_hist)):
        if x_hist[i-1] < L <= x_hist[i]:
            dx = x_hist[i] - x_hist[i-1]
            dt = t_hist[i] - t_hist[i-1]
            if dx == 0:
                return t_hist[i]
            a = (L - x_hist[i-1]) / dx
            return t_hist[i-1] + a * dt
    return t_hist[-1]


def compute_strong_fitness(x_hist: list[float], t_hist: list[float], track_length: float) -> tuple[float, dict]:
    """
    STRONGER: More aggressive rewards for distance.
    Goal: Make it to 5.8m as fast as possible.
    """
    if len(x_hist) < 2:
        return 0.0, dict(finished=0, dist=0.0, speed=0.0, ttf=None, progress=0.0)
    
    dist = max(0.0, float(x_hist[-1] - x_hist[0]))
    dur = max(1e-6, float(t_hist[-1] - t_hist[0]))
    avg_speed = dist / dur
    progress_pct = (dist / track_length) * 100.0
    
    # Check if finished
    finished = dist >= (track_length - 0.05)
    
    if finished:
        ttf = _estimate_ttf(x_hist, t_hist, track_length)
        # HUGE bonus for finishing + speed reward
        finish_bonus = 500.0
        time_bonus = (track_length / max(1e-6, ttf)) * 50  # Reward faster completion
        score = dist * 10 + finish_bonus + time_bonus
        return score, dict(finished=1, dist=track_length, speed=track_length/ttf, ttf=ttf, progress=100.0)
    else:
        # STRONGER progressive rewards
        
        # 1. Distance is PRIMARY - scale it up heavily
        dist_score = dist * 10  # 1m = 10 points, 5m = 50 points
        
        # 2. Exponential bonus for getting further
        # Reward reaching milestones exponentially
        milestone_bonus = 0.0
        if dist >= 1.0:
            milestone_bonus += 5
        if dist >= 2.0:
            milestone_bonus += 10
        if dist >= 3.0:
            milestone_bonus += 20
        if dist >= 4.0:
            milestone_bonus += 40
        if dist >= 5.0:
            milestone_bonus += 80
        
        # 3. Speed bonus (faster = better)
        speed_bonus = min(10.0, avg_speed * 2.0)
        
        # 4. Forwardness (penalize backward movement heavily)
        fwd_ratio = 0.0
        backward_penalty = 0.0
        if len(x_hist) > 1:
            dx = np.diff(np.array(x_hist, dtype=float))
            fwd_steps = np.sum(dx > 0)
            total_steps = len(dx)
            fwd_ratio = fwd_steps / max(1, total_steps)
            backward_penalty = float(np.sum(np.abs(dx[dx < 0]))) * 2.0  # Penalize backward
        
        forwardness_bonus = fwd_ratio * 5.0
        
        # 5. Consistency bonus
        consistency = 0.0
        if len(x_hist) > 10:
            # Check if robot maintains forward motion
            recent_progress = x_hist[-1] - x_hist[-min(50, len(x_hist)//2)]
            if recent_progress > 0:
                consistency = min(3.0, recent_progress)
        
        score = (dist_score + 
                milestone_bonus + 
                speed_bonus + 
                forwardness_bonus + 
                consistency - 
                backward_penalty)
        
        return score, dict(finished=0, dist=dist, speed=avg_speed, ttf=None, progress=progress_pct)


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
# Episode runner
# =========================
_debug_counter = {"episodes": 0}

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


def run_episode_with_controller(body_arch: BodyArchitecture, theta: np.ndarray, steps: int = 800) -> tuple[float, list[list[float]]]:
    if not body_arch or not body_arch.viable:
        return -1e6, [[0, 0, 0], [0, 0, 0]]

    model = body_arch.model
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    mj.set_mjcb_control(None)

    INP = body_arch.inp_size
    OUT = body_arch.out_size
    params = unpack_controller_theta(theta, INP, CTRL_HIDDEN, OUT)
    if params is None:
        return -1e6, [[0, 0, 0], [0, 0, 0]]

    x_hist: list[float] = []
    t_hist: list[float] = []
    u_prev = np.zeros(OUT, dtype=float)
    start_x = _body_x(model, data, body_arch.track_body_name)

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

        x = _body_x(model, data, body_arch.track_body_name)
        x_hist.append(x - start_x)
        t_hist.append(float(data.time))

        if x_hist[-1] >= (TRACK_LENGTH - 1e-2):
            break

        if t_hist[-1] - last_progress_check_t >= STALL_WINDOW_SEC:
            t_goal = t_hist[-1] - STALL_WINDOW_SEC
            j = 0
            while j < len(t_hist) - 1 and t_hist[j + 1] < t_goal:
                j += 1
            if (x_hist[-1] - x_hist[j]) < STALL_EPS:
                break
            last_progress_check_t = t_hist[-1]

    fitness, _ = compute_strong_fitness(x_hist, t_hist, TRACK_LENGTH)

    path = [[start_x, 0.0, 0.0], [start_x + (x_hist[-1] if x_hist else 0.0), 0.0, 0.0]]
    return float(fitness), path

# =========================
# Fast controller search with MORE diversity
# =========================
def quick_controller_search_diverse(body_arch: BodyArchitecture, body_geno, 
                                   n_samples: int = 40, n_refine: int = 20):
    """
    MORE DIVERSE: Try wider range of initializations.
    """
    key = body_geno_to_key(body_geno)
    
    if key in _BEST_CTRL_CACHE:
        cached_theta, cached_fit = _BEST_CTRL_CACHE[key]
        # Still try to improve cached
        if cached_fit < 10.0:  # If not great, keep searching
            pass
        else:
            return cached_theta, cached_fit
    
    if not body_arch.viable:
        return None, -1e6
    
    INP, OUT = body_arch.inp_size, body_arch.out_size
    theta_size = controller_theta_size(INP, CTRL_HIDDEN, OUT)
    
    # Phase 1: DIVERSE random sampling
    best_theta = None
    best_fit = -1e9
    
    scales = [0.3, 0.5, 0.7, 1.0]  # Try multiple scales
    for i in range(n_samples):
        scale = scales[i % len(scales)]
        theta = RNG.normal(0.0, scale, size=theta_size).astype(float)
        theta = np.clip(theta, CTRL_W_LOW, CTRL_W_HIGH)
        fit, _ = run_episode_with_controller(body_arch, theta, steps=600)
        if fit > best_fit:
            best_fit = fit
            best_theta = theta.copy()
    
    if best_theta is None:
        return None, -1e6
    
    # Phase 2: Aggressive hill climbing
    step_size = 0.3
    for _ in range(n_refine):
        for _ in range(5):  # Try 5 candidates per iteration
            noise = RNG.normal(0.0, step_size, size=theta_size)
            candidate = np.clip(best_theta + noise, CTRL_W_LOW, CTRL_W_HIGH)
            fit, _ = run_episode_with_controller(body_arch, candidate, steps=600)
            
            if fit > best_fit:
                best_fit = fit
                best_theta = candidate.copy()
                break
        step_size *= 0.9  # Gradually reduce
    
    _BEST_CTRL_CACHE[key] = (best_theta, best_fit)
    return best_theta, best_fit


# =========================
# Controller EA
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


def controller_polynomial_mutation(individual, eta=12.0, indpb=0.20, rng=None):
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
    theta = rng.normal(0.0, 0.6, size=w_size).astype(float)  # INCREASED from 0.5
    return np.clip(theta, CTRL_W_LOW, CTRL_W_HIGH)


def evolve_controller_for_body(body_geno, verbose=False):
    key = body_geno_to_key(body_geno)
    arch = get_body_architecture(key, body_geno)
    if not arch.viable:
        if verbose:
            console.log(f"[CtrlEA] Body not viable: {arch.error_msg}")
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
        theta = init_controller_genotype_for_body(arch.inp_size, arch.out_size, RNG)
        return creator.ControllerIndividual(theta.tolist())

    pop = [init_ctrl_ind() for _ in range(CTRL_POP_SIZE)]

    # Seed from cache if available
    cached = _BEST_CTRL_CACHE.get(key)
    if cached is not None:
        seed_theta, _ = cached
        pop[0] = creator.ControllerIndividual(seed_theta.tolist())

    # Initial eval
    for ind in pop:
        fit, _ = run_episode_with_controller(arch, np.array(ind), steps=600)
        ind.fitness.values = (fit,)

    for gen in range(CTRL_N_GEN):
        elite = tools.selBest(pop, 1)[0]

        offspring = tools.selTournament(pop, len(pop), tournsize=CTRL_TOURNSIZE)
        offspring = [creator.ControllerIndividual(ind[:]) for ind in offspring]
        offspring[0] = creator.ControllerIndividual(elite[:])

        for c1, c2 in zip(offspring[1::2], offspring[2::2]):
            if random.random() < CTRL_CXPB:
                a, b = controller_sbx_crossover(c1, c2, eta=CTRL_SBX_ETA, rng=RNG)
                c1[:] = a.tolist(); c2[:] = b.tolist()
                if hasattr(c1.fitness, "values"): del c1.fitness.values
                if hasattr(c2.fitness, "values"): del c2.fitness.values
        for m in offspring[1:]:
            if random.random() < CTRL_MUTPB:
                x = controller_polynomial_mutation(m, eta=12.0, indpb=0.20, rng=RNG)
                m[:] = x.tolist()
                if hasattr(m.fitness, "values"): del m.fitness.values
        
        n_imm = max(0, int(CTRL_IMM_FRAC * len(offspring)))
        for _ in range(n_imm):
            offspring[random.randrange(len(offspring))] = creator.ControllerIndividual(
                init_controller_genotype_for_body(arch.inp_size, arch.out_size, RNG).tolist()
            )

        for ind in offspring:
            if not ind.fitness.valid:
                fit, _ = run_episode_with_controller(arch, np.array(ind), steps=600)
                ind.fitness.values = (fit,)
        pop = offspring

    best = tools.selBest(pop, 1)[0]
    _BEST_CTRL_CACHE[key] = (np.array(best), best.fitness.values[0])
    return np.array(best), best.fitness.values[0]


# =========================
# Adaptive body evaluation
# =========================
def evaluate_body_individual_adaptive(ind, generation: int):
    """Use fast search longer, switch to EA later."""
    geno = ind[0]
    key = body_geno_to_key(geno)
    arch = get_body_architecture(key, geno)
    
    if not arch.viable:
        return (-1e6,)
    
    try:
        # Use fast search for first 12 generations
        if generation < 12:
            best_theta, best_fit = quick_controller_search_diverse(
                arch, geno,
                n_samples=40,
                n_refine=20
            )
            if best_theta is None:
                return (-1e6,)
            return (float(best_fit),)
        
        # Mid generations: light EA
        elif generation < 20:
            old_g, old_p = CTRL_N_GEN, CTRL_POP_SIZE
            try:
                globals()["CTRL_N_GEN"] = 5
                globals()["CTRL_POP_SIZE"] = 10
                best_theta, best_fit = evolve_controller_for_body(geno, verbose=False)
                if best_theta is None:
                    return (-1e6,)
                return (float(best_fit),)
            finally:
                globals()["CTRL_N_GEN"] = old_g
                globals()["CTRL_POP_SIZE"] = old_p
        
        # Late generations: full EA
        else:
            best_theta, best_fit = evolve_controller_for_body(geno, verbose=False)
            if best_theta is None:
                return (-1e6,)
            return (float(best_fit),)
            
    except Exception as e:
        console.log(f"[Body Eval] Exception: {e}")
        return (-1e6,)


# =========================
# Co-evolution setup
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
    geno = init_body_genotype(RNG, BODY_L)
    return creator.BodyIndividual([geno])

toolbox.register("individual", init_body_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def mate_bodies(ind1, ind2):
    g1, g2 = ind1[0], ind2[0]
    c1, c2 = sbx_body(g1, g2, eta=BODY_SBX_ETA, rng=RNG)
    ind1[0] = c1; ind2[0] = c2
    if hasattr(ind1.fitness, "values"): del ind1.fitness.values
    if hasattr(ind2.fitness, "values"): del ind2.fitness.values
    return ind1, ind2


def mutate_body_aggressive(ind):
    ind[0] = block_mutation_aggressive(ind[0], indpb=8.0 / BODY_L, rng=RNG)
    if hasattr(ind.fitness, "values"): del ind.fitness.values
    return (ind,)


toolbox.register("mate", mate_bodies)
toolbox.register("mutate", mutate_body_aggressive)
toolbox.register("select", tools.selTournament, tournsize=BODY_TOURNSIZE)


def _hof_similar(a, b) -> bool:
    try:
        g1, g2 = a[0], b[0]
        v1 = np.concatenate([np.ravel(g1[0]), np.ravel(g1[1]), np.ravel(g1[2])])
        v2 = np.concatenate([np.ravel(g2[0]), np.ravel(g2[1]), np.ravel(g2[2])])
        return bool(np.allclose(v1, v2, atol=1e-12, rtol=1e-12))
    except Exception:
        return False

# =========================
# Telemetry
# =========================
def probe_best_metrics(arch: BodyArchitecture, theta: np.ndarray, steps: int = PROBE_STEPS) -> tuple[float, float]:
    if not arch or not arch.viable:
        return 0.0, float("inf")

    model = arch.model
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    INP, OUT = arch.inp_size, arch.out_size
    params = unpack_controller_theta(theta, INP, CTRL_HIDDEN, OUT)
    if params is None:
        return 0.0, float("inf")

    try:
        start_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, arch.track_body_name) if arch.track_body_name else 1
        if start_bid < 0 and model.nbody >= 2:
            start_bid = 1
    except Exception:
        start_bid = 1
    start_x = float(data.xpos[start_bid, 0])

    u_prev = np.zeros(OUT, dtype=float)
    dt = float(model.opt.timestep)
    max_steps = min(steps, int(SIM_DURATION / max(1e-6, dt)))

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

        x_now = float(data.xpos[start_bid, 0])
        if (x_now - start_x) >= (TRACK_LENGTH - 1e-2):
            break

    end_pos = np.array(data.xpos[start_bid, :], dtype=float)
    dist_x = max(0.0, float(end_pos[0] - start_x))
    dist_to_target = float(np.linalg.norm(end_pos - np.array(TARGET_POSITION, dtype=float)))
    return dist_x, dist_to_target

# =========================
# IMPROVED EA loop with better tracking
# =========================
def run_co_evolution_fixed():
    """
    FIXED: Stronger fitness, more exploration, better tracking.
    """
    random.seed(SEED); np.random.seed(SEED)
    console.log(f"[Co-Evolution] FIXED version - Goal: reach {TRACK_LENGTH:.2f}m fast!")

    # Initialize population
    pop = toolbox.population(n=BODY_POP_SIZE)
    
    # Initial evaluation
    console.log(f"Evaluating {len(pop)} initial bodies (diverse search)...")
    for ind in pop:
        ind.fitness.values = evaluate_body_individual_adaptive(ind, generation=0)

    # Hall of fame - keep fewer elites to allow more diversity
    ELITE_K = max(2, BODY_POP_SIZE // 10)
    hof = tools.HallOfFame(ELITE_K, similar=_hof_similar)
    hof.update(pop)
    
    best_per_gen = [max(ind.fitness.values[0] for ind in pop)]
    dist_per_gen = []
    progress_per_gen = []
    
    best_so_far = best_per_gen[-1]
    no_improve = 0

    # Save gen 0 best
    best0 = max(pop, key=lambda x: x.fitness.values[0])
    try:
        built0 = build_body(best0[0], nde_modules=NUM_OF_MODULES, rng=RNG)
        save_body_artifacts(DATA, built0, tag="gen_000_best")
    except Exception:
        pass

    # Initial telemetry
    try:
        best_key = body_geno_to_key(best0[0])
        best_arch = _BODY_ARCH_CACHE.get(best_key)
        dist_x = 0.0
        if best_arch and best_arch.viable:
            cached = _BEST_CTRL_CACHE.get(best_key)
            if cached:
                theta_best = cached[0]
                dist_x, _ = probe_best_metrics(best_arch, theta_best, steps=PROBE_STEPS)
        dist_per_gen.append(dist_x)
        progress_per_gen.append((dist_x / TRACK_LENGTH) * 100)
    except Exception:
        dist_per_gen.append(0.0)
        progress_per_gen.append(0.0)

    console.log(f"Gen 0: best_fit={best_per_gen[0]:.2f}, dist={dist_per_gen[0]:.2f}m ({progress_per_gen[0]:.1f}%)")
    
    t_wall = time.time()
    
    for gen in range(1, BODY_N_GEN + 1):
        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < BODY_CXPB:
                toolbox.mate(c1, c2)
        
        # Adaptive mutation rate
        adapt_mutpb = BODY_MUTPB
        if no_improve >= 2:
            adapt_mutpb = min(0.7, BODY_MUTPB * 1.4)
        
        for m in offspring:
            if random.random() < adapt_mutpb:
                toolbox.mutate(m)
        
        # Always add some immigrants for diversity
        n_imm = max(2, int(BODY_IMM_FRAC * len(offspring)))
        if no_improve >= 2:
            n_imm = max(3, int(BODY_IMM_FRAC * 1.5 * len(offspring)))
        
        for _ in range(n_imm):
            offspring[random.randrange(len(offspring))] = toolbox.individual()
        
        # Evaluate offspring
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate_body_individual_adaptive(ind, gen)
        
        # Update hall of fame
        hof.update(offspring)
        
        # Build next generation with diversity
        elites = [toolbox.clone(ind) for ind in hof.items]
        offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        
        # Take top performers but leave room for diversity
        remaining = BODY_POP_SIZE - len(elites)
        top_performers = offspring[:remaining]
        
        pop = elites + top_performers
        
        # Track best
        best = max(pop, key=lambda ind: ind.fitness.values[0])
        best_fit = best.fitness.values[0]
        best_per_gen.append(best_fit)
        
        if best_fit > best_so_far + 0.1:  # Require meaningful improvement
            best_so_far = best_fit
            no_improve = 0
        else:
            no_improve += 1
        
        # Compute stats
        avg_fit = np.mean([ind.fitness.values[0] for ind in pop])
        
        # DETAILED TELEMETRY
        dist_x = 0.0
        progress_pct = 0.0
        try:
            best_key = body_geno_to_key(best[0])
            best_arch = _BODY_ARCH_CACHE.get(best_key)
            if best_arch and best_arch.viable:
                cached = _BEST_CTRL_CACHE.get(best_key)
                if cached:
                    theta_best = cached[0]
                    dist_x, _ = probe_best_metrics(best_arch, theta_best, steps=PROBE_STEPS)
                    progress_pct = (dist_x / TRACK_LENGTH) * 100
        except Exception:
            pass
        
        dist_per_gen.append(dist_x)
        progress_per_gen.append(progress_pct)
        
        dt_wall = time.time() - t_wall
        
        # Enhanced logging with progress tracking
        finished_marker = "🏁" if dist_x >= TRACK_LENGTH - 0.05 else ""
        console.log(
            f"[EA] Gen {gen:02d} | fit={best_fit:.1f} | avg={avg_fit:.1f} | "
            f"dist={dist_x:.2f}m/{TRACK_LENGTH:.1f}m ({progress_pct:.1f}%) {finished_marker} | "
            f"no_imp={no_improve} | t={dt_wall:.1f}s"
        )
        
        # Save artifacts every 5 generations
        if gen % 5 == 0:
            try:
                built = build_body(best[0], nde_modules=NUM_OF_MODULES, rng=RNG)
                save_body_artifacts(DATA, built, tag=f"gen_{gen:03d}_best")
            except Exception:
                pass
        
        t_wall = time.time()

    # Save final best
    final_best = max(pop, key=lambda ind: ind.fitness.values[0])
    try:
        built_final = build_body(final_best[0], nde_modules=NUM_OF_MODULES, rng=RNG)
        save_body_artifacts(DATA, built_final, tag="final_best")
        console.log(f"[EA] Saved final best to {DATA}")
    except Exception as e:
        console.log(f"[Save] Error: {e}")

    # Plot fitness AND distance progress
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Fitness
        ax1.plot(best_per_gen, marker="o", linewidth=2, markersize=5, label="Best Fitness", color="blue")
        ax1.set_xlabel("Generation", fontsize=11)
        ax1.set_ylabel("Fitness", fontsize=11)
        ax1.set_title("Fitness Progress", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Distance & Progress
        ax2.plot(dist_per_gen, marker="s", linewidth=2, markersize=5, label="Distance (m)", color="green")
        ax2.axhline(y=TRACK_LENGTH, color='r', linestyle='--', label=f"Goal ({TRACK_LENGTH:.1f}m)")
        ax2.set_xlabel("Generation", fontsize=11)
        ax2.set_ylabel("Distance (m)", fontsize=11)
        ax2.set_title("Distance to Goal Progress", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(EA_FITNESS_PNG, dpi=150)
        plt.close()
        console.log(f"[EA] Saved progress plots to {EA_FITNESS_PNG}")
    except Exception as e:
        console.log(f"[Plot] {e}")

    return final_best, best_per_gen, dist_per_gen, progress_per_gen


# =========================
# Main
# =========================
def main():
    console.log("[Co-Evolution] Starting FIXED race co-evolution...")
    console.log(f"[Goal] Reach {TRACK_LENGTH:.2f}m as fast as possible!")
    console.log(f"[Config] Pop={BODY_POP_SIZE}, Gens={BODY_N_GEN}, MutPB={BODY_MUTPB}")
    console.log(f"[Config] Warmup={WARMUP_STEPS}, RateLimit={RATE_LIMIT_FRAC}")
    
    best, fit_curve, dist_curve, prog_curve = run_co_evolution_fixed()
    
    console.log(f"\n{'='*60}")
    console.log(f"[FINAL RESULTS]")
    console.log(f"{'='*60}")
    console.log(f"Best fitness: {best.fitness.values[0]:.2f}")
    console.log(f"Final distance: {dist_curve[-1]:.2f}m / {TRACK_LENGTH:.2f}m")
    console.log(f"Progress: {prog_curve[-1]:.1f}%")
    
    if dist_curve[-1] >= TRACK_LENGTH - 0.05:
        console.log(f"🎉 SUCCESS! Robot reached the finish line!")
    else:
        console.log(f"Distance remaining: {TRACK_LENGTH - dist_curve[-1]:.2f}m")
    
    console.log(f"\nFitness progression: {[round(f, 1) for f in fit_curve[::5]]}")
    console.log(f"Distance progression: {[round(d, 2) for d in dist_curve[::5]]}")
    console.log(f"{'='*60}")

if __name__ == "__main__":
    main()