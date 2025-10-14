# A3_erik_race_SELECTION_FIXED.py
# Fixed selection pressure + reduced search overhead
# Key insight: The EA selection was broken (tournament size 2 = random walk)
# Now: Proper selection + moderate controller search = faster AND better

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

console = Console()

# =========================
# PROPER SELECTION + MODERATE SEARCH
# =========================
SCRIPT_NAME = "A3_selection_fixed"
CWD = Path(".").resolve()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# EA parameters - AGGRESSIVE EXPLORATION
BODY_POP_SIZE = 30            # INCREASED from 20 (more diversity)
BODY_N_GEN = 40               # More time to explore
BODY_TOURNSIZE = 3            # REDUCED from 5 (less selection pressure = more diversity)
BODY_CXPB = 0.5               # REDUCED from 0.6 (less recombination)
BODY_MUTPB = 0.8              # INCREASED from 0.6 (more mutation!)
BODY_SBX_ETA = 8.0            # REDUCED from 12 (more variation in crossover)
BODY_ELITE_K = 4

# Controller search - ADAPTIVE BUDGET
CTRL_SAMPLES_MIN = 30         # Quick filter for bad bodies
CTRL_SAMPLES_MAX = 200        # Deep search for promising bodies
CTRL_REFINE_MIN = 10          # Quick refine
CTRL_REFINE_MAX = 50          # Deep refine
CTRL_PARALLEL_CANDIDATES = 8

# Sim settings
SIM_DURATION = 15.0
WARMUP_STEPS = 30
STALL_WINDOW_SEC = 3.0
STALL_EPS = 0.002
RATE_LIMIT_FRAC = 0.12

SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]
TRACK_LENGTH = float(TARGET_POSITION[0] - SPAWN_POS[0])

# Body encoding
BODY_L = 64
NUM_OF_MODULES = 30

# Controller encoding
CTRL_HIDDEN = 16
CTRL_W_LOW, CTRL_W_HIGH = -4.0, 4.0

EA_FITNESS_PNG = DATA / "ea_best_over_generations.png"
PROBE_STEPS = 700

VERBOSE = False
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


def strong_mutation(g, indpb: float = 15.0 / BODY_L, rng: np.random.Generator | None = None):
    """VERY STRONG mutation: ~15 genes (was 10)"""
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
# Body architecture
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
# Controller
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
# Quality-aware fitness
# =========================
def compute_quality_fitness(x_hist: list[float], t_hist: list[float], track_length: float) -> tuple[float, dict]:
    """Quality-aware fitness that rewards smooth forward motion."""
    if len(x_hist) < 2:
        return 0.0, dict(finished=0, dist=0.0, speed=0.0, quality=0.0, progress=0.0)
    
    dist = max(0.0, float(x_hist[-1] - x_hist[0]))
    dur = max(1e-6, float(t_hist[-1] - t_hist[0]))
    avg_speed = dist / dur
    progress_pct = (dist / track_length) * 100.0
    
    finished = dist >= (track_length - 0.05)
    
    if finished:
        ttf = t_hist[-1]
        for i in range(1, len(x_hist)):
            if x_hist[i-1] < track_length <= x_hist[i]:
                dx = x_hist[i] - x_hist[i-1]
                dt = t_hist[i] - t_hist[i-1]
                if dx > 0:
                    a = (track_length - x_hist[i-1]) / dx
                    ttf = t_hist[i-1] + a * dt
                break
        
        finish_bonus = 1000.0
        speed_reward = (track_length / max(1e-6, ttf)) * 100
        score = dist * 20 + finish_bonus + speed_reward
        return score, dict(finished=1, dist=track_length, speed=track_length/ttf, quality=1.0, progress=100.0)
    
    # STANDARD EC FITNESS: Just maximize distance!
    # No milestones, no tricks - pure distance optimization
    
    finished = dist >= (track_length - 0.05)
    
    if finished:
        # Huge bonus for finishing + time reward
        ttf = t_hist[-1]
        for i in range(1, len(x_hist)):
            if x_hist[i-1] < track_length <= x_hist[i]:
                dx = x_hist[i] - x_hist[i-1]
                dt = t_hist[i] - t_hist[i-1]
                if dx > 0:
                    a = (track_length - x_hist[i-1]) / dx
                    ttf = t_hist[i-1] + a * dt
                break
        
        finish_bonus = 10000.0
        time_reward = (track_length / max(1e-6, ttf)) * 100
        score = finish_bonus + time_reward
        return score, dict(finished=1, dist=track_length, speed=track_length/ttf, quality=1.0, progress=100.0)
    
    # Standard fitness: Distance squared (creates gradient)
    # d^2 means: 2m is 4x better than 1m, 3m is 9x better
    fitness = dist * dist * 50.0  # Squared distance
    
    # Small speed bonus (rewards efficiency)
    speed_bonus = avg_speed * 5.0
    
    # Quality: only used to break ties at same distance
    quality_score = 0.0
    if len(x_hist) > 10:
        dx = np.diff(np.array(x_hist, dtype=float))
        forward_ratio = np.sum(dx > 0) / max(1, len(dx))
        quality_score = forward_ratio
    
    quality_bonus = quality_score * 2.0  # Max 2 points
    
    score = fitness + speed_bonus + quality_bonus
    
    return score, dict(finished=0, dist=dist, speed=avg_speed, quality=quality_score, progress=progress_pct)


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


def run_episode_with_controller(body_arch: BodyArchitecture, theta: np.ndarray, steps: int = 800) -> tuple[float, dict]:
    if not body_arch or not body_arch.viable:
        return -1e6, dict(finished=0, dist=0.0, quality=0.0)

    model = body_arch.model
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    mj.set_mjcb_control(None)

    INP = body_arch.inp_size
    OUT = body_arch.out_size
    params = unpack_controller_theta(theta, INP, CTRL_HIDDEN, OUT)
    if params is None:
        return -1e6, dict(finished=0, dist=0.0, quality=0.0)

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

    fitness, metrics = compute_quality_fitness(x_hist, t_hist, TRACK_LENGTH)
    return float(fitness), metrics

# =========================
# MODERATE controller search (50 samples instead of 100)
# =========================
def adaptive_controller_search(body_arch: BodyArchitecture, body_geno, generation: int):
    """
    ADAPTIVE: Spend more time on promising bodies, less on bad ones.
    Phase 1: Quick filter (30 samples)
    Phase 2: If promising (>20 fitness), do deep search (200 samples)
    """
    key = body_geno_to_key(body_geno)
    
    if key in _BEST_CTRL_CACHE:
        cached_theta, cached_fit = _BEST_CTRL_CACHE[key]
        # Only trust cache if it's good
        if cached_fit > 30.0:
            return cached_theta, cached_fit
    
    if not body_arch.viable:
        return None, -1e6
    
    INP, OUT = body_arch.inp_size, body_arch.out_size
    theta_size = controller_theta_size(INP, CTRL_HIDDEN, OUT)
    
    # PHASE 1: Quick filter (30 samples)
    best_theta = None
    best_fit = -1e9
    best_quality = 0.0
    
    scales = [0.3, 0.5, 0.7, 1.0]
    for i in range(CTRL_SAMPLES_MIN):
        scale = scales[i % len(scales)]
        theta = RNG.normal(0.0, scale, size=theta_size).astype(float)
        theta = np.clip(theta, CTRL_W_LOW, CTRL_W_HIGH)
        
        fit, metrics = run_episode_with_controller(body_arch, theta, steps=600)
        quality = metrics.get('quality', 0.0)
        
        if fit > best_fit or (abs(fit - best_fit) < 5.0 and quality > best_quality):
            best_fit = fit
            best_theta = theta.copy()
            best_quality = quality
    
    if best_theta is None:
        return None, -1e9
    
    # PHASE 2: Adaptive deep search
    # If body shows promise (fitness > 20), invest more samples
    if best_fit > 20.0:  # Promising body!
        # Calculate how many more samples based on fitness
        extra_samples = int((best_fit / 50.0) * (CTRL_SAMPLES_MAX - CTRL_SAMPLES_MIN))
        extra_samples = min(extra_samples, CTRL_SAMPLES_MAX - CTRL_SAMPLES_MIN)
        
        console.log(f"[Promising] fit={best_fit:.1f} → {CTRL_SAMPLES_MIN + extra_samples} total samples")
        
        # More exploration for promising bodies
        for i in range(extra_samples):
            scale = scales[i % len(scales)]
            theta = RNG.normal(0.0, scale, size=theta_size).astype(float)
            theta = np.clip(theta, CTRL_W_LOW, CTRL_W_HIGH)
            
            fit, metrics = run_episode_with_controller(body_arch, theta, steps=650)
            quality = metrics.get('quality', 0.0)
            
            if fit > best_fit or (abs(fit - best_fit) < 5.0 and quality > best_quality):
                best_fit = fit
                best_theta = theta.copy()
                best_quality = quality
        
        # Adaptive refinement
        refine_iters = CTRL_REFINE_MIN + int((best_fit / 50.0) * (CTRL_REFINE_MAX - CTRL_REFINE_MIN))
        refine_iters = min(refine_iters, CTRL_REFINE_MAX)
    else:
        # Bad body - minimal refinement
        refine_iters = CTRL_REFINE_MIN
    
    # Phase 3: Refinement
    step_size = 0.4
    for _ in range(refine_iters):
        improved = False
        for _ in range(CTRL_PARALLEL_CANDIDATES):
            noise = RNG.normal(0.0, step_size, size=theta_size)
            candidate = np.clip(best_theta + noise, CTRL_W_LOW, CTRL_W_HIGH)
            
            fit, metrics = run_episode_with_controller(body_arch, candidate, steps=650)
            quality = metrics.get('quality', 0.0)
            
            if fit > best_fit or (abs(fit - best_fit) < 3.0 and quality > best_quality):
                best_fit = fit
                best_theta = candidate.copy()
                best_quality = quality
                improved = True
        
        if not improved:
            step_size *= 0.85
    
    _BEST_CTRL_CACHE[key] = (best_theta, best_fit)
    return best_theta, best_fit


def evaluate_body_adaptive(ind, generation: int):
    """Adaptive evaluation: fast for bad bodies, deep for good ones."""
    geno = ind[0]
    key = body_geno_to_key(geno)
    arch = get_body_architecture(key, geno)
    
    if not arch.viable:
        return (-1e6,)
    
    try:
        best_theta, best_fit = adaptive_controller_search(arch, geno, generation)
        if best_theta is None:
            return (-1e6,)
        return (float(best_fit),)
    except Exception as e:
        console.log(f"[Eval] Error: {e}")
        return (-1e6,)


# =========================
# EA setup with PROPER SELECTION
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


def mutate_body_strong(ind):
    ind[0] = strong_mutation(ind[0], indpb=15.0 / BODY_L, rng=RNG)
    if hasattr(ind.fitness, "values"): del ind.fitness.values
    return (ind,)


toolbox.register("mate", mate_bodies)
toolbox.register("mutate", mutate_body_strong)
toolbox.register("evaluate", lambda ind: evaluate_body_adaptive(ind, generation=0))
toolbox.register("select", tools.selTournament, tournsize=BODY_TOURNSIZE)


def _hof_similar(a, b) -> bool:
    try:
        g1, g2 = a[0], b[0]
        v1 = np.concatenate([np.ravel(g1[0]), np.ravel(g1[1]), np.ravel(g1[2])])
        v2 = np.concatenate([np.ravel(g2[0]), np.ravel(g2[1]), np.ravel(g2[2])])
        return bool(np.allclose(v1, v2, atol=1e-12, rtol=1e-12))
    except Exception:
        return False


def probe_best_metrics(arch: BodyArchitecture, theta: np.ndarray) -> tuple[float, float, float]:
    if not arch or not arch.viable:
        return 0.0, 0.0, 0.0
    fit, metrics = run_episode_with_controller(arch, theta, steps=PROBE_STEPS)
    return metrics.get('dist', 0.0), metrics.get('quality', 0.0), metrics.get('speed', 0.0)


# =========================
# EA with FIXED SELECTION
# =========================
def run_fixed_selection_ea():
    """
    EA with PROPER selection pressure (tournament size 5, elite preservation).
    This should maintain and improve fitness over generations.
    """
    random.seed(SEED); np.random.seed(SEED)
    console.log(f"[EXPLORATION BOOST] Aggressive diversity maintenance")
    console.log(f"[Config] Pop={BODY_POP_SIZE} (larger), Gens={BODY_N_GEN}")
    console.log(f"[Config] Tournament={BODY_TOURNSIZE} (lower pressure), MutPB={BODY_MUTPB} (high!)")
    console.log(f"[Config] Mutation: ~15 genes/mut, Diversity detection enabled")
    console.log(f"[Speed] ~2min/gen\n")

    # Initialize
    pop = toolbox.population(n=BODY_POP_SIZE)
    
    console.log(f"[Gen 0] Evaluating {len(pop)} bodies...")
    t_start = time.time()
    for idx, ind in enumerate(pop):
        if (idx + 1) % 5 == 0:
            console.log(f"[Gen 0] Progress: {idx+1}/{len(pop)}")
        ind.fitness.values = toolbox.evaluate(ind)
    t_elapsed = time.time() - t_start
    console.log(f"[Gen 0] Complete in {t_elapsed/60:.1f} min")

    # Statistics
    hof = tools.HallOfFame(BODY_ELITE_K, similar=_hof_similar)
    hof.update(pop)
    
    best_per_gen = [max(ind.fitness.values[0] for ind in pop)]
    avg_per_gen = [np.mean([ind.fitness.values[0] for ind in pop])]
    dist_per_gen = []
    quality_per_gen = []
    
    # Initial telemetry
    best0 = max(pop, key=lambda x: x.fitness.values[0])
    try:
        built0 = build_body(best0[0], nde_modules=NUM_OF_MODULES, rng=RNG)
        save_body_artifacts(DATA, built0, tag="gen_000_best")
        
        best_key = body_geno_to_key(best0[0])
        best_arch = _BODY_ARCH_CACHE.get(best_key)
        if best_arch and best_arch.viable:
            cached = _BEST_CTRL_CACHE.get(best_key)
            if cached:
                theta_best = cached[0]
                dist, quality, _ = probe_best_metrics(best_arch, theta_best)
                dist_per_gen.append(dist)
                quality_per_gen.append(quality)
    except Exception:
        dist_per_gen.append(0.0)
        quality_per_gen.append(0.0)
    
    progress = (dist_per_gen[0] / TRACK_LENGTH) * 100
    console.log(
        f"[Gen 0] best={best_per_gen[0]:.1f} | avg={avg_per_gen[0]:.1f} | "
        f"dist={dist_per_gen[0]:.2f}m ({progress:.1f}%) | quality={quality_per_gen[0]:.2f}\n"
    )
    
    best_so_far = best_per_gen[0]
    gens_without_improvement = 0
    
    for gen in range(1, BODY_N_GEN + 1):
        console.log(f"[Gen {gen}] Starting...")
        t_gen_start = time.time()
        
        # PROPER SELECTION with tournament size 5
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # Crossover (ALL offspring, we'll select survivors later)
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < BODY_CXPB:
                toolbox.mate(c1, c2)
        
        # Mutation (ALL offspring) - AGGRESSIVE
        for m in offspring:
            if random.random() < BODY_MUTPB:
                toolbox.mutate(m)
        
        # AGGRESSIVE diversity injection when stuck
        # Detect convergence: if avg is close to best, population has converged!
        population_diversity = best_so_far - np.mean([ind.fitness.values[0] for ind in pop])
        
        if gens_without_improvement >= 3:  # EARLIER injection (was 5)
            console.log(f"[Gen {gen}] Stuck {gens_without_improvement} gens - AGGRESSIVE diversity injection")
            
            # Calculate how many to inject based on convergence
            if population_diversity < 2.0:  # Highly converged
                n_random = BODY_POP_SIZE // 3  # Replace 1/3 of population!
                console.log(f"[Gen {gen}] Population converged (div={population_diversity:.1f}) - injecting {n_random} randoms")
            else:
                n_random = max(3, BODY_POP_SIZE // 8)
            
            for _ in range(n_random):
                offspring.append(toolbox.individual())
        
        # Evaluate ALL offspring with generation number
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        console.log(f"[Gen {gen}] Evaluating {len(invalid)} new bodies (adaptive budget)...")
        for idx, ind in enumerate(invalid):
            if (idx + 1) % 5 == 0:
                console.log(f"[Gen {gen}] Progress: {idx+1}/{len(invalid)}")
            ind.fitness.values = evaluate_body_adaptive(ind, generation=gen)
        
        # CRITICAL FIX: (μ + λ) selection - parents + offspring compete!
        # Combine current population with offspring
        combined = pop + offspring
        
        # Update HOF with ALL individuals
        hof.update(combined)
        
        # Select best from combined pool (elitism is automatic!)
        pop = tools.selBest(combined, BODY_POP_SIZE)
        
        # VERIFY best never decreases (with proper (μ+λ), this is guaranteed)
        pop_fits = [ind.fitness.values[0] for ind in pop]
        if max(pop_fits) < best_so_far - 0.01:
            console.log(f"[BUG] Elite lost! best_so_far={best_so_far:.1f} but max(pop)={max(pop_fits):.1f}")
        else:
            console.log(f"[OK] Elite preserved: {max(pop_fits):.1f} >= {best_so_far:.1f}")
        
        # Track statistics
        best = max(pop, key=lambda ind: ind.fitness.values[0])
        best_fit = best.fitness.values[0]
        avg_fit = np.mean([ind.fitness.values[0] for ind in pop])
        min_fit = min([ind.fitness.values[0] for ind in pop])
        
        best_per_gen.append(best_fit)
        avg_per_gen.append(avg_fit)
        
        # Calculate diversity metric
        diversity = best_fit - avg_fit
        convergence_status = "🔴 CONVERGED" if diversity < 2.0 else "🟢 DIVERSE"
        
        # Check improvement
        if best_fit > best_so_far + 1.0:
            best_so_far = best_fit
            gens_without_improvement = 0
            console.log(f"[Gen {gen}] 🎉 NEW BEST! Fitness: {best_fit:.1f}")
        else:
            gens_without_improvement += 1
        
        # Detailed telemetry
        dist, quality, speed = 0.0, 0.0, 0.0
        try:
            best_key = body_geno_to_key(best[0])
            best_arch = _BODY_ARCH_CACHE.get(best_key)
            if best_arch and best_arch.viable:
                cached = _BEST_CTRL_CACHE.get(best_key)
                if cached:
                    theta_best = cached[0]
                    dist, quality, speed = probe_best_metrics(best_arch, theta_best)
        except Exception:
            pass
        
        dist_per_gen.append(dist)
        quality_per_gen.append(quality)
        progress = (dist / TRACK_LENGTH) * 100
        
        t_gen_elapsed = time.time() - t_gen_start
        
        # Check if avg fitness is improving (not collapsing)
        avg_change = avg_fit - avg_per_gen[-2] if len(avg_per_gen) > 1 else 0
        avg_status = "✓" if avg_change >= -0.5 else "⚠"
        
        finished = "🏁" if dist >= TRACK_LENGTH - 0.05 else ""
        console.log(
            f"[Gen {gen}] best={best_fit:.1f} | avg={avg_fit:.1f} (min={min_fit:.1f}) {avg_status} (Δ{avg_change:+.1f}) | "
            f"div={diversity:.1f} {convergence_status} | "
            f"dist={dist:.2f}m ({progress:.1f}%) {finished} | "
            f"qual={quality:.2f} | stuck={gens_without_improvement} | t={t_gen_elapsed/60:.1f}min\n"
        )
        
        # Save every 5 gens
        if gen % 5 == 0:
            try:
                built = build_body(best[0], nde_modules=NUM_OF_MODULES, rng=RNG)
                save_body_artifacts(DATA, built, tag=f"gen_{gen:03d}_best")
            except Exception:
                pass
        
        # Early stopping
        if dist >= TRACK_LENGTH - 0.05:
            console.log(f"[SUCCESS] Robot reached finish at gen {gen}!")
            break
    
    # Save final
    final_best = max(pop, key=lambda ind: ind.fitness.values[0])
    try:
        built_final = build_body(final_best[0], nde_modules=NUM_OF_MODULES, rng=RNG)
        save_body_artifacts(DATA, built_final, tag="final_best")
    except Exception:
        pass
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        
        # Fitness
        axes[0].plot(best_per_gen, marker="o", linewidth=2, markersize=4, label="Best", color="blue")
        axes[0].plot(avg_per_gen, marker="s", linewidth=2, markersize=3, label="Avg", color="cyan", alpha=0.7)
        axes[0].set_ylabel("Fitness", fontsize=11)
        axes[0].set_title("Fitness Progress (Best & Avg)", fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Distance
        axes[1].plot(dist_per_gen, marker="s", linewidth=2, markersize=4, color="green")
        axes[1].axhline(y=TRACK_LENGTH, color='r', linestyle='--', label=f"Goal ({TRACK_LENGTH:.1f}m)")
        axes[1].set_ylabel("Distance (m)", fontsize=11)
        axes[1].set_title("Distance Progress", fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Quality
        axes[2].plot(quality_per_gen, marker="^", linewidth=2, markersize=4, color="orange")
        axes[2].set_ylabel("Movement Quality", fontsize=11)
        axes[2].set_title("Quality Score (0=wiggling, 1=smooth)", fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Average fitness change (detect collapse)
        avg_changes = [0] + [avg_per_gen[i] - avg_per_gen[i-1] for i in range(1, len(avg_per_gen))]
        axes[3].plot(avg_changes, marker="d", linewidth=2, markersize=4, color="purple")
        axes[3].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[3].set_xlabel("Generation", fontsize=11)
        axes[3].set_ylabel("Avg Fitness Change", fontsize=11)
        axes[3].set_title("Population Health (should stay ≥0)", fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(EA_FITNESS_PNG, dpi=150)
        plt.close()
        console.log(f"[Plot] Saved to {EA_FITNESS_PNG}")
    except Exception as e:
        console.log(f"[Plot] Error: {e}")
    
    return final_best, best_per_gen, avg_per_gen, dist_per_gen, quality_per_gen


# =========================
# Main
# =========================
def main():
    console.log("="*70)
    console.log("[ADAPTIVE BUDGET] Smart resource allocation for speed")
    console.log("="*70)
    console.log(f"Goal: Reach {TRACK_LENGTH:.2f}m FASTER")
    console.log(f"Strategy: 30 samples for bad bodies, 200 for promising ones")
    console.log(f"Fitness: dist² × 50 (pure distance optimization)")
    console.log(f"Expected: 2-3x faster generations, same or better results")
    console.log("="*70 + "\n")
    
    best, fit_curve, avg_curve, dist_curve, qual_curve = run_fixed_selection_ea()
    
    console.log("\n" + "="*70)
    console.log("[FINAL RESULTS]")
    console.log("="*70)
    console.log(f"Best fitness: {best.fitness.values[0]:.1f}")
    console.log(f"Final avg fitness: {avg_curve[-1]:.1f} (should be stable, not collapsing)")
    console.log(f"Final distance: {dist_curve[-1]:.2f}m / {TRACK_LENGTH:.2f}m ({(dist_curve[-1]/TRACK_LENGTH)*100:.1f}%)")
    console.log(f"Final quality: {qual_curve[-1]:.2f}")
    
    if dist_curve[-1] >= TRACK_LENGTH - 0.05:
        console.log("🎉🎉🎉 SUCCESS! Robot completed the track!")
    else:
        console.log(f"Distance remaining: {TRACK_LENGTH - dist_curve[-1]:.2f}m")
    
    # Check for fitness collapse
    if len(avg_curve) > 5:
        avg_trend = avg_curve[-1] - avg_curve[-5]
        if avg_trend < -2.0:
            console.log(f"⚠ WARNING: Average fitness dropped {avg_trend:.1f} over last 5 gens (selection may still be weak)")
        else:
            console.log(f"✓ Selection working: avg fitness change = {avg_trend:+.1f} over last 5 gens")
    
    console.log(f"\nFitness evolution: {[round(f, 1) for f in fit_curve[::max(1, len(fit_curve)//10)]]}")
    console.log(f"Average evolution: {[round(a, 1) for a in avg_curve[::max(1, len(avg_curve)//10)]]}")
    console.log(f"Distance evolution: {[round(d, 2) for d in dist_curve[::max(1, len(dist_curve)//10)]]}")
    console.log("="*70)


if __name__ == "__main__":
    main()