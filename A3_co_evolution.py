"""
Assignment 3 — Robot Olympics (co-evolution with DISTANCE-FOCUSED FITNESS)

This version implements a GOAL-ORIENTED fitness function that strongly motivates robots
to reach the end of the course at position [5, 0, 0.5].

DISTANCE-FOCUSED FITNESS IMPROVEMENTS:
1. Primary reward: Distance toward goal (exponentially increasing)
2. Progress reward: Incremental reward for each meter forward
3. Speed bonus: Reward for reaching goal faster
4. Minimum movement requirement: Penalty for staying stationary
5. Simple but effective: Focus purely on forward locomotion performance

GOAL: Make robots travel from [-0.8, 0, 0.1] to [5, 0, 0.5] = 5.8 meters forward

PARAMETERS:
- Body: 12 population × 6 generations (focused)  
- Controller: 12 population × 8 generations (thorough training)
- Simulation: 15 seconds each (more time to reach goal)
- Estimated total runtime: ~5-6 hours
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

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
# Global settings - GOAL-FOCUSED
# =========================
SCRIPT_NAME = "A3_co_evolution_goal_focused"
CWD = Path(".").resolve()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# EA parameters - Body evolution (focused)
BODY_POP_SIZE = 12  
BODY_N_GEN = 6      
BODY_TOURNSIZE = 3
BODY_CXPB = 0.7
BODY_MUTPB = 0.18   
BODY_SBX_ETA = 20.0
BODY_IMM_FRAC = 0.25  
BODY_STAGNATION_STEPS = (3, 5)
BODY_MUTPB_BOOSTS = (1.8, 2.5)

# Controller EA - FOCUSED ON PERFORMANCE
CTRL_POP_SIZE = 12  
CTRL_N_GEN = 8      
CTRL_TOURNSIZE = 3
CTRL_CXPB = 0.8
CTRL_MUTPB = 0.25   
CTRL_SBX_ETA = 15.0
CTRL_IMM_FRAC = 0.20

# Sim + environment - GOAL-ORIENTED
SIM_DURATION = 15    # MORE TIME to reach the goal
RATE_LIMIT_DU = 0.05  # Even more aggressive control for speed
HINGE_LIMIT = math.pi / 2
WARMUP_STEPS = int(0.2 * 240)  # Even shorter warmup for quick action
HARD_TIMEOUT = 15
SPAWN_POS = [-0.8, 0, 0.1]  # Start position
TARGET_POSITION = [5, 0, 0.5]  # GOAL POSITION - 5.8 meters forward!

# Body encoding
BODY_L = 64
NUM_OF_MODULES = 30

# Controller encoding - High capacity for complex behaviors
CTRL_HIDDEN = 16    
CTRL_W_LOW, CTRL_W_HIGH = -2.5, 2.5  # Even less conservative for dynamic movement

# Plot outputs
EA_FITNESS_PNG = DATA / "ea_best_over_generations.png"

# =========================
# GOAL-FOCUSED FITNESS FUNCTION
# =========================
def compute_goal_focused_fitness(start_pos, end_pos, history, duration):
    """
    Goal-focused fitness that strongly rewards progress toward target [5, 0, 0.5].
    
    The course is: from [-0.8, 0, 0.1] to [5, 0, 0.5] = 5.8 meters forward
    
    Components (in order of importance):
    1. DISTANCE PROGRESS: Exponentially increasing reward for forward progress
    2. GOAL ACHIEVEMENT: Massive bonus for reaching near the goal
    3. SPEED BONUS: Extra reward for reaching goal quickly
    4. MOVEMENT REQUIREMENT: Penalty for barely moving
    5. SURVIVAL: Small penalty for falling/instability only if it prevents progress
    """
    
    # Calculate forward progress (x-direction is primary)
    start_x = float(start_pos[0])  # Should be around -0.8
    end_x = float(end_pos[0])
    goal_x = TARGET_POSITION[0]  # 5.0
    
    # Forward distance covered
    forward_distance = end_x - start_x
    
    # Distance to goal (how close are we to the finish line?)
    distance_to_goal = goal_x - end_x  # Positive = still need to go forward
    
    # 1. EXPONENTIAL PROGRESS REWARD - This is the main driver
    # We want MASSIVE rewards for each additional meter forward
    if forward_distance > 0:
        # Exponential reward: 1m = 1 point, 2m = 4 points, 3m = 9 points, etc.
        progress_reward = forward_distance * forward_distance
        # Additional linear component for consistent progress
        progress_reward += 2.0 * forward_distance
    else:
        # Heavy penalty for moving backward
        progress_reward = 3.0 * forward_distance  # Amplify backward penalty
    
    # 2. GOAL ACHIEVEMENT BONUS - Massive bonus for getting close to target
    if distance_to_goal <= 0:
        # REACHED OR PASSED THE GOAL! 
        goal_bonus = 50.0 + max(0, -distance_to_goal * 5.0)  # Extra bonus for going past
    elif distance_to_goal <= 1.0:
        # Very close to goal
        goal_bonus = 30.0 * (1.0 - distance_to_goal)  # Up to 30 points for being within 1m
    elif distance_to_goal <= 2.0:  
        # Close to goal
        goal_bonus = 10.0 * (2.0 - distance_to_goal)  # Up to 10 points for being within 2m
    else:
        goal_bonus = 0.0
    
    # 3. SPEED BONUS - Reward for reaching goal faster
    if distance_to_goal <= 1.0:  # Only give speed bonus if robot is near goal
        # More time remaining = higher speed bonus
        time_efficiency = max(0, (duration - time.time() if hasattr(time, 'time') else duration * 0.3))
        speed_bonus = min(5.0, time_efficiency * 0.5)
    else:
        speed_bonus = 0.0
    
    # 4. MINIMUM MOVEMENT REQUIREMENT - Prevent robots from just standing
    total_distance_traveled = 0.0
    if len(history) >= 2:
        for i in range(1, len(history)):
            if len(history[i]) >= 3 and len(history[i-1]) >= 3:
                dx = history[i][0] - history[i-1][0]
                dy = history[i][1] - history[i-1][1]
                total_distance_traveled += np.sqrt(dx*dx + dy*dy)
    
    # Require at least some movement - penalty for being stationary
    if total_distance_traveled < 0.1:
        movement_penalty = -2.0  # Penalty for barely moving
    else:
        movement_penalty = 0.0
    
    # 5. BASIC SURVIVAL CHECK - Only penalize if robot completely fails
    final_height = float(end_pos[2])
    if final_height < -0.5:  # Robot completely fell through floor
        survival_penalty = -5.0
    else:
        survival_penalty = 0.0
    
    # COMBINE ALL COMPONENTS - Heavily weighted toward distance progress
    total_fitness = (
        progress_reward +      # PRIMARY: Quadratic + linear progress reward
        goal_bonus +          # SECONDARY: Massive bonus near goal  
        speed_bonus +         # TERTIARY: Speed efficiency
        movement_penalty +    # CONSTRAINT: Must move  
        survival_penalty      # CONSTRAINT: Must not completely fail
    )
    
    # Log promising individuals with detailed breakdown
    if total_fitness > 0.5 or forward_distance > 0.5:
        console.log(f"[Goal-Focused] fwd_dist={forward_distance:.3f}m, to_goal={distance_to_goal:.3f}m, "
                   f"prog_rew={progress_reward:.2f}, goal_bon={goal_bonus:.2f}, "
                   f"total={total_fitness:.2f} | POS: {end_x:.3f}")
    
    return total_fitness

# =========================
# Body architecture caching system (unchanged)
# =========================
@dataclass
class BodyArchitecture:
    """Cached information about a body's controller requirements."""
    inp_size: int
    out_size: int 
    viable: bool
    error_msg: str
    world: Optional[Any] = None
    model: Optional[Any] = None
    track_body_name: Optional[str] = None

# Global cache for body architectures
_BODY_ARCH_CACHE = {}

def get_body_architecture(body_geno_key: str, body_geno) -> BodyArchitecture:
    """Get or create cached body architecture information."""
    if body_geno_key in _BODY_ARCH_CACHE:
        return _BODY_ARCH_CACHE[body_geno_key]
    
    try:
        temp_rng = np.random.default_rng(hash(body_geno_key) % 2**32)
        
        world = OlympicArena()
        built = build_body(body_geno, nde_modules=NUM_OF_MODULES, rng=temp_rng)
        
        if not built.viable:
            arch = BodyArchitecture(0, 0, False, built.error_msg)
            _BODY_ARCH_CACHE[body_geno_key] = arch
            return arch
        
        world.spawn(built.mjspec.spec, position=SPAWN_POS)
        model = world.spec.compile()
        data = mj.MjData(model)
        
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        
        if model.nu == 0 or model.nv == 0 or model.nbody < 2:
            arch = BodyArchitecture(0, 0, False, f"Invalid model: nu={model.nu}, nv={model.nv}, nbody={model.nbody}")
            _BODY_ARCH_CACHE[body_geno_key] = arch
            return arch
        
        track_body_name = _find_core_body_name(model)
        if not track_body_name:
            arch = BodyArchitecture(0, 0, False, "No core body found")
            _BODY_ARCH_CACHE[body_geno_key] = arch
            return arch
        
        try:
            core_body = data.body(track_body_name)
            start_height = core_body.xpos[2]
            if start_height < -0.2:
                arch = BodyArchitecture(0, 0, False, f"Robot starts too low: {start_height:.3f}")
                _BODY_ARCH_CACHE[body_geno_key] = arch
                return arch
            if start_height > 2.0:
                arch = BodyArchitecture(0, 0, False, f"Robot starts too high: {start_height:.3f}")
                _BODY_ARCH_CACHE[body_geno_key] = arch  
                return arch
        except Exception as e:
            arch = BodyArchitecture(0, 0, False, f"Cannot access core body: {e}")
            _BODY_ARCH_CACHE[body_geno_key] = arch
            return arch
        
        inp_size = len(data.qpos) + len(data.qvel) + 3
        out_size = model.nu
        
        arch = BodyArchitecture(inp_size, out_size, True, "OK", world, model, track_body_name)
        _BODY_ARCH_CACHE[body_geno_key] = arch
        console.log(f"[Architecture] Cached viable body: inp={inp_size}, out={out_size}, height={start_height:.3f}")
        return arch
        
    except Exception as e:
        arch = BodyArchitecture(0, 0, False, str(e))
        _BODY_ARCH_CACHE[body_geno_key] = arch
        return arch

def body_geno_to_key(body_geno) -> str:
    """Convert body genotype to a hashable cache key."""
    t, c, r = body_geno
    combined = np.concatenate([t.flatten(), c.flatten(), r.flatten()])
    return str(hash(combined.tobytes()))

def _find_core_body_name(model):
    """Find the core body, with robust fallback."""
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "robot-core")
    if bid != -1:
        return "robot-core"
    
    candidates = []
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        if not name or "world" in name.lower():
            continue
        if "core" in name.lower():
            candidates.append(name)
    
    if candidates:
        candidates.sort(key=len)
        return candidates[0]
    
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        if name and "world" not in name.lower():
            return name
    
    return None

# =========================
# Controller genome and operators - AGGRESSIVE SETTINGS
# =========================
def controller_theta_size(inp, hidden=CTRL_HIDDEN, out_dim=1):
    """Calculate size of flattened NN weights vector."""
    return inp*hidden + hidden + hidden*hidden + hidden + hidden*out_dim + out_dim

def unpack_controller_theta(theta, inp, hidden=CTRL_HIDDEN, out_dim=1):
    """Unpack flattened weights into network layers."""
    theta = np.asarray(theta, dtype=float)
    expected = controller_theta_size(inp, hidden, out_dim)
    if theta.size != expected:
        console.log(f"[ERROR] unpack_controller_theta: theta size {theta.size} != expected {expected}")
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
    """Forward pass through MLP."""
    if params is None:
        return np.zeros(1)
    
    try:
        W1, b1, W2, b2, W3, b3 = params
        h1 = np.tanh(x @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        y = np.tanh(h2 @ W3 + b3)
        return y
    except Exception as e:
        console.log(f"[MLP Error] {e}")
        return np.zeros(1)

def init_controller_genotype_for_body(inp_size, out_size, rng: np.random.Generator):
    """Initialize controller weights for AGGRESSIVE LOCOMOTION."""
    w_size = controller_theta_size(inp_size, CTRL_HIDDEN, out_size)
    
    # More aggressive initialization for dynamic movement
    w1_scale = np.sqrt(3.0 / (inp_size + CTRL_HIDDEN))  # Slightly larger
    w2_scale = np.sqrt(3.0 / (CTRL_HIDDEN + CTRL_HIDDEN))  
    w3_scale = np.sqrt(2.0 / (CTRL_HIDDEN + out_size))
    
    theta = np.zeros(w_size, dtype=float)
    i = 0
    
    # W1 weights - More variation for diverse behaviors
    n_w1 = inp_size * CTRL_HIDDEN
    theta[i:i+n_w1] = rng.normal(0.0, w1_scale, size=n_w1)
    i += n_w1
    
    # b1 biases - Small positive bias to encourage activation
    theta[i:i+CTRL_HIDDEN] = rng.normal(0.1, 0.1, size=CTRL_HIDDEN)  # Higher variance
    i += CTRL_HIDDEN
    
    # W2 weights
    n_w2 = CTRL_HIDDEN * CTRL_HIDDEN  
    theta[i:i+n_w2] = rng.normal(0.0, w2_scale, size=n_w2)
    i += n_w2
    
    # b2 biases
    theta[i:i+CTRL_HIDDEN] = rng.normal(0.05, 0.1, size=CTRL_HIDDEN)
    i += CTRL_HIDDEN
    
    # W3 weights - Output layer
    n_w3 = CTRL_HIDDEN * out_size
    theta[i:i+n_w3] = rng.normal(0.0, w3_scale, size=n_w3)
    i += n_w3
    
    # b3 biases - Small bias toward forward movement
    theta[i:i+out_size] = rng.normal(0.02, 0.05, size=out_size)  # Slight forward bias
    
    theta = np.clip(theta, CTRL_W_LOW, CTRL_W_HIGH)
    
    return theta

def controller_sbx_crossover(parent1, parent2, eta=CTRL_SBX_ETA, rng=None):
    """SBX crossover for controller weights."""
    if rng is None:
        rng = np.random.default_rng()
    
    parent1 = np.asarray(parent1, dtype=float)
    parent2 = np.asarray(parent2, dtype=float)
    
    if parent1.shape != parent2.shape:
        console.log(f"[SBX Error] Shape mismatch: {parent1.shape} vs {parent2.shape}")
        return parent1.copy(), parent2.copy()
    
    u = rng.random(parent1.shape)
    beta = np.empty_like(parent1)
    
    mask = u <= 0.5
    beta[mask] = (2.0 * u[mask]) ** (1.0 / (eta + 1.0))
    beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (eta + 1.0))
    
    child1 = 0.5 * ((parent1 + parent2) - beta * (parent2 - parent1))
    child2 = 0.5 * ((parent1 + parent2) + beta * (parent2 - parent1))
    
    child1 = np.clip(child1, CTRL_W_LOW, CTRL_W_HIGH)
    child2 = np.clip(child2, CTRL_W_LOW, CTRL_W_HIGH)
    
    return child1, child2

def controller_polynomial_mutation(individual, eta=12.0, indpb=0.15, rng=None):
    """Polynomial mutation with AGGRESSIVE parameters for exploration."""
    if rng is None:
        rng = np.random.default_rng()
    
    individual = np.asarray(individual, dtype=float)
    mutated = individual.copy()
    
    for i in range(len(individual)):
        if rng.random() < indpb:
            u = rng.random()
            delta_1 = (individual[i] - CTRL_W_LOW) / (CTRL_W_HIGH - CTRL_W_LOW)
            delta_2 = (CTRL_W_HIGH - individual[i]) / (CTRL_W_HIGH - CTRL_W_LOW)
            
            delta_1 = np.clip(delta_1, 1e-8, 1.0 - 1e-8)
            delta_2 = np.clip(delta_2, 1e-8, 1.0 - 1e-8)
            
            if u <= 0.5:
                mut_pow = 1.0 / (eta + 1.0)
                delta_q = (2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta_1) ** (eta + 1.0)) ** mut_pow - 1.0
            else:
                mut_pow = 1.0 / (eta + 1.0)
                delta_q = 1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta_2) ** (eta + 1.0)) ** mut_pow
            
            mutated[i] = individual[i] + delta_q * (CTRL_W_HIGH - CTRL_W_LOW)
            mutated[i] = np.clip(mutated[i], CTRL_W_LOW, CTRL_W_HIGH)
    
    return mutated

# =========================
# Body genotype and operators (unchanged)
# =========================
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

def sbx_body(
    g1: tuple[np.ndarray, np.ndarray, np.ndarray],
    g2: tuple[np.ndarray, np.ndarray, np.ndarray],
    eta: float = BODY_SBX_ETA,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    a1, b1 = _sbx_pair(g1[0], g2[0], eta, rng=rng)
    a2, b2 = _sbx_pair(g1[1], g2[1], eta, rng=rng)
    a3, b3 = _sbx_pair(g1[2], g2[2], eta, rng=rng)
    return (a1, a2, a3), (b1, b2, b3)

def block_mutation(
    g: tuple[np.ndarray, np.ndarray, np.ndarray],
    indpb: float = 2.5 / BODY_L,  # Slightly higher for more exploration
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
# Body building (unchanged)
# =========================
@dataclass
class BuiltBody:
    nde: NeuralDevelopmentalEncoding
    decoded_graph: Any
    mjspec: Any
    viable: bool
    error_msg: str

def build_body(geno: tuple[np.ndarray, np.ndarray, np.ndarray], nde_modules: int, rng: np.random.Generator) -> BuiltBody:
    """Build body with comprehensive error handling."""
    try:
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
        
        return BuiltBody(nde=nde, decoded_graph=graph, mjspec=spec, viable=True, error_msg="OK")
        
    except Exception as e:
        dummy_nde = NeuralDevelopmentalEncoding(number_of_modules=1)
        return BuiltBody(nde=dummy_nde, decoded_graph=None, mjspec=None, viable=False, error_msg=str(e))

def save_body_artifacts(run_dir: Path, built: BuiltBody, tag: str):
    """Save body artifacts with error handling."""
    if not built.viable:
        return
        
    try:
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
            "error_msg": built.error_msg
        }
        with open(run_dir / f"{tag}_nde.json", "w") as f:
            json.dump(nde_json, f, indent=2)
    except Exception as e:
        console.log(f"[Save Error] {e}")

# =========================
# Episode evaluation - GOAL-FOCUSED FITNESS
# =========================
def run_episode_with_controller(body_geno_key: str, controller_theta, duration: int = SIM_DURATION):
    """Run episode with GOAL-FOCUSED FITNESS FUNCTION."""
    try:
        arch = _BODY_ARCH_CACHE.get(body_geno_key)
        if not arch or not arch.viable:
            return -1e6, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        
        world = arch.world
        model = arch.model
        track_body_name = arch.track_body_name
        
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
            # VERY AGGRESSIVE CONTROL for fast movement
            ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=float).reshape(arch.out_size, 2)
            low = ctrlrange[:, 0].copy()
            high = ctrlrange[:, 1].copy()
            limited = np.array(model.actuator_ctrllimited, dtype=bool).reshape(-1)
            bad = (~limited) | ~np.isfinite(low) | ~np.isfinite(high) | (high <= low)
            low[bad] = -HINGE_LIMIT
            high[bad] = HINGE_LIMIT
            center = 0.5 * (low + high)
            halfspan = 0.5 * (high - low)
            u_apply = center.copy()
            base_rate = 0.05 * (high - low)  # Very aggressive
            rate = np.minimum(base_rate, RATE_LIMIT_DU)
            
            def _cb(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
                nonlocal u_apply, step
                try:
                    t_now = d.time
                    
                    qpv = np.concatenate([d.qpos, d.qvel]).astype(float, copy=False)
                    time_features = np.array([t_now, math.sin(2 * math.pi * t_now), math.cos(2 * math.pi * t_now)], dtype=float)
                    x_in = np.concatenate([qpv, time_features])
                    
                    if len(x_in) != arch.inp_size:
                        return np.zeros(arch.out_size, dtype=np.float64)
                    
                    y_out = controller_mlp_forward(x_in, controller_params)
                    y_out = y_out.flatten()
                    
                    if len(y_out) != arch.out_size:
                        y_out = np.resize(y_out, arch.out_size)
                    
                    y_out = np.nan_to_num(y_out, nan=0.0, posinf=0.0, neginf=0.0)
                    u_target = center + halfspan * y_out
                    
                    # VERY SHORT WARMUP for immediate action
                    ramp = 1.0 if WARMUP_STEPS <= 0 else min(1.0, step / max(1, WARMUP_STEPS))
                    u_cmd = center + ramp * (u_target - center)
                    
                    # AGGRESSIVE RATE LIMITING
                    du = np.clip(u_cmd - u_apply, -rate, rate)
                    u_apply = np.clip(u_apply + du, low, high)
                    step += 1
                    
                    return u_apply.astype(np.float64, copy=False)
                    
                except Exception as e:
                    console.log(f"[Controller Callback Error] {e}")
                    return np.zeros(arch.out_size, dtype=np.float64)
            
            return _cb
        
        episode_cb = _episode_controller()
        ctrl = Controller(controller_callback_function=episode_cb, tracker=tracker)
        
        if ctrl.tracker is not None:
            ctrl.tracker.setup(world.spec, data)
        
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
        
        # Run simulation for LONGER DURATION
        simple_runner(model, data, duration=duration)
        
        end_pos = np.array(data.body(track_body_name).xpos[:3], dtype=float)
        
        hist = tracker.history.get("xpos", [])
        if not hist or len(hist) < 2:
            hist = [start_pos.tolist(), end_pos.tolist()]
        
        # GOAL-FOCUSED FITNESS CALCULATION
        fitness = compute_goal_focused_fitness(start_pos, end_pos, hist, duration)
        
        fitness = np.clip(fitness, -100.0, 1000.0)  # Wider range for big rewards
        
        return fitness, hist
        
    except Exception as e:
        console.log(f"[Episode Error] {e}")
        return -1e6, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

# =========================
# Controller evolution (unchanged logic)
# =========================
def evolve_controller_for_body(body_geno, verbose=False):
    """Evolve a controller for the given body."""
    body_geno_key = body_geno_to_key(body_geno)
    arch = get_body_architecture(body_geno_key, body_geno)
    
    if not arch.viable:
        if verbose:
            console.log(f"[Controller EA] Body not viable: {arch.error_msg}")
        return None, -1e6
    
    if verbose:
        console.log(f"[Controller EA] Body viable - inp:{arch.inp_size}, out:{arch.out_size}")
    
    try:
        creator.ControllerFitnessMax
    except AttributeError:
        creator.create("ControllerFitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.ControllerIndividual
    except AttributeError:
        creator.create("ControllerIndividual", list, fitness=creator.ControllerFitnessMax)
    
    def init_controller_individual():
        theta = init_controller_genotype_for_body(arch.inp_size, arch.out_size, RNG)
        return creator.ControllerIndividual(theta.tolist())
    
    ctrl_pop = [init_controller_individual() for _ in range(CTRL_POP_SIZE)]
    
    if verbose:
        expected_size = controller_theta_size(arch.inp_size, CTRL_HIDDEN, arch.out_size)
        console.log(f"[Controller EA] Created population with theta size: {len(ctrl_pop[0])} (expected: {expected_size})")
    
    for ind in ctrl_pop:
        try:
            fitness, _ = run_episode_with_controller(body_geno_key, np.array(ind), SIM_DURATION)
            ind.fitness.values = (fitness,)
        except Exception as e:
            if verbose:
                console.log(f"[Controller EA] Initial eval error: {e}")
            ind.fitness.values = (-1e6,)
    
    best_fitness = max(ind.fitness.values[0] for ind in ctrl_pop)
    if verbose:
        console.log(f"[Controller EA] Initial best: {best_fitness:.4f}")
    
    for gen in range(CTRL_N_GEN):
        offspring = tools.selTournament(ctrl_pop, len(ctrl_pop), tournsize=CTRL_TOURNSIZE)
        offspring = [creator.ControllerIndividual(ind[:]) for ind in offspring]
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CTRL_CXPB:
                try:
                    c1, c2 = controller_sbx_crossover(child1, child2, eta=CTRL_SBX_ETA, rng=RNG)
                    child1[:] = c1.tolist()
                    child2[:] = c2.tolist()
                    del child1.fitness.values
                    del child2.fitness.values
                except Exception as e:
                    if verbose:
                        console.log(f"[Controller EA] Crossover error: {e}")
        
        for mutant in offspring:
            if random.random() < CTRL_MUTPB:
                try:
                    mutated = controller_polynomial_mutation(mutant, eta=12.0, indpb=0.15, rng=RNG)
                    mutant[:] = mutated.tolist()
                    del mutant.fitness.values
                except Exception as e:
                    if verbose:
                        console.log(f"[Controller EA] Mutation error: {e}")
        
        n_imm = max(0, int(CTRL_IMM_FRAC * len(offspring)))
        for _ in range(n_imm):
            offspring[random.randrange(len(offspring))] = init_controller_individual()
        
        for ind in offspring:
            if not hasattr(ind.fitness, 'values') or not ind.fitness.valid:
                try:
                    fitness, _ = run_episode_with_controller(body_geno_key, np.array(ind), SIM_DURATION)
                    ind.fitness.values = (fitness,)
                except Exception as e:
                    if verbose:
                        console.log(f"[Controller EA] Eval error: {e}")
                    ind.fitness.values = (-1e6,)
        
        ctrl_pop = offspring
        
        gen_best = max(ind.fitness.values[0] for ind in ctrl_pop)
        if gen_best > best_fitness:
            best_fitness = gen_best
            if verbose and gen_best > 0.5:
                console.log(f"[Controller EA] Gen {gen}: BREAKTHROUGH! best = {gen_best:.4f}")
    
    best_ctrl = tools.selBest(ctrl_pop, 1)[0]
    final_fitness = best_ctrl.fitness.values[0]
    
    if verbose:
        console.log(f"[Controller EA] Final best: {final_fitness:.4f}")
    
    return np.array(best_ctrl), final_fitness

# =========================
# DEAP scaffolding (unchanged)
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
    g1 = ind1[0]
    g2 = ind2[0]
    c1, c2 = sbx_body(g1, g2, eta=BODY_SBX_ETA, rng=RNG)
    ind1[0] = c1
    ind2[0] = c2
    if hasattr(ind1.fitness, "values"):
        del ind1.fitness.values
    if hasattr(ind2.fitness, "values"):
        del ind2.fitness.values
    return ind1, ind2

def mutate_body(ind):
    g = ind[0]
    ind[0] = block_mutation(g, indpb=2.5 / BODY_L, rng=RNG)
    if hasattr(ind.fitness, "values"):
        del ind.fitness.values
    return (ind,)

toolbox.register("mate", mate_bodies)
toolbox.register("mutate", mutate_body)

def evaluate_body_individual(ind):
    """Evaluate a body by evolving a goal-focused controller for it."""
    geno = ind[0]
    try:
        best_controller, fitness = evolve_controller_for_body(geno, verbose=False)
        if best_controller is None:
            return (-1e6,)
        
        if not np.isfinite(fitness):
            fitness = -1e6
        
        fitness = float(np.clip(fitness, -1e6, 1e6))
        
        return (fitness,)
    except Exception as e:
        console.log(f"[Body Eval] Exception: {e}")
        return (-1e6,)

toolbox.register("evaluate", evaluate_body_individual)

def _hof_similar(a, b) -> bool:
    try:
        g1 = a[0]
        g2 = b[0]
        v1 = np.concatenate([np.ravel(g1[0]), np.ravel(g1[1]), np.ravel(g1[2])])
        v2 = np.concatenate([np.ravel(g2[0]), np.ravel(g2[1]), np.ravel(g2[2])])
        return bool(np.allclose(v1, v2, atol=1e-12, rtol=1e-12))
    except Exception:
        return False

toolbox.register("select", tools.selTournament, tournsize=BODY_TOURNSIZE)

# =========================
# Rendering helpers (unchanged)
# =========================
def _topdown_camera() -> mj.MjvCamera:
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat = [2.5, 0.0, 0.0]
    cam.distance = 10.0
    cam.azimuth = 0
    cam.elevation = -90
    return cam

def render_snapshot(world, save_path: str | None = None):
    """Compile a fresh model and render one frame - with error handling."""
    try:
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
    except Exception as e:
        console.log(f"[Render Error] {e}")

# =========================
# Main EA loop - GOAL FOCUSED
# =========================
def run_co_evolution():
    random.seed(SEED)
    np.random.seed(SEED)
    
    console.log(f"[Co-Evolution] GOAL-FOCUSED VERSION - Targeting position {TARGET_POSITION}")
    console.log(f"[Co-Evolution] Course: {SPAWN_POS} → {TARGET_POSITION} = {TARGET_POSITION[0] - SPAWN_POS[0]:.1f}m forward")
    console.log(f"[Co-Evolution] Body: {BODY_POP_SIZE} pop × {BODY_N_GEN} gen")
    console.log(f"[Co-Evolution] Controller: {CTRL_POP_SIZE} pop × {CTRL_N_GEN} gen")
    console.log(f"[Co-Evolution] Simulation: {SIM_DURATION}s each (goal-focused fitness)")
    
    body_pop = toolbox.population(n=BODY_POP_SIZE)
    
    console.log("[Co-Evolution] Evaluating initial population...")
    invalid = [ind for ind in body_pop if not ind.fitness.valid]
    
    viable_count = 0
    for i, ind in enumerate(invalid):
        console.log(f"[Co-Evolution] Evaluating body {i+1}/{len(invalid)}")
        fitness_tuple = toolbox.evaluate(ind)
        ind.fitness.values = fitness_tuple
        
        if fitness_tuple[0] > -1e6:
            viable_count += 1
            if fitness_tuple[0] > 1.0:
                console.log(f"  -> PROMISING body found! Fitness: {fitness_tuple[0]:.4f}")
            else:
                console.log(f"  -> Viable body found! Fitness: {fitness_tuple[0]:.4f}")
    
    console.log(f"[Co-Evolution] Initial population: {viable_count}/{len(invalid)} viable bodies")
    console.log(f"[Co-Evolution] Architecture cache size: {len(_BODY_ARCH_CACHE)}")
    
    ELITE_K = max(1, BODY_POP_SIZE // 6)
    hof = tools.HallOfFame(ELITE_K, similar=_hof_similar)
    hof.update(body_pop)
    
    best_per_gen = [tools.selBest(body_pop, 1)[0].fitness.values[0]]
    
    no_improve = 0
    best_so_far = best_per_gen[-1]
    
    console.log(f"[Co-Evolution] Initial best fitness: {best_so_far:.4f}")
    
    # Success thresholds
    if best_so_far > 10.0:
        console.log("🎉 AMAZING START! Robot shows strong forward progress!")
    elif best_so_far > 2.0:
        console.log("🚀 GREAT START! Robot shows forward movement!")
    elif best_so_far > 0.5:
        console.log("📈 GOOD START! Robot shows some forward progress!")
    
    best0 = tools.selBest(body_pop, 1)[0]
    if best0.fitness.values[0] > -1e6:
        body_key = body_geno_to_key(best0[0])
        arch = _BODY_ARCH_CACHE.get(body_key)
        if arch and arch.viable:
            built0 = build_body(best0[0], nde_modules=NUM_OF_MODULES, rng=RNG)
            save_body_artifacts(DATA, built0, tag="gen_000_best")
            try:
                render_snapshot(arch.world, save_path=str(DATA / "gen_000_best.png"))
                console.log("[Co-Evolution] Saved initial best snapshot")
            except Exception as e:
                console.log(f"[Render init] {e}")
    
    t_wall = time.time()
    
    for gen in range(1, BODY_N_GEN + 1):
        console.log(f"[Co-Evolution] Generation {gen}/{BODY_N_GEN}")
        
        offspring = list(map(toolbox.clone, toolbox.select(body_pop, len(body_pop))))
        
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < BODY_CXPB:
                toolbox.mate(c1, c2)
        
        adapt_mutpb = BODY_MUTPB
        if no_improve >= BODY_STAGNATION_STEPS[0]:
            adapt_mutpb = min(1.0, BODY_MUTPB * BODY_MUTPB_BOOSTS[0])
            console.log(f"  -> Boosting mutation rate to {adapt_mutpb:.3f}")
        if no_improve >= BODY_STAGNATION_STEPS[1]:
            adapt_mutpb = min(1.0, BODY_MUTPB * BODY_MUTPB_BOOSTS[1])
            console.log(f"  -> Boosting mutation rate to {adapt_mutpb:.3f}")
        
        for m in offspring:
            if random.random() < adapt_mutpb:
                toolbox.mutate(m)
        
        n_imm = max(0, int(BODY_IMM_FRAC * len(offspring)))
        for _ in range(n_imm):
            offspring[random.randrange(len(offspring))] = toolbox.individual()
        
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        
        gen_viable_count = 0
        promising_count = 0
        for i, ind in enumerate(invalid):
            console.log(f"[Co-Evolution] Gen {gen} - Evaluating body {i+1}/{len(invalid)}")
            fitness_tuple = toolbox.evaluate(ind)
            ind.fitness.values = fitness_tuple
            
            if fitness_tuple[0] > -1e6:
                gen_viable_count += 1
                if fitness_tuple[0] > 2.0:
                    promising_count += 1
                    console.log(f"  -> BREAKTHROUGH! Fitness: {fitness_tuple[0]:.4f}")
                elif fitness_tuple[0] > 0.5:
                    console.log(f"  -> Promising! Fitness: {fitness_tuple[0]:.4f}")
        
        console.log(f"  -> Gen {gen}: {gen_viable_count}/{len(invalid)} viable offspring ({promising_count} promising)")
        console.log(f"  -> Architecture cache size: {len(_BODY_ARCH_CACHE)}")
        
        hof.update(offspring)
        elites = list(map(toolbox.clone, hof.items))
        
        offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        body_pop = elites + offspring[: max(0, BODY_POP_SIZE - len(elites))]
        
        best = tools.selBest(body_pop, 1)[0]
        best_per_gen.append(best.fitness.values[0])
        
        improvement = best.fitness.values[0] - best_so_far
        if improvement > 0.1:  # Meaningful improvement threshold
            best_so_far = best.fitness.values[0]
            no_improve = 0
            console.log(f"  -> MAJOR IMPROVEMENT! +{improvement:.4f} | NEW BEST: {best_so_far:.4f}")
            
            # Distance analysis
            if best_so_far > 50:
                console.log("    🏆 GOAL ACHIEVED! Robot reached the target!")
            elif best_so_far > 20:
                console.log("    🚀 EXCELLENT! Robot moved several meters forward!")
            elif best_so_far > 5:
                console.log("    📈 GREAT! Robot shows strong forward movement!")
            elif best_so_far > 1:
                console.log("    ✅ GOOD! Robot moving in right direction!")
                
        else:
            no_improve += 1
        
        dt_wall = time.time() - t_wall
        console.log(
            f"[Co-Evolution] Gen {gen:3d} | best = {best.fitness.values[0]:.4f} | "
            f"no_improve={no_improve:2d} | t={dt_wall:.1f}s"
        )
        
        if best.fitness.values[0] > -1e6:
            body_key = body_geno_to_key(best[0])
            arch = _BODY_ARCH_CACHE.get(body_key)
            if arch and arch.viable:
                built = build_body(best[0], nde_modules=NUM_OF_MODULES, rng=RNG)
                tag = f"gen_{gen:03d}_best"
                save_body_artifacts(DATA, built, tag=tag)
                
                try:
                    render_snapshot(arch.world, save_path=str(DATA / f"{tag}.png"))
                except Exception as e:
                    console.log(f"[Render gen {gen}] {e}")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        plt.plot(best_per_gen, marker="o", linewidth=2, markersize=6)
        plt.xlabel("Generation")
        plt.ylabel("Best fitness (goal-focused)")
        plt.title(f"Goal-Focused Co-Evolution: Progress toward target {TARGET_POSITION}")
        plt.grid(True, alpha=0.3)
        
        # Add reference lines for progress milestones
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No movement')
        plt.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='1m+ forward')
        plt.axhline(y=5, color='blue', linestyle='--', alpha=0.5, label='Strong movement')
        plt.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Major progress')
        plt.axhline(y=50, color='purple', linestyle='--', alpha=0.5, label='Goal achieved!')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(EA_FITNESS_PNG, dpi=150)
        plt.close()
        console.log(f"[Co-Evolution] Saved curve to {EA_FITNESS_PNG}")
    except Exception as e:
        console.log(f"[Plot] {e}")
    
    return tools.selBest(body_pop, 1)[0], best_per_gen

# =========================
# Main
# =========================
def main():
    console.log("[Co-Evolution] Starting GOAL-FOCUSED co-evolution...")
    console.log(f"[Co-Evolution] Target: Move {TARGET_POSITION[0] - SPAWN_POS[0]:.1f}m forward to reach {TARGET_POSITION}")
    
    best, curve = run_co_evolution()
    final_fitness = best.fitness.values[0]
    
    console.log(f"[Co-Evolution] Done. Best fitness = {final_fitness:.4f}")
    
    # Result analysis
    if final_fitness > 50:
        console.log("🏆 MISSION ACCOMPLISHED! Robot successfully reached the goal!")
    elif final_fitness > 20:
        console.log("🚀 EXCELLENT RESULT! Robot made major progress toward goal!")
    elif final_fitness > 5:
        console.log("📈 GREAT RESULT! Robot shows strong forward locomotion!")
    elif final_fitness > 1:
        console.log("✅ GOOD RESULT! Robot achieved forward movement!")
    elif final_fitness > 0:
        console.log("📊 PROGRESS! Robot shows some forward movement!")
    else:
        console.log("🔄 LEARNING! Robot needs more evolution to achieve forward movement!")
    
    # Calculate estimated distance
    # Using rough approximation: fitness ≈ distance² + 2*distance for positive movement
    if final_fitness > 0:
        # Solve quadratic: x² + 2x - fitness = 0 → x = (-2 + √(4 + 4*fitness)) / 2
        est_distance = (-2 + np.sqrt(4 + 4 * final_fitness)) / 2
        console.log(f"[Co-Evolution] Estimated forward distance: {est_distance:.2f} meters")
        console.log(f"[Co-Evolution] Progress: {100 * est_distance / 5.8:.1f}% toward goal")
    
    if best.fitness.values[0] > -1e6:
        console.log("[Co-Evolution] Evolving final controller for best body...")
        best_controller, best_fitness = evolve_controller_for_body(best[0], verbose=True)
        
        if best_controller is not None:
            body_key = body_geno_to_key(best[0])
            arch = _BODY_ARCH_CACHE[body_key]
            
            controller_data = {
                "controller_type": "mlp_co_evolved_goal_focused",
                "robot": "evolved_body",
                "environment": "OlympicArena", 
                "goal": f"Reach {TARGET_POSITION} from {SPAWN_POS}",
                "target_distance": f"{TARGET_POSITION[0] - SPAWN_POS[0]:.1f}m",
                "input_features": ["qpos", "qvel", "t", "sin(2πt)", "cos(2πt)"],
                "architecture": {
                    "input_size": int(arch.inp_size),
                    "hidden_size": CTRL_HIDDEN,
                    "output_size": int(arch.out_size)
                },
                "w_bounds": [CTRL_W_LOW, CTRL_W_HIGH],
                "theta_len": len(best_controller),
                "theta": [float(x) for x in best_controller],
                "final_fitness": float(best_fitness),
                "fitness_description": "goal_focused (exponential progress + goal bonus + speed bonus)",
                "optimizations_applied": [
                    "Goal-focused fitness function",
                    "Exponential progress rewards",
                    "Longer simulation (15s)",
                    "Aggressive control parameters",
                    "Forward-biased initialization"
                ],
                "generations": {
                    "body_gens": BODY_N_GEN,
                    "ctrl_gens": CTRL_N_GEN,
                    "body_pop": BODY_POP_SIZE,
                    "ctrl_pop": CTRL_POP_SIZE
                },
                "performance_summary": {
                    "best_fitness": float(best_fitness),
                    "estimated_distance_forward": float((-2 + np.sqrt(4 + 4 * max(0, best_fitness))) / 2) if best_fitness > 0 else 0.0,
                    "goal_progress_percent": float(100 * max(0, (-2 + np.sqrt(4 + 4 * max(0, best_fitness))) / 2) / 5.8) if best_fitness > 0 else 0.0
                },
                "cache_stats": {
                    "total_architectures_cached": len(_BODY_ARCH_CACHE),
                    "viable_architectures": sum(1 for arch in _BODY_ARCH_CACHE.values() if arch.viable)
                }
            }
            
            with open(DATA / "best_controller_goal_focused.json", "w") as f:
                json.dump(controller_data, f, indent=2)
            
            console.log(f"[Co-Evolution] Saved best controller to {DATA / 'best_controller_goal_focused.json'}")
            console.log(f"[Co-Evolution] Controller architecture: {arch.inp_size} → {CTRL_HIDDEN} → {arch.out_size}")
            console.log(f"[Co-Evolution] Final controller fitness: {best_fitness:.4f}")
            console.log(f"[Co-Evolution] Total architectures cached: {len(_BODY_ARCH_CACHE)}")
    else:
        console.log("[Co-Evolution] No viable solutions found!")

if __name__ == "__main__":
    main()