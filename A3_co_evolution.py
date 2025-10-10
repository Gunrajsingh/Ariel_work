"""
Assignment 3 — Robot Olympics (co-evolution with DETERMINISTIC body analysis)

CRITICAL FIX:
- The issue was that the same body genotype was producing different architectures 
  when built multiple times due to stochastic elements in the NDE process
- This caused dimension mismatches between controller analysis and actual use
- Solution: Build the body ONCE, cache its architecture, and reuse for all controller evaluations

APPROACH:
- Deterministic body analysis with caching
- More conservative controller evolution (larger population, more generations)
- Focus on getting stable, working solutions rather than optimal ones
- Simplified architecture with robust fallbacks

PARAMETERS:
- Body: 12 population × 6 generations (smaller, more focused)  
- Controller: 10 population × 6 generations (more thorough)
- Simulation: 8 seconds each (good evaluation time)
- Estimated total runtime: ~3-4 hours
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
# Global settings - CONSERVATIVE BUT RELIABLE
# =========================
SCRIPT_NAME = "A3_co_evolution_stable"
CWD = Path(".").resolve()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# EA parameters - Conservative for reliability
BODY_POP_SIZE = 12  # Smaller population, more focused
BODY_N_GEN = 6      # Fewer generations, better solutions
BODY_TOURNSIZE = 3
BODY_CXPB = 0.7
BODY_MUTPB = 0.15
BODY_SBX_ETA = 20.0
BODY_IMM_FRAC = 0.20
BODY_STAGNATION_STEPS = (3, 5)
BODY_MUTPB_BOOSTS = (1.8, 2.5)

# Controller EA - More thorough evolution per body
CTRL_POP_SIZE = 10  # Larger population for better search
CTRL_N_GEN = 6      # More generations for better controllers
CTRL_TOURNSIZE = 3
CTRL_CXPB = 0.8
CTRL_MUTPB = 0.2
CTRL_SBX_ETA = 15.0
CTRL_IMM_FRAC = 0.20

# Sim + environment - Good evaluation time
SIM_DURATION = 8     # Longer evaluation for stability
RATE_LIMIT_DU = 0.025
HINGE_LIMIT = math.pi / 2
WARMUP_STEPS = int(0.4 * 240)  # Good warmup time
HARD_TIMEOUT = 15
SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]

# Body encoding
BODY_L = 64
NUM_OF_MODULES = 30

# Controller encoding - Conservative
CTRL_HIDDEN = 10    # Smaller network for stability
CTRL_W_LOW, CTRL_W_HIGH = -1.5, 1.5  # Conservative bounds

# Plot outputs
EA_FITNESS_PNG = DATA / "ea_best_over_generations.png"

# =========================
# Body architecture caching system
# =========================
@dataclass
class BodyArchitecture:
    """Cached information about a body's controller requirements."""
    inp_size: int
    out_size: int 
    viable: bool
    error_msg: str
    world: Optional[Any] = None  # Keep world for reuse
    model: Optional[Any] = None  # Keep model for reuse
    track_body_name: Optional[str] = None

# Global cache for body architectures
_BODY_ARCH_CACHE = {}

def get_body_architecture(body_geno_key: str, body_geno) -> BodyArchitecture:
    """
    Get or create cached body architecture information.
    This ensures the same genotype always produces the same architecture.
    """
    if body_geno_key in _BODY_ARCH_CACHE:
        return _BODY_ARCH_CACHE[body_geno_key]
    
    # Build body deterministically
    try:
        # Use fixed random seed for deterministic building
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
        
        # Reset and check
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        
        # Check basic viability
        if model.nu == 0 or model.nv == 0 or model.nbody < 2:
            arch = BodyArchitecture(0, 0, False, f"Invalid model: nu={model.nu}, nv={model.nv}, nbody={model.nbody}")
            _BODY_ARCH_CACHE[body_geno_key] = arch
            return arch
        
        # Find core body
        track_body_name = _find_core_body_name(model)
        if not track_body_name:
            arch = BodyArchitecture(0, 0, False, "No core body found")
            _BODY_ARCH_CACHE[body_geno_key] = arch
            return arch
        
        # Check if robot is reasonable
        try:
            core_body = data.body(track_body_name)
            if core_body.xpos[2] < -0.5:
                arch = BodyArchitecture(0, 0, False, "Robot below ground")
                _BODY_ARCH_CACHE[body_geno_key] = arch
                return arch
        except Exception as e:
            arch = BodyArchitecture(0, 0, False, f"Cannot access core body: {e}")
            _BODY_ARCH_CACHE[body_geno_key] = arch
            return arch
        
        # Calculate controller dimensions
        inp_size = len(data.qpos) + len(data.qvel) + 3  # qpos + qvel + time features
        out_size = model.nu
        
        arch = BodyArchitecture(inp_size, out_size, True, "OK", world, model, track_body_name)
        _BODY_ARCH_CACHE[body_geno_key] = arch
        console.log(f"[Architecture] Cached body: inp={inp_size}, out={out_size}, joints={len(data.qpos)}, actuators={out_size}")
        return arch
        
    except Exception as e:
        arch = BodyArchitecture(0, 0, False, str(e))
        _BODY_ARCH_CACHE[body_geno_key] = arch
        return arch

def body_geno_to_key(body_geno) -> str:
    """Convert body genotype to a hashable cache key."""
    t, c, r = body_geno
    # Create a deterministic hash from the genotype arrays
    combined = np.concatenate([t.flatten(), c.flatten(), r.flatten()])
    return str(hash(combined.tobytes()))

def _find_core_body_name(model):
    """Find the core body, with robust fallback."""
    # Try robot-core first
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "robot-core")
    if bid != -1:
        return "robot-core"
    
    # Look for any body with 'core' in name
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
    
    # Fallback: first non-world body
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        if name and "world" not in name.lower():
            return name
    
    return None

# =========================
# Controller genome and operators - DETERMINISTIC
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
        y = np.tanh(h2 @ W3 + b3)  # [-1, 1]
        return y
    except Exception as e:
        console.log(f"[MLP Error] {e}")
        return np.zeros(1)

def init_controller_genotype_for_body(inp_size, out_size, rng: np.random.Generator):
    """Initialize random controller weights for specific body dimensions."""
    w_size = controller_theta_size(inp_size, CTRL_HIDDEN, out_size)
    
    # Xavier initialization with conservative scaling
    scale = np.sqrt(1.0 / inp_size)  # Conservative initialization
    theta = rng.normal(0.0, scale, size=w_size).astype(float)
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

def controller_polynomial_mutation(individual, eta=20.0, indpb=0.1, rng=None):
    """Polynomial bounded mutation for controller weights."""
    if rng is None:
        rng = np.random.default_rng()
    
    individual = np.asarray(individual, dtype=float)
    mutated = individual.copy()
    
    for i in range(len(individual)):
        if rng.random() < indpb:
            u = rng.random()
            delta_1 = (individual[i] - CTRL_W_LOW) / (CTRL_W_HIGH - CTRL_W_LOW)
            delta_2 = (CTRL_W_HIGH - individual[i]) / (CTRL_W_HIGH - CTRL_W_LOW)
            
            # Clamp deltas to valid range
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
# Body genotype and operators
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
    indpb: float = 1.5 / BODY_L,  # Conservative mutation rate
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
# Body building
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
        # Return a placeholder for failed builds
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
# Episode evaluation - USING CACHED ARCHITECTURE
# =========================
def run_episode_with_controller(body_geno_key: str, controller_theta, duration: int = SIM_DURATION):
    """
    Run episode using cached body architecture.
    This ensures consistent dimensions throughout.
    """
    try:
        # Get cached architecture
        arch = _BODY_ARCH_CACHE.get(body_geno_key)
        if not arch or not arch.viable:
            return -1e6, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        
        # Use the cached world and model
        world = arch.world
        model = arch.model
        track_body_name = arch.track_body_name
        
        # Create fresh data
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        
        # Reset MuJoCo callback
        mj.set_mjcb_control(None)
        
        # Set up tracker
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind=track_body_name)
        
        # Record start position
        start_pos = np.array(data.body(track_body_name).xpos[:3], dtype=float).copy()
        
        # Unpack controller with cached dimensions
        controller_params = unpack_controller_theta(controller_theta, arch.inp_size, CTRL_HIDDEN, arch.out_size)
        if controller_params is None:
            return -1e6, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        
        def _episode_controller():
            step = 0
            # Setup control bounds
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
            base_rate = 0.02 * (high - low)
            rate = np.minimum(base_rate, RATE_LIMIT_DU)
            
            def _cb(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
                nonlocal u_apply, step
                try:
                    t_now = d.time
                    
                    # Create input vector: qpos + qvel + time features
                    qpv = np.concatenate([d.qpos, d.qvel]).astype(float, copy=False)
                    time_features = np.array([t_now, math.sin(2 * math.pi * t_now), math.cos(2 * math.pi * t_now)], dtype=float)
                    x_in = np.concatenate([qpv, time_features])
                    
                    # Verify input size matches cached architecture
                    if len(x_in) != arch.inp_size:
                        console.log(f"[ERROR] Input size mismatch: {len(x_in)} != {arch.inp_size}")
                        return np.zeros(arch.out_size, dtype=np.float64)
                    
                    # Forward pass through controller
                    y_out = controller_mlp_forward(x_in, controller_params)
                    y_out = y_out.flatten()
                    
                    # Verify output size
                    if len(y_out) != arch.out_size:
                        console.log(f"[ERROR] Output size mismatch: {len(y_out)} != {arch.out_size}")
                        y_out = np.resize(y_out, arch.out_size)
                    
                    # Clean output
                    y_out = np.nan_to_num(y_out, nan=0.0, posinf=0.0, neginf=0.0)
                    u_target = center + halfspan * y_out
                    
                    # Warm-up ramp
                    ramp = 1.0 if WARMUP_STEPS <= 0 else min(1.0, step / max(1, WARMUP_STEPS))
                    u_cmd = center + ramp * (u_target - center)
                    
                    # Rate limiting
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
        
        # Run simulation
        simple_runner(model, data, duration=duration)
        
        # Get final position and compute displacement
        end_pos = np.array(data.body(track_body_name).xpos[:3], dtype=float)
        
        # Get tracker history or fallback
        hist = tracker.history.get("xpos", [])
        if not hist or len(hist) < 2:
            hist = [start_pos.tolist(), end_pos.tolist()]
        
        # Compute fitness as x-displacement
        displacement = float(end_pos[0] - start_pos[0])
        
        # Bound the fitness to reasonable values
        displacement = np.clip(displacement, -10.0, 10.0)
        
        return displacement, hist
        
    except Exception as e:
        console.log(f"[Episode Error] {e}")
        return -1e6, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

# =========================
# Controller evolution - CACHED ARCHITECTURE
# =========================
def evolve_controller_for_body(body_geno, verbose=False):
    """
    Evolve a controller for the given body using cached architecture.
    """
    # Get cache key and architecture
    body_geno_key = body_geno_to_key(body_geno)
    arch = get_body_architecture(body_geno_key, body_geno)
    
    if not arch.viable:
        if verbose:
            console.log(f"[Controller EA] Body not viable: {arch.error_msg}")
        return None, -1e6
    
    if verbose:
        console.log(f"[Controller EA] Body viable - inp:{arch.inp_size}, out:{arch.out_size}")
    
    # Create DEAP types for controller evolution
    try:
        creator.ControllerFitnessMax
    except AttributeError:
        creator.create("ControllerFitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.ControllerIndividual
    except AttributeError:
        creator.create("ControllerIndividual", list, fitness=creator.ControllerFitnessMax)
    
    # Initialize controller population with correct dimensions
    def init_controller_individual():
        theta = init_controller_genotype_for_body(arch.inp_size, arch.out_size, RNG)
        return creator.ControllerIndividual(theta.tolist())
    
    # Create population
    ctrl_pop = [init_controller_individual() for _ in range(CTRL_POP_SIZE)]
    
    if verbose:
        expected_size = controller_theta_size(arch.inp_size, CTRL_HIDDEN, arch.out_size)
        console.log(f"[Controller EA] Created population with theta size: {len(ctrl_pop[0])} (expected: {expected_size})")
    
    # Evaluate initial population
    for ind in ctrl_pop:
        try:
            fitness, _ = run_episode_with_controller(body_geno_key, np.array(ind), SIM_DURATION)
            ind.fitness.values = (fitness,)
        except Exception as e:
            if verbose:
                console.log(f"[Controller EA] Initial eval error: {e}")
            ind.fitness.values = (-1e6,)
    
    # Track best fitness
    best_fitness = max(ind.fitness.values[0] for ind in ctrl_pop)
    if verbose:
        console.log(f"[Controller EA] Initial best: {best_fitness:.4f}")
    
    # Controller evolution loop
    for gen in range(CTRL_N_GEN):
        # Selection
        offspring = tools.selTournament(ctrl_pop, len(ctrl_pop), tournsize=CTRL_TOURNSIZE)
        offspring = [creator.ControllerIndividual(ind[:]) for ind in offspring]
        
        # Crossover
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
        
        # Mutation
        for mutant in offspring:
            if random.random() < CTRL_MUTPB:
                try:
                    mutated = controller_polynomial_mutation(mutant, eta=20.0, indpb=0.1, rng=RNG)
                    mutant[:] = mutated.tolist()
                    del mutant.fitness.values
                except Exception as e:
                    if verbose:
                        console.log(f"[Controller EA] Mutation error: {e}")
        
        # Random immigrants with correct dimensions
        n_imm = max(0, int(CTRL_IMM_FRAC * len(offspring)))
        for _ in range(n_imm):
            offspring[random.randrange(len(offspring))] = init_controller_individual()
        
        # Evaluate offspring
        for ind in offspring:
            if not hasattr(ind.fitness, 'values') or not ind.fitness.valid:
                try:
                    fitness, _ = run_episode_with_controller(body_geno_key, np.array(ind), SIM_DURATION)
                    ind.fitness.values = (fitness,)
                except Exception as e:
                    if verbose:
                        console.log(f"[Controller EA] Eval error: {e}")
                    ind.fitness.values = (-1e6,)
        
        # Replace population
        ctrl_pop = offspring
        
        # Track progress
        gen_best = max(ind.fitness.values[0] for ind in ctrl_pop)
        if gen_best > best_fitness:
            best_fitness = gen_best
            if verbose and gen_best > -1e6:
                console.log(f"[Controller EA] Gen {gen}: IMPROVEMENT! best = {gen_best:.4f}")
    
    # Return best controller and fitness
    best_ctrl = tools.selBest(ctrl_pop, 1)[0]
    final_fitness = best_ctrl.fitness.values[0]
    
    if verbose:
        console.log(f"[Controller EA] Final best: {final_fitness:.4f}")
    
    return np.array(best_ctrl), final_fitness

# =========================
# DEAP scaffolding for body evolution
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
    ind[0] = block_mutation(g, indpb=1.5 / BODY_L, rng=RNG)
    if hasattr(ind.fitness, "values"):
        del ind.fitness.values
    return (ind,)

toolbox.register("mate", mate_bodies)
toolbox.register("mutate", mutate_body)

def evaluate_body_individual(ind):
    """Evaluate a body by evolving a cached controller for it."""
    geno = ind[0]
    try:
        best_controller, fitness = evolve_controller_for_body(geno, verbose=False)
        if best_controller is None:
            return (-1e6,)
        
        # Additional check: make sure fitness is reasonable
        if not np.isfinite(fitness):
            fitness = -1e6
        
        fitness = float(np.clip(fitness, -1e6, 1e6))  # Bound fitness
        
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
# Rendering helpers
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
# Main EA loop - DETERMINISTIC
# =========================
def run_co_evolution():
    random.seed(SEED)
    np.random.seed(SEED)
    
    console.log(f"[Co-Evolution] STABLE VERSION - Deterministic body architecture caching")
    console.log(f"[Co-Evolution] Body: {BODY_POP_SIZE} pop × {BODY_N_GEN} gen")
    console.log(f"[Co-Evolution] Controller: {CTRL_POP_SIZE} pop × {CTRL_N_GEN} gen")
    console.log(f"[Co-Evolution] Simulation: {SIM_DURATION}s each")
    
    # Initialize body population
    body_pop = toolbox.population(n=BODY_POP_SIZE)
    
    # Evaluate initial population with progress tracking
    console.log("[Co-Evolution] Evaluating initial population...")
    invalid = [ind for ind in body_pop if not ind.fitness.valid]
    
    viable_count = 0
    for i, ind in enumerate(invalid):
        console.log(f"[Co-Evolution] Evaluating body {i+1}/{len(invalid)}")
        fitness_tuple = toolbox.evaluate(ind)
        ind.fitness.values = fitness_tuple
        
        if fitness_tuple[0] > -1e6:
            viable_count += 1
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
    
    # Render/save snapshot of gen 0 best (if viable)
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
    
    # Main evolution loop
    for gen in range(1, BODY_N_GEN + 1):
        console.log(f"[Co-Evolution] Generation {gen}/{BODY_N_GEN}")
        
        # Selection + cloning
        offspring = list(map(toolbox.clone, toolbox.select(body_pop, len(body_pop))))
        
        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < BODY_CXPB:
                toolbox.mate(c1, c2)
        
        # Adaptive mutation
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
        
        # Random immigrants
        n_imm = max(0, int(BODY_IMM_FRAC * len(offspring)))
        for _ in range(n_imm):
            offspring[random.randrange(len(offspring))] = toolbox.individual()
        
        # Evaluate
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        
        gen_viable_count = 0
        for i, ind in enumerate(invalid):
            console.log(f"[Co-Evolution] Gen {gen} - Evaluating body {i+1}/{len(invalid)}")
            fitness_tuple = toolbox.evaluate(ind)
            ind.fitness.values = fitness_tuple
            
            if fitness_tuple[0] > -1e6:
                gen_viable_count += 1
                console.log(f"  -> Viable! Fitness: {fitness_tuple[0]:.4f}")
        
        console.log(f"  -> Gen {gen}: {gen_viable_count}/{len(invalid)} viable offspring")
        console.log(f"  -> Architecture cache size: {len(_BODY_ARCH_CACHE)}")
        
        # Elitism
        hof.update(offspring)
        elites = list(map(toolbox.clone, hof.items))
        
        offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        body_pop = elites + offspring[: max(0, BODY_POP_SIZE - len(elites))]
        
        # Logging
        best = tools.selBest(body_pop, 1)[0]
        best_per_gen.append(best.fitness.values[0])
        
        improvement = best.fitness.values[0] - best_so_far
        if improvement > 1e-6:  # More lenient improvement threshold
            best_so_far = best.fitness.values[0]
            no_improve = 0
            console.log(f"  -> IMPROVEMENT! +{improvement:.4f}")
        else:
            no_improve += 1
        
        dt_wall = time.time() - t_wall
        console.log(
            f"[Co-Evolution] Gen {gen:3d} | best = {best.fitness.values[0]:.4f} | "
            f"no_improve={no_improve:2d} | t={dt_wall:.1f}s"
        )
        
        # Save artifacts + per-gen snapshot (if viable)
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
    
    # Plot curve
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(best_per_gen, marker="o", linewidth=2, markersize=6)
        plt.xlabel("Generation")
        plt.ylabel("Best fitness (x-displacement)")
        plt.title(f"Co-Evolution STABLE: Best fitness over {BODY_N_GEN} generations")
        plt.grid(True, alpha=0.3)
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
    console.log("[Co-Evolution] Starting STABLE co-evolution with deterministic caching...")
    best, curve = run_co_evolution()
    console.log(f"[Co-Evolution] Done. Best fitness = {best.fitness.values[0]:.4f}")
    
    # Save final results
    if best.fitness.values[0] > -1e6:
        # Evolve final controller for best body
        console.log("[Co-Evolution] Evolving final controller for best body...")
        best_controller, best_fitness = evolve_controller_for_body(best[0], verbose=True)
        
        if best_controller is not None:
            body_key = body_geno_to_key(best[0])
            arch = _BODY_ARCH_CACHE[body_key]
            
            # Save the best controller
            controller_data = {
                "controller_type": "mlp_co_evolved_stable",
                "robot": "evolved_body",
                "environment": "OlympicArena", 
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
                "generations": {
                    "body_gens": BODY_N_GEN,
                    "ctrl_gens": CTRL_N_GEN,
                    "body_pop": BODY_POP_SIZE,
                    "ctrl_pop": CTRL_POP_SIZE
                },
                "cache_stats": {
                    "total_architectures_cached": len(_BODY_ARCH_CACHE),
                    "viable_architectures": sum(1 for arch in _BODY_ARCH_CACHE.values() if arch.viable)
                }
            }
            
            with open(DATA / "best_controller_stable.json", "w") as f:
                json.dump(controller_data, f, indent=2)
            
            console.log(f"[Co-Evolution] Saved best controller to {DATA / 'best_controller_stable.json'}")
            console.log(f"[Co-Evolution] Controller architecture: {arch.inp_size} → {CTRL_HIDDEN} → {arch.out_size}")
            console.log(f"[Co-Evolution] Final controller fitness: {best_fitness:.4f}")
            console.log(f"[Co-Evolution] Total architectures cached: {len(_BODY_ARCH_CACHE)}")
    else:
        console.log("[Co-Evolution] No viable solutions found!")

if __name__ == "__main__":
    main()