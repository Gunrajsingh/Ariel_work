"""
Assignment 2 — Gecko robot
1) Runs random controller N times, collects fitness, plots distribution.
2) Evolves oscillator controller with DEAP, plots best-per-generation curve.
3) Renders the best EA individual in a Mujoco viewer.

The fitness/comparison metric is unified via `compute_displacement` and, by default,
is the robot's forward displacement projected onto its initial facing direction.
"""

import math
import numpy as np
import random
import time
import mujoco
import matplotlib.pyplot as plt
from deap import base, creator, tools

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Mujoco viewer
from mujoco import viewer

# ----------------------------
# Global sim settings
# ----------------------------
HINGE_LIMIT = math.pi / 2
DT = 1 / 240
STEPS = 4000

# Unified fitness mode: 'projected' | 'x' | 'euclidean'
FITNESS_MODE = "projected"

# Output files
RAND_LINE_PNG = "random_fitness_line.png"
EA_FITNESS_PNG = "ea_best_over_generations.png"

# Rate limiter (per-step max change, radians)
RATE_LIMIT_DU = 0.015  # tune 0.01–0.03 as needed

# ----------------------------
# Unified fitness/comparison metric
# ----------------------------
def compute_displacement(history_xy, forward_xy0, mode="projected"):
    """
    history_xy: list/array of XY positions (start..end)
    forward_xy0: unit XY vector of initial facing (ignored if mode != 'projected')
    mode: 'projected' | 'x' | 'euclidean'
    returns scalar fitness in meters
    """
    disp = np.asarray(history_xy[-1]) - np.asarray(history_xy[0])
    if mode == "projected":
        return float(np.dot(disp, forward_xy0))
    elif mode == "x":
        return float(disp[0])
    elif mode == "euclidean":
        return float(np.linalg.norm(disp))
    else:
        raise ValueError(f"Unknown mode: {mode}")

# ----------------------------
# Helpers for robust body binding & facing
# ----------------------------
def _find_core_body_name(model):
    """
    Prefer 'robot-core'. If not present, pick a body whose name contains 'core'
    but is not 'world', with a shortest-name heuristic.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot-core")
    if bid != -1:
        return "robot-core"

    candidates = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        if "core" in name and "world" not in name:
            candidates.append(name)

    if not candidates:
        # Fallback: root body (index 0)
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0)

    candidates.sort(key=len)
    return candidates[0]


def _bind_core_body_and_forward_xy(model, data):
    """
    Returns (core_body, forward_xy0). forward_xy0 is the XY projection of the
    body's local +X axis in world frame at t=0 (normalized).
    """
    core_name = _find_core_body_name(model)
    core_body = data.body(core_name)
    fwd3 = np.array(core_body.xmat[0:3], dtype=float)  # local +X in world coords
    n = np.linalg.norm(fwd3)
    if n > 0:
        fwd3 /= n
    return core_body, fwd3[:2].copy()

# ----------------------------
# Random controller
# ----------------------------
def random_move(data, rng):
    delta = 0.05
    u = data.ctrl.copy()
    u = u + rng.uniform(-HINGE_LIMIT, HINGE_LIMIT, size=data.ctrl.shape) * delta
    u = np.clip(u, -HINGE_LIMIT, HINGE_LIMIT)
    return u

def run_random_episode(rng, steps):
    """One episode with random control, fitness = unified displacement metric."""
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    # Settle & bind core body + initial facing
    mujoco.mj_forward(model, data)
    core_body, forward_xy0 = _bind_core_body_and_forward_xy(model, data)

    history = [core_body.xpos[:2].copy()]  # log initial position BEFORE stepping
    for _ in range(steps):
        u = random_move(data, rng)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        history.append(core_body.xpos[:2].copy())

    # Unified metric
    fit = compute_displacement(history, forward_xy0, mode=FITNESS_MODE)
    return fit

# ----------------------------
# Rendering helpers
# ----------------------------
def render_episode_with_genome_exact(genes, steps=STEPS):
    """
    Replay exactly what EA evaluated (same t*DT timing), and keep the window open
    until the user closes it.
    """
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore
    mujoco.mj_forward(model, data)

    genes = np.asarray(genes, dtype=float)
    A      = genes[0::4]
    F      = genes[1::4]
    PHASE  = genes[2::4]
    OFFSET = genes[3::4]
    assert len(A) == model.nu, "Genome size does not match actuator count."

    with viewer.launch_passive(model, data) as v:
        for t in range(steps):
            if not v.is_running():
                return
            t_sec = t * DT
            u_cmd = OFFSET + A * np.sin(2.0 * math.pi * F * t_sec + PHASE)
            data.ctrl[:] = np.clip(u_cmd, -HINGE_LIMIT, HINGE_LIMIT)
            mujoco.mj_step(model, data)
            v.sync()
        while v.is_running():
            v.sync()

def render_episode_with_genome_realtime(genes, duration_sec=None):
    """Real-time viewer using data.time pacing; runs until closed (or duration)."""
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore
    mujoco.mj_forward(model, data)

    genes = np.asarray(genes, dtype=float)
    A, F, PHASE, OFFSET = genes[0::4], genes[1::4], genes[2::4], genes[3::4]
    assert len(A) == model.nu, "Genome size does not match actuator count."

    with viewer.launch_passive(model, data) as v:
        t0_wall = time.perf_counter()
        while v.is_running():
            if duration_sec is not None and (time.perf_counter() - t0_wall) >= duration_sec:
                break
            target = time.perf_counter() - t0_wall
            while data.time < target and v.is_running():
                t_sim = data.time
                u_cmd = OFFSET + A * np.sin(2.0 * math.pi * F * t_sim + PHASE)
                data.ctrl[:] = np.clip(u_cmd, -HINGE_LIMIT, HINGE_LIMIT)
                mujoco.mj_step(model, data)
            v.sync()

# ----------------------------
# EA evaluation
# ----------------------------
def build_model():
    """Build a minimal world+model to query actuator count, etc."""
    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    return world, model

PROBE_WORLD, PROBE_MODEL = build_model()
NU = PROBE_MODEL.nu  # number of actuators

# Oscillator genome structure: [A, f, phase, offset] per actuator
GENES_PER_JOINT = 4
GENOME_SIZE = NU * GENES_PER_JOINT

# Controller bounds (tightened frequency to discourage twitching)
LOW_A, HI_A = 0.0, HINGE_LIMIT
LOW_F, HI_F = 0.5, 2.0          # Hz (was 0.0–3.0)
LOW_P, HI_P = -math.pi, math.pi
LOW_O, HI_O = -HINGE_LIMIT, HINGE_LIMIT

LOW_BOUNDS = np.array([LOW_A, LOW_F, LOW_P, LOW_O] * NU, dtype=float)
HI_BOUNDS  = np.array([HI_A, HI_F, HI_P, HI_O] * NU, dtype=float)

def run_episode_with_genome(genes, steps=STEPS):
    """One episode with oscillator parameters; fitness = unified displacement metric (with rate limiting)."""
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    # Bind core body and initial facing (t=0)
    mujoco.mj_forward(model, data)
    core_body, forward_xy0 = _bind_core_body_and_forward_xy(model, data)

    genes = np.asarray(genes, dtype=float)
    A      = genes[0::4]
    F      = genes[1::4]
    PHASE  = genes[2::4]
    OFFSET = genes[3::4]
    assert len(A) == model.nu, "Genome size does not match actuator count."

    history = [core_body.xpos[:2].copy()]  # log initial pose BEFORE stepping

    # Rate-limited control application
    # Start from current ctrl (usually zeros)
    u_apply = data.ctrl.copy()

    for t in range(steps):
        t_sec = t * DT
        # Desired command from oscillator
        u_cmd = OFFSET + A * np.sin(2.0 * math.pi * F * t_sec + PHASE)
        u_cmd = np.clip(u_cmd, -HINGE_LIMIT, HINGE_LIMIT)

        # Per-step rate limiting
        du = u_cmd - u_apply
        du = np.clip(du, -RATE_LIMIT_DU, RATE_LIMIT_DU)
        u_apply = np.clip(u_apply + du, -HINGE_LIMIT, HINGE_LIMIT)

        data.ctrl[:] = u_apply
        mujoco.mj_step(model, data)
        history.append(core_body.xpos[:2].copy())

    fit = compute_displacement(history, forward_xy0, mode=FITNESS_MODE)
    return fit

# ----------------------------
# DEAP scaffolding
# ----------------------------
# Guarded creation to avoid re-definition errors in interactive runs
try:
    creator.FitnessMax
except AttributeError:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
try:
    creator.Individual
except AttributeError:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def init_ind():
    return creator.Individual(np.random.uniform(LOW_BOUNDS, HI_BOUNDS).tolist())

toolbox.register("individual", init_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    dx = run_episode_with_genome(ind, steps=STEPS)
    return (dx,)

toolbox.register("evaluate", evaluate)
ETA = 20.0
low_list = LOW_BOUNDS.tolist()
up_list  = HI_BOUNDS.tolist()

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register(
    "mate",
    tools.cxSimulatedBinaryBounded,
    low=low_list,
    up=up_list,
    eta=ETA,
)
toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=low_list,
    up=up_list,
    eta=ETA,
    indpb=1.0/GENOME_SIZE,
)

# ----------------------------
# Experiment runners
# ----------------------------
def run_random_experiment(num_runs=100, steps=500, out_png=RAND_LINE_PNG):
    rng = np.random.default_rng(42)
    fitnesses = []
    for i in range(num_runs):
        fit = run_random_episode(rng, steps=steps)
        fitnesses.append(fit)
        print(f"[Random] Run {i+1:3d}: fitness = {fit:.4f}")

    fitnesses = np.asarray(fitnesses, dtype=float)

    # Plot as line chart (run index vs fitness)
    plt.figure(figsize=(8, 4))
    plt.plot(fitnesses, marker="o", linestyle="-", label="Random run fitness")
    plt.axhline(np.mean(fitnesses), linestyle="--", linewidth=1, label=f"Mean = {np.mean(fitnesses):.3f}")
    plt.axhline(np.median(fitnesses), linestyle=":", linewidth=1, label=f"Median = {np.median(fitnesses):.3f}")
    label_desc = {
        "projected": "forward displacement",
        "x": "ΔX displacement",
        "euclidean": "Euclidean displacement",
    }[FITNESS_MODE]
    plt.xlabel("Run index")
    plt.ylabel(f"Fitness ({label_desc}, m)")
    plt.title(f"Random Controller Fitness over {num_runs} Runs (steps={steps})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[Random] Saved line chart to {out_png}")
    return fitnesses

def run_ea_experiment(pop_size=100, n_gen=100, cxpb=0.9, mutpb=0.2, steps=STEPS, out_png=EA_FITNESS_PNG):
    # init population
    pop = toolbox.population(n=pop_size)

    # evaluate initial population
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    best_per_gen = [tools.selBest(pop, 1)[0].fitness.values[0]]

    for gen in range(1, n_gen + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                if hasattr(c1.fitness, "values"):
                    del c1.fitness.values
                if hasattr(c2.fitness, "values"):
                    del c2.fitness.values

        # mutation
        for m in offspring:
            if random.random() < mutpb:
                toolbox.mutate(m)
                if hasattr(m.fitness, "values"):
                    del m.fitness.values

        # evaluate new/changed individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # replace population
        pop[:] = offspring

        # record best of this generation
        best = tools.selBest(pop, 1)[0].fitness.values[0]
        best_per_gen.append(best)
        print(f"[EA] Gen {gen:3d} | best fitness = {best:.4f}")
        if gen % 5 == 0:
            Fs = np.array([ind[1::4] for ind in pop], dtype=float).ravel()
            print(f"[EA]   F stats — mean {Fs.mean():.2f} Hz | "f"median {np.median(Fs):.2f} | min {Fs.min():.2f} | max {Fs.max():.2f}")

    # Plot best fitness by generation
    plt.figure(figsize=(8, 4))
    plt.plot(best_per_gen, marker="o")
    label_desc = {
        "projected": "forward displacement",
        "x": "ΔX displacement",
        "euclidean": "Euclidean displacement",
    }[FITNESS_MODE]
    plt.xlabel("Generation")
    plt.ylabel(f"Best Fitness ({label_desc}, m)")
    plt.title(f"EA: Best Fitness over {n_gen} Generations (pop={pop_size}, steps={steps})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[EA] Saved best-per-generation plot to {out_png}")

    best_ind = tools.selBest(pop, 1)[0]
    return best_ind, best_per_gen

# ----------------------------
# Main
# ----------------------------
def main():
    # 1) Random experiment
    NUM_RANDOM_RUNS = 100
    run_random_experiment(num_runs=NUM_RANDOM_RUNS, steps=STEPS, out_png=RAND_LINE_PNG)

    # 2) EA experiment
    POP_SIZE = 100
    NGEN = 100
    CXPB = 0.9
    MUTPB = 0.2
    best_ind, best_per_gen = run_ea_experiment(
        pop_size=POP_SIZE,
        n_gen=NGEN,
        cxpb=CXPB,
        mutpb=MUTPB,
        steps=STEPS,
        out_png=EA_FITNESS_PNG
    )

    # 3) Render final best individual
    print("[Render] Rendering final best individual (real-time)...")
    render_episode_with_genome_realtime(best_ind, duration_sec=None)
    # If you prefer exact replay instead:
    # print("[Render] Rendering final best individual (exact replay)...")
    # render_episode_with_genome_exact(best_ind, steps=STEPS)

if __name__ == "__main__":
    main()
