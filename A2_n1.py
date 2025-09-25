"""
Assignment 2 — Gecko robot (improved EA & exploration)

Key changes:
- **Do NOT change the fitness** (compute_displacement stays identical).
- Stronger exploration towards **larger joint amplitudes** while keeping frequencies in a non-twitchy band.
- Bias the **initialization** toward higher amplitudes and small offsets.
- **Heavy-tailed amplitude kicks** during mutation to escape low-amplitude traps.
- Add **elitism**, **random immigrants**, and **adaptive mutation** on stagnation.
- Gentle **amplitude warm-up** in simulation so large A doesn’t cause instability, and a slightly higher rate limit to avoid over-damping.
- Extra logging of amplitude stats.
- EA1 projected-distance comparison metric tracked for all EA runs.
- Runs each algorithm 3 times and saves **all** run data to CSV.
- FINAL: Everything runs for **30 generations/episodes** (EA gens == Random episodes).
"""

import math
import numpy as np
import random
import time
import csv
import os
import mujoco
import matplotlib.pyplot as plt
from collections import defaultdict
from deap import base, creator, tools
import json

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

# Output files (base names; per-run suffixes will be added)
RAND_LINE_PNG = "random_fitness_line.png"
EA_FITNESS_PNG = "ea_best_over_generations.png"

# Rate limiter (per-step max change, radians)
# (Slightly higher so high-A solutions can actually manifest in sim)
RATE_LIMIT_DU = 0.025  # was 0.015

# Smoothly ramp amplitude during the first WARMUP_STEPS to avoid shocks
WARMUP_STEPS = int(0.5 / DT)  # ~0.5 s

# ----------------------------
# Unified fitness/comparison metric
# ----------------------------
def compute_displacement(history_xy, forward_xy0, mode="EA1", k=0.5):
    """
    history_xy: list/array of XY positions (start..end)
    forward_xy0: unit XY vector of initial facing (ignored if mode != 'projected')
    mode: EA1 ='projected'; EA2B = forward_minus_sideways
    returns scalar fitness in meters
    """
    disp = np.asarray(history_xy[-1]) - np.asarray(history_xy[0])
    if mode == "EA1":
        # Original projected mode
        return float(np.dot(disp, forward_xy0))
    elif mode == "EA2B":
        # compute_forward_minus_sideways
        start = np.asarray(history_xy[0], dtype=float)
        end   = np.asarray(history_xy[-1], dtype=float)
        disp  = end - start
        forward  = float(np.dot(disp, forward_xy0))  # forward component
        eucl     = float(np.linalg.norm(disp))       # total displacement
        sideways = max(0.0, eucl - abs(forward))     # non-forward component
        return forward - k * sideways
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


def run_random_episode(rng, steps, fitness_variant="EA1", k_side=0.5):
    """One episode with random control, fitness = unified displacement metric."""
    model, data = build_model()
    core_body, forward_xy0 = _bind_core_body_and_forward_xy(model, data)
    history = [core_body.xpos[:2].copy()]
    for _ in range(steps):
        u = random_move(data, rng)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        history.append(core_body.xpos[:2].copy())
    fit = compute_displacement(history, forward_xy0, mode=fitness_variant, k=k_side)
    return fit


def split_genes(genes, model):
    genes = np.asarray(genes, dtype=float)
    A, F, PHASE, OFFSET = genes[0::4], genes[1::4], genes[2::4], genes[3::4]
    assert len(A) == model.nu, "Genome size does not match actuator count."
    return A, F, PHASE, OFFSET

# ----------------------------
# Rendering helpers
# ----------------------------

def render_episode_with_genome_exact(genes, steps=STEPS):
    model, data = build_model()
    A, F, PHASE, OFFSET = split_genes(genes, model)
    with viewer.launch_passive(model, data) as v:
        for t in range(steps):
            if not v.is_running():
                return
            t_sec = t * DT
            ramp = min(1.0, t / max(1, WARMUP_STEPS))
            u_cmd = OFFSET + (A * ramp) * np.sin(2.0 * math.pi * F * t_sec + PHASE)
            data.ctrl[:] = np.clip(u_cmd, -HINGE_LIMIT, HINGE_LIMIT)
            mujoco.mj_step(model, data)
            v.sync()
        while v.is_running():
            v.sync()


def render_episode_with_genome_realtime(genes, duration_sec=None):
    model, data = build_model()
    A, F, PHASE, OFFSET = split_genes(genes, model)
    with viewer.launch_passive(model, data) as v:
        t0_wall = time.perf_counter()
        while v.is_running():
            if duration_sec is not None and (time.perf_counter() - t0_wall) >= duration_sec:
                break
            target = time.perf_counter() - t0_wall
            while data.time < target and v.is_running():
                t_sim = data.time
                ramp = min(1.0, (t_sim / DT) / max(1, WARMUP_STEPS))
                u_cmd = OFFSET + (A * ramp) * np.sin(2.0 * math.pi * F * t_sim + PHASE)
                data.ctrl[:] = np.clip(u_cmd, -HINGE_LIMIT, HINGE_LIMIT)
                mujoco.mj_step(model, data)
            v.sync()

# ----------------------------
# EA evaluation
# ----------------------------

def build_model(spawn_pos=(0, 0, 0.1)):
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=list(spawn_pos))
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


PROBE_MODEL , _ = build_model()
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


def run_episode_with_genome(genes, steps=STEPS, fitness_variant="EA1", k_side=0.5):
    """One episode with oscillator parameters; fitness = unified displacement metric (with rate limiting + warm-up)."""
    model, data = build_model()
    core_body, forward_xy0 = _bind_core_body_and_forward_xy(model, data)
    A, F, PHASE, OFFSET = split_genes(genes, model)

    history = [core_body.xpos[:2].copy()]
    u_apply = data.ctrl.copy()

    for t in range(steps):
        t_sec = t * DT
        ramp = min(1.0, t / max(1, WARMUP_STEPS))
        u_cmd = OFFSET + (A * ramp) * np.sin(2.0 * math.pi * F * t_sec + PHASE)
        u_cmd = np.clip(u_cmd, -HINGE_LIMIT, HINGE_LIMIT)

        du = u_cmd - u_apply
        du = np.clip(du, -RATE_LIMIT_DU, RATE_LIMIT_DU)
        u_apply = np.clip(u_apply + du, -HINGE_LIMIT, HINGE_LIMIT)

        data.ctrl[:] = u_apply
        mujoco.mj_step(model, data)
        history.append(core_body.xpos[:2].copy())

    fit = compute_displacement(history, forward_xy0, mode=fitness_variant, k=k_side)
    return fit

# ----------------------------
# DEAP scaffolding
# ----------------------------
try:
    creator.FitnessMax
except AttributeError:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
try:
    creator.Individual
except AttributeError:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def make_evaluator(steps=STEPS, fitness_variant="EA1", k_side=0.5):
    def _eval(ind):
        val = run_episode_with_genome(
            ind, steps=steps, fitness_variant=fitness_variant, k_side=k_side
        )
        return (val,)
    return _eval


# --- Initialization biased toward larger amplitudes & modest offsets ---
def init_ind():
    genes = np.empty(GENOME_SIZE, dtype=float)
    for j in range(NU):
        # High-amplitude bias via Beta(5,2) ∈ [0,1]
        a = float(np.clip(np.random.beta(5.0, 2.0) * HI_A, LOW_A, HI_A))
        # Keep frequencies in comfortable range (still within bounds)
        f = float(np.clip(np.random.uniform(0.6, 1.8), LOW_F, HI_F))
        p = float(np.random.uniform(LOW_P, HI_P))
        # Small offsets around 0 (helps avoid saturating hinges)
        o = float(np.clip(np.random.normal(loc=0.0, scale=0.2 * HINGE_LIMIT), LOW_O, HI_O))
        genes[4*j:4*j+4] = [a, f, p, o]
    return creator.Individual(genes.tolist())


toolbox.register("individual", init_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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
_base_mut = tools.mutPolynomialBounded


def _amplitude_kick(ind, frac_joints=0.3, scale=0.25):
    """Heavy-tailed kicks on A genes for a random subset of joints."""
    n_pick = max(1, int(frac_joints * NU))
    idx_A = [4*j for j in range(NU)]
    pick = random.sample(idx_A, n_pick)
    for i in pick:
        # Cauchy step (heavy-tailed) encourages occasional big jumps
        cauchy = math.tan(math.pi * (random.random() - 0.5))
        step = cauchy * (scale * HI_A)
        newA = float(np.clip(ind[i] + step, LOW_A, HI_A))
        if newA < 0.4 * HI_A:
            newA = min(HI_A, newA + 0.3 * HI_A)
        ind[i] = newA


def mutate(offspring, mutpb, amp_kick_prob=0.5):
    for m in offspring:
        changed = False
        if random.random() < mutpb:
            _base_mut(m, low=low_list, up=up_list, eta=ETA, indpb=1.0/GENOME_SIZE)
            changed = True
        # Occasionally layer on amplitude kicks
        if random.random() < amp_kick_prob:
            _amplitude_kick(m, frac_joints=0.3, scale=0.25)
            changed = True
        if changed and hasattr(m.fitness, "values"):
            del m.fitness.values
    return offspring


# ----------------------------
# CSV helpers
# ----------------------------
def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def write_csv(path, rows, header):
    ensure_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ----------------------------
# Experiment runners
# ----------------------------

def run_random_experiment(num_runs=100, steps=500, out_png=RAND_LINE_PNG, fitness_variant="EA1", k_side=0.5, seed=None, csv_path=None, run_idx=None):
    rng = np.random.default_rng(seed)
    fitnesses = []
    for i in range(num_runs):
        fit = run_random_episode(rng, steps=steps, fitness_variant=fitness_variant, k_side=k_side)
        fitnesses.append(fit)
        print(f"[Random] Run {i+1:3d}: fitness = {fit:.4f}")

    fitnesses = np.asarray(fitnesses, dtype=float)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(fitnesses, marker="o", linestyle="-", label="Random run fitness")
    plt.axhline(np.mean(fitnesses), linestyle="--", linewidth=1, label=f"Mean = {np.mean(fitnesses):.3f}")
    plt.axhline(np.median(fitnesses), linestyle=":", linewidth=1, label=f"Median = {np.median(fitnesses):.3f}")
    plt.xlabel("Episode index")
    plt.ylabel(f"Fitness ({fitness_variant}, m)")
    plt.title(f"Random Controller Fitness over {num_runs} Episodes (steps={steps})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[Random] Saved line chart to {out_png}")

    # CSV (per episode)
    if csv_path is not None:
        rows = []
        for i, fval in enumerate(fitnesses.tolist()):
            rows.append({
                "algorithm": "Random",
                "outer_run": run_idx,
                "episode_index": i,
                "fitness_variant": fitness_variant,
                "fitness": fval,
                "steps": steps,
                "seed": seed,
            })
        header = ["algorithm","outer_run","episode_index","fitness_variant","fitness","steps","seed"]
        write_csv(csv_path, rows, header)

    return fitnesses


def crossover(offspring, cxpb):
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(c1, c2)
            if hasattr(c1.fitness, "values"):
                del c1.fitness.values
            if hasattr(c2.fitness, "values"):
                del c2.fitness.values
    return offspring


def run_ea_experiment(pop_size=100, n_gen=100, cxpb=0.9, mutpb=0.2, steps=STEPS, out_png=EA_FITNESS_PNG, fitness_variant="EA1", k_side=0.5):
    """Runs EA and returns best individual, per-gen native fitness, per-gen EA1 projected distance."""
    toolbox.register("evaluate", make_evaluator(
        steps=steps,
        fitness_variant=fitness_variant,
        k_side=k_side
    ))
    pop = toolbox.population(n=pop_size)

    # evaluate initial population
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    ELITE_K = max(1, pop_size // 20)
    hof = tools.HallOfFame(ELITE_K)
    hof.update(pop)

    best_per_gen = [tools.selBest(pop, 1)[0].fitness.values[0]]

    # EA1 comparison metric tracking
    best_proj_per_gen = []
    best_initial = tools.selBest(pop, 1)[0]
    best_initial_proj = run_episode_with_genome(
        best_initial, steps=steps, fitness_variant="EA1", k_side=k_side
    )
    best_proj_per_gen.append(best_initial_proj)

    no_improve = 0
    best_so_far = best_per_gen[-1]
    IMM_FRAC = 0.10

    for gen in range(1, n_gen + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        offspring = crossover(offspring, cxpb)

        adapt_mutpb = mutpb
        amp_kick_prob = 0.5
        if no_improve >= 5:
            adapt_mutpb = min(1.0, mutpb * 1.8)
            amp_kick_prob = 0.6
        if no_improve >= 10:
            adapt_mutpb = min(1.0, mutpb * 2.5)
            amp_kick_prob = 0.8

        offspring = mutate(offspring, adapt_mutpb, amp_kick_prob=amp_kick_prob)

        n_imm = max(0, int(IMM_FRAC * len(offspring)))
        for _ in range(n_imm):
            idx = random.randrange(len(offspring))
            offspring[idx] = toolbox.individual()

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        hof.update(offspring)
        elites = list(map(toolbox.clone, hof.items))

        offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        pop = elites + offspring[: max(0, pop_size - len(elites))]

        best = tools.selBest(pop, 1)[0]
        best_per_gen.append(best.fitness.values[0])

        best_proj = run_episode_with_genome(
            best, steps=steps, fitness_variant="EA1", k_side=k_side
        )
        best_proj_per_gen.append(best_proj)

        if best.fitness.values[0] > best_so_far + 1e-9:
            best_so_far = best.fitness.values[0]
            no_improve = 0
        else:
            no_improve += 1

        Fs = np.array([ind[1::4] for ind in pop], dtype=float).ravel()
        As = np.array([ind[0::4] for ind in pop], dtype=float).ravel()
        bigA = (As > 0.7 * HI_A).mean() * 100.0
        print(
            f"[EA] Gen {gen:3d} | best = {best.fitness.values[0]:.4f} | no_improve={no_improve:2d} | "
            f"F mean {Fs.mean():.2f} Hz (min {Fs.min():.2f}, max {Fs.max():.2f}) | "
            f"A mean {As.mean():.2f} rad | >0.6*HI {bigA:4.1f}% | "
            f"EA1-proj(best) = {best_proj:.4f}"
        )

    # Native fitness plot
    plt.figure(figsize=(8, 4))
    plt.plot(best_per_gen, marker="o")
    plt.xlabel("Generation")
    plt.ylabel(f"Best Fitness ({fitness_variant}, m)")
    plt.title(f"EA: Best Fitness over {n_gen} Generations (pop={pop_size}, steps={steps})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[EA] Saved best-per-generation plot to {out_png}")

    # EA1 comparison plot
    proj_png = out_png.replace(".png", "_projected_distance.png")
    plt.figure(figsize=(8, 4))
    plt.plot(best_proj_per_gen, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Projected Distance (EA1, m)")
    plt.title(f"EA Comparison Metric: EA1 Projected Distance per Generation\n(pop={pop_size}, steps={steps})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(proj_png, dpi=150)
    plt.close()
    print(f"[EA] Saved EA1 projected-distance comparison plot to {proj_png}")

    best_ind = tools.selBest(pop, 1)[0]
    return best_ind, best_per_gen, best_proj_per_gen


def get_z_scores(results, eps=1e-12):
    out = defaultdict(dict)
    for mode, d in results.items():
        rand = np.asarray(d.get("Random_experiment_fitnesses", []), dtype=float)
        ea   = np.asarray(d.get("EA_experiment_fitness", []), dtype=float)
        rand = rand[~np.isnan(rand)]
        ea   = ea[~np.isnan(ea)]
        mu = float(rand.mean()) if rand.size else float("nan")
        sd = float(rand.std())  if rand.size else float("nan")
        if sd == 0:
            sd = eps
        if rand.size == 0 or np.isnan(sd):
            z = []
        else:
            z = ((ea - mu) / sd).astype(float).tolist()
        out[mode]["random_mean"] = mu
        out[mode]["random_std"] = sd
        out[mode]["ea_z_scores"] = z
    return out

def save_controller(genes):
    data = {
        "controller_type": "sinewave",
        "robot": "gecko",
        "environment": "SimpleFlatWorld",
        "per_joint_format": ["A","F","PHASE","OFFSET"],
        "nu": NU,
        "genes": [float(x) for x in genes],
    }
    with open("controller_data", "w") as f:
        json.dump(data, f, indent=2)

# ----------------------------
# Main: run each algorithm 3 times, NGEN=30 (random episodes = 30), and save CSVs
# ----------------------------

def main():
    # Global experiment parameters
    POP_SIZE = 120
    NGEN = 30            # <-- FINAL: 30 generations
    CXPB = 0.9
    MUTPB = 0.25
    STEPS_EA = 4000

    # Random baseline episodes set equal to generations
    NUM_RANDOM_RUNS_PER_EXP = NGEN

    # Output directories (optional: keep root clean)
    out_dir = "results"
    ensure_dir(os.path.join(out_dir, "dummy"))
    # Master CSV to aggregate all EA runs
    ea_master_rows = []
    ea_master_header = [
        "algorithm","outer_run","generation","native_fitness","projected_fitness",
        "pop_size","n_gen","steps","cxpb","mutpb","seed"
    ]

    total_runs = 1
    base_seed = 42
    best_ea2b_last = None

    for run_idx in range(1, total_runs + 1):
        seed = base_seed + 100 * run_idx
        print(f"\n================ RUN {run_idx} / {total_runs} (seed {seed}) ================")
        random.seed(seed)
        np.random.seed(seed)

        # ---- Random controller baseline (EA1 metric) ----
        rand_png = os.path.join(out_dir, f"{RAND_LINE_PNG.replace('.png','')}_run{run_idx}.png")
        rand_csv = os.path.join(out_dir, f"random_baseline_run{run_idx}.csv")
        _ = run_random_experiment(
            num_runs=NUM_RANDOM_RUNS_PER_EXP,
            steps=STEPS,
            out_png=rand_png,
            fitness_variant="EA1",
            k_side=0.5,
            seed=seed,
            csv_path=rand_csv,
            run_idx=run_idx
        )

        # ---- EA1 ----
        ea1_png = os.path.join(out_dir, f"ea1_best_over_generations_run{run_idx}.png")
        best1, ea1_native, ea1_proj = run_ea_experiment(
            pop_size=POP_SIZE, n_gen=NGEN, cxpb=CXPB, mutpb=MUTPB,
            steps=STEPS_EA, out_png=ea1_png,
            fitness_variant="EA1"
        )
        # Save per-run EA1 CSV
        rows = []
        for gen, (nf, pf) in enumerate(zip(ea1_native, ea1_proj)):  # includes generation 0
            rows.append({
                "algorithm": "EA1",
                "outer_run": run_idx,
                "generation": gen,
                "native_fitness": nf,
                "projected_fitness": pf,
                "pop_size": POP_SIZE,
                "n_gen": NGEN,
                "steps": STEPS_EA,
                "cxpb": CXPB,
                "mutpb": MUTPB,
                "seed": seed
            })
        ea1_csv = os.path.join(out_dir, f"ea1_run{run_idx}.csv")
        write_csv(ea1_csv, rows, ea_master_header)
        ea_master_rows.extend(rows)

        # ---- EA2B ----
        ea2b_png = os.path.join(out_dir, f"ea2b_best_over_generations_run{run_idx}.png")
        best2, ea2b_native, ea2b_proj = run_ea_experiment(
            pop_size=POP_SIZE, n_gen=NGEN, cxpb=CXPB, mutpb=MUTPB,
            steps=STEPS_EA, out_png=ea2b_png,
            fitness_variant="EA2B", k_side=0.5
        )
        # Save per-run EA2B CSV
        rows = []
        for gen, (nf, pf) in enumerate(zip(ea2b_native, ea2b_proj)):
            rows.append({
                "algorithm": "EA2B",
                "outer_run": run_idx,
                "generation": gen,
                "native_fitness": nf,
                "projected_fitness": pf,
                "pop_size": POP_SIZE,
                "n_gen": NGEN,
                "steps": STEPS_EA,
                "cxpb": CXPB,
                "mutpb": MUTPB,
                "seed": seed
            })
        ea2b_csv = os.path.join(out_dir, f"ea2b_run{run_idx}.csv")
        write_csv(ea2b_csv, rows, ea_master_header)
        ea_master_rows.extend(rows)

    save_controller(best1)
    # Write combined EA master CSV (all runs, both algorithms)
    ea_master_csv = os.path.join(out_dir, "ea_all_runs_master.csv")
    write_csv(ea_master_csv, ea_master_rows, ea_master_header)
    print(f"[CSV] Wrote EA master CSV: {ea_master_csv}")

    # Optional: render the best of the last EA2B run (demo)
    #if best_ea2b_last is not None:
     #   print("[Render] Rendering EA2B best individual from last run (real-time)...")
    #    render_episode_with_genome_realtime(best_ea2b_last, duration_sec=None)


if __name__ == "__main__":
    main()
