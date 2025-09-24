"""
Assignment 2 — Gecko robot (improved EA & exploration)

Key changes:
- **Do NOT change the fitness** (compute_displacement stays identical).
- Stronger exploration towards **larger joint amplitudes** while keeping frequencies in a non-twitchy band.
- Bias the **initialization** toward higher amplitudes and small offsets.
- **Heavy‑tailed amplitude kicks** during mutation to escape low‑amplitude traps.
- Add **elitism**, **random immigrants**, and **adaptive mutation** on stagnation.
- Gentle **amplitude warm‑up** in simulation so large A doesn’t cause instability, and a slightly higher rate limit to avoid over‑damping.
- Extra logging of amplitude stats.

The overall structure/order of the original script is preserved (random → EA → plots → render).
"""

import math
import numpy as np
import random
import time
import mujoco
import matplotlib.pyplot as plt
from collections import defaultdict
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

# Output files
RAND_LINE_PNG = "random_fitness_line.png"
EA_FITNESS_PNG = "ea_best_over_generations.png"

# Rate limiter (per-step max change, radians)
# (Slightly higher so high‑A solutions can actually manifest in sim)
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

    history = [core_body.xpos[:2].copy()]  # log initial position BEFORE stepping
    for _ in range(steps):
        u = random_move(data, rng)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        history.append(core_body.xpos[:2].copy())

    # Unified metric
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
    """
    Replay exactly what EA evaluated (same t*DT timing), and keep the window open
    until the user closes it.
    """

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
    """Real-time viewer using data.time pacing; runs until closed (or duration)."""

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

    history = [core_body.xpos[:2].copy()]  # log initial pose BEFORE stepping

    # Rate-limited control application
    u_apply = data.ctrl.copy()

    for t in range(steps):
        t_sec = t * DT
        # Amplitude warm-up to let large-A solutions settle
        ramp = min(1.0, t / max(1, WARMUP_STEPS))
        # Desired command from oscillator
        u_cmd = OFFSET + (A * ramp) * np.sin(2.0 * math.pi * F * t_sec + PHASE)
        u_cmd = np.clip(u_cmd, -HINGE_LIMIT, HINGE_LIMIT)

        # Per-step rate limiting
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
        # High‑amplitude bias via Beta(5,2) ∈ [0,1]
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

# Tournament selection stays the same
toolbox.register("select", tools.selTournament, tournsize=3)

# Baseline SBX crossover (bounded)
toolbox.register(
    "mate",
    tools.cxSimulatedBinaryBounded,
    low=low_list,
    up=up_list,
    eta=ETA,
)

# We'll keep DEAP's polynomial mutation as a base and add amplitude "kicks" below
_base_mut = tools.mutPolynomialBounded


def _amplitude_kick(ind, frac_joints=0.3, scale=0.25):
    """Heavy‑tailed kicks on A genes for a random subset of joints.
    scale is relative to HI_A.
    """
    n_pick = max(1, int(frac_joints * NU))
    idx_A = [4*j for j in range(NU)]
    pick = random.sample(idx_A, n_pick)
    for i in pick:
        # Cauchy step (heavy‑tailed) encourages occasional big jumps
        cauchy = math.tan(math.pi * (random.random() - 0.5))
        step = cauchy * (scale * HI_A)
        newA = float(np.clip(ind[i] + step, LOW_A, HI_A))
        # If amplitude stayed too small, bias upward a bit
        if newA < 0.4 * HI_A:
            newA = min(HI_A, newA + 0.3 * HI_A)
        ind[i] = newA


# Wrapper mutate used by our EA loop
# (keeps same signature usage as before)
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
# Experiment runners
# ----------------------------

def run_random_experiment(num_runs=100, steps=500, out_png=RAND_LINE_PNG, fitness_variant="EA1", k_side=0.5):
    rng = np.random.default_rng(42)
    fitnesses = []
    for i in range(num_runs):
        fit = run_random_episode(rng, steps=steps, fitness_variant=fitness_variant, k_side=k_side)
        fitnesses.append(fit)
        print(f"[Random] Run {i+1:3d}: fitness = {fit:.4f}")

    fitnesses = np.asarray(fitnesses, dtype=float)

    # Plot as line chart (run index vs fitness)
    plt.figure(figsize=(8, 4))
    plt.plot(fitnesses, marker="o", linestyle="-", label="Random run fitness")
    plt.axhline(np.mean(fitnesses), linestyle="--", linewidth=1, label=f"Mean = {np.mean(fitnesses):.3f}")
    plt.axhline(np.median(fitnesses), linestyle=":", linewidth=1, label=f"Median = {np.median(fitnesses):.3f}")
    plt.xlabel("Run index")
    plt.ylabel(f"Fitness ({fitness_variant}, m)")
    plt.title(f"Random Controller Fitness over {num_runs} Runs (steps={steps})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[Random] Saved line chart to {out_png}")
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


# Improved EA: elitism + random immigrants + adaptive mutation on stagnation
def run_ea_experiment(pop_size=100, n_gen=100, cxpb=0.9, mutpb=0.2, steps=STEPS, out_png=EA_FITNESS_PNG, fitness_variant="EA1", k_side=0.5):
    toolbox.register("evaluate", make_evaluator(
        steps=steps,
        fitness_variant=fitness_variant,
        k_side=k_side
    ))
    # initial population
    pop = toolbox.population(n=pop_size)

    # evaluate initial population
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    # Elitism via Hall of Fame
    ELITE_K = max(1, pop_size // 20)  # top 5%
    hof = tools.HallOfFame(ELITE_K)
    hof.update(pop)

    best_per_gen = [tools.selBest(pop, 1)[0].fitness.values[0]]

    # Stagnation tracking
    no_improve = 0
    best_so_far = best_per_gen[-1]

    # Random immigrants fraction
    IMM_FRAC = 0.10

    for gen in range(1, n_gen + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # crossover & mutation
        offspring = crossover(offspring, cxpb)

        # Adapt mutation if stagnating
        adapt_mutpb = mutpb
        amp_kick_prob = 0.5
        if no_improve >= 5:
            adapt_mutpb = min(1.0, mutpb * 1.8)
            amp_kick_prob = 0.6  # push harder on amplitude exploration
        if no_improve >= 10:
            adapt_mutpb = min(1.0, mutpb * 2.5)
            amp_kick_prob = 0.8

        offspring = mutate(offspring, adapt_mutpb, amp_kick_prob=amp_kick_prob)

        # Random immigrants replace a fraction before evaluation
        n_imm = max(0, int(IMM_FRAC * len(offspring)))
        for _ in range(n_imm):
            idx = random.randrange(len(offspring))
            offspring[idx] = toolbox.individual()

        # evaluate new/changed individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Elitism: keep top ELITE_K from previous pop (tracked by HOF)
        hof.update(offspring)
        elites = list(map(toolbox.clone, hof.items))

        # replace population (elitism injected)
        # Keep the best ELITE_K overall, fill the rest with offspring
        offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        pop = elites + offspring[: max(0, pop_size - len(elites))]

        # record best of this generation
        best = tools.selBest(pop, 1)[0].fitness.values[0]
        best_per_gen.append(best)

        # stagnation update
        if best > best_so_far + 1e-9:
            best_so_far = best
            no_improve = 0
        else:
            no_improve += 1

        # Logging
        Fs = np.array([ind[1::4] for ind in pop], dtype=float).ravel()
        As = np.array([ind[0::4] for ind in pop], dtype=float).ravel()
        bigA = (As > 0.7 * HI_A).mean() * 100.0
        if gen % 1 == 0:
            print(
                f"[EA] Gen {gen:3d} | best = {best:.4f} | no_improve={no_improve:2d} | "
                f"F mean {Fs.mean():.2f} Hz (min {Fs.min():.2f}, max {Fs.max():.2f}) | "
                f"A mean {As.mean():.2f} rad | >0.6*HI {bigA:4.1f}%"
            )

    # Plot best fitness by generation
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

    best_ind = tools.selBest(pop, 1)[0]
    return best_ind, best_per_gen



def get_z_scores(results, eps=1e-12):
    out = defaultdict(dict)

    for mode, d in results.items():

        rand = np.asarray(d.get("Random_experiment_fitnesses", []), dtype=float)
        ea   = np.asarray(d.get("EA_experiment_fitness", []), dtype=float)
        rand = rand[~np.isnan(rand)]
        ea   = ea[~np.isnan(ea)]

        mu = float(rand.mean()) if rand.size else float("nan")
        sd = float(rand.std())  if rand.size else float("nan")

        # minimal error management
        if sd == 0:
            sd = eps
        # Z scores
        if rand.size == 0 or np.isnan(sd):
            z = []
        else:
            z = ((ea - mu) / sd).astype(float).tolist()

        out[mode]["random_mean"] = mu
        out[mode]["random_std"] = sd
        out[mode]["ea_z_scores"] = z

    return out

# ----------------------------
# Main
# ----------------------------

def main():
    random.seed(42)
    np.random.seed(42)
    results = defaultdict(dict)
    # 1) Random experiment
    NUM_RANDOM_RUNS = 20
    Fits_EA1_rand = run_random_experiment(num_runs=NUM_RANDOM_RUNS, steps=STEPS, out_png=RAND_LINE_PNG, fitness_variant="EA1")
    Fits_EA2B_rand = run_random_experiment(num_runs=NUM_RANDOM_RUNS, steps=STEPS, out_png=RAND_LINE_PNG, fitness_variant="EA2B", k_side=0.5)

    # 2) EA experiments: EA1 (distance) vs EA2B (distance - k*sideways)
    POP_SIZE = 120
    NGEN = 20
    CXPB = 0.9
    MUTPB = 0.25
    STEPS_EA = 4000  # quicker for testing; raise to STEPS later

    # EA1: projected forward distance
    best1, Fits_EA1 = run_ea_experiment(
        pop_size=POP_SIZE, n_gen=NGEN, cxpb=CXPB, mutpb=MUTPB,
        steps=STEPS_EA, out_png="ea1_best_over_generations.png",
        fitness_variant="EA1"
    )

    # EA2B: forward - 0.5 * sideways
    best2, Fits_EA2B = run_ea_experiment(
        pop_size=POP_SIZE, n_gen=NGEN, cxpb=CXPB, mutpb=MUTPB,
        steps=STEPS_EA, out_png="ea2b_best_over_generations.png",
        fitness_variant="EA2B", k_side=0.5
    )
    results["EA1"] = {"EA_experiment_fitness":Fits_EA1, "Random_experiment_fitnesses":Fits_EA1_rand}
    results["EA2B"] = {"EA_experiment_fitness":Fits_EA2B, "Random_experiment_fitnesses":Fits_EA2B_rand}
    Z_results = get_z_scores(results)

    plt.figure(figsize=(8, 4))
    for model, stats in Z_results.items():
        z = stats["ea_z_scores"]
        x = np.arange(len(z))
        legend_label = f"{model} z_scores"
        plt.plot(x, z, marker="o", linestyle="-", label=legend_label)

    plt.axhline(0.0, linestyle="--", linewidth=1, label="Z = 0")
    plt.xlabel("N Gen")
    plt.ylabel("Z score")
    plt.title("Z-scores per Model Across generations")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Z_scores_comparison_n1_20_gen", dpi=150)
    plt.close()

    # 4) Render the best of EA2B (straighter gait, better for demo)
    print("[Render] Rendering EA2B best individual (real-time)...")
    render_episode_with_genome_realtime(best2, duration_sec=None)


if __name__ == "__main__":
    main()
