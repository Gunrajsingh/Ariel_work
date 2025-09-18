"""
Assignment 2 — Gecko robot, repeated random controller runs
Runs random controller 100 times, collects fitness, plots fitness distribution.
"""

import math
import numpy as np
import random
import mujoco
import matplotlib.pyplot as plt
from deap import base, creator, tools

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# NEW: same viewer API as in your A2_template.py
from mujoco import viewer

HINGE_LIMIT = math.pi / 2
DT = 1 / 240
STEPS = 500

#####start of random definition

def random_move(data, rng):
    delta = 0.05
    u = data.ctrl.copy()
    u = u + rng.uniform(-HINGE_LIMIT, HINGE_LIMIT, size=data.ctrl.shape) * delta
    u = np.clip(u, -HINGE_LIMIT, HINGE_LIMIT)
    return u

def run_random_episode(rng, steps=500):
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0, 0, 0.1])  # small lift

    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    # Track core geom (for position)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    core = to_track[0]

    history = []
    for t in range(steps):
        u = random_move(data, rng)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        history.append(core.xpos[:2].copy())

    # Fitness = displacement along X-axis
    fit = float(history[-1][0] - history[0][0])
    return fit

######end of random definition
######start of DEAP

#helper function for repeat world creation code
def build_model():
    """Build a minimal world+model to query actuator count, etc."""
    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    return world, model

PROBE_WORLD, PROBE_MODEL = build_model()
NU = PROBE_MODEL.nu  # number of actuators

#creating gene "shape" - oscillator
GENES_PER_JOINT = 4   # [A, f, phase, offset]
GENOME_SIZE = NU * GENES_PER_JOINT

# Bounds per joint (repeated across all joints)
LOW_A, HI_A = 0.0, HINGE_LIMIT
LOW_F, HI_F = 0.0, 3.0          # Hz
LOW_P, HI_P = -math.pi, math.pi
LOW_O, HI_O = -HINGE_LIMIT, HINGE_LIMIT

LOW_BOUNDS = np.array([LOW_A, LOW_F, LOW_P, LOW_O] * NU, dtype=float)
HI_BOUNDS  = np.array([HI_A, HI_F, HI_P, HI_O] * NU, dtype=float)



def run_episode_with_genome(genes, steps=STEPS):
    """Run one simulation episode with oscillator parameters as the controller."""
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0, 0, 0.1])  # small lift
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    # Track core geom (for position)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    core = to_track[0]

    genes = np.asarray(genes, dtype=float)
    A      = genes[0::4]
    F      = genes[1::4]
    PHASE  = genes[2::4]
    OFFSET = genes[3::4]
    assert len(A) == model.nu, "Genome size does not match actuator count."

    history = []
    for t in range(steps):
        t_sec = t * DT
        u = OFFSET + A * np.sin(2.0 * math.pi * F * t_sec + PHASE)
        u = np.clip(u, -HINGE_LIMIT, HINGE_LIMIT)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        history.append(core.xpos[:2].copy())

    # Fitness = displacement along X-axis
    fit = float(history[-1][0] - history[0][0])
    return fit

# ======================
# DEAP: basic scaffolding
# ======================
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)  # <-- list

toolbox = base.Toolbox()

def init_ind():
    return creator.Individual(np.random.uniform(LOW_BOUNDS, HI_BOUNDS).tolist())

toolbox.register("individual", init_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation: calls the simulator with the genome
def evaluate(ind):
    dx = run_episode_with_genome(ind, steps=STEPS)
    # Optional: add small effort penalty to discourage extremes (disabled by default)
    # effort = float(np.mean(ind[0::4]) + np.mean(np.abs(ind[3::4])))  # A + |offset|
    # return (dx - 0.02 * effort,)
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
# Main simulation loop
# ----------------------------
def main():
    rng = np.random.default_rng(42)

    num_runs = 100
    fitnesses = []
    #random run x 1
 #   for i in range(num_runs):
  #      fit = run_random_episode(rng, steps=500)
   #     fitnesses.append(fit)
    #    print(f"Run {i+1:3d}: fitness = {fit:.4f}")
    
    
    POP_SIZE = 100
    NGEN = 20
    CXPB = 0.9
    MUTPB = 0.2

    # init population
    pop = toolbox.population(n=POP_SIZE)

    # evaluate initial population
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    # track best per generation for plotting
    best_per_gen = [tools.selBest(pop, 1)[0].fitness.values[0]]

    # generational loop
    for gen in range(1, NGEN + 1):
        # selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)
                if hasattr(c1.fitness, "values"):
                    del c1.fitness.values
                if hasattr(c2.fitness, "values"):
                    del c2.fitness.values

        # mutation
        for m in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(m)
                if hasattr(m.fitness, "values"):
                    del m.fitness.values

        # evaluate new/changed individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # replace population
        pop[:] = offspring

        # record best
        best = tools.selBest(pop, 1)[0].fitness.values[0]
        best_per_gen.append(best)
        print(f"Gen {gen:3d} | best fitness = {best:.4f}")

    # Plot: best fitness by generation
    plt.figure(figsize=(8, 4))
    plt.plot(best_per_gen, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (ΔX)")
    plt.title("EA: Best Fitness over 100 Generations")
    plt.grid(True)
    plt.savefig("fitness_ea_best.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
