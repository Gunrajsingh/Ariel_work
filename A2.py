"""
Assignment 2 — Gecko robot, template-style
Baseline setup with random controller (from template),
plus oscillator controller and numeric fitness.
"""

import math
import numpy as np
import mujoco
import matplotlib.pyplot as plt

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

HINGE_LIMIT = math.pi / 2
DT = 1/240

# ----------------------------
# Template function: random move
# ----------------------------
def random_move(data, rng):
    delta = 0.05
    u = data.ctrl.copy()
    u = u + rng.uniform(-HINGE_LIMIT, HINGE_LIMIT, size=data.ctrl.shape) * delta
    u = np.clip(u, -HINGE_LIMIT, HINGE_LIMIT)
    return u

# ----------------------------
# Template function: plot history
# ----------------------------
def show_qpos_history(history, label="trajectory"):
    history = np.array(history)
    plt.plot(history[:, 0], history[:, 1], label=label)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Gecko core trajectory")
    plt.show()

# ----------------------------
# New: oscillator controller
# ----------------------------
def oscillator_move(t, data, params):
    nu = data.ctrl.shape[0]
    A = params[0:nu]
    F = params[nu:2*nu]
    P = params[2*nu:3*nu]
    tt = t * DT
    u = A * np.sin(2*np.pi*F*tt + P) * HINGE_LIMIT
    return np.clip(u, -HINGE_LIMIT, HINGE_LIMIT)

# ----------------------------
# Main simulation
# ----------------------------
def main():
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0, 0, 0.1])  # small lift

    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    # Find core geom
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    core = to_track[0]

    rng = np.random.default_rng(42)

    # --- Random controller run ---
    history_random = []
    for t in range(500):
        u = random_move(data, rng)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        history_random.append(core.xpos[:2].copy())

    fit_random = float(np.linalg.norm(history_random[-1] - history_random[0]))
    print(f"[Random] fitness = {fit_random:.4f}")

    # --- Oscillator controller run ---
    nu = model.nu
    params = np.concatenate([
        np.full(nu, 0.6),                            # amplitudes
        np.full(nu, 1.0),                            # frequencies
        np.linspace(0.0, 2*np.pi, nu, endpoint=False) # phases
    ])

    history_osc = []
    for t in range(1000):
        u = oscillator_move(t, data, params)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        history_osc.append(core.xpos[:2].copy())

    fit_osc = float(np.linalg.norm(history_osc[-1] - history_osc[0]))
    print(f"[Oscillator] fitness = {fit_osc:.4f}")

    # Optional: plot for report
    show_qpos_history(history_random, label="Random")
    show_qpos_history(history_osc, label="Oscillator")

if __name__ == "__main__":
    main()
