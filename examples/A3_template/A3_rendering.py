"""
Assignment 3 — Robot Olympics (EA for BODY only, full utilities + rendering fix)

- Evolves: body genome = 3 vectors in [0,1]^64 (NDE inputs)
- Controller: loads your A2 MLP from 'controller_data' if present; else a small random NN
- Environment: OlympicArena
- Outputs per run:
  - __data__/A3_full/ea_best_over_generations.png      (fitness curve)
  - __data__/A3_full/gen_XXX_best.png                  (rendered snapshots, top-down)
  - __data__/A3_full/gen_XXX_path_overlay.png          (optional XY path overlay)
  - __data__/A3_full/gen_XXX_nde.json                  (t,c,r vectors)
  - __data__/A3_full/gen_XXX_decoded_graph.json        (body graph)
  - __data__/A3_full/ea_best_per_gen.csv               (CSV per-generation best fitness)

Notes:
- This version keeps "all the things" while fixing the renderer usage.
- You can switch the run mode via EA_RUN_MODE ("simple" recommended for EA speed).
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco as mj
import numpy as np
from deap import base, creator, tools
from rich.console import Console

# Ariel imports
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from mujoco import viewer  # only used if EA_RUN_MODE="launcher"/"no_control"

console = Console()

# =========================
# Paths / globals
# =========================
SCRIPT_NAME = "A3_full"
DATA = Path("__data__") / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)

# EA params
POP_SIZE = 60
N_GEN = 20
TOURNSIZE = 3
CXPB = 0.9
MUTPB = 0.25
SBX_ETA = 20.0
IMM_FRAC = 0.10
STAGNATION_STEPS = (5, 10)
MUTPB_BOOSTS = (1.8, 2.5)

# Sim / env
SIM_DURATION = 15.0  # seconds
SPAWN_POS = [-0.8, 0.0, 0.15]  # small bump on Z to reduce initial collisions
WARMUP_STEPS = 120             # ~0.5s at 240Hz
RATE_LIMIT_DU = 0.025
EA_RUN_MODE: Literal["simple", "frame", "video", "launcher", "no_control"] = "simple"

# Body encoding
BODY_L = 64
NUM_OF_MODULES = 30

# Plots/CSVs
EA_FITNESS_PNG = DATA / "ea_best_over_generations.png"
EA_FITNESS_CSV = DATA / "ea_best_per_gen.csv"

# Optional path overlay toggle
SAVE_PATH_OVERLAY = True

# =========================
# Load A2 MLP brain (optional)
# =========================
CONTROLLER_JSON_PATH = "controller_data"
_A2_BRAIN_CACHE = None  # lazy cache for params/shape

def _solve_inp_len(theta_len: int, hidden: int, out_dim: int) -> int:
    # theta = inp*h + h + h*h + h + h*out + out
    # => inp = (theta - out)/h - h - out - 2
    return int(round((theta_len - out_dim) / float(hidden) - hidden - out_dim - 2))

def _unpack_theta(theta: np.ndarray, inp: int, hidden: int, out_dim: int):
    i = 0
    W1 = theta[i : i + inp * hidden].reshape(inp, hidden); i += inp * hidden
    b1 = theta[i : i + hidden]; i += hidden
    W2 = theta[i : i + hidden * hidden].reshape(hidden, hidden); i += hidden * hidden
    b2 = theta[i : i + hidden]; i += hidden
    W3 = theta[i : i + hidden * out_dim].reshape(hidden, out_dim); i += hidden * out_dim
    b3 = theta[i : i + out_dim]
    return (W1, b1, W2, b2, W3, b3)

def _mlp_forward(x: np.ndarray, params) -> np.ndarray:
    W1, b1, W2, b2, W3, b3 = params
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return np.tanh(h2 @ W3 + b3)  # [-1, 1]

def _load_a2_brain(path: str = CONTROLLER_JSON_PATH):
    global _A2_BRAIN_CACHE
    if _A2_BRAIN_CACHE is not None:
        return _A2_BRAIN_CACHE
    try:
        with open(path, "r") as f:
            blob = json.load(f)
        theta = np.asarray(blob["theta"], dtype=float)
        hidden = int(blob["hidden"])
        out_saved = int(blob["nu"])
        inp_saved = _solve_inp_len(len(theta), hidden, out_saved)
        params = _unpack_theta(theta, inp_saved, hidden, out_saved)
        _A2_BRAIN_CACHE = dict(params=params, inp_saved=inp_saved, hidden=hidden, out_saved=out_saved)
        console.log(f"[Brain] Loaded A2 controller: inp={inp_saved}, hidden={hidden}, out={out_saved}")
    except Exception as e:
        console.log(f"[Brain] Could not load 'controller_data' ({e}); using random NN fallback.")
        _A2_BRAIN_CACHE = None
    return _A2_BRAIN_CACHE

# =========================
# Body genotype & operators
# =========================
def init_body_genotype(rng: np.random.Generator, n: int):
    return (rng.random(n).astype(np.float32),
            rng.random(n).astype(np.float32),
            rng.random(n).astype(np.float32))

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

def sbx_body(g1, g2, eta=SBX_ETA, rng: np.random.Generator | None = None):
    a1, b1 = _sbx_pair(g1[0], g2[0], eta, rng=rng)
    a2, b2 = _sbx_pair(g1[1], g2[1], eta, rng=rng)
    a3, b3 = _sbx_pair(g1[2], g2[2], eta, rng=rng)
    return (a1, a2, a3), (b1, b2, b3)

def block_mutation(g, indpb=1.0 / BODY_L, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    t, c, r = g
    mt = rng.random(t.shape) < indpb
    mc = rng.random(c.shape) < indpb
    mr = rng.random(r.shape) < indpb
    t = t.copy(); c = c.copy(); r = r.copy()
    t[mt] = rng.random(mt.sum()).astype(np.float32)
    c[mc] = rng.random(mc.sum()).astype(np.float32)
    r[mr] = rng.random(mr.sum()).astype(np.float32)
    return (t, c, r)

# =========================
# Build body (decode NDE -> graph -> spec)
# =========================
@dataclass
class BuiltBody:
    nde: NeuralDevelopmentalEncoding
    decoded_graph: Any
    mjspec: Any

def build_body(geno, nde_modules: int) -> BuiltBody:
    t, c, r = [np.asarray(x, dtype=np.float32) for x in geno]
    nde = NeuralDevelopmentalEncoding(number_of_modules=nde_modules)
    p_mats = nde.forward([t, c, r])
    decoder = HighProbabilityDecoder(nde_modules)
    graph = decoder.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])
    spec = construct_mjspec_from_graph(graph)
    return BuiltBody(nde=nde, decoded_graph=graph, mjspec=spec)

def save_body_artifacts(run_dir: Path, built: BuiltBody, tag: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    save_graph_as_json(built.decoded_graph, str(run_dir / f"{tag}_decoded_graph.json"))
    # Save the NDE vectors (t,c,r) in a simple JSON for reproducibility
    try:
        nde_json = {
            "t": built.nde.model_t.tolist() if hasattr(built.nde, "model_t") else [],
            "c": built.nde.model_c.tolist() if hasattr(built.nde, "model_c") else [],
            "r": built.nde.model_r.tolist() if hasattr(built.nde, "model_r") else [],
            "n_modules": NUM_OF_MODULES,
        }
    except Exception:
        nde_json = {"n_modules": NUM_OF_MODULES}
    with open(run_dir / f"{tag}_nde.json", "w") as f:
        json.dump(nde_json, f, indent=2)

# =========================
# Rendering helpers — FIXED to use model + data (+ camera)
# =========================
def topdown_camera() -> mj.MjvCamera:
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat = [2.5, 0.0, 0.0]
    cam.distance = 10
    cam.azimuth = 0
    cam.elevation = -90
    return cam

def render_snapshot(model: mj.MjModel, data: mj.MjData, save_path: str):
    single_frame_renderer(
        model,
        data,
        camera=topdown_camera(),
        save=True,
        save_path=save_path,
    )

def render_path_overlay(history_xyz: list[list[float]], save_path: str):
    """
    Recreates a background and overlays the XY path. Uses the same topdown camera so
    the overlay roughly matches the arena footprint.
    """
    camera = topdown_camera()
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    # Save background image
    bg_path = str(Path(save_path).with_suffix(".bg.png"))
    single_frame_renderer(
        model,
        data,
        camera=camera,
        save=True,
        save_path=bg_path,
    )

    import matplotlib.pyplot as plt
    img = plt.imread(bg_path)

    # Convert path history into a rough pixel path using simple proportional mapping.
    # (This is approximate; for exact reprojection you'd calibrate pixels->meters.)
    h, w, _ = img.shape
    xs = [p[0] for p in history_xyz]
    ys = [p[1] for p in history_xyz]
    if not xs or not ys:
        return
    # Normalize XY to background size (simple min/max scaling across path extent)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    spanx = max(1e-6, xmax - xmin)
    spany = max(1e-6, ymax - ymin)
    px = [int((x - xmin) / spanx * (w * 0.8) + w * 0.1) for x in xs]
    py = [int((y - ymin) / spany * (h * 0.8) + h * 0.1) for y in ys]

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.plot(px, py, "b-", linewidth=2, label="Path")
    ax.plot(px[0], py[0], "go", label="Start")
    ax.plot(px[-1], py[-1], "ro", label="End")
    ax.set_title("Robot Path (approx. overlay)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# =========================
# Evaluation (supports multiple run modes)
# =========================
def _choose_track_body(model: mj.MjModel) -> str:
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "robot-core")
    if bid != -1:
        return "robot-core"
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, 1 if model.nbody > 1 else 0)

def _episode_controller_factory(model: mj.MjModel, data: mj.MjData, tracker: Tracker):
    """Returns a MuJoCo control callback for this episode."""
    brain = _load_a2_brain()
    out = model.nu  # actuators
    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=float).reshape(out, 2)
    low = ctrlrange[:, 0].copy(); high = ctrlrange[:, 1].copy()
    limited = np.array(model.actuator_ctrllimited, dtype=bool).reshape(-1)
    bad = (~limited) | ~np.isfinite(low) | ~np.isfinite(high) | (high <= low)
    low[bad] = -1.0; high[bad] = 1.0
    center = 0.5 * (low + high); halfspan = 0.5 * (high - low)

    u_apply = center.copy()
    rate = np.minimum(0.02 * (high - low), RATE_LIMIT_DU)

    def cb(m: mj.MjModel, d: mj.MjData):
        nonlocal u_apply
        t_now = d.time
        x_pq = np.concatenate([d.qpos, d.qvel]).astype(float, copy=False)

        if brain is not None:
            inp_saved = brain["inp_saved"]; out_saved = brain["out_saved"]; params = brain["params"]
            x_in = np.zeros(inp_saved, dtype=float)
            x_in[-3:] = [t_now, math.sin(2 * math.pi * t_now), math.cos(2 * math.pi * t_now)]
            k = min(x_pq.size, inp_saved - 3)
            if k > 0:
                x_in[:k] = x_pq[:k]
            y = _mlp_forward(x_in, params)
            y_out = y[:out] if out_saved >= out else np.resize(y, out)
        else:
            # tiny random NN fallback
            hid = 8
            w1 = RNG.normal(0.0, 0.1, size=(x_pq.size + 3, hid))
            w2 = RNG.normal(0.0, 0.1, size=(hid, hid))
            w3 = RNG.normal(0.0, 0.1, size=(hid, out))
            x_in = np.concatenate([x_pq, [t_now, math.sin(2 * math.pi * t_now), math.cos(2 * math.pi * t_now)]])
            y_tmp = np.tanh(x_in @ w1)
            y_tmp = np.tanh(y_tmp @ w2)
            y_out = np.tanh(y_tmp @ w3)

        y_out = np.nan_to_num(y_out, nan=0.0, posinf=0.0, neginf=0.0)
        u_target = center + halfspan * y_out
        # warm-up
        ramp = 1.0 if WARMUP_STEPS <= 0 else min(1.0, d.time / (WARMUP_STEPS * (1 / m.opt.timestep)))
        u_cmd = center + ramp * (u_target - center)
        du = np.clip(u_cmd - u_apply, -rate, rate)
        u_apply = np.clip(u_apply + du, low, high)
        return u_apply.astype(np.float64, copy=False)

    return cb

def run_episode_for_genotype(
    body_geno,
    duration: float = SIM_DURATION,
    mode: Literal["simple", "frame", "video", "launcher", "no_control"] = EA_RUN_MODE,
):
    """
    Build body, run in OlympicArena with controller, return:
      (fitness, history_xyz, (model, data))
    Fitness = x-displacement of tracked body over the episode.
    """
    # Build world/model
    mj.set_mjcb_control(None)
    world = OlympicArena()
    built = build_body(body_geno, nde_modules=NUM_OF_MODULES)

    world.spawn(built.mjspec.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # Tracker on main body
    track_name = _choose_track_body(model)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY, name_to_bind=track_name)

    # Controller
    ctrl_cb = _episode_controller_factory(model, data, tracker)
    ctrl = Controller(controller_callback_function=ctrl_cb, tracker=tracker)
    if ctrl.tracker is not None:
        ctrl.tracker.setup(world.spec, data)

    # Control callback
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Run by mode
    if mode == "simple":
        simple_runner(model, data, duration=duration)
    elif mode == "frame":
        # Single snapshot and return
        render_snapshot(model, data, save_path=str(DATA / "debug_frame.png"))
    elif mode == "video":
        recorder = VideoRecorder(output_folder=str(DATA / "videos"))
        video_renderer(model, data, duration=duration, video_recorder=recorder)
    elif mode == "launcher":
        with viewer.launch_passive(model, data) as v:
            t0 = time.perf_counter()
            while v.is_running() and (time.perf_counter() - t0) < duration:
                target = time.perf_counter() - t0
                while data.time < target and v.is_running():
                    mj.mj_step(model, data)
                v.sync()
    elif mode == "no_control":
        mj.set_mjcb_control(None)
        with viewer.launch_passive(model, data) as v:
            t0 = time.perf_counter()
            while v.is_running() and (time.perf_counter() - t0) < duration:
                target = time.perf_counter() - t0
                while data.time < target and v.is_running():
                    mj.mj_step(model, data)
                v.sync()
    else:
        simple_runner(model, data, duration=duration)

    # History for fitness
    hist = tracker.history.get("xpos", [[]])
    if not hist or not isinstance(hist[0], (list, tuple, np.ndarray)):
        # fallback: start/end from body pose
        start = np.array(data.body(track_name).xpos[:3], dtype=float)
        end = np.array(data.body(track_name).xpos[:3], dtype=float)
        hist = [start.tolist(), end.tolist()]

    disp_x = float(hist[-1][0] - hist[0][0])
    return disp_x, hist, (model, data)

# =========================
# CSV helpers
# =========================
def write_csv(path: Path, rows: list[dict], header: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# =========================
# DEAP setup
# =========================
try:
    creator.FitnessMax
except AttributeError:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
try:
    creator.BodyIndividual
except AttributeError:
    creator.create("BodyIndividual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def init_individual():
    geno = init_body_genotype(RNG, BODY_L)
    return creator.BodyIndividual([geno])

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

def mate(ind1, ind2):
    c1, c2 = sbx_body(ind1[0], ind2[0], eta=SBX_ETA, rng=RNG)
    ind1[0] = c1; ind2[0] = c2
    if hasattr(ind1.fitness, "values"): del ind1.fitness.values
    if hasattr(ind2.fitness, "values"): del ind2.fitness.values
    return ind1, ind2

def mutate(ind):
    ind[0] = block_mutation(ind[0], indpb=1.0 / BODY_L, rng=RNG)
    if hasattr(ind.fitness, "values"): del ind.fitness.values
    return (ind,)

toolbox.register("mate", mate)
toolbox.register("mutate", mutate)

def evaluate_individual(ind):
    geno = ind[0]
    try:
        fit, hist, md = run_episode_for_genotype(geno, duration=SIM_DURATION, mode=EA_RUN_MODE)
    except Exception as e:
        console.log(f"[Eval] Exception: {e}. Penalizing.")
        fit, hist, md = -1e9, [[0, 0, 0]], None
    return (fit,)

toolbox.register("evaluate", evaluate_individual)

# =========================
# EA loop with rendering + CSV + path overlays
# =========================
def run_ea():
    random.seed(SEED); np.random.seed(SEED)
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate initial pop
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    ELITE_K = max(1, POP_SIZE // 20)
    hof = tools.HallOfFame(ELITE_K)
    hof.update(pop)

    best_per_gen = [tools.selBest(pop, 1)[0].fitness.values[0]]
    csv_rows = [{"generation": 0, "best_fitness": best_per_gen[-1]}]

    # Save & render gen 0 best
    best0 = tools.selBest(pop, 1)[0]
    built0 = build_body(best0[0], NUM_OF_MODULES)
    tag0 = "gen_000_best"
    save_body_artifacts(DATA, built0, tag0)
    try:
        world0 = OlympicArena()
        world0.spawn(built0.mjspec.spec, spawn_position=SPAWN_POS)
        m0 = world0.spec.compile(); d0 = mj.MjData(m0)
        render_snapshot(m0, d0, save_path=str(DATA / f"{tag0}.png"))
    except Exception as e:
        console.log(f"[Render gen 0] {e}")

    no_improve = 0
    best_so_far = best_per_gen[-1]
    t0 = time.time()

    for gen in range(1, N_GEN + 1):
        # selection + clone
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)

        # adaptive mutation
        adapt_mutpb = MUTPB
        if no_improve >= STAGNATION_STEPS[0]:
            adapt_mutpb = min(1.0, MUTPB * MUTPB_BOOSTS[0])
        if no_improve >= STAGNATION_STEPS[1]:
            adapt_mutpb = min(1.0, MUTPB * MUTPB_BOOSTS[1])

        for m in offspring:
            if random.random() < adapt_mutpb:
                toolbox.mutate(m)

        # random immigrants
        n_imm = max(0, int(IMM_FRAC * len(offspring)))
        for _ in range(n_imm):
            offspring[random.randrange(len(offspring))] = toolbox.individual()

        # eval
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        hof.update(offspring)
        elites = list(map(toolbox.clone, hof.items))

        offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        pop = elites + offspring[: max(0, POP_SIZE - len(elites))]

        # logging
        best = tools.selBest(pop, 1)[0]
        best_per_gen.append(best.fitness.values[0])
        csv_rows.append({"generation": gen, "best_fitness": best_per_gen[-1]})

        if best.fitness.values[0] > best_so_far + 1e-12:
            best_so_far = best.fitness.values[0]; no_improve = 0
        else:
            no_improve += 1
        console.log(
            f"[EA] Gen {gen:02d} | best={best.fitness.values[0]:.4f} | no_improve={no_improve:2d} | t={time.time()-t0:.1f}s"
        )

        # Save artifacts + rendered snapshot (+ path overlay from a quick headless sim)
        built = build_body(best[0], NUM_OF_MODULES)
        tag = f"gen_{gen:03d}_best"
        save_body_artifacts(DATA, built, tag)
        try:
            # render static snapshot
            world_g = OlympicArena()
            world_g.spawn(built.mjspec.spec, spawn_position=SPAWN_POS)
            mg = world_g.spec.compile(); dg = mj.MjData(mg)
            render_snapshot(mg, dg, save_path=str(DATA / f"{tag}.png"))
        except Exception as e:
            console.log(f"[Render {tag}] {e}")

        # Optional: run a quick headless sim to capture path and overlay
        if SAVE_PATH_OVERLAY:
            try:
                fit_tmp, hist_tmp, _ = run_episode_for_genotype(best[0], duration=SIM_DURATION, mode="simple")
                render_path_overlay(hist_tmp, save_path=str(DATA / f"{tag}_path_overlay.png"))
            except Exception as e:
                console.log(f"[Path overlay {tag}] {e}")

    # Write CSV
    write_csv(EA_FITNESS_CSV, csv_rows, header=["generation", "best_fitness"])
    console.log(f"[EA] Wrote per-gen CSV: {EA_FITNESS_CSV}")

    # Plot curve
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(best_per_gen, marker="o")
        plt.xlabel("Generation"); plt.ylabel("Best fitness (x-displacement)")
        plt.title(f"EA best over {N_GEN} generations (pop={POP_SIZE})")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(EA_FITNESS_PNG, dpi=150); plt.close()
        console.log(f"[EA] Saved curve to {EA_FITNESS_PNG}")
    except Exception as e:
        console.log(f"[Plot] {e}")

    return tools.selBest(pop, 1)[0], best_per_gen

# =========================
# Main
# =========================
def main():
    best, curve = run_ea()
    print(f"[EA] Done. Best fitness = {best.fitness.values[0]:.4f}")

if __name__ == "__main__":
    main()
