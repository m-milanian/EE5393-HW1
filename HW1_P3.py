#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import multiprocessing as mp
import os
import random
import signal
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
if "MPLCONFIGDIR" not in os.environ:
    mpl_cfg = os.path.join(os.getcwd(), ".mplconfig")
    os.makedirs(mpl_cfg, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_cfg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------
# Gillespie SSA core
# ---------------------------

@dataclass(frozen=True)
class Reaction:
    reactants: Dict[str, int]
    products: Dict[str, int]
    rate: float

def mass_action_propensity(state: Dict[str, int], rxn: Reaction) -> float:
    """Standard mass-action with integer stoichiometries (assumes small stoich)."""
    a = rxn.rate
    for sp, sto in rxn.reactants.items():
        n = state.get(sp, 0)
        if n < sto:
            return 0.0
        # n choose sto  (sto is 1/2/3 in our constructions)
        if sto == 1:
            a *= n
        elif sto == 2:
            a *= n * (n - 1) / 2.0
        elif sto == 3:
            a *= n * (n - 1) * (n - 2) / 6.0
        else:
            # generic
            comb = 1.0
            for k in range(sto):
                comb *= (n - k) / (k + 1.0)
            a *= comb
        if a == 0.0:
            return 0.0
    return float(a)

def fire(state: Dict[str, int], rxn: Reaction) -> None:
    for sp, sto in rxn.reactants.items():
        state[sp] -= sto
    for sp, sto in rxn.products.items():
        state[sp] = state.get(sp, 0) + sto
    # keep nonnegative
    for sp in list(state.keys()):
        if state[sp] == 0:
            # optional cleanup
            pass

def ssa_simulate(
    reactions: List[Reaction],
    init: Dict[str, int],
    T: float,
    rng: random.Random,
) -> Dict[str, int]:
    t = 0.0
    state = dict(init)

    while t < T:
        props = [mass_action_propensity(state, r) for r in reactions]
        a0 = sum(props)
        if a0 <= 0.0:
            break
        u1 = rng.random()
        u2 = rng.random()

        tau = -math.log(u1) / a0
        t += tau
        # select reaction
        thresh = u2 * a0
        s = 0.0
        for r, a in zip(reactions, props):
            s += a
            if s >= thresh:
                fire(state, r)
                break

    return state


G_REACTIONS: List[Reaction] = []
G_INIT: Dict[str, int] = {}
G_T: float = 0.0


def _init_worker(reactions: List[Reaction], init: Dict[str, int], T: float) -> None:
    global G_REACTIONS, G_INIT, G_T
    G_REACTIONS = reactions
    G_INIT = init
    G_T = T
    # parent handles Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_one_trial(seed: int) -> Dict[str, int]:
    rng = random.Random(seed)
    return ssa_simulate(G_REACTIONS, G_INIT, G_T, rng)


def _mp_context():
    if os.name == "posix":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    return mp.get_context()


# ---------------------------
# Problem 3a CRN
# ---------------------------

def build_crn_3a() -> Tuple[List[Reaction], List[str]]:
    # Species list is not strictly required, but helpful for debugging
    species = ["b","d","Y","Yp","c","m","mp","X","a","Z","w"]

    # Choose rates with the intended separation:
    # fastest ~ 1e5-1e6, fast ~ 1e4, medium ~ 1e3, slow ~ 1e0-1e2
    r = []
    # Log module
    r.append(Reaction({"b":1}, {"b":1,"d":1}, 100.0))             # b -> b + d  (slow)
    r.append(Reaction({"d":1,"Y":2}, {"d":1,"c":1,"Yp":1}, 1e5))  # d + 2Y -> c + Yp + d (very fast)
    r.append(Reaction({"c":2}, {"c":1}, 1e6))                     # 2c -> c (fastest)
    r.append(Reaction({"d":1}, {}, 1e4))                          # d -> 0 (fast)
    r.append(Reaction({"Yp":1}, {"Y":1}, 1e3))                    # Yp -> Y (medium)
    r.append(Reaction({"c":1}, {"m":1}, 1e3))                     # c -> m (medium)

    # Multiplication module (repeated addition)
    r.append(Reaction({"X":1}, {"a":1}, 1.0))                     # X -> a (slow)
    r.append(Reaction({"a":1,"m":1}, {"a":1,"mp":1,"Z":1}, 1e5))  # a + m -> a + mp + Z (very fast)
    r.append(Reaction({"a":1}, {}, 1e4))                          # a -> 0 (fast)
    r.append(Reaction({"mp":1}, {"m":1}, 1e3))                    # mp -> m (medium)
    return r, species


# ---------------------------
# Problem 3b CRN
# ---------------------------

def build_crn_3b() -> Tuple[List[Reaction], List[str]]:
    # Here we compute Z = 2^{log2(X0)}; with powers-of-two inputs this equals X0.
    species = ["b","dlog","X","Xp","c","m","w","dexp","Z","Zp"]

    r = []
    # Log module but driven by X instead of Y (same pairing idea):
    r.append(Reaction({"b":1}, {"b":1,"dlog":1}, 100.0))               # b -> b + dlog  (slow)
    r.append(Reaction({"dlog":1,"X":2}, {"dlog":1,"c":1,"Xp":1}, 1e4)) # dlog + 2X -> c + Xp + dlog (fast)
    r.append(Reaction({"c":2}, {"c":1}, 1e4))                          # 2c -> c (fast)
    r.append(Reaction({"dlog":1}, {}, 1600.0))                         # dlog -> 0 (fast)
    r.append(Reaction({"Xp":1}, {"X":1}, 100.0))                       # Xp -> X (medium)
    r.append(Reaction({"c":1}, {"m":1}, 100.0))                        # c -> m (medium)

    # Exponentiation module: Z starts at 1 and doubles once per m
    r.append(Reaction({"m":1}, {"dexp":1}, 10.0))                      # m -> dexp (slow-ish)
    r.append(Reaction({"dexp":1,"Z":1}, {"dexp":1,"Zp":2}, 1e4))        # dexp + Z -> dexp + 2Zp (fast)
    r.append(Reaction({"dexp":1}, {}, 1600.0))                         # dexp -> 0 (fast)
    r.append(Reaction({"Zp":1}, {"Z":1}, 100.0))                       # Zp -> Z (medium)

    return r, species


# ---------------------------
# Experiments + plots
# ---------------------------

def run_trials(
    reactions: List[Reaction],
    init: Dict[str, int],
    T: float,
    trials: int,
    seed: int,
    nproc: int = 1,
    chunksize: int = 32,
    progress_label: str = "",
) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    if nproc <= 1:
        rng = random.Random(seed)
        step = max(1, trials // 5)
        for i in range(trials):
            s = ssa_simulate(reactions, init, T, rng)
            out.append(s)
            if progress_label and ((i + 1) % step == 0 or i + 1 == trials):
                print(f"  {progress_label}: {i+1}/{trials} trials", flush=True)
        return out

    ctx = _mp_context()
    step = max(1, trials // 5)
    seeds = ((seed + 104729 * k) & 0xFFFFFFFF for k in range(trials))
    with ctx.Pool(
        processes=nproc,
        initializer=_init_worker,
        initargs=(reactions, init, T),
    ) as pool:
        for i, s in enumerate(pool.imap_unordered(_run_one_trial, seeds, chunksize=max(1, chunksize)), 1):
            out.append(s)
            if progress_label and (i % step == 0 or i == trials):
                print(f"  {progress_label}: {i}/{trials} trials", flush=True)
    return out

def mean_of_species(states: List[Dict[str,int]], sp: str) -> float:
    return float(sum(st.get(sp,0) for st in states)) / len(states)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["3a","3b"], default="3a",
                    help="Which subproblem to run (default: 3a).")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nproc", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--chunksize", type=int, default=32)
    args = ap.parse_args()

    if args.part == "3a":
        reactions, _ = build_crn_3a()
        X_vals = [1,2,4,8,16,32]
        Y_vals = [2,4,8,16,32,64,128]
        pairs = [(x,y) for x in X_vals for y in Y_vals]

        computed = []
        truth = []
        print(
            f"Running part 3a with {len(pairs)} input pairs, trials={args.trials}, "
            f"T={args.T}, nproc={args.nproc}"
        )
        for k, (x0, y0) in enumerate(pairs, 1):
            t0 = time.time()
            init = {"b":1, "d":0, "Y":y0, "Yp":0, "c":0, "m":0, "mp":0, "X":x0, "a":0, "Z":0, "w":0}
            lbl = f"[{k}/{len(pairs)}] X={x0}, Y={y0}"
            states = run_trials(
                reactions,
                init,
                args.T,
                args.trials,
                args.seed + 1009 * k,
                nproc=args.nproc,
                chunksize=args.chunksize,
                progress_label=lbl,
            )
            zbar = mean_of_species(states, "Z")
            computed.append(zbar)
            truth.append(x0 * int(math.log2(y0)))
            print(f"  {lbl}: done in {time.time()-t0:.1f}s, mean Z={zbar:.3f}", flush=True)

        # scatter
        plt.figure()
        plt.plot(truth, truth, linestyle="--", label="Perfect fit")
        plt.scatter(truth, computed, s=20)
        plt.xlabel("True value  $X_0\\log_2(Y_0)$")
        plt.ylabel("Computed (mean final $Z$)")
        plt.title("Problem 3a: computed vs true")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("p3a_scatter.png", dpi=200)
        plt.close()
        print("Wrote p3a_scatter.png")

    else:
        reactions, _ = build_crn_3b()
        X_vals = [1,2,4,8,16,32,64,128,256]
        computed = []
        truth = []
        print(
            f"Running part 3b with {len(X_vals)} inputs, trials={args.trials}, "
            f"T={args.T}, nproc={args.nproc}"
        )
        for k, x0 in enumerate(X_vals, 1):
            t0 = time.time()
            init = {"b":1, "dlog":0, "X":x0, "Xp":0, "c":0, "m":0, "w":0, "dexp":0, "Z":1, "Zp":0}
            lbl = f"[{k}/{len(X_vals)}] X={x0}"
            states = run_trials(
                reactions,
                init,
                args.T,
                args.trials,
                args.seed + 1009 * k,
                nproc=args.nproc,
                chunksize=args.chunksize,
                progress_label=lbl,
            )
            zbar = mean_of_species(states, "Z")
            computed.append(zbar)
            truth.append(x0)  # since x0 is power of two, 2^{log2 x0} = x0
            print(f"  {lbl}: done in {time.time()-t0:.1f}s, mean Z={zbar:.3f}", flush=True)

        plt.figure()
        plt.plot(truth, truth, linestyle="--", label="Perfect fit")
        plt.scatter(truth, computed, s=20)
        plt.xlabel("True value  $X_0$")
        plt.ylabel("Computed (mean final $Z$)")
        plt.title("Problem 3b: computed vs true")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("p3b_scatter.png", dpi=200)
        plt.close()
        print("Wrote p3b_scatter.png")

if __name__ == "__main__":
    main()
