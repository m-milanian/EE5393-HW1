#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import random
import time
import multiprocessing as mp
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ============================================================
# Fenwick Tree (Binary Indexed Tree) for weighted sampling O(log R)
# ============================================================

class Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.bit = [0.0] * (n + 1)

    def build(self, arr: List[float]) -> None:
        for i, v in enumerate(arr):
            self.add(i, v)

    def add(self, i: int, delta: float) -> None:
        j = i + 1
        while j <= self.n:
            self.bit[j] += delta
            j += j & -j

    def total(self) -> float:
        s = 0.0
        j = self.n
        while j > 0:
            s += self.bit[j]
            j -= j & -j
        return s

    def find_prefix_index(self, u: float) -> int:
        """
        Return smallest index i s.t. prefix_sum(i) > u.
        Requires 0 <= u < total.
        """
        idx = 0
        # largest power of two <= n
        bitmask = 1
        while (bitmask << 1) <= self.n:
            bitmask <<= 1
        while bitmask:
            nxt = idx + bitmask
            if nxt <= self.n and self.bit[nxt] <= u:
                u -= self.bit[nxt]
                idx = nxt
            bitmask >>= 1
        return idx  # 0-based (because idx is count <= target)


# ============================================================
# Parsing lambda.r and lambda.in
# ============================================================

def _parse_side(side: str) -> Dict[str, int]:
    side = side.strip()
    if side == "":
        return {}
    toks = side.split()
    if len(toks) % 2 != 0:
        raise ValueError(f"Bad reaction side (odd token count): {side}")
    out: Dict[str, int] = {}
    for i in range(0, len(toks), 2):
        sp = toks[i]
        sto = int(toks[i + 1])
        out[sp] = out.get(sp, 0) + sto
    return out


def parse_lambda_r(path: str) -> Tuple[List[Tuple[Dict[str, int], Dict[str, int], float]], List[str]]:
    reactions = []
    species = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(":")]
            if len(parts) != 3:
                continue
            lhs_s, rhs_s, k_s = parts
            lhs = _parse_side(lhs_s)
            rhs = _parse_side(rhs_s)
            k = float(k_s)
            for sp in lhs:
                species.add(sp)
            for sp in rhs:
                species.add(sp)
            reactions.append((lhs, rhs, k))
    return reactions, sorted(species)


def parse_lambda_in(path: str) -> Dict[str, int]:
    init: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            sp = toks[0]
            val = int(toks[1])
            init[sp] = val
    return init


# ============================================================
# Model compilation
# ============================================================

def comb_small(n: int, k: int) -> int:
    if k == 1:
        return n
    if k == 2:
        return n * (n - 1) // 2
    if k == 3:
        return n * (n - 1) * (n - 2) // 6
    return 0


@dataclass(frozen=True)
class Rxn:
    react: Tuple[Tuple[int, int], ...]
    prod:  Tuple[Tuple[int, int], ...]
    k: float
    affected: Tuple[int, ...]


def build_model(lambda_r_path: str, lambda_in_path: str):
    rxns_raw, species = parse_lambda_r(lambda_r_path)
    init_map = parse_lambda_in(lambda_in_path)

    idx = {sp: i for i, sp in enumerate(species)}
    nS = len(species)

    x0 = [0] * nS
    for sp, v in init_map.items():
        x0[idx[sp]] = v

    # compile reactions
    reactant_sets: List[set] = []
    rxns_tmp = []
    for lhs, rhs, k in rxns_raw:
        react = tuple((idx[sp], sto) for sp, sto in lhs.items())
        prod  = tuple((idx[sp], sto) for sp, sto in rhs.items())
        reactant_sets.append(set(j for (j, _) in react))
        rxns_tmp.append((react, prod, k))

    # dependency map: species -> reactions whose propensity depends on it (reactants)
    dep: Dict[int, List[int]] = defaultdict(list)
    for i, sp_set in enumerate(reactant_sets):
        for j in sp_set:
            dep[j].append(i)

    rxns: List[Rxn] = []
    for i, (react, prod, k) in enumerate(rxns_tmp):
        changed = set([j for (j, _) in react] + [j for (j, _) in prod])
        affected = set()
        for j in changed:
            affected.update(dep.get(j, []))
        rxns.append(Rxn(
            react=react,
            prod=prod,
            k=k,
            affected=tuple(sorted(affected))
        ))

    for needed in ["MOI", "cI2", "Cro2"]:
        if needed not in idx:
            raise ValueError(f"Missing required species '{needed}'. Check lambda.r/.in.")

    return species, idx, x0, rxns


def propensity(x: List[int], r: Rxn) -> float:
    a = r.k
    for j, sto in r.react:
        n = x[j]
        if n < sto:
            return 0.0
        a *= comb_small(n, sto)
        if a == 0.0:
            return 0.0
    return float(a)


# ============================================================
# Global model for multiprocessing workers (avoid pickling each job)
# ============================================================

G_X0: List[int] = []
G_RXNS: List[Rxn] = []
G_IDX: Dict[str, int] = {}

def _init_worker(x0: List[int], rxns: List[Rxn], idx: Dict[str, int]):
    global G_X0, G_RXNS, G_IDX
    G_X0 = x0
    G_RXNS = rxns
    G_IDX = idx


def simulate_one_run(moi_value: int, seed: int, max_steps: int) -> int:
    """
    Return:
      1 => stealth hit first  (cI2 > 145)
      2 => hijack hit first   (Cro2 > 55)
      0 => censored / dead by max_steps
    """
    rng = random.Random(seed)

    x = G_X0[:]  # copy
    x[G_IDX["MOI"]] = moi_value

    cI2_i = G_IDX["cI2"]
    Cro2_i = G_IDX["Cro2"]

    # build propensities + Fenwick
    a = [0.0] * len(G_RXNS)
    for i, r in enumerate(G_RXNS):
        a[i] = propensity(x, r)
    fw = Fenwick(len(G_RXNS))
    fw.build(a)
    total = fw.total()

    for _ in range(max_steps):
        if x[cI2_i] > 145:
            return 1
        if x[Cro2_i] > 55:
            return 2
        if total <= 0.0:
            return 0

        u = rng.random() * total
        i = fw.find_prefix_index(u)
        r = G_RXNS[i]

        # fire
        for j, sto in r.react:
            x[j] -= sto
        for j, sto in r.prod:
            x[j] += sto

        # update affected propensities
        for k in r.affected:
            old = a[k]
            new = propensity(x, G_RXNS[k])
            if new != old:
                a[k] = new
                fw.add(k, new - old)
                total += (new - old)

    return 0


def run_batch(pool: mp.Pool, moi: int, n: int, max_steps: int, base_seed: int) -> Tuple[int, int, int]:
    """
    Run n independent trajectories in parallel.
    Returns (stealth, hijack, censored).
    """
    # deterministic unique seeds
    jobs = [(moi, (base_seed + 10_000*moi + t) & 0xFFFFFFFF, max_steps) for t in range(n)]
    outs = pool.starmap(simulate_one_run, jobs)

    stealth = sum(1 for o in outs if o == 1)
    hijack  = sum(1 for o in outs if o == 2)
    cens    = n - stealth - hijack
    return stealth, hijack, cens


# ============================================================
# Progressive estimator per MOI (fast + prints immediately)
# ============================================================

def estimate_for_moi(pool: mp.Pool, moi: int,
                     base_seed: int = 123,
                     pilot_n: int = 200,
                     pilot_steps: int = 10_000,
                     batch_n: int = 1000,
                     max_total: int = 10_000,
                     target_hw: float = 0.02,
                     max_steps_cap: int = 400_000) -> Tuple[float, float, float, float, float, int, int]:
    """
    Returns:
      pS, seS, pH, seH, pC, used_trials, max_steps_used
    """

    print(f"\n[MOI={moi}] pilot: {pilot_n} runs with max_steps={pilot_steps} ...", flush=True)
    st, hj, ce = run_batch(pool, moi, pilot_n, pilot_steps, base_seed)
    cens_rate = ce / pilot_n

    # choose starting max_steps based on pilot censoring
    if cens_rate > 0.90:
        max_steps = 200_000
    elif cens_rate > 0.70:
        max_steps = 100_000
    else:
        max_steps = 50_000

    stealth = hijack = cens = 0
    used = 0

    while used < max_total:
        n = min(batch_n, max_total - used)
        st, hj, ce = run_batch(pool, moi, n, max_steps, base_seed + used + 777)
        stealth += st
        hijack += hj
        cens += ce
        used += n

        pS = stealth / used
        pH = hijack / used
        pC = cens / used
        seS = math.sqrt(max(pS*(1-pS), 0.0) / used)
        seH = math.sqrt(max(pH*(1-pH), 0.0) / used)
        hwS = 1.96 * seS
        hwH = 1.96 * seH

        print(f"[MOI={moi}] trials={used:6d}  pS={pS:6.3f}±{hwS:5.3f}  "
              f"pH={pH:6.3f}±{hwH:5.3f}  pC={pC:6.3f}  max_steps={max_steps}",
              flush=True)

        # stop if tight enough
        if hwS <= target_hw and hwH <= target_hw:
            break

        # if still heavily censored, increase step cap
        if pC > 0.85 and max_steps < max_steps_cap:
            max_steps = min(max_steps_cap, int(1.5 * max_steps))

    pS = stealth / used
    pH = hijack / used
    pC = cens / used
    seS = math.sqrt(max(pS*(1-pS), 0.0) / used)
    seH = math.sqrt(max(pH*(1-pH), 0.0) / used)
    return pS, seS, pH, seH, pC, used, max_steps


# ============================================================
# Main
# ============================================================

def main():
    lambda_r_path = "lambda.r"
    lambda_in_path = "lambda.in"

    t0 = time.time()
    _, idx, x0, rxns = build_model(lambda_r_path, lambda_in_path)
    print(f"Loaded model: {len(rxns)} reactions, {len(idx)} species. Build time: {time.time()-t0:.2f}s")

    # Use fewer processes on laptops to avoid overhead
    nproc = max(1, mp.cpu_count() - 1)
    print(f"Using {nproc} worker processes.\n")

    print("Lambda phage randomness: estimate P(stealth) vs P(hijack) for MOI=1..10")
    print("Stealth  := hit cI2 > 145 first")
    print("Hijack   := hit Cro2 > 55 first\n")

    print("MOI  P(stealth)   SE      P(hijack)    SE      P(censored)   trials   max_steps")

    with mp.Pool(processes=nproc, initializer=_init_worker, initargs=(x0, rxns, idx)) as pool:
        for moi in range(1, 11):
            pS, seS, pH, seH, pC, used, max_steps = estimate_for_moi(
                pool=pool,
                moi=moi,
                base_seed=123,
                pilot_n=200,
                pilot_steps=10_000,
                batch_n=1000,
                max_total=10_000,
                target_hw=0.02,
                max_steps_cap=400_000
            )
            print(f"{moi:>3d}  {pS:>10.4f}  {seS:>7.4f}   {pH:>10.4f}  {seH:>7.4f}   "
                  f"{pC:>10.4f}   {used:>6d}   {max_steps:>8d}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    mp.freeze_support()  # important for macOS/Windows
    main()