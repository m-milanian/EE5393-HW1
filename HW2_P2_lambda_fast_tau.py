#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import multiprocessing as mp
import os
import signal
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


_PLT = None


def _get_pyplot():
    """
    Lazy import to avoid matplotlib/font-cache overhead in worker processes.
    """
    global _PLT
    if _PLT is None:
        if "MPLCONFIGDIR" not in os.environ:
            cfg = os.path.join(os.getcwd(), ".mplconfig")
            os.makedirs(cfg, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = cfg
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _PLT = plt
    return _PLT


# -------------------------
# Parsing
# -------------------------
def _parse_side(side: str) -> Dict[str, int]:
    side = side.strip()
    if side == "":
        return {}
    toks = side.split()
    if len(toks) % 2 != 0:
        raise ValueError(f"Bad reaction side: {side}")
    out: Dict[str, int] = {}
    for i in range(0, len(toks), 2):
        sp = toks[i]
        sto = int(toks[i + 1])
        out[sp] = out.get(sp, 0) + sto
    return out


def parse_lambda_r(path: str):
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
            init[toks[0]] = int(toks[1])
    return init


# -------------------------
# Model
# -------------------------
@dataclass(frozen=True)
class Rxn:
    react_j: Tuple[int, ...]
    react_s: Tuple[int, ...]
    prod_j: Tuple[int, ...]
    prod_s: Tuple[int, ...]
    k: float


def build_model(lambda_r_path: str, lambda_in_path: str):
    rxns_raw, species = parse_lambda_r(lambda_r_path)
    init_map = parse_lambda_in(lambda_in_path)
    idx = {sp: i for i, sp in enumerate(species)}

    x0 = np.zeros(len(species), dtype=np.int64)
    for sp, v in init_map.items():
        x0[idx[sp]] = v

    rxns: List[Rxn] = []
    for lhs, rhs, k in rxns_raw:
        react = [(idx[sp], sto) for sp, sto in lhs.items()]
        prod = [(idx[sp], sto) for sp, sto in rhs.items()]
        rxns.append(
            Rxn(
                react_j=tuple(j for j, _ in react),
                react_s=tuple(sto for _, sto in react),
                prod_j=tuple(j for j, _ in prod),
                prod_s=tuple(sto for _, sto in prod),
                k=float(k),
            )
        )

    for needed in ["MOI", "cI2", "Cro2"]:
        if needed not in idx:
            raise ValueError(f"Missing required species '{needed}'.")

    # Dense stoichiometry for vectorized state updates: x += K @ nu
    nu = np.zeros((len(rxns), len(species)), dtype=np.int64)
    for i, r in enumerate(rxns):
        for j, sto in zip(r.react_j, r.react_s):
            nu[i, j] -= int(sto)
        for j, sto in zip(r.prod_j, r.prod_s):
            nu[i, j] += int(sto)

    return species, idx, x0, rxns, nu


# -------------------------
# Propensities
# -------------------------
def propensity(x: np.ndarray, r: Rxn) -> float:
    a = r.k
    for j, sto in zip(r.react_j, r.react_s):
        n = int(x[j])
        if n < sto:
            return 0.0
        if sto == 1:
            a *= n
        elif sto == 2:
            a *= (n * (n - 1)) // 2
        elif sto == 3:
            a *= (n * (n - 1) * (n - 2)) // 6
        else:
            # Fallback for unexpected stoichiometry
            out = 1
            for t in range(sto):
                out = out * (n - t) // (t + 1)
            a *= out
        if a == 0.0:
            return 0.0
    return float(a)


# -------------------------
# Numeric safety
# -------------------------
def _draw_poisson_bounded(
    rng: np.random.Generator,
    lam: np.ndarray,
    lam_cap: float,
) -> np.ndarray:
    """
    Guard Poisson rates from inf/NaN/overflow in aggressive tau-leaps.
    """
    lam = np.nan_to_num(
        lam,
        nan=0.0,
        posinf=lam_cap if lam_cap > 0 else 1e6,
        neginf=0.0,
        copy=False,
    )
    if lam_cap > 0:
        np.clip(lam, 0.0, lam_cap, out=lam)
    else:
        np.maximum(lam, 0.0, out=lam)
    return rng.poisson(lam)


# -------------------------
# Tau-leaping simulation
# -------------------------
def simulate_one_tau(
    x0: np.ndarray,
    rxns: List[Rxn],
    nu: np.ndarray,
    idx: Dict[str, int],
    moi: int,
    T: float,
    dt: float,
    rng: np.random.Generator,
    stealth_thr: int = 145,
    hijack_thr: int = 55,
    lam_cap: float = 1e6,
) -> int:
    """
    Returns:
      1 = stealth
      2 = hijack
    Always returns 1 or 2; if no threshold hit by T, final-state rule decides.
    """
    x = x0.copy()
    x[idx["MOI"]] = moi

    cI2_i = idx["cI2"]
    cro2_i = idx["Cro2"]
    n_rxn = len(rxns)
    n_steps = int(math.ceil(T / dt))

    for _ in range(n_steps):
        if x[cI2_i] > stealth_thr:
            return 1
        if x[cro2_i] > hijack_thr:
            return 2

        a = np.fromiter((propensity(x, r) for r in rxns), dtype=np.float64, count=n_rxn)
        K = _draw_poisson_bounded(rng, a * dt, lam_cap=lam_cap)
        if np.any(K):
            x += K @ nu
            # tau-leaping can overshoot reactants; keep state valid
            np.maximum(x, 0, out=x)

    # Final-state tie-break when horizon ends before threshold hit
    s_score = float(x[cI2_i]) / float(stealth_thr)
    h_score = float(x[cro2_i]) / float(hijack_thr)
    return 2 if h_score >= s_score else 1


# -------------------------
# Multiprocessing worker state
# -------------------------
G_X0 = None
G_RXNS = None
G_NU = None
G_IDX = None
G_T = 500.0
G_DT = 0.2
G_STEALTH_THR = 145
G_HIJACK_THR = 55
G_LAM_CAP = 1e6


def _set_worker_globals(
    x0: np.ndarray,
    rxns: List[Rxn],
    nu: np.ndarray,
    idx: Dict[str, int],
    T: float,
    dt: float,
    stealth_thr: int,
    hijack_thr: int,
    lam_cap: float,
) -> None:
    global G_X0, G_RXNS, G_NU, G_IDX, G_T, G_DT, G_STEALTH_THR, G_HIJACK_THR, G_LAM_CAP
    G_X0 = x0
    G_RXNS = rxns
    G_NU = nu
    G_IDX = idx
    G_T = T
    G_DT = dt
    G_STEALTH_THR = stealth_thr
    G_HIJACK_THR = hijack_thr
    G_LAM_CAP = lam_cap


def _init_worker(
    x0: np.ndarray,
    rxns: List[Rxn],
    nu: np.ndarray,
    idx: Dict[str, int],
    T: float,
    dt: float,
    stealth_thr: int,
    hijack_thr: int,
    lam_cap: float,
) -> None:
    _set_worker_globals(x0, rxns, nu, idx, T, dt, stealth_thr, hijack_thr, lam_cap)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _simulate_job(job: Tuple[int, int]) -> int:
    moi, seed = job
    rng = np.random.default_rng(seed)
    return simulate_one_tau(
        G_X0,
        G_RXNS,
        G_NU,
        G_IDX,
        moi,
        G_T,
        G_DT,
        rng,
        stealth_thr=G_STEALTH_THR,
        hijack_thr=G_HIJACK_THR,
        lam_cap=G_LAM_CAP,
    )


def _run_batch_serial(
    x0: np.ndarray,
    rxns: List[Rxn],
    nu: np.ndarray,
    idx: Dict[str, int],
    moi: int,
    n: int,
    base_seed: int,
    T: float,
    dt: float,
    stealth_thr: int,
    hijack_thr: int,
    lam_cap: float,
) -> Tuple[int, int]:
    stealth = hijack = 0
    for t in range(n):
        seed = (base_seed + 100_003 * moi + t) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        out = simulate_one_tau(
            x0,
            rxns,
            nu,
            idx,
            moi,
            T,
            dt,
            rng,
            stealth_thr=stealth_thr,
            hijack_thr=hijack_thr,
            lam_cap=lam_cap,
        )
        if out == 1:
            stealth += 1
        else:
            hijack += 1
    return stealth, hijack


def _run_batch_parallel(
    pool: mp.Pool,
    moi: int,
    n: int,
    base_seed: int,
    chunksize: int,
) -> Tuple[int, int]:
    stealth = hijack = 0
    jobs = ((moi, (base_seed + 100_003 * moi + t) & 0xFFFFFFFF) for t in range(n))
    for out in pool.imap_unordered(_simulate_job, jobs, chunksize=max(1, chunksize)):
        if out == 1:
            stealth += 1
        else:
            hijack += 1
    return stealth, hijack


def _estimate_moi(
    moi: int,
    trials: int,
    max_trials: int,
    target_hw: float,
    min_trials: int,
    batch_n: int,
    seed: int,
    x0: np.ndarray,
    rxns: List[Rxn],
    nu: np.ndarray,
    idx: Dict[str, int],
    T: float,
    dt: float,
    stealth_thr: int,
    hijack_thr: int,
    lam_cap: float,
    pool: mp.Pool,
    chunksize: int,
) -> dict:
    stop_by_hw = target_hw > 0.0
    cap_trials = max_trials if stop_by_hw else trials

    stealth = hijack = 0
    used = 0
    batch_id = 0

    while used < cap_trials:
        n = min(batch_n, cap_trials - used)
        t0 = time.time()
        if pool is None:
            st, hj = _run_batch_serial(
                x0,
                rxns,
                nu,
                idx,
                moi,
                n,
                seed + 777 * used,
                T,
                dt,
                stealth_thr,
                hijack_thr,
                lam_cap,
            )
        else:
            st, hj = _run_batch_parallel(
                pool, moi, n, seed + 777 * used, chunksize=chunksize
            )
        stealth += st
        hijack += hj
        used += n
        batch_id += 1

        pS = stealth / used
        pH = hijack / used
        seS = math.sqrt(max(pS * (1 - pS), 0.0) / used)
        seH = math.sqrt(max(pH * (1 - pH), 0.0) / used)
        hwS = 1.96 * seS
        hwH = 1.96 * seH
        print(
            f"MOI={moi:2d} batch={batch_id:3d} trials={used:5d}  "
            f"P(H)={pH:.3f}±{hwH:.3f}  P(S)={pS:.3f}±{hwS:.3f}  dt={time.time()-t0:5.1f}s",
            flush=True,
        )

        if stop_by_hw and used >= min_trials and hwH <= target_hw and hwS <= target_hw:
            break

    return {
        "MOI": moi,
        "trials": used,
        "pHijack": hijack / used,
        "seHijack": math.sqrt(max((hijack / used) * (1 - hijack / used), 0.0) / used),
        "pStealth": stealth / used,
        "seStealth": math.sqrt(max((stealth / used) * (1 - stealth / used), 0.0) / used),
    }


# -------------------------
# Trajectory plotting (sample paths)
# -------------------------
def sample_trajectories(
    x0: np.ndarray,
    rxns: List[Rxn],
    nu: np.ndarray,
    idx: Dict[str, int],
    moi: int,
    T: float,
    dt: float,
    n_traj: int,
    seed: int = 0,
    lam_cap: float = 1e6,
):
    rng = np.random.default_rng(seed)
    cI2_i = idx["cI2"]
    cro2_i = idx["Cro2"]
    n_rxn = len(rxns)
    n_steps = int(math.ceil(T / dt))
    paths = []
    for _ in range(n_traj):
        x = x0.copy()
        x[idx["MOI"]] = moi
        cro2_s = [int(x[cro2_i])]
        cI2_s = [int(x[cI2_i])]
        for _ in range(n_steps):
            a = np.fromiter((propensity(x, r) for r in rxns), dtype=np.float64, count=n_rxn)
            K = _draw_poisson_bounded(rng, a * dt, lam_cap=lam_cap)
            if np.any(K):
                x += K @ nu
                np.maximum(x, 0, out=x)
            cro2_s.append(int(x[cro2_i]))
            cI2_s.append(int(x[cI2_i]))
        paths.append((cro2_s, cI2_s))
    return paths


def _mp_context():
    if os.name == "posix":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    return mp.get_context()


def _parse_args():
    ap = argparse.ArgumentParser(description="Tau-leaping estimator for lambda phage mode probabilities.")
    ap.add_argument("--lambda-r", default="lambda.r")
    ap.add_argument("--lambda-in", default="lambda.in")
    ap.add_argument("--outdir", default="out_tau")
    ap.add_argument("--moi-min", type=int, default=1)
    ap.add_argument("--moi-max", type=int, default=10)
    ap.add_argument("--trials", type=int, default=1000, help="Fixed trials per MOI when --target-hw <= 0.")
    ap.add_argument("--target-hw", type=float, default=0.0, help="95%% CI half-width target; 0 disables early stop.")
    ap.add_argument("--min-trials", type=int, default=300, help="Minimum trials before CI early-stop check.")
    ap.add_argument("--max-trials", type=int, default=3000, help="Max trials per MOI when using --target-hw.")
    ap.add_argument("--batch-n", type=int, default=200, help="Batch size per MOI update.")
    ap.add_argument("--T", type=float, default=500.0, help="Time horizon.")
    ap.add_argument("--dt", type=float, default=0.2, help="Tau-leap step.")
    ap.add_argument("--lam-cap", type=float, default=1e6, help="Cap for Poisson lambda; 0 keeps only finite/nonnegative.")
    ap.add_argument("--stealth-thr", type=int, default=145)
    ap.add_argument("--hijack-thr", type=int, default=55)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--nproc", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--chunksize", type=int, default=64)
    ap.add_argument("--traj-moi", type=int, default=5)
    ap.add_argument("--traj-n", type=int, default=10)
    ap.add_argument("--skip-plots", action="store_true")
    return ap.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = _parse_args()
    if args.moi_max < args.moi_min:
        raise ValueError("--moi-max must be >= --moi-min")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.target_hw < 0:
        raise ValueError("--target-hw must be >= 0")
    if args.max_trials <= 0 or args.min_trials <= 0:
        raise ValueError("--max-trials and --min-trials must be positive")
    if args.batch_n <= 0:
        raise ValueError("--batch-n must be positive")
    if args.dt <= 0 or args.T <= 0:
        raise ValueError("--dt and --T must be positive")
    if args.lam_cap < 0:
        raise ValueError("--lam-cap must be >= 0")
    if args.nproc <= 0:
        raise ValueError("--nproc must be positive")

    os.makedirs(args.outdir, exist_ok=True)

    species, idx, x0, rxns, nu = build_model(args.lambda_r, args.lambda_in)
    print(f"Loaded model: {len(rxns)} reactions, {len(species)} species.")
    if args.target_hw > 0:
        print(
            f"Running tau-leaping with CI early-stop: target_hw={args.target_hw}, "
            f"min_trials={args.min_trials}, max_trials={args.max_trials}, "
            f"batch_n={args.batch_n}, T={args.T}, dt={args.dt}, nproc={args.nproc}\n"
        )
    else:
        print(
            f"Running tau-leaping: trials={args.trials}, batch_n={args.batch_n}, "
            f"T={args.T}, dt={args.dt}, nproc={args.nproc}\n"
        )

    rows: List[dict] = []
    pool = None
    interrupted = False
    try:
        if args.nproc > 1:
            ctx = _mp_context()
            pool = ctx.Pool(
                processes=args.nproc,
                initializer=_init_worker,
                initargs=(
                    x0,
                    rxns,
                    nu,
                    idx,
                    args.T,
                    args.dt,
                    args.stealth_thr,
                    args.hijack_thr,
                    args.lam_cap,
                ),
            )

        for moi in range(args.moi_min, args.moi_max + 1):
            r = _estimate_moi(
                moi=moi,
                trials=args.trials,
                max_trials=args.max_trials,
                target_hw=args.target_hw,
                min_trials=args.min_trials,
                batch_n=args.batch_n,
                seed=args.seed,
                x0=x0,
                rxns=rxns,
                nu=nu,
                idx=idx,
                T=args.T,
                dt=args.dt,
                stealth_thr=args.stealth_thr,
                hijack_thr=args.hijack_thr,
                lam_cap=args.lam_cap,
                pool=pool,
                chunksize=args.chunksize,
            )
            rows.append(r)
            print(
                f"MOI={moi:2d} done  trials={r['trials']:5d}  "
                f"P(H)={r['pHijack']:.3f}  P(S)={r['pStealth']:.3f}",
                flush=True,
            )

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Stopping remaining simulations.")
    finally:
        if pool is not None:
            if interrupted:
                pool.terminate()
            else:
                pool.close()
            pool.join()
    if interrupted:
        return

    # Write CSV
    csv_path = os.path.join(args.outdir, "lambda_tau_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["MOI", "trials", "pHijack", "seHijack", "pStealth", "seStealth"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot probabilities with error bars
    if not args.skip_plots:
        plt = _get_pyplot()
        xs = [r["MOI"] for r in rows]
        pH = [r["pHijack"] for r in rows]
        eH = [1.96 * r["seHijack"] for r in rows]
        pS = [r["pStealth"] for r in rows]
        eS = [1.96 * r["seStealth"] for r in rows]

        plt.figure()
        plt.errorbar(xs, pH, yerr=eH, marker="o", capsize=4, linewidth=2, label="Hijack")
        plt.errorbar(xs, pS, yerr=eS, marker="s", capsize=4, linewidth=2, label="Stealth")
        plt.xlabel("MOI")
        plt.ylabel("Estimated probability")
        plt.xticks(xs)
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "prob_vs_moi_tau.png"), dpi=200)
        plt.close()

        # Trajectory phase plot
        paths = sample_trajectories(
            x0,
            rxns,
            nu,
            idx,
            args.traj_moi,
            args.T,
            args.dt,
            args.traj_n,
            seed=args.seed + 999,
            lam_cap=args.lam_cap,
        )
        plt.figure()
        for cro2_s, cI2_s in paths:
            plt.plot(cro2_s, cI2_s, linewidth=1, alpha=0.85)
        plt.axvline(args.hijack_thr, linestyle="--", linewidth=2)
        plt.axhline(args.stealth_thr, linestyle="--", linewidth=2)
        plt.xlabel("Cro2")
        plt.ylabel("cI2")
        plt.title(f"Sample trajectories in (Cro2, cI2), MOI={args.traj_moi} (tau-leaping)")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"traj_phase_moi{args.traj_moi}_tau.png"), dpi=200)
        plt.close()

    # LaTeX table
    tex_path = os.path.join(args.outdir, "table_tau.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[t]" "\n")
        f.write(r"\centering" "\n")
        f.write(
            r"\caption{Estimated hijack vs.\ stealth probabilities vs.\ MOI (tau-leaping, fixed horizon). "
            r"Error bars use $\pm1.96\,\mathrm{SE}$.}" "\n"
        )
        f.write(r"\label{tab:lambda-tau}" "\n")
        f.write(r"\begin{tabular}{rccccc}" "\n")
        f.write(r"\hline" "\n")
        f.write(r"MOI & trials & $\hat p_H$ & SE$(\hat p_H)$ & $\hat p_S$ & SE$(\hat p_S)$ \\" "\n")
        f.write(r"\hline" "\n")
        for r in rows:
            f.write(
                f"{r['MOI']} & {r['trials']} & {r['pHijack']:.4f} & {r['seHijack']:.4f} & "
                f"{r['pStealth']:.4f} & {r['seStealth']:.4f} \\\\\n"
            )
        f.write(r"\hline" "\n")
        f.write(r"\end{tabular}" "\n")
        f.write(r"\end{table}" "\n")

    print(f"\nWrote outputs to ./{args.outdir}/")
    print("  - lambda_tau_results.csv")
    if not args.skip_plots:
        print("  - prob_vs_moi_tau.png")
        print(f"  - traj_phase_moi{args.traj_moi}_tau.png")
    print("  - table_tau.tex")


if __name__ == "__main__":
    mp.freeze_support()
    main()
