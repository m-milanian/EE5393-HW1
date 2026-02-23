import math
import random
import multiprocessing
from collections import defaultdict
from typing import Tuple, Dict, List

State = Tuple[int, int, int]

# ----------------------------
# Reaction model (discrete)
# ----------------------------

def propensities(s: State) -> Tuple[float, float, float]:
    """
    Returns (a1, a2, a3) for state s = (x1, x2, x3), using the propensities
    consistent with the prompt:
      a1 = (1/2) x1 (x1-1) x2
      a2 = x1 x3 (x3-1)
      a3 = 3 x2 x3
    """
    x1, x2, x3 = s
    a1 = 0.5 * x1 * (x1 - 1) * x2 if (x1 >= 2 and x2 >= 1) else 0.0
    a2 = x1 * x3 * (x3 - 1) if (x1 >= 1 and x3 >= 2) else 0.0
    a3 = 3.0 * x2 * x3 if (x2 >= 1 and x3 >= 1) else 0.0
    return a1, a2, a3


def step(s: State, rng: random.Random) -> State:
    """
    One jump of the embedded discrete-time chain:
    pick reaction i with probability a_i / (a1+a2+a3), then update state.

    Reactions:
      R1: 2X1 + X2 -> 4X3   : (x1,x2,x3) -> (x1-2, x2-1, x3+4)
      R2: X1 + 2X3 -> 3X2   : (x1,x2,x3) -> (x1-1, x2+3, x3-2)
      R3: X2 + X3 -> 2X1    : (x1,x2,x3) -> (x1+2, x2-1, x3-1)
    """
    a1, a2, a3 = propensities(s)
    a0 = a1 + a2 + a3
    if a0 <= 0.0:
        # No reaction can fire; return unchanged (absorbing dead state).
        return s

    u = rng.random() * a0
    x1, x2, x3 = s

    if u < a1:
        return (x1 - 2, x2 - 1, x3 + 4)
    elif u < a1 + a2:
        return (x1 - 1, x2 + 3, x3 - 2)
    else:
        return (x1 + 2, x2 - 1, x3 - 1)


# ----------------------------
# Problem 1(a): outcomes
# ----------------------------

def outcome_label(s: State) -> int:
    """
    Outcomes:
      C1: x1 >= 150  -> label 1
      C2: x2 < 10    -> label 2
      C3: x3 > 100   -> label 3

    Returns 0 if none of the outcomes holds.
    If multiple hold simultaneously, we return the smallest label by priority.
    (You can change this rule if your assignment defines a different convention.)
    """
    x1, x2, x3 = s
    if x1 >= 150:
        return 1
    if x2 < 10:
        return 2
    if x3 > 100:
        return 3
    return 0


def simulate_first_hit(start: State, rng: random.Random, max_steps: int = 10_000) -> Tuple[int, int]:
    """
    Simulate until first hit of C1/C2/C3, returning (label, steps_to_hit).
    Stops early if no reactions can fire.
    """
    s = start
    for t in range(max_steps + 1):
        lab = outcome_label(s)
        if lab != 0:
            return lab, t
        s_next = step(s, rng)
        if s_next == s:  # dead (no reaction)
            return 0, t
        s = s_next
    return 0, max_steps


def _worker_run_chunk(start: State, n_trials: int, seed: int, seed_offset: int, max_steps: int):
    rng = random.Random(seed + seed_offset)
    chunk_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    chunk_hits: List[int] = []
    for _ in range(n_trials):
        lab, t = simulate_first_hit(start, rng, max_steps=max_steps)
        chunk_counts[lab] += 1
        if lab != 0:
            chunk_hits.append(t)
    return chunk_counts, chunk_hits


def clopper_pearson_upper_zero_success(N: int, alpha: float = 0.05) -> float:
    """
    One-sided (1-alpha) upper bound when 0 successes in N Bernoulli trials:
      p <= 1 - alpha^(1/N)
    """
    return 1.0 - (alpha ** (1.0 / N))


def estimate_outcome_probabilities(
    start: State = (110, 26, 55),
    trials: int = 50_000,
    seed: int = 0,
    max_steps: int = 10_000,
    use_multiprocessing: bool = False,
) -> None:
    # use module-level worker to allow multiprocessing

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    hit_steps: List[int] = []

    if use_multiprocessing and trials > 0:
        cpu = multiprocessing.cpu_count()
        # split trials roughly evenly
        base = trials // cpu
        extras = trials % cpu
        tasks = []
        for i in range(cpu):
            n = base + (1 if i < extras else 0)
            if n > 0:
                tasks.append((start, n, seed, i + 1, max_steps))

        with multiprocessing.Pool(processes=len(tasks)) as pool:
            results = pool.starmap(_worker_run_chunk, tasks)
        for ccounts, chits in results:
            for k in counts:
                counts[k] += ccounts.get(k, 0)
            hit_steps.extend(chits)
    else:
        # single-process
        chunk_counts, chunk_hits = _worker_run_chunk(start, trials, seed, 1, max_steps)
        for k in counts:
            counts[k] += chunk_counts.get(k, 0)
        hit_steps.extend(chunk_hits)

    p1 = counts[1] / trials
    p2 = counts[2] / trials
    p3 = counts[3] / trials
    p0 = counts[0] / trials

    print("Problem 1(a): First-hit estimates from start =", start)
    print(f"  trials = {trials}, seed = {seed}")
    print(f"  Pr(C1) ~= {p1:.8f}   (count {counts[1]})")
    print(f"  Pr(C2) ~= {p2:.8f}   (count {counts[2]})")
    print(f"  Pr(C3) ~= {p3:.8f}   (count {counts[3]})")
    print(f"  Pr(no hit / dead / max) ~= {p0:.8f}   (count {counts[0]})")

    if counts[1] == 0:
        ub1 = clopper_pearson_upper_zero_success(trials, alpha=0.05)
        print(f"  95% one-sided upper bound for Pr(C1): {ub1:.3e}")
    if counts[2] == 0:
        ub2 = clopper_pearson_upper_zero_success(trials, alpha=0.05)
        print(f"  95% one-sided upper bound for Pr(C2): {ub2:.3e}")

    if hit_steps:
        hit_steps_sorted = sorted(hit_steps)
        n = len(hit_steps_sorted)
        mean_steps = sum(hit_steps_sorted) / n
        median_steps = hit_steps_sorted[n // 2]
        q01 = hit_steps_sorted[int(0.01 * (n - 1))]
        q05 = hit_steps_sorted[int(0.05 * (n - 1))]
        q95 = hit_steps_sorted[int(0.95 * (n - 1))]
        q99 = hit_steps_sorted[int(0.99 * (n - 1))]
        print("  Steps-to-hit summary (conditional on hitting an outcome):")
        print(f"    mean   ~ {mean_steps:.2f}")
        print(f"    median = {median_steps}")
        print(f"    q01,q05,q95,q99 = {q01}, {q05}, {q95}, {q99}")


# ----------------------------
# Problem 1(b): distribution after exactly n steps
# ----------------------------

def transition_probabilities(s: State) -> Tuple[Tuple[State, float], ...]:
    """
    Returns the possible next states and their probabilities from state s,
    i.e., ((s1,p1), (s2,p2), (s3,p3)) but omits reactions with zero propensity.
    """
    a1, a2, a3 = propensities(s)
    a0 = a1 + a2 + a3
    if a0 <= 0.0:
        return tuple()  # no outgoing transitions

    x1, x2, x3 = s
    out = []
    if a1 > 0:
        out.append(((x1 - 2, x2 - 1, x3 + 4), a1 / a0))
    if a2 > 0:
        out.append(((x1 - 1, x2 + 3, x3 - 2), a2 / a0))
    if a3 > 0:
        out.append(((x1 + 2, x2 - 1, x3 - 1), a3 / a0))
    return tuple(out)


def exact_distribution_after_n_steps(start: State, n: int) -> Dict[State, float]:
    """
    Exact distribution over states after exactly n steps via dynamic programming:
      dist_{t+1}(s') = sum_s dist_t(s) P(s->s')
    Reachable states <= 3^n (often less), so n=7 is tiny.
    """
    dist = {start: 1.0}
    for _ in range(n):
        newdist = defaultdict(float)
        for s, ps in dist.items():
            trans = transition_probabilities(s)
            if not trans:
                # dead state persists (absorbing) in our step-counting model
                newdist[s] += ps
                continue
            for s2, p in trans:
                newdist[s2] += ps * p
        dist = dict(newdist)
    return dist


def mean_variance_from_distribution(dist: Dict[State, float]) -> Dict[str, Tuple[float, float]]:
    """
    From distribution over full states, compute mean and variance of X1, X2, X3 separately.
    """
    EX1 = EX2 = EX3 = 0.0
    EX1_2 = EX2_2 = EX3_2 = 0.0

    for (x1, x2, x3), p in dist.items():
        EX1 += p * x1
        EX2 += p * x2
        EX3 += p * x3
        EX1_2 += p * (x1 ** 2)
        EX2_2 += p * (x2 ** 2)
        EX3_2 += p * (x3 ** 2)

    Var1 = EX1_2 - EX1 ** 2
    Var2 = EX2_2 - EX2 ** 2
    Var3 = EX3_2 - EX3 ** 2

    return {
        "X1": (EX1, Var1),
        "X2": (EX2, Var2),
        "X3": (EX3, Var3),
    }


def solve_part_b_exact(start: State = (9, 8, 7), steps: int = 7) -> None:
    dist = exact_distribution_after_n_steps(start, steps)
    stats = mean_variance_from_distribution(dist)

    mass = sum(dist.values())
    print("Problem 1(b): exact distribution after", steps, "steps from start =", start)
    print(f"  #reachable states = {len(dist)}")
    print(f"  total probability mass = {mass:.12f} (should be 1.0)")
    for k in ["X1", "X2", "X3"]:
        mu, var = stats[k]
        print(f"  {k}: mean = {mu:.8f}, variance = {var:.8f}")

    # (Optional) show a marginal PMF for X1 (or X2/X3) if desired:
    # pmf_X1 = defaultdict(float)
    # for (x1,x2,x3), p in dist.items():
    #     pmf_X1[x1] += p
    # for x1 in sorted(pmf_X1):
    #     print(x1, pmf_X1[x1])


def solve_part_b_monte_carlo(start: State = (9, 8, 7), steps: int = 7, trials: int = 200_000, seed: int = 1) -> None:
    rng = random.Random(seed)
    sum1 = sum2 = sum3 = 0.0
    sum1_2 = sum2_2 = sum3_2 = 0.0

    for _ in range(trials):
        s = start
        for _ in range(steps):
            s = step(s, rng)
        x1, x2, x3 = s
        sum1 += x1; sum2 += x2; sum3 += x3
        sum1_2 += x1*x1; sum2_2 += x2*x2; sum3_2 += x3*x3

    m1 = sum1 / trials; m2 = sum2 / trials; m3 = sum3 / trials
    v1 = (sum1_2 / trials) - m1*m1
    v2 = (sum2_2 / trials) - m2*m2
    v3 = (sum3_2 / trials) - m3*m3

    print("Problem 1(b): Monte Carlo check after", steps, "steps from start =", start)
    print(f"  trials = {trials}, seed = {seed}")
    print(f"  X1: mean ~= {m1:.8f}, variance ~= {v1:.8f}")
    print(f"  X2: mean ~= {m2:.8f}, variance ~= {v2:.8f}")
    print(f"  X3: mean ~= {m3:.8f}, variance ~= {v3:.8f}")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    # Part (a)
    estimate_outcome_probabilities(start=(110, 26, 55), trials=50_000, seed=0, max_steps=10000, use_multiprocessing=True)

    print("\n" + "-"*60 + "\n")

    # Part (b) exact (recommended)
    solve_part_b_exact(start=(9, 8, 7), steps=7)

    # Optional Monte Carlo sanity check
    # solve_part_b_monte_carlo(start=(9, 8, 7), steps=7, trials=200_000, seed=1)