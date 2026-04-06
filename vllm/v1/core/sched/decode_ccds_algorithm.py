"""
Decode-side scheduling reference algorithm (CCDS style).

This file is intentionally self-contained so it can be used as a design spec
before integrating into vLLM runtime code.

Key ideas:
1) Update each request's output-length distribution online (posterior update).
2) Convert posterior to remaining-length moments (mean/std).
3) Build a risk-aware pressure score per request.
4) Use dynamic batch target control around the observed goodput knee.
5) Admit waiting requests in risk tiers; preempt only when needed.

This is pseudocode-like Python:
- Executable for toy examples.
- Not wired into vLLM internal classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class Mode(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class BucketSpec:
    # Example: edges=[0,100,200,300,400,8192]
    edges: List[float]

    def __post_init__(self) -> None:
        if len(self.edges) < 2:
            raise ValueError("Bucket edges must contain at least two values")
        if any(self.edges[i] >= self.edges[i + 1] for i in range(len(self.edges) - 1)):
            raise ValueError("Bucket edges must be strictly increasing")

    @property
    def num_buckets(self) -> int:
        return len(self.edges) - 1

    def center_of_interval(self, lo: float, hi: float) -> float:
        return 0.5 * (lo + hi)


@dataclass
class RequestState:
    req_id: str
    prompt_len: int
    # Initial output-length bucket probabilities.
    # One probability for each [edges[i], edges[i+1]) interval.
    probs: List[float]

    generated_tokens: int = 0
    preempt_count: int = 0
    step_token_demand: int = 1
    last_pressure: float = 0.0

    # Online-updated posterior over total output length L.
    posterior_probs: List[float] = field(default_factory=list)

    def ensure_posterior(self) -> None:
        if not self.posterior_probs:
            self.posterior_probs = list(self.probs)


@dataclass
class RuntimeSignals:
    goodput_short: float
    goodput_prev: float
    preempt_rate_short: float
    staging_watermark: float
    oom_alert: bool = False


@dataclass
class SchedulerState:
    batch_target: int
    batch_min: int
    batch_max: int

    # Risk budget target: allowed overload probability.
    epsilon_overload: float = 0.1

    # Lightweight history buffers for quantile-like adaptive mode switching.
    preempt_hist: List[float] = field(default_factory=list)
    water_hist: List[float] = field(default_factory=list)
    goodput_hist: List[float] = field(default_factory=list)

    # Optional accounting for analysis.
    wasted_decode_tokens: int = 0
    scheduler_overhead_units: int = 0

    def append_history(self, sig: RuntimeSignals, max_len: int = 128) -> None:
        self.preempt_hist.append(sig.preempt_rate_short)
        self.water_hist.append(sig.staging_watermark)
        self.goodput_hist.append(sig.goodput_short)
        if len(self.preempt_hist) > max_len:
            self.preempt_hist.pop(0)
        if len(self.water_hist) > max_len:
            self.water_hist.pop(0)
        if len(self.goodput_hist) > max_len:
            self.goodput_hist.pop(0)


@dataclass
class ScheduleDecision:
    mode: Mode
    admitted: List[str]
    preempted: List[str]
    running_after: List[str]
    waiting_after: List[str]
    batch_target_after: int
    note: str


def _normalize(vals: Sequence[float]) -> List[float]:
    s = sum(vals)
    if s <= 0:
        n = len(vals)
        return [1.0 / n] * n
    return [v / s for v in vals]


def update_posterior_probs(
    req: RequestState,
    buckets: BucketSpec,
) -> List[float]:
    """
    Condition the total-length distribution on "L > generated_tokens".

    Example with edges [0,100,200,300,400,8192]:
    - If generated_tokens == 100, bucket[0] mass becomes 0.
    - Remaining buckets are renormalized.
    """
    req.ensure_posterior()
    g = float(req.generated_tokens)
    kept: List[float] = []

    for i, p in enumerate(req.posterior_probs):
        lo, hi = buckets.edges[i], buckets.edges[i + 1]
        width = hi - lo
        if width <= 0:
            kept.append(0.0)
            continue

        if g >= hi:
            # This bucket is fully impossible after generating g tokens.
            kept.append(0.0)
            continue

        if g <= lo:
            # Entire bucket remains valid.
            kept.append(p)
            continue

        # Partial survival in current bucket with a uniform-within-bucket
        # assumption. Only [g, hi) remains.
        remain_ratio = (hi - g) / width
        kept.append(p * max(0.0, min(1.0, remain_ratio)))

    req.posterior_probs = _normalize(kept)
    return req.posterior_probs


def remaining_moments(
    req: RequestState,
    buckets: BucketSpec,
) -> Tuple[float, float]:
    """
    Compute mean/std of remaining output tokens R = L - generated_tokens,
    from posterior bucket probabilities.
    """
    probs = update_posterior_probs(req, buckets)
    g = float(req.generated_tokens)

    rem_centers: List[float] = []
    for i, p in enumerate(probs):
        lo, hi = buckets.edges[i], buckets.edges[i + 1]
        # Truncated interval after conditioning on L > g.
        lo_eff = max(lo, g)
        center_full = buckets.center_of_interval(lo_eff, hi)
        rem_center = max(1.0, center_full - g)
        rem_centers.append(rem_center)

    mu = sum(p * c for p, c in zip(probs, rem_centers))
    var = sum(p * ((c - mu) ** 2) for p, c in zip(probs, rem_centers))
    std = sqrt(max(0.0, var))
    return mu, std


def risk_beta(epsilon_overload: float) -> float:
    # Cantelli-style scale; epsilon in (0,1).
    e = min(max(epsilon_overload, 1e-6), 1.0 - 1e-6)
    return sqrt((1.0 - e) / e)


def classify_requests(
    waiting: Sequence[RequestState],
    pressure_map: Dict[str, float],
    std_map: Dict[str, float],
) -> Dict[str, List[RequestState]]:
    """
    Split waiting requests into four groups:
    - short_certain, short_uncertain, long_certain, long_uncertain

    Thresholds are data-driven (medians), reducing manual tuning.
    """
    if not waiting:
        return {
            "short_certain": [],
            "short_uncertain": [],
            "long_certain": [],
            "long_uncertain": [],
        }

    pressure_med = median(pressure_map[r.req_id] for r in waiting)
    std_med = median(std_map[r.req_id] for r in waiting)

    groups = {
        "short_certain": [],
        "short_uncertain": [],
        "long_certain": [],
        "long_uncertain": [],
    }

    for r in waiting:
        is_short = pressure_map[r.req_id] <= pressure_med
        is_certain = std_map[r.req_id] <= std_med
        if is_short and is_certain:
            groups["short_certain"].append(r)
        elif is_short and not is_certain:
            groups["short_uncertain"].append(r)
        elif (not is_short) and is_certain:
            groups["long_certain"].append(r)
        else:
            groups["long_uncertain"].append(r)

    for key in groups:
        groups[key].sort(key=lambda x: pressure_map[x.req_id])
    return groups


def choose_mode(state: SchedulerState, sig: RuntimeSignals) -> Mode:
    """
    Adaptive mode without many fixed thresholds.
    Uses recent medians as baselines plus a hard OOM alert override.
    """
    if sig.oom_alert:
        return Mode.RED

    pre_med = median(state.preempt_hist) if state.preempt_hist else sig.preempt_rate_short
    wat_med = median(state.water_hist) if state.water_hist else sig.staging_watermark
    gp_med = median(state.goodput_hist) if state.goodput_hist else sig.goodput_short

    goodput_down = sig.goodput_short < min(gp_med, sig.goodput_prev)
    pre_high = sig.preempt_rate_short > pre_med
    wat_high = sig.staging_watermark > wat_med

    if (pre_high and wat_high) or (goodput_down and pre_high):
        return Mode.RED
    if pre_high or wat_high or goodput_down:
        return Mode.YELLOW
    return Mode.GREEN


def adjust_batch_target(state: SchedulerState, mode: Mode) -> None:
    if mode == Mode.GREEN:
        state.batch_target = min(state.batch_target + 1, state.batch_max)
    elif mode == Mode.RED:
        state.batch_target = max(state.batch_target - 1, state.batch_min)


def _admission_order(mode: Mode, groups: Dict[str, List[RequestState]]) -> List[RequestState]:
    if mode == Mode.GREEN:
        order = [
            "short_certain",
            "short_uncertain",
            "long_certain",
            "long_uncertain",
        ]
    elif mode == Mode.YELLOW:
        order = ["short_certain", "short_uncertain", "long_certain"]
    else:
        order = ["short_certain", "short_uncertain"]

    out: List[RequestState] = []
    for key in order:
        out.extend(groups[key])
    return out


def pick_preemption_victim(
    running: Sequence[RequestState],
    pressure_map: Dict[str, float],
) -> Optional[RequestState]:
    """
    Preempt only if required.
    Victim preference:
    1) high pressure
    2) low completion ratio
    """
    if not running:
        return None

    def completion_ratio(r: RequestState) -> float:
        denom = max(1.0, r.prompt_len + r.generated_tokens)
        return r.generated_tokens / denom

    # Higher key => more likely to preempt.
    return max(
        running,
        key=lambda r: (
            pressure_map.get(r.req_id, 0.0),
            -completion_ratio(r),
        ),
    )


def schedule_step(
    running: List[RequestState],
    waiting: List[RequestState],
    state: SchedulerState,
    buckets: BucketSpec,
    sig: RuntimeSignals,
    token_budget: int,
) -> ScheduleDecision:
    """
    One decode scheduling step.

    token_budget models the fact that per-step token demand is not constant.
    For normal decode demand is often 1 token/request, but this hook supports
    different demand values (e.g., speculative decode path).
    """
    state.scheduler_overhead_units += 1
    state.append_history(sig)

    mode = choose_mode(state, sig)
    adjust_batch_target(state, mode)

    beta = risk_beta(state.epsilon_overload)

    pressure_map: Dict[str, float] = {}
    std_map: Dict[str, float] = {}

    for req in list(running) + list(waiting):
        mu_rem, std_rem = remaining_moments(req, buckets)
        risk_len = mu_rem + beta * std_rem
        pressure = (req.prompt_len + risk_len) * risk_len
        req.last_pressure = pressure
        pressure_map[req.req_id] = pressure
        std_map[req.req_id] = std_rem

    groups = classify_requests(waiting, pressure_map, std_map)
    ordered_waiting = _admission_order(mode, groups)

    admitted: List[str] = []
    used_tokens = 0

    # Keep existing running requests, then admit up to batch_target.
    target_slots = max(0, state.batch_target - len(running))

    for req in ordered_waiting:
        if target_slots <= 0:
            break

        demand = max(1, req.step_token_demand)
        if used_tokens + demand > token_budget:
            break

        waiting.remove(req)
        running.append(req)
        admitted.append(req.req_id)
        used_tokens += demand
        target_slots -= 1

    preempted: List[str] = []

    # Optional emergency preemption path in RED mode when budget is exceeded.
    while used_tokens > token_budget and running:
        victim = pick_preemption_victim(running, pressure_map)
        if victim is None:
            break
        running.remove(victim)
        waiting.append(victim)
        victim.preempt_count += 1
        preempted.append(victim.req_id)
        # Approximate wasted decode-token accounting.
        state.wasted_decode_tokens += max(1, victim.generated_tokens)
        used_tokens = max(0, used_tokens - max(1, victim.step_token_demand))

    note = (
        "Decision made with posterior-updated risk pressure. "
        "Admit by mode-tiered order; preempt only if required."
    )

    return ScheduleDecision(
        mode=mode,
        admitted=admitted,
        preempted=preempted,
        running_after=[r.req_id for r in running],
        waiting_after=[r.req_id for r in waiting],
        batch_target_after=state.batch_target,
        note=note,
    )


def _demo_case_1() -> None:
    print("\\n=== Demo 1: posterior update at boundary g=100 ===")
    buckets = BucketSpec(edges=[0, 100, 200, 300, 400, 8192])
    r = RequestState(req_id="r1", prompt_len=600, probs=[0.20, 0.25, 0.25, 0.20, 0.10])
    r.generated_tokens = 100
    post = update_posterior_probs(r, buckets)
    mu, std = remaining_moments(r, buckets)
    print("posterior:", [round(x, 4) for x in post])
    print("remaining mean/std:", round(mu, 2), round(std, 2))


def _demo_case_2() -> None:
    print("\\n=== Demo 2: one scheduling step ===")
    buckets = BucketSpec(edges=[0, 100, 200, 300, 400, 8192])

    running = [
        RequestState("run_a", prompt_len=1024, probs=[0.1, 0.2, 0.3, 0.25, 0.15], generated_tokens=80),
        RequestState("run_b", prompt_len=768, probs=[0.2, 0.2, 0.2, 0.2, 0.2], generated_tokens=40),
    ]
    waiting = [
        RequestState("w1", prompt_len=256, probs=[0.5, 0.2, 0.15, 0.1, 0.05]),
        RequestState("w2", prompt_len=2048, probs=[0.05, 0.1, 0.2, 0.25, 0.4]),
        RequestState("w3", prompt_len=512, probs=[0.3, 0.3, 0.2, 0.1, 0.1]),
    ]

    state = SchedulerState(batch_target=4, batch_min=2, batch_max=8, epsilon_overload=0.1)
    sig = RuntimeSignals(
        goodput_short=1200,
        goodput_prev=1230,
        preempt_rate_short=0.18,
        staging_watermark=0.72,
        oom_alert=False,
    )

    decision = schedule_step(
        running=running,
        waiting=waiting,
        state=state,
        buckets=buckets,
        sig=sig,
        token_budget=3,
    )

    print("mode:", decision.mode)
    print("admitted:", decision.admitted)
    print("preempted:", decision.preempted)
    print("running_after:", decision.running_after)
    print("waiting_after:", decision.waiting_after)
    print("batch_target_after:", decision.batch_target_after)
    print("note:", decision.note)


if __name__ == "__main__":
    _demo_case_1()
    _demo_case_2()
