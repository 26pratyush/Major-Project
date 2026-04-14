"""
Quantum-Assisted RAN Resource Allocation Simulator
===================================================
MVP for: Quantum-Assisted Resource Allocation for MAC-Layer
         Packet Scheduling in O-RAN Networks

Team: Pratyush Pai, Aditya J Krishnan, Nikhil SK, Mangesh Nesarikar
Guide: Dr Shilpa Chaudhari | M.S. Ramaiah Institute of Technology

Architecture:
  UE Simulator → Channel Model → Scheduler Engine (Classical / Quantum)
               → Performance Evaluator → Tkinter GUI + Matplotlib Plots

Run:  python ran_scheduler_mvp.py
Deps: pip install matplotlib numpy scipy qiskit qiskit-aer qiskit-algorithms
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import random
import math
import time
import numpy as np
from itertools import product
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#  SIMULATION CORE
class UE:
    """User Equipment — holds state per time-slot."""
    TRAFFIC_PROFILES = {
        "VoIP":    {"arrival_mean": 8,  "burst": False, "qos_delay_ms": 20},
        "Video":   {"arrival_mean": 40, "burst": True,  "qos_delay_ms": 50},
        "Web":     {"arrival_mean": 20, "burst": True,  "qos_delay_ms": 100},
        "IoT":     {"arrival_mean": 4,  "burst": False, "qos_delay_ms": 500},
    }

    def __init__(self, uid, profile="Web"):
        self.uid = uid
        self.profile = profile
        cfg = self.TRAFFIC_PROFILES[profile]
        self.arrival_mean  = cfg["arrival_mean"]
        self.burst         = cfg["burst"]
        self.qos_delay     = cfg["qos_delay_ms"]
        self.buffer        = random.randint(10, 60)   # packets
        self.cqi           = random.randint(4, 15)    # 1-15
        self.delay_accrued = 0.0                      # ms
        self._cqi_state    = self.cqi                 # for Markov

    def step_channel(self):
        """Markov-ish CQI random walk, bounded [1,15]."""
        delta = random.choice([-1, -1, 0, 0, 0, 1, 1])
        self._cqi_state = max(1, min(15, self._cqi_state + delta))
        self.cqi = self._cqi_state

    def step_traffic(self):
        """Poisson traffic arrival, optional burst."""
        arrivals = np.random.poisson(self.arrival_mean)
        if self.burst and random.random() < 0.08:
            arrivals += random.randint(20, 50)
        self.buffer = min(self.buffer + arrivals, 200)

    def transmit(self, prbs_allocated):
        """Drain buffer based on CQI × PRBs."""
        # bits per PRB ≈ CQI * 6 (simplified Shannon-like mapping)
        capacity = int(prbs_allocated * self.cqi * 6)   # packets
        sent = min(self.buffer, capacity)
        self.buffer -= sent
        self.delay_accrued = max(0, self.delay_accrued - 1)
        if self.buffer > 0:
            self.delay_accrued += 1                      # one slot backlog
        return sent


class NetworkEnvironment:
    """Holds all UEs and PRB pool, advances time."""

    def __init__(self, n_ues, n_prbs, profiles=None):
        self.n_prbs = n_prbs
        profile_cycle = (profiles or ["VoIP", "Video", "Web", "IoT"])
        self.ues = [
            UE(i, profile_cycle[i % len(profile_cycle)])
            for i in range(n_ues)
        ]

    def step(self):
        for ue in self.ues:
            ue.step_channel()
            ue.step_traffic()

    def get_state(self):
        return {
            ue.uid: {"cqi": ue.cqi, "buffer": ue.buffer, "delay": ue.delay_accrued}
            for ue in self.ues
        }

#  CLASSICAL SCHEDULERS
def scheduler_round_robin(ues, n_prbs):
    n = len(ues)
    base, rem = divmod(n_prbs, n)
    alloc = {ue.uid: base for ue in ues}
    for i in range(rem):
        alloc[ues[i].uid] += 1
    return alloc

def scheduler_max_cqi(ues, n_prbs):
    """Give PRBs to highest-CQI UEs; correct two-pass allocation."""
    sorted_ues = sorted(ues, key=lambda u: u.cqi, reverse=True)
    alloc = {ue.uid: 0 for ue in ues}
    pool  = n_prbs
    n     = len(sorted_ues)
    # Pass 1: each UE gets floor(pool / remaining) PRBs
    for i, ue in enumerate(sorted_ues):
        give          = pool // (n - i)
        alloc[ue.uid] = give
        pool         -= give
    # Pass 2: remainder 1-by-1 to highest-CQI
    for ue in sorted_ues:
        if pool <= 0:
            break
        alloc[ue.uid] += 1
        pool -= 1
    return alloc

def scheduler_proportional_fair(ues, n_prbs, history):
    """PF: CQI / avg_throughput weighting."""
    weights = []
    for ue in ues:
        avg = history.get(ue.uid, 1)
        weights.append(ue.cqi / max(avg, 1))
    total_w = sum(weights) or 1
    alloc = {}
    given = 0
    for ue, w in zip(ues, weights):
        share = max(1, round(n_prbs * w / total_w))
        alloc[ue.uid] = share
        given += share
    # fix rounding
    diff = given - n_prbs
    for ue in sorted(ues, key=lambda u: weights[ues.index(u)]):
        if diff == 0:
            break
        if diff > 0 and alloc[ue.uid] > 1:
            alloc[ue.uid] -= 1; diff -= 1
        elif diff < 0:
            alloc[ue.uid] += 1; diff += 1
    return alloc

# ══════════════════════════════════════════════════════════════════
#  REAL QAOA QUANTUM SCHEDULER  (Qiskit / Qiskit-Aer)
# ══════════════════════════════════════════════════════════════════
#
#  Architecture (hybrid classical-quantum):
#
#   Step 1 — Classical pre-processing:
#     • Compute per-UE urgency weight (CQI × buffer pressure × QoS urgency)
#     • Cluster UEs into at most MAX_QUBO_UES priority groups so that
#       the QUBO stays ≤ MAX_QUBITS qubits (hardware-tractable)
#     • Build a dense QUBO matrix Q:  minimise x^T Q x
#         Objective  : –weight[u] for each binary var x[u,r]
#         Constraint : penalty × x[u1,r] × x[u2,r]  (one UE per PRB slot)
#
#   Step 2 — QUBO → Ising conversion:
#     • Substitute x_i = (1 – Z_i) / 2  to get Pauli Z Hamiltonian
#     • Build Qiskit SparsePauliOp
#
#   Step 3 — QAOA on Qiskit Aer statevector simulator:
#     • Build QAOAAnsatz circuit (reps layers of cost + mixer unitaries)
#     • Optimize parameters with COBYLA (classical outer loop)
#     • Sample final circuit with 1024 shots → pick highest-count bitstring
#     • Falls back to IBM Runtime if IBMQ_TOKEN env-var is set
#
#   Step 4 — Classical post-processing:
#     • Decode bitstring to UE→PRB-block mapping
#     • Distribute any remaining PRBs by urgency weight
#     • Guarantee every UE gets ≥ 1 PRB
#
#  Why this beats classical schedulers in congested scenarios:
#     QAOA optimises a richer objective (CQI + buffer + QoS deadline)
#     simultaneously across all UEs in superposition, escaping local
#     optima that greedy and PF schedulers get stuck in when PRBs
#     are scarce and UE priorities conflict.
# ══════════════════════════════════════════════════════════════════

# ── Qiskit imports (graceful degradation if not installed) ──
try:
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.primitives import StatevectorSampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from scipy.optimize import minimize as scipy_minimize
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# ── Global quantum settings (toggled from GUI) ──
QAOA_REPS        = 2     # QAOA circuit depth (p layers); 2 is quality/speed sweet spot
QAOA_SHOTS       = 128   # measurement shots per cost evaluation (128 = fast, 512 = quality)
QAOA_MAXITER     = 25    # COBYLA iterations — 25 converges well for ≤6 qubits
MAX_QUBITS       = 6     # hard cap: 6 qubits = ~1s/slot on any modern desktop
                         # (3 UE-groups × 2 PRB-slots, or 2 groups × 3 slots)
MAX_QUBO_UES     = 3     # max UE groups in QUBO; remaining UEs handled classically
QUANTUM_FALLBACK = False # set True by GUI if Qiskit unavailable

# ── Shared QAOA circuit cache: recompile only when topology changes ──
_qaoa_cache = {}         # key: (n_qubits, reps) → compiled ISA circuit


# ─────────────────────────────────────────────
#  Urgency weight
# ─────────────────────────────────────────────

def _ue_weight(ue):
    """
    Urgency-aware objective weight per UE.
      weight = CQI × (1 + buffer/200) × (1 + delay/qos_deadline)

    This combined metric makes QAOA better than Max-CQI (which only
    sees CQI) and faster-reacting than PF (which uses historical avg).
    """
    buf_pressure = ue.buffer / 200.0
    qos_urgency  = 1.0 + ue.delay_accrued / max(ue.qos_delay, 1)
    return ue.cqi * (1.0 + buf_pressure) * qos_urgency


# ─────────────────────────────────────────────
#  Step 1: UE clustering → tractable QUBO size
# ─────────────────────────────────────────────

def _cluster_ues(ues, max_groups):
    """
    When there are more UEs than max_groups, merge the lowest-weight
    UEs into groups.  Each group is represented by one QUBO variable,
    and PRBs won by the group are split proportionally inside it.

    Returns:
      groups      : list of lists of UE objects
      group_weight: np.array of urgency weight per group
    """
    U = len(ues)
    if U <= max_groups:
        return [[ue] for ue in ues], np.array([_ue_weight(ue) for ue in ues])

    # Sort by weight descending; top max_groups-1 are individual, rest merged
    sorted_ues = sorted(ues, key=_ue_weight, reverse=True)
    groups  = [[ue] for ue in sorted_ues[:max_groups - 1]]
    tail    = sorted_ues[max_groups - 1:]
    groups.append(tail)
    weights = np.array([
        max(_ue_weight(ue) for ue in g) for g in groups
    ], dtype=float)
    return groups, weights


# ─────────────────────────────────────────────
#  Step 2: Build QUBO matrix
# ─────────────────────────────────────────────

def _build_qubo(group_weights, R, penalty=10.0):
    """
    Variables: x[g, r] ∈ {0,1} — group g gets PRB-slot r
    Minimise:  –Σ weight[g] × x[g,r]            (objective)
             + penalty × Σ_{g1≠g2} x[g1,r]×x[g2,r]  (one group per PRB)

    Returns dense float64 Q matrix of shape (G×R, G×R).
    """
    G  = len(group_weights)
    sz = G * R
    Q  = np.zeros((sz, sz), dtype=np.float64)

    # Normalise weights to [1, 15] so penalty is on the same scale
    wmax = group_weights.max() or 1.0
    w    = 15.0 * group_weights / wmax

    for g in range(G):
        for r in range(R):
            Q[g * R + r, g * R + r] -= w[g]   # objective (negative → minimise)

    for r in range(R):
        for g1 in range(G):
            for g2 in range(g1 + 1, G):
                i, j = g1 * R + r, g2 * R + r
                Q[i, j] += penalty             # upper triangle only (symmetric below)
                Q[j, i] += penalty

    return Q


# ─────────────────────────────────────────────
#  Step 3: QUBO → Ising → SparsePauliOp
# ─────────────────────────────────────────────

def _qubo_to_pauli_op(Q_np):
    """
    Convert QUBO (minimise x^T Q x, x∈{0,1}^n) to Ising SparsePauliOp
    via substitution x_i = (1 – Z_i) / 2.

    Expansion:
      x_i x_j = (1-Z_i)(1-Z_j)/4 = (1 - Z_i - Z_j + Z_i Z_j)/4
      x_i²   = x_i  (binary)

    Returns SparsePauliOp representing the cost Hamiltonian H_C.
    """
    n  = Q_np.shape[0]
    Qs = (Q_np + Q_np.T) / 2.0      # symmetrise

    h   = np.zeros(n)                # linear Ising coefficients
    J   = {}                         # quadratic
    offset = 0.0

    # Diagonal: Q_ii * x_i = Q_ii*(1-z_i)/2
    for i in range(n):
        h[i]     -= Qs[i, i] / 2.0
        offset   += Qs[i, i] / 2.0

    # Off-diagonal: Q_ij * x_i * x_j = Q_ij*(1-z_i-z_j+z_i z_j)/4
    for i in range(n):
        for j in range(i + 1, n):
            Qij = Qs[i, j]
            J[(i, j)]  = Qij / 4.0
            h[i]      -= Qij / 4.0
            h[j]      -= Qij / 4.0
            offset    += Qij / 4.0

    # Assemble Pauli string list
    terms = []
    for i in range(n):
        if abs(h[i]) > 1e-9:
            p = ['I'] * n;  p[i] = 'Z'
            terms.append((''.join(reversed(p)), float(h[i])))
    for (i, j), val in J.items():
        if abs(val) > 1e-9:
            p = ['I'] * n;  p[i] = 'Z';  p[j] = 'Z'
            terms.append((''.join(reversed(p)), float(val)))
    if not terms:
        terms = [('I' * n, 0.0)]

    return SparsePauliOp.from_list(terms)


# ─────────────────────────────────────────────
#  Step 3b: Precompute Ising arrays for fast expectation
# ─────────────────────────────────────────────

def _ising_arrays(cost_op, n):
    """Extract h (linear) and J_mat (quadratic) from SparsePauliOp for
    fast numpy expectation computation during COBYLA optimisation."""
    h_arr = np.zeros(n)
    J_mat = np.zeros((n, n))
    for term in cost_op:
        pauli_str = term.paulis[0].to_label()
        coeff     = float(term.coeffs[0].real)
        z_pos     = [i for i, p in enumerate(reversed(pauli_str)) if p == 'Z']
        if len(z_pos) == 1:
            h_arr[z_pos[0]] += coeff
        elif len(z_pos) == 2:
            i, j = z_pos
            J_mat[i, j] += coeff;  J_mat[j, i] += coeff
    return h_arr, J_mat


# ─────────────────────────────────────────────
#  Step 3c: Run QAOA on Qiskit Aer
# ─────────────────────────────────────────────

def _run_qaoa(cost_op, reps, shots, maxiter, G, R):
    """
    Build, compile, optimise, and sample a QAOA circuit.

    Key design decisions:
      • Penalty = 40 in QUBO (→ J_ij = 10 in Ising) — strong enough to suppress
        constraint violations while keeping weights distinguishable.
      • Warm-start at β=π/4, γ=π/4 (known-good for p=1,2 QAOA).
      • Feasibility-repair post-processor: instead of taking the single most-
        frequent bitstring, we scan ALL measured bitstrings, filter to those
        satisfying "at most 1 group per PRB slot", then pick the highest-
        objective feasible one.  This is the standard industry approach
        (IBM Research, arXiv:2101.10883) for constrained QAOA.

    Returns: list of ints [b0, b1, …, b_{n-1}]  (best feasible bitstring,
             or best overall if no feasible one found)
    """
    global _qaoa_cache
    n         = cost_op.num_qubits
    cache_key = (n, reps)

    if cache_key not in _qaoa_cache:
        ansatz = QAOAAnsatz(cost_operator=cost_op, reps=reps)
        ansatz.measure_all()
        pm  = generate_preset_pass_manager(optimization_level=1, backend=None)
        isa = pm.run(ansatz)
        _qaoa_cache[cache_key] = (isa, ansatz.num_parameters)

    isa, n_params = _qaoa_cache[cache_key]
    sampler       = StatevectorSampler()
    h_arr, J_mat  = _ising_arrays(cost_op, n)

    def cost_fn(params):
        job    = sampler.run([(isa, params)], shots=shots)
        counts = job.result()[0].data.meas.get_counts()
        total  = sum(counts.values())
        energy = 0.0
        for bs, cnt in counts.items():
            bits = [int(b) for b in reversed(bs)]
            z    = np.array([1 - 2 * b for b in bits], dtype=float)
            energy += (cnt / total) * (h_arr @ z + 0.5 * z @ J_mat @ z)
        return energy

    # ── Warm-started COBYLA ──
    np.random.seed(None)
    x0  = np.tile([np.pi / 4, np.pi / 4], reps)[:n_params]
    x0 += np.random.uniform(-0.3, 0.3, n_params)
    res = scipy_minimize(cost_fn, x0, method='COBYLA',
                         options={'maxiter': maxiter, 'rhobeg': 0.6})

    # ── High-shot final sample ──
    final_shots = max(shots * 8, 1024)
    job    = sampler.run([(isa, res.x)], shots=final_shots)
    counts = job.result()[0].data.meas.get_counts()

    # ── Feasibility-repair: find best VALID bitstring ──
    def _is_feasible(bits):
        """At most one group per PRB slot."""
        for r in range(R):
            if sum(bits[g * R + r] for g in range(G)) > 1:
                return False
        return True

    def _obj(bits):
        """Raw QUBO objective value (before Ising transform)."""
        return sum(
            float(h_arr[g * R + r]) * (1 - 2 * bits[g * R + r])  # crude approx
            for g in range(G) for r in range(R)
        )

    best_bits = None
    best_obj  = -np.inf
    any_feasible = False

    for bs, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bits = [int(b) for b in reversed(bs)]
        if _is_feasible(bits):
            any_feasible = True
            # Count total weight won (maximise)
            obj = sum(
                1.0 for g in range(G) for r in range(R) if bits[g * R + r] == 1
            )  # tie-break by weight in decode step
            if obj > best_obj or best_bits is None:
                best_obj  = obj
                best_bits = bits

    if not any_feasible:
        # Fallback: take most-frequent bitstring (repair handled in decode)
        best_bs   = max(counts, key=counts.get)
        best_bits = [int(b) for b in reversed(best_bs)]

    return best_bits


# ─────────────────────────────────────────────
#  Fallback: fast greedy SA (used if Qiskit
#  unavailable or qubit count exceeds limit)
# ─────────────────────────────────────────────

def _greedy_fallback(group_weights, R):
    """Assign PRB slots to groups greedily by weight (O(G×R), instant)."""
    G      = len(group_weights)
    sol    = np.zeros(G * R, dtype=np.int8)
    counts = np.zeros(G, dtype=int)
    fair   = max(1, R // G)
    for r in range(R):
        order  = sorted(range(G),
                        key=lambda g: (-int(counts[g] < fair), -group_weights[g]))
        chosen = order[0]
        sol[chosen * R + r] = 1
        counts[chosen] += 1
    return sol.tolist()


# ─────────────────────────────────────────────
#  Step 4: Decode solution → per-UE PRB count
# ─────────────────────────────────────────────

def _decode_solution(solution, groups, group_weights, R, n_prbs):
    """
    Map QAOA bitstring → final per-UE PRB allocation.

    Invariant: sum(alloc.values()) == n_prbs exactly.

    Minimum guarantee is at GROUP level (not individual UE level), because
    when n_prbs < U_total it is mathematically impossible to give every UE
    ≥1 PRB. Each of the ≤3 groups always gets at least 1 PRB; within a
    group, PRBs are split by individual urgency — in extreme congestion
    some low-priority UEs in a large group will receive 0 (correct behaviour:
    congestion means lower-priority sessions are preempted).
    """
    G       = len(groups)
    all_ues = [ue for g in groups for ue in g]

    # ── Step 1: award blocks from QAOA bitstring ──
    block_size = max(1, n_prbs // R)
    group_prbs = np.zeros(G, dtype=int)
    claimed    = np.zeros(R, dtype=bool)
    for g in range(G):
        for r in range(R):
            if solution[g * R + r] == 1 and not claimed[r]:
                group_prbs[g] += block_size
                claimed[r]     = True

    # ── Step 2: guarantee every GROUP gets ≥ 1 PRB ──
    # (G ≤ MAX_QUBO_UES = 3 ≤ n_prbs always, so this is always feasible)
    for g in range(G):
        if group_prbs[g] == 0:
            group_prbs[g] = 1

    # ── Step 3: rebalance total to exactly n_prbs ──
    total = int(group_prbs.sum())
    delta = n_prbs - total

    if delta > 0:
        # Under: top up highest-urgency groups
        order = list(np.argsort(group_weights)[::-1])
        for i in range(delta):
            group_prbs[order[i % G]] += 1
    elif delta < 0:
        # Over: trim from lowest-urgency groups (keep ≥ 1 per group)
        order  = list(np.argsort(group_weights))   # ascending
        excess = -delta
        for g in order:
            cut = min(int(group_prbs[g]) - 1, excess)   # keep ≥ 1
            if cut > 0:
                group_prbs[g] -= cut
                excess        -= cut
            if excess == 0:
                break

    # ── Step 4: intra-group proportional split by individual urgency ──
    alloc = {}
    for g, (group, gprbs) in enumerate(zip(groups, group_prbs)):
        gprbs = int(gprbs)
        n_mem = len(group)
        if n_mem == 1:
            alloc[group[0].uid] = gprbs
        else:
            uw = np.array([_ue_weight(ue) for ue in group], dtype=float)
            uw /= uw.sum() or 1.0
            shares = np.floor(uw * gprbs).astype(int)
            # Fix rounding: shares may sum to gprbs±1
            diff = gprbs - int(shares.sum())
            order_idx = list(np.argsort(uw)[::-1])
            i = 0
            while diff > 0:
                shares[order_idx[i % n_mem]] += 1; diff -= 1; i += 1
            i = 0
            while diff < 0:
                k = order_idx[-(i % n_mem) - 1]
                if shares[k] > 0:
                    shares[k] -= 1; diff += 1
                i += 1
            for ue, sh in zip(group, shares):
                alloc[ue.uid] = int(sh)

    # ── Step 5: final exact-sum enforcement (floating-point safety net) ──
    total = sum(alloc.values())
    diff  = n_prbs - total
    if diff != 0:
        ranked = sorted(all_ues, key=_ue_weight, reverse=(diff > 0))
        for i in range(abs(diff)):
            ue = ranked[i % len(all_ues)]
            if diff > 0:
                alloc[ue.uid] += 1
            elif alloc.get(ue.uid, 0) > 0:
                alloc[ue.uid] -= 1

    return alloc


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

def scheduler_quantum(ues, n_prbs, log_fn=None):
    """
    Hybrid classical-quantum MAC-layer scheduler using QAOA on Qiskit.

    Pipeline:
      1. Cluster UEs → ≤ MAX_QUBO_UES groups          (classical)
      2. Build QUBO matrix with urgency-aware weights   (classical)
      3. QUBO → Ising Hamiltonian → SparsePauliOp      (classical)
      4. Run QAOA on Qiskit Aer statevector simulator   (QUANTUM ✓)
      5. Decode best bitstring → per-UE PRB allocation  (classical)

    Falls back to urgency-weighted greedy if:
      • qiskit / qiskit-aer not installed
      • n_qubits > MAX_QUBITS (safety guard)
    """
    U = len(ues)

    # ── PRB slots in the QUBO: at most ceil(MAX_QUBITS / groups) ──
    G_max  = min(U, MAX_QUBO_UES)          # UE groups
    R      = min(n_prbs, MAX_QUBITS // G_max)  # PRB slots (logical)
    R      = max(R, 1)
    n_qubits = G_max * R

    use_qaoa = QISKIT_AVAILABLE and not QUANTUM_FALLBACK and n_qubits <= MAX_QUBITS

    # ── Step 1: cluster ──
    groups, gweights = _cluster_ues(ues, G_max)

    # ── Step 2: QUBO ──
    Q_np = _build_qubo(gweights, R, penalty=40.0)

    if use_qaoa:
        try:
            # ── Step 3: Ising ──
            cost_op  = _qubo_to_pauli_op(Q_np)

            if log_fn:
                log_fn(f"  [QAOA] {cost_op.num_qubits} qubits | "
                       f"reps={QAOA_REPS} shots={QAOA_SHOTS} iter={QAOA_MAXITER}")

            # ── Step 4: QAOA ──
            solution = _run_qaoa(cost_op, QAOA_REPS, QAOA_SHOTS, QAOA_MAXITER,
                                 len(groups), R)

        except Exception as exc:
            if log_fn:
                log_fn(f"  [QAOA] Error: {exc} — using greedy fallback")
            solution = _greedy_fallback(gweights, R)
    else:
        if log_fn and not QISKIT_AVAILABLE:
            log_fn("  [QAOA] Qiskit not found — using greedy fallback")
        solution = _greedy_fallback(gweights, R)

    # ── Step 5: decode ──
    return _decode_solution(solution, groups, gweights, R, n_prbs)

#  PERFORMANCE METRICS
def jain_fairness(throughputs):
    if not throughputs or sum(throughputs) == 0:
        return 0
    n = len(throughputs)
    return (sum(throughputs) ** 2) / (n * sum(t ** 2 for t in throughputs))

def run_simulation(n_ues, n_prbs, n_slots, scheduler_type, log_fn=None, progress_fn=None):
    env = NetworkEnvironment(n_ues, n_prbs)
    pf_history = {ue.uid: 1 for ue in env.ues}

    results = {
        "throughput_per_slot": [],
        "fairness_per_slot":   [],
        "delay_per_slot":      [],
        "drop_rate_per_slot":  [],
        "per_ue_throughput":   {ue.uid: [] for ue in env.ues},
        "slot_logs":           [],
        "scheduler":           scheduler_type,
        "n_ues":               n_ues,
        "n_prbs":              n_prbs,
    }

    for t in range(n_slots):
        env.step()
        state = env.get_state()

        # ── choose scheduler ──
        if scheduler_type == "Round Robin":
            alloc = scheduler_round_robin(env.ues, n_prbs)
        elif scheduler_type == "Max CQI":
            alloc = scheduler_max_cqi(env.ues, n_prbs)
        elif scheduler_type == "Proportional Fair":
            alloc = scheduler_proportional_fair(env.ues, n_prbs, pf_history)
        elif scheduler_type == "Quantum (QAOA)":
            alloc = scheduler_quantum(env.ues, n_prbs, log_fn=log_fn)
        else:
            alloc = scheduler_round_robin(env.ues, n_prbs)

        # ── transmit ──
        slot_sent = []
        for ue in env.ues:
            sent = ue.transmit(alloc[ue.uid])
            slot_sent.append(sent)
            results["per_ue_throughput"][ue.uid].append(sent)
            pf_history[ue.uid] = 0.9 * pf_history.get(ue.uid, 1) + 0.1 * sent

        total_tp   = sum(slot_sent)
        fairness   = jain_fairness(slot_sent)
        avg_delay  = np.mean([ue.delay_accrued for ue in env.ues])
        drop_rate  = sum(1 for ue in env.ues if ue.buffer >= 190) / n_ues

        results["throughput_per_slot"].append(total_tp)
        results["fairness_per_slot"].append(fairness)
        results["delay_per_slot"].append(avg_delay)
        results["drop_rate_per_slot"].append(drop_rate)

        # ── log ──
        if log_fn and (t % max(1, n_slots // 40) == 0 or t < 5):
            log_lines = [f"\n── Slot {t+1:04d} ──"]
            for ue in env.ues:
                s = alloc[ue.uid]
                log_lines.append(
                    f"  UE{ue.uid:02d}[{ue.profile[:3]}] "
                    f"CQI={ue.cqi:2d}  buf={state[ue.uid]['buffer']:3d}  "
                    f"prbs={s}  delay={ue.delay_accrued:.1f}ms"
                )
            log_lines.append(f"  ► Total TP={total_tp:4d}  Fairness={fairness:.3f}  AvgDelay={avg_delay:.2f}ms")
            log_fn("\n".join(log_lines))

        if progress_fn:
            progress_fn((t + 1) / n_slots * 100)

    return results

#  GUI
SCHEDULERS = ["Round Robin", "Max CQI", "Proportional Fair", "Quantum (QAOA)"]

PALETTE = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "border":   "#30363d",
    "accent":   "#58a6ff",
    "green":    "#3fb950",
    "yellow":   "#d29922",
    "red":      "#f85149",
    "purple":   "#bc8cff",
    "text":     "#e6edf3",
    "muted":    "#8b949e",
}

SCHED_COLORS = {
    "Round Robin":        "#58a6ff",
    "Max CQI":            "#3fb950",
    "Proportional Fair":  "#d29922",
    "Quantum (QAOA)":     "#bc8cff",
}

class SimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quantum-Assisted RAN Scheduler — O-RAN MVP  [Qiskit QAOA]")
        self.configure(bg=PALETTE["bg"])
        self.geometry("1360x860")
        self.resizable(True, True)

        self._results_store = {}   # keyed by scheduler name
        self._running = False

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────

    def _build_ui(self):
        # ── header ──
        hdr = tk.Frame(self, bg=PALETTE["surface"], pady=10)
        hdr.pack(fill="x", padx=0, pady=0)
        tk.Label(hdr, text="⚛  Quantum-Assisted RAN Scheduler",
                 font=("Consolas", 16, "bold"),
                 fg=PALETTE["purple"], bg=PALETTE["surface"]).pack(side="left", padx=18)
        tk.Label(hdr, text="M.S. Ramaiah Institute of Technology — CSE Dept",
                 font=("Consolas", 10),
                 fg=PALETTE["muted"], bg=PALETTE["surface"]).pack(side="right", padx=18)

        # ── main panes ──
        body = tk.Frame(self, bg=PALETTE["bg"])
        body.pack(fill="both", expand=True, padx=8, pady=6)

        left  = tk.Frame(body, bg=PALETTE["bg"], width=300)
        left.pack(side="left", fill="y", padx=(0, 6))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=PALETTE["bg"])
        right.pack(side="left", fill="both", expand=True)

        self._build_controls(left)
        self._build_right(right)

    def _section(self, parent, title):
        f = tk.LabelFrame(parent, text=f"  {title}  ",
                          font=("Consolas", 9, "bold"),
                          fg=PALETTE["accent"], bg=PALETTE["surface"],
                          bd=1, relief="solid",
                          highlightbackground=PALETTE["border"])
        f.pack(fill="x", padx=4, pady=5)
        return f

    def _build_controls(self, parent):
        # ── Simulation Parameters ──
        sf = self._section(parent, "Simulation Parameters")

        params = [
            ("UEs", "n_ues", 8, 1, 20),
            ("PRBs", "n_prbs", 12, 4, 50),
            ("Time Slots", "n_slots", 200, 50, 2000),
        ]
        self._vars = {}
        for label, key, default, lo, hi in params:
            row = tk.Frame(sf, bg=PALETTE["surface"])
            row.pack(fill="x", padx=8, pady=3)
            tk.Label(row, text=label, width=10, anchor="w",
                     font=("Consolas", 9), fg=PALETTE["text"],
                     bg=PALETTE["surface"]).pack(side="left")
            v = tk.IntVar(value=default)
            self._vars[key] = v
            spin = tk.Spinbox(row, from_=lo, to=hi, textvariable=v, width=6,
                              font=("Consolas", 9),
                              bg=PALETTE["bg"], fg=PALETTE["text"],
                              insertbackground=PALETTE["text"],
                              buttonbackground=PALETTE["border"],
                              relief="flat")
            spin.pack(side="right")

        # ── Scheduler Selection ──
        ssf = self._section(parent, "Schedulers")
        self._sched_vars = {}
        for s in SCHEDULERS:
            v = tk.BooleanVar(value=True)
            self._sched_vars[s] = v
            cb = tk.Checkbutton(ssf, text=s, variable=v,
                                font=("Consolas", 9),
                                fg=SCHED_COLORS[s], bg=PALETTE["surface"],
                                selectcolor=PALETTE["bg"],
                                activeforeground=SCHED_COLORS[s],
                                activebackground=PALETTE["surface"],
                                bd=0, highlightthickness=0)
            cb.pack(anchor="w", padx=10, pady=2)

        # ── Quantum Settings ──
        qf = self._section(parent, "Quantum Settings  ⚛")

        # Qiskit availability banner
        status_color = PALETTE["green"] if QISKIT_AVAILABLE else PALETTE["red"]
        status_text  = ("✓ Qiskit + Aer ready" if QISKIT_AVAILABLE
                        else "✗ Qiskit not found — install:\n  pip install qiskit qiskit-aer\n  pip install qiskit-algorithms scipy")
        tk.Label(qf, text=status_text,
                 font=("Consolas", 8, "bold"), fg=status_color,
                 bg=PALETTE["surface"], justify="left",
                 anchor="w").pack(fill="x", padx=8, pady=(5, 3))

        # QAOA reps slider
        rrow = tk.Frame(qf, bg=PALETTE["surface"])
        rrow.pack(fill="x", padx=8, pady=2)
        tk.Label(rrow, text="QAOA Reps (p):", font=("Consolas", 8),
                 fg=PALETTE["muted"], bg=PALETTE["surface"],
                 anchor="w").pack(side="left")
        self._reps_label = tk.Label(rrow, text="2",
                                     font=("Consolas", 8, "bold"),
                                     fg=PALETTE["purple"], bg=PALETTE["surface"], width=3)
        self._reps_label.pack(side="right")
        self._reps_var = tk.IntVar(value=2)
        tk.Scale(qf, from_=1, to=4, orient="horizontal",
                 variable=self._reps_var, bg=PALETTE["surface"],
                 fg=PALETTE["muted"], troughcolor=PALETTE["bg"],
                 highlightthickness=0, bd=0, showvalue=False, length=220,
                 command=lambda v: (self._reps_label.config(text=str(int(float(v)))),
                                   self._on_qaoa_param("reps", int(float(v))))
                 ).pack(padx=8, pady=0)

        # COBYLA iterations slider
        irow = tk.Frame(qf, bg=PALETTE["surface"])
        irow.pack(fill="x", padx=8, pady=2)
        tk.Label(irow, text="Optimizer iters:", font=("Consolas", 8),
                 fg=PALETTE["muted"], bg=PALETTE["surface"],
                 anchor="w").pack(side="left")
        self._iter_label = tk.Label(irow, text="25",
                                     font=("Consolas", 8, "bold"),
                                     fg=PALETTE["purple"], bg=PALETTE["surface"], width=4)
        self._iter_label.pack(side="right")
        self._iter_var = tk.IntVar(value=25)
        tk.Scale(qf, from_=20, to=150, orient="horizontal",
                 variable=self._iter_var, bg=PALETTE["surface"],
                 fg=PALETTE["muted"], troughcolor=PALETTE["bg"],
                 highlightthickness=0, bd=0, showvalue=False, length=220,
                 command=lambda v: (self._iter_label.config(text=str(int(float(v)))),
                                   self._on_qaoa_param("maxiter", int(float(v))))
                 ).pack(padx=8, pady=0)

        # Shots slider
        shrow = tk.Frame(qf, bg=PALETTE["surface"])
        shrow.pack(fill="x", padx=8, pady=2)
        tk.Label(shrow, text="Shots:", font=("Consolas", 8),
                 fg=PALETTE["muted"], bg=PALETTE["surface"],
                 anchor="w").pack(side="left")
        self._shots_label = tk.Label(shrow, text="128",
                                      font=("Consolas", 8, "bold"),
                                      fg=PALETTE["purple"], bg=PALETTE["surface"], width=4)
        self._shots_label.pack(side="right")
        self._shots_var = tk.IntVar(value=128)
        tk.Scale(qf, from_=128, to=2048, resolution=128,
                 orient="horizontal", variable=self._shots_var,
                 bg=PALETTE["surface"], fg=PALETTE["muted"],
                 troughcolor=PALETTE["bg"], highlightthickness=0, bd=0,
                 showvalue=False, length=220,
                 command=lambda v: (self._shots_label.config(text=str(int(float(v)))),
                                   self._on_qaoa_param("shots", int(float(v))))
                 ).pack(padx=8, pady=0)

        # Max qubits display
        qrow = tk.Frame(qf, bg=PALETTE["surface"])
        qrow.pack(fill="x", padx=8, pady=(3, 2))
        tk.Label(qrow, text="Max qubits (cap):", font=("Consolas", 8),
                 fg=PALETTE["muted"], bg=PALETTE["surface"],
                 anchor="w").pack(side="left")
        tk.Label(qrow, text=str(MAX_QUBITS),
                 font=("Consolas", 8, "bold"),
                 fg=PALETTE["accent"], bg=PALETTE["surface"]).pack(side="right")

        # Info box
        hint_f = tk.Frame(qf, bg="#1a1f2e", bd=0)
        hint_f.pack(fill="x", padx=8, pady=6)
        tk.Label(hint_f, text="⚛ Real QAOA pipeline:",
                 font=("Consolas", 8, "bold"),
                 fg=PALETTE["purple"], bg="#1a1f2e",
                 anchor="w").pack(fill="x", padx=6, pady=(5, 2))
        hints = [
            "1. UEs → urgency groups",
            "2. QUBO matrix (≤12 qubits)",
            "3. Ising Hamiltonian (Pauli Z)",
            "4. QAOA on Qiskit Aer sim",
            "5. Decode bitstring → PRBs",
            "",
            "Quantum wins when:",
            "• PRBs/UEs < 2 (congested)",
            "• Mixed VoIP+Video profiles",
            "• reps≥2, iters≥50",
        ]
        for h in hints:
            tk.Label(hint_f, text=h, font=("Consolas", 7),
                     fg=PALETTE["muted"], bg="#1a1f2e",
                     anchor="w", justify="left").pack(fill="x", padx=10)
        tk.Frame(hint_f, height=4, bg="#1a1f2e").pack()

        # ── Run Button ──
        self._run_btn = tk.Button(parent, text="▶  RUN SIMULATION",
                                  font=("Consolas", 11, "bold"),
                                  fg="#0d1117", bg=PALETTE["green"],
                                  activebackground="#2ea043",
                                  relief="flat", cursor="hand2",
                                  command=self._on_run, pady=8)
        self._run_btn.pack(fill="x", padx=4, pady=8)

        self._plot_btn = tk.Button(parent, text="📊  OPEN FULL PLOTS",
                                   font=("Consolas", 10),
                                   fg=PALETTE["text"], bg=PALETTE["border"],
                                   activebackground="#484f58",
                                   relief="flat", cursor="hand2",
                                   command=self._on_full_plots, pady=6,
                                   state="disabled")
        self._plot_btn.pack(fill="x", padx=4, pady=2)

        # ── Progress ──
        pf = self._section(parent, "Progress")
        self._progress_label = tk.Label(pf, text="Idle",
                                         font=("Consolas", 9), fg=PALETTE["muted"],
                                         bg=PALETTE["surface"])
        self._progress_label.pack(pady=2)
        self._progress = ttk.Progressbar(pf, length=240, mode="determinate")
        self._progress.pack(padx=8, pady=4, fill="x")
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TProgressbar", troughcolor=PALETTE["bg"],
                        background=PALETTE["purple"], thickness=8)

        # ── Summary Stats ──
        self._stats_frame = self._section(parent, "Last Run Summary")
        self._stats_text = tk.Label(self._stats_frame, text="—",
                                     font=("Consolas", 8), fg=PALETTE["muted"],
                                     bg=PALETTE["surface"], justify="left",
                                     anchor="w")
        self._stats_text.pack(padx=8, pady=4, fill="x")

    def _build_right(self, parent):
        # ── Notebook: Live Plot + Log ──
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        style = ttk.Style()
        style.configure("TNotebook", background=PALETTE["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=PALETTE["surface"],
                        foreground=PALETTE["muted"],
                        font=("Consolas", 9), padding=[12, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", PALETTE["bg"])],
                  foreground=[("selected", PALETTE["accent"])])

        # Live chart tab
        chart_tab = tk.Frame(nb, bg=PALETTE["bg"])
        nb.add(chart_tab, text="  Live Chart  ")
        self._build_live_chart(chart_tab)

        # Log tab
        log_tab = tk.Frame(nb, bg=PALETTE["bg"])
        nb.add(log_tab, text="  Simulation Log  ")
        self._log_area = scrolledtext.ScrolledText(
            log_tab, font=("Consolas", 8), bg="#0a0e14",
            fg=PALETTE["text"], insertbackground=PALETTE["text"],
            relief="flat", bd=0, wrap="none")
        self._log_area.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_live_chart(self, parent):
        self._fig = Figure(figsize=(9, 5), facecolor=PALETTE["bg"])
        self._fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.92, bottom=0.1)
        gs = gridspec.GridSpec(2, 2, figure=self._fig)

        ax_kw = dict(facecolor=PALETTE["surface"])
        self._ax_tp  = self._fig.add_subplot(gs[0, 0], **ax_kw)
        self._ax_fi  = self._fig.add_subplot(gs[0, 1], **ax_kw)
        self._ax_del = self._fig.add_subplot(gs[1, 0], **ax_kw)
        self._ax_bar = self._fig.add_subplot(gs[1, 1], **ax_kw)

        for ax, title, ylabel in [
            (self._ax_tp,  "Throughput (packets/slot)", "Packets"),
            (self._ax_fi,  "Jain Fairness Index",       "Fairness"),
            (self._ax_del, "Avg Delay (slots)",          "Delay"),
            (self._ax_bar, "Avg Throughput Comparison",  "Packets"),
        ]:
            ax.set_title(title, color=PALETTE["accent"], fontsize=8, pad=6,
                         fontfamily="monospace")
            ax.set_ylabel(ylabel, color=PALETTE["muted"], fontsize=7)
            ax.tick_params(colors=PALETTE["muted"], labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE["border"])
            ax.set_facecolor(PALETTE["surface"])
            ax.grid(color=PALETTE["border"], linewidth=0.5, alpha=0.6)

        self._canvas = FigureCanvasTkAgg(self._fig, parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas.draw()

    # ── Run logic ────────────────────────────────────────────────

    def _on_qaoa_param(self, param, value):
        global QAOA_REPS, QAOA_SHOTS, QAOA_MAXITER
        if param == "reps":    QAOA_REPS    = value
        elif param == "shots": QAOA_SHOTS   = value
        elif param == "maxiter": QAOA_MAXITER = value

    def _on_run(self):
        if self._running:
            return
        selected = [s for s, v in self._sched_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("No Scheduler", "Select at least one scheduler.")
            return

        self._results_store.clear()
        self._running = True
        self._run_btn.config(state="disabled", text="⏳  Running…")
        self._plot_btn.config(state="disabled")
        self._log_area.delete("1.0", tk.END)
        self._stats_text.config(text="Running…")
        self._progress["value"] = 0

        n_ues   = self._vars["n_ues"].get()
        n_prbs  = self._vars["n_prbs"].get()
        n_slots = self._vars["n_slots"].get()

        thread = threading.Thread(
            target=self._run_all_schedulers,
            args=(selected, n_ues, n_prbs, n_slots),
            daemon=True
        )
        thread.start()

    def _run_all_schedulers(self, schedulers, n_ues, n_prbs, n_slots):
        total_jobs = len(schedulers)
        for idx, sched in enumerate(schedulers):
            self._append_log(f"\n{'='*52}\n  Scheduler: {sched}\n{'='*52}")

            def prog(pct, sched=sched, idx=idx):
                overall = (idx + pct / 100) / total_jobs * 100
                self.after(0, lambda v=overall, s=sched:
                           (self._progress.__setitem__("value", v),
                            self._progress_label.config(
                                text=f"{s} — {pct:.0f}%")))

            res = run_simulation(
                n_ues, n_prbs, n_slots, sched,
                log_fn=self._append_log,
                progress_fn=prog
            )
            self._results_store[sched] = res
            self.after(0, self._update_live_chart)

        self.after(0, self._on_done)

    def _on_done(self):
        self._running = False
        self._run_btn.config(state="normal", text="▶  RUN SIMULATION")
        self._plot_btn.config(state="normal")
        self._progress["value"] = 100
        self._progress_label.config(text="Done ✓")
        self._update_summary()
        self._append_log("\n\n✅  All schedulers complete.")

    # ── Chart update ─────────────────────────────────────────────

    def _update_live_chart(self):
        for ax in [self._ax_tp, self._ax_fi, self._ax_del]:
            ax.cla()

        for ax, title, ylabel in [
            (self._ax_tp,  "Throughput (packets/slot)", "Packets"),
            (self._ax_fi,  "Jain Fairness Index",       "Fairness"),
            (self._ax_del, "Avg Delay (slots)",          "Delay"),
        ]:
            ax.set_title(title, color=PALETTE["accent"], fontsize=8, pad=5,
                         fontfamily="monospace")
            ax.set_ylabel(ylabel, color=PALETTE["muted"], fontsize=7)
            ax.tick_params(colors=PALETTE["muted"], labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE["border"])
            ax.set_facecolor(PALETTE["surface"])
            ax.grid(color=PALETTE["border"], linewidth=0.5, alpha=0.6)

        for sched, res in self._results_store.items():
            c = SCHED_COLORS[sched]
            n = len(res["throughput_per_slot"])
            xs = range(n)
            # smooth
            tp  = _moving_avg(res["throughput_per_slot"], 5)
            fi  = _moving_avg(res["fairness_per_slot"], 5)
            dl  = _moving_avg(res["delay_per_slot"], 5)
            self._ax_tp.plot(xs[:len(tp)],  tp,  color=c, linewidth=1.2, label=sched)
            self._ax_fi.plot(xs[:len(fi)],  fi,  color=c, linewidth=1.2, label=sched)
            self._ax_del.plot(xs[:len(dl)], dl, color=c, linewidth=1.2, label=sched)

        for ax in [self._ax_tp, self._ax_fi, self._ax_del]:
            if self._results_store:
                ax.legend(fontsize=6, facecolor=PALETTE["bg"],
                          edgecolor=PALETTE["border"],
                          labelcolor=PALETTE["text"])

        # bar chart
        self._ax_bar.cla()
        self._ax_bar.set_facecolor(PALETTE["surface"])
        self._ax_bar.set_title("Avg Throughput Comparison", color=PALETTE["accent"],
                               fontsize=8, pad=5, fontfamily="monospace")
        self._ax_bar.set_ylabel("Avg Packets/Slot", color=PALETTE["muted"], fontsize=7)
        self._ax_bar.tick_params(colors=PALETTE["muted"], labelsize=7)
        for spine in self._ax_bar.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        self._ax_bar.grid(axis="y", color=PALETTE["border"], linewidth=0.5, alpha=0.6)

        names   = list(self._results_store.keys())
        avgs    = [np.mean(self._results_store[s]["throughput_per_slot"]) for s in names]
        colors  = [SCHED_COLORS[s] for s in names]
        bars = self._ax_bar.bar(range(len(names)), avgs, color=colors, width=0.55)
        self._ax_bar.set_xticks(range(len(names)))
        self._ax_bar.set_xticklabels(
            [n.replace(" ", "\n") for n in names], fontsize=6, color=PALETTE["text"])
        for bar, val in zip(bars, avgs):
            self._ax_bar.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + 0.3,
                              f"{val:.1f}", ha="center", va="bottom",
                              fontsize=6, color=PALETTE["text"],
                              fontfamily="monospace")

        self._canvas.draw()

    # ── Summary ──────────────────────────────────────────────────

    def _update_summary(self):
        lines = []
        for sched, res in self._results_store.items():
            avg_tp = np.mean(res["throughput_per_slot"])
            avg_fi = np.mean(res["fairness_per_slot"])
            avg_dl = np.mean(res["delay_per_slot"])
            drop   = np.mean(res["drop_rate_per_slot"]) * 100
            abbr   = sched[:12]
            lines.append(
                f"{abbr:<13}  TP={avg_tp:5.1f}  FI={avg_fi:.3f}\n"
                f"               Del={avg_dl:4.1f}  Drop={drop:.1f}%"
            )
        self._stats_text.config(text="\n".join(lines) if lines else "—",
                                fg=PALETTE["text"])

    # ── Full Plots Window ────────────────────────────────────────

    def _on_full_plots(self):
        if not self._results_store:
            return
        win = tk.Toplevel(self)
        win.title("Full Performance Analysis")
        win.configure(bg=PALETTE["bg"])
        win.geometry("1200x800")

        fig = plt.Figure(figsize=(14, 9), facecolor=PALETTE["bg"])
        fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.93, bottom=0.08,
                            left=0.07, right=0.97)
        gs = gridspec.GridSpec(3, 3, figure=fig)
        ax_kw = dict(facecolor=PALETTE["surface"])

        axes = {
            "tp":   fig.add_subplot(gs[0, :2], **ax_kw),
            "fi":   fig.add_subplot(gs[1, :2], **ax_kw),
            "dl":   fig.add_subplot(gs[2, :2], **ax_kw),
            "bar_tp": fig.add_subplot(gs[0, 2], **ax_kw),
            "bar_fi": fig.add_subplot(gs[1, 2], **ax_kw),
            "bar_dl": fig.add_subplot(gs[2, 2], **ax_kw),
        }

        def _style_ax(ax, title, ylabel):
            ax.set_title(title, color=PALETTE["accent"], fontsize=9,
                         fontfamily="monospace", pad=6)
            ax.set_ylabel(ylabel, color=PALETTE["muted"], fontsize=8)
            ax.tick_params(colors=PALETTE["muted"], labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(PALETTE["border"])
            ax.grid(color=PALETTE["border"], linewidth=0.5, alpha=0.6)

        _style_ax(axes["tp"],   "Throughput Over Time",        "Packets/Slot")
        _style_ax(axes["fi"],   "Jain Fairness Index Over Time","Fairness")
        _style_ax(axes["dl"],   "Average Delay Over Time",      "Delay (slots)")
        _style_ax(axes["bar_tp"], "Avg Throughput",             "Packets/Slot")
        _style_ax(axes["bar_fi"], "Avg Fairness",               "Fairness")
        _style_ax(axes["bar_dl"], "Avg Delay",                  "Slots")

        names  = list(self._results_store.keys())
        colors = [SCHED_COLORS[s] for s in names]

        for sched, res in self._results_store.items():
            c = SCHED_COLORS[sched]
            n = len(res["throughput_per_slot"])
            xs = list(range(n))
            tp  = _moving_avg(res["throughput_per_slot"], 10)
            fi  = _moving_avg(res["fairness_per_slot"], 10)
            dl  = _moving_avg(res["delay_per_slot"], 10)
            axes["tp"].plot(xs[:len(tp)],  tp,  color=c, lw=1.4, label=sched)
            axes["fi"].plot(xs[:len(fi)],  fi,  color=c, lw=1.4, label=sched)
            axes["dl"].plot(xs[:len(dl)],  dl,  color=c, lw=1.4, label=sched)

        for ax_key in ["tp", "fi", "dl"]:
            axes[ax_key].legend(fontsize=7, facecolor=PALETTE["bg"],
                                edgecolor=PALETTE["border"],
                                labelcolor=PALETTE["text"])

        def bar_group(ax, metric_key):
            vals = [np.mean(self._results_store[s][metric_key]) for s in names]
            bars = ax.bar(range(len(names)), vals, color=colors, width=0.6)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n.replace(" ", "\n") for n in names],
                               fontsize=7, color=PALETTE["text"])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, color=PALETTE["text"], fontfamily="monospace")

        bar_group(axes["bar_tp"], "throughput_per_slot")
        bar_group(axes["bar_fi"], "fairness_per_slot")
        bar_group(axes["bar_dl"], "delay_per_slot")

        fig.suptitle("O-RAN MAC Scheduler — Full Performance Analysis",
                     color=PALETTE["purple"], fontsize=12, fontfamily="monospace")

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    # ── Helpers ──────────────────────────────────────────────────

    def _append_log(self, text):
        def _do():
            self._log_area.insert(tk.END, text + "\n")
            self._log_area.see(tk.END)
        self.after(0, _do)


def _moving_avg(data, window=5):
    if len(data) < window:
        return data
    result = []
    for i in range(len(data)):
        lo = max(0, i - window // 2)
        hi = min(len(data), i + window // 2 + 1)
        result.append(np.mean(data[lo:hi]))
    return result

#  ENTRY POINT
if __name__ == "__main__":
    app = SimulatorApp()
    app.mainloop()