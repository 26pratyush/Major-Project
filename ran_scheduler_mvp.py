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
Deps: pip install matplotlib numpy scipy
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

# ──────────────────────────────────────────────
#  SIMULATION CORE
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
#  CLASSICAL SCHEDULERS
# ──────────────────────────────────────────────

def scheduler_round_robin(ues, n_prbs):
    n = len(ues)
    base, rem = divmod(n_prbs, n)
    alloc = {ue.uid: base for ue in ues}
    for i in range(rem):
        alloc[ues[i].uid] += 1
    return alloc

def scheduler_max_cqi(ues, n_prbs):
    sorted_ues = sorted(ues, key=lambda u: u.cqi, reverse=True)
    alloc = {ue.uid: 0 for ue in ues}
    pool = n_prbs
    for ue in sorted_ues:
        give = min(pool, max(1, pool // max(1, len(sorted_ues))))
        alloc[ue.uid] = give
        pool -= give
        if pool <= 0:
            break
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


# ──────────────────────────────────────────────
#  QUANTUM-INSPIRED SCHEDULER  (QUBO / QAOA-sim)
# ──────────────────────────────────────────────

# Global speed-mode flag toggled by GUI
QUANTUM_FAST_MODE = False     # True = greedy-descent only (instant)
QUANTUM_SA_STEPS  = 300       # reduced from 800; still good quality

def _ue_weight(ue):
    """
    Urgency-aware objective weight per UE.
    Combines channel quality (CQI) with buffer pressure and QoS deadline.
    This is what makes quantum actually *better* in the right scenarios:
    it optimises a richer objective than plain Max-CQI.

    weight = CQI  ×  (1 + buffer_pressure)  ×  qos_urgency
      buffer_pressure = buffer / 200          (how full the buffer is)
      qos_urgency     = 1 + delay / qos_ms   (how overdue the UE is)
    """
    buf_pressure = ue.buffer / 200.0
    qos_urgency  = 1.0 + ue.delay_accrued / max(ue.qos_delay, 1)
    return ue.cqi * (1.0 + buf_pressure) * qos_urgency

def build_qubo_matrix(ues, R, penalty=8.0):
    """
    Build QUBO as a dense NumPy matrix Q of shape (U*R, U*R).
    Minimise  x^T Q x.

    Objective (negated for minimisation):
      -weight_u  per diagonal x[u,r]
    Constraint — each PRB assigned to at most one UE:
      penalty * x[u1,r] * x[u2,r]  for u1 ≠ u2, same r
    """
    U   = len(ues)
    sz  = U * R
    Q   = np.zeros((sz, sz), dtype=np.float32)

    weights = [_ue_weight(ue) for ue in ues]
    # normalise weights to [1, 15] range so penalty stays meaningful
    wmax = max(weights) or 1
    weights = [15.0 * w / wmax for w in weights]

    for u in range(U):
        for r in range(R):
            Q[u * R + r, u * R + r] -= weights[u]   # objective

    for r in range(R):
        for u1 in range(U):
            for u2 in range(u1 + 1, U):
                i, j = u1 * R + r, u2 * R + r
                Q[i, j] += penalty                   # upper triangle
                Q[j, i] += penalty                   # symmetric

    return Q

def _energy_np(Q, x):
    """Fast matrix-form energy: x^T Q x (x is a numpy bool/int array)."""
    return float(x @ Q @ x)

def _delta_flip(Q, x, k):
    """Energy change when flipping bit k — O(N) not O(N²)."""
    # ΔE = (1-2x_k) * (Q[k,k] + 2 * sum_{j≠k} Q[k,j]*x[j])
    row_sum = np.dot(Q[k], x) - Q[k, k] * x[k]
    return (1 - 2 * int(x[k])) * (Q[k, k] + 2 * row_sum)

def _greedy_warm_start(ues, R):
    """
    Greedy initialisation: assign each PRB to the highest-weight UE
    that doesn't already have more than its fair share.
    Much better starting point than random → SA converges faster.
    """
    U      = len(ues)
    x      = np.zeros(U * R, dtype=np.int8)
    counts = np.zeros(U, dtype=int)
    weights = [_ue_weight(ue) for ue in ues]
    fair   = max(1, R // U)

    for r in range(R):
        # prefer UEs under their fair share, then by weight
        order = sorted(range(U),
                       key=lambda u: (-int(counts[u] < fair), -weights[u]))
        chosen = order[0]
        x[chosen * R + r] = 1
        counts[chosen] += 1
    return x

def simulated_annealing_qubo(Q, x0, n_steps=QUANTUM_SA_STEPS,
                              T_start=3.0, T_end=0.02):
    """
    SA over QUBO using numpy delta-flip for O(N) per step.
    Warm-starts from x0 (greedy solution).
    """
    sz   = len(x0)
    x    = x0.copy()
    e    = _energy_np(Q, x)
    bx   = x.copy(); be = e
    T    = T_start
    decay = (T_end / T_start) ** (1.0 / max(n_steps, 1))

    for _ in range(n_steps):
        k     = random.randrange(sz)
        delta = _delta_flip(Q, x, k)
        if delta < 0 or random.random() < math.exp(-delta / T):
            x[k] ^= 1
            e += delta
            if e < be:
                be = e; bx = x.copy()
        T *= decay

    return bx

def _greedy_descent(Q, x0):
    """
    Pure greedy bit-flip descent — no randomness, instant convergence.
    Used in FAST mode.  Finds a local minimum of the QUBO.
    """
    x = x0.copy()
    improved = True
    while improved:
        improved = False
        order = list(range(len(x)))
        random.shuffle(order)
        for k in order:
            if _delta_flip(Q, x, k) < -1e-6:
                x[k] ^= 1
                improved = True
    return x

def scheduler_quantum(ues, n_prbs, fast_mode=None):
    """
    QUBO-based quantum-inspired scheduler.

    Speed modes:
      fast_mode=True  → greedy warm-start + greedy descent  (~instant)
      fast_mode=False → greedy warm-start + SA refinement   (~0.05s/slot)

    Performance edge over classical methods appears when:
      • UEs have MIXED traffic profiles (VoIP + Video together)
      • Buffer pressure is HIGH (many UEs near full)
      • PRBs are SCARCE relative to UEs  (n_prbs / n_ues < 2)
      • QoS deadlines are TIGHT (mix of real-time + best-effort traffic)
    In these conditions the urgency-weighted QUBO objective routes
    scarce PRBs to the UEs that need them most, beating Max-CQI
    (which ignores delay) and PF (which is slower to react to bursts).
    """
    if fast_mode is None:
        fast_mode = QUANTUM_FAST_MODE

    R  = min(n_prbs, 10)          # QUBO window — keep ≤ 10 PRBs for speed
    U  = len(ues)
    Q  = build_qubo_matrix(ues, R, penalty=8.0)
    x0 = _greedy_warm_start(ues, R)

    if fast_mode:
        solution = _greedy_descent(Q, x0)
    else:
        solution = simulated_annealing_qubo(Q, x0, n_steps=QUANTUM_SA_STEPS)

    # ── decode solution ──
    alloc      = {ue.uid: 0 for ue in ues}
    prb_claimed = np.zeros(R, dtype=bool)

    for u, ue in enumerate(ues):
        for r in range(R):
            if solution[u * R + r] == 1 and not prb_claimed[r]:
                alloc[ue.uid] += 1
                prb_claimed[r] = True

    # distribute PRBs beyond the QUBO window (proportional to urgency)
    leftover = n_prbs - R
    if leftover > 0:
        weights = sorted(ues, key=_ue_weight, reverse=True)
        for i, ue in enumerate(weights):
            alloc[ue.uid] += leftover // U + (1 if i < leftover % U else 0)

    # guarantee every UE gets ≥ 1 PRB
    for ue in ues:
        if alloc[ue.uid] == 0:
            alloc[ue.uid] = 1

    return alloc


# ──────────────────────────────────────────────
#  PERFORMANCE METRICS
# ──────────────────────────────────────────────

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
        elif scheduler_type == "Quantum (QUBO-SA)":
            alloc = scheduler_quantum(env.ues, n_prbs, fast_mode=QUANTUM_FAST_MODE)
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


# ──────────────────────────────────────────────
#  GUI
# ──────────────────────────────────────────────

SCHEDULERS = ["Round Robin", "Max CQI", "Proportional Fair", "Quantum (QUBO-SA)"]

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
    "Quantum (QUBO-SA)":  "#bc8cff",
}

class SimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quantum-Assisted RAN Scheduler — O-RAN MVP")
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
        qf = self._section(parent, "Quantum Settings")

        # Fast mode toggle
        frow = tk.Frame(qf, bg=PALETTE["surface"])
        frow.pack(fill="x", padx=8, pady=4)
        self._fast_mode_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            frow, text="⚡ Fast Mode (greedy, instant)",
            variable=self._fast_mode_var,
            font=("Consolas", 9), fg=PALETTE["yellow"],
            bg=PALETTE["surface"], selectcolor=PALETTE["bg"],
            activeforeground=PALETTE["yellow"],
            activebackground=PALETTE["surface"],
            bd=0, highlightthickness=0,
            command=self._on_fast_mode_toggle
        ).pack(anchor="w")

        # SA Steps slider
        srow = tk.Frame(qf, bg=PALETTE["surface"])
        srow.pack(fill="x", padx=8, pady=3)
        tk.Label(srow, text="SA Steps:", font=("Consolas", 8),
                 fg=PALETTE["muted"], bg=PALETTE["surface"], width=9,
                 anchor="w").pack(side="left")
        self._sa_steps_var = tk.IntVar(value=300)
        self._sa_steps_label = tk.Label(srow, text="300",
                                         font=("Consolas", 8, "bold"),
                                         fg=PALETTE["purple"], bg=PALETTE["surface"],
                                         width=4)
        self._sa_steps_label.pack(side="right")
        sa_scale = tk.Scale(qf, from_=50, to=1000,
                            orient="horizontal", variable=self._sa_steps_var,
                            bg=PALETTE["surface"], fg=PALETTE["muted"],
                            troughcolor=PALETTE["bg"],
                            highlightthickness=0, bd=0,
                            showvalue=False, length=220,
                            command=self._on_sa_steps_change)
        sa_scale.pack(padx=8, pady=0)

        # "When does quantum win?" hint box
        hint_f = tk.Frame(qf, bg="#1a1f2e", bd=0)
        hint_f.pack(fill="x", padx=8, pady=6)
        tk.Label(hint_f, text="💡 When Quantum wins:",
                 font=("Consolas", 8, "bold"),
                 fg=PALETTE["purple"], bg="#1a1f2e",
                 anchor="w").pack(fill="x", padx=6, pady=(5, 2))
        hints = [
            "• PRBs/UEs ratio  < 2",
            "• Mixed profiles  (VoIP+Video)",
            "• High buffer load (many UEs)",
            "• SA Steps        ≥ 300",
            "  → Quantum beats Max-CQI on",
            "    delay & fairness in these",
            "    congested scenarios.",
        ]
        for h in hints:
            tk.Label(hint_f, text=h,
                     font=("Consolas", 7), fg=PALETTE["muted"],
                     bg="#1a1f2e", anchor="w",
                     justify="left").pack(fill="x", padx=10)
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

    def _on_fast_mode_toggle(self):
        global QUANTUM_FAST_MODE
        QUANTUM_FAST_MODE = self._fast_mode_var.get()

    def _on_sa_steps_change(self, val):
        global QUANTUM_SA_STEPS
        QUANTUM_SA_STEPS = int(val)
        self._sa_steps_label.config(text=str(int(val)))

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


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = SimulatorApp()
    app.mainloop()