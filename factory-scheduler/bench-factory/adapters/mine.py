from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import (  # noqa: E402
    Scheduler as BaseScheduler,
    decode,
    dispatch,
    interleaved_job_sequence,
    schedule,
    schedule_cost,
)
from benchkit.rng import Rng  # noqa: E402
from data import Instance  # noqa: E402

SUBMIT_INTERVAL = 0.05
RESERVE = 0.15
MAX_RESTARTS = 5


class Scheduler(BaseScheduler):
    def prepare(self, instance: Instance, deadline: float, submit=None) -> None:
        self.inst = instance
        self.submit = submit if submit is not None else self._local_sink
        self.started = time.perf_counter()
        self.deadline = deadline
        self.last_submit = 0.0
        self.best_cost: Optional[int] = None
        self.best_sequences: Optional[List[List[int]]] = None
        self.rng = Rng(instance.digest() ^ 0xD15_9A77)

    def _local_sink(self, plan) -> dict:
        cost = schedule_cost(self.inst, plan["sequences"])
        return {
            "accepted": cost is not None,
            "reason": None if cost is not None else "deadlock",
            "cost": cost,
            "best": self.best_cost,
            "elapsed_s": self._elapsed(),
            "remaining_s": None,
        }

    def _remaining(self) -> float:
        return self.deadline - time.perf_counter()

    def _elapsed(self) -> float:
        return time.perf_counter() - self.started

    def _offer(self, sequences: Sequence[Sequence[int]], force: bool = True) -> None:
        now = self._elapsed()
        if not force and now - self.last_submit < SUBMIT_INTERVAL:
            return
        receipt = self.submit({"sequences": [list(s) for s in sequences]})
        self.last_submit = now
        if receipt.get("remaining_s") is not None:
            self.deadline = time.perf_counter() + receipt["remaining_s"]
        if receipt["accepted"]:
            cost = receipt["cost"]
            if self.best_cost is None or cost < self.best_cost:
                self.best_cost = cost
                self.best_sequences = [list(s) for s in sequences]

    def _instant(self) -> None:
        inst = self.inst
        order = sorted(range(inst.n_jobs), key=lambda j: (inst.due[j], inst.release[j], j))
        self._offer(decode(inst, interleaved_job_sequence(inst, order)))

    def _portfolio(self) -> None:
        inst = self.inst
        duration = inst.op_duration
        due = inst.due
        weight = inst.weight
        remaining_work = [sum(duration[op] for op in inst.job_ops[job]) for job in range(inst.n_jobs)]

        def edd(job, op, start):
            return (due[job], start, duration[op], job)

        def spt(job, op, start):
            return (duration[op], start, job)

        def non_delay(job, op, start):
            return (start, duration[op], job)

        def weighted_critical_ratio(job, op, start):
            slack = due[job] - start - remaining_work[job]
            return (slack * 100 // max(1, weight[job]), start, job)

        def wspt(job, op, start):
            return (-(weight[job] * 1000 // max(1, duration[op])), start, job)

        def tardy_hybrid(job, op, start):
            slack = due[job] - start - remaining_work[job]
            return (slack - 16 * weight[job], start + duration[op], job)

        def weighted_slack(job, op, start):
            slack = due[job] - start - remaining_work[job]
            return (slack * 1000 // max(1, weight[job] + 1), start, job)

        rules = [weighted_critical_ratio, edd, non_delay, wspt, tardy_hybrid, weighted_slack, spt]
        cutoff = self._remaining() / 2.0
        for rule in rules:
            if self._remaining() < max(RESERVE * 2, cutoff):
                break
            sequences = dispatch(self.inst, rule)
            self._offer(sequences)

        self.remaining_work = remaining_work

    def _descend_and_perturb(self) -> None:
        if self.best_sequences is None:
            return
        sequences = [list(s) for s in self.best_sequences]
        cost = self._descend(sequences, self.best_cost)
        self._offer(sequences)

        for _ in range(MAX_RESTARTS):
            if self._remaining() <= RESERVE * 2:
                break
            trial = [list(s) for s in (self.best_sequences or sequences)]
            if not self._perturb(trial):
                break
            trial_cost = self._descend(trial, schedule_cost(self.inst, trial))
            if trial_cost is not None and (cost is None or trial_cost < cost):
                sequences, cost = trial, trial_cost
                self._offer(sequences)

    def _descend(self, sequences: List[List[int]], cost: Optional[int]) -> Optional[int]:
        if cost is None:
            cost = schedule_cost(self.inst, sequences)
            if cost is None:
                return None

        improved = True
        while improved and self._remaining() > RESERVE:
            improved = False
            best_swap = None
            best_swap_cost = cost
            critical = self._critical_operations(sequences)
            for machine in range(self.inst.n_machines):
                sequence = sequences[machine]
                for position in range(len(sequence) - 1):
                    if critical and sequence[position] not in critical and sequence[position + 1] not in critical:
                        continue
                    if self._remaining() <= RESERVE:
                        return cost
                    sequence[position], sequence[position + 1] = (
                        sequence[position + 1],
                        sequence[position],
                    )
                    candidate = schedule_cost(self.inst, sequences)
                    if candidate is not None and candidate < best_swap_cost:
                        best_swap = (machine, position)
                        best_swap_cost = candidate
                    sequence[position], sequence[position + 1] = (
                        sequence[position + 1],
                        sequence[position],
                    )
            if best_swap is not None:
                machine, position = best_swap
                sequences[machine][position], sequences[machine][position + 1] = (
                    sequences[machine][position + 1],
                    sequences[machine][position],
                )
                cost = best_swap_cost
                self._offer(sequences, force=False)
                improved = True
        return cost

    def _critical_operations(self, sequences: Sequence[Sequence[int]]) -> set[int]:
        result = schedule(self.inst, sequences)
        if result is None:
            return set()

        finish = result["finish"]
        starts = result["starts"]
        successors = [[] for _ in range(self.inst.total_ops)]
        for ops in self.inst.job_ops:
            for left, right in zip(ops, ops[1:]):
                successors[left].append((right, 0))
        for machine, ops in enumerate(sequences):
            for left, right in zip(ops, ops[1:]):
                setup = 0
                if machine in self.inst._setup_set:
                    setup = self.inst.setup_matrix[
                        self.inst.op_group[left]
                    ][self.inst.op_group[right]]
                successors[left].append((right, setup))

        latest_finish = [result["makespan"]] * self.inst.total_ops
        for op in sorted(range(self.inst.total_ops), key=finish.__getitem__, reverse=True):
            for successor, setup in successors[op]:
                limit = starts[successor] - setup
                if limit < latest_finish[op]:
                    latest_finish[op] = limit

        return {
            op
            for op in range(self.inst.total_ops)
            if starts[op] == latest_finish[op] - self.inst.op_duration[op]
        }

    def _perturb(self, sequences: List[List[int]]) -> bool:
        candidates = [m for m in range(self.inst.n_machines) if len(sequences[m]) > 1]
        if not candidates:
            return False
        for _ in range(self.rng.randint(2, 7)):
            machine = self.rng.choice(candidates)
            sequence = sequences[machine]
            position = self.rng.below(len(sequence) - 1)
            sequence[position], sequence[position + 1] = (
                sequence[position + 1],
                sequence[position],
            )
        return True

    def _late_job_rescue(self) -> None:
        if self.best_sequences is None:
            return
        sequences = [list(s) for s in self.best_sequences]
        cost = schedule_cost(self.inst, sequences)
        if cost is None:
            return

        while self._remaining() > RESERVE:
            best_move = None
            best_score = cost
            current = schedule(self.inst, sequences)
            if current is None:
                return
            late_jobs = [
                j
                for j in range(self.inst.n_jobs)
                if current["completion"][j] > self.inst.due[j]
            ]
            late_jobs.sort(
                key=lambda j: (
                    self.inst.weight[j]
                    * (current["completion"][j] - self.inst.due[j]),
                    current["completion"][j] - self.inst.due[j],
                ),
                reverse=True,
            )
            if not late_jobs:
                late_jobs = sorted(
                    range(self.inst.n_jobs),
                    key=lambda j: (self.inst.due[j], -self.inst.weight[j], j),
                )[: max(1, self.inst.n_jobs // 4)]

            for job in late_jobs:
                ops = self.inst.job_ops[job]
                for op in ops:
                    machine = self.inst.op_machine[op]
                    seq = sequences[machine]
                    pos = seq.index(op)
                    for shift in range(pos):
                        if self._remaining() <= RESERVE:
                            return
                        seq.pop(pos)
                        seq.insert(shift, op)
                        candidate = schedule_cost(self.inst, sequences)
                        if candidate is not None and candidate < best_score:
                            best_move = (machine, pos, shift)
                            best_score = candidate
                        seq.pop(shift)
                        seq.insert(pos, op)
            if best_move is None:
                break
            machine, pos, shift = best_move
            seq = sequences[machine]
            op = seq[pos]
            seq.pop(pos)
            seq.insert(shift, op)
            cost = best_score
            self._offer(sequences, force=False)

    def build(self):
        self._instant()
        self._portfolio()
        self._descend_and_perturb()
        self._late_job_rescue()
        return self.best_sequences, self.best_cost

    def solve(self, instance: Instance, submit_candidate) -> object:
        self.prepare(instance, time.perf_counter() + 1e9, submit_candidate)
        self.build()
        return {"sequences": self.best_sequences} if self.best_sequences else None


Scheduler_ = Scheduler
