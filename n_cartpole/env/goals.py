"""Target configurations for goal-conditioned control (transitions between poses).

Each link is either upright (target angle 0) or hanging down (target angle π), so
an ``n``-link rig has ``2**n`` equilibrium configurations. A goal is identified by
its index into :func:`goal_configs` or by a label string with one character per
link, base link first: ``"U"`` = up, ``"D"`` = down. For the double link:

    ``UU`` both up · ``DU`` link 1 down, link 2 up · ``UD`` · ``DD`` both down

Only ``DD…D`` is passively stable; every other configuration is an unstable
equilibrium the policy must actively balance.

:class:`TransitionStats` tracks per ``(from, to)`` success so the environment can
oversample the transitions that still fail (an automatic curriculum) and the
trainers can log a success matrix.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

# "from" index used when a segment did not start at a settled equilibrium
# (random reset, or the previous goal was never reached).
FROM_OTHER = -1


def goal_configs(n_links: int) -> np.ndarray:
    """Return all ``2**n`` target-angle vectors, shape ``(2**n, n)``.

    Row ``i`` encodes link ``k`` as down (π) when bit ``k`` of ``i`` is set, so
    index 0 is all-up and the last index is all-down.
    """
    idx = np.arange(2**n_links)[:, None]
    bits = (idx >> np.arange(n_links)[None, :]) & 1
    return bits.astype(np.float64) * np.pi


def goal_labels(n_links: int) -> list[str]:
    """Return the ``U``/``D`` label for each row of :func:`goal_configs`."""
    return [
        "".join("D" if (i >> k) & 1 else "U" for k in range(n_links))
        for i in range(2**n_links)
    ]


def goal_index(goal: int | str, n_links: int) -> int:
    """Resolve a goal index or ``U``/``D`` label to an index into the configs."""
    if isinstance(goal, str):
        label = goal.strip().upper()
        if len(label) != n_links or set(label) - {"U", "D"}:
            raise ValueError(
                f"goal label {goal!r} must be {n_links} characters of U/D "
                f"(one per link, base link first)"
            )
        return sum(1 << k for k, c in enumerate(label) if c == "D")
    if not 0 <= int(goal) < 2**n_links:
        raise ValueError(f"goal index {goal} out of range for {n_links} links")
    return int(goal)


def parse_goal_schedule(spec: str, n_links: int) -> list[tuple[float, int]]:
    """Parse ``"UU@0,DU@5,DD@10"`` into sorted ``[(time_s, goal_index), ...]``.

    A bare label without ``@t`` is only allowed as the first entry (time 0).
    """
    out: list[tuple[float, int]] = []
    for i, part in enumerate(p for p in spec.split(",") if p.strip()):
        label, _, t = part.partition("@")
        if not t and i > 0:
            raise ValueError(f"goal {part!r} needs a time, e.g. {label}@5")
        out.append((float(t) if t else 0.0, goal_index(label, n_links)))
    if not out:
        raise ValueError("empty goal schedule")
    return sorted(out)


def parse_goal_weights(spec: str, n_links: int) -> tuple[tuple[float, ...], ...]:
    """Parse ``"UU=2,DD>UU=3"`` into a goal-sampling weight table.

    ``TO=w`` scales every transition into ``TO`` (also from a random start);
    ``FROM>TO=w`` scales one transition. Entries multiply and unlisted ones stay
    1. The table is indexed ``[from + 1][to]`` like :class:`TransitionStats`, so
    row 0 is :data:`FROM_OTHER`.
    """
    n_goals = 2**n_links
    w = np.ones((n_goals + 1, n_goals))
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        key, eq, value = part.partition("=")
        if not eq:
            raise ValueError(f"goal weight {part!r} needs a value, e.g. UU=2")
        weight = float(value)
        if weight < 0:
            raise ValueError(f"goal weight {part!r} must be >= 0")
        frm, arrow, to = key.partition(">")
        if arrow:
            w[goal_index(frm, n_links) + 1, goal_index(to, n_links)] *= weight
        else:
            w[:, goal_index(key, n_links)] *= weight
    if not (w.sum(axis=1) > 0).all():
        raise ValueError(f"goal weights {spec!r} leave some start with no goal")
    return tuple(tuple(float(x) for x in row) for row in w)


class TransitionStats:
    """Success statistics per ``(from, to)`` transition.

    Rows are indexed by ``from + 1`` so :data:`FROM_OTHER` (``-1``) maps to row 0.
    ``ema`` is an exponential moving average of success used for curriculum
    sampling; ``count``/``success`` accumulate raw totals for logging.
    """

    def __init__(self, n_goals: int, ema_rate: float = 0.05) -> None:
        """Start with no successes (uniform sampling until data arrives)."""
        self.n_goals = n_goals
        self.ema_rate = ema_rate
        self.ema = np.zeros((n_goals + 1, n_goals))
        self.count = np.zeros((n_goals + 1, n_goals), dtype=np.int64)
        self.success = np.zeros((n_goals + 1, n_goals), dtype=np.int64)

    def record(self, frm: int, to: int, ok: bool) -> None:
        """Record the outcome of one segment."""
        r = frm + 1
        self.ema[r, to] += self.ema_rate * (float(ok) - self.ema[r, to])
        self.count[r, to] += 1
        self.success[r, to] += int(ok)

    def extend(self, segments: Iterable[tuple[int, int, bool]]) -> None:
        """Record many ``(from, to, ok)`` segments."""
        for frm, to, ok in segments:
            self.record(frm, to, ok)

    def sample_weights(self, frm: int, floor: float = 0.2) -> np.ndarray:
        """Goal-sampling probabilities from ``frm``, favoring failing transitions."""
        w = floor + (1.0 - self.ema[frm + 1])
        return w / w.sum()

    def success_rate(self) -> float:
        """Overall success fraction (NaN when nothing has been recorded)."""
        n = self.count.sum()
        return float(self.success.sum() / n) if n else float("nan")

    def format_matrix(self, labels: Sequence[str]) -> str:
        """Render the ``from`` by ``to`` success matrix as a fixed-width table."""
        rows = ["random", *labels]
        w = max(6, *(len(s) for s in labels))
        head = "from\\to".ljust(8) + "".join(s.rjust(w + 2) for s in labels)
        lines = [head]
        for r, name in enumerate(rows):
            if r == 0 and not self.count[0].any():
                continue  # no segments started from a random state
            cells = []
            for c in range(self.n_goals):
                n = self.count[r, c]
                cells.append(
                    (f"{self.success[r, c] / n:.0%}" if n else "-").rjust(w + 2)
                )
            lines.append(name.ljust(8) + "".join(cells))
        return "\n".join(lines)
