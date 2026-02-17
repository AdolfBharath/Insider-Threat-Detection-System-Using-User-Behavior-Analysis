from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SequenceBatch:
    X: np.ndarray  # shape: (n, L, d)
    meta: list[dict[str, Any]]  # per-sequence metadata incl. original events


def build_user_sequences(
    events: list[dict[str, Any]],
    vectors: list[np.ndarray],
    seq_len: int,
    stride: int,
) -> SequenceBatch:
    # Events must be in the same order as vectors.
    by_user: dict[str, list[int]] = {}
    for idx, e in enumerate(events):
        by_user.setdefault(str(e.get("user") or "unknown"), []).append(idx)

    X_list: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    for user, indices in by_user.items():
        # Sort by timestamp (lexicographic ISO works for Z timestamps)
        indices_sorted = sorted(indices, key=lambda i: str(events[i].get("ts") or ""))
        if len(indices_sorted) < seq_len:
            continue

        for start in range(0, len(indices_sorted) - seq_len + 1, stride):
            win = indices_sorted[start : start + seq_len]
            seq_vec = np.stack([vectors[i] for i in win], axis=0)
            seq_events = [events[i] for i in win]

            X_list.append(seq_vec)
            meta.append(
                {
                    "user": user,
                    "seq_len": seq_len,
                    "start_ts": seq_events[0].get("ts"),
                    "end_ts": seq_events[-1].get("ts"),
                    "events": seq_events,
                }
            )

    if not X_list:
        return SequenceBatch(X=np.zeros((0, seq_len, 0), dtype=float), meta=[])

    X = np.stack(X_list, axis=0)
    return SequenceBatch(X=X, meta=meta)
