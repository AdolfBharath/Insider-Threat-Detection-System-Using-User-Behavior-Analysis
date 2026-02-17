from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FeatureSpec:
    action_vocab: list[str]
    resource_vocab: list[str]
    median: list[float]
    iqr: list[float]
    feature_names: list[str]


def _safe_iqr(x: np.ndarray) -> float:
    q75 = float(np.percentile(x, 75))
    q25 = float(np.percentile(x, 25))
    v = q75 - q25
    return v if v > 1e-9 else 1.0


def _extract_resource_type(e: dict) -> str:
    # If upstream log already has resource_type, prefer it.
    rt = e.get("resource_type")
    if isinstance(rt, str) and rt:
        rt = rt.lower()
        if rt in {"file", "api", "database"}:
            return rt

    # Heuristic for this repo's demo logs.
    # app_access is typically file/api access; infer database if resource path looks like export/db.
    resource = str(e.get("resource") or "").lower()
    if "db" in resource or "export" in resource or "customers" in resource:
        return "database"
    if resource.startswith("/api/") or "api" in resource:
        return "api"

    return "file"


def _extract_action(e: dict) -> str:
    # Normalize actions into a small stable vocabulary.
    a = str(e.get("action") or "unknown").lower()
    if a in {"view", "download", "exec", "login", "query"}:
        return a

    # Map project-specific actions.
    if "login" in a:
        return "login"
    if "sudo" in a or "command" in a or e.get("event_type") == "sudo":
        return "exec"

    return "view" if e.get("event_type") == "app_access" else "exec"


def _extract_privilege(e: dict) -> float:
    # privilege indicator: explicit field or inferred from sudo.
    p = e.get("privilege")
    if isinstance(p, (int, float)):
        return 1.0 if float(p) > 0 else 0.0
    return 1.0 if e.get("event_type") == "sudo" else 0.0


def _extract_success(e: dict) -> float:
    s = str(e.get("status") or "").lower()
    if s in {"success", "ok", "200"}:
        return 1.0
    if s in {"fail", "failed", "403", "401"}:
        return 0.0
    # fallback: interpret numeric http status
    extra = e.get("extra") or {}
    http = extra.get("http_status") if isinstance(extra, dict) else None
    if isinstance(http, str) and http.startswith("2"):
        return 1.0
    return 0.0


def _extract_bytes(e: dict) -> float:
    b = e.get("bytes")
    if isinstance(b, int) and b >= 0:
        return float(b)
    if isinstance(b, str) and b.isdigit():
        return float(int(b))
    return 0.0


def _extract_time_features(ts_iso: str) -> tuple[float, float]:
    # ts expected ISO-8601 with Z; only uses hour-of-day.
    try:
        # Fast parse: YYYY-MM-DDTHH:...
        hour = int(ts_iso[11:13])
    except Exception:
        hour = 0

    angle = 2.0 * math.pi * (hour / 24.0)
    return (math.sin(angle), math.cos(angle))


def build_feature_spec(events: list[dict[str, Any]]) -> FeatureSpec:
    actions = sorted({ _extract_action(e) for e in events })
    resources = sorted({ _extract_resource_type(e) for e in events })

    feature_names: list[str] = []
    feature_names += [f"action:{a}" for a in actions]
    feature_names += [f"resource:{r}" for r in resources]
    feature_names += ["bytes_log", "success", "privilege", "hour_sin", "hour_cos"]

    # numeric stats for robust scaling
    bytes_logs = []
    success_vals = []
    priv_vals = []
    hsin = []
    hcos = []

    for e in events:
        b = _extract_bytes(e)
        bytes_logs.append(math.log1p(b))
        success_vals.append(_extract_success(e))
        priv_vals.append(_extract_privilege(e))
        s, c = _extract_time_features(str(e.get("ts") or ""))
        hsin.append(s)
        hcos.append(c)

    numeric = np.array([bytes_logs, success_vals, priv_vals, hsin, hcos], dtype=float).T
    median = np.median(numeric, axis=0)
    iqr = np.array([_safe_iqr(numeric[:, i]) for i in range(numeric.shape[1])], dtype=float)

    return FeatureSpec(
        action_vocab=actions,
        resource_vocab=resources,
        median=[float(x) for x in median],
        iqr=[float(x) for x in iqr],
        feature_names=feature_names,
    )


def event_to_vector(e: dict, spec: FeatureSpec) -> np.ndarray:
    a = _extract_action(e)
    r = _extract_resource_type(e)

    action_onehot = np.zeros(len(spec.action_vocab), dtype=float)
    if a in spec.action_vocab:
        action_onehot[spec.action_vocab.index(a)] = 1.0

    resource_onehot = np.zeros(len(spec.resource_vocab), dtype=float)
    if r in spec.resource_vocab:
        resource_onehot[spec.resource_vocab.index(r)] = 1.0

    b = math.log1p(_extract_bytes(e))
    s = _extract_success(e)
    p = _extract_privilege(e)
    hour_sin, hour_cos = _extract_time_features(str(e.get("ts") or ""))

    numeric = np.array([b, s, p, hour_sin, hour_cos], dtype=float)
    numeric_scaled = (numeric - np.array(spec.median, dtype=float)) / np.array(spec.iqr, dtype=float)

    return np.concatenate([action_onehot, resource_onehot, numeric_scaled], axis=0)


def save_feature_spec(path: str, spec: FeatureSpec) -> None:
    payload = {
        "action_vocab": spec.action_vocab,
        "resource_vocab": spec.resource_vocab,
        "median": spec.median,
        "iqr": spec.iqr,
        "feature_names": spec.feature_names,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_feature_spec(path: str) -> FeatureSpec:
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    return FeatureSpec(
        action_vocab=list(p["action_vocab"]),
        resource_vocab=list(p["resource_vocab"]),
        median=list(p["median"]),
        iqr=list(p["iqr"]),
        feature_names=list(p["feature_names"]),
    )
