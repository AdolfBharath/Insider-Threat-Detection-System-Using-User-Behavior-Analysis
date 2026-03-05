from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


_SPEC_VERSION = 2


@dataclass
class FeatureSpec:
    # NOTE: Versioned to support safe upgrades without breaking older artifacts.
    version: int
    action_vocab: list[str]
    resource_vocab: list[str]
    role_vocab: list[str]
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


def _extract_hour(ts_iso: str) -> int:
    try:
        return int(ts_iso[11:13])
    except Exception:
        return 0


def _working_hours_from_cfg(cfg: dict) -> tuple[int, int]:
    # Improvement #5: use configured working hours to compute time anomaly flags.
    rules = cfg.get("rules", {}) if isinstance(cfg, dict) else {}
    wh = rules.get("working_hours", {}) if isinstance(rules, dict) else {}

    def _h(x: Any, default: int) -> int:
        try:
            return int(str(x).split(":", 1)[0])
        except Exception:
            return default

    return _h(wh.get("start", "09:00"), 9), _h(wh.get("end", "18:00"), 18)


def resolve_user_role(cfg: dict, user: str) -> str:
    # Improvement #1: incorporate roles (admin/manager/employee) into behavior modeling.
    roles = cfg.get("roles", {}) if isinstance(cfg, dict) else {}
    default_role = str(roles.get("default", "employee"))
    users = roles.get("users", {}) if isinstance(roles.get("users"), dict) else {}
    role = str(users.get(user, default_role))
    return role


def _privileged_command_flag(e: dict) -> float:
    # Improvement #5: privileged command signal.
    if str(e.get("event_type") or "").lower() == "sudo":
        return 1.0
    cmd = str(e.get("resource") or "").lower()
    if any(x in cmd for x in ("/etc/shadow", "useradd", "usermod", "passwd", "chmod", "chown")):
        return 1.0
    return 0.0


def _large_transfer_flag(cfg: dict, e: dict) -> float:
    # Improvement #5: large data transfer indicator.
    rules = cfg.get("rules", {}) if isinstance(cfg, dict) else {}
    mb = float(rules.get("large_data_threshold_mb", 200)) if isinstance(rules, dict) else 200.0
    threshold = int(mb * 1024 * 1024)
    b = e.get("bytes")
    return 1.0 if isinstance(b, int) and b >= threshold else 0.0


def _after_hours_flag(cfg: dict, ts_iso: str) -> float:
    start_h, end_h = _working_hours_from_cfg(cfg)
    h = _extract_hour(ts_iso)
    return 1.0 if (h < start_h or h >= end_h) else 0.0


def _hour_deviation(cfg: dict, ts_iso: str) -> float:
    # Improvement #5: login hour anomaly (distance from working-hours midpoint).
    start_h, end_h = _working_hours_from_cfg(cfg)
    mid = (start_h + end_h) / 2.0
    h = float(_extract_hour(ts_iso))
    # Normalize to ~0..1 by dividing by 12 hours.
    return min(1.0, abs(h - mid) / 12.0)


def build_feature_spec(events: list[dict[str, Any]], cfg: dict | None = None) -> FeatureSpec:
    """Build a versioned feature spec.

    Improvements:
    - #1 Role-based behavioral modeling (role one-hot in vectors)
    - #5 Enhanced feature engineering (time anomaly, failure burst, large transfer, privileged command, novelty)
    """
    cfg = cfg or {}
    actions = sorted({_extract_action(e) for e in events})
    resources = sorted({_extract_resource_type(e) for e in events})

    roles_cfg = cfg.get("roles", {}) if isinstance(cfg, dict) else {}
    roles_list = roles_cfg.get("list") if isinstance(roles_cfg, dict) else None
    if isinstance(roles_list, list) and roles_list:
        roles = sorted({str(x) for x in roles_list})
    else:
        # fall back to whatever appears in data/mapping
        roles = sorted({resolve_user_role(cfg, str(e.get("user") or "unknown")) for e in events} | {"employee"})

    feature_names: list[str] = []
    feature_names += [f"action:{a}" for a in actions]
    feature_names += [f"resource:{r}" for r in resources]
    feature_names += [f"role:{r}" for r in roles]
    feature_names += [
        # Base numeric features
        "bytes_log",
        "success",
        "privilege",
        "hour_sin",
        "hour_cos",
        # Enhanced numeric features
        "after_hours",
        "fail_burst_10",
        "large_transfer",
        "privileged_command",
        "resource_novelty",
        "hour_deviation",
    ]

    # numeric stats for robust scaling
    bytes_logs = []
    success_vals = []
    priv_vals = []
    hsin = []
    hcos = []

    after_hours = []
    fail_burst_10 = []
    large_transfer = []
    priv_cmd = []
    res_novel = []
    hour_dev = []

    # stateful signals
    recent_failures: dict[str, deque[int]] = {}
    seen_resources: dict[str, set[str]] = {}

    for e in events:
        user = str(e.get("user") or "unknown")
        b = _extract_bytes(e)
        bytes_logs.append(math.log1p(b))
        success_vals.append(_extract_success(e))
        priv_vals.append(_extract_privilege(e))
        s, c = _extract_time_features(str(e.get("ts") or ""))
        hsin.append(s)
        hcos.append(c)

        ts = str(e.get("ts") or "")
        after_hours.append(_after_hours_flag(cfg, ts))

        dq = recent_failures.setdefault(user, deque(maxlen=10))
        dq.append(1 if _extract_success(e) == 0.0 else 0)
        fail_burst_10.append(float(sum(dq)) / 10.0)

        large_transfer.append(_large_transfer_flag(cfg, e))
        priv_cmd.append(_privileged_command_flag(e))

        r = str(e.get("resource") or "")
        sset = seen_resources.setdefault(user, set())
        res_novel.append(0.0 if (not r or r in sset) else 1.0)
        if r:
            sset.add(r)

        hour_dev.append(_hour_deviation(cfg, ts))

    numeric = np.array(
        [
            bytes_logs,
            success_vals,
            priv_vals,
            hsin,
            hcos,
            after_hours,
            fail_burst_10,
            large_transfer,
            priv_cmd,
            res_novel,
            hour_dev,
        ],
        dtype=float,
    ).T
    median = np.median(numeric, axis=0)
    iqr = np.array([_safe_iqr(numeric[:, i]) for i in range(numeric.shape[1])], dtype=float)

    return FeatureSpec(
        version=_SPEC_VERSION,
        action_vocab=actions,
        resource_vocab=resources,
        role_vocab=roles,
        median=[float(x) for x in median],
        iqr=[float(x) for x in iqr],
        feature_names=feature_names,
    )


class EventVectorizer:
    """Stateful event→vector transformer.

    Improvements:
    - #1 adds role one-hot
    - #5 adds behavioral signals requiring short-term state (fail burst, novelty)
    """

    def __init__(self, cfg: dict, spec: FeatureSpec):
        self._cfg = cfg
        self._spec = spec
        self._recent_failures: dict[str, deque[int]] = {}
        self._seen_resources: dict[str, set[str]] = {}

    def vectorize(self, e: dict) -> np.ndarray:
        user = str(e.get("user") or "unknown")
        role = str(e.get("role") or resolve_user_role(self._cfg, user))

        a = _extract_action(e)
        rtype = _extract_resource_type(e)

        action_onehot = np.zeros(len(self._spec.action_vocab), dtype=float)
        if a in self._spec.action_vocab:
            action_onehot[self._spec.action_vocab.index(a)] = 1.0

        resource_onehot = np.zeros(len(self._spec.resource_vocab), dtype=float)
        if rtype in self._spec.resource_vocab:
            resource_onehot[self._spec.resource_vocab.index(rtype)] = 1.0

        role_onehot = np.zeros(len(self._spec.role_vocab), dtype=float)
        if role in self._spec.role_vocab:
            role_onehot[self._spec.role_vocab.index(role)] = 1.0

        ts = str(e.get("ts") or "")
        b = math.log1p(_extract_bytes(e))
        success = _extract_success(e)
        priv = _extract_privilege(e)
        hour_sin, hour_cos = _extract_time_features(ts)
        after_hours = _after_hours_flag(self._cfg, ts)

        dq = self._recent_failures.setdefault(user, deque(maxlen=10))
        dq.append(1 if success == 0.0 else 0)
        fail_burst = float(sum(dq)) / 10.0

        large_transfer = _large_transfer_flag(self._cfg, e)
        priv_cmd = _privileged_command_flag(e)

        res = str(e.get("resource") or "")
        seen = self._seen_resources.setdefault(user, set())
        novelty = 0.0 if (not res or res in seen) else 1.0
        if res:
            seen.add(res)

        hour_dev = _hour_deviation(self._cfg, ts)

        numeric = np.array(
            [
                b,
                success,
                priv,
                hour_sin,
                hour_cos,
                after_hours,
                fail_burst,
                large_transfer,
                priv_cmd,
                novelty,
                hour_dev,
            ],
            dtype=float,
        )
        numeric_scaled = (numeric - np.array(self._spec.median, dtype=float)) / np.array(self._spec.iqr, dtype=float)

        return np.concatenate([action_onehot, resource_onehot, role_onehot, numeric_scaled], axis=0)


def event_to_vector(e: dict, spec: FeatureSpec) -> np.ndarray:
    """Backward-compatible stateless vectorization.

    NOTE: The pipeline uses `EventVectorizer` for improved features.
    """
    return EventVectorizer(cfg={}, spec=spec).vectorize(e)


def save_feature_spec(path: str, spec: FeatureSpec) -> None:
    payload = {
        "version": spec.version,
        "action_vocab": spec.action_vocab,
        "resource_vocab": spec.resource_vocab,
        "role_vocab": spec.role_vocab,
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
        version=int(p.get("version", 1)),
        action_vocab=list(p.get("action_vocab", [])),
        resource_vocab=list(p.get("resource_vocab", [])),
        role_vocab=list(p.get("role_vocab", ["employee"])),
        median=list(p.get("median", [])),
        iqr=list(p.get("iqr", [])),
        feature_names=list(p.get("feature_names", [])),
    )
