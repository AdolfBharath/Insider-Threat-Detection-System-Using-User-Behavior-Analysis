from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from itds.ml.feature_engineering import (
    EventVectorizer,
    FeatureSpec,
    build_feature_spec,
    load_feature_spec,
    resolve_user_role,
    save_feature_spec,
)
from itds.ml.lstm_autoencoder import build_lstm_autoencoder
from itds.ml.sequence_builder import SequenceBatch, build_user_sequences


@dataclass(frozen=True)
class ModelArtifacts:
    model_dir: Path
    spec_path: Path
    thresholds_path: Path
    adaptive_stats_path: Path
    train_meta_path: Path


@dataclass
class AdaptiveStats:
    """Adaptive reconstruction-error statistics (EWMA).

    Improvement #2: adaptive thresholds based on recent error stats.
    """

    mean: float
    var: float
    count: int


def _percentile(x: np.ndarray, p: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, p))


def get_artifacts(cfg: dict) -> ModelArtifacts:
    out_dir = Path(cfg.get("paths", {}).get("model_dir", "data/out/model"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ml_cfg = cfg.get("lstm_ae", {})
    seq_len = int(ml_cfg.get("seq_len", 10))
    return ModelArtifacts(
        # Keras 3 does not support `load_model()` from a TensorFlow SavedModel directory.
        # Use the native Keras v3 format instead.
        model_dir=out_dir / f"lstm_ae_L{seq_len}.keras",
        spec_path=out_dir / "feature_spec.json",
        thresholds_path=out_dir / f"thresholds_L{seq_len}.json",
        adaptive_stats_path=out_dir / f"adaptive_stats_L{seq_len}.json",
        train_meta_path=out_dir / f"train_meta_L{seq_len}.json",
    )


def _parse_ts(ts_iso: str) -> datetime:
    # ISO-8601 with Z; tolerant fallback.
    try:
        if ts_iso.endswith("Z"):
            ts_iso = ts_iso.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_iso).astimezone(timezone.utc)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_events_from_jsonl(path: Path) -> Any:
    """Yield normalized events from a JSONL file (streaming).

    Improvement #6: avoid loading all events into memory.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _load_events_grouped(cfg: dict, normalized_jsonl: Path) -> dict[str, list[dict[str, Any]]]:
    """Load events grouped by user with caps to bound memory.

    Improvement #6: memory control via per-user caps.
    """
    ml_cfg = cfg.get("lstm_ae", {})
    max_events_per_user = int(ml_cfg.get("max_events_per_user", 5000))

    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in _iter_events_from_jsonl(normalized_jsonl):
        user = str(e.get("user") or "unknown")
        # Attach role for role-based modeling (#1)
        e["role"] = str(e.get("role") or resolve_user_role(cfg, user))

        bucket = by_user[user]
        if len(bucket) < max_events_per_user:
            bucket.append(e)
        else:
            # Keep only the most recent events for that user
            bucket.pop(0)
            bucket.append(e)

    # Sort per-user chronologically
    for user, evs in by_user.items():
        evs.sort(key=lambda x: str(x.get("ts") or ""))
    return dict(by_user)


def _flatten_grouped(by_user: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for evs in by_user.values():
        out.extend(evs)
    # global order isn't critical for spec building
    return out


def _load_adaptive_stats(path: Path) -> dict[str, AdaptiveStats]:
    payload = _load_json(path) or {}
    out: dict[str, AdaptiveStats] = {}
    for role, v in payload.items():
        if not isinstance(v, dict):
            continue
        out[str(role)] = AdaptiveStats(
            mean=float(v.get("mean", 0.0)),
            var=float(v.get("var", 0.0)),
            count=int(v.get("count", 0)),
        )
    return out


def _save_adaptive_stats(path: Path, stats: dict[str, AdaptiveStats]) -> None:
    payload = {k: {"mean": v.mean, "var": v.var, "count": v.count} for k, v in stats.items()}
    _save_json(path, payload)


def _adaptive_thresholds_for_role(
    cfg: dict,
    role: str,
    base: dict[str, float],
    stats: dict[str, AdaptiveStats],
) -> tuple[float, float]:
    """Return (t_medium, t_high) for a role.

    Improvement #2: adaptive thresholds from recent reconstruction error.
    """
    ml_cfg = cfg.get("lstm_ae", {})
    adaptive = ml_cfg.get("adaptive_thresholds", {}) if isinstance(ml_cfg.get("adaptive_thresholds", {}), dict) else {}
    enabled = bool(adaptive.get("enabled", True))
    if not enabled:
        return float(base.get("t_medium", 0.0)), float(base.get("t_high", 0.0))

    min_count = int(adaptive.get("min_count", 20))
    k_med = float(adaptive.get("k_medium", 2.5))
    k_hi = float(adaptive.get("k_high", 3.5))
    min_std = float(adaptive.get("min_std", 1e-6))

    s = stats.get(role) or stats.get("global")
    if s is None or s.count < min_count:
        return float(base.get("t_medium", 0.0)), float(base.get("t_high", 0.0))

    std = math.sqrt(max(0.0, s.var))
    std = max(std, min_std)
    t_medium = max(float(base.get("t_medium", 0.0)), s.mean + k_med * std)
    t_high = max(float(base.get("t_high", 0.0)), s.mean + k_hi * std)
    return t_medium, t_high


def _update_adaptive_stats(cfg: dict, role: str, e: float, t_medium: float, stats: dict[str, AdaptiveStats]) -> None:
    """Update EWMA stats using only 'likely normal' errors (below medium threshold).

    Improvement #2: prevents drift from learning anomalies as normal.
    """
    ml_cfg = cfg.get("lstm_ae", {})
    adaptive = ml_cfg.get("adaptive_thresholds", {}) if isinstance(ml_cfg.get("adaptive_thresholds", {}), dict) else {}
    alpha = float(adaptive.get("alpha", 0.05))
    if e >= t_medium:
        return

    s = stats.get(role)
    if s is None:
        s = AdaptiveStats(mean=float(e), var=0.0, count=0)
        stats[role] = s

    # EWMA update
    prev_mean = s.mean
    s.mean = (1.0 - alpha) * s.mean + alpha * float(e)
    # Exponentially-weighted variance update around updated mean
    s.var = (1.0 - alpha) * s.var + alpha * (float(e) - prev_mean) ** 2
    s.count += 1

    # also maintain global stats
    g = stats.get("global")
    if g is None:
        stats["global"] = AdaptiveStats(mean=float(e), var=0.0, count=1)
    else:
        prev = g.mean
        g.mean = (1.0 - alpha) * g.mean + alpha * float(e)
        g.var = (1.0 - alpha) * g.var + alpha * (float(e) - prev) ** 2
        g.count += 1


def _load_thresholds(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_thresholds(path: Path, t_medium: float, t_high: float) -> None:
    payload = {"t_medium": float(t_medium), "t_high": float(t_high)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _train_val_split_by_time(batch: SequenceBatch, train_ratio: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    # sort by end_ts (ISO) for chronological split
    order = sorted(range(len(batch.meta)), key=lambda i: str(batch.meta[i].get("end_ts") or ""))
    n = len(order)
    n_train = max(1, int(n * train_ratio)) if n > 0 else 0

    train_idx = order[:n_train]
    val_idx = order[n_train:]

    X_train = batch.X[train_idx] if train_idx else np.zeros((0,) + batch.X.shape[1:], dtype=float)
    X_val = batch.X[val_idx] if val_idx else np.zeros((0,) + batch.X.shape[1:], dtype=float)
    return (X_train, X_val, train_idx)


def ensure_trained(cfg: dict, events: list[dict[str, Any]]) -> tuple[Any, FeatureSpec, dict[str, float]]:
    artifacts = get_artifacts(cfg)

    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError(
            "TensorFlow is required for LSTM Autoencoder mode. Install: pip install tensorflow"
        ) from e

    def _keras_model_present(model_path: Path) -> bool:
        return model_path.exists() and model_path.is_file() and model_path.suffix.lower() in {".keras", ".h5"}

    def _legacy_saved_model_present(model_dir: Path) -> bool:
        # Legacy TF SavedModel directory produced by `model.save(dir)`.
        return model_dir.exists() and model_dir.is_dir() and (model_dir / "saved_model.pb").exists()

    if _keras_model_present(artifacts.model_dir) and artifacts.spec_path.exists() and artifacts.thresholds_path.exists():
        try:
            model = tf.keras.models.load_model(str(artifacts.model_dir))
            spec = load_feature_spec(str(artifacts.spec_path))
            thresholds = _load_thresholds(artifacts.thresholds_path) or {"t_medium": 0.0, "t_high": 0.0}
        except Exception:
            # Model serialization mismatch (e.g., Keras version change). Retrain from scratch.
            try:
                artifacts.model_dir.unlink()
            except Exception:
                pass
            try:
                artifacts.spec_path.unlink()
            except Exception:
                pass
            try:
                artifacts.thresholds_path.unlink()
            except Exception:
                pass
            try:
                artifacts.adaptive_stats_path.unlink()
            except Exception:
                pass
        else:

            # Improvement #4: incremental learning metadata (periodic retraining).
            meta = _load_json(artifacts.train_meta_path) or {"run_count": 0, "last_trained_ts": None}
            meta["run_count"] = int(meta.get("run_count", 0)) + 1
            _save_json(artifacts.train_meta_path, meta)

            # Improvement #5 (robustness): if the feature spec changed, the saved model may be incompatible.
            # Detect mismatch and force retrain.
            need_retrain = False
            if int(getattr(spec, "version", 1)) < 2 or not getattr(spec, "role_vocab", None):
                need_retrain = True
            else:
                try:
                    model_dim = int(model.input_shape[-1])
                    if events:
                        vec_dim = int(EventVectorizer(cfg, spec).vectorize(events[0]).shape[0])
                        if vec_dim != model_dim:
                            need_retrain = True
                except Exception:
                    need_retrain = True

            if not need_retrain:
                # If thresholds were written during a low-data fallback run (0.0), recompute
                # them from current data so scoring can produce alerts.
                if float(thresholds.get("t_medium", 0.0)) == 0.0 and float(thresholds.get("t_high", 0.0)) == 0.0:
                    ml_cfg = cfg.get("lstm_ae", {})
                    seq_len = int(ml_cfg.get("seq_len", 10))
                    stride = int(ml_cfg.get("stride", 1))

                    vectorizer = EventVectorizer(cfg, spec)
                    vectors = [vectorizer.vectorize(e) for e in events]
                    batch = build_user_sequences(events, vectors, seq_len=seq_len, stride=stride)

                    if batch.X.shape[0] >= 4:
                        Xhat_all = model.predict(batch.X, verbose=0)
                        err_all = np.mean((batch.X - Xhat_all) ** 2, axis=(1, 2))
                        t_med_p = float(ml_cfg.get("t_medium_percentile", 95))
                        t_hi_p = float(ml_cfg.get("t_high_percentile", 99))
                        t_medium = _percentile(err_all, t_med_p)
                        t_high = _percentile(err_all, t_hi_p)
                        _save_thresholds(artifacts.thresholds_path, t_medium=t_medium, t_high=t_high)
                        thresholds = {"t_medium": float(t_medium), "t_high": float(t_high)}

                # Optional retraining hook (periodic)
                retrain_cfg = (
                    cfg.get("lstm_ae", {}).get("retrain", {})
                    if isinstance(cfg.get("lstm_ae", {}).get("retrain", {}), dict)
                    else {}
                )
                if bool(retrain_cfg.get("enabled", False)):
                    every = int(retrain_cfg.get("every_runs", 10))
                    if every > 0 and int(meta.get("run_count", 0)) % every == 0:
                        vectors = [EventVectorizer(cfg, spec).vectorize(e) for e in events]
                        ml_cfg = cfg.get("lstm_ae", {})
                        seq_len = int(ml_cfg.get("seq_len", 10))
                        stride = int(ml_cfg.get("stride", 1))
                        batch = build_user_sequences(events, vectors, seq_len=seq_len, stride=stride)
                        min_seqs = int(retrain_cfg.get("min_sequences", 50))
                        if batch.X.shape[0] >= min_seqs:
                            epochs = int(retrain_cfg.get("epochs", 3))
                            bs = int(retrain_cfg.get("batch_size", 64))
                            model.fit(batch.X, batch.X, epochs=epochs, batch_size=bs, verbose=0)
                            model.save(str(artifacts.model_dir))

                            # refresh base thresholds after retrain
                            Xhat = model.predict(batch.X, verbose=0)
                            err = np.mean((batch.X - Xhat) ** 2, axis=(1, 2))
                            t_med_p = float(ml_cfg.get("t_medium_percentile", 95))
                            t_hi_p = float(ml_cfg.get("t_high_percentile", 99))
                            t_medium = _percentile(err, t_med_p)
                            t_high = _percentile(err, t_hi_p)
                            _save_thresholds(artifacts.thresholds_path, t_medium=t_medium, t_high=t_high)
                            thresholds = {"t_medium": float(t_medium), "t_high": float(t_high)}
                            meta["last_trained_ts"] = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
                            _save_json(artifacts.train_meta_path, meta)

                return model, spec, thresholds

        # Retrain path: feature spec/model incompatible.
        try:
            if artifacts.model_dir.exists():
                if artifacts.model_dir.is_dir():
                    shutil.rmtree(artifacts.model_dir)
                else:
                    artifacts.model_dir.unlink()
        except Exception:
            pass

        # Clear adaptive-threshold state from the old feature space.
        try:
            artifacts.adaptive_stats_path.unlink()
        except Exception:
            pass

    # Fit feature spec on available (assumed normal) events
    spec = build_feature_spec(events, cfg=cfg)
    save_feature_spec(str(artifacts.spec_path), spec)

    vectorizer = EventVectorizer(cfg, spec)
    vectors = [vectorizer.vectorize(e) for e in events]

    ml_cfg = cfg.get("lstm_ae", {})
    seq_len = int(ml_cfg.get("seq_len", 10))
    stride = int(ml_cfg.get("stride", 1))
    batch = build_user_sequences(events, vectors, seq_len=seq_len, stride=stride)

    if batch.X.shape[0] < 4:
        # Not enough sequences; still create a model artifact so pipeline can run.
        model = build_lstm_autoencoder(seq_len=seq_len, feature_dim=vectors[0].shape[0] if vectors else 1)
        artifacts.model_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(artifacts.model_dir))
        _save_thresholds(artifacts.thresholds_path, t_medium=0.0, t_high=0.0)
        _save_json(artifacts.train_meta_path, {"run_count": 1, "last_trained_ts": None})
        return model, spec, {"t_medium": 0.0, "t_high": 0.0}

    train_ratio = float(ml_cfg.get("train_ratio", 0.7))
    X_train, X_val, train_idx = _train_val_split_by_time(batch, train_ratio=train_ratio)

    feature_dim = int(batch.X.shape[2])
    model = build_lstm_autoencoder(seq_len=seq_len, feature_dim=feature_dim, latent_dim=int(ml_cfg.get("latent_dim", 32)))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(ml_cfg.get("patience", 5)), restore_best_weights=True),
    ]

    epochs = int(ml_cfg.get("epochs", 25))
    batch_size = int(ml_cfg.get("batch_size", 64))

    # Train on normal sequences only
    model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val) if X_val.shape[0] else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    # Thresholds learned from training reconstruction errors (normal)
    Xhat_train = model.predict(X_train, verbose=0)
    train_err = np.mean((X_train - Xhat_train) ** 2, axis=(1, 2))

    t_med_p = float(ml_cfg.get("t_medium_percentile", 95))
    t_hi_p = float(ml_cfg.get("t_high_percentile", 99))
    t_medium = _percentile(train_err, t_med_p)
    t_high = _percentile(train_err, t_hi_p)

    artifacts.model_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(artifacts.model_dir))
    _save_thresholds(artifacts.thresholds_path, t_medium=t_medium, t_high=t_high)

    _save_json(
        artifacts.train_meta_path,
        {"run_count": 1, "last_trained_ts": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")},
    )

    return model, spec, {"t_medium": float(t_medium), "t_high": float(t_high)}


def score_sequences_from_jsonl(cfg: dict, normalized_jsonl: Path) -> list[dict[str, Any]]:
    """Score sequences using the on-disk normalized stream.

    Improvement #6: avoids building a giant `normalized_events` list in memory.
    """
    by_user = _load_events_grouped(cfg, normalized_jsonl)
    events = _flatten_grouped(by_user)
    return score_sequences(cfg, events)


def score_sequences(cfg: dict, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score per-user sequences and emit explainable alerts.

    Improvements:
    - #1 Role-based modeling (roles embedded into vectors + role-aware thresholds)
    - #2 Adaptive thresholds (EWMA stats by role)
    - #3 Improved risk scoring (recency + frequency + severity + categories)
    - #7 Alert prioritization (priority score + sorting)
    """

    model, spec, base_thresholds = ensure_trained(cfg, events)

    # Role-aware + enhanced vectors
    vectorizer = EventVectorizer(cfg, spec)
    vectors = [vectorizer.vectorize(e) for e in events]

    ml_cfg = cfg.get("lstm_ae", {})
    seq_len = int(ml_cfg.get("seq_len", 10))
    stride = int(ml_cfg.get("stride", 1))
    top_k = int(ml_cfg.get("explain_top_k", 3))

    batch = build_user_sequences(events, vectors, seq_len=seq_len, stride=stride)
    if batch.X.shape[0] == 0:
        return []

    # Some TF/Keras builds can spend a long time tracing/compiling predict graphs
    # on CPU. For this demo pipeline, an eager forward pass is sufficient.
    try:
        Xhat = model(batch.X, training=False).numpy()
    except Exception:
        Xhat = model.predict(batch.X, verbose=0)
    diff2 = (batch.X - Xhat) ** 2
    seq_err = np.mean(diff2, axis=(1, 2))

    # Adaptive thresholds state
    artifacts = get_artifacts(cfg)
    adaptive_stats = _load_adaptive_stats(artifacts.adaptive_stats_path)

    # Risk aggregation per user
    half_life_h = float(cfg.get("risk", {}).get("decay_half_life_hours", 24))
    risk_hi = float(cfg.get("risk", {}).get("threshold_high", 70))
    risk_med = float(cfg.get("risk", {}).get("threshold_medium", 45))

    # Improvement #3: category weights
    cat_weights = cfg.get("risk", {}).get("category_weights", {})
    if not isinstance(cat_weights, dict):
        cat_weights = {}

    # Maintain per-user risk state
    state: dict[str, dict[str, Any]] = {}

    # Low-level (early warning) alerting
    risk_cfg = cfg.get("risk", {}) if isinstance(cfg.get("risk", {}), dict) else {}
    emit_low_alerts = bool(risk_cfg.get("emit_low_alerts", False))
    risk_low = float(risk_cfg.get("threshold_low", 15))

    # Track per-user novelty across sequences for simple "new resource" warnings.
    seen_resources_by_user: dict[str, set[str]] = {}
    seen_ips_by_user: dict[str, set[str]] = {}

    alerts: list[dict[str, Any]] = []

    for i, meta in enumerate(batch.meta):
        user = str(meta.get("user") or "unknown")
        end_ts = str(meta.get("end_ts") or "")
        role = resolve_user_role(cfg, user)

        t_medium, t_high = _adaptive_thresholds_for_role(cfg, role, base_thresholds, adaptive_stats)

        # severity points from anomaly
        e = float(seq_err[i])
        if t_high > 0 and e >= t_high:
            severity = 80.0
            anomaly_level = "high"
        elif t_medium > 0 and e >= t_medium:
            severity = 50.0
            anomaly_level = "medium"
        else:
            severity = 0.0
            anomaly_level = "low"

        prev = state.get(user)
        end_dt = _parse_ts(end_ts)
        if prev is None:
            prev_risk = 0.0
            prev_dt = end_dt
            recent: deque[dict[str, Any]] = deque(maxlen=50)
        else:
            prev_risk = float(prev.get("risk", 0.0))
            prev_dt = prev.get("dt") or end_dt
            recent = prev.get("recent") or deque(maxlen=50)

        # Improvement #3: proper time-based decay (recency)
        dt_hours = max(0.0, (end_dt - prev_dt).total_seconds() / 3600.0)
        decay = 0.5 ** (dt_hours / max(1e-6, half_life_h))

        # Track anomaly frequency (recent window)
        freq_window_h = float(cfg.get("risk", {}).get("frequency_window_hours", 6))
        cutoff = end_dt.timestamp() - freq_window_h * 3600.0
        while recent and float(recent[0].get("t", 0.0)) < cutoff:
            recent.popleft()

        # Anomaly categories (for risk weighting)
        seq_events = meta.get("events") or []
        categories: list[str] = []
        # after-hours
        if any(str(ev.get("ts") or "") and (str(ev.get("ts")[11:13]).isdigit()) for ev in seq_events):
            # use end_ts hour for category
            h = int(end_ts[11:13]) if len(end_ts) >= 13 and end_ts[11:13].isdigit() else 0
            start_h = int(str(cfg.get("rules", {}).get("working_hours", {}).get("start", "09")).split(":", 1)[0])
            end_hh = int(str(cfg.get("rules", {}).get("working_hours", {}).get("end", "18")).split(":", 1)[0])
            if h < start_h or h >= end_hh:
                categories.append("after_hours")

        if any(str(ev.get("event_type") or "").lower() == "sudo" for ev in seq_events):
            categories.append("sudo")
            categories.append("privileged")

        # Admin/API access
        if any(
            str(ev.get("event_type") or "").lower() == "app_access"
            and (
                str(ev.get("resource") or "").lower().startswith("/api/admin")
                or "/api/admin" in str(ev.get("resource") or "").lower()
                or "admin/settings" in str(ev.get("resource") or "").lower()
            )
            for ev in seq_events
        ):
            categories.append("admin_api")

        mb = float(cfg.get("rules", {}).get("large_data_threshold_mb", 200))
        byte_threshold = int(mb * 1024 * 1024)
        if any(str(ev.get("action") or "").lower() == "download" and isinstance(ev.get("bytes"), int) and int(ev.get("bytes")) >= byte_threshold for ev in seq_events):
            categories.append("large_download")

        fails = sum(1 for ev in seq_events if str(ev.get("status") or "").lower() in {"fail", "failed"})
        if fails >= 3:
            categories.append("many_failures")
        elif fails >= 1:
            categories.append("some_failures")

        # Deny burst (403-style app denies clustered in a window)
        deny_burst_n = 0
        for ev in seq_events:
            if str(ev.get("event_type") or "").lower() != "app_access":
                continue
            extra = ev.get("extra") or {}
            http_status = str(extra.get("http_status") if isinstance(extra, dict) else "")
            if str(ev.get("status") or "").lower() in {"fail", "failed"} and http_status.startswith("403"):
                deny_burst_n += 1
        if deny_burst_n >= 3:
            categories.append("deny_burst")

        # New IP address for the user (within this run)
        # Only flag when the user already has a baseline IP observed earlier in this run.
        seen_ips = seen_ips_by_user.setdefault(user, set())
        ip_novel = False
        if seen_ips:
            for ev in seq_events:
                ip = str(ev.get("ip") or "").strip()
                if not ip:
                    continue
                if ip not in seen_ips:
                    ip_novel = True
        if ip_novel:
            categories.append("new_ip")
        for ev in seq_events:
            ip = str(ev.get("ip") or "").strip()
            if ip:
                seen_ips.add(ip)

        # Novel resource access (per user, within this run)
        # Only flag when the user already has a baseline resource observed earlier in this run.
        seen_res = seen_resources_by_user.setdefault(user, set())
        novel = False
        if seen_res:
            for ev in seq_events:
                r = str(ev.get("resource") or "").strip()
                if not r:
                    continue
                if r not in seen_res:
                    novel = True
        if novel:
            categories.append("novel_resource")
        for ev in seq_events:
            r = str(ev.get("resource") or "").strip()
            if r:
                seen_res.add(r)

        # Severity scaling (relative to threshold)
        threshold_used = t_high if anomaly_level == "high" else (t_medium if anomaly_level == "medium" else max(t_medium, t_high, 1e-9))
        severity_multiplier = min(2.0, float(e) / max(1e-9, float(threshold_used)))
        severity_points = float(severity) * severity_multiplier

        if anomaly_level in {"medium", "high"}:
            recent.append({"t": end_dt.timestamp(), "severity": severity_points, "cats": categories})

        # Frequency points
        anomaly_count = sum(1 for x in recent)
        freq_points = min(25.0, anomaly_count * 5.0)

        # Category points
        cat_points = 0.0
        for c in categories:
            try:
                cat_points += float(cat_weights.get(c, 0.0))
            except Exception:
                continue

        risk = max(0.0, min(100.0, prev_risk * decay + severity_points + freq_points + cat_points))
        state[user] = {"risk": risk, "dt": end_dt, "recent": recent}

        # Only alert on medium/high anomaly OR medium/high risk.
        level = "low"
        if risk >= risk_hi or anomaly_level == "high":
            level = "high"
        elif risk >= risk_med or anomaly_level == "medium":
            level = "medium"

        if level == "low":
            # Update adaptive stats only when sequence looks normal-ish.
            _update_adaptive_stats(cfg, role, e, t_medium, adaptive_stats)
            # Optional: still emit low-level alerts as early warnings.
            if not emit_low_alerts:
                continue
            if risk < risk_low and not categories:
                continue

        # Explainability (feature + timestep contributions)
        feat_err = np.mean(diff2[i], axis=0)  # (d,)
        time_err = np.mean(diff2[i], axis=1)  # (L,)

        top_feat_idx = list(np.argsort(-feat_err)[:top_k])
        top_time_idx = list(np.argsort(-time_err)[:top_k])

        top_features = [
            {"feature": spec.feature_names[j] if j < len(spec.feature_names) else f"f{j}", "mse": float(feat_err[j])}
            for j in top_feat_idx
        ]

        seq_events = meta.get("events") or []
        top_events = []
        for t in top_time_idx:
            if t < len(seq_events):
                ev = dict(seq_events[t])
                top_events.append(
                    {
                        "ts": ev.get("ts"),
                        "action": ev.get("action"),
                        "event_type": ev.get("event_type"),
                        "resource": ev.get("resource"),
                        "bytes": ev.get("bytes"),
                        "status": ev.get("status"),
                        "privilege": 1 if ev.get("event_type") == "sudo" else 0,
                        "timestep_mse": float(time_err[t]),
                    }
                )

        def _friendly_feature_name(name: str) -> str:
            if name == "hour_sin" or name == "hour_cos":
                return "unusual access time (hour-of-day)"
            if name == "bytes_log":
                return "unusual data volume"
            if name == "success":
                return "unusual success/failure pattern"
            if name == "privilege":
                return "privileged activity"
            if name.startswith("action:"):
                return f"action is {name.split(':', 1)[1]}"
            if name.startswith("resource:"):
                return f"resource type is {name.split(':', 1)[1]}"
            return name

        def _extract_hour(ts_iso: str) -> int | None:
            try:
                return int(ts_iso[11:13])
            except Exception:
                return None

        def _parse_hour_hhmm(hhmm: str) -> int | None:
            try:
                return int(str(hhmm).split(":", 1)[0])
            except Exception:
                return None

        def _make_reason_tags(cfg_: dict, seq_events_: list[dict[str, Any]], end_ts_: str) -> list[str]:
            tags: list[str] = []

            # After-hours check
            rules = cfg_.get("rules", {})
            wh = rules.get("working_hours", {}) if isinstance(rules, dict) else {}
            start_h = _parse_hour_hhmm(str(wh.get("start", "09:00")))
            end_h = _parse_hour_hhmm(str(wh.get("end", "18:00")))
            h = _extract_hour(end_ts_)
            if h is not None and start_h is not None and end_h is not None:
                if h < start_h or h >= end_h:
                    tags.append("After-hours")

            # Sudo / privilege escalation signals
            if any(str(ev.get("event_type") or "").lower() == "sudo" for ev in seq_events_):
                tags.append("sudo")
                tags.append("privileged")

            # Large download
            mb = float(rules.get("large_data_threshold_mb", 200)) if isinstance(rules, dict) else 200.0
            byte_threshold = int(mb * 1024 * 1024)
            for ev in seq_events_:
                if str(ev.get("action") or "").lower() == "download":
                    b = ev.get("bytes")
                    if isinstance(b, int) and b >= byte_threshold:
                        tags.append("large download")
                        break

            # Many failures (useful for bob's 403 burst)
            fails = sum(1 for ev in seq_events_ if str(ev.get("status") or "").lower() in {"fail", "failed"})
            if fails >= 3:
                tags.append("many failures")

            # De-dup while preserving order
            seen = set()
            out = []
            for t in tags:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return out

        threshold_used = 0.0
        threshold_name = ""
        if anomaly_level == "high":
            threshold_used = t_high
            threshold_name = "t_high"
        elif anomaly_level == "medium":
            threshold_used = t_medium
            threshold_name = "t_medium"
        else:
            # Low anomaly-level: report relative to t_medium for clarity (if available)
            threshold_used = t_medium if float(t_medium) > 0 else float(t_high)
            threshold_name = "t_medium" if float(t_medium) > 0 else "t_high"

        top_feature_text = ", ".join(_friendly_feature_name(x["feature"]) for x in top_features[:top_k])
        top_event_text = ""
        if top_events:
            ev0 = top_events[0]
            r = ev0.get("resource")
            top_event_text = f"Top event: {ev0.get('action')} {r if r else ''} at {ev0.get('ts')}".strip()

        reason_short_tags = _make_reason_tags(cfg, seq_events, end_ts)
        # Extend tags for newer low-level categories (keeps output explainable)
        if "novel_resource" in categories:
            reason_short_tags.append("new resource")
        if "some_failures" in categories and "many_failures" not in categories:
            reason_short_tags.append("few failures")
        if "new_ip" in categories:
            reason_short_tags.append("new IP")
        if "admin_api" in categories:
            reason_short_tags.append("admin API")
        if "deny_burst" in categories:
            reason_short_tags.append("deny burst")
        reason_short = " + ".join(reason_short_tags) if reason_short_tags else "Unusual behavior"

        if anomaly_level in {"medium", "high"}:
            reason = (
                f"{reason_short}. Reconstruction error {e:.4f} exceeded {threshold_name}={threshold_used:.4f}; "
                f"drivers: {top_feature_text}. {top_event_text}"
            ).strip()
        else:
            # Low anomaly-level: avoid misleading 'exceeded =0.0000' messaging.
            if float(threshold_used) > 0:
                reason = (
                    f"{reason_short}. Reconstruction error {e:.4f} is below {threshold_name}={threshold_used:.4f}; "
                    f"drivers: {top_feature_text}. {top_event_text}"
                ).strip()
            else:
                reason = (
                    f"{reason_short}. Reconstruction error {e:.4f}; drivers: {top_feature_text}. {top_event_text}"
                ).strip()

        alerts.append(
            {
                "ts": end_ts,
                "user": user,
                "role": role,
                "level": level,
                "score": round(float(risk), 2),
                "event": {
                    "seq_start": meta.get("start_ts"),
                    "seq_end": end_ts,
                    "seq_len": seq_len,
                    "anomaly_score": float(e),
                    "t_medium": float(t_medium),
                    "t_high": float(t_high),
                    "risk_score": float(risk),
                },
                # Improvement #7: prioritization signal for sorting/triage
                "priority": {
                    "score": round(
                        float(
                            (0.7 * risk + 0.3 * severity_points + (10.0 if "sudo" in reason_short_tags else 0.0) + (10.0 if "large download" in reason_short_tags else 0.0))
                            * (0.6 if level == "low" else 1.0)
                        ),
                        2,
                    ),
                    "critical": [t for t in ("sudo", "large download") if t in reason_short_tags],
                },
                "explanations": [
                    {
                        "kind": "dl",
                        "name": "lstm_autoencoder_reconstruction_error",
                        "weight": float(e),
                        "evidence": {
                            "anomaly_level": anomaly_level,
                            "reason_short": reason_short,
                            "reason": reason,
                            "reconstruction_error": float(e),
                            "threshold_name": threshold_name,
                            "threshold": float(threshold_used),
                            "top_features": top_features,
                            "top_events": top_events,
                            # Improvement #3: categories for downstream analytics
                            "categories": categories,
                        },
                    }
                ],
            }
        )

        # Update adaptive stats only when below medium threshold.
        _update_adaptive_stats(cfg, role, e, t_medium, adaptive_stats)

    # Persist adaptive stats for next run (adaptive thresholds)
    _save_adaptive_stats(artifacts.adaptive_stats_path, adaptive_stats)

    # Improvement #7: sort alerts by priority score (descending)
    alerts.sort(key=lambda a: float((a.get("priority") or {}).get("score", 0.0)), reverse=True)

    # Demo-only: force a fixed level distribution for presentation.
    # Remaps the top alerts (by priority) into requested buckets while preserving
    # the original model-derived severity in `level_raw`.
    alerting_cfg = cfg.get("alerting", {}) if isinstance(cfg.get("alerting", {}), dict) else {}
    demo_dist = alerting_cfg.get("demo_level_distribution")
    if isinstance(demo_dist, dict):
        try:
            n_high = int(demo_dist.get("high", 0))
            n_med = int(demo_dist.get("medium", 0))
            n_low = int(demo_dist.get("low", 0))
            unique_users = bool(demo_dist.get("unique_users", False))
        except Exception:
            n_high, n_med, n_low = 0, 0, 0
            unique_users = False

        total = max(0, n_high) + max(0, n_med) + max(0, n_low)
        if total > 0 and alerts:
            if unique_users:
                chosen: list[dict[str, Any]] = []
                seen: set[str] = set()
                for a in alerts:
                    u = str(a.get("user") or "unknown")
                    if u in seen:
                        continue
                    seen.add(u)
                    chosen.append(a)
                    if len(chosen) >= total:
                        break
            else:
                chosen = alerts[: min(total, len(alerts))]

            if len(chosen) < min(total, len(alerts)):
                # Not enough unique users; fall back to filling remaining slots.
                for a in alerts:
                    if len(chosen) >= min(total, len(alerts)):
                        break
                    if a in chosen:
                        continue
                    chosen.append(a)

            out: list[dict[str, Any]] = []
            for idx, a in enumerate(chosen):
                a2 = dict(a)
                a2.setdefault("level_raw", a2.get("level"))
                if idx < n_high:
                    a2["level"] = "high"
                elif idx < n_high + n_med:
                    a2["level"] = "medium"
                else:
                    a2["level"] = "low"
                out.append(a2)
            alerts = out
    return alerts
