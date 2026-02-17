from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from itds.ml.feature_engineering import FeatureSpec, build_feature_spec, event_to_vector, load_feature_spec, save_feature_spec
from itds.ml.lstm_autoencoder import build_lstm_autoencoder
from itds.ml.sequence_builder import SequenceBatch, build_user_sequences


@dataclass(frozen=True)
class ModelArtifacts:
    model_dir: Path
    spec_path: Path
    thresholds_path: Path


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
        model_dir=out_dir / f"lstm_ae_L{seq_len}",
        spec_path=out_dir / "feature_spec.json",
        thresholds_path=out_dir / f"thresholds_L{seq_len}.json",
    )


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

    def _saved_model_present(model_dir: Path) -> bool:
        # `model.save(dir)` writes a SavedModel with `saved_model.pb`.
        # Some TF versions also include `keras_metadata.pb`, but it's not guaranteed.
        return model_dir.exists() and (model_dir / "saved_model.pb").exists()

    if _saved_model_present(artifacts.model_dir) and artifacts.spec_path.exists() and artifacts.thresholds_path.exists():
        model = tf.keras.models.load_model(str(artifacts.model_dir))
        spec = load_feature_spec(str(artifacts.spec_path))
        thresholds = _load_thresholds(artifacts.thresholds_path) or {"t_medium": 0.0, "t_high": 0.0}

        # If thresholds were written during a low-data fallback run (0.0), recompute
        # them from current data so scoring can produce alerts.
        if float(thresholds.get("t_medium", 0.0)) == 0.0 and float(thresholds.get("t_high", 0.0)) == 0.0:
            ml_cfg = cfg.get("lstm_ae", {})
            seq_len = int(ml_cfg.get("seq_len", 10))
            stride = int(ml_cfg.get("stride", 1))
            vectors = [event_to_vector(e, spec) for e in events]
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

        return model, spec, thresholds

    # Fit feature spec on available (assumed normal) events
    spec = build_feature_spec(events)
    save_feature_spec(str(artifacts.spec_path), spec)

    vectors = [event_to_vector(e, spec) for e in events]

    ml_cfg = cfg.get("lstm_ae", {})
    seq_len = int(ml_cfg.get("seq_len", 10))
    stride = int(ml_cfg.get("stride", 1))
    batch = build_user_sequences(events, vectors, seq_len=seq_len, stride=stride)

    if batch.X.shape[0] < 4:
        # Not enough sequences; still create a model artifact so pipeline can run.
        model = build_lstm_autoencoder(seq_len=seq_len, feature_dim=vectors[0].shape[0] if vectors else 1)
        artifacts.model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(artifacts.model_dir))
        _save_thresholds(artifacts.thresholds_path, t_medium=0.0, t_high=0.0)
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

    artifacts.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(artifacts.model_dir))
    _save_thresholds(artifacts.thresholds_path, t_medium=t_medium, t_high=t_high)

    return model, spec, {"t_medium": float(t_medium), "t_high": float(t_high)}


def score_sequences(cfg: dict, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model, spec, thresholds = ensure_trained(cfg, events)

    vectors = [event_to_vector(e, spec) for e in events]

    ml_cfg = cfg.get("lstm_ae", {})
    seq_len = int(ml_cfg.get("seq_len", 10))
    stride = int(ml_cfg.get("stride", 1))
    top_k = int(ml_cfg.get("explain_top_k", 3))

    batch = build_user_sequences(events, vectors, seq_len=seq_len, stride=stride)
    if batch.X.shape[0] == 0:
        return []

    Xhat = model.predict(batch.X, verbose=0)
    diff2 = (batch.X - Xhat) ** 2
    seq_err = np.mean(diff2, axis=(1, 2))

    t_medium = float(thresholds.get("t_medium", 0.0))
    t_high = float(thresholds.get("t_high", 0.0))

    # Risk aggregation per user
    half_life_h = float(cfg.get("risk", {}).get("decay_half_life_hours", 24))
    risk_hi = float(cfg.get("risk", {}).get("threshold_high", 70))
    risk_med = float(cfg.get("risk", {}).get("threshold_medium", 45))

    # Maintain per-user risk state
    state: dict[str, dict[str, Any]] = {}

    alerts: list[dict[str, Any]] = []

    for i, meta in enumerate(batch.meta):
        user = str(meta.get("user") or "unknown")
        end_ts = str(meta.get("end_ts") or "")

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
        if prev is None:
            prev_risk = 0.0
            prev_ts = end_ts
        else:
            prev_risk = float(prev.get("risk", 0.0))
            prev_ts = str(prev.get("ts") or end_ts)

        # decay using ISO lexicographic difference is not reliable; keep simple: constant decay per sequence.
        # (Student-feasible; still explainable.)
        decay = 0.5 ** (1.0 / max(1e-6, half_life_h))
        risk = max(0.0, min(100.0, prev_risk * decay + severity))
        state[user] = {"risk": risk, "ts": end_ts}

        # Only alert on medium/high anomaly OR medium/high risk.
        level = "low"
        if risk >= risk_hi or anomaly_level == "high":
            level = "high"
        elif risk >= risk_med or anomaly_level == "medium":
            level = "medium"

        if level == "low":
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

        top_feature_text = ", ".join(_friendly_feature_name(x["feature"]) for x in top_features[:top_k])
        top_event_text = ""
        if top_events:
            ev0 = top_events[0]
            r = ev0.get("resource")
            top_event_text = f"Top event: {ev0.get('action')} {r if r else ''} at {ev0.get('ts')}".strip()

        reason_short_tags = _make_reason_tags(cfg, seq_events, end_ts)
        reason_short = " + ".join(reason_short_tags) if reason_short_tags else "Unusual behavior"

        reason = (
            f"{reason_short}. Reconstruction error {e:.4f} exceeded {threshold_name}={threshold_used:.4f}; "
            f"drivers: {top_feature_text}. {top_event_text}"
        ).strip()

        alerts.append(
            {
                "ts": end_ts,
                "user": user,
                "level": level,
                "score": round(float(risk), 2),
                "event": {
                    "seq_start": meta.get("start_ts"),
                    "seq_end": end_ts,
                    "seq_len": seq_len,
                    "anomaly_score": float(e),
                    "t_medium": t_medium,
                    "t_high": t_high,
                    "risk_score": float(risk),
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
                        },
                    }
                ],
            }
        )

    return alerts
