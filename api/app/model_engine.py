from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    CRITICAL_SEQUENCE_CHANNELS,
    EXPECTED_ARTIFACT_SHA256,
    MODEL_DIR,
    MODEL_VERSION,
    REVIEW_CONFIDENCE_THRESHOLD,
)


class Encoder(nn.Module):
    def __init__(self, n_channels: int, emb_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(48, emb_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).squeeze(-1)


class MultiTaskSupCon(nn.Module):
    def __init__(self, n_channels: int, emb_dim: int = 48):
        super().__init__()
        self.encoder = Encoder(n_channels, emb_dim)
        self.dropout = nn.Dropout(0.20)
        self.head3 = nn.Linear(emb_dim, 3)
        self.heel_head = nn.Linear(emb_dim, 2)
        self.mf_head = nn.Linear(emb_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.encoder(x)
        z = self.dropout(emb)
        return {
            "embedding": emb,
            "logits3": self.head3(z),
            "heel_logits": self.heel_head(z),
            "mf_logits": self.mf_head(z),
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    out = np.empty(arr.shape, dtype=np.float32)
    it = np.nditer(arr, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
    for v in it:
        x = v.item()
        try:
            out[it.multi_index] = np.nan if x is None else float(x)
        except Exception:
            out[it.multi_index] = np.nan
    return out


def _manual_impute(X: np.ndarray, imputer) -> np.ndarray:
    X = np.asarray(X, dtype=float).copy()
    stats = np.asarray(imputer.statistics_, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    bad = ~np.isfinite(X)
    if bad.any():
        rows, cols = np.where(bad)
        X[rows, cols] = stats[cols]
    return X


@dataclass
class EngineStatus:
    artifact_integrity: bool
    model_version: str
    labels: list[str]
    sequence_channels: list[str]
    time_grid_sec: np.ndarray


class FrozenModelEngine:
    def __init__(self, model_dir: Path | None = None, device: str | None = None):
        self.model_dir = Path(model_dir or MODEL_DIR)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._verify_artifacts()
        self.anchor = torch.load(
            self.model_dir / "strike_anchor_supcon_15model_v0_16.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.rescue = joblib.load(
            self.model_dir / "strike_forefoot_heel_rescue_5model_v0_16.joblib"
        )
        self.overstride = joblib.load(
            self.model_dir / "overstride_cv_ensemble_candidate_v0_16.joblib"
        )
        self.overstride_final = json.loads(
            (self.model_dir / "overstride_final_inference_v1.json").read_text(encoding="utf-8")
        )
        self.labels = list(self.anchor["labels"])
        self.channels = list(self.anchor["sequence_channels"])
        self.time_grid = np.asarray(self.anchor["time_grid_sec"], dtype=float)
        self.channel_to_idx = {c: i for i, c in enumerate(self.channels)}
        self._validate_artifact_contract()
        self._torch_members = self._load_torch_members()

    def _verify_artifacts(self) -> None:
        for name, expected in EXPECTED_ARTIFACT_SHA256.items():
            path = self.model_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Frozen model artifact missing: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise RuntimeError(
                    f"Frozen artifact SHA256 mismatch: {name}; expected={expected}; actual={actual}"
                )

    def _validate_artifact_contract(self) -> None:
        if int(self.anchor.get("n_models", -1)) != 15:
            raise RuntimeError("Strike anchor artifact must contain 15 models.")
        if int(self.rescue.get("n_models", -1)) != 5:
            raise RuntimeError("Strike rescue artifact must contain 5 models.")
        if int(self.overstride.get("n_model_pairs", -1)) != 30:
            raise RuntimeError("Overstride artifact must contain 30 model pairs.")
        expected_features = [
            "base_pred_v016",
            "knee_landing_event_median_deg",
            "thigh_flexion_recalc_deg",
            "hip_rom_recalc_deg",
        ]
        if list(self.overstride_final.get("features", [])) != expected_features:
            raise RuntimeError("Final Overstride feature schema mismatch.")
        if self.overstride_final.get("model_version") != MODEL_VERSION:
            raise RuntimeError("Final Overstride model version mismatch.")
        if abs(float(self.rescue.get("heel_override_threshold")) - 0.65) > 1e-12:
            raise RuntimeError("Frozen rescue threshold must remain 0.65.")
        if len(self.channels) != 17 or len(self.time_grid) != 21:
            raise RuntimeError("Frozen Strike input schema must be 21x17.")
        if list(self.rescue.get("feature_columns", [])) != self._expected_rescue_features():
            raise RuntimeError("Frozen rescue feature schema mismatch.")

    def _load_torch_members(self):
        loaded = []
        for member in self.anchor["members"]:
            model = MultiTaskSupCon(len(self.channels), 48).to(self.device)
            model.load_state_dict(member["state_dict"])
            model.eval()
            centroids = F.normalize(
                torch.tensor(member["centroids"], dtype=torch.float32, device=self.device),
                dim=1,
            )
            loaded.append((member, model, centroids))
        return loaded

    @property
    def status(self) -> EngineStatus:
        return EngineStatus(
            artifact_integrity=True,
            model_version=MODEL_VERSION,
            labels=self.labels,
            sequence_channels=self.channels,
            time_grid_sec=self.time_grid.copy(),
        )

    def predict_overstride(self, features: dict[str, float | None]) -> dict:
        f = {k: (np.nan if v is None else float(v)) for k, v in features.items()}
        clip = f.get("os_clip_selected_mm", np.nan)
        event = f.get("os_event_abs_mean_mm", np.nan)
        if not np.isfinite(clip) and not np.isfinite(event):
            return {
                "status": "unmeasurable",
                "reason": "both_overstride_anchor_features_missing",
                "prediction_mm": None,
                "prediction_std_mm": None,
                "model_pair_count": 0,
            }
        for anchor in ["os_clip_selected_mm", "os_event_abs_mean_mm"]:
            x = f.get(anchor, np.nan)
            f[f"{anchor}__sq100"] = (x / 100.0) ** 2 if np.isfinite(x) else np.nan
            f[f"{anchor}__log1p"] = math.log1p(max(x, 0.0)) if np.isfinite(x) else np.nan

        pair_predictions: list[float] = []
        for member in self.overstride["members"]:
            component = []
            for key in ["model_clip_selected", "model_event_abs_mean"]:
                ref = member[key]
                cols = list(ref["features"])
                X = np.asarray([[f.get(c, np.nan) for c in cols]], dtype=float)
                pipe = ref["model"]
                imputer = pipe.named_steps["imp"]
                estimator = pipe.named_steps["model"]
                Xi = _manual_impute(X, imputer)
                component.append(float(estimator.predict(Xi)[0]))
            w = float(member["blend_weight"])
            pair_predictions.append(w * component[0] + (1.0 - w) * component[1])

        base_pred = float(np.mean(pair_predictions))
        base_std = float(np.std(pair_predictions, ddof=1)) if len(pair_predictions) > 1 else 0.0

        final_features = {
            "base_pred_v016": base_pred,
            "knee_landing_event_median_deg": f.get("knee_landing_event_median_deg", np.nan),
            "thigh_flexion_recalc_deg": f.get("thigh_flexion_recalc_deg", np.nan),
            "hip_rom_recalc_deg": f.get("hip_rom_recalc_deg", np.nan),
        }
        missing = [k for k, v in final_features.items() if not np.isfinite(v)]
        if missing:
            return {
                "status": "unmeasurable",
                "reason": "final_overstride_features_missing:" + "|".join(missing),
                "prediction_mm": None,
                "prediction_std_mm": None,
                "model_pair_count": len(pair_predictions),
            }

        coefficients = self.overstride_final["coefficients"]
        pred = float(self.overstride_final["intercept"])
        for name, value in final_features.items():
            pred += float(coefficients[name]) * float(value)

        std = abs(float(coefficients["base_pred_v016"])) * base_std
        return {
            "status": "completed",
            "reason": None,
            "prediction_mm": float(pred),
            "prediction_std_mm": float(std),
            "model_pair_count": len(pair_predictions),
        }

    def _validate_strike_input(self, feet: list[dict]) -> tuple[dict[str, np.ndarray], dict[str, list[str | None]]]:
        by_foot: dict[str, np.ndarray] = {}
        rules: dict[str, list[str | None]] = {}
        seen = set()
        for foot_item in feet:
            foot = str(foot_item["foot"]).lower()
            if foot not in {"left", "right"} or foot in seen:
                raise ValueError("Strike payload must contain exactly one left and one right foot.")
            seen.add(foot)
            events = foot_item.get("events", [])
            if not events:
                raise ValueError(f"No Strike events supplied for {foot} foot.")
            seqs = []
            rr = []
            for event in events:
                arr = _safe_float_array(event["sequence"])
                if arr.shape != (21, 17):
                    raise ValueError(
                        f"Each Strike event must be 21x17; got {arr.shape} for {foot}."
                    )
                seqs.append(arr)
                rule = event.get("rule_class")
                if rule is not None and rule not in self.labels:
                    raise ValueError(f"Unknown rule_class: {rule}")
                rr.append(rule)
            by_foot[foot] = np.stack(seqs).astype(np.float32)
            rules[foot] = rr
        if seen != {"left", "right"}:
            raise ValueError("Strike payload requires both left and right feet.")
        return by_foot, rules

    def _critical_channel_failures(self, by_foot: dict[str, np.ndarray]) -> list[str]:
        all_events = np.concatenate([by_foot["left"], by_foot["right"]], axis=0)
        failed = []
        for channel in CRITICAL_SEQUENCE_CHANNELS:
            ci = self.channel_to_idx[channel]
            if not np.isfinite(all_events[:, :, ci]).any():
                failed.append(channel)
        return failed

    def _member_foot_probabilities(self, by_foot, rules, member, model, centroids):
        out = {}
        for foot in ["left", "right"]:
            X = by_foot[foot].copy()
            mean = np.asarray(member["channel_mean"], dtype=np.float32)
            std = np.asarray(member["channel_std"], dtype=np.float32)
            for c in range(X.shape[2]):
                z = X[:, :, c]
                z[~np.isfinite(z)] = mean[c]
                X[:, :, c] = (z - mean[c]) / std[c]
            with torch.no_grad():
                xb = torch.tensor(X, dtype=torch.float32, device=self.device)
                o = model(xb)
                p3 = torch.softmax(o["logits3"], dim=1)
                ph = torch.softmax(o["heel_logits"], dim=1)
                pmf = torch.softmax(o["mf_logits"], dim=1)
                pheel = ph[:, 0]
                hier = torch.stack(
                    [pheel, (1 - pheel) * pmf[:, 0], (1 - pheel) * pmf[:, 1]], dim=1
                )
                neural = (
                    (1.0 - float(self.anchor["neural_hier_blend"])) * p3
                    + float(self.anchor["neural_hier_blend"]) * hier
                )
                proto = torch.softmax(
                    (F.normalize(o["embedding"], dim=1) @ centroids.T) / 0.20, dim=1
                )
                p = (
                    (1.0 - float(self.anchor["prototype_blend"])) * neural
                    + float(self.anchor["prototype_blend"]) * proto
                ).cpu().numpy()
            foot_p = np.median(p, axis=0)
            foot_p = foot_p / max(float(foot_p.sum()), 1e-12)
            rule_rows = []
            for r in rules[foot]:
                if r in self.labels:
                    one = np.zeros(3, dtype=float)
                    one[self.labels.index(r)] = 1.0
                    rule_rows.append(one)
            if rule_rows:
                rp = np.mean(rule_rows, axis=0)
                rp = rp / max(float(rp.sum()), 1e-12)
                a = float(self.anchor["rule_blend_fixed"])
                foot_p = (1.0 - a) * foot_p + a * rp
                foot_p = foot_p / max(float(foot_p.sum()), 1e-12)
            out[foot] = foot_p.astype(float)
        return out

    def _expected_rescue_features(self) -> list[str]:
        key_channels = [
            "heel_ground_gap_norm",
            "toe_ground_gap_norm",
            "heel_toe_y_diff_norm",
            "foot_angle_canonical_deg",
            "foot_angle_delta_ic_deg",
            "shank_angle_abs_deg",
            "knee_flexion_deg",
            "heel_gap_velocity_norm_s",
            "toe_gap_velocity_norm_s",
            "heel_toe_velocity_norm_s",
            "foot_angle_velocity_deg_s",
        ]
        times = [-100, -50, 0, 50, 100]
        cols = []
        for c in key_channels:
            for ms in times:
                cols.append(f"{c}_t{ms:+d}ms")
            cols.append(f"{c}_event_iqr_ic")
        cols.extend(
            [
                "gap_toe_minus_heel_pre100",
                "gap_toe_minus_heel_ic",
                "gap_toe_minus_heel_post50",
            ]
        )
        return cols

    def _rescue_feature_vector(self, by_foot: dict[str, np.ndarray]) -> np.ndarray:
        key_channels = [
            "heel_ground_gap_norm",
            "toe_ground_gap_norm",
            "heel_toe_y_diff_norm",
            "foot_angle_canonical_deg",
            "foot_angle_delta_ic_deg",
            "shank_angle_abs_deg",
            "knee_flexion_deg",
            "heel_gap_velocity_norm_s",
            "toe_gap_velocity_norm_s",
            "heel_toe_velocity_norm_s",
            "foot_angle_velocity_deg_s",
        ]
        times = [-100, -50, 0, 50, 100]
        time_indices = {
            ms: int(np.argmin(np.abs(self.time_grid - ms / 1000.0))) for ms in times
        }
        foot_vectors = []
        for foot in ["left", "right"]:
            arr = by_foot[foot]
            row: dict[str, float] = {}
            for c in key_channels:
                ci = self.channel_to_idx[c]
                for ms in times:
                    ti = time_indices[ms]
                    row[f"{c}_t{ms:+d}ms"] = float(np.nanmedian(arr[:, ti, ci]))
                v = arr[:, time_indices[0], ci]
                if np.isfinite(v).any():
                    row[f"{c}_event_iqr_ic"] = float(
                        np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
                    )
                else:
                    row[f"{c}_event_iqr_ic"] = np.nan
            hci = self.channel_to_idx["heel_ground_gap_norm"]
            tci = self.channel_to_idx["toe_ground_gap_norm"]
            for ms, suffix in [(-100, "pre100"), (0, "ic"), (50, "post50")]:
                ti = time_indices[ms]
                toe = float(np.nanmedian(arr[:, ti, tci]))
                heel = float(np.nanmedian(arr[:, ti, hci]))
                row[f"gap_toe_minus_heel_{suffix}"] = toe - heel
            foot_vectors.append(
                np.asarray([row[c] for c in self.rescue["feature_columns"]], dtype=float)
            )
        return (foot_vectors[0] + foot_vectors[1]) / 2.0

    def _rescue_probability(self, vector: np.ndarray) -> float:
        values = []
        for model_ref in self.rescue["models"]:
            pipe = model_ref["pipeline"]
            imp = pipe.named_steps["imputer"]
            clf = pipe.named_steps["extra_trees"]
            Xi = _manual_impute(vector[None, :], imp)
            p = clf.predict_proba(Xi)[0]
            classes = list(clf.classes_)
            values.append(float(p[classes.index("heel")]))
        return float(np.mean(values))

    def predict_strike(self, feet: list[dict]) -> dict:
        by_foot, rules = self._validate_strike_input(feet)
        failed = self._critical_channel_failures(by_foot)
        if failed:
            return {
                "status": "unmeasurable",
                "reason": "critical_channel_complete_loss:" + "|".join(failed),
                "feet": [
                    {
                        "foot": foot,
                        "prediction": None,
                        "confidence": None,
                        "review_required": True,
                        "local_probabilities": None,
                    }
                    for foot in ["left", "right"]
                ],
                "patient_anchor_class": None,
                "patient_anchor_confidence": None,
                "final_class": None,
                "final_confidence": None,
                "review_required": True,
                "rescue_p_heel": None,
                "rescue_activated": False,
            }

        member_foot = {"left": [], "right": []}
        for member, model, centroids in self._torch_members:
            probs = self._member_foot_probabilities(by_foot, rules, member, model, centroids)
            for foot in ["left", "right"]:
                member_foot[foot].append(probs[foot])

        local = {
            foot: np.mean(np.stack(member_foot[foot]), axis=0) for foot in ["left", "right"]
        }
        patient_p = (local["left"] + local["right"]) / 2.0
        patient_p = patient_p / max(float(patient_p.sum()), 1e-12)
        anchor_class = self.labels[int(np.argmax(patient_p))]
        anchor_conf = float(np.max(patient_p))
        rescue_p = self._rescue_probability(self._rescue_feature_vector(by_foot))
        activated = bool(
            anchor_class == "forefoot"
            and rescue_p >= float(self.rescue["heel_override_threshold"])
        )
        final_class = "heel" if activated else anchor_class
        final_conf = rescue_p if activated else anchor_conf
        foot_rows = []
        for foot in ["left", "right"]:
            local_class = self.labels[int(np.argmax(local[foot]))]
            local_conf = float(np.max(local[foot]))
            foot_rows.append(
                {
                    "foot": foot,
                    "prediction": local_class,
                    "confidence": local_conf,
                    "review_required": local_conf < REVIEW_CONFIDENCE_THRESHOLD,
                    "local_probabilities": {
                        label: float(local[foot][i]) for i, label in enumerate(self.labels)
                    },
                }
            )
        return {
            "status": "completed",
            "reason": None,
            "feet": foot_rows,
            "patient_anchor_class": anchor_class,
            "patient_anchor_confidence": anchor_conf,
            "final_class": final_class,
            "final_confidence": final_conf,
            "review_required": final_conf < REVIEW_CONFIDENCE_THRESHOLD,
            "rescue_p_heel": rescue_p,
            "rescue_activated": activated,
        }
