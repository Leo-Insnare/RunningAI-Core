import numpy as np

from app.main import get_engine


def test_per_foot_prediction_is_separate_from_final_consensus(monkeypatch):
    engine = get_engine()

    def fake_member_foot_probabilities(by_foot, rules, member, model, centroids):
        return {
            "left": np.asarray([0.80, 0.10, 0.10], dtype=float),
            "right": np.asarray([0.10, 0.80, 0.10], dtype=float),
        }

    monkeypatch.setattr(engine, "_member_foot_probabilities", fake_member_foot_probabilities)
    monkeypatch.setattr(engine, "_rescue_probability", lambda vector: 0.20)

    sequence = np.zeros((21, 17), dtype=float).tolist()
    out = engine.predict_strike([
        {"foot": "left", "events": [{"sequence": sequence, "rule_class": None}]},
        {"foot": "right", "events": [{"sequence": sequence, "rule_class": None}]},
    ])

    feet = {row["foot"]: row for row in out["feet"]}
    assert feet["left"]["prediction"] == "heel"
    assert feet["right"]["prediction"] == "midfoot"
    assert out["patient_anchor_class"] == "heel"
    assert out["final_class"] == "heel"
