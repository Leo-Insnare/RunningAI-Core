from app.debug_export_adapter import DebugExportAdapter, SEQUENCE_CHANNELS
from tests.fixture_debug_export import build_debug_export_bytes


def test_adapter_builds_frozen_request():
    result = DebugExportAdapter.from_bytes(build_debug_export_bytes()).build()
    req = result.request
    assert req.patient_meta.patient_id == "test001"
    assert len(req.strike_feet) == 2
    assert {x.foot for x in req.strike_feet} == {"left", "right"}
    for foot in req.strike_feet:
        assert len(foot.events) >= 3
        assert len(foot.events[0].sequence) == 21
        assert len(foot.events[0].sequence[0]) == len(SEQUENCE_CHANNELS) == 17
    assert req.overstride_features["os_clip_selected_mm"] == 72.8
    assert req.overstride_features["thigh_flexion_recalc_deg"] is not None
    assert req.overstride_features["hip_rom_recalc_deg"] is not None
