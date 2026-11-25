from __future__ import annotations

import pytest

from deformeot.wormid_report import summarize_benchmark


def test_summary_bootstraps_complete_held_out_recordings():
    rows = []
    for recording, delta in (("a", 0.1), ("b", 0.2)):
        for method, accuracy, switches in (
            ("first_affine", 0.7, 10),
            ("dual_anchor", 0.7 + delta, 4),
        ):
            rows.append(
                {
                    "recording_id": recording,
                    "split": "held_out",
                    "stride": 5,
                    "method": method,
                    "identity_accuracy": accuracy,
                    "severe_identity_accuracy": accuracy,
                    "correct": int(accuracy * 100),
                    "total": 100,
                    "identity_switches": switches,
                    "severe_frames": 2,
                    "frames": 11,
                    "neurons": 10,
                    "runtime_seconds": 1.0,
                }
            )
    summary = summarize_benchmark(
        {"schema_version": 1, "methods": ["first_affine", "dual_anchor"], "results": rows},
        samples=1000,
        seed=3,
    )
    paired = summary["paired_by_stride"][0]
    assert paired["identity_accuracy_difference"]["mean"] == pytest.approx(0.15)
    assert summary["held_out_recording_ids"] == ["a", "b"]
