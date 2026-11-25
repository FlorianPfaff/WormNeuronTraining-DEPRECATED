"""Validate provenance and eligibility claims without network or data scoring."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import socket

import pytest
import yaml

from deformeot.paper_pilot_inventory import _validated_inventory, run_inventory


CONFIG = Path(__file__).resolve().parents[1] / "configs/paper_pilot_inventory.yaml"


def test_inventory_reproduces_reviewed_evidence_without_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("reviewed inventory must not access the network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    shutil.copyfile(CONFIG, root / "configs/paper_pilot_inventory.yaml")
    result = run_inventory(root, tmp_path / "out")
    assert result["reference_identity_classification"] == "algorithm_assisted_expert_verified_reference_trajectories"
    assert result["external_cohort_status"] == "conditional"
    assert len(result["sources"]) == 3
    assert all(source["verified_eligible_new_animals"] == 0 for source in result["sources"])
    assert json.loads((tmp_path / "out/inventory.json").read_text()) == result
    assert result["execution"]["external_test_scoring"] is False
    for source in [result["reference_identity"], *result["sources"]]:
        assert all(claim["citations"] for claim in source["claims"])


def test_inventory_rejects_unsupported_evidence_and_unapproved_families():
    original = yaml.safe_load(CONFIG.read_text())
    missing = deepcopy(original)
    missing["sources"][0]["claims"][0]["evidence"] = ["unreviewed"]
    with pytest.raises(ValueError, match="existing reviewed evidence"):
        _validated_inventory(missing)
    extra = deepcopy(original)
    extra["sources"].append(deepcopy(extra["sources"][0]))
    with pytest.raises(ValueError, match="exactly the three"):
        _validated_inventory(extra)


def test_inventory_does_not_promote_unverified_animals_to_eligible():
    inventory = yaml.safe_load(CONFIG.read_text())
    inventory["external_cohort_status"] = "eligible"
    with pytest.raises(ValueError, match="aggregate eligibility"):
        _validated_inventory(inventory)
    inventory["sources"][0]["status"] = "eligible"
    with pytest.raises(ValueError, match="at least two verified"):
        _validated_inventory(inventory)
