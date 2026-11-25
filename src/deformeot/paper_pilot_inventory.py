"""Render the reviewed public-data feasibility ledger without network access.

This records source observations and unresolved eligibility checks. It neither
downloads images nor scores external recordings, and does not promote published
tracker outputs into independently annotated ground truth.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import yaml


EXTERNAL_FAMILIES = {"wormid_non_ey", "zephir_public", "three_dee_cell_tracker_public"}
STATUSES = {"eligible", "conditional", "none_verified"}


def _validated_inventory(raw: object) -> dict:
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("inventory must use schema_version 1")
    inventory = deepcopy(raw)
    date.fromisoformat(inventory["reviewed_on"])
    if inventory.get("external_cohort_status") not in STATUSES:
        raise ValueError("invalid external cohort status")
    catalog = inventory.get("evidence", {})
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("inventory requires a reviewed evidence catalog")
    for source in catalog.values():
        parsed = urlparse(source["url"])
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("each evidence source requires an HTTPS URL")
        date.fromisoformat(source["retrieved_on"])
    sources = inventory.get("sources", [])
    ids = [source["id"] for source in sources]
    if len(ids) != 3 or set(ids) != EXTERNAL_FAMILIES:
        raise ValueError("inventory must contain exactly the three approved families")
    claim_ids = set()
    for section in [inventory["reference_identity"], *sources]:
        if not section.get("claims"):
            raise ValueError("every inventory section requires source-backed observations")
        for claim in section["claims"]:
            if claim["id"] in claim_ids:
                raise ValueError("claim identifiers must be unique")
            claim_ids.add(claim["id"])
            references = claim.get("evidence", [])
            if not references or any(key not in catalog for key in references):
                raise ValueError("every claim must cite existing reviewed evidence")
            claim["citations"] = [
                {"source_id": key, "url": catalog[key]["url"],
                 "retrieved_on": catalog[key]["retrieved_on"]}
                for key in references
            ]
    for source in sources:
        if source.get("status") not in STATUSES:
            raise ValueError("invalid source eligibility status")
        verified = source.get("verified_eligible_new_animals")
        if isinstance(verified, bool) or not isinstance(verified, int) or verified < 0:
            raise ValueError("verified eligible animal counts must be nonnegative integers")
        if source["status"] == "eligible" and verified < 2:
            raise ValueError("eligible external cohorts require at least two verified new animals")
    if inventory["external_cohort_status"] == "eligible" and not any(
        source["status"] == "eligible" for source in sources
    ):
        raise ValueError("aggregate eligibility requires an eligible source family")
    return inventory


def run_inventory(root: str | Path, output_dir: str | Path) -> dict:
    """Write and return the versioned, reviewed inventory; never access the web."""
    root = Path(root).resolve()
    source_path = root / "configs" / "paper_pilot_inventory.yaml"
    source_bytes = source_path.read_bytes()
    inventory = _validated_inventory(yaml.safe_load(source_bytes))
    inventory.update({
        "source_config": str(source_path),
        "source_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "reference_identity_classification": inventory["reference_identity"]["classification"],
        "execution": {
            "network_access": False,
            "external_test_scoring": False,
            "new_annotations": False,
            "author_contact": False,
            "large_image_downloads": False,
            "interpretation": "Reviewed observations are reproduced; availability is not re-polled.",
        },
    })
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "inventory.json"
    path.write_text(json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return inventory
