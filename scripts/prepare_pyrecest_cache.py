"""Extract the one missing tuning coordinate cache on the calculation server."""

from dataclasses import fields
import json
from pathlib import Path
import sys

from deformeot.wormid_dandi import DandiAsset, extract_coordinate_cache


def main() -> None:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    canonical = json.loads(source.read_text())
    field_names = {field.name for field in fields(DandiAsset)}
    for row in canonical["dandiset"]["assets"]:
        if row["recording_id"] != "sub-20190925-04_ses-20190925_ophys":
            continue
        asset = DandiAsset(**{key: row[key] for key in field_names})
        recording, audit = extract_coordinate_cache(asset, destination)
        print(json.dumps({"recording": recording.recording_id,
                          "frames": recording.frames, "neurons": recording.neurons,
                          "audit": audit}, indent=2), flush=True)
        return
    raise ValueError("canonical inventory is missing the tuning recording")


if __name__ == "__main__":
    main()
