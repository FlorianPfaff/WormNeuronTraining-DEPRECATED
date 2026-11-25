"""Data loading and validation for point-track benchmarks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


COORD_COLUMNS = ("t", "z", "y", "x")
TRACK_COLUMNS = ("TrackID", "ObjectID", "t", "z", "y", "x")


def read_coordinates(path: str | Path) -> pd.DataFrame:
    """Read an ASCENT-style coordinate table with globally unique object IDs."""

    frame = pd.read_csv(path)
    required = {"object_id", *COORD_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing coordinate columns: {sorted(missing)}")
    frame = frame.loc[:, ["object_id", *COORD_COLUMNS]].copy()
    frame["object_id"] = frame["object_id"].astype(str)
    frame["t"] = frame["t"].astype(int)
    for column in ("z", "y", "x"):
        frame[column] = frame[column].astype(float)
    if frame["object_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate global object_id values")
    return frame.sort_values(["t", "object_id"], kind="stable").reset_index(drop=True)


def read_tracks(path: str | Path) -> pd.DataFrame:
    """Read a Napari/ASCENT-style point-track table."""

    frame = pd.read_csv(path)
    missing = set(TRACK_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing track columns: {sorted(missing)}")
    frame = frame.loc[:, list(TRACK_COLUMNS)].copy()
    frame["TrackID"] = frame["TrackID"].astype(str)
    frame["ObjectID"] = frame["ObjectID"].astype(str)
    frame["t"] = frame["t"].astype(int)
    for column in ("z", "y", "x"):
        frame[column] = frame[column].astype(float)

    if frame["ObjectID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate global ObjectID values")
    if frame.duplicated(["TrackID", "t"]).any():
        raise ValueError(f"{path} assigns a track to multiple objects in one frame")
    return frame.sort_values(["t", "TrackID"], kind="stable").reset_index(drop=True)


def track_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    """Return compact integrity statistics for a track table."""

    counts = frame.groupby("t", sort=True).size()
    lengths = frame.groupby("TrackID", sort=False).size()
    return {
        "frames": int(counts.size),
        "first_frame": int(frame["t"].min()),
        "last_frame": int(frame["t"].max()),
        "detections": int(len(frame)),
        "tracks": int(frame["TrackID"].nunique()),
        "mean_detections_per_frame": float(counts.mean()),
        "min_detections_per_frame": int(counts.min()),
        "max_detections_per_frame": int(counts.max()),
        "median_track_length": float(lengths.median()),
    }

