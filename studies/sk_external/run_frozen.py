"""External SK comparison with the archived WHA phase-two tracking and scoring code."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
HIST = Path.home() / "wha-paper-phase2-20261004-iPztef/project"
ARCHIVED = HIST / "work/results/paper_phase2_20261004"
SIGNATURE = "903c64ebaf9d58706d1633b9c7a45af7592e5121138a9091feba73415fbf3e68"
RUNTIME = Path.home() / "wha-paper-pilot-20261003-CEtun5/runtime/bin/python"
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
if socket.gethostname() != "workstation2" or sys.executable != str(RUNTIME):
    raise RuntimeError("Use the archived interpreter on gpuserver6000.")
if any(os.environ.get(k) != "1" for k in THREADS) or os.environ.get("PYTHONHASHSEED") != "0":
    raise RuntimeError("Expected the archived single-thread settings and PYTHONHASHSEED=0")

sys.path.insert(0, str(HIST / "src"))
import numpy as np
import yaml
import pycpd
from deformeot.paper_phase2 import dispatch, save_assignments
from deformeot.paper_pilot import atomic_json
from deformeot.wormid import WormIDRecording, identity_blind_sequence
from deformeot.wormid_benchmark import ReferenceTrackerConfig

METHODS = ("dual_anchor", "cpd_reference")
# Original candidate inventory order and calibrated calcium spacing, from the outcome-blind audit.
COHORT = {
    "SK1_20230322_185704": (1500,118,[0.3208,0.3208,2.5],1.04),
    "SK1_20230322_201650": (1500,117,[0.3208,0.3208,2.5],1.04),
    "SK1_20230506_150145": (1500,93,[0.1604,0.1604,3.0],1.04),
    "SK1_20230506_153351": (1500,103,[0.1604,0.1604,3.0],1.04),
    "SK2_20220426_w00": (3000,121,[0.3208,0.3208,1.5],3.3),
    "SK2_20220426_w01": (3000,140,[0.3208,0.3208,1.5],3.3),
}

def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()

def read(path):
    return json.loads(path.read_text())

def now():
    return datetime.now(timezone.utc).isoformat()

def freeze():
    ctx = read(ARCHIVED / "active_context.json")
    selected = read(ARCHIVED / "cpd_selection.json")
    if not (selected["source_signature"] == ctx["signature"] == SIGNATURE):
        raise ValueError("Historical signature mismatch")
    selected_method = next(m for m in selected["selected"] if m["name"] == "cpd_reference")
    cfg = yaml.safe_load((HIST / "configs/wormid_prospective.yaml").read_text())
    geometry = ReferenceTrackerConfig(**cfg["association"]["geometry"])
    source_paths = [p for p in ctx["source_hashes"] if p.startswith("src/") or p in
                    {"configs/wormid_prospective.yaml","configs/paper_phase2.yaml"}]
    source_hashes = {p: sha(HIST / p) for p in source_paths}
    if any(ctx["source_hashes"][p] != h for p,h in source_hashes.items()):
        raise ValueError("Archived project sources differ from the frozen record")
    dep_root = Path(pycpd.__file__).parent
    dep_hashes = {str(p.relative_to(dep_root)):sha(p) for p in dep_root.rglob("*.py")}
    packages = {k:metadata.version(k) for k in ("numpy","scipy","pycpd","h5py","PyYAML")}
    if dep_hashes != ctx["runtime_dependency_source_hashes"]["pycpd"] or any(
            ctx["packages"][k] != v for k,v in packages.items()):
        raise ValueError("Numeric dependency verification failed")
    settings = {
        "historical_source_signature":SIGNATURE,"historical_source_hashes":source_hashes,
        "packages":packages, "pycpd_source_hashes":dep_hashes, "geometry":asdict(geometry),
        "methods":{"dual_anchor":{"method":"dual_anchor","name":"dual_anchor"},"cpd_reference":selected_method},
        "protocol_sha256":sha(HIST/"configs/wormid_prospective.yaml"),
        "selection_sha256":sha(ARCHIVED/"cpd_selection.json"),
        "runner_sha256":sha(Path(__file__)), "single_thread_env":{k:os.environ[k] for k in THREADS},
        "stride":1, "seed_rule":"20260720 + 1000 * zero_based_sorted_SK_inventory_index + 1",
        "animals":{a:{"frames":COHORT[a][0],"n_rois":COHORT[a][1],"spacing_xyz_um":COHORT[a][2],
                     "nominal_hz":COHORT[a][3],"seed":cfg["association"]["base_seed"]+1000*i+1}
                   for i,a in enumerate(sorted(COHORT))},
        "endpoints":"Primary accuracy includes initialized frame. Switch rate per 1000 adjacent track-transition opportunities. Duration is mean consecutive wrong scans per episode including terminal episodes. Summarize with equal animal weight.",
        "replication_direction":"For the observed three-way pattern, compare equal-animal means for higher accuracy, lower switch rate and longer wrong episodes under dual anchoring. Preserve the original practical accuracy/switch screen separately. If no wrong episodes, report duration as undefined biologically and flag historical zero encoding.",
    }
    target=ROOT/"frozen_protocol.json"
    if target.exists():
        previous=read(target)
        if {k:v for k,v in previous.items() if k!="created_utc"} != settings:
            raise ValueError("Prospective settings differ from the file already frozen")
    else:
        atomic_json(target,{**settings,"created_utc":now()})
    return settings

def run_case(animal, method):
    frozen = read(ROOT/"frozen_protocol.json")
    settings=frozen["animals"][animal]
    cache=ROOT/"inputs"/(animal+".npz")
    source_hash=sha(cache)
    out=ROOT/"results"/animal
    out.mkdir(parents=True,exist_ok=True)
    file=out/(method+".npz")
    meta=out/(method+".json")
    if meta.exists():
        saved=read(meta)
        if (saved["input_sha256"],saved["protocol_sha256"],saved["assignment_file_sha256"]) != (
                source_hash,sha(ROOT/"frozen_protocol.json"),sha(file)):
            raise ValueError("Saved task inputs or output differ")
        return
    with np.load(cache,allow_pickle=False) as data:
        xyz=np.asarray(data["points"],dtype=np.float64)
        times=np.asarray(data["times"],dtype=np.float64)
        ids=data["roi_ids"]
        spacing=np.asarray(settings["spacing_xyz_um"])
        if "spacing_xyz_um" in data and not np.allclose(data["spacing_xyz_um"],spacing,rtol=1e-6,atol=1e-12):
            raise ValueError("Mismatched calcium calibration")
        if xyz.shape!=(settings["frames"],settings["n_rois"],3) or not np.array_equal(ids,np.arange(settings["n_rois"])):
            raise ValueError("Incorrect cohort")
        if not np.allclose(times - times[0],np.arange(len(times))/settings["nominal_hz"],rtol=1e-6,atol=1e-6):
            raise ValueError("Frame clock differs from the pre-run audit")
        recording=WormIDRecording(str(cache), animal, np.ascontiguousarray(xyz[:,:,::-1]),
            tuple(f"ROI_{i}" for i in ids),times,tuple(spacing[::-1]),xyz / spacing)
    sequence=identity_blind_sequence(recording,frame_indices=range(recording.frames),seed=settings["seed"])
    geometry=ReferenceTrackerConfig(**frozen["geometry"])
    config=yaml.safe_load((HIST/"configs/paper_phase2.yaml").read_text())
    started=now()
    tick=time.monotonic()
    result=dispatch(sequence,frozen["methods"][method],HIST,config,geometry)
    elapsed=time.monotonic()-tick
    metrics=save_assignments(file,sequence,result)
    atomic_json(meta,{**metrics,"animal":animal,"method":method,"permutation_seed":settings["seed"],
        "started_utc":started,"finished_utc":now(),"tracker_seconds":elapsed,"input_sha256":source_hash,
        "protocol_sha256":sha(ROOT/"frozen_protocol.json"),"host":socket.gethostname()})

def run_all():
    lock=(ROOT/"run.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze()
    if not (ROOT/"inputs/input_manifest.json").is_file():
        raise ValueError("Extraction manifest is not complete")
    def invoke(task):
        animal,method=task
        logdir=ROOT/"logs"
        logdir.mkdir(exist_ok=True)
        logfile=logdir/f"{animal}__{method}.log"
        print(now(),"starting",animal,method,flush=True)
        with logfile.open("a") as log:
            done=subprocess.run([sys.executable,str(Path(__file__).resolve()),"case",animal,method],stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:
            raise RuntimeError(f"Task failed: {animal}/{method}, see {logfile}")
        return animal,method
    # Longest acquisitions first. Each process uses the archived one-thread execution.
    tasks=[(a,m) for m in ("cpd_reference","dual_anchor") for a in sorted(COHORT,key=lambda a: -COHORT[a][0]*COHORT[a][1])]
    with ThreadPoolExecutor(max_workers=6) as pool:
        fs=[pool.submit(invoke,t) for t in tasks]
        for f in as_completed(fs):
            print(now(),"completed",*f.result(),flush=True)
    print("Complete: 12/12",flush=True)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("action",choices=["freeze","run","case"])
    args.add_argument("animal",nargs="?",choices=sorted(COHORT))
    args.add_argument("method",nargs="?",choices=METHODS)
    arg=args.parse_args()
    if arg.action=="freeze": freeze()
    elif arg.action=="case": run_case(arg.animal,arg.method)
    else: run_all()
