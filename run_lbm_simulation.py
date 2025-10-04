#!/usr/bin/env python3
"""Submit and collect 2D LBM simulations on Slurm."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

COMPLETED_STATES = {"COMPLETED"}
FAILED_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "REVOKED",
    "DEADLINE",
    "STOPPED",
}


@dataclass
class Submission:
    project_root: Path
    run_id: str
    run_dir: Path
    geometry_source: Path
    staged_geometry: Path
    result_filename: str
    sbatch_path: Path
    manifest_path: Path
    resolution: int
    partition: str
    gpus: int
    cpus: int
    mem: str
    walltime: str
    job_name: str
    job_id: Optional[str] = None

    @property
    def result_path(self) -> Path:
        return self.run_dir / self.result_filename


@dataclass
class SimulationResult:
    job_id: str
    run_id: str
    run_dir: Path
    result_path: Path
    raw_value: str
    numeric_value: Optional[float]
    state: str
    finished_at: str


def resolve_geometry(name: str, project_root: Path) -> Path:
    """Locate the geometry file ignoring case within known directories."""
    candidate = Path(name)
    if candidate.is_absolute() and candidate.is_file():
        return candidate

    relative_candidate = project_root / candidate
    if relative_candidate.is_file():
        return relative_candidate

    target_lower = candidate.name.lower()
    search_roots: Iterable[Path] = (
        project_root / "sim_2D",
        project_root / "sim_2D" / "ellipses",
        project_root / "geometries",
    )
    for root in search_roots:
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and path.name.lower() == target_lower:
                    return path

    raise FileNotFoundError(
        f"Geometry file '{name}' not found (case-insensitive search)."
    )


def ensure_ascii(text: str) -> str:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Non-ASCII text encountered where ASCII required") from exc
    return text


def build_sbatch_script(
    *,
    project_root: Path,
    run_id: str,
    run_dir: Path,
    staged_geometry: Path,
    resolution: int,
    partition: str,
    gpus: int,
    cpus: int,
    mem: str,
    walltime: str,
    job_name: str,
    result_filename: str,
) -> str:
    stdout_name = "stdout.log"
    stderr_name = "stderr.log"

    project_root_quoted = shlex.quote(str(project_root))
    run_script_quoted = shlex.quote(str(project_root / "sim_2D" / "run"))
    geometry_name = staged_geometry.name
    geometry_name_quoted = shlex.quote(geometry_name)
    geometry_abs_quoted = shlex.quote(str(staged_geometry))
    value_basename = f"value_{geometry_name}"
    value_source = Path("sim_2D") / "values" / value_basename
    value_source_abs_quoted = shlex.quote(str(project_root / value_source))
    result_name_quoted = shlex.quote(result_filename)

    script = f"""#!/bin/bash
#SBATCH --job-name={ensure_ascii(job_name)}
#SBATCH --output={stdout_name}
#SBATCH --error={stderr_name}
#SBATCH --time={ensure_ascii(walltime)}
#SBATCH --partition={ensure_ascii(partition)}
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={ensure_ascii(mem)}

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PROJECT_ROOT={project_root_quoted}
RUN_SCRIPT={run_script_quoted}
GEOMETRY_FILE={geometry_name_quoted}
GEOMETRY_ABS={geometry_abs_quoted}
VALUE_SOURCE={value_source_abs_quoted}
RESULT_FILE={result_name_quoted}

rm -f "$VALUE_SOURCE"

"$RUN_SCRIPT" sim2d_2 {resolution} "$GEOMETRY_ABS"

if [ ! -f "$VALUE_SOURCE" ]; then
    echo "Result file $VALUE_SOURCE not produced" >&2
    exit 2
fi

mv "$VALUE_SOURCE" "$RESULT_FILE"

echo "Stored result in $RESULT_FILE"
"""
    return script


def submit_job(sbatch_path: Path) -> str:
    proc = subprocess.run(
        ["sbatch", "--parsable", sbatch_path.name],
        cwd=sbatch_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip().split(";", 1)[0]


def make_manifest(run_dir: Path, data: dict) -> Path:
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return manifest_path


def update_manifest(run_dir: Path, updates: dict) -> dict:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {}
    data.update(updates)
    make_manifest(run_dir, data)
    return data


def generate_run_id() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"run-{timestamp}-{suffix}"


def query_job_state(job_id: str) -> str:
    def first_nonempty(text: str) -> Optional[str]:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    squeue = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        capture_output=True,
        text=True,
    )
    if squeue.returncode == 0:
        state = first_nonempty(squeue.stdout)
        if state:
            return state.upper()

    sacct = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--parsable2", "--noheader"],
        capture_output=True,
        text=True,
    )
    if sacct.returncode == 0:
        state = first_nonempty(sacct.stdout)
        if state:
            primary = state.split("|", 1)[0]
            return primary.upper()

    return "UNKNOWN"


def wait_for_job_completion(
    job_id: str,
    *,
    poll_interval: float = 30.0,
    timeout: Optional[float] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    poll_interval = max(poll_interval, 1.0)
    deadline = time.monotonic() + timeout if timeout else None
    last_state = ""
    while True:
        state = query_job_state(job_id)
        if state and state != last_state and progress_callback:
            progress_callback(state)
        if state:
            last_state = state
        if state in COMPLETED_STATES:
            return state
        if state in FAILED_STATES:
            return state
        if deadline and time.monotonic() > deadline:
            raise TimeoutError(f"Timed out while waiting for job {job_id}.")
        time.sleep(poll_interval)


def wait_for_result_file(
    path: Path,
    *,
    poll_interval: float = 5.0,
    timeout: Optional[float] = None,
) -> Path:
    poll_interval = max(poll_interval, 1.0)
    deadline = time.monotonic() + timeout if timeout else None
    while True:
        if path.exists():
            return path
        if deadline and time.monotonic() > deadline:
            raise TimeoutError(f"Timed out while waiting for result file {path}.")
        time.sleep(poll_interval)


def read_result_file(path: Path) -> tuple[str, Optional[float]]:
    raw = path.read_text(encoding="ascii").strip()
    try:
        numeric = float(raw)
    except ValueError:
        numeric = None
    return raw, numeric


def prepare_submission(
    geometry: str,
    resolution: int,
    *,
    partition: str,
    walltime: str,
    gpus: int,
    cpus: int,
    mem: str,
    runs_root: str,
    job_name: Optional[str],
) -> Submission:
    project_root = Path(__file__).resolve().parent
    geometry_path = resolve_geometry(geometry, project_root)

    run_id = generate_run_id()
    run_root = project_root / runs_root
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)

    staged_geometry_name = f"{run_id}_{geometry_path.name}"
    staged_geometry_path = run_dir / staged_geometry_name
    shutil.copy2(geometry_path, staged_geometry_path)

    result_filename = f"tke_{run_id}.txt"
    unique_suffix = run_id.rsplit("-", 1)[-1]
    actual_job_name = job_name or f"lbm-{unique_suffix}"
    sbatch_text = build_sbatch_script(
        project_root=project_root,
        run_id=run_id,
        run_dir=run_dir,
        staged_geometry=staged_geometry_path,
        resolution=resolution,
        partition=partition,
        gpus=gpus,
        cpus=cpus,
        mem=mem,
        walltime=walltime,
        job_name=actual_job_name,
        result_filename=result_filename,
    )

    sbatch_path = run_dir / "job.sbatch"
    sbatch_path.write_text(sbatch_text, encoding="ascii")

    prepared_at = datetime.utcnow().isoformat() + "Z"
    manifest = {
        "run_id": run_id,
        "geometry_source": str(geometry_path),
        "staged_geometry": staged_geometry_path.name,
        "resolution": resolution,
        "partition": partition,
        "gpus": gpus,
        "cpus": cpus,
        "mem": mem,
        "walltime": walltime,
        "result_file": result_filename,
        "prepared_at": prepared_at,
        "job_name": actual_job_name,
    }
    manifest_path = make_manifest(run_dir, manifest)

    return Submission(
        project_root=project_root,
        run_id=run_id,
        run_dir=run_dir,
        geometry_source=geometry_path,
        staged_geometry=staged_geometry_path,
        result_filename=result_filename,
        sbatch_path=sbatch_path,
        manifest_path=manifest_path,
        resolution=resolution,
        partition=partition,
        gpus=gpus,
        cpus=cpus,
        mem=mem,
        walltime=walltime,
        job_name=actual_job_name,
    )


def submit_prepared(submission: Submission, *, dry_run: bool = False) -> Submission:
    if dry_run:
        return submission

    job_id = submit_job(submission.sbatch_path)
    submission.job_id = job_id
    update_manifest(
        submission.run_dir,
        {
            "job_id": job_id,
            "submitted_at": datetime.utcnow().isoformat() + "Z",
        },
    )
    return submission


def collect_submission(
    submission: Submission,
    *,
    poll_interval: float = 30.0,
    timeout: Optional[float] = None,
    result_timeout: Optional[float] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> SimulationResult:
    if not submission.job_id:
        raise ValueError("Cannot collect submission without a job id.")

    final_state = wait_for_job_completion(
        submission.job_id,
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
    )
    finished_at = datetime.utcnow().isoformat() + "Z"
    update_manifest(
        submission.run_dir,
        {
            "final_state": final_state,
            "finished_at": finished_at,
        },
    )

    if final_state not in COMPLETED_STATES:
        raise RuntimeError(f"Job {submission.job_id} ended with state {final_state}.")

    wait_for_result_file(
        submission.result_path,
        poll_interval=5.0,
        timeout=result_timeout,
    )

    raw, numeric = read_result_file(submission.result_path)
    update_manifest(submission.run_dir, {"result_value": raw})

    return SimulationResult(
        job_id=submission.job_id,
        run_id=submission.run_id,
        run_dir=submission.run_dir,
        result_path=submission.result_path,
        raw_value=raw,
        numeric_value=numeric,
        state=final_state,
        finished_at=finished_at,
    )


def submit_and_collect(
    geometry: str,
    resolution: int,
    *,
    partition: str = "gp",
    walltime: str = "01:00:00",
    gpus: int = 1,
    cpus: int = 4,
    mem: str = "16G",
    runs_root: str = "lbm_runs",
    job_name: Optional[str] = None,
    poll_interval: float = 30.0,
    timeout: Optional[float] = None,
    result_timeout: Optional[float] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> SimulationResult:
    submission = prepare_submission(
        geometry=geometry,
        resolution=resolution,
        partition=partition,
        walltime=walltime,
        gpus=gpus,
        cpus=cpus,
        mem=mem,
        runs_root=runs_root,
        job_name=job_name,
    )
    submission = submit_prepared(submission)
    return collect_submission(
        submission,
        poll_interval=poll_interval,
        timeout=timeout,
        result_timeout=result_timeout,
        progress_callback=progress_callback,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit sim2d_2 runs to Slurm and manage per-run outputs.",
    )
    parser.add_argument("geometry", help="Geometry file name or path (case-insensitive search).")
    parser.add_argument("resolution", type=int, help="Simulation resolution (e.g. 8).")
    parser.add_argument("--partition", default="gp", help="Slurm partition name (default: gp).")
    parser.add_argument("--walltime", default="10:00:00", help="Walltime in HH:MM:SS (default: 01:00:00).")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1).")
    parser.add_argument("--cpus", type=int, default=4, help="CPUs (default: 4).")
    parser.add_argument("--mem", default="16G", help="Memory request (default: 16G).")
    parser.add_argument(
        "--job-name",
        default=None,
        help="Custom job name; defaults to a generated value.",
    )
    parser.add_argument(
        "--runs-root",
        default="lbm_runs",
        help="Directory (relative to repo root) where run folders are created (default: lbm_runs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare run directory without submitting to Slurm.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for completion and print the resulting TKE value.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Polling interval (seconds) while waiting for the job (default: 30).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Abort waiting after this many seconds (default: no timeout).",
    )
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=None,
        help="After job completion, wait this many seconds for the result file (default: unlimited).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    try:
        submission = prepare_submission(
            geometry=args.geometry,
            resolution=args.resolution,
            partition=args.partition,
            walltime=args.walltime,
            gpus=args.gpus,
            cpus=args.cpus,
            mem=args.mem,
            runs_root=args.runs_root,
            job_name=args.job_name,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to prepare submission: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Prepared run directory {submission.run_dir} (dry run)")
        print(f"Geometry staged as {submission.staged_geometry.name}")
        print(f"Job script: {submission.sbatch_path.name}")
        return 0

    try:
        submission = submit_prepared(submission)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else str(exc)
        print(stderr, file=sys.stderr)
        return exc.returncode or 1

    print("Submitted job", submission.job_id)
    print("Run directory:", submission.run_dir)
    print("Result will be copied to:", submission.result_path)
    print("Monitor with: squeue -j", submission.job_id)
    print("After completion, see:", submission.run_dir / "stdout.log")

    if not args.wait:
        return 0

    print("Waiting for job completion...")

    def on_state_change(state: str) -> None:
        print("  state ->", state)

    try:
        result = collect_submission(
            submission,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            result_timeout=args.result_timeout,
            progress_callback=on_state_change,
        )
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    if result.numeric_value is not None:
        print("Final TKE:", result.numeric_value)
    else:
        print("Final TKE (raw text):", result.raw_value)

    print("Result stored at:", result.result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
