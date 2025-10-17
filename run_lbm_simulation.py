#!/usr/bin/env python3
"""Submit and collect 2D LBM simulations via Slurm or locally.

This script is intended to be both an importable helper module and a small CLI.
The typical lifecycle is:

1. Locate and copy a geometry file into a dedicated run directory.
2. Generate an ``sbatch`` file that launches ``sim_2D/run`` with the requested
   resolution and optional solver flags, or run the solver locally.
3. Submit the job to Slurm (or execute it immediately) and optionally wait
   until a numerical result is produced.

The CLI exposes these steps with sensible defaults::

    python run_lbm_simulation.py 32.txt 8 --wait

For programmatic use, ``prepare_submission`` returns a :class:`Submission`
object that can be fed into ``submit_prepared`` and ``collect_submission``.

The solver binary must be built ahead of time, for example::

    cmake -S . -B build
    cmake --build build --target sim2d_2

If the binary is missing, submissions abort before hitting the queue.
"""

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

EXECUTOR_SLURM = "slurm"
EXECUTOR_LOCAL = "local"
DEFAULT_SIMULATION_TARGET = "sim2d_2"

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
    """Describe the on-disk artefacts and scheduler metadata for a run."""

    project_root: Path
    run_id: str
    run_dir: Path
    geometry_source: Path
    staged_geometry: Path
    result_filename: str
    manifest_path: Path
    executor: str
    run_script: Path
    stdout_path: Path
    stderr_path: Path
    simulation_target: str
    sbatch_path: Optional[Path] = None
    resolution: int
    partition: str
    gpus: int
    cpus: int
    mem: str
    walltime: str
    job_name: str
    type1_bouzidi: str
    job_id: Optional[str] = None
    state: Optional[str] = None
    finished_at: Optional[str] = None

    @property
    def result_path(self) -> Path:
        return self.run_dir / self.result_filename


@dataclass
class SimulationResult:
    """Capture the essential data produced by a finished run."""

    job_id: str
    run_id: str
    run_dir: Path
    result_path: Path
    raw_value: str
    numeric_value: Optional[float]
    state: str
    finished_at: str


def resolve_geometry(name: str, project_root: Path) -> Path:
    """Locate a geometry file, performing a case-insensitive search."""
    candidate = Path(name)
    if candidate.is_absolute() and candidate.is_file():
        return candidate

    relative_candidate = project_root / candidate
    if relative_candidate.is_file():
        return relative_candidate

    target_lower = candidate.name.lower()
    search_roots: Iterable[Path] = (
        project_root / "sim_2D",
        project_root / "sim_2D" / "geometries",
        project_root / "sim_2D" / "ellipses",
        project_root / "geometries",
    )
    for root in search_roots:
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and path.name.lower() == target_lower:
                    return path
    search_list = ", ".join(str(root) for root in search_roots if root.exists())
    raise FileNotFoundError(
        f"Geometry file '{name}' not found (case-insensitive search). "
        f"Searched: {search_list or 'no geometry directories present'}."
    )


def ensure_ascii(text: str) -> str:
    """Validate that ``text`` contains ASCII characters only."""
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Non-ASCII text encountered where ASCII required") from exc
    return text


def default_run_script(project_root: Path) -> Path:
    """Return the canonical launcher script that builds and runs solver targets."""
    return project_root / "sim_2D" / "run"


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
    type1_bouzidi: str,
    run_script: Path,
    simulation_target: str,
) -> str:
    """Return the contents of the ``sbatch`` script for a single run."""
    stdout_name = "stdout.log"
    stderr_name = "stderr.log"

    project_root_quoted = shlex.quote(str(project_root))
    geometry_name = staged_geometry.name
    geometry_name_quoted = shlex.quote(geometry_name)
    geometry_abs_quoted = shlex.quote(str(staged_geometry))
    value_basename = f"value_{geometry_name}"
    value_source = Path("sim_2D") / "values" / value_basename
    value_source_abs_quoted = shlex.quote(str(project_root / value_source))
    result_name_quoted = shlex.quote(result_filename)
    type1_mode = ensure_ascii(type1_bouzidi)
    type1_mode_quoted = shlex.quote(type1_mode)
    run_script_quoted = shlex.quote(str(run_script))
    simulation_target = ensure_ascii(simulation_target)
    simulation_target_quoted = shlex.quote(simulation_target)

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
SIM_TARGET={simulation_target_quoted}
GEOMETRY_FILE={geometry_name_quoted}
GEOMETRY_ABS={geometry_abs_quoted}
VALUE_SOURCE={value_source_abs_quoted}
RESULT_FILE={result_name_quoted}
TYPE1_BOUZIDI={type1_mode_quoted}

EXTRA_ARGS=()
if [ "$TYPE1_BOUZIDI" != "auto" ]; then
    EXTRA_ARGS+=(--type1-bouzidi "$TYPE1_BOUZIDI")
fi

rm -f "$VALUE_SOURCE"

# the launcher ensures the target is built before execution
if [ ! -x "$RUN_SCRIPT" ]; then
    echo "Launcher script $RUN_SCRIPT is missing or not executable." >&2
    exit 3
fi

# run the solver from the repo root so values land under sim_2D/values
(
    cd "$PROJECT_ROOT"
    "$RUN_SCRIPT" "$SIM_TARGET" {resolution} "$GEOMETRY_ABS" "${{EXTRA_ARGS[@]}}"
)

if [ ! -f "$VALUE_SOURCE" ]; then
    echo "Result file $VALUE_SOURCE not produced" >&2
    exit 2
fi

mv "$VALUE_SOURCE" "$RESULT_FILE"

echo "Stored result in $RESULT_FILE"
"""
    return script


def iso_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def create_run_directory(project_root: Path, runs_root: str, run_id: str) -> Path:
    """Create and return the fresh run directory."""
    run_root = project_root / runs_root
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def stage_geometry_file(geometry_path: Path, run_dir: Path, run_id: str) -> Path:
    """Copy the geometry into the run directory with a unique prefix."""
    staged_name = f"{run_id}_{geometry_path.name}"
    staged_path = run_dir / staged_name
    shutil.copy2(geometry_path, staged_path)
    return staged_path


def write_sbatch(run_dir: Path, script_text: str) -> Path:
    """Persist the generated ``sbatch`` script."""
    sbatch_path = run_dir / "job.sbatch"
    sbatch_path.write_text(script_text, encoding="ascii")
    return sbatch_path


def default_job_name(run_id: str) -> str:
    """Create a short, stable job name from a run identifier."""
    unique_suffix = run_id.rsplit("-", 1)[-1]
    return f"lbm-{unique_suffix}"


def submit_job(sbatch_path: Path) -> str:
    """Submit the given ``sbatch`` script and return the Slurm job id."""
    proc = subprocess.run(
        ["sbatch", "--parsable", sbatch_path.name],
        cwd=sbatch_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip().split(";", 1)[0]


def make_manifest(run_dir: Path, data: dict) -> Path:
    """Write the manifest JSON file and return its path."""
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return manifest_path


def update_manifest(run_dir: Path, updates: dict) -> dict:
    """Merge updates into the manifest JSON file and return the combined data."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {}
    data.update(updates)
    make_manifest(run_dir, data)
    return data


def generate_run_id() -> str:
    """Create a monotonic, collisions-resistant identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"run-{timestamp}-{suffix}"


def query_job_state(job_id: str) -> str:
    """Return Slurm's best-known state for ``job_id``."""
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
    """Poll Slurm until the job reaches a terminal state."""
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
    """Wait until ``path`` exists on disk."""
    poll_interval = max(poll_interval, 1.0)
    deadline = time.monotonic() + timeout if timeout else None
    while True:
        if path.exists():
            return path
        if deadline and time.monotonic() > deadline:
            raise TimeoutError(f"Timed out while waiting for result file {path}.")
        time.sleep(poll_interval)


def read_result_file(path: Path) -> tuple[str, Optional[float]]:
    """Return the raw text and optional float value stored in ``path``."""
    raw = path.read_text(encoding="ascii").strip()
    try:
        numeric = float(raw)
    except ValueError:
        numeric = None
    return raw, numeric


def execute_local(submission: Submission) -> None:
    """Run the sim_2D launcher directly and capture artefacts locally."""
    values_dir = submission.project_root / "sim_2D" / "values"
    values_dir.mkdir(parents=True, exist_ok=True)
    value_source = values_dir / f"value_{submission.staged_geometry.name}"
    if value_source.exists():
        value_source.unlink()

    cmd = [
        str(submission.run_script),
        submission.simulation_target,
        str(submission.resolution),
        str(submission.staged_geometry),
    ]
    if submission.type1_bouzidi != "auto":
        cmd.extend(["--type1-bouzidi", submission.type1_bouzidi])

    submitted_at = iso_timestamp()
    update_manifest(
        submission.run_dir,
        {
            "executor": submission.executor,
            "job_id": submission.job_id,
            "submitted_at": submitted_at,
        },
    )

    with submission.stdout_path.open("w", encoding="utf-8") as stdout_file, submission.stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        proc = subprocess.run(
            cmd,
            cwd=submission.project_root,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
        )

    finished_at = iso_timestamp()
    manifest_updates = {"finished_at": finished_at}
    if proc.returncode != 0:
        submission.state = f"FAILED_EXIT_{proc.returncode}"
        submission.finished_at = finished_at
        manifest_updates["final_state"] = submission.state
        manifest_updates["exit_code"] = proc.returncode
        update_manifest(submission.run_dir, manifest_updates)
        raise RuntimeError(
            f"Local solver exited with code {proc.returncode}. "
            f"Inspect '{submission.stderr_path}' for details."
        )

    if not value_source.exists():
        submission.state = "FAILED_NO_RESULT"
        submission.finished_at = finished_at
        manifest_updates["final_state"] = submission.state
        update_manifest(submission.run_dir, manifest_updates)
        raise RuntimeError(
            f"Simulation completed but result file '{value_source}' is missing."
        )

    if submission.result_path.exists():
        submission.result_path.unlink()
    shutil.move(str(value_source), submission.result_path)

    submission.state = "COMPLETED"
    submission.finished_at = finished_at
    manifest_updates["final_state"] = submission.state
    update_manifest(submission.run_dir, manifest_updates)


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
    type1_bouzidi: str,
    executor: str = EXECUTOR_SLURM,
    simulation_target: str = DEFAULT_SIMULATION_TARGET,
) -> Submission:
    """Prepare on-disk artefacts for a single run, regardless of executor."""
    project_root = Path(__file__).resolve().parent
    geometry_path = resolve_geometry(geometry, project_root)
    normalized_executor = executor.lower()
    if normalized_executor not in {EXECUTOR_SLURM, EXECUTOR_LOCAL}:
        raise ValueError(
            f"Unsupported executor '{executor}'. Expected '{EXECUTOR_SLURM}' or '{EXECUTOR_LOCAL}'."
        )
    simulation_target = ensure_ascii(simulation_target)
    if not simulation_target:
        raise ValueError("Simulation target must be a non-empty string.")

    run_script = default_run_script(project_root)
    if not run_script.is_file():
        raise FileNotFoundError(
            f"Launcher script '{run_script}' not found. Ensure submodules are initialised."
        )

    run_id = generate_run_id()
    run_dir = create_run_directory(project_root, runs_root, run_id)
    staged_geometry_path = stage_geometry_file(geometry_path, run_dir, run_id)

    result_filename = f"tke_{run_id}.txt"
    actual_job_name = job_name or default_job_name(run_id)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    sbatch_path: Optional[Path] = None
    if normalized_executor == EXECUTOR_SLURM:
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
            type1_bouzidi=type1_bouzidi,
            run_script=run_script,
            simulation_target=simulation_target,
        )
        sbatch_path = write_sbatch(run_dir, sbatch_text)

    prepared_at = iso_timestamp()
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
        "type1_bouzidi": type1_bouzidi,
        "executor": normalized_executor,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
        "run_script": str(run_script),
        "simulation_target": simulation_target,
    }
    manifest_path = make_manifest(run_dir, manifest)

    return Submission(
        project_root=project_root,
        run_id=run_id,
        run_dir=run_dir,
        geometry_source=geometry_path,
        staged_geometry=staged_geometry_path,
        result_filename=result_filename,
        manifest_path=manifest_path,
        executor=normalized_executor,
        run_script=run_script,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        simulation_target=simulation_target,
        sbatch_path=sbatch_path,
        resolution=resolution,
        partition=partition,
        gpus=gpus,
        cpus=cpus,
        mem=mem,
        walltime=walltime,
        job_name=actual_job_name,
        type1_bouzidi=type1_bouzidi,
    )


def submit_prepared(submission: Submission, *, dry_run: bool = False) -> Submission:
    """Submit a prepared run directory to Slurm unless ``dry_run`` is set."""
    if dry_run:
        return submission

    if submission.executor == EXECUTOR_LOCAL:
        submission.job_id = submission.job_id or f"{EXECUTOR_LOCAL}-{submission.run_id}"
        execute_local(submission)
        return submission

    if submission.sbatch_path is None:
        raise ValueError("Missing Slurm submission script for executor 'slurm'.")

    job_id = submit_job(submission.sbatch_path)
    submission.job_id = job_id
    update_manifest(
        submission.run_dir,
        {
            "job_id": job_id,
            "executor": submission.executor,
            "submitted_at": iso_timestamp(),
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
    """Block until the submitted job has finished and a result file is available."""
    if not submission.job_id:
        raise ValueError("Cannot collect submission without a job id.")

    if submission.executor == EXECUTOR_LOCAL:
        finished_at = submission.finished_at or iso_timestamp()
        state = submission.state or "COMPLETED"
        if not submission.result_path.exists():
            raise RuntimeError(f"Local run did not produce result file {submission.result_path}.")
        raw, numeric = read_result_file(submission.result_path)
        update_manifest(
            submission.run_dir,
            {
                "final_state": state,
                "finished_at": finished_at,
                "result_value": raw,
            },
        )
        return SimulationResult(
            job_id=submission.job_id,
            run_id=submission.run_id,
            run_dir=submission.run_dir,
            result_path=submission.result_path,
            raw_value=raw,
            numeric_value=numeric,
            state=state,
            finished_at=finished_at,
        )

    final_state = wait_for_job_completion(
        submission.job_id,
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
    )
    finished_at = iso_timestamp()
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
    type1_bouzidi: str = "auto",
    poll_interval: float = 30.0,
    timeout: Optional[float] = None,
    result_timeout: Optional[float] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    executor: str = EXECUTOR_SLURM,
    simulation_target: str = DEFAULT_SIMULATION_TARGET,
) -> SimulationResult:
    """Convenience helper that prepares, submits, and collects in one call."""
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
        type1_bouzidi=type1_bouzidi,
        executor=executor,
        simulation_target=simulation_target,
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
    """Return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run sim_2D targets either via Slurm submissions or directly on the local machine.",
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
        "--type1-bouzidi",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control mapping of geometry type 1 cells to Bouzidi near-wall (auto|on|off); 'auto' uses build default.",
    )
    parser.add_argument(
        "--executor",
        choices=[EXECUTOR_SLURM, EXECUTOR_LOCAL],
        default=EXECUTOR_SLURM,
        help="Execution backend: 'slurm' uses sbatch; 'local' runs the solver synchronously.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_SIMULATION_TARGET,
        help="Simulation target passed to sim_2D/run (default: sim2d_2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare run directory without executing the simulation.",
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
    """CLI entry point."""
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
            type1_bouzidi=args.type1_bouzidi,
            executor=args.executor,
            simulation_target=args.target,
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
        print(f"Executor: {submission.executor}")
        print(f"Target: {submission.simulation_target}")
        if submission.sbatch_path:
            print(f"Job script: {submission.sbatch_path.name}")
        else:
            print("Job script: (not used for local executor)")
        return 0

    try:
        submission = submit_prepared(submission)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else str(exc)
        print(stderr, file=sys.stderr)
        return exc.returncode or 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    if submission.executor == EXECUTOR_SLURM:
        print("Submitted job", submission.job_id)
        print("Run directory:", submission.run_dir)
        print("Target:", submission.simulation_target)
        print("Result will be copied to:", submission.result_path)
        print("Monitor with: squeue -j", submission.job_id)
        print("After completion, see:", submission.stdout_path)
    else:
        print("Executed local run", submission.job_id)
        print("Run directory:", submission.run_dir)
        print("Target:", submission.simulation_target)
        print("Stdout:", submission.stdout_path)
        print("Stderr:", submission.stderr_path)
        print("Result stored at:", submission.result_path)

    should_wait = args.wait or submission.executor == EXECUTOR_LOCAL
    if not should_wait:
        return 0

    progress_callback: Optional[Callable[[str], None]]
    if submission.executor == EXECUTOR_SLURM:
        print("Waiting for job completion...")

        def on_state_change(state: str) -> None:
            print("  state ->", state)

        progress_callback = on_state_change
    else:
        print("Collecting local result...")
        progress_callback = None

    try:
        result = collect_submission(
            submission,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            result_timeout=args.result_timeout,
            progress_callback=progress_callback,
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
