#!/usr/bin/env python3
"""Batch submit LBM simulations for numbered geometries and collect TKE results."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import run_lbm_simulation


@dataclass
class GeometryRecord:
    geometry_number: int
    geometry_name: str
    run_id: str = ""
    job_id: str = ""
    tke_value: str = ""
    state: str = "PENDING"
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit run_lbm_simulation jobs for a range of numbered geometries "
            "and collect their TKE integrals into a CSV file."
        )
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First geometry number to include (default: 0).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=180,
        help="Last geometry number to include (inclusive, default: 180).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lbm_runs") / "batch_results.csv",
        help="Path to the CSV file to produce (default: lbm_runs/batch_results.csv).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=8,
        help="Simulation resolution to pass to run_lbm_simulation (default: 8).",
    )
    parser.add_argument(
        "--partition",
        default="gp",
        help="Slurm partition (default: gp).",
    )
    parser.add_argument(
        "--walltime",
        default="10:00:00",
        help="Walltime for each simulation job (default: 10:00:00).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs per job (default: 1).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Number of CPUs per job (default: 4).",
    )
    parser.add_argument(
        "--mem",
        default="16G",
        help="Memory request per job (default: 16G).",
    )
    parser.add_argument(
        "--runs-root",
        default="lbm_runs",
        help="Directory (relative to repo root) where job run folders are stored (default: lbm_runs).",
    )
    parser.add_argument(
        "--type1-bouzidi",
        choices=["auto", "on", "off"],
        default="on",
        help="Value for --type1-bouzidi passed to run_lbm_simulation (default: on).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between job status polls while waiting (default: 30).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for individual job completion (default: unlimited).",
    )
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for the result file after job completion (default: unlimited).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and submit jobs but do not wait for completion or write CSV.",
    )
    return parser.parse_args()


def submit_geometries(
    args: argparse.Namespace,
) -> tuple[list[GeometryRecord], list[tuple[GeometryRecord, run_lbm_simulation.Submission]]]:
    records: list[GeometryRecord] = []
    submissions: list[tuple[GeometryRecord, run_lbm_simulation.Submission]] = []

    for geometry_number in range(args.start, args.end + 1):
        geometry_name = f"{geometry_number}.txt"
        record = GeometryRecord(
            geometry_number=geometry_number,
            geometry_name=geometry_name,
        )
        records.append(record)
        print(f"[submit] Preparing geometry {geometry_name}")
        try:
            submission = run_lbm_simulation.prepare_submission(
                geometry=geometry_name,
                resolution=args.resolution,
                partition=args.partition,
                walltime=args.walltime,
                gpus=args.gpus,
                cpus=args.cpus,
                mem=args.mem,
                runs_root=args.runs_root,
                job_name=None,
                type1_bouzidi=args.type1_bouzidi,
            )
        except FileNotFoundError as exc:
            record.state = "MISSING_GEOMETRY"
            record.error = str(exc)
            print(f"  ! Geometry not found: {exc}", file=sys.stderr)
            continue
        except Exception as exc:  # noqa: BLE001
            record.state = "PREPARE_FAILED"
            record.error = str(exc)
            print(f"  ! Failed to prepare submission: {exc}", file=sys.stderr)
            continue

        record.run_id = submission.run_id
        try:
            submission = run_lbm_simulation.submit_prepared(
                submission,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            record.state = "SUBMIT_FAILED"
            record.error = str(exc)
            print(f"  ! Failed to submit job for {geometry_name}: {exc}", file=sys.stderr)
            continue

        if args.dry_run:
            record.state = "SUBMITTED_DRY_RUN"
            submissions.append((record, submission))
            print(f"  -> prepared (dry run) run_id={record.run_id}")
            continue

        if submission.job_id is None:
            record.state = "NO_JOB_ID"
            record.error = "Job ID missing after submission."
            print(f"  ! Job ID missing for {geometry_name}", file=sys.stderr)
            continue

        record.job_id = submission.job_id
        record.state = "SUBMITTED"
        submissions.append((record, submission))
        print(f"  -> submitted job_id={record.job_id} run_id={record.run_id}")

    return records, submissions


def collect_results(
    submissions: list[tuple[GeometryRecord, run_lbm_simulation.Submission]],
    *,
    poll_interval: float,
    timeout: Optional[float],
    result_timeout: Optional[float],
) -> None:
    for record, submission in submissions:
        if submission.job_id is None:
            continue
        print(
            f"[wait] geometry={record.geometry_name} job_id={submission.job_id} run_id={record.run_id}"
        )

        def on_state_change(state: str, *, geom: str = record.geometry_name) -> None:
            print(f"    state -> {state} ({geom})")

        try:
            result = run_lbm_simulation.collect_submission(
                submission,
                poll_interval=poll_interval,
                timeout=timeout,
                result_timeout=result_timeout,
                progress_callback=on_state_change,
            )
        except TimeoutError as exc:
            record.state = "TIMEOUT"
            record.error = str(exc)
            print(f"  ! Timeout waiting for job {submission.job_id}", file=sys.stderr)
        except RuntimeError as exc:
            record.state = "JOB_FAILED"
            record.error = str(exc)
            print(f"  ! Job {submission.job_id} failed: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            record.state = "COLLECT_ERROR"
            record.error = str(exc)
            print(f"  ! Unexpected error collecting job {submission.job_id}: {exc}", file=sys.stderr)
        else:
            record.state = result.state
            record.tke_value = result.raw_value
            print(
                f"  -> completed state={result.state} tke={record.tke_value}"
            )


def write_csv(records: list[GeometryRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["geometry_number", "tke_integral", "run_id"])
        for record in sorted(records, key=lambda r: r.geometry_number):
            writer.writerow(
                [
                    record.geometry_number,
                    record.tke_value,
                    record.run_id,
                ]
            )
    print(f"[done] Wrote results to {output_path}")


def main() -> int:
    args = parse_args()
    if args.start > args.end:
        print("--start must be <= --end", file=sys.stderr)
        return 2

    records, submissions = submit_geometries(args)

    if args.dry_run:
        print(
            f"Prepared {len(submissions)} submissions (dry run); skipping wait and CSV output."
        )
        return 0

    collect_results(
        submissions,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        result_timeout=args.result_timeout,
    )

    write_csv(records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
