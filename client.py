#!/usr/bin/env python3
"""Simple client to call the interpretable-ml Flask API.

Usage examples:
  python client.py \
    --input-path /abs/path/input.json \
    --instances-path /abs/path/ecg_tracings.hdf5 \
    --ground-truth-path /abs/path/labels.csv \
    --instance-idx 0

Notes:
- Reads files locally and sends content to the API.
- Requires `h5py` to load HDF5 instances.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import h5py
import pandas as pd

EXPLAIN_URL = "http://localhost:5001/explain"


def _ensure_http_url(url: str) -> None:
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with 'http://' or 'https://'")


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    _ensure_http_url(url)
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")  # noqa: S310
    with urlopen(req) as resp:  # noqa: S310
        charset = resp.headers.get_content_charset() or "utf-8"
        body = resp.read().decode(charset)
        return json.loads(body)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Client for interpretable-ml /explain endpoint")
    p.add_argument("--input-path", required=True, help="Absolute path to config JSON")
    p.add_argument("--instances-path", required=True, help="Absolute path to HDF5 with 'tracings'")
    p.add_argument("--ground-truth-path", default=None, help="Absolute path to CSV with labels")
    p.add_argument("--instance-idx", type=int, default=0, help="Index of the instance (default: 0)")
    return p.parse_args()


def _load_input_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_instances_hdf5(path: Path) -> list:
    with h5py.File(path, "r") as f:
        if "tracings" not in f:
            raise ValueError("HDF5 must contain a 'tracings' dataset")
        data = f["tracings"][()]
    return data.tolist()


def _load_ground_truth_csv(path: Path) -> list:
    ground_truth = pd.read_csv(path)
    return ground_truth.values.tolist()


def main() -> int:
    args = parse_args()

    # Validate file paths exist locally
    for label, path in (
        ("input_path", args.input_path),
        ("instances_path", args.instances_path),
    ):
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            return 2
    if args.ground_truth_path and not Path(args.ground_truth_path).exists():
        print(f"Error: ground_truth_path not found: {args.ground_truth_path}", file=sys.stderr)
        return 2

    # Load files and build content payload
    try:
        input_config = _load_input_config(Path(args.input_path))
        instances = _load_instances_hdf5(Path(args.instances_path))
        ground_truth = (
            _load_ground_truth_csv(Path(args.ground_truth_path)) if args.ground_truth_path else None
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading inputs: {exc}", file=sys.stderr)
        return 2

    payload: dict[str, object] = {
        "input_config": input_config,
        "instances": instances,
        "instance_idx": args.instance_idx,
    }
    if ground_truth is not None:
        payload["ground_truth"] = ground_truth

    explain_url = EXPLAIN_URL
    try:
        resp = _post_json(explain_url, payload)
    except Exception as exc:  # noqa: BLE001
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    status = resp.get("status")
    print(json.dumps(resp, indent=2))

    if status != "success":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
