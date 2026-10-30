#!/usr/bin/env python3
"""Download pinned V2 weights; reuse frozen base files only after hash checks.

No accelerator is used. The native VAE snapshot is populated for offline Omni
loading; the V2 transformer remains in its released Diffusers layout.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def matches(path, spec):
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        return False
    digest = hashlib.sha256() if "sha256" in spec else hashlib.sha1()
    if "git_blob_sha1" in spec:
        digest.update(f"blob {spec['bytes']}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == spec.get("sha256", spec.get("git_blob_sha1"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, help="Existing native MiniMax-H3 root or its FL2VA directory")
    parser.add_argument("--native-cache", type=Path, required=True, help="HF_HUB_CACHE for native VAE snapshot")
    parser.add_argument("--endpoint", default=os.getenv("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    spec = json.loads((Path(__file__).resolve().parents[1] / "configs/fasth3-v2.json").read_text())
    args.output = args.output.resolve()
    args.native_cache = args.native_cache.resolve()
    base = args.base.resolve() if args.base else None
    if base and (base / "FL2VA").is_dir():
        base = base / "FL2VA"
    native = spec["native_vaes"]
    native_root = args.native_cache / ("models--" + native["repo_id"].replace("/", "--")) / "snapshots" / native["revision"]

    def ensure(job):
        entry, root, repo, revision, candidate = job
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if matches(target, entry):
            return dict(**entry, source="verified-existing")
        if target.is_symlink():
            raise ValueError(f"Existing symlink does not match pinned weights: {target}")
        if candidate and matches(candidate, entry):
            target.symlink_to(candidate.resolve())
            print("VERIFIED_BASE_REUSE", entry["path"], flush=True)
            return dict(**entry, source="verified-base", original=str(candidate.resolve()))
        partial = target.with_suffix(target.suffix + ".partial")
        if matches(partial, entry):
            partial.replace(target)
            return dict(**entry, source="verified-complete-partial")
        for attempt in range(8):
            # Fixed revision and final digest pin the bytes. Fresh query avoids
            # stale CDN signed redirects, without trusting a moving branch.
            url = f"{args.endpoint.rstrip('/')}/{repo}/resolve/{revision}/{entry['path']}?download=true&attempt={time.time_ns()}"
            print("DOWNLOAD", entry["path"], "attempt", attempt + 1, flush=True)
            result = subprocess.run(["curl", "-fLsS", "--connect-timeout", "20", "--speed-limit", "1024",
                "--speed-time", "120", "--retry", "2", "--retry-delay", "5", "--continue-at", "-",
                "--output", str(partial), url])
            if result.returncode == 0 and matches(partial, entry):
                partial.replace(target)
                print("VERIFIED_DOWNLOAD", entry["path"], flush=True)
                return dict(**entry, source="verified-download")
            if partial.exists() and partial.stat().st_size >= entry["bytes"]:
                raise ValueError(f"Downloaded bytes failed pinned digest: {target}")
            time.sleep(min(5 * (attempt + 1), 30))
        raise RuntimeError(f"Download failed: {entry['path']}")

    jobs = []
    for entry in spec["files"]:
        # The student DiT must always come from V2, never the original model.
        candidate = base / entry["path"] if base and entry["path"].startswith(("text_encoder/", "tokenizer/", "processor/")) else None
        jobs.append((entry, args.output, spec["repo_id"], spec["revision"], candidate))
    for entry in native["files"]:
        candidate = base / entry["path"].removeprefix("FL2VA/") if base else None
        jobs.append((entry, native_root, native["repo_id"], native["revision"], candidate))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(ensure, jobs))
    receipt = dict(repo_id=spec["repo_id"], revision=spec["revision"], steps=8, sparsity=.8,
                   native_vae_revision=native["revision"], native_cache=str(args.native_cache), records=records)
    (args.output / "ascend-h3-download-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("ALL_FILES_HASH_VERIFIED", len(records), flush=True)
    print(f"Use HF_HUB_CACHE={args.native_cache} HF_HUB_OFFLINE=1 for fully local loading.")


if __name__ == "__main__":
    main()
