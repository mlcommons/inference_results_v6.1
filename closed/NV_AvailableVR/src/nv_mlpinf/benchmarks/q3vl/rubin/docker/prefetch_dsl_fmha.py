#!/usr/bin/env python3
"""Prefetch FlashInfer's pinned cute-dsl FMHA exports into FLASHINFER_CUBIN_DIR at image build time.

FlashInfer loads a cute-dsl FMHA export (<ArtifactPath.DSL_FMHA><cpu>/<sm>/<name>.so) through
get_artifact(), which downloads it on first use. Its bulk download (download_artifacts(), which
builds the flashinfer-cubin wheel) lists only *.cubin and *.h files, so the cubin wheel carries
just the checksums.txt manifests of this collection. This script runs FlashInfer's own per-file
path for every manifest entry, so the leaf ends up exactly as runtime downloads would leave it
and the image needs no network to load any export:

  flashinfer.attention.cute_dsl.fmha._get_checksums(sm)                manifest, sha256 = the pin's
  flashinfer.jit.cubin_loader.get_artifact(<leaf>/<name>.so, sha256)  each export, sha256-checked

The source is FlashInfer's default repository (its public cubin repository) unless --repository
names an http(s) mirror in the published layout. FLASHINFER_CUBIN_DIR is FlashInfer's own default, the
installed flashinfer_cubin package's cubins/ dir. Nothing is loaded (no GPU at build time).
An export that get_artifact() could not fetch (after FlashInfer's own 4 tries: 35-70 s of backoff
plus the requests, 148 s when DNS lookups hang) is tried RETRIES more times, RETRY_PAUSE_S apart,
since a build sandbox can lose DNS for minutes.

usage: prefetch_dsl_fmha.py --sm sm_107a --record OUT.json [--repository URL]
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

RETRIES = 3
RETRY_PAUSE_S = 60


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--repository", default="", help="an http(s) mirror of FlashInfer's cubin repository (default: FlashInfer's)")
    a = ap.parse_args()

    for var in ("FLASHINFER_CUBIN_DIR", "FLASHINFER_DSL_FMHA_LOCAL_DIR", "FLASHINFER_NO_DOWNLOAD",
                "FLASHINFER_CUBIN_CHECKSUM_DISABLED", "FLASHINFER_CUBINS_REPOSITORY"):
        if os.environ.get(var):
            sys.exit(f"prefetch: {var} must not be set")
    if a.repository:
        # Read once at import by flashinfer.jit.cubin_loader: set it before importing flashinfer.
        os.environ["FLASHINFER_CUBINS_REPOSITORY"] = a.repository.rstrip("/") + "/"

    import flashinfer
    import flashinfer.artifacts as artifacts
    import flashinfer_cubin
    from flashinfer.attention.cute_dsl import fmha
    from flashinfer.jit import cubin_loader
    from flashinfer.jit.env import FLASHINFER_CUBIN_DIR

    cubin_dir = pathlib.Path(FLASHINFER_CUBIN_DIR).resolve()
    pkg_dir = pathlib.Path(flashinfer_cubin.get_cubin_dir()).resolve()
    if cubin_dir != pkg_dir:
        sys.exit(f"prefetch: FLASHINFER_CUBIN_DIR {cubin_dir} is not the flashinfer_cubin package dir {pkg_dir}")
    cpu = fmha._get_cpu_arch()
    pin_path = artifacts.ArtifactPath.DSL_FMHA
    if fmha.DSL_FMHA_ARTIFACT_PATH != pin_path:
        sys.exit(f"prefetch: fmha uses {fmha.DSL_FMHA_ARTIFACT_PATH!r}, artifacts pins {pin_path!r}")
    manifest_sha256 = artifacts.CheckSumHash.DSL_FMHA_CHECKSUMS[cpu][a.sm]

    leaf_rel = os.path.join(pin_path, cpu, a.sm)
    leaf = cubin_dir / leaf_rel
    # The flashinfer-cubin wheel ships the checksums.txt manifest of every pinned DSL arch (never
    # a .so), so the leaf may already hold that one file. Accept exactly it, with the pinned hash.
    preexisting = sorted(p.name for p in leaf.iterdir()) if leaf.exists() else []
    if preexisting and (preexisting != ["checksums.txt"] or sha256(leaf / "checksums.txt") != manifest_sha256):
        sys.exit(f"prefetch: {leaf} is not empty: {preexisting[:5]}")

    checksums = fmha._get_checksums(a.sm)  # FlashInfer's manifest fetch (sha256-checked)
    if not checksums:
        sys.exit("prefetch: empty manifest")
    if sha256(leaf / "checksums.txt") != manifest_sha256:
        sys.exit("prefetch: manifest on disk does not match the pin")
    for name, want in sorted(checksums.items()):
        for attempt in range(RETRIES + 1):
            # b"" on failure; the download is written to a temp file and renamed, so a retry is clean.
            if cubin_loader.get_artifact(os.path.join(leaf_rel, f"{name}.so"), want):
                break
            if attempt == RETRIES:
                sys.exit(f"prefetch: get_artifact failed for {name}")
            print(f"prefetch: get_artifact failed for {name}, retry {attempt + 1}/{RETRIES} in {RETRY_PAUSE_S} s",
                  flush=True)
            time.sleep(RETRY_PAUSE_S)
        if sha256(leaf / f"{name}.so") != want:
            sys.exit(f"prefetch: sha256 mismatch on disk for {name}")

    files = sorted(p.name for p in leaf.iterdir())
    so = [f for f in files if f.endswith(".so")]
    other = [f for f in files if not f.endswith(".so") and f != "checksums.txt" and not f.endswith(".lock")]
    if len(so) != len(checksums) or other:
        sys.exit(f"prefetch: leaf has {len(so)} .so (manifest {len(checksums)}), unexpected files {other[:5]}")
    record = {
        "flashinfer_version": flashinfer.__version__,
        "flashinfer_cubin_version": flashinfer_cubin.__version__,
        "artifact_path": pin_path,
        "cpu_arch": cpu,
        "sm_arch": a.sm,
        "manifest_sha256": manifest_sha256,
        "so_count": len(so),
        "so_bytes": sum((leaf / f).stat().st_size for f in so),
        "manifest_from_cubin_wheel": preexisting == ["checksums.txt"],
        "leaf": str(leaf),
        "repository": cubin_loader.FLASHINFER_CUBINS_REPOSITORY,
        "method": "flashinfer fmha._get_checksums + cubin_loader.get_artifact (runtime download path)",
    }
    os.makedirs(os.path.dirname(a.record), exist_ok=True)
    with open(a.record, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(record, indent=1, sort_keys=True))
    print(f"DSL_FMHA_PREFETCHED {len(so)} exports in {leaf}")


if __name__ == "__main__":
    main()
