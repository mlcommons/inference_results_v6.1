# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Fail unless protected base packages are unchanged and all nvidia-cutlass-dsl* distributions are at the public pin.

Usage: CUTLASS_DSL_PUBLIC_VERSION=<v> CUTLASS_DSL_PACKAGES="<names>" python3 check_pkgs.py <protect.txt>
"""
import importlib.metadata as m
import os
import re
import sys


def canon(name):
    return re.sub(r"[-_.]+", "-", name or "").lower()


def version(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return None


# Protected packages: exact versions recorded in <protect.txt>.
moved = []
with open(sys.argv[1]) as f:
    for line in f:
        name, want = line.strip().split("==", 1)
        if version(name) != want:
            moved.append((name, want, version(name)))
print("protected packages unchanged" if not moved else f"PROTECTED PACKAGE MOVED: {moved}")

# CuTe DSL: the pinned packages are installed and every installed nvidia-cutlass-dsl* distribution is at the pin.
pin = os.environ["CUTLASS_DSL_PUBLIC_VERSION"]
installed = {}
for d in m.distributions():
    n = canon(d.metadata["Name"])
    if n.startswith("nvidia-cutlass-dsl"):
        installed[n] = d.version
missing = [n for n in map(canon, os.environ["CUTLASS_DSL_PACKAGES"].split()) if n not in installed]
off_pin = {n: v for n, v in installed.items() if v != pin}
cutlass_ok = not missing and not off_pin
print(f"cutlass-dsl {dict(sorted(installed.items()))}"
      + ("" if cutlass_ok else f" NOT THE PUBLIC PIN {pin}: missing {missing}, other versions {off_pin}"))

sys.exit(0 if not moved and cutlass_ok else 1)
