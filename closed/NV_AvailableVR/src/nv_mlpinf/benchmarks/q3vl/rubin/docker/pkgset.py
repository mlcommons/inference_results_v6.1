"""Snapshot the installed Python distributions and prove what one image build step changed.

usage:
  pkgset.py snapshot OUT.json
  pkgset.py expect BEFORE.json [--added-file REQ.txt] [--added NAME==VER ...]
                                [--replaced NAME==VER ...] [--removed NAME ...] [--allow-new]

`expect` compares BEFORE.json with the current environment and fails unless the
difference is exactly the declared one:
  --added      the distribution was absent before and is now installed at VER; or (apt
               shadowing) it existed before ONLY as apt python3-* copies in /usr/lib/python3/
               dist-packages, those copies are unchanged, and VER is now installed next to them
               outside that directory (pip/uv into /usr/local, which comes first on sys.path),
               e.g. oauthlib / pyparsing from the Dynamo dependencies over the base's apt copies;
  --added-file one `NAME==VER` per line (comments and blank lines ignored);
  --replaced   the distribution was installed before (any version) and is now VER;
  --removed    the distribution was installed before and is now absent;
  --allow-new  a distribution that was absent before may appear without being declared (it is
               listed); every distribution that was installed before still must keep its version.
Every other distribution must keep its exact version. Names are compared after PEP 503
normalization. Read-only apart from OUT.json.
"""

import argparse
import importlib.metadata as metadata
import json
import re
import sys
from pathlib import Path


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


APT_SITE = "/usr/lib/python3/dist-packages"


def installed():
    # A name can be installed twice (e.g. an apt python3-* copy under /usr/lib/python3 and a
    # pip copy under /usr/local); keep every version so such a pair is compared as a whole.
    seen = {}
    for dist in metadata.distributions():
        seen.setdefault(normalize(dist.metadata["Name"]), set()).add(dist.version)
    return {name: "|".join(sorted(versions)) for name, versions in sorted(seen.items())}


def locations():
    """{name: {version: set of site dirs}} of the current environment."""
    where = {}
    for dist in metadata.distributions():
        site = str(Path(dist.locate_file("")).resolve())
        where.setdefault(normalize(dist.metadata["Name"]), {}).setdefault(dist.version, set()).add(site)
    return where


def pin(text):
    name, sep, version = text.strip().partition("==")
    if not sep or not name or not version:
        raise SystemExit(f"expected NAME==VERSION, got {text!r}")
    return normalize(name), version.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("out", type=Path)
    exp = sub.add_parser("expect")
    exp.add_argument("before", type=Path)
    exp.add_argument("--added-file", type=Path)
    exp.add_argument("--added", nargs="*", default=[])
    exp.add_argument("--replaced", nargs="*", default=[])
    exp.add_argument("--removed", nargs="*", default=[])
    exp.add_argument("--allow-new", action="store_true")
    args = parser.parse_args()

    now = installed()
    if args.mode == "snapshot":
        args.out.write_text(json.dumps(now, indent=2) + "\n")
        print(f"pkgset snapshot: {len(now)} distributions -> {args.out}")
        return

    before = json.loads(args.before.read_text())
    added = dict(pin(p) for p in args.added)
    if args.added_file:
        for line in args.added_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                name, version = pin(line)
                assert name not in added, f"duplicate pin {name}"
                added[name] = version
    replaced = dict(pin(p) for p in args.replaced)
    removed = {normalize(n) for n in args.removed}
    errors, alongside_apt = [], []
    where = locations()
    for name, version in added.items():
        if name not in before:
            if now.get(name) != version:
                errors.append(f"{name}: expected {version}, installed {now.get(name)}")
            continue
        # Present before: allowed only as a new copy next to unchanged apt python3-* copies.
        prev = set(before[name].split("|"))
        cur = set(now.get(name, "").split("|")) - {""}
        prev_sites = {s for v in prev for s in where.get(name, {}).get(v, set())}
        new_sites = where.get(name, {}).get(version, set())
        if version in prev:
            errors.append(f"{name}: expected new {version}, was already installed ({before[name]})")
        elif cur != prev | {version}:
            errors.append(f"{name}: expected {version} next to {before[name]}, installed {now.get(name)}")
        elif not prev_sites or any(s != APT_SITE for s in prev_sites):
            errors.append(f"{name}: was already installed ({before[name]}) outside {APT_SITE}: "
                          f"{sorted(prev_sites)}")
        elif not new_sites or APT_SITE in new_sites:
            errors.append(f"{name}: {version} not installed outside {APT_SITE}: {sorted(new_sites)}")
        else:
            alongside_apt.append(f"{name} {version} in {','.join(sorted(new_sites))} "
                                 f"(apt {before[name]} unchanged in {APT_SITE})")
    for name, version in replaced.items():
        if name not in before:
            errors.append(f"{name}: expected a replacement, was not installed before")
        if now.get(name) != version:
            errors.append(f"{name}: expected {version}, installed {now.get(name)}")
    for name in removed:
        if name not in before:
            errors.append(f"{name}: expected a removal, was not installed before")
        if name in now:
            errors.append(f"{name}: expected removed, still installed ({now[name]})")
    declared = set(added) | set(replaced) | removed
    extra_new = []
    for name in sorted((set(before) | set(now)) - declared):
        if before.get(name) != now.get(name):
            if args.allow_new and name not in before:
                extra_new.append(f"{name}=={now[name]}")
                continue
            errors.append(f"{name}: undeclared change {before.get(name)} -> {now.get(name)}")
    if errors:
        print("pkgset FAIL:\n  " + "\n  ".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(
        f"pkgset OK: {len(before)} -> {len(now)} distributions; added {len(added)}, "
        f"replaced {len(replaced)}, removed {len(removed)}; no undeclared change"
        + (f"; added next to an apt copy: {'; '.join(alongside_apt)}" if alongside_apt else "")
        + (f"; also new (allowed): {' '.join(extra_new)}" if extra_new else "")
    )


if __name__ == "__main__":
    main()
