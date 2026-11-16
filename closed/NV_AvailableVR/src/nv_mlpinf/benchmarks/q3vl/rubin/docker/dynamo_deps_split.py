#!/usr/bin/env python3
"""Split the Dynamo runtime dependency set (dynamo-deps.txt) against the installed vLLM base image.

usage: dynamo_deps_split.py DEPS.txt INSTALL_OUT.txt KEPT_OUT.txt
  INSTALL_OUT: `name==version` lines of DEPS.txt whose distribution is not installed at all
  KEPT_OUT:    `name pinned=<listed version> installed=<base version>` for names already installed;
               the base keeps its version (installing the listed pin would move a base package)
A distribution present only as an apt python3-* copy in /usr/lib/python3/dist-packages does not
count as installed: uv installs into /usr/local next to it without seeing it (e.g. oauthlib /
pyparsing), which pkgset.py --added accepts when the apt copy is unchanged.
"""
import importlib.metadata as metadata
import re
import sys
from pathlib import Path

APT_SITE = "/usr/lib/python3/dist-packages"


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


installed, apt_only = {}, {}
for dist in metadata.distributions():
    site = str(Path(dist.locate_file("")).resolve())
    target = apt_only if site == APT_SITE else installed
    target.setdefault(norm(dist.metadata["Name"]), set()).add(dist.version)

deps, kept, install = sys.argv[1], [], []
for line in open(deps):
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    name, _, version = line.partition("==")
    assert name and version, line
    n = norm(name)
    if n in installed:
        kept.append(f"{n} pinned={version} installed={'|'.join(sorted(installed[n]))}")
    else:
        install.append(f"{name}=={version}")
        if n in apt_only:
            print(f"  install {name}=={version} next to the apt copy {'|'.join(sorted(apt_only[n]))} in {APT_SITE}")
with open(sys.argv[2], "w") as f:
    f.write("".join(x + "\n" for x in install))
with open(sys.argv[3], "w") as f:
    f.write("".join(x + "\n" for x in kept))
print(f"dynamo deps: {len(install)} to install, {len(kept)} already provided by the base image")
for x in kept:
    print("  kept", x)
