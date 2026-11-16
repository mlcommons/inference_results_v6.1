#!/usr/bin/env python3
"""
Inject a live list of frontend endpoint URLs into an endpoint.yaml config.

Reads an existing endpoint.yaml template, replaces the ``endpoint_config.endpoints``
placeholder (``endpoints: []``) with the provided URL list using regex substitution,
and writes the result. Other content is preserved verbatim so the endpoints repo's
own env-var interpolation still runs at ``benchmark from-config`` load time.

Uses only Python stdlib (re, argparse, pathlib) - no PyYAML required.

Usage:
    python3 generate_endpoint_yaml.py \\
        --input  /work/.../endpoint.yaml \\
        --frontend-urls 10.0.0.1:8000,10.0.0.2:8000 \\
        --output /tmp/endpoint_resolved.yaml \\
        [--scheme http]
"""
import argparse
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Source endpoint.yaml template (must contain 'endpoints: []')")
    p.add_argument("--frontend-urls", required=True,
                   help="Comma-separated frontend addresses, e.g. 10.0.0.1:8000,10.0.0.2:8000. "
                        "Each entry may include a scheme; otherwise --scheme is prepended.")
    p.add_argument("--output", required=True, type=Path,
                   help="Destination endpoint.yaml")
    p.add_argument("--scheme", default="http", choices=["http", "https"],
                   help="Scheme to prepend to bare ip:port entries (default: http)")
    return p.parse_args()


def parse_urls(raw: str, scheme: str) -> list:
    urls = []
    for entry in raw.split(","):
        e = entry.strip()
        if not e:
            continue
        if not (e.startswith("http://") or e.startswith("https://")):
            e = f"{scheme}://{e}"
        urls.append(e)
    if not urls:
        print("ERROR: no URLs provided in --frontend-urls", file=sys.stderr)
        sys.exit(1)
    return urls


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    text = args.input.read_text()

    # Match only non-comment lines: capture leading indent so we can restore it.
    # Negative lookahead ensures the line doesn't start with '#' (possibly preceded by spaces).
    _PLACEHOLDER = re.compile(r"^(?![ \t]*#)([ \t]*)endpoints:\s*\[\]", re.MULTILINE)
    if not _PLACEHOLDER.search(text):
        print("ERROR: input YAML does not contain 'endpoints: []' placeholder", file=sys.stderr)
        sys.exit(1)

    urls = parse_urls(args.frontend_urls, args.scheme)
    url_block = "\n".join(f"  - {u}" for u in urls)
    text = _PLACEHOLDER.sub(lambda m: f"{m.group(1)}endpoints:\n{url_block}", text, count=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)

    print(f"Written: {args.output}")
    print(f"endpoints ({len(urls)}):")
    for u in urls:
        print(f"  - {u}")


if __name__ == "__main__":
    main()
