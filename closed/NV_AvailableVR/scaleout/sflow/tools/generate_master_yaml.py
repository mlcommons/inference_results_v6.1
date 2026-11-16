#\!/usr/bin/env python3
"""
Generate a trtllm-serve disaggregated master/frontend server config YAML.

Accepts comma-separated URL lists for context and generation servers,
combines them with the frontend's hostname and port, and writes a valid
server_config.yaml.

Optionally appends "hostname:port" to a FRONTEND_URLS registry file so the
MLPerf harness can discover all live frontend endpoints.

Usage:
    python3 generate_master_yaml.py \
        --hostname 10.52.103.50 \
        --port 8000 \
        --ctx-urls 10.52.97.47:8337,10.52.97.47:8336 \
        --gen-urls 10.52.97.48:8338 \
        --output    /path/to/server_config.yaml \
        [--register-url-to /path/to/frontend_urls.txt]
"""
import argparse
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hostname", required=True, help="IP/hostname this frontend listens on")
    p.add_argument("--port", required=True, type=int, help="Port this frontend listens on")
    p.add_argument("--ctx-urls", required=True,
                   help="Comma-separated list of context server URLs, e.g. 10.0.0.1:8337,10.0.0.2:8337")
    p.add_argument("--gen-urls", required=True,
                   help="Comma-separated list of generation server URLs, e.g. 10.0.0.3:8338")
    p.add_argument("--output", required=True, type=Path, help="Destination server_config.yaml")
    p.add_argument("--register-url-to", type=Path, default=None,
                   help="Append 'hostname:port' to this file (e.g. frontend_urls.txt)")
    return p.parse_args()


def parse_urls(raw: str) -> list[str]:
    """Split a comma-separated URL string, stripping whitespace and empty entries."""
    urls = [u.strip() for u in raw.split(",") if u.strip()]
    if not urls:
        print("ERROR: no URLs provided", file=sys.stderr)
        sys.exit(1)
    return urls


def make_yaml_config(hostname: str, port: int, ctx_urls: list[str], gen_urls: list[str]) -> dict:
    """Build the server config dict for a trtllm-serve disaggregated frontend."""
    return {
        "hostname": hostname,
        "port": port,
        "backend": "pytorch",
        "context_servers": {
            "num_instances": len(ctx_urls),
            "urls": ctx_urls,
        },
        "generation_servers": {
            "num_instances": len(gen_urls),
            "urls": gen_urls,
        },
    }


def main() -> None:
    args = parse_args()

    ctx_urls = parse_urls(args.ctx_urls)
    gen_urls = parse_urls(args.gen_urls)

    config = make_yaml_config(args.hostname, args.port, ctx_urls, gen_urls)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Written: {args.output}")
    with open(args.output) as f:
        print(f.read())

    if args.register_url_to:
        url = f"{args.hostname}:{args.port}"
        with open(args.register_url_to, "a") as f:
            f.write(f"{url}\n")
        print(f"Registered URL: {url} → {args.register_url_to}")


if __name__ == "__main__":
    main()
