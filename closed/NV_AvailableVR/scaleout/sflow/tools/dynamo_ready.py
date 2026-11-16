#!/usr/bin/env python3
"""Block until every Dynamo frontend serves the model with the expected worker population.

Usage: dynamo_ready.py --frontend-urls ip:port[,ip:port...] --model NAME --num-prefill P --num-decode D [--timeout S]

/v1/models goes green as soon as the first prefill+decode pair registers, so the gate also counts the distinct
`generate` instances per component in /health. Dynamo wires worker sets to HTTP routes asynchronously after
discovery, so the gate ends with one tiny completion per frontend (not from the benchmark dataset).
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.load(r)


def population(health):
    inst = [i for i in health.get("instances", []) if i.get("endpoint") == "generate"]
    pre = len({i.get("instance_id") for i in inst if "prefill" in str(i.get("component", ""))})
    dec = len({i.get("instance_id") for i in inst if "prefill" not in str(i.get("component", ""))})
    comps = sorted({(i.get("namespace"), i.get("component"), i.get("endpoint")) for i in health.get("instances", [])})
    return pre, dec, comps


def wait_one(base, model, want, deadline):
    last = None
    while time.monotonic() < deadline:
        try:
            models = [m.get("id") for m in get(base, "/v1/models").get("data", [])]
            pre, dec, comps = population(get(base, "/health"))
            state = (tuple(models), pre, dec)
            if state != last:
                print(f"[dynamo_ready] {base} models={models} prefill={pre} decode={dec} instances={comps}", flush=True)
                last = state
            if model in models and (pre, dec) == want:
                break
        except (OSError, ValueError, urllib.error.URLError) as e:
            if last != "err":
                print(f"[dynamo_ready] {base} not up yet: {e}", flush=True)
                last = "err"
        time.sleep(10)
    else:
        sys.exit(f"[dynamo_ready] {base}: timed out waiting for {want} (prefill, decode)")
    body = json.dumps({"model": model, "prompt": "The capital of France is", "max_tokens": 8,
                       "temperature": 0.0, "stream": False}).encode()
    while time.monotonic() < deadline:
        req = urllib.request.Request(base + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                out = json.load(r)
            print(f"[dynamo_ready] {base} live gate OK: {json.dumps(out)[:400]}", flush=True)
            return
        except urllib.error.HTTPError as e:
            print(f"[dynamo_ready] {base} gate HTTP {e.code}: {e.read()[:300]!r}", flush=True)
        except (OSError, ValueError) as e:
            print(f"[dynamo_ready] {base} gate error: {e}", flush=True)
        time.sleep(15)
    sys.exit(f"[dynamo_ready] {base}: serving path never went live")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend-urls", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prefill", type=int, required=True)
    ap.add_argument("--num-decode", type=int, required=True)
    ap.add_argument("--timeout", type=float, default=5400)
    a = ap.parse_args()
    deadline = time.monotonic() + a.timeout
    for u in [x.strip() for x in a.frontend_urls.split(",") if x.strip()]:
        wait_one(u if u.startswith("http") else f"http://{u}", a.model, (a.num_prefill, a.num_decode), deadline)
    print("DYNAMO_READY", flush=True)


if __name__ == "__main__":
    main()
