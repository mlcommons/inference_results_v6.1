#!/usr/bin/env python3
"""
Print shell export statements from a flat YAML env file.

Usage:
    eval "$(python3 run_with_env.py <env.yaml> | tee /dev/stderr)"

The tee /dev/stderr makes each export line visible in the log,
while eval applies them to the current shell.
"""
import argparse
import sys

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("env_yaml", help="Path to a flat YAML file of KEY: value env vars")
    return p.parse_args()


def print_exports(env_vars: dict) -> None:
    """Print 'export KEY=value' lines for each entry in env_vars."""
    for k, v in env_vars.items():
        print(f"export {k}='{v}'")


def main() -> None:
    args = parse_args()
    with open(args.env_yaml) as f:
        env_vars = yaml.safe_load(f) or {}
    print_exports(env_vars)


if __name__ == "__main__":
    main()
