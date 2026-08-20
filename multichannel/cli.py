"""Minimal operator-only diagnostics entrypoint for the bootstrap milestone."""

from __future__ import annotations

import argparse
import os

from multichannel.config import diagnostic_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("diagnostics",))
    args = parser.parse_args()
    if args.command == "diagnostics":
        print(diagnostic_report(dict(os.environ)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
