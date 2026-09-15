#!/usr/bin/env python3
"""Render an existing metrics.json without training or evaluating again."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolox.model_report_html import render_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    with Path(args.output).open("x", encoding="utf-8") as stream:
        stream.write(render_html(report))
    print(args.output)


if __name__ == "__main__":
    main()
