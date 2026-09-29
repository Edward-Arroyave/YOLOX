#!/usr/bin/env python3
"""Regenerate the Excel ficha from an edited Markdown and existing metrics.json."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolox.model_report_excel import write_excel_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="metrics.json de la ejecución")
    parser.add_argument("--markdown", required=True, help="model_report.md editado")
    parser.add_argument("--output", required=True, help="Excel que se creará o actualizará")
    args = parser.parse_args()
    report = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    write_excel_report(report, args.output, args.markdown)
    print(args.output)


if __name__ == "__main__":
    main()
