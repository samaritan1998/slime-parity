"""CLI exit codes distinguish accepted, report-only, numerical, and evidence states."""
import argparse
import json
import sys
from . import __version__
from .common import EvidenceError, write_new
from .plan import prepare
from .runner import run_stage
from .compare import compare


def main(argv=None):
    parser = argparse.ArgumentParser(prog="training-parity")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare", help="Freeze a spec into a new run directory")
    create.add_argument("--spec", required=True)
    create.add_argument("--output", required=True)
    run = commands.add_parser("run", help="Execute one declared local stage")
    run.add_argument("--plan", required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--timeout", type=float, default=3600)
    check = commands.add_parser("compare", help="Verify evidence and evaluate the frozen rules")
    check.add_argument("--plan", required=True)
    check.add_argument("--output", required=True, help="New report file; never overwrite")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = {"status": "prepared", "plan": str(prepare(args.spec, args.output))}
        elif args.command == "run":
            result = run_stage(args.plan, args.stage, timeout=args.timeout)
        else:
            report = compare(args.plan)
            write_new(args.output, report)
            result = {"status": report["status"], "ungated_tensors": report["ungated_tensors"], "report": args.output}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return {"numerical_failure": 2, "report_only": 4}.get(result["status"], 0)
    except (EvidenceError, OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"status": "incomplete_evidence", "error": str(error)}), file=sys.stderr)
        return 3
