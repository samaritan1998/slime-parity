import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from training_parity.common import EvidenceError, write_new, read
from . import __version__
from .suite import prepare, execute, report
from .reporting import save_report

EXIT_CODES = {"passed": 0, "numerical_failure": 2, "unverified": 3, "report_only": 4}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="slime-parity", description="Slime model and parallel correctness suites")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Write an executable CPU demonstration spec")
    init.add_argument("--output", required=True)
    for name in ("plan", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--spec", required=True)
        p.add_argument("--output", required=True)
    run = sub.add_parser("run", help="Run pending stages, preserving all previous attempts")
    run.add_argument("--suite", required=True)
    summary = sub.add_parser("report")
    summary.add_argument("--suite", required=True)
    summary.add_argument("--output")
    sub.add_parser("doctor")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            from .adapters.mlp import example_spec
            write_new(args.output, example_spec())
            print(str(Path(args.output).resolve()))
            return 0
        if args.command == "doctor":
            versions = {}
            for name in ("slime-parity", "torch", "numpy", "transformers", "megatron-core", "transformer-engine"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            data = {"versions": versions, "python": sys.version.split()[0], "cuda_devices": 0}
            try:
                import torch
                data["cuda_devices"] = torch.cuda.device_count()
            except ImportError:
                pass
            print(json.dumps(data, indent=2))
            return 0
        if args.command in ("plan", "validate"):
            path = prepare(args.spec, args.output)
            if args.command == "plan":
                suite = read(path)
                print(json.dumps({"suite": str(path), "requested": sum(len(c["matrix"]) for c in suite["cases"]),
                                  "eligible": sum(e["eligibility"] == "eligible" for c in suite["cases"] for e in c["matrix"])}, indent=2))
                return 0
            result = execute(path)
        elif args.command == "run":
            result = execute(args.suite)
        else:
            result = report(args.suite)
            if args.output:
                save_report(result, args.output)
                result["report_directory"] = str(Path(args.output).resolve())
        print(json.dumps({k: result[k] for k in ("status", "counts", "report_directory") if k in result}, indent=2))
        return EXIT_CODES[result["status"]]
    except (EvidenceError, OSError, ValueError, ImportError, AttributeError, KeyError) as error:
        print(json.dumps({"status": "unverified", "error": str(error)}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
