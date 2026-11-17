"""Metadata-only CLI. Run `py -3 -m shared_protocol --help`."""

import argparse
import json

from .manifest import ProtocolError, load_manifest, read_json, verification_issues
from .splits import create_split, load_split, save_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Audit draft metadata; no adapter access")
    validate.add_argument("manifest")
    validate.add_argument("--require-verified", action="store_true")
    split = sub.add_parser("split", help="Create a new immutable primary split artifact")
    split.add_argument("manifest")
    split.add_argument("policy")
    split.add_argument("output")
    check = sub.add_parser("check", help="Revalidate saved splits against metadata")
    check.add_argument("manifest")
    check.add_argument("split")
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest, require_verified=getattr(args, "require_verified", False))
        if args.command == "validate":
            issues = {row["adapter_id"]: verification_issues(row) for row in manifest["adapters"]}
            print(json.dumps({"verification_required": {k: v for k, v in issues.items() if v}}, indent=2))
        elif args.command == "split":
            plan = create_split(manifest, read_json(args.policy))
            save_split(args.output, plan, manifest)
            print(f"Created immutable split: {args.output}")
        else:
            load_split(args.split, manifest)
            print("Saved split verified; no adapter artifacts read.")
    except (ProtocolError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
