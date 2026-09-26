"""Thin command-line layer: business logic lives in importable modules."""
import argparse
import json
import logging
from pathlib import Path

from .data import read_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Train, evaluate, and use a Telco churn model.')
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Check a labeled or feature-only CSV.')
    validate.add_argument('--data', type=Path, required=True)
    validate.add_argument('--unlabeled', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        frame = read_dataset(args.data, labeled=not args.unlabeled)
        print(json.dumps({'rows': len(frame), 'missing': frame.isna().sum().to_dict()}, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        logging.error('%s', exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
