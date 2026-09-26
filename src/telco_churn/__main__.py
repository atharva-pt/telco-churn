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
    explore = commands.add_parser('explore', help='Reserve splits and explore training rows only.')
    explore.add_argument('--data', type=Path, required=True)
    explore.add_argument('--config', type=Path, required=True)
    explore.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        if args.command == 'validate':
            frame = read_dataset(args.data, labeled=not args.unlabeled)
            print(json.dumps({'rows': len(frame), 'missing': frame.isna().sum().to_dict()}, indent=2))
        elif args.command == 'explore':
            from .artifacts import write_json_new
            from .config import load_config
            from .plots import explore_training
            from .splits import create_manifest, select_partition
            config = load_config(args.config)
            frame = read_dataset(args.data, labeled=True)
            manifest = create_manifest(args.data, frame, config)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json_new(args.output/'split.json', manifest)
            explore_training(select_partition(frame, manifest['ids'], 'train'), args.output)
            logging.info('Training exploration saved to %s', args.output)
        return 0
    except (ValueError, OSError) as exc:
        logging.error('%s', exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
