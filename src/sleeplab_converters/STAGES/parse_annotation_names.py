"""Go through all csv files and parse the unique annotation names."""
import argparse
import csv
import json
import logging
import sleeplab_format as slf

from pathlib import Path

logger = logging.getLogger(__name__)


def parse_csvs(base_dir: Path):
    unique_names = set()
    name_counts = {}
    for csvpath in base_dir.glob('**/*.csv'):
        logger.info(f'Parsing names from {csvpath}...')
        with open(csvpath, 'r') as f:
            reader = csv.reader(f, delimiter=',', skipinitialspace=True)
            _ = next(reader, None)
            try:
                for row in reader:
                    if len(row) > 3:
                        row = row[:2] + [', '.join(row[2:])]
                    _, _, name_str = row
                    name_str = name_str.strip()
                    unique_names.add(name_str)
                    name_counts[name_str] = name_counts.get(name_str, 0) + 1
            except Exception as e:
                logger.info(f'Skipping Exception: {e}')

    return unique_names, name_counts


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', required=True)
    parser.add_argument('--save-dir', required=True)

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    logger.info(f'Start parsing CSVs from {args.base_dir}')
    unique_names, name_counts = parse_csvs(Path(args.base_dir))
    logger.info(f'Writing results to {args.save_dir}...')

    with open(Path(args.save_dir) / 'unique_annotation_names.json', 'w') as f:
        json.dump(sorted(list(unique_names)), f, indent=4)
    
    with open(Path(args.save_dir) / 'annotation_name_counts.json', 'w') as f:
        # Sort by count in descending order
        json.dump(dict(sorted(name_counts.items(), key=lambda item: -item[1])), f, indent=4)
