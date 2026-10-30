"""
Build dataset_splits.csv from one or more preprocessing result tables.

Examples
--------
# Leave AIBL out: AIBL -> 'external', other cohorts -> train/val/test by subject
python scripts/make_splits.py --scans out/preprocessing_results*.csv --holdout_cohort AIBL --out splits_loco_AIBL.csv

# Reuse the benchmark's assignments, joined on subject (and cohort, if present in both)
python scripts/make_splits.py --scans out/preprocessing_results*.csv \
    --splits_from benchmark_manifest.csv --splits_key sub_id --splits_col split --out splits_bench.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import splits as S


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scans', nargs='+', required=True, help='Preprocessing result CSV(s); globs allowed')
    p.add_argument('--out', required=True, help='Output dataset_splits.csv')
    p.add_argument('--holdout_cohort', nargs='+', default=None, help='Cohort(s) to hold out as "external"')
    p.add_argument('--splits_from', default=None, help='CSV whose split assignments should be reused')
    p.add_argument('--splits_key', default='subject_id', help='Subject column in --splits_from')
    p.add_argument('--splits_col', default='split', help='Split column in --splits_from')
    p.add_argument('--splits_cohort_col', default=None, help='Cohort column in --splits_from, if any')
    p.add_argument('--split_map', default=None,
                   help='Map split values, e.g. "0:train,1:val,2:test,3:external"')
    p.add_argument('--val_size', type=float, default=0.1)
    p.add_argument('--test_size', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--drop_conflicting', action='store_true',
                   help='Drop subjects with scans under both labels instead of failing')
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted({f for pattern in args.scans for f in glob.glob(pattern)})
    if not paths:
        sys.exit(f"No files match {args.scans}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if 'error' in df.columns:
        if 'final_file' not in df.columns:
            sys.exit('No scan in these result files was preprocessed successfully')
        failed = df['final_file'].isna()
        if failed.any():
            print(f"Excluding {failed.sum()} scans that failed preprocessing")
        df = df[~failed]
    if 'cohort' not in df.columns:
        df['cohort'] = 'all'
    df = df.drop(columns=['split'], errors='ignore')
    df = S.check_label_consistency(df, drop=args.drop_conflicting)

    if args.splits_from:
        t = pd.read_csv(args.splits_from).rename(columns={args.splits_key: 'subject_id'})
        if args.splits_cohort_col:
            t = t.rename(columns={args.splits_cohort_col: 'cohort'})
        mapping = dict(kv.split(':') for kv in args.split_map.split(',')) if args.split_map else None
        df = S.splits_from_table(df, t, key='subject_id', split_col=args.splits_col, split_map=mapping)
    elif args.holdout_cohort:
        df = S.loco_split(df, args.holdout_cohort, args.val_size, args.test_size, args.seed)
    else:
        df = S.random_split(df, args.test_size, args.val_size, args.seed)

    df = S.finalise(df)
    df.to_csv(args.out, index=False)
    with pd.option_context('display.width', 200):
        print(S.summarise(df))
    print(f"Wrote {len(df)} scans to {args.out}")


if __name__ == '__main__':
    main()
