"""
How much cohort information is linearly decodable from a run's penultimate features?

Fits a logistic-regression cohort probe with subject-grouped stratified CV on the features
saved by train.py --export_features, overall and within each diagnostic class. Balanced
accuracy at chance (1 / n_cohorts) means no linearly decodable cohort signal. Compare a
--dann run against the same configuration without it.

Example:
  python scripts/cohort_decodability.py --run_dir output/cnn_loco_AIBL_dann --model cnn --splits train val test
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load(run_dir, model, splits):
    tables, feats = [], []
    for s in splits:
        csv = os.path.join(run_dir, f'{model}_{s}_predictions.csv')
        npy = os.path.join(run_dir, f'{model}_{s}_features.npy')
        if not (os.path.exists(csv) and os.path.exists(npy)):
            print(f"Skipping '{s}': no exported features (run train.py with --export_features)")
            continue
        t, f = pd.read_csv(csv), np.load(npy)
        if len(t) != len(f):
            raise ValueError(f"{s}: {len(t)} prediction rows but {len(f)} feature rows")
        tables.append(t)
        feats.append(f)
    if not tables:
        raise SystemExit("No features found")
    return pd.concat(tables, ignore_index=True), np.concatenate(feats)


def probe(X, cohort, groups, folds, seed):
    k = min(folds, pd.Series(groups).groupby(cohort).nunique().min())
    if k < 2 or len(np.unique(cohort)) < 2:
        return None
    pred = np.empty(len(cohort), dtype=object)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, cohort, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced'))
        clf.fit(X[tr], cohort[tr])
        pred[te] = clf.predict(X[te])
    return {'balanced_acc': float(balanced_accuracy_score(cohort, pred)),
            'chance': 1.0 / len(np.unique(cohort)), 'n': int(len(cohort)), 'folds': int(k)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                   help="Splits to pool. Leave out 'external' to probe only the training cohorts.")
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    table, X = load(args.run_dir, args.model, args.splits)
    ccol = next(c for c in ('cohort', 'dataset', 'Dataset') if c in table.columns)
    cohort = table[ccol].astype(str).values
    groups = (table[ccol].astype(str) + '|' + table['subject_id'].astype(str)).values

    results = {'overall': probe(X, cohort, groups, args.folds, args.seed)}
    for label, name in ((0, 'within_control'), (1, 'within_AD')):
        m = table['label'].values == label
        results[name] = probe(X[m], cohort[m], groups[m], args.folds, args.seed)
    results['cohorts'] = pd.Series(cohort).value_counts().to_dict()

    for k, v in results.items():
        if isinstance(v, dict) and 'balanced_acc' in v:
            print(f"{k:>15}: cohort balanced acc {v['balanced_acc']:.3f} (chance {v['chance']:.3f}, n={v['n']})")
    out = os.path.join(args.run_dir, f'{args.model}_cohort_decodability.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
