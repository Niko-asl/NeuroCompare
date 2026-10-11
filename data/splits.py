"""
Subject-level split construction with leakage checks.

All functions work on a scan table with columns:
  final_file, subject_id, group ('AD' / 'Control'), cohort
Subjects are keyed by (cohort, subject_id), so IDs that repeat across cohorts
never merge, and every scan of a subject lands in the same split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED = ('final_file', 'subject_id', 'group', 'cohort')


def _key(df):
    return df['cohort'].astype(str) + '|' + df['subject_id'].astype(str)


def check_label_consistency(df, drop=False):
    """Subjects with scans under more than one label leak across classes."""
    n_labels = df.assign(_k=_key(df)).groupby('_k')['group'].nunique()
    bad = n_labels[n_labels > 1].index
    if len(bad) == 0:
        return df
    if not drop:
        raise ValueError(
            f"{len(bad)} subjects have scans under both labels, e.g. {list(bad[:5])}. "
            "Resolve them in the manifest, or pass drop=True to exclude them."
        )
    keep = ~_key(df).isin(bad)
    print(f"Dropping {len(bad)} subjects ({(~keep).sum()} scans) with conflicting labels")
    return df[keep].copy()


def assert_no_leakage(df):
    n_splits = df.assign(_k=_key(df)).groupby('_k')['split'].nunique()
    leaked = n_splits[n_splits > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} subjects appear in more than one split, e.g. {list(leaked.index[:5])}")


def _split_subjects(subjects, size, seed):
    """Split a one-row-per-subject table, stratified by cohort x label where possible."""
    if size <= 0 or len(subjects) == 0:
        return subjects, subjects.iloc[0:0]
    for strat in (subjects['cohort'].astype(str) + '|' + subjects['group'], subjects['group'], None):
        try:
            return train_test_split(subjects, test_size=size, random_state=seed, stratify=strat)
        except ValueError:
            continue


def _subject_table(df):
    return df.assign(_k=_key(df)).drop_duplicates('_k')[['_k', 'cohort', 'group']]


def random_split(df, test_size=0.1, val_size=0.1, seed=42):
    """Train / val / test across all cohorts, grouped by subject."""
    subjects = _subject_table(df)
    rest, test = _split_subjects(subjects, test_size, seed)
    train, val = _split_subjects(rest, val_size / (1 - test_size), seed)
    return _assign(df, {'train': train, 'val': val, 'test': test})


def loco_split(df, holdout, val_size=0.1, test_size=0.1, seed=42):
    """Leave-one-cohort-out. The held-out cohort becomes 'external'; the remaining
    cohorts are split into train / val / test (test = in-distribution held-out subjects)."""
    holdout = [holdout] if isinstance(holdout, str) else list(holdout)
    missing = set(holdout) - set(df['cohort'].astype(str))
    if missing:
        raise ValueError(f"Hold-out cohort(s) {missing} not in data; cohorts are {sorted(df['cohort'].unique())}")
    is_ext = df['cohort'].astype(str).isin(holdout)
    subjects = _subject_table(df[~is_ext])
    rest, test = _split_subjects(subjects, test_size, seed)
    train, val = _split_subjects(rest, val_size / (1 - test_size) if test_size < 1 else val_size, seed)
    out = _assign(df[~is_ext], {'train': train, 'val': val, 'test': test})
    ext = df[is_ext].copy()
    ext['split'] = 'external'
    return pd.concat([out, ext], ignore_index=True)


def splits_from_table(df, splits, key='subject_id', split_col='split', split_map=None):
    """Reuse split assignments from another table (e.g. the benchmark's folds).
    `key` must exist in both tables; when 'cohort' exists in both, it is joined on too."""
    s = splits.copy()
    if split_map:
        s[split_col] = s[split_col].astype(str).map(split_map)
    on = [key] + (['cohort'] if 'cohort' in s.columns and key != 'cohort' else [])
    s = s[on + [split_col]].drop_duplicates()
    if s.duplicated(on).any():
        raise ValueError(f"Split table assigns some {on} to more than one split")
    merged = df.drop(columns=['split'], errors='ignore').merge(
        s.rename(columns={split_col: 'split'}), on=on, how='left')
    unmatched = merged['split'].isna().sum()
    if unmatched:
        print(f"Warning: {unmatched} scans have no split assignment and are dropped")
        merged = merged.dropna(subset=['split'])
    return merged


def _assign(df, parts):
    lookup = {k: name for name, part in parts.items() for k in part['_k']}
    out = df.copy()
    out['split'] = _key(out).map(lookup)
    return out


def summarise(df):
    return df.groupby(['split', 'cohort', 'group']).agg(
        scans=('final_file', 'size'), subjects=('subject_id', 'nunique')).unstack('group', fill_value=0)


def finalise(df):
    for c in REQUIRED:
        if c not in df.columns:
            raise ValueError(f"Scan table is missing column '{c}'")
    assert_no_leakage(df)
    return df
