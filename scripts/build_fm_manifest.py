"""
Build preprocessing manifests restricted to the scans used in the foundation-model benchmark.

Walks scans/<cohort>/<diagnosis>/**.nii[.gz], keeps the scans whose name matches the
Image column of every model (files of the same model, e.g. a main and a QMIN eval-only file,
are combined first; see --model_regex), i.e. the FM-common set, and writes two manifests:

  <out>_first_scan.csv     FM-common scans that are first sessions (per --sessions)
  <out>_all_sessions.csv   every scan under scans/ belonging to the same subjects
                           (for the all-visits sensitivity analysis; evaluate on first scans only)
  <out>_report.csv         per cohort x label counts at each step (aggregate only)

Matching is on file names with directories and extensions removed, lower-cased, after any
--strip patterns. Columns in the sessions CSV are auto-detected (single columns, or two
columns joined with '_', e.g. subject + image ID) unless given.

Example:
  python scripts/build_fm_manifest.py --scans_root scans \
      --embeddings emb/common_jepa.csv emb/common_vfm.csv emb/common_nsc.csv emb/common_iac.csv \
      --sessions sessions.csv --out manifests/fm
then:
  python scripts/preprocess.py --manifest manifests/fm_first_scan.csv --path_col path \
      --subject_col sub_id --label_col label --cohort_col Dataset --scan_col scan_id ...
"""

import argparse
import glob
import itertools
import os
import re
import sys

import pandas as pd

EXTENSIONS = ('.gz', '.nii', '.npy', '.npz', '.h5', '.pt', '.mgz')
AD_DIRS = ('ad', 'alzheimers', 'dementia')
CN_DIRS = ('cn', 'control', 'controls', 'hc', 'normal')
DEFAULT_SUBJECT_PATTERNS = (r'\d{3}_S_\d{4}', r'NACC\d{6}', r'sub-[A-Za-z0-9]+')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scans_root', required=True, help='Folder with <cohort>/<diagnosis>/ subfolders of NIfTIs')
    p.add_argument('--embeddings', nargs='+', required=True, help='Common embedding CSV(s); globs allowed')
    p.add_argument('--image_col', default='Image', help='Scan identifier column in the embedding files')
    p.add_argument('--emb_label_col', default='label',
                   help='Label column in the embedding files (0=CN, 1=AD) for a cross-check; "" to skip')
    p.add_argument('--model_regex', default=r'emb_([^_]+)_',
                   help='Regex on file names whose group 1 is the model; files of the same model are '
                        'combined (union) before intersecting across models')
    p.add_argument('--sessions', default=None, help='CSV listing first sessions; omit to skip that filter')
    p.add_argument('--sessions_cols', nargs='+', default=None,
                   help='Sessions column(s) identifying a scan (joined with "_"); default: auto-detect')
    p.add_argument('--sessions_subject_col', default=None,
                   help='Sessions column with subject IDs (otherwise taken from file names)')
    p.add_argument('--strip', nargs='*', default=[],
                   help='Regexes removed from both scan names and Image values before matching, e.g. "_t1w$" "_brain"')
    p.add_argument('--subject_regex', nargs='*', default=[],
                   help='Per-cohort subject-ID regex, e.g. AIBL="^(\\d+)_" (group 1 used if present)')
    p.add_argument('--out', required=True, help='Output prefix, e.g. manifests/fm')
    return p.parse_args()


def make_key(name, strip):
    k = os.path.basename(str(name))
    while k.lower().endswith(EXTENSIONS):
        k = k[:k.rfind('.')]
    for pat in strip:
        k = re.sub(pat, '', k, flags=re.IGNORECASE)
    return k.lower()


def walk_scans(root, strip):
    rows, skipped = [], set()
    for cohort in sorted(os.listdir(root)):
        cdir = os.path.join(root, cohort)
        if not os.path.isdir(cdir):
            continue
        for dx in sorted(os.listdir(cdir)):
            ddir = os.path.join(cdir, dx)
            if not os.path.isdir(ddir):
                continue
            label = 'AD' if dx.lower() in AD_DIRS else 'CN' if dx.lower() in CN_DIRS else None
            if label is None:
                skipped.add(f"{cohort}/{dx}")
                continue
            for f in glob.glob(os.path.join(ddir, '**', '*.nii*'), recursive=True):
                stem = os.path.basename(f)
                stem = stem[:stem.lower().rfind('.nii')]
                rows.append(dict(path=os.path.abspath(f), Dataset=cohort, label=label,
                                 scan_id=stem, key=make_key(f, strip)))
    if skipped:
        print(f"Ignored folders not recognised as AD or CN: {sorted(skipped)}")
    if not rows:
        sys.exit(f"No NIfTIs found under {root}/<cohort>/<AD|CN>/")
    return pd.DataFrame(rows)


def model_of(path, pattern):
    m = re.search(pattern, os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def load_embedding_keys(paths, image_col, label_col, strip, scan_keys, model_pattern):
    """Union the files of each model (e.g. a main file plus a QMIN eval-only file),
    then intersect across models."""
    by_model, labels = {}, {}
    for path in paths:
        cols = pd.read_csv(path, nrows=0).columns
        if image_col not in cols:
            sys.exit(f"{path} has no '{image_col}' column (first columns: {list(cols[:8])})")
        ccol = next((c for c in ('Dataset', 'dataset', 'cohort') if c in cols), None)
        use = [image_col] + ([label_col] if label_col and label_col in cols else []) + ([ccol] if ccol else [])
        df = pd.read_csv(path, usecols=use)
        keys = df[image_col].map(lambda x: make_key(x, strip))
        model = model_of(path, model_pattern)
        by_model.setdefault(model, set()).update(keys)
        if label_col and label_col in cols:
            for k, l in zip(keys, df[label_col]):
                labels.setdefault(k, set()).add(str(l).split('.')[0])
        hit = keys.isin(scan_keys)
        print(f"[{model}] {os.path.basename(path)}: {len(df)} rows, {keys.nunique()} unique images, "
              f"{hit.sum()} match a scan")
        if (~hit).any():
            miss = df.loc[~hit.values]
            groups = miss.groupby(ccol) if ccol else [('all', miss)]
            for cohort, g in groups:
                ex = [str(v) for v in g[image_col].head(2)]
                print(f"    {len(g)} unmatched in {cohort}, e.g. {ex}")
    print(f"\nModels (files grouped by --model_regex {model_pattern!r}):")
    for model, keys in by_model.items():
        print(f"  {model}: {len(keys)} images, {len(keys & scan_keys)} match a scan")
    common = set.intersection(*by_model.values())
    print(f"Images present in all {len(by_model)} models: {len(common)} "
          f"({len(common & scan_keys)} match a scan)")
    if len(by_model) < 2:
        print("Warning: only one model found; check --model_regex")
    return common, labels


def detect_session_keys(sessions, scan_keys, cols, strip):
    s = sessions.astype(str)
    if cols:
        cands = [tuple(cols)]
    else:
        cands = [(c,) for c in s.columns] + list(itertools.permutations(s.columns, 2))
    best, best_n, best_keys = None, -1, None
    for c in cands:
        joined = s[c[0]] + '_' + s[c[1]] if len(c) > 1 else s[c[0]]
        keys = joined.map(lambda x: make_key(x, strip))
        n = len(set(keys) & scan_keys)
        if n > best_n:
            best, best_n, best_keys = c, n, keys
    if best_n <= 0:
        sys.exit(f"No column (or pair of columns) in the sessions CSV matches scan names. Columns: {list(s.columns)}")
    print(f"Sessions matched on {' + '.join(best)}: {best_n} of {sessions.shape[0]} rows match a scan")
    return best_keys


def subject_ids(df, overrides):
    per_cohort = {}
    for o in overrides:
        cohort, pat = o.split('=', 1)
        per_cohort[cohort.lower()] = pat
    out = []
    for name, cohort in zip(df['scan_id'], df['Dataset']):
        pats = [per_cohort[cohort.lower()]] if cohort.lower() in per_cohort else DEFAULT_SUBJECT_PATTERNS
        sid = None
        for pat in pats:
            m = re.search(pat, name, flags=re.IGNORECASE)
            if m:
                sid = m.group(1) if m.groups() else m.group(0)
                break
        out.append(sid)
    return pd.Series(out, index=df.index)


def main():
    args = parse_args()
    emb_paths = sorted({f for pattern in args.embeddings for f in glob.glob(pattern)})
    if not emb_paths:
        sys.exit(f"No embedding files match {args.embeddings}")

    scans = walk_scans(args.scans_root, args.strip)
    report = scans.groupby(['Dataset', 'label']).size().rename('scans_found').to_frame()

    # Keys repeated across cohorts or labels cannot be matched to embeddings unambiguously
    dup = scans.groupby('key')[['Dataset', 'label']].nunique().max(axis=1) > 1
    if dup.any():
        print(f"Warning: {dup.sum()} scan names occur in more than one cohort/label folder; excluded")
        scans = scans[~scans['key'].isin(dup[dup].index)]

    common, emb_labels = load_embedding_keys(emb_paths, args.image_col, args.emb_label_col, args.strip,
                                             set(scans['key']), args.model_regex)
    scans['in_fm'] = scans['key'].isin(common)
    report['in_all_fm'] = scans[scans['in_fm']].groupby(['Dataset', 'label']).size()
    if not scans['in_fm'].any():
        sys.exit("No scan matches an image present in every model; see the unmatched examples above "
                 "and adjust --strip (or --model_regex) so both sides reduce to the same name.")
    unmatched_fm = len(common - set(scans['key']))
    if unmatched_fm:
        print(f"Warning: {unmatched_fm} FM-common images have no file under {args.scans_root}")

    # Folder diagnosis vs the label the FM benchmark used
    if emb_labels:
        want = {'AD': '1', 'CN': '0'}
        bad = scans['in_fm'] & scans.apply(
            lambda r: r['key'] in emb_labels and emb_labels[r['key']] != {want[r['label']]}, axis=1)
        report['label_mismatch'] = scans[bad].groupby(['Dataset', 'label']).size()
        if bad.any():
            print(f"Warning: {bad.sum()} FM-common scans have a folder label that disagrees with the "
                  f"embedding label; excluded")
            scans.loc[bad, 'in_fm'] = False

    first = scans[scans['in_fm']].copy()
    if args.sessions:
        sessions = pd.read_csv(args.sessions, dtype=str)
        skeys = detect_session_keys(sessions, set(first['key']), args.sessions_cols, args.strip)
        first = first[first['key'].isin(set(skeys))]
        report['first_session'] = first.groupby(['Dataset', 'label']).size()
        if args.sessions_subject_col:
            sub_map = dict(zip(skeys, sessions[args.sessions_subject_col]))
            first['sub_id'] = first['key'].map(sub_map)

    if 'sub_id' not in first or first['sub_id'].isna().any():
        missing = first['sub_id'].isna() if 'sub_id' in first else pd.Series(True, index=first.index)
        first.loc[missing, 'sub_id'] = subject_ids(first[missing], args.subject_regex)
    no_sub = first['sub_id'].isna()
    if no_sub.any():
        ex = first.loc[no_sub].groupby('Dataset')['scan_id'].first().to_dict()
        sys.exit(f"No subject ID for {no_sub.sum()} scans; add --subject_regex. Example name per cohort: {ex}")

    multi = first.groupby(['Dataset', 'sub_id']).size()
    if (multi > 1).any():
        print(f"Warning: {(multi > 1).sum()} subjects have more than one scan in the first-scan manifest "
              "(check the sessions filter)")
    report['first_scan_rows'] = first.groupby(['Dataset', 'label']).size()
    report['first_scan_subjects'] = first.groupby(['Dataset', 'label'])['sub_id'].nunique()

    # Every scan of those subjects, wherever it sits under scans/
    scans['sub_id'] = subject_ids(scans, args.subject_regex)
    if args.sessions_subject_col and args.sessions:
        known = dict(zip(first['key'], first['sub_id']))
        scans['sub_id'] = scans['key'].map(known).fillna(scans['sub_id'])
    subj = set(zip(first['Dataset'], first['sub_id']))
    alls = scans[[(d, s) in subj for d, s in zip(scans['Dataset'], scans['sub_id'])]].copy()
    conflict = alls.groupby(['Dataset', 'sub_id'])['label'].nunique()
    conflict = set(conflict[conflict > 1].index)
    if conflict:
        print(f"Warning: {len(conflict)} subjects have scans in both AD and CN folders; dropped from all-sessions")
        alls = alls[[(d, s) not in conflict for d, s in zip(alls['Dataset'], alls['sub_id'])]]
    report['all_sessions_rows'] = alls.groupby(['Dataset', 'label']).size()

    cols = ['path', 'sub_id', 'scan_id', 'label', 'Dataset', 'key']
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    first[cols].rename(columns={'key': 'Image'}).to_csv(f'{args.out}_first_scan.csv', index=False)
    alls[cols].rename(columns={'key': 'Image'}).to_csv(f'{args.out}_all_sessions.csv', index=False)
    report = report.fillna(0).astype(int)
    report.to_csv(f'{args.out}_report.csv')
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(report)
    print(f"\nFirst-scan manifest: {len(first)} scans; all-sessions manifest: {len(alls)} scans "
          f"({len(alls) - len(first)} additional visits)")


if __name__ == '__main__':
    main()
