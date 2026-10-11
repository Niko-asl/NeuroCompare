"""
Datasets and data loaders.

Each item is (volume, label, domain, index):
  volume : float tensor (1, D, H, W)
  label  : 1 = AD, 0 = Control
  domain : integer cohort id from the training split's cohorts (-1 if the
           cohort was not seen in training, or if the CSV has no cohort column)
  index  : row position in dataset.df, so predictions and features can be
           joined back to subject / scan / cohort without relying on order

Use `unpack_batch` in loops so code works with this 4-tuple.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

DEFAULT_SHAPE = (182, 218, 182)
COHORT_COLUMNS = ('cohort', 'dataset', 'Dataset')


def cohort_column(df):
    for col in COHORT_COLUMNS:
        if col in df.columns:
            return col
    return None


def unpack_batch(batch):
    """Return (inputs, labels, domains, indices); tolerates old 2-tuples."""
    if len(batch) == 4:
        return batch
    inputs, labels = batch
    n = labels.shape[0]
    return inputs, labels, torch.full((n,), -1, dtype=torch.long), torch.full((n,), -1, dtype=torch.long)


class _ScanDataset(Dataset):
    def __init__(self, split_csv_path, split='train', transform=None, base_dir=None,
                 resize=None, resize_method='trilinear', domain_map=None):
        df = pd.read_csv(split_csv_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.split = split
        self.transform = transform
        self.base_dir = base_dir
        self.resize = resize
        self.resize_method = resize_method

        self.labels = np.array([1 if g == 'AD' else 0 for g in self.df['group']], dtype=np.int64)
        col = cohort_column(self.df)
        if col is not None and domain_map:
            self.domains = np.array([domain_map.get(c, -1) for c in self.df[col]], dtype=np.int64)
        else:
            self.domains = np.full(len(self.df), -1, dtype=np.int64)

        unique, counts = np.unique(self.labels, return_counts=True)
        self.class_counts = dict(zip(unique.tolist(), counts.tolist()))
        self._validate_files()
        print(f"Loaded {len(self.df)} scans for '{split}' split, class counts {self.class_counts}")

    def _resolve_path(self, file_path):
        return os.path.join(self.base_dir, file_path) if self.base_dir is not None else file_path

    def _validate_files(self):
        missing = [p for p in self.df['final_file'] if not os.path.exists(self._resolve_path(p))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(self.df)} files in split '{self.split}' not found, "
                f"e.g. {self._resolve_path(missing[0])} (base_dir={self.base_dir})"
            )

    def _load_volume(self, path):
        raise NotImplementedError

    def get_sampler(self, generator=None):
        weights = [1.0 / self.class_counts[l] for l in self.labels]
        return WeightedRandomSampler(weights=weights, num_samples=len(weights),
                                     replacement=True, generator=generator)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self._resolve_path(self.df.iloc[idx]['final_file'])
        # No silent fallback: an unreadable volume is a data error, not a zero image.
        try:
            data = np.ascontiguousarray(self._load_volume(path))
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}") from e
        if self.transform:
            data = np.ascontiguousarray(self.transform(data))
        tensor = torch.from_numpy(data).float().unsqueeze(0)
        if self.resize is not None and tuple(tensor.shape[-3:]) != tuple(self.resize):
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=tuple(self.resize), mode=self.resize_method,
                align_corners=False if self.resize_method == 'trilinear' else None,
            ).squeeze(0)
        return tensor, int(self.labels[idx]), int(self.domains[idx]), idx


class AlzheimerDataset(_ScanDataset):
    """NIfTI volumes."""
    def _load_volume(self, path):
        import nibabel as nib
        return nib.load(path).get_fdata(dtype=np.float32)


class AlzheimerDatasetHDF5(_ScanDataset):
    """HDF5 volumes stored under the 'volume' key."""
    def _load_volume(self, path):
        import h5py
        with h5py.File(path, 'r') as f:
            return f['volume'][:]


def _dataset_class(split_csv_path):
    first = pd.read_csv(split_csv_path)['final_file'].iloc[0]
    return AlzheimerDatasetHDF5 if str(first).endswith('.h5') else AlzheimerDataset


def build_domain_map(split_csv_path):
    """Cohort -> integer id, from cohorts present in the training split only."""
    df = pd.read_csv(split_csv_path)
    col = cohort_column(df)
    if col is None:
        return {}
    return {c: i for i, c in enumerate(sorted(df.loc[df['split'] == 'train', col].astype(str).unique()))}


def eval_splits(split_csv_path):
    """Every split other than train/val (e.g. 'test', 'external'), in a stable order."""
    splits = pd.read_csv(split_csv_path)['split'].astype(str).unique().tolist()
    return sorted(s for s in splits if s not in ('train', 'val'))


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def _worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def create_data_loaders(split_csv_path, batch_size=8, num_workers=4, transforms=None, base_dir=None,
                        use_balanced_sampler=True, pin_memory=True, resize=None, resize_method='trilinear',
                        seed=None, domain_map=None, eval_batch_size=None):
    """Loaders for 'train', 'val' and every other split value in the CSV (see eval_splits)."""
    transforms = transforms or {}
    Dataset_ = _dataset_class(split_csv_path)
    if domain_map is None:
        domain_map = build_domain_map(split_csv_path)

    names = ['train', 'val'] + eval_splits(split_csv_path)
    datasets = {
        s: Dataset_(split_csv_path, split=s, transform=transforms.get(s) if s == 'train' else None,
                    base_dir=base_dir, resize=resize, resize_method=resize_method, domain_map=domain_map)
        for s in names
    }
    for s in ('train', 'val'):
        if len(datasets[s]) == 0:
            raise ValueError(f"Split '{s}' is empty in {split_csv_path}")

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    sampler = datasets['train'].get_sampler(generator) if use_balanced_sampler else None

    common = dict(num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=_worker_init)
    loaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=sampler is None,
                            sampler=sampler, generator=generator, **common),
    }
    for s in names[1:]:
        loaders[s] = DataLoader(datasets[s], batch_size=eval_batch_size or batch_size, shuffle=False, **common)
    return loaders


def create_evaluation_dataloader(split_csv_path, batch_size=4, transform=None, base_dir=None,
                                 num_workers=4, pin_memory=True, resize=None, resize_method='trilinear',
                                 split='test', domain_map=None):
    Dataset_ = _dataset_class(split_csv_path)
    dataset = Dataset_(split_csv_path, split=split, transform=transform, base_dir=base_dir,
                       resize=resize, resize_method=resize_method, domain_map=domain_map)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory)
