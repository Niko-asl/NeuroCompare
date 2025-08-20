import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# Added for local HDF5 format training

class AlzheimerDatasetHDF5(Dataset):
    def __init__(self, split_csv_path, split='train', transform=None, base_dir=None, resize=None, resize_method='trilinear'):
        """
        Initialize the dataset.
        
        Parameters:
        split_csv_path : str
            N.B path files must lead to .h5 files to use this
        split : str
            Which split to use
        transform : callable, optional
        base_dir : str, optional
        """
        
        self.df = pd.read_csv(split_csv_path)
        self.df = self.df[self.df['split'] == split]
        self.transform = transform
        self.base_dir = base_dir
        self.resize = resize
        self.resize_method = resize_method
        
        self._validate_files()
        
        # Calculate class distribution
        self.calculate_class_distribution()
        
        print(f"Loaded {len(self.df)} HDF5 files for {split} split")
    
    def _validate_files(self):
        # File validator
        if len(self.df) > 0:
            first_file = self._resolve_path(self.df.iloc[0]['final_file'])
            if not os.path.exists(first_file):
                raise FileNotFoundError(
                    f"File not found: {first_file}\n"
                    f"Please check if the base_dir (currently {self.base_dir}) is set correctly"
                )
            
            if not first_file.endswith('.h5'):
                raise ValueError(
                    f"Expected HDF5 file (.h5), got: {first_file}\n"
                    f"Please run the NIfTI to HDF5 converter first"
                )
    
    def calculate_class_distribution(self):
        # class dist for balanced sampling
        self.labels = np.array([1 if row['group'] == 'AD' else 0 for _, row in self.df.iterrows()])
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.class_counts = dict(zip(unique_labels, counts))
        print(f"Class distribution for {len(self.labels)} samples: {self.class_counts}")
    	
    def get_sampler(self):
        class_weights = {
            class_idx: 1.0/count
            for class_idx, count in self.class_counts.items()
        }
        weights = [class_weights[label] for label in self.labels]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        return sampler
    
    def _resolve_path(self, file_path):
        if self.base_dir is not None:
            return os.path.join(self.base_dir, file_path)
        return file_path
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Path to HDF5 file
        hdf5_path = self._resolve_path(row['final_file'])
        label = 1 if row['group'] == 'AD' else 0
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                mri_data = f['volume'][:] 
            
            mri_data = np.ascontiguousarray(mri_data)
            
            if self.transform:
                mri_data = self.transform(mri_data)
                mri_data = np.ascontiguousarray(mri_data)
            
            mri_tensor = torch.from_numpy(mri_data).float().unsqueeze(0)
            
            if self.resize is not None:
                mri_tensor = self._apply_resize(mri_tensor)
                
            return mri_tensor, label
            
        except Exception as e:
            print(f"Error loading {hdf5_path}: {e}")
            dummy_data = np.zeros((182, 218, 182), dtype=np.float32)
            dummy_tensor = torch.from_numpy(dummy_data).float().unsqueeze(0)
            return dummy_tensor, label
      
    def _apply_resize(self, mri_tensor):
        """Apply resize if specified."""
        import torch.nn.functional as F
	    
	# mri_tensor shape: (1, D, H, W) - has channel dimension from unsqueeze(0)
        original_shape = mri_tensor.shape[-3:]  # (D, H, W)
        target_shape = tuple(self.resize)
	    
        if original_shape != target_shape:
	# Add batch dimension for F.interpolate: (1, 1, D, H, W)
            resized = F.interpolate(
		mri_tensor.unsqueeze(0),
		size=target_shape,
		mode=self.resize_method,
		align_corners=False if self.resize_method == 'trilinear' else None
            ).squeeze(0)  # Remove batch dim: (1, D, H, W)
            return resized
	    
        return mri_tensor  
      

def create_data_loaders_hdf5(split_csv_path, batch_size=8, num_workers=4, transforms=None, base_dir=None, use_balanced_sampler=True, pin_memory=True, resize=None, resize_method='trilinear'):
# Like the create data laoders of the original but for HDF5 files
    if transforms is None:
        transforms = {'train': None, 'val': None, 'test': None}
    
    datasets = {
        split: AlzheimerDatasetHDF5(
            split_csv_path=split_csv_path,
            split=split,
            transform=transforms[split],
            base_dir=base_dir,
            resize=resize,
            resize_method=resize_method
        )
        for split in ['train', 'val', 'test']
    }
    
    train_sampler = None
    if use_balanced_sampler:
        train_sampler = datasets['train'].get_sampler()
        print("Using balanced sampling for training data")
        
        # Test sampler
        indices = [train_sampler.__iter__().__next__() for _ in range(32)]
        labels = [datasets['train'].labels[i] for i in indices]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Example balanced batch distribution: {dict(zip(unique, counts))}")
    
    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    return dataloaders


def create_data_loaders(split_csv_path, batch_size=8, num_workers=4, transforms=None, base_dir=None, use_balanced_sampler=True, pin_memory=True, force_nifti=False, resize=None, resize_method='trilinear'):
# backwards compatible auto data loader
# loads whatever filetype it sees (hdf5 or nifti)
    try:
        df = pd.read_csv(split_csv_path)
        sample_file = df['final_file'].iloc[0]
        
        if sample_file.endswith('.h5') and not force_nifti:
            print("Detected HDF5 files - using fast HDF5 loader")
            return create_data_loaders_hdf5(split_csv_path, batch_size, num_workers, transforms, base_dir, use_balanced_sampler, pin_memory, resize, resize_method)
        else:
            print("Detected NIfTI files - using standard NIfTI loader")
            # Import the original function
            from data.dataset_original import create_data_loaders as create_data_loaders_nifti
            return create_data_loaders_nifti(split_csv_path, batch_size, num_workers, transforms, base_dir, use_balanced_sampler, pin_memory, resize, resize_method)
            
    except Exception as e:
        print(f"Error detecting file type: {e}")
        print("Falling back to NIfTI loader")
        from data.dataset_original import create_data_loaders as create_data_loaders_nifti
        return create_data_loaders_nifti(split_csv_path, batch_size, num_workers, transforms, base_dir, use_balanced_sampler, pin_memory, resize, resize_method)
        
        
def create_evaluation_dataloader(split_csv_path, batch_size=4, transform=None, 
                                base_dir=None, num_workers=4, pin_memory=True, 
                                resize=None, resize_method='trilinear'):
    """
    Create a single dataloader for evaluation (no train/val splits needed)
    """
    # Determine file type
    df = pd.read_csv(split_csv_path)
    sample_file = df['final_file'].iloc[0]
    
    if sample_file.endswith('.h5'):
        from data.dataset import AlzheimerDatasetHDF5 as Dataset
    else:
        from data.dataset_original import AlzheimerDataset as Dataset
    
    # Create dataset (assumes all data is marked as 'test' split)
    dataset = Dataset(
        split_csv_path=split_csv_path,
        split='test',
        transform=transform,
        base_dir=base_dir,
        resize=resize,
        resize_method=resize_method
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader
