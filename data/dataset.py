import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class AlzheimerDataset(Dataset):
    
    def __init__(self, split_csv_path, split='train', transform=None, base_dir=None):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        split_csv_path : str
            Path to CSV file containing dataset split information
        split : str
            Which split to use ('train', 'val', or 'test')
        transform : callable, optional
            Optional transform to apply to the MRI data
        base_dir : str, optional
            Base directory to resolve relative paths in the CSV
        """
        self.df = pd.read_csv(split_csv_path)
        self.df = self.df[self.df['split'] == split]
        self.transform = transform
        self.base_dir = base_dir
        
        self._validate_files()
        
        # Calculate class weights
        self.calculate_class_distribution()
        
        print(f"Loaded {len(self.df)} scans for {split} split")
    
    # path issues catcher
    def _validate_files(self):
        if len(self.df) > 0:
            first_file = self._resolve_path(self.df.iloc[0]['norm_file'])
            if not os.path.exists(first_file):
                raise FileNotFoundError(
                    f"File not found: {first_file}\n"
                    f"Please check if the base_dir (currently {self.base_dir}) is set correctly"
                )
    
    # class distribution for balanced sampling
    def calculate_class_distribution(self):
    	self.labels = np.array([1 if row['group'] == 'AD' else 0 for _, row in self.df.iterrows()])
    	unique_labels, counts = np.unique(self.labels, return_counts=True)
    	self.class_counts = dict(zip(unique_labels, counts))
    	print(f"class distribution for {len(self.labels)} samples: {self.class_counts}")
    	
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
    
    # I had filepath problems so had to add --base_dir functionality
    def _resolve_path(self, file_path):
        if self.base_dir is not None:
            return os.path.join(self.base_dir, file_path)
        return file_path
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Path to normalized MRI file (which should be the end of the preprocessing pipeline
        # IF the pipeline changes this might have to be revisited
        mri_path = self._resolve_path(row['norm_file'])
        label = 1 if row['group'] == 'AD' else 0
        
        try:
           
            mri_img = nib.load(mri_path)
            mri_data = mri_img.get_fdata()
            # This line is necessary otherwise we get negative strides errors 
            # I have no clue what negative strides are. 
            mri_data = np.ascontiguousarray(mri_data)
            
            if self.transform:
                mri_data = self.transform(mri_data)
                mri_data = np.ascontiguousarray(mri_data)
            
            # Convert to tensor
            mri_tensor = torch.from_numpy(mri_data).float().unsqueeze(0)
            
            return mri_tensor, label
            
        except Exception as e:
            print(f"Error loading {mri_path}: {e}")
            # Return a dummy tensor of the correct shape (182, 218, 182) and the correct label
            # This is a fallback in case of any issues
            dummy_data = np.zeros((182, 218, 182), dtype=np.float32)
            dummy_tensor = torch.from_numpy(dummy_data).float().unsqueeze(0)
            return dummy_tensor, label



def create_data_loaders(split_csv_path, batch_size=8, num_workers=4, transforms=None, base_dir=None, use_balanced_sampler=True, pin_memory=True):
    """
    Create data loaders for train, validation and test sets.
    
    Parameters:
    -----------
    split_csv_path : str
        Path to CSV file containing dataset split information
    batch_size : int
        Batch size for data loaders
    num_workers : int
        Number of workers for data loading
    transforms : dict, optional
        Dictionary of transforms for each split
    base_dir : str, optional
        Base directory to resolve relative paths in the CSV
    use_balanced_sampler : bool
        Whether to use balanced sampling for training data
    pin_memory : bool
        Whether to pin memory for faster data transfer to GPU
        
    Returns:
    --------
    dict
        Dictionary of data loaders for each split
    """
    if transforms is None:
        transforms = {'train': None, 'val': None, 'test': None}
    
    datasets = {
        split: AlzheimerDataset(
            split_csv_path=split_csv_path,
            split=split,
            transform=transforms[split],
            base_dir=base_dir
        )
        for split in ['train', 'val', 'test']
    }
    
    train_sampler = None
    if use_balanced_sampler:
        train_sampler = datasets['train'].get_sampler()
        print("Using balanced sampling for training data")
        
        indices = [train_sampler.__iter__().__next__() for _ in range(32)]
        labels = [datasets['train'].labels[i] for i in indices]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"example balanced batch distribution: {dict(zip(unique, counts))}")
    
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
