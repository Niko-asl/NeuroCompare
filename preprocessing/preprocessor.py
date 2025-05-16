import os
import logging
from tqdm import tqdm

# dicom to nifti converter
from .conversion import DicomConverter

# Skull stripper using HD-BET from https://github.com/MIC-DKFZ/HD-BET
from .brain_extraction import BrainExtractor

# Spatial and intensity normalization 
from .normalization import SpatialNormalizer, IntensityNormalizer
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


#Heart and soul of the pipeline
class AlzheimerPreprocessor:
    
    def __init__(self, ad_dir, control_dir, output_dir):
        self.ad_dir = ad_dir
        self.control_dir = control_dir
        self.output_dir = output_dir
        
        os.makedirs(os.path.join(output_dir, "nifti"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "skull_stripped"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "normalized_space"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "normalized"), exist_ok=True)
        
        self.converter = DicomConverter(output_dir)
        self.brain_extractor = BrainExtractor(output_dir)
        self.spatial_normalizer = SpatialNormalizer(output_dir)
        self.intensity_normalizer = IntensityNormalizer(output_dir)
        
        # Initialize metadata
        self.metadata = []
        
        logger.info(f"Initialized preprocessor with AD dir: {ad_dir}, Control dir: {control_dir}")
    
    
    # This function scans dirs and collects scans for preprocessing
    def scan_directories(self):
        logger.info("Scanning directories for DICOM series...")
    
        # label inclided as a param here because directories are split AD vs CN
        # Might need retweaking if this is not the case in future
        def process_directory(base_dir, label):
            subjects = []
            for subject_id in os.listdir(base_dir):
                subject_path = os.path.join(base_dir, subject_id)
                if not os.path.isdir(subject_path):
                    continue
            
                # Omitted MP-RAGE repeats, correct if wrong
                for scan_type in ["MP-RAGE"]:
                    scan_type_path = os.path.join(subject_path, scan_type)
                    if not os.path.isdir(scan_type_path):
                        continue
                
                    # Scans are organised by date in ADNI
                    for scan_date in os.listdir(scan_type_path):
                        scan_date_path = os.path.join(scan_type_path, scan_date)
                        if not os.path.isdir(scan_date_path):
                            continue
                    
                       
                        for series_folder in os.listdir(scan_date_path):
                            series_path = os.path.join(scan_date_path, series_folder)
                            if not os.path.isdir(series_path):
                                continue
                            
         
                            dicom_files = glob.glob(os.path.join(series_path, "*.dcm"))
                            if dicom_files:
                                subjects.append({
                                    "subject_id": subject_id,
                                    "scan_type": scan_type,
                                    "scan_date": scan_date,
                                    "series_folder": series_folder,
                                    "dicom_path": series_path,
                                    "group": label
                                })
        
            return subjects
    
        # Process dirs separately then fuse into a single dataframe
        ad_subjects = process_directory(self.ad_dir, "AD")
        control_subjects = process_directory(self.control_dir, "Control")
        all_subjects = ad_subjects + control_subjects
        self.metadata = pd.DataFrame(all_subjects)
    
        logger.info(f"Found {len(ad_subjects)} AD scans and {len(control_subjects)} Control scans")
    
        # Tabulate all samples (N.B, this is saved to the output folder)
        self.metadata.to_csv(os.path.join(self.output_dir, "metadata.csv"), index=False)
        return self.metadata
        
        
        
    # Pipline is fundamentally 4 steps, see below:    
    def process_single_subject(self, row):
        """Process a single subject through the entire pipeline."""
        subject_id = row['subject_id']
        logger.info(f"Processing subject {subject_id}")
        
        # Step 1: Convert DICOM to NIfTI
        nifti_file = self.converter.convert(row)
        if nifti_file is None:
            return {'subject_id': subject_id, 'group': row['group'], 'error': 'DICOM conversion 	failed'}
        
        # Step 2: Apply skull stripping
        brain_file = self.brain_extractor.extract(nifti_file)
        if brain_file is None:
            return {'subject_id': subject_id, 'group': row['group'], 
                   'nifti_file': nifti_file, 'error': 'Skull stripping failed'}
        
        # Step 3: Apply spatial normalization
        mni_file = self.spatial_normalizer.normalize(brain_file)
        if mni_file is None:
            return {'subject_id': subject_id, 'group': row['group'], 
                   'nifti_file': nifti_file, 'brain_file': brain_file, 
                   'error': 'Spatial normalization failed'}
        
        # Step 4: Normalize intensity
        norm_file = self.intensity_normalizer.normalize(mni_file)
        if norm_file is None:
            return {'subject_id': subject_id, 'group': row['group'], 
                   'nifti_file': nifti_file, 'brain_file': brain_file, 
                   'mni_file': mni_file, 'error': 'Intensity normalization failed'}
        
        
        return {
            'subject_id': subject_id,
            'group': row['group'],
            'nifti_file': nifti_file,
            'brain_file': brain_file,
            'mni_file': mni_file,
            'norm_file': norm_file
        }
       
    # Does what it says
    def run_preprocessing_pipeline(self):
    
    	# Grabs the data produced by scan_directories
        if not hasattr(self, 'metadata') or len(self.metadata) == 0:
            self.scan_directories()
        
        results = []
        
        logger.info("Starting preprocessing pipeline...")
        
        # goes through the rows of the csv and runs the 4 step pipeline sequentially 
        for idx, row in tqdm(self.metadata.iterrows(), desc="Processing subjects", total=len(self.metadata)):
            result = self.process_single_subject(row)
            results.append(result)
        
        # here I found it pretty useful to make this CSV that includes any errors that occured in the pipeline 
        # helps for review on a file by file basis if anything messes up
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "preprocessing_results.csv"), index=False)
        
        logger.info("Preprocessing pipeline complete")
        return results_df
        
    # quick and diry 1-fold validation dataset maker
    # might need review when we think about multi fold validation but this should work for now
    def create_dataset_splits(self, test_size=0.1, val_size=0.1, random_state=42):
        
            
            results_path = os.path.join(self.output_dir, "preprocessing_results.csv")
            if not os.path.exists(results_path):
                logger.error("Cannot create splits: preprocessing_results.csv not found")
                return None
                
            results_df = pd.read_csv(results_path)
            
            # fastest way to remove scans that didnt make it all the way throught the pipeline
            valid_results = results_df.dropna(subset=['norm_file'])
            
            
            subjects = valid_results[['subject_id', 'group']].drop_duplicates()
            
            # Absolutely essential to split by subjects at this point 
            # (And all others when making datasets)
            # IMPERATIVE to avoid data leakage
            
            train_val_subjects, test_subjects = train_test_split(
                subjects, test_size=test_size, random_state=random_state, stratify=subjects['group']
            )
            
            # I had to do two splits here (train/val + test)
            # There has to be a way to do a 3 way split in less lines of code (??)
            adjusted_val_size = val_size / (1 - test_size)
            train_subjects, val_subjects = train_test_split(
                train_val_subjects, test_size=adjusted_val_size, 
                random_state=random_state, stratify=train_val_subjects['group']
            )
            
            # Add split information 
            train_subjects['split'] = 'train'
            val_subjects['split'] = 'val'
            test_subjects['split'] = 'test'
            
           
            all_splits = pd.concat([train_subjects, val_subjects, test_subjects])
            
           
            split_results = valid_results.merge(all_splits[['subject_id', 'split']], on='subject_id')
            
            # Keeping in theme, make CSV
            split_results.to_csv(os.path.join(self.output_dir, "dataset_splits.csv"), index=False)
            
            # This is a very surface level analysis of final train/test/val numbers
            # Mostly just for a sanity check that the balance of datasets is about right
            # And that the number of scans is what was expected
                    
            train_ad_subjects = len(train_subjects[train_subjects['group'] == 'AD'])
            train_control_subjects = len(train_subjects[train_subjects['group'] == 'Control'])
            val_ad_subjects = len(val_subjects[val_subjects['group'] == 'AD'])
            val_control_subjects = len(val_subjects[val_subjects['group'] == 'Control'])
            test_ad_subjects = len(test_subjects[test_subjects['group'] == 'AD'])
            test_control_subjects = len(test_subjects[test_subjects['group'] == 'Control'])
            train_scans = len(split_results[split_results['split'] == 'train'])
            train_ad_scans = len(split_results[(split_results['split'] == 'train') & (split_results['group'] == 'AD')])
            train_control_scans = len(split_results[(split_results['split'] == 'train') & (split_results['group'] == 'Control')])
            val_scans = len(split_results[split_results['split'] == 'val'])
            val_ad_scans = len(split_results[(split_results['split'] == 'val') & (split_results['group'] == 'AD')])
            val_control_scans = len(split_results[(split_results['split'] == 'val') & (split_results['group'] == 'Control')])
            test_scans = len(split_results[split_results['split'] == 'test'])
            test_ad_scans = len(split_results[(split_results['split'] == 'test') & (split_results['group'] == 'AD')])
            test_control_scans = len(split_results[(split_results['split'] == 'test') & (split_results['group'] == 'Control')])
            logger.info(f"Dataset splits created:")
            logger.info("=" * 50)
            logger.info(f"SUBJECT COUNTS:")
            logger.info(f"  Train: {len(train_subjects)} subjects total")
            logger.info(f"    - AD: {train_ad_subjects} subjects")
            logger.info(f"    - Control: {train_control_subjects} subjects")
            logger.info(f"  Validation: {len(val_subjects)} subjects total")
            logger.info(f"    - AD: {val_ad_subjects} subjects")
            logger.info(f"    - Control: {val_control_subjects} subjects")
            logger.info(f"  Test: {len(test_subjects)} subjects total")
            logger.info(f"    - AD: {test_ad_subjects} subjects")
            logger.info(f"    - Control: {test_control_subjects} subjects")
            logger.info("-" * 50)
            logger.info(f"SCAN COUNTS:")
            logger.info(f"  Train: {train_scans} scans total")
            logger.info(f"    - AD: {train_ad_scans} scans")
            logger.info(f"    - Control: {train_control_scans} scans")
            logger.info(f"  Validation: {val_scans} scans total")
            logger.info(f"    - AD: {val_ad_scans} scans")
            logger.info(f"    - Control: {val_control_scans} scans")
            logger.info(f"  Test: {test_scans} scans total")
            logger.info(f"    - AD: {test_ad_scans} scans")
            logger.info(f"    - Control: {test_control_scans} scans")
            logger.info("=" * 50)
        
            # Average scans per subject, if needed/wanted
            avg_scans_per_subject = len(split_results) / len(all_splits)
            logger.info(f"Average scans per subject: {avg_scans_per_subject:.2f}")
        
            return split_results
            
                
                
    # This is here to COMPLEMENT NOT REPLACE checking the scans manually with a viewer  
    # Just nice to have      
    def visualize_samples(self, n_samples=3):
        try:
            splits_path = os.path.join(self.output_dir, "dataset_splits.csv")
            if not os.path.exists(splits_path):
                logger.error("Cannot visualize: dataset_splits.csv not found")
                return
            
            splits_df = pd.read_csv(splits_path)
            ad_samples = splits_df[splits_df['group'] == 'AD'].sample(n_samples)
            control_samples = splits_df[splits_df['group'] == 'Control'].sample(n_samples)
        
            samples = pd.concat([ad_samples, control_samples])
        
            fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*4, 8))
        
            for i, (_, row) in enumerate(samples.iterrows()):
                group_idx = 0 if row['group'] == 'AD' else 1
                sample_idx = i % n_samples
            
                img = nib.load(row['norm_file'])
                data = img.get_fdata()
                mid_slice = data.shape[2] // 2
            
                axes[group_idx, sample_idx].imshow(data[:, :, mid_slice], cmap='gray')
                axes[group_idx, sample_idx].set_title(f"{row['group']}: {row['subject_id']}")
                axes[group_idx, sample_idx].axis('off')
        
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "sample_visualization.png"))
            plt.close()
        
            logger.info(f"Sample visualization saved to {os.path.join(self.output_dir, 'sample_visualization.png')}")
        
        except Exception as e:
            logger.error(f"Error visualizing samples: {str(e)}")       
