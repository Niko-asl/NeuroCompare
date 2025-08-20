import os
import logging
from tqdm import tqdm
import tempfile
import shutil
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nibabel as nib

# Import the same processing classes
from .conversion import DicomConverter
from .brain_extraction import BrainExtractor
from .normalization import SpatialNormalizer, IntensityNormalizer

logger = logging.getLogger(__name__)


class StreamlinedAlzheimerPreprocessor:
    
    def __init__(self, ad_dir, control_dir, output_dir):
        self.ad_dir = ad_dir
        self.control_dir = control_dir
        self.output_dir = output_dir
        
        # Only create final output directory
        os.makedirs(os.path.join(output_dir, "final_processed"), exist_ok=True)
        
        # Create temporary directory for intermediate processing
        self.temp_dir = tempfile.mkdtemp(prefix="alzheimer_preprocessing_")
        
        # Create all necessary temp subdirectories
        os.makedirs(os.path.join(self.temp_dir, "nifti"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "skull_stripped"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "normalized_space"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "normalized"), exist_ok=True)
        
        # Initialize processors with temp directory
        self.converter = DicomConverter(self.temp_dir)
        self.brain_extractor = BrainExtractor(self.temp_dir)
        self.spatial_normalizer = SpatialNormalizer(self.temp_dir)
        self.intensity_normalizer = IntensityNormalizer(self.temp_dir)
        
        # Initialize metadata
        self.metadata = []
        
        logger.info(f"Initialized streamlined preprocessor with AD dir: {ad_dir}, Control dir: {control_dir}")
        logger.info(f"Using temporary directory: {self.temp_dir}")
    
    def __del__(self):
        """Clean up temporary directory when object is destroyed"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def scan_directories(self):
        logger.info("Scanning directories for DICOM series and NIfTI files...")
        
        def process_directory(base_dir, label):
            subjects = []
            
            # Walk through all directories recursively
            for root, dirs, files in os.walk(base_dir):
                # Check if this directory contains DICOM files
                dicom_files = [f for f in files if f.lower().endswith('.dcm')]
                # Check if this directory contains NIfTI files
                nifti_files = [f for f in files if f.lower().endswith(('.nii', '.nii.gz'))]
                
                if dicom_files or nifti_files:
                    # Extract subject info from path
                    path_parts = os.path.relpath(root, base_dir).split(os.sep)
                    
                    if dicom_files:
                        # DICOM processing
                        subject_id = path_parts[0] if len(path_parts) > 0 else os.path.basename(root)
                        scan_type = path_parts[1] if len(path_parts) > 1 else "Unknown"
                        scan_date = path_parts[2] if len(path_parts) > 2 else "Unknown"
                        series_folder = path_parts[3] if len(path_parts) > 3 else "Unknown"
                        
                        subjects.append({
                            "subject_id": subject_id,
                            "scan_type": scan_type,
                            "scan_date": scan_date,
                            "series_folder": series_folder,
                            "dicom_path": root,
                            "group": label,
                            "file_type": "DICOM"
                        })
                    
                    if nifti_files:
                        # NIfTI processing - one entry per NIfTI file
                        for nifti_file in nifti_files:
                            subject_id = path_parts[0] if len(path_parts) > 0 else os.path.splitext(nifti_file)[0]
                            scan_info = os.path.splitext(nifti_file)[0]  # Use filename as scan info
                            
                            subjects.append({
                                "subject_id": subject_id,
                                "scan_type": scan_info,
                                "scan_date": "Unknown",
                                "series_folder": "NIfTI",
                                "dicom_path": os.path.join(root, nifti_file),
                                "group": label,
                                "file_type": "NIfTI"
                            })
            
            return subjects
        
        # Process directories separately then fuse
        ad_subjects = process_directory(self.ad_dir, "AD")
        control_subjects = process_directory(self.control_dir, "Control")
        all_subjects = ad_subjects + control_subjects
        self.metadata = pd.DataFrame(all_subjects)
        
        logger.info(f"Found {len(ad_subjects)} AD scans and {len(control_subjects)} Control scans")
        logger.info(f"DICOM files: {len(self.metadata[self.metadata['file_type'] == 'DICOM'])}")
        logger.info(f"NIfTI files: {len(self.metadata[self.metadata['file_type'] == 'NIfTI'])}")
        
        # Save metadata
        self.metadata.to_csv(os.path.join(self.output_dir, "metadata.csv"), index=False)
        return self.metadata
    
    # Replace the process_single_subject_streamlined method in preprocessing/preprocessor.py

    def process_single_subject_streamlined(self, row):
        """Process a single subject through the entire pipeline without saving intermediates."""
        subject_id = row['subject_id']
        logger.info(f"Processing subject {subject_id}")
    
        # FIXED: More robust scan ID extraction that doesn't fail
        scan_id = self._extract_scan_id_robust(row)
    
        # Check if final output already exists
        final_filename = f"{subject_id}_{row['group']}_{scan_id}_processed.nii.gz"
        final_output = os.path.join(self.output_dir, "final_processed", final_filename)
    
        if os.path.exists(final_output):
            logger.info(f"Final processed file already exists for {subject_id} ({scan_id}), skipping entire pipeline")
            return {
                'subject_id': subject_id,
                'group': row['group'],
                'final_file': final_output,
                'scan_type': row['scan_type'],
                'file_type': row['file_type'],
                'scan_id': scan_id
            }
    
        try:
            # Step 1: Handle input (DICOM conversion or direct NIfTI)
            if row['file_type'] == 'DICOM':
                logger.info(f"Converting DICOM for {subject_id}")
                nifti_file = self.converter.convert(row)
                if nifti_file is None:
                    logger.error(f"DICOM conversion failed for {subject_id}")
                    return {'subject_id': subject_id, 'group': row['group'], 'error': 'DICOM conversion failed'}
                logger.info(f"DICOM conversion successful for {subject_id}: {nifti_file}")
            else:  # NIfTI file
                nifti_file = row['dicom_path']  # This is actually the NIfTI file path
                if not os.path.exists(nifti_file):
                    logger.error(f"NIfTI file not found for {subject_id}: {nifti_file}")
                    return {'subject_id': subject_id, 'group': row['group'], 'error': 'NIfTI file not found'}
                logger.info(f"Using existing NIfTI file for {subject_id}: {nifti_file}")
        
            # Step 2: Apply skull stripping with better error handling
            logger.info(f"Starting skull stripping for {subject_id}")
            try:
                brain_file = self.brain_extractor.extract(nifti_file)
                if brain_file is None:
                    logger.error(f"HD-BET returned None for {subject_id} - likely processing error")
                    return {'subject_id': subject_id, 'group': row['group'], 'error': 'Skull stripping failed - HD-BET returned None'}
                elif not os.path.exists(brain_file):
                    logger.error(f"HD-BET output file does not exist: {brain_file}")
                    return {'subject_id': subject_id, 'group': row['group'], 'error': 'Skull stripping failed - output file missing'}
                else:
                    logger.info(f"Skull stripping successful for {subject_id}: {brain_file}")
            except Exception as e:
                logger.error(f"Exception during skull stripping for {subject_id}: {str(e)}")
                return {'subject_id': subject_id, 'group': row['group'], 'error': f'Skull stripping exception: {str(e)}'}
        
            # Step 3: Apply spatial normalization (creates temporary file)
            logger.info(f"Starting spatial normalization for {subject_id}")
            mni_file = self.spatial_normalizer.normalize(brain_file)
            if mni_file is None:
                logger.error(f"Spatial normalization failed for {subject_id}")
                return {'subject_id': subject_id, 'group': row['group'], 'error': 'Spatial normalization failed'}
            logger.info(f"Spatial normalization successful for {subject_id}: {mni_file}")
        
            # Step 4: Apply intensity normalization (creates temporary file)
            logger.info(f"Starting intensity normalization for {subject_id}")
            temp_norm_file = self.intensity_normalizer.normalize(mni_file)
            if temp_norm_file is None:
                logger.error(f"Intensity normalization failed for {subject_id}")
                return {'subject_id': subject_id, 'group': row['group'], 'error': 'Intensity normalization failed'}
            logger.info(f"Intensity normalization successful for {subject_id}: {temp_norm_file}")
        
            logger.info(f"All processing steps completed for {subject_id}, temp file: {temp_norm_file}")
        
            # Step 5: Move final result to output directory (scan_id already calculated above)
            # final_filename and final_output already defined above
        
            # Ensure output directory exists
            os.makedirs(os.path.dirname(final_output), exist_ok=True)
            logger.info(f"Copying {temp_norm_file} to {final_output}")
        
            # Copy the final processed file to output directory
            if os.path.exists(temp_norm_file):
                shutil.copy2(temp_norm_file, final_output)
                logger.info(f"Successfully saved final processed file: {final_output}")
            else:
                logger.error(f"Temp file does not exist: {temp_norm_file}")
                return {'subject_id': subject_id, 'group': row['group'], 'error': 'Temp file missing for final copy'}
        
            # Clean up intermediate files immediately after processing this subject
            self._cleanup_subject_intermediates(subject_id)
        
            return {
                'subject_id': subject_id,
                'group': row['group'],
                'final_file': final_output,
                'scan_type': row['scan_type'],
                'file_type': row['file_type'],
                'scan_id': scan_id
            }
        
        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {str(e)}")
            return {'subject_id': subject_id, 'group': row['group'], 'error': str(e)}

    def _extract_scan_id_robust(self, row):
        # Strategy 1: Look for ADNI-style scan IDs (original logic)
        if row['file_type'] == 'DICOM':
            path_parts = row['dicom_path'].split(os.sep)
        else:
            path_parts = row['dicom_path'].split(os.sep)
    
        # Try to find ADNI-style scan ID (I followed by digits)
        for part in path_parts:
            if part.startswith('I') and len(part) > 1 and part[1:].isdigit():
                return part.lower()
    
        # Strategy 2: Use scan_type if it's informative
        if row['scan_type'] and row['scan_type'] not in ['Unknown', 'T1', 'T1W', 'T1w']:
            # Clean up scan_type to be filename-safe
            clean_scan_type = row['scan_type'].replace(' ', '_').replace('/', '_')
            clean_scan_type = ''.join(c for c in clean_scan_type if c.isalnum() or c in ['_', '-'])
            if clean_scan_type:
                return clean_scan_type.lower()
    
        # Strategy 3: Extract from filename
        if row['file_type'] == 'NIfTI':
            filename = os.path.basename(row['dicom_path'])
            # Remove extension
            name_without_ext = filename.replace('.nii.gz', '').replace('.nii', '')
            # If it's not just the subject ID, use it
            if name_without_ext != row['subject_id']:
                 # Clean it up
                clean_name = name_without_ext.replace(row['subject_id'], '').strip('_-')
                if clean_name:
                    return clean_name.lower()
    
        # Strategy 4: Use parts of the path
        useful_parts = []
        for part in path_parts[-3:]:  # Look at last 3 parts of path
            if part and part not in [row['subject_id'], row['group'], '.', '..']:
                # Clean and add if it looks useful
                clean_part = ''.join(c for c in part if c.isalnum() or c in ['_', '-'])
                if len(clean_part) > 1:
                    useful_parts.append(clean_part)
    
        if useful_parts:
            return '_'.join(useful_parts).lower()
    
        # Strategy 5: Generate a hash-based ID from the full path
        import hashlib
        path_hash = hashlib.md5(row['dicom_path'].encode()).hexdigest()[:8]
        logger.info(f"Generated hash-based scan ID for {row['subject_id']}: {path_hash}")
        return f"scan_{path_hash}"

# Add this method to the StreamlinedAlzheimerPreprocessor class
    
    def _cleanup_subject_intermediates(self, subject_id):
        """Clean up intermediate files for a specific subject"""
        try:
            # Find all files in temp directory that contain the subject_id
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    if subject_id in file and file.endswith(('.nii.gz', '.nii')):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up intermediate files for {subject_id}: {str(e)}")
    
    def run_streamlined_preprocessing_pipeline(self):
        """Run the preprocessing pipeline without saving intermediate files"""
        
        if not hasattr(self, 'metadata') or len(self.metadata) == 0:
            self.scan_directories()
        
        results = []
        
        logger.info("Starting streamlined preprocessing pipeline...")
        
        for idx, row in tqdm(self.metadata.iterrows(), desc="Processing subjects", total=len(self.metadata)):
            result = self.process_single_subject_streamlined(row)
            results.append(result)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "preprocessing_results.csv"), index=False)
        
        logger.info("Streamlined preprocessing pipeline complete")
        
        # Final cleanup of temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up all temporary files")
        
        return results_df
    
    def create_dataset_splits(self, test_size=0.1, val_size=0.1, random_state=42):
        """Create dataset splits using only final processed files"""
        
        results_path = os.path.join(self.output_dir, "preprocessing_results.csv")
        if not os.path.exists(results_path):
            logger.error("Cannot create splits: preprocessing_results.csv not found")
            return None
            
        results_df = pd.read_csv(results_path)
        
        # Filter valid results (those with final_file)
        valid_results = results_df.dropna(subset=['final_file'])
        
        # Get unique subjects for splitting
        subjects = valid_results[['subject_id', 'group']].drop_duplicates()
        
        # Split by subjects to avoid data leakage
        train_val_subjects, test_subjects = train_test_split(
            subjects, test_size=test_size, random_state=random_state, stratify=subjects['group']
        )
        
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
        
        # Merge with results
        split_results = valid_results.merge(all_splits[['subject_id', 'split']], on='subject_id')
        
        # Save splits
        split_results.to_csv(os.path.join(self.output_dir, "dataset_splits.csv"), index=False)
        
        # Log statistics
        self._log_split_statistics(train_subjects, val_subjects, test_subjects, split_results)
        
        return split_results
    
    def _log_split_statistics(self, train_subjects, val_subjects, test_subjects, split_results):
        """Log detailed statistics about the dataset splits"""
        
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
        
        avg_scans_per_subject = len(split_results) / len(pd.concat([train_subjects, val_subjects, test_subjects]))
        logger.info(f"Average scans per subject: {avg_scans_per_subject:.2f}")
    
    def visualize_samples(self, n_samples=3):
        """Visualize samples from the final processed files"""
        try:
            splits_path = os.path.join(self.output_dir, "dataset_splits.csv")
            if not os.path.exists(splits_path):
                logger.error("Cannot visualize: dataset_splits.csv not found")
                return
            
            splits_df = pd.read_csv(splits_path)
            ad_samples = splits_df[splits_df['group'] == 'AD'].sample(min(n_samples, len(splits_df[splits_df['group'] == 'AD'])))
            control_samples = splits_df[splits_df['group'] == 'Control'].sample(min(n_samples, len(splits_df[splits_df['group'] == 'Control'])))
        
            samples = pd.concat([ad_samples, control_samples])
        
            fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*4, 8))
            if n_samples == 1:
                axes = axes.reshape(2, 1)
        
            for i, (_, row) in enumerate(samples.iterrows()):
                group_idx = 0 if row['group'] == 'AD' else 1
                sample_idx = i % n_samples
            
                img = nib.load(row['final_file'])
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
