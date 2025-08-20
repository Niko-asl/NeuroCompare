#!/usr/bin/env python3
"""
External Dataset Preprocessing Script
Preprocesses external AD/Ctrl dataset using the same pipeline as training data
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocessor import StreamlinedAlzheimerPreprocessor

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess external AD/Ctrl dataset for model evaluation")
    
    parser.add_argument('--external_data_dir', type=str, required=True,
                       help='Root directory containing AD and Ctrl folders')
    parser.add_argument('--output_dir', type=str, default='external_preprocessed_data',
                       help='Directory to save preprocessed external data')
    parser.add_argument('--ad_folder', type=str, default='AD',
                       help='Name of AD folder within external_data_dir (default: AD)')
    parser.add_argument('--ctrl_folder', type=str, default='Ctrl',
                       help='Name of control folder within external_data_dir (default: Ctrl)')
    parser.add_argument('--n_viz_samples', type=int, default=5,
                       help='Number of samples to visualize per group (default: 5)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()

def validate_external_dataset(external_data_dir, ad_folder, ctrl_folder):
    """Validate that the external dataset has the expected structure"""
    ad_path = os.path.join(external_data_dir, ad_folder)
    ctrl_path = os.path.join(external_data_dir, ctrl_folder)
    
    if not os.path.exists(external_data_dir):
        raise FileNotFoundError(f"External data directory not found: {external_data_dir}")
    
    if not os.path.exists(ad_path):
        raise FileNotFoundError(f"AD folder not found: {ad_path}")
    
    if not os.path.exists(ctrl_path):
        raise FileNotFoundError(f"Control folder not found: {ctrl_path}")
    
    # Check if folders contain any files
    ad_files = []
    ctrl_files = []
    
    for root, dirs, files in os.walk(ad_path):
        ad_files.extend([f for f in files if f.lower().endswith(('.dcm', '.nii', '.nii.gz'))])
    
    for root, dirs, files in os.walk(ctrl_path):
        ctrl_files.extend([f for f in files if f.lower().endswith(('.dcm', '.nii', '.nii.gz'))])
    
    if len(ad_files) == 0:
        raise ValueError(f"No DICOM or NIfTI files found in AD folder: {ad_path}")
    
    if len(ctrl_files) == 0:
        raise ValueError(f"No DICOM or NIfTI files found in Control folder: {ctrl_path}")
    
    return ad_path, ctrl_path, len(ad_files), len(ctrl_files)

def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("external_preprocessing.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting external dataset preprocessing...")
    logger.info(f"External data directory: {args.external_data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Validate external dataset structure
        logger.info("Step 1: Validating external dataset structure...")
        ad_path, ctrl_path, n_ad_files, n_ctrl_files = validate_external_dataset(
            args.external_data_dir, args.ad_folder, args.ctrl_folder
        )
        
        logger.info(f"Found {n_ad_files} AD files and {n_ctrl_files} Control files")
        logger.info(f"AD path: {ad_path}")
        logger.info(f"Control path: {ctrl_path}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize preprocessor with external dataset paths
        logger.info("Step 2: Initializing preprocessor...")
        preprocessor = StreamlinedAlzheimerPreprocessor(
            ad_dir=ad_path,
            control_dir=ctrl_path,
            output_dir=args.output_dir
        )
        
        # Scan directories and identify all files
        logger.info("Step 3: Scanning external dataset...")
        metadata = preprocessor.scan_directories()
        logger.info(f"Found {len(metadata)} total scans to process")
        
        # Run preprocessing pipeline
        logger.info("Step 4: Running preprocessing pipeline...")
        results = preprocessor.run_streamlined_preprocessing_pipeline()
        
        # Count successful processing
        successful = len(results.dropna(subset=['final_file']))
        failed = len(results) - successful
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        if successful == 0:
            logger.error("No scans were successfully processed. Check logs for errors.")
            return
        
        # Create evaluation dataset (no train/val split needed for external test set)
        logger.info("Step 5: Creating evaluation dataset...")
        eval_results = results.dropna(subset=['final_file']).copy()
        eval_results['split'] = 'external_test'  # Mark all as external test data
        
        # Save evaluation dataset
        eval_results.to_csv(os.path.join(args.output_dir, "external_dataset.csv"), index=False)
        logger.info(f"External evaluation dataset saved with {len(eval_results)} scans")
        
        # Create visualizations
        logger.info("Step 6: Creating sample visualizations...")
        preprocessor.visualize_samples(n_samples=args.n_viz_samples)
        
        # Summary statistics
        logger.info("="*60)
        logger.info("EXTERNAL DATASET PREPROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total scans found: {len(metadata)}")
        logger.info(f"Successfully processed: {successful}")
        logger.info(f"Failed processing: {failed}")
        
        ad_processed = len(eval_results[eval_results['group'] == 'AD'])
        ctrl_processed = len(eval_results[eval_results['group'] == 'Control'])
        logger.info(f"AD scans processed: {ad_processed}")
        logger.info(f"Control scans processed: {ctrl_processed}")
        
        if successful > 0:
            logger.info(f"Processed files saved to: {os.path.join(args.output_dir, 'final_processed')}")
            logger.info(f"Evaluation dataset saved to: {os.path.join(args.output_dir, 'external_dataset.csv')}")
            logger.info(f"Sample visualization saved to: {os.path.join(args.output_dir, 'sample_visualization.png')}")
        
        logger.info("External dataset preprocessing completed successfully!")
        logger.info(f"Ready for model evaluation using: {os.path.join(args.output_dir, 'external_dataset.csv')}")
        
    except Exception as e:
        logger.error(f"External dataset preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
