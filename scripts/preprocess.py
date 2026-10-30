import sys
import os
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the new streamlined preprocessor
from preprocessing.preprocessor import StreamlinedAlzheimerPreprocessor

def parse_args():
    parser = argparse.ArgumentParser(description="Streamlined Alzheimer's MRI data preprocessing")
    
    parser.add_argument('--ad_dir', type=str, default=None,
                       help='Directory containing AD patient data (folder mode)')
    parser.add_argument('--control_dir', type=str, default=None,
                       help='Directory containing control subject data (folder mode)')
    parser.add_argument('--cohort', type=str, default=None,
                       help='Cohort name for folder mode, or fallback when the manifest has no cohort column')
    parser.add_argument('--manifest', type=str, default=None,
                       help='CSV with one row per scan; replaces --ad_dir/--control_dir')
    parser.add_argument('--path_col', type=str, default='path', help='Manifest column with the NIfTI path')
    parser.add_argument('--subject_col', type=str, default='sub_id', help='Manifest subject column')
    parser.add_argument('--label_col', type=str, default='label', help='Manifest label column')
    parser.add_argument('--cohort_col', type=str, default='Dataset', help='Manifest cohort column')
    parser.add_argument('--scan_col', type=str, default=None,
                       help='Manifest column with a per-scan ID (e.g. ses_id); default: file name')
    parser.add_argument('--ad_values', nargs='+', default=['AD', '1'], help='Label values meaning AD')
    parser.add_argument('--cn_values', nargs='+', default=['Control', 'CN', '0'], help='Label values meaning control')
    parser.add_argument('--shard_index', type=int, default=None,
                       help='Shard to process (default: $SLURM_ARRAY_TASK_ID, else 0)')
    parser.add_argument('--num_shards', type=int, default=1,
                       help='Total shards; with >1, splits are not created (run make_splits.py after all shards)')
    parser.add_argument('--output_dir', type=str, default='streamlined_preprocessed_data',
                       help='Directory to save final preprocessed data')
    parser.add_argument('--test_size', type=float, default=0.1,
                       help='Proportion of data for test set (default: 0.1)')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Proportion of data for validation set (default: 0.1)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducible splits (default: 42)')
    parser.add_argument('--n_viz_samples', type=int, default=3,
                       help='Number of samples to visualize per group (default: 3)')
    
    return parser.parse_args()

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("streamlined_preprocessing.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    args = parse_args()
    
    if args.manifest is None:
        if not (args.ad_dir and args.control_dir):
            logger.error("Give either --manifest or both --ad_dir and --control_dir")
            return
        for d in (args.ad_dir, args.control_dir):
            if not os.path.exists(d):
                logger.error(f"Directory does not exist: {d}")
                return
    if args.shard_index is None:
        args.shard_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting streamlined Alzheimer's MRI preprocessing pipeline")
    logger.info(f"AD directory: {args.ad_dir}")
    logger.info(f"Control directory: {args.control_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Initialize streamlined preprocessor
        preprocessor = StreamlinedAlzheimerPreprocessor(
            ad_dir=args.ad_dir,
            control_dir=args.control_dir,
            output_dir=args.output_dir,
            cohort=args.cohort
        )
        
        if args.manifest:
            logger.info("Step 1: Reading manifest...")
            metadata = preprocessor.load_manifest(
                args.manifest, path_col=args.path_col, subject_col=args.subject_col,
                label_col=args.label_col, cohort_col=args.cohort_col, scan_col=args.scan_col,
                ad_values=tuple(args.ad_values), cn_values=tuple(args.cn_values))
        else:
            logger.info("Step 1: Scanning directories...")
            metadata = preprocessor.scan_directories()
        logger.info(f"Found {len(metadata)} total scans to process")
        
        # Run the streamlined preprocessing pipeline
        logger.info("Step 2: Running streamlined preprocessing pipeline...")
        results = preprocessor.run_streamlined_preprocessing_pipeline(args.shard_index, args.num_shards)
        
        # Count successful processing
        # No 'final_file' column at all when every scan in the run failed
        successful = int(results['final_file'].notna().sum()) if 'final_file' in results else 0
        failed = len(results) - successful
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        if successful == 0:
            logger.error("No scans were successfully processed. Check logs for errors.")
            return
        
        if args.num_shards > 1:
            logger.info("Sharded run: skipping splits; run scripts/make_splits.py once all shards finish.")
            return
        
        # Create dataset splits
        logger.info("Step 3: Creating dataset splits...")
        splits = preprocessor.create_dataset_splits(
            test_size=args.test_size, 
            val_size=args.val_size,
            random_state=args.random_state
        )
        
        if splits is not None:
            logger.info(f"Dataset splits created with {len(splits)} total processed scans")
        
        # Create visualizations
        logger.info("Step 4: Creating sample visualizations...")
        preprocessor.visualize_samples(n_samples=args.n_viz_samples)
        
        # Summary statistics
        logger.info("="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total scans found: {len(metadata)}")
        logger.info(f"Successfully processed: {successful}")
        logger.info(f"Failed processing: {failed}")
        
        if successful > 0:
            logger.info(f"Final processed files saved to: {os.path.join(args.output_dir, 'final_processed')}")
            logger.info(f"Dataset splits saved to: {os.path.join(args.output_dir, 'dataset_splits.csv')}")
            logger.info(f"Sample visualization saved to: {os.path.join(args.output_dir, 'sample_visualization.png')}")
        
        logger.info("Streamlined preprocessing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise
    
    finally:
        # Cleanup is handled automatically by the preprocessor destructor
        logger.info("Pipeline finished.")

if __name__ == "__main__":
    main()
