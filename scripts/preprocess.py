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
    
    parser.add_argument('--ad_dir', type=str, required=True,
                       help='Directory containing AD patient data')
    parser.add_argument('--control_dir', type=str, required=True,
                       help='Directory containing control subject data')
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
    
    # Validate input directories
    if not os.path.exists(args.ad_dir):
        logger.error(f"AD directory does not exist: {args.ad_dir}")
        return
    
    if not os.path.exists(args.control_dir):
        logger.error(f"Control directory does not exist: {args.control_dir}")
        return
    
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
            output_dir=args.output_dir
        )
        
        # Scan directories and identify all files
        logger.info("Step 1: Scanning directories...")
        metadata = preprocessor.scan_directories()
        logger.info(f"Found {len(metadata)} total scans to process")
        
        # Run the streamlined preprocessing pipeline
        logger.info("Step 2: Running streamlined preprocessing pipeline...")
        results = preprocessor.run_streamlined_preprocessing_pipeline()
        
        # Count successful processing
        successful = len(results.dropna(subset=['final_file']))
        failed = len(results) - successful
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        if successful == 0:
            logger.error("No scans were successfully processed. Check logs for errors.")
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
