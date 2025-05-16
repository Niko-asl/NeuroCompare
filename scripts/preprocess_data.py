import sys
import os


import os
import logging
import argparse
from tqdm import tqdm

from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocessor import AlzheimerPreprocessor

def parse_args():
   
    parser = argparse.ArgumentParser(description="Preprocess Alzheimer's MRI data")
    
    parser.add_argument('--ad_dir', type=str, required=True,
                       help='Directory containing AD patient data')
    parser.add_argument('--control_dir', type=str, required=True,
                       help='Directory containing control subject data')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data',
                       help='Directory to save preprocessed data')
    
    return parser.parse_args()

def main():

    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("preprocessing.log"),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    
    
    args = parse_args()
    
    # Initialize preprocessor
    preprocessor = AlzheimerPreprocessor(
        ad_dir=args.ad_dir,
        control_dir=args.control_dir,
        output_dir=args.output_dir
    )
    
    
    preprocessor.scan_directories()
    
    # Complete preprocessing pipeline
    results = preprocessor.run_preprocessing_pipeline()
    
    # Create dataset splits
    splits = preprocessor.create_dataset_splits(test_size=0.1, val_size=0.1)
    
    # Visualize
    preprocessor.visualize_samples(n_samples=3)
    
    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    main()
