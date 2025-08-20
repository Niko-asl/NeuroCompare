import os
import glob
import subprocess
import logging
import torch

logger = logging.getLogger(__name__)


class BrainExtractor:
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.skull_stripped_dir = os.path.join(output_dir, "skull_stripped")
        
        # HD-BET runs on both CPU and GPU
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.device_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU available for skull stripping: {self.device_name}")
            else:
                logger.info("No GPU available, will use CPU for skull stripping")
        except ImportError:
            self.gpu_available = False
            logger.info("PyTorch not installed, will use CPU for skull stripping")
    
    def extract(self, nifti_file):
        try:
            if nifti_file is None or not os.path.exists(nifti_file):
                logger.warning(f"Cannot perform skull stripping on non-existent file: {nifti_file}")
                return None
                
            # Create output paths
            basename = os.path.basename(nifti_file).replace('.nii.gz', '').replace('.nii', '')
            output_dir = os.path.join(self.output_dir, "skull_stripped")
            os.makedirs(output_dir, exist_ok=True)
            
            # Final output file we want
            final_output_file = os.path.join(output_dir, f"{basename}_brain.nii.gz")
            
            # Skip if already processed
            if os.path.exists(final_output_file):
                logger.info(f"Skull-stripped file already exists for {basename}, skipping")
                return final_output_file
            
            # HD-BET output will have the exact name we specify (no automatic suffix)
            hd_bet_output = os.path.join(output_dir, f"{basename}.nii.gz")
            
            # Set device parameter
            try:
                import torch
                if torch.cuda.is_available():
                    device_param = "cuda"
                    logger.info(f"Using GPU for HD-BET: {torch.cuda.get_device_name(0)}")
                else:
                    device_param = "cpu"
                    logger.info("Using CPU for HD-BET")
            except ImportError:
                device_param = "cpu"
                logger.warning("PyTorch not found, defaulting to CPU for HD-BET")
            
            # Run HD-BET command
            logger.info(f"Running HD-BET for {basename}")
            
            cmd = [
                "hd-bet",
                "-i", nifti_file,
                "-o", hd_bet_output,  # HD-BET will create exactly this file
                "-device", device_param
            ]
            
            logger.info(f"HD-BET command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if process.returncode != 0:
                logger.error(f"HD-BET failed with return code {process.returncode}")
                logger.error(f"HD-BET stderr: {process.stderr}")
                logger.error(f"HD-BET stdout: {process.stdout}")
                return None
            
            # HD-BET creates the file with the exact name we specified
            if os.path.exists(hd_bet_output):
                # Rename to our desired output filename
                os.rename(hd_bet_output, final_output_file)
                logger.info(f"HD-BET processing finished for: {os.path.basename(final_output_file)}")
                return final_output_file
            else:
                logger.error(f"HD-BET output not found: {hd_bet_output}")
                # List what files are actually in the directory for debugging
                all_files = os.listdir(output_dir)
                logger.error(f"Files in output directory: {all_files}")
                return None
            
        except Exception as e:
            logger.error(f"Error in HD-BET skull stripping for {nifti_file}: {str(e)}")
            return None
