import os
import glob
import subprocess
import logging
import torch

logger = logging.getLogger(__name__)


# Skull stripping function 
# Makes use of the HD-BET, a deep learning skull stripping framework, found at https://github.com/MIC-DKFZ/HD-BET

class BrainExtractor:
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.skull_stripped_dir = os.path.join(output_dir, "skull_stripped")
        
        # HD-BET runs on both CPU and GPU, this is just here to log which one is being used
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
                
            # output path
            basename = os.path.basename(nifti_file).replace('.nii.gz', '')
            output_dir = os.path.join(self.output_dir, "skull_stripped")
            output_file = os.path.join(output_dir, f"{basename}_brain.nii.gz")  
            hd_bet_output = os.path.join(output_dir, f"{basename}.nii.gz")  #
            
            # Skip files already processed
            if os.path.exists(output_file):
                logger.info(f"Skull-stripped file already exists for {basename}, skipping")
                return output_file
            
            # This just exists to set the device_param in the HD-BET command
            # I probably could have done it above but I am not smart
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
                "-o", hd_bet_output,  
                "-device", device_param
                #"--save_bet_mask"  # Uncomment if you want to keep the brain mask, No immediate use for it in this case
            ]
            
            logger.info(f"HD-BET command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if process.returncode != 0:
                logger.error(f"HD-BET error: {process.stderr}")
                return None
            
            
            
            expected_bet_file = hd_bet_output.replace('.nii.gz', '_bet.nii.gz')
            
            if os.path.exists(expected_bet_file):
                os.rename(expected_bet_file, output_file)
                return output_file
            else:
                potential_files = glob.glob(os.path.join(output_dir, f"{basename}*.nii.gz"))
                brain_files = [f for f in potential_files if "mask" not in os.path.basename(f)]
                
                if brain_files:
                    found_file = brain_files[0]
                    os.rename(found_file, output_file)
                    logger.info(f"HD-BET Processing finished for: {os.path.basename(output_file)}")
                    return output_file
                
                return None
            
        except Exception as e:
            logger.error(f"Error in HD-BET skull stripping: {str(e)}")
            return None
