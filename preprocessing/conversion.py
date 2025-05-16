import os
import glob
import subprocess
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Does what it says
class DicomConverter:
   
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.nifti_dir = os.path.join(output_dir, "nifti")
    
    def convert(self, row):
        try:
            subject_id = row['subject_id']
            group = row['group']
            scan_type = row['scan_type']
            scan_date = row['scan_date']
            dicom_path = row['dicom_path']
            
            # filename housekeeping
            safe_scan_date = scan_date.replace(':', '_').replace(' ', '_')
            
            output_basename = f"{subject_id}_{group}_{scan_type}_{safe_scan_date}"
            output_dir = os.path.join(self.output_dir, "nifti")
            
            # Check if output already exists
            expected_output = os.path.join(output_dir, f"{output_basename}.nii.gz")
            if os.path.exists(expected_output):
                logger.info(f"NIfTI file already exists for {output_basename}, skipping conversion")
                return expected_output
            
            # Run dcm2niix
            cmd = [
                'dcm2niix',
                '-z', 'y',  
                '-f', output_basename,  
                '-o', output_dir,  
                dicom_path  
            ]
            
            logger.info(f"Converting DICOM to NIfTI for {output_basename}")
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if process.returncode != 0:
                logger.error(f"dcm2niix error: {process.stderr}")
                return None
            
            
            output_files = glob.glob(os.path.join(output_dir, f"{output_basename}*.nii.gz"))
            if not output_files:
                logger.warning(f"No output files found for {output_basename}")
                return None
            
            return output_files[0]
            
        except Exception as e:
            logger.error(f"Error converting DICOM to NIfTI for {row['subject_id']}: {str(e)}")
            return None
