import os
import subprocess
import nibabel as nib
import numpy as np
import logging

logger = logging.getLogger(__name__)

# does what it says
class SpatialNormalizer:
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.normalized_space_dir = os.path.join(output_dir, "normalized_space")
        
        self.template = os.path.join(os.environ.get('FSLDIR', '/usr/local/fsl'), 
                                     'data/standard/MNI152_T1_1mm_brain.nii.gz')
        
        if not os.path.exists(self.template):
            logger.warning(f"MNI template not found at {self.template}")
    
    
    #Spatial normalisation function, leveraging FSL
    def normalize(self, brain_file):
        try:
            if brain_file is None or not os.path.exists(brain_file):
                logger.warning(f"Cannot normalize non-existent file: {brain_file}")
                return None
                
            basename = os.path.basename(brain_file).replace('_brain.nii.gz', '')
            output_file = os.path.join(self.output_dir, "normalized_space", f"{basename}_mni.nii.gz")
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if os.path.exists(output_file):
                logger.info(f"Spatially normalized file already exists for {basename}, skipping")
                return output_file
            
            # Path to MNI template (standard in FSL)
            # N.B this path may need to be adjusted based on your FSL installation
            template = os.path.join(os.environ.get('FSLDIR', '/usr/local/fsl'), 
                                   'data/standard/MNI152_T1_1mm_brain.nii.gz')
            
            if not os.path.exists(template):
                logger.error(f"MNI template not found at {template}")
                return None
            
            # Run FLIRT for registration to MNI space
            cmd = [
                'flirt',
                '-in', brain_file,
                '-ref', template,
                '-out', output_file,
                '-dof', '12',  
                '-interp', 'spline'  
            ]
            
            logger.info(f"Running spatial normalization for {basename}")
            
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            if process.returncode != 0:
                logger.error(f"Spatial normalization error: {process.stderr}")
                return None
                
            return output_file
        
        except Exception as e:
            logger.error(f"Error in spatial normalization: {str(e)}")
            return None

# Standard 2nd-98th percentile clipping normalizer that scales to 1-0 intensities
class IntensityNormalizer:
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.normalized_dir = os.path.join(output_dir, "normalized")
    
    def normalize(self, brain_file):
        try:
            if brain_file is None or not os.path.exists(brain_file):
                logger.warning(f"Cannot normalize non-existent file: {brain_file}")
                return None
                
            basename = os.path.basename(brain_file).replace('_brain.nii.gz', '')
            output_file = os.path.join(self.output_dir, "normalized", f"{basename}_norm.nii.gz")
            
            if os.path.exists(output_file):
                logger.info(f"Normalized file already exists for {basename}, skipping")
                return output_file
            
            # Load the skull-stripped image passed as param
            img = nib.load(brain_file)
            data = img.get_fdata()
            
            # Only include non-zero voxels in percentile calculation
            non_zero_mask = data > 0
            
            # Empty image warning (just in case)
            if not np.any(non_zero_mask):
                logger.warning(f"Image {basename} is empty (all zeros)")
                norm_img = nib.Nifti1Image(data, img.affine, img.header)
                nib.save(norm_img, output_file)
                return output_file
            
            # Intensity normalization
            non_zero_data = data[non_zero_mask]
            p2, p98 = np.percentile(non_zero_data, (2, 98))
            
            # only within the brain mask
            data_clipped = data.copy()
            data_clipped[non_zero_mask] = np.clip(data_clipped[non_zero_mask], p2, p98)
            
            # norm to 0-1
            if p98 > p2:  
                data_norm = np.zeros_like(data_clipped)
                data_norm[non_zero_mask] = (data_clipped[non_zero_mask] - p2) / (p98 - p2)
            else:
                data_norm = data_clipped
            
            # Save the normalized image
            norm_img = nib.Nifti1Image(data_norm, img.affine, img.header)
            nib.save(norm_img, output_file)
            
            logger.info(f"Normalized {basename}, output saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error in normalization for {brain_file}: {str(e)}")
            return None
