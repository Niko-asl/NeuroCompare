import numpy as np
from scipy.ndimage import rotate, shift, zoom, gaussian_filter
import torch.nn.functional as F
import torch
import logging

# Add basic logging
logger = logging.getLogger(__name__)

class MRIAugmentation:
    
    # Added to fix the fact that applying augmentation used to literally break the entire pipeline
    # Just makes sure that augmented MRI matches the dimensions of non-augmented MRI
    @staticmethod
    def ensure_shape(data, target_shape=(182, 218, 182)):
        current_shape = data.shape
        result = np.copy(data)
        
        for dim in range(3):
            if current_shape[dim] > target_shape[dim]:
                start_idx = (current_shape[dim] - target_shape[dim]) // 2
                slices = [slice(None)] * 3
                slices[dim] = slice(start_idx, start_idx + target_shape[dim])
                result = result[tuple(slices)]
            
            elif current_shape[dim] < target_shape[dim]:
                pad_total = target_shape[dim] - current_shape[dim]
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                padding = [(0, 0)] * 3
                padding[dim] = (pad_before, pad_after)
                
                result = np.pad(result, padding, mode='constant', constant_values=0)
        
        return result
    
    # RIGID/AFFINE GEOMETRY
    @staticmethod
    def random_rotation(data, max_angle=10, target_shape=(182, 218, 182)):
        """Apply random rotation with shape preservation. (±5-10° per axis)"""
        try:
            axes = [(0, 1), (0, 2), (1, 2)]
            axis = axes[np.random.choice(len(axes))]
            angle = np.random.uniform(-max_angle, max_angle)
            
            # Use rotate with reshape=False to maintain size
            # Otherwise everything breaks
            rotated = rotate(data, angle, axes=axis, reshape=False, mode='constant', 
                             cval=0, prefilter=False, order=1)
            
            # Ensure correct shape
            rotated = MRIAugmentation.ensure_shape(rotated, target_shape)
            
            # Ensure contiguous array
            # Otherwise everything breaks again
            return np.ascontiguousarray(rotated)
        except Exception as e:
            logger.error(f"Error in random_rotation: {e}")
            # Return original data if rotation fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_translation(data, max_shift=5, target_shape=(182, 218, 182)):
        try:
            # Generate random shifts in each dimension - max 5 voxels according to literature
            shifts = [np.random.randint(-max_shift, max_shift+1) for _ in range(3)]
            
            shifted = shift(data, shifts, mode='constant', cval=0, prefilter=False, order=1)
            
            shifted = MRIAugmentation.ensure_shape(shifted, target_shape)
            
            return np.ascontiguousarray(shifted)
        except Exception as e:
            logger.error(f"Error in random_translation: {e}")
            # Return original data if translation fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_scale(data, scale_range=(0.9, 1.1), target_shape=(182, 218, 182)):
        """Apply random scaling (zoom). (0.9 - 1.1×)"""
        try:
            # Use same zoom factor for all dimensions to maintain aspect ratio
            zoom_factor = np.random.uniform(*scale_range)
            
            zoomed = zoom(data, zoom_factor, mode='constant', cval=0, prefilter=False, order=1)
            
            zoomed = MRIAugmentation.ensure_shape(zoomed, target_shape)
            
            return np.ascontiguousarray(zoomed)
        except Exception as e:
            logger.error(f"Error in random_scale: {e}")
            # Return original data if scaling fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    # MIRRORING
    @staticmethod
    def left_right_flip(data, target_shape=(182, 218, 182)):
        """Random left-right flip (along x-axis). (prob = 0.5)"""
        try:
            # Flip along the 0 axis (Left-Right)
            # pretty important aug in literature, consider pushing to default
            flipped = np.flip(data, axis=0).copy()
            
            # Ensure correct shape (should already be correct, but for consistency)
            flipped = MRIAugmentation.ensure_shape(flipped, target_shape)
            
            return np.ascontiguousarray(flipped)
        except Exception as e:
            logger.error(f"Error in left_right_flip: {e}")
            # Return original data if flip fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    # ELASTIC DEFORMATION
    @staticmethod
    def random_elastic_deformation(data, sigma=3, magnitude=10, target_shape=(182, 218, 182)):
        """Random elastic deformation using displacement fields. (σ ≈ 3-5 vox, mag ≤ 10 vox)"""
        try:
            from scipy.ndimage import gaussian_filter, map_coordinates
       
            # First ensure the data is properly shaped
            data = MRIAugmentation.ensure_shape(data, target_shape)
            shape = data.shape
        
            # Create coordinate grid that matches the shape exactly
            coords = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing='ij'
            )
        
            downscale_factor = 4  # Reduce memory usage
            small_shape = (shape[0]//downscale_factor, shape[1]//downscale_factor, shape[2]//downscale_factor)
        
            # Generate small random fields and upscale them
            for i in range(3):
                # Create small displacement field
                field = np.random.rand(*small_shape) * 2 - 1
                field = gaussian_filter(field, sigma, mode="constant", cval=0)
                
                # Upscale to exactly match our target shape
                # We need to be careful about the exact shape here
                upscale_factors = [shape[j]/small_shape[j] for j in range(3)]
                field = zoom(field, upscale_factors, order=1, mode='constant', cval=0)
                
                # Ensure the field matches exactly by reshaping if needed
                if field.shape != shape:
                    field = MRIAugmentation.ensure_shape(field, shape)
                
                # Apply magnitude and add to coordinates
                coords[i] = coords[i] + field * magnitude
                
            # Clip coordinates to valid range
            for i in range(3):
                coords[i] = np.clip(coords[i], 0, shape[i] - 1)
        
            # Apply the deformation using map_coordinates (much faster than loops)
            deformed = map_coordinates(data, coords, order=1, mode='constant', cval=0)
        
            # Ensure output shape
            deformed = MRIAugmentation.ensure_shape(deformed, target_shape)
        
            return np.ascontiguousarray(deformed)
        except Exception as e:
            logger.error(f"Error in random_elastic_deformation: {e}")
            # Return original data if deformation fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    # GAUSSIAN NOISE
    @staticmethod
    def random_noise(data, std_percent=0.03, target_shape=(182, 218, 182)):
        try:
            # Gaussian noiser @ max 3% max intensity
            
            std = std_percent * np.max(np.abs(data))
            
            noise = np.random.normal(0, std, data.shape)
            noisy = data + noise
            
            noisy = MRIAugmentation.ensure_shape(noisy, target_shape)
            
            return np.ascontiguousarray(noisy)
        except Exception as e:
            logger.error(f"Error in random_noise: {e}")
            # Return original data if adding noise fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    # MRI-SPECIFIC ARTIFACTS
    @staticmethod
    def random_bias_field(data, coeff=0.5, target_shape=(182, 218, 182)):
        try:
            # This is a method to mimic scanner variability (?)
            
            shape = data.shape
            
            x_coords = np.linspace(-np.pi, np.pi, shape[0])
            y_coords = np.linspace(-np.pi, np.pi, shape[1])
            z_coords = np.linspace(-np.pi, np.pi, shape[2])
            
            num_basis = 3
            
            bias = np.zeros(shape)
            
            for i in range(num_basis):
                for j in range(num_basis):
                    for k in range(num_basis):
                        random_coeff = np.random.uniform(-coeff, coeff)
                        
                        x_component = np.cos(i * x_coords)[:, np.newaxis, np.newaxis]
                        y_component = np.cos(j * y_coords)[np.newaxis, :, np.newaxis]
                        z_component = np.cos(k * z_coords)[np.newaxis, np.newaxis, :]
                        
                        basis_function = x_component * y_component * z_component
                        bias += random_coeff * basis_function
            
            bias = np.exp(bias)
            
            biased_data = data * bias
            
            biased_data = MRIAugmentation.ensure_shape(biased_data, target_shape)
            
            return np.ascontiguousarray(biased_data)
        except Exception as e:
            logger.error(f"Error in random_bias_field: {e}")
            # Return original data if bias field fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_ghosting(data, max_ghosts=3, target_shape=(182, 218, 182)):
        try:
            # Simulate MRI ghosting
            
            result = data.copy()
            
            # Randomly select a direction for ghosting
            axis = np.random.randint(0, 3)
     
            num_ghosts = np.random.randint(1, max_ghosts + 1)
            
            # Calculate ghost intensity (decreasing with distance)
            for i in range(1, num_ghosts + 1):
                ghost_intensity = 0.5 / i
                
                shift_amount = i * np.random.randint(5, 15)
                
                shifts = [0, 0, 0]
                
                # Positive direction ghost
                shifts[axis] = shift_amount
                ghost_pos = shift(data, shifts, mode='constant', cval=0) * ghost_intensity
                result += ghost_pos
                
                # Negative direction ghost
                shifts[axis] = -shift_amount
                ghost_neg = shift(data, shifts, mode='constant', cval=0) * ghost_intensity
                result += ghost_neg
            
            result = MRIAugmentation.ensure_shape(result, target_shape)
            
            result = result * (np.max(data) / np.max(result)) if np.max(result) > 0 else result
            
            return np.ascontiguousarray(result)
        except Exception as e:
            logger.error(f"Error in random_ghosting: {e}")
            # Return original data if ghosting fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    # PHOTOMETRIC / INTENSITY
    @staticmethod
    def random_gamma(data, gamma_range=(0.7, 1.5), target_shape=(182, 218, 182)):
        try:
            # Apply random gamma adjustments (within range)
            min_val = data.min()
            data_pos = data - min_val
            
            max_val = data_pos.max()
            if max_val > 0:
                data_norm = data_pos / max_val
            else:
                return data.copy() 
            
            # Apply random gamma
            gamma = np.random.uniform(*gamma_range)
            data_gamma = np.power(data_norm, gamma)
            
            # Rescale back
            data_rescaled = data_gamma * max_val + min_val
            
            data_rescaled = MRIAugmentation.ensure_shape(data_rescaled, target_shape)
            
            return np.ascontiguousarray(data_rescaled)
        except Exception as e:
            logger.error(f"Error in random_gamma: {e}")
            # Return original data if gamma fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_contrast(data, contrast_range=(0.7, 1.5), target_shape=(182, 218, 182)):
        try:
            # As above, but for contrast
            mean = np.mean(data)
            
            # Apply contrast adjustment
            factor = np.random.uniform(*contrast_range)
            contrasted = factor * (data - mean) + mean
            
            contrasted = MRIAugmentation.ensure_shape(contrasted, target_shape)
            
            return np.ascontiguousarray(contrasted)
        except Exception as e:
            logger.error(f"Error in random_contrast: {e}")
            # Return original data if contrast fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    # SPATIAL CROPPING / PATCHING
    @staticmethod
    def random_crop(data, crop_size=(80, 80, 80), target_shape=(182, 218, 182)):
        """Apply random cropping, then resize back to original. (80-128 vox cubed)"""
        try:
            # Applies random cropping and resizes back to original
            # I'm apprehensive about this but apparently its used in the literature
            # I wanna say Perez et al..?
            
            original_shape = data.shape
            
            if isinstance(crop_size, int):
                crop_size = (crop_size, crop_size, crop_size)
            
            # Calculate valid ranges for the crop's top-left corner
            valid_ranges = [original_shape[i] - crop_size[i] for i in range(3)]
            
            # Check if cropping is valid (crop size smaller than image)
            if any(r < 0 for r in valid_ranges):
                crop_size = tuple(min(crop_size[i], original_shape[i]) for i in range(3))
                valid_ranges = [original_shape[i] - crop_size[i] for i in range(3)]
            
            # Generate random top-left corner within valid ranges
            corner = [np.random.randint(0, r+1) if r > 0 else 0 for r in valid_ranges]
            
            # Perform the crop
            cropped = data[
                corner[0]:corner[0] + crop_size[0],
                corner[1]:corner[1] + crop_size[1],
                corner[2]:corner[2] + crop_size[2]
            ]
            
            # Scale back to target shape
            # Use scipy's zoom for 3D
            scale_factors = [target_shape[i] / crop_size[i] for i in range(3)]
            scaled = zoom(cropped, scale_factors, mode='constant', cval=0, prefilter=False, order=1)
            
            scaled = MRIAugmentation.ensure_shape(scaled, target_shape)
            
            return np.ascontiguousarray(scaled)
        except Exception as e:
            logger.error(f"Error in random_crop: {e}")
            # Return original data if crop fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    # CUTOUT
    @staticmethod
    def random_cutout(data, size_range=(20, 40), target_shape=(182, 218, 182)):
        try:
            # Cuts out random 20-40 voxel patches, kinda like dropout 
            
            result = data.copy()
            
            shape = data.shape

            cutout_size = np.random.randint(*size_range)
            
            # Number of cutouts (randomly 1-3)
            num_cutouts = np.random.randint(1, 4)
            
            for _ in range(num_cutouts):
                # Random center point
                center = [np.random.randint(0, s) for s in shape]
                
                # Calculate boundaries
                start = [max(0, c - cutout_size // 2) for c in center]
                end = [min(s, c + cutout_size // 2) for c, s in zip(center, shape)]
                
                # Apply cutout (set to zero)
                result[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 0
            
            result = MRIAugmentation.ensure_shape(result, target_shape)
            
            return np.ascontiguousarray(result)
        except Exception as e:
            logger.error(f"Error in random_cutout: {e}")
            # Return original data if cutout fails
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))

    # Compose method that accepts specific augmentations to apply
    @classmethod
    def compose(cls, augmentations=None, p=0.5, target_shape=(182, 218, 182)):
        """
        Compose specified augmentations with probability p.
        
        Parameters:
        -----------
        augmentations : list of str or None
            List of augmentation names to apply. If None, apply all.
        p : float
            Probability of applying each augmentation.
        target_shape : tuple
            Target shape for output volume.
        """
        # Define mapping of augmentation names to methods
        augmentation_map = {
            # Rigid/affine
            'rotation': cls.random_rotation,
            'translation': cls.random_translation,
            'scale': cls.random_scale,
            # Mirroring
            'flip': cls.left_right_flip, 
            # Elastic
            'deformation': cls.random_elastic_deformation,
            # Noise
            'noise': cls.random_noise,
            # MRI artifacts
            'bias': cls.random_bias_field,
            'ghosting': cls.random_ghosting,
            # Photometric
            'gamma': cls.random_gamma,
            'contrast': cls.random_contrast,
            # Spatial
            'crop': cls.random_crop,
            # Cutout
            'cutout': cls.random_cutout
        }
        
        if augmentations is None:
            augmentations = list(augmentation_map.keys())
        
        valid_augmentations = []
        for aug in augmentations:
            if aug in augmentation_map:
                valid_augmentations.append(aug)
            else:
                logger.warning(f"Warning: Unknown augmentation '{aug}'. Skipping.")
        
        def augment(data):
            try:
                # Check for None input
                if data is None:
                    logger.error("Input data is None")
                    return None
                    
                # Check for NaN/Inf values
                if np.isnan(data).any() or np.isinf(data).any():
                    logger.warning("Input data contains NaN or Inf values - replacing with zeros")
                    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
                
                augmented = cls.ensure_shape(data.copy(), target_shape)

                for aug in valid_augmentations:
                    if np.random.random() < p:
                        try:
                            augmented = augmentation_map[aug](augmented, target_shape=target_shape)
                            
                            # Check for NaN/Inf values after augmentation
                            if np.isnan(augmented).any() or np.isinf(augmented).any():
                                logger.warning(f"{aug} produced NaN/Inf values - using original")
                                augmented = cls.ensure_shape(data.copy(), target_shape)
                                break
                        except Exception as e:
                            logger.error(f"Error applying {aug} augmentation: {e}")
                            # Continue with current result
                
                # Final shape check
                augmented = cls.ensure_shape(augmented, target_shape)
                
                # Final NaN/Inf check
                if np.isnan(augmented).any() or np.isinf(augmented).any():
                    logger.error("Augmented data contains NaN/Inf values - returning original")
                    return np.ascontiguousarray(cls.ensure_shape(data, target_shape))
                
                return np.ascontiguousarray(augmented)
            except Exception as e:
                logger.error(f"Error in augmentation pipeline: {e}")
                # Return original data on error
                return np.ascontiguousarray(cls.ensure_shape(data, target_shape))
        
        return augment
