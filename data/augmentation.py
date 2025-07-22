import numpy as np
from scipy.ndimage import rotate, shift, zoom, gaussian_filter, map_coordinates
import torch.nn.functional as F
import torch
import logging

# QOL changes to bind augmentation to GPU instead of CPU
# Should save some train time
logger = logging.getLogger(__name__)
MONAI_AVAILABLE = False

try:
    from monai.transforms.spatial.array import (
        RandRotate as RandRotateArray,
        RandAffine as RandAffineArray,
        Flip as FlipArray,
        Rand3DElastic as Rand3DElasticArray
    )
    from monai.transforms import (
        RandFlip, RandGaussianNoise, RandAdjustContrast, 
        RandKSpaceSpikeNoise, RandHistogramShift
    )
    # debug readouts because I can't call the right function names
    MONAI_AVAILABLE = True
    print("  MONAI transforms loaded successfully - GPU acceleration enabled")
    print("  Using array-based spatial transforms (4D tensor format)")
    print("  GPU-accelerated: rotation, translation, scale, flip, noise, contrast")
    print("  GPU-accelerated MRI-specific: elastic deformation, k-space artifacts, histogram shift")
    print("  CPU fallback: bias field, crop, cutout")

# import error readouts
except ImportError as e:
    MONAI_AVAILABLE = False
    print(f"MONAI ImportError - using CPU fallback: {e}")
except Exception as e:
    MONAI_AVAILABLE = False
    print(f"MONAI error - using CPU fallback: {e}")


class MRIAugmentation:
    
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
        
        
        
    ######################################### EVERY METHOD SHOULD LOOK LIKE THIS ##################################################################################
    # Sample method documented - others follow identical structure
    @staticmethod
    def random_rotation(data, max_angle=10, target_shape=(182, 218, 182)):
        
        # every augment calls this otherwise everything breaks
        try:
            if MONAI_AVAILABLE and torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
                    
                    if isinstance(data, np.ndarray):
                        tensor_data = torch.from_numpy(data).float().unsqueeze(0)
                        tensor_data = tensor_data.cuda()
                    else:
                        tensor_data = data
                    
                    # Define function with monai call
                    transform = RandRotateArray(
                        range_x=np.radians(max_angle),
                        range_y=np.radians(max_angle), 
                        range_z=np.radians(max_angle),
                        prob=1.0,
                        keep_size=True,
                        mode='bilinear',
                        padding_mode='zeros'
                    )
                    
                    # apply function
                    rotated = transform(tensor_data)
                    
                    # Convert back to numpy
                    rotated = rotated.cpu().squeeze(0).numpy()  # Remove channel dim
                    
                    # Ensure correct shape
                    rotated = MRIAugmentation.ensure_shape(rotated, target_shape)
                    return np.ascontiguousarray(rotated)
                    
                except Exception: # on ANY monai failure
                    pass  # go back to CPU based implementation
            
            # CPU fallback called if pass
            axes = [(0, 1), (0, 2), (1, 2)]
            axis = axes[np.random.choice(len(axes))]
            angle = np.random.uniform(-max_angle, max_angle)
            
            rotated = rotate(data, angle, axes=axis, reshape=False, mode='constant', 
                             cval=0, prefilter=False, order=1)
            
            # Ensure correct shape
            rotated = MRIAugmentation.ensure_shape(rotated, target_shape)
            return np.ascontiguousarray(rotated)
            
        except Exception as e: 
            logger.error(f"Error in random_rotation: {e}")
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    ###############################################################################################################################################################
    
    @staticmethod
    def random_translation(data, max_shift=5, target_shape=(182, 218, 182)):
        try:
            if MONAI_AVAILABLE and torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
                  
                    if isinstance(data, np.ndarray):
                        tensor_data = torch.from_numpy(data).float().unsqueeze(0)
                        tensor_data = tensor_data.cuda()
                    else:
                        tensor_data = data
                    
                    transform = RandAffineArray(
                        translate_range=(max_shift, max_shift, max_shift),
                        prob=1.0,
                        mode='bilinear',
                        padding_mode='zeros'
                    )
                    
                    shifted = transform(tensor_data)
                    shifted = shifted.cpu().squeeze(0).numpy()
                    shifted = MRIAugmentation.ensure_shape(shifted, target_shape)
                    return np.ascontiguousarray(shifted)
                    
                except Exception:
                    pass  
            

            shifts = [np.random.randint(-max_shift, max_shift+1) for _ in range(3)]
            shifted = shift(data, shifts, mode='constant', cval=0, prefilter=False, order=1)
            shifted = MRIAugmentation.ensure_shape(shifted, target_shape)
            return np.ascontiguousarray(shifted)
            
        except Exception as e:
            logger.error(f"Error in random_translation: {e}")
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_scale(data, scale_range=(0.9, 1.1), target_shape=(182, 218, 182)):
        """Apply random scaling (zoom). (0.9 - 1.1×)"""
        try:
            if MONAI_AVAILABLE and torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
               
                    if isinstance(data, np.ndarray):
                        tensor_data = torch.from_numpy(data).float().unsqueeze(0)
                        tensor_data = tensor_data.cuda()
                    else:
                        tensor_data = data
                    
                    scale_factor = np.random.uniform(*scale_range)
                    transform = RandAffineArray(
                        scale_range=(scale_factor-1, scale_factor-1, scale_factor-1),
                        prob=1.0,
                        mode='bilinear',
                        padding_mode='zeros'
                    )
                    
                    zoomed = transform(tensor_data)
                    
                    zoomed = zoomed.cpu().squeeze(0).numpy()
                    zoomed = MRIAugmentation.ensure_shape(zoomed, target_shape)
                    return np.ascontiguousarray(zoomed)
                    
                except Exception:
                    pass
            
            zoom_factor = np.random.uniform(*scale_range)
            zoomed = zoom(data, zoom_factor, mode='constant', cval=0, prefilter=False, order=1)
            zoomed = MRIAugmentation.ensure_shape(zoomed, target_shape)
            return np.ascontiguousarray(zoomed)
            
        except Exception as e:
            logger.error(f"Error in random_scale: {e}")
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def left_right_flip(data, target_shape=(182, 218, 182)):
        """Random left-right flip (along x-axis). (prob = 0.5)"""
        try:
            if MONAI_AVAILABLE and torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
                    
                    if isinstance(data, np.ndarray):
                        tensor_data = torch.from_numpy(data).float().unsqueeze(0)
                        tensor_data = tensor_data.cuda()
                    else:
                        tensor_data = data
                    
                    transform = FlipArray(spatial_axis=0)
                    
                    flipped = transform(tensor_data)
                    
                    flipped = flipped.cpu().squeeze(0).numpy()
                    flipped = MRIAugmentation.ensure_shape(flipped, target_shape)
                    return np.ascontiguousarray(flipped)
                    
                except Exception:
                    pass  
            
            flipped = np.flip(data, axis=0).copy()
            flipped = MRIAugmentation.ensure_shape(flipped, target_shape)
            return np.ascontiguousarray(flipped)
            
        except Exception as e:
            logger.error(f"Error in left_right_flip: {e}")
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_elastic_deformation(data, sigma=3, magnitude=10, target_shape=(182, 218, 182)):
        """Random elastic deformation using displacement fields. (σ ≈ 3-5 vox, mag ≤ 10 vox)"""
        try:
            if MONAI_AVAILABLE and torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
                    
                    if isinstance(data, np.ndarray):
                        tensor_data = torch.from_numpy(data).float().unsqueeze(0)
                        tensor_data = tensor_data.cuda()
                    else:
                        tensor_data = data

                    transform = Rand3DElasticArray(
                        sigma_range=(sigma, sigma),
                        magnitude_range=(magnitude, magnitude),
                        prob=1.0
                    )
                    
                    deformed = transform(tensor_data)

                    deformed = deformed.cpu().squeeze(0).numpy()
                    deformed = MRIAugmentation.ensure_shape(deformed, target_shape)
                    return np.ascontiguousarray(deformed)
                    
                except Exception:
                    pass  

            data = MRIAugmentation.ensure_shape(data, target_shape)
            shape = data.shape
        
            coords = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing='ij'
            )
        
            downscale_factor = 4  
            small_shape = (shape[0]//downscale_factor, shape[1]//downscale_factor, shape[2]//downscale_factor)
        
            for i in range(3):
                field = np.random.rand(*small_shape) * 2 - 1
                field = gaussian_filter(field, sigma, mode="constant", cval=0)
                
                upscale_factors = [shape[j]/small_shape[j] for j in range(3)]
                field = zoom(field, upscale_factors, order=1, mode='constant', cval=0)
                
                if field.shape != shape:
                    field = MRIAugmentation.ensure_shape(field, shape)
                
                coords[i] = coords[i] + field * magnitude
                
            for i in range(3):
                coords[i] = np.clip(coords[i], 0, shape[i] - 1)
        
            deformed = map_coordinates(data, coords, order=1, mode='constant', cval=0)
            deformed = MRIAugmentation.ensure_shape(deformed, target_shape)
            return np.ascontiguousarray(deformed)
                
        except Exception as e:
            logger.error(f"Error in random_elastic_deformation: {e}")
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_noise(data, std_percent=0.03, target_shape=(182, 218, 182)):
        try:
            if MONAI_AVAILABLE and torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
                    
                    if isinstance(data, np.ndarray):
                        tensor_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
                        tensor_data = tensor_data.cuda()
                    else:
                        tensor_data = data
                    
                    std = std_percent * torch.max(torch.abs(tensor_data))
                    transform = RandGaussianNoise(prob=1.0, mean=0.0, std=std.item())
                    
                    noisy = transform(tensor_data)
                    
                    noisy = noisy.cpu().squeeze(0).squeeze(0).numpy()
                    noisy = MRIAugmentation.ensure_shape(noisy, target_shape)
                    return np.ascontiguousarray(noisy)
                    
                except Exception:
                    pass  
            
            std = std_percent * np.max(np.abs(data))
            noise = np.random.normal(0, std, data.shape)
            noisy = data + noise
            noisy = MRIAugmentation.ensure_shape(noisy, target_shape)
            return np.ascontiguousarray(noisy)
            
        except Exception as e:
            logger.error(f"Error in random_noise: {e}")
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    
    # CPU is generally fine for bias field - no monai here
    @staticmethod
    def random_bias_field(data, coeff=0.5, target_shape=(182, 218, 182)):
        try:
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
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_ghosting(data, max_ghosts=3, target_shape=(182, 218, 182)):
    # monai has no "ghosting" per se, but forums say k spike noise is basically the same thing
        try:
            if MONAI_AVAILABLE and torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
                    if isinstance(data, np.ndarray):
                        tensor_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
                        tensor_data = tensor_data.cuda()
                    else:
                        tensor_data = data
                    transform = RandKSpaceSpikeNoise(
                        prob=1.0,
                        intensity_range=(0.5, 1.0),
                        channel_wise=False
                    )
                    
                    ghosted = transform(tensor_data)
                    ghosted = ghosted.cpu().squeeze(0).squeeze(0).numpy()
                    ghosted = MRIAugmentation.ensure_shape(ghosted, target_shape)
                    return np.ascontiguousarray(ghosted)
                    
                except Exception:
                    pass  
            
            result = data.copy()
            axis = np.random.randint(0, 3)
            num_ghosts = np.random.randint(1, max_ghosts + 1)
            
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
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_gamma(data, gamma_range=(0.7, 1.5), target_shape=(182, 218, 182)):
    # again here randhistogramshift serves the same purpose as gamma change
        try:
            if MONAI_AVAILABLE and torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
                    
                    if isinstance(data, np.ndarray):
                        tensor_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
                        tensor_data = tensor_data.cuda()
                    else:
                        tensor_data = data

                    transform = RandHistogramShift(num_control_points=10, prob=1.0)
                    
                    adjusted = transform(tensor_data)
                    
                    adjusted = adjusted.cpu().squeeze(0).squeeze(0).numpy()
                    adjusted = MRIAugmentation.ensure_shape(adjusted, target_shape)
                    return np.ascontiguousarray(adjusted)
                    
                except Exception:
                    pass  
            
            min_val = data.min()
            data_pos = data - min_val
            
            max_val = data_pos.max()
            if max_val > 0:
                data_norm = data_pos / max_val
            else:
                return data.copy()
            
            gamma = np.random.uniform(*gamma_range)
            data_gamma = np.power(data_norm, gamma)
            data_rescaled = data_gamma * max_val + min_val
            data_rescaled = MRIAugmentation.ensure_shape(data_rescaled, target_shape)
            return np.ascontiguousarray(data_rescaled)
                
        except Exception as e:
            logger.error(f"Error in random_gamma: {e}")
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_contrast(data, contrast_range=(0.7, 1.5), target_shape=(182, 218, 182)):
        try:
            if MONAI_AVAILABLE and torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1).cuda()
                    
                    if isinstance(data, np.ndarray):
                        tensor_data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
                        tensor_data = tensor_data.cuda()
                    else:
                        tensor_data = data
                    
                    transform = RandAdjustContrast(gamma=contrast_range, prob=1.0)
                    
                    contrasted = transform(tensor_data)
                    contrasted = contrasted.cpu().squeeze(0).squeeze(0).numpy()
                    contrasted = MRIAugmentation.ensure_shape(contrasted, target_shape)
                    return np.ascontiguousarray(contrasted)
                    
                except Exception:
                    pass  
            mean = np.mean(data)
            factor = np.random.uniform(*contrast_range)
            contrasted = factor * (data - mean) + mean
            contrasted = MRIAugmentation.ensure_shape(contrasted, target_shape)
            return np.ascontiguousarray(contrasted)
                
        except Exception as e:
            logger.error(f"Error in random_contrast: {e}")
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_crop(data, crop_size=(80, 80, 80), target_shape=(182, 218, 182)):
    # cpu imp is fine here
        try:
            original_shape = data.shape
            
            if isinstance(crop_size, int):
                crop_size = (crop_size, crop_size, crop_size)
            
            valid_ranges = [original_shape[i] - crop_size[i] for i in range(3)]
            
            if any(r < 0 for r in valid_ranges):
                crop_size = tuple(min(crop_size[i], original_shape[i]) for i in range(3))
                valid_ranges = [original_shape[i] - crop_size[i] for i in range(3)]
            
            corner = [np.random.randint(0, r+1) if r > 0 else 0 for r in valid_ranges]
            
            cropped = data[
                corner[0]:corner[0] + crop_size[0],
                corner[1]:corner[1] + crop_size[1],
                corner[2]:corner[2] + crop_size[2]
            ]
            
            scale_factors = [target_shape[i] / crop_size[i] for i in range(3)]
            scaled = zoom(cropped, scale_factors, mode='constant', cval=0, prefilter=False, order=1)
            scaled = MRIAugmentation.ensure_shape(scaled, target_shape)
            return np.ascontiguousarray(scaled)
                
        except Exception as e:
            logger.error(f"Error in random_crop: {e}")
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))
    
    @staticmethod
    def random_cutout(data, size_range=(20, 40), target_shape=(182, 218, 182)):
    #cpu is fine here too
        try:
            result = data.copy()
            shape = data.shape
            cutout_size = np.random.randint(*size_range)
            num_cutouts = np.random.randint(1, 4)
            
            for _ in range(num_cutouts):
                center = [np.random.randint(0, s) for s in shape]
                start = [max(0, c - cutout_size // 2) for c in center]
                end = [min(s, c + cutout_size // 2) for c, s in zip(center, shape)]
                
                result[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 0
            
            result = MRIAugmentation.ensure_shape(result, target_shape)
            return np.ascontiguousarray(result)
                
        except Exception as e:
            logger.error(f"Error in random_cutout: {e}")
            return np.ascontiguousarray(MRIAugmentation.ensure_shape(data, target_shape))

    @classmethod
    def compose(cls, augmentations=None, p=0.5, target_shape=(182, 218, 182)):
        # Define mapping of augmentation names to methods
        augmentation_map = {
            'rotation': cls.random_rotation,
            'translation': cls.random_translation,
            'scale': cls.random_scale,
            'flip': cls.left_right_flip, 
            'deformation': cls.random_elastic_deformation,
            'noise': cls.random_noise,
            'bias': cls.random_bias_field,
            'ghosting': cls.random_ghosting,
            'gamma': cls.random_gamma,
            'contrast': cls.random_contrast,
            'crop': cls.random_crop,
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
                if data is None:
                    logger.error("Input data is None")
                    return None
                if np.isnan(data).any() or np.isinf(data).any():
                    logger.warning("Input data contains NaN or Inf values - replacing with zeros")
                    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
                
                augmented = cls.ensure_shape(data.copy(), target_shape)

                for aug in valid_augmentations:
                    if np.random.random() < p:
                        try:
                            augmented = augmentation_map[aug](augmented, target_shape=target_shape)
                            if np.isnan(augmented).any() or np.isinf(augmented).any():
                                logger.warning(f"{aug} produced NaN/Inf values - using original")
                                augmented = cls.ensure_shape(data.copy(), target_shape)
                                break
                        except Exception as e:
                            logger.error(f"Error applying {aug} augmentation: {e}")

                augmented = cls.ensure_shape(augmented, target_shape)
                
                if np.isnan(augmented).any() or np.isinf(augmented).any():
                    logger.error("Augmented data contains NaN/Inf values - returning original")
                    return np.ascontiguousarray(cls.ensure_shape(data, target_shape))
                
                return np.ascontiguousarray(augmented)
            except Exception as e:
                logger.error(f"Error in augmentation pipeline: {e}")
                # on error return base scan
                return np.ascontiguousarray(cls.ensure_shape(data, target_shape))
        
        return augment
