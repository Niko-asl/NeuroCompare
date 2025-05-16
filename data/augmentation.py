import numpy as np
from scipy.ndimage import rotate, shift, zoom, gaussian_filter
import torch.nn.functional as F
import torch

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
    
    @staticmethod
    def random_translation(data, max_shift=5, target_shape=(182, 218, 182)):
        # Generate random shifts in each dimension - max 5 voxels according to literature
        shifts = [np.random.randint(-max_shift, max_shift+1) for _ in range(3)]
        
        shifted = shift(data, shifts, mode='constant', cval=0, prefilter=False, order=1)
        
        shifted = MRIAugmentation.ensure_shape(shifted, target_shape)
        
        return np.ascontiguousarray(shifted)
    
    @staticmethod
    def random_scale(data, scale_range=(0.9, 1.1), target_shape=(182, 218, 182)):
        """Apply random scaling (zoom). (0.9 - 1.1×)"""
        # Use same zoom factor for all dimensions to maintain aspect ratio
        zoom_factor = np.random.uniform(*scale_range)
        
        zoomed = zoom(data, zoom_factor, mode='constant', cval=0, prefilter=False, order=1)
        
        zoomed = MRIAugmentation.ensure_shape(zoomed, target_shape)
        
        return np.ascontiguousarray(zoomed)
    
    # MIRRORING
    @staticmethod
    def left_right_flip(data, target_shape=(182, 218, 182)):
        """Random left-right flip (along x-axis). (prob = 0.5)"""
        # Flip along the 0 axis (Left-Right)
        # pretty important aug in literature, consider pushing to default
        flipped = np.flip(data, axis=0).copy()
        
        # Ensure correct shape (should already be correct, but for consistency)
        flipped = MRIAugmentation.ensure_shape(flipped, target_shape)
        
        return np.ascontiguousarray(flipped)
    
    # ELASTIC DEFORMATION
    @staticmethod
    def random_elastic_deformation(data, sigma=3, magnitude=10, target_shape=(182, 218, 182)):
        """Apply random elastic deformation using B-splines. (σ ≈ 3-5 vox, mag ≤ 10 vox)"""
        # Implementation using displacement fields
        shape = data.shape
        
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * magnitude
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * magnitude
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * magnitude
        
        # Generate mesh grid
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        
        indices = [np.reshape(x + dx, (-1, 1)), 
                   np.reshape(y + dy, (-1, 1)), 
                   np.reshape(z + dz, (-1, 1))]
        
        indices[0] = np.clip(indices[0], 0, shape[0] - 1)
        indices[1] = np.clip(indices[1], 0, shape[1] - 1)
        indices[2] = np.clip(indices[2], 0, shape[2] - 1)
        
        deformed = np.zeros_like(data)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    di = int(indices[0][i*shape[1]*shape[2] + j*shape[2] + k])
                    dj = int(indices[1][i*shape[1]*shape[2] + j*shape[2] + k])
                    dk = int(indices[2][i*shape[1]*shape[2] + j*shape[2] + k])
                    deformed[i, j, k] = data[di, dj, dk]
        
        deformed = MRIAugmentation.ensure_shape(deformed, target_shape)
        
        return np.ascontiguousarray(deformed)
    
    # GAUSSIAN NOISE
    @staticmethod
    def random_noise(data, std_percent=0.03, target_shape=(182, 218, 182)):
        # Gaussian noiser @ max 3% max intensity
        
        std = std_percent * np.max(np.abs(data))
        
        noise = np.random.normal(0, std, data.shape)
        noisy = data + noise
        
        noisy = MRIAugmentation.ensure_shape(noisy, target_shape)
        
        return np.ascontiguousarray(noisy)
    
    # MRI-SPECIFIC ARTIFACTS
    @staticmethod
    def random_bias_field(data, coeff=0.5, target_shape=(182, 218, 182)):
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
    
    @staticmethod
    def random_ghosting(data, max_ghosts=3, target_shape=(182, 218, 182)):

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
    
    # PHOTOMETRIC / INTENSITY
    @staticmethod
    def random_gamma(data, gamma_range=(0.7, 1.5), target_shape=(182, 218, 182)):
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
    
    @staticmethod
    def random_contrast(data, contrast_range=(0.7, 1.5), target_shape=(182, 218, 182)):
        # As above, but for contrast
        mean = np.mean(data)
        
        # Apply contrast adjustment
        factor = np.random.uniform(*contrast_range)
        contrasted = factor * (data - mean) + mean
        
        contrasted = MRIAugmentation.ensure_shape(contrasted, target_shape)
        
        return np.ascontiguousarray(contrasted)
    
    # SPATIAL CROPPING / PATCHING
    @staticmethod
    def random_crop(data, crop_size=(80, 80, 80), target_shape=(182, 218, 182)):
        """Apply random cropping, then resize back to original. (80-128 vox cubed)"""
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
    
    # CUTOUT
    @staticmethod
    def random_cutout(data, size_range=(20, 40), target_shape=(182, 218, 182)):
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
                print(f"Warning: Unknown augmentation '{aug}'. Skipping.")
        
        def augment(data):
            augmented = cls.ensure_shape(data.copy(), target_shape)

            for aug in valid_augmentations:
                if np.random.random() < p:
                    augmented = augmentation_map[aug](augmented, target_shape=target_shape)
            
            augmented = cls.ensure_shape(augmented, target_shape)
            return np.ascontiguousarray(augmented)
        
        return augment
