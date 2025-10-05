import glob
import torch
import random
import math
import numpy as np
from torch.utils.data import Dataset
from trainer.patches_funs import patchify, unpatchify, generate_random_patch_mask, mask_patches


class CustomDataset(Dataset):
    def __init__(self, data_directory, patch_size=(16, 16), mask_ratio=0.75, fixed_masking=False, return_directory=False):
        self.data_directory = data_directory
        self.files = []
        for directory in self.data_directory:
            self.files.extend(glob.glob(directory + "*.pt"))
        self.size = len([name for name in self.files])
        self.loaded = {}
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.fixed_masking = fixed_masking
        self.return_directory = return_directory

    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        if self.fixed_masking:
            random.seed(idx)
            np.random.seed(idx)
        else:
            seed_value = random.randint(1, 3000)
            random.seed(seed_value)
            np.random.seed(seed_value)
    
        if idx not in self.loaded:
            content = torch.load(self.files[idx], weights_only=False)
            self.loaded[idx] = content
    
        spectrogram = self.loaded[idx]
    
        # Crop
        ph, pw = self.patch_size
        H, W = spectrogram.shape
        H_crop = (H // ph) * ph
        W_crop = (W // pw) * pw
        spectrogram = spectrogram[:H_crop, :W_crop]
        spectrogram = spectrogram.numpy() if isinstance(spectrogram, torch.Tensor) else spectrogram
    
        patches, positions = patchify(spectrogram, self.patch_size)
        mask = generate_random_patch_mask(len(patches), self.mask_ratio)
        masked_patches = mask_patches(patches, mask)
        masked_spec = unpatchify(masked_patches, positions, spectrogram.shape)
    
        # Binary mask
        full_mask = np.zeros((H_crop, W_crop), dtype=bool)
        for m, (y, x) in zip(mask, positions):
            if m:
                full_mask[y:y+ph, x:x+pw] = True
    
        if self.return_directory: # for peaks evaluation or data labelling
            return torch.tensor(masked_spec).float(), (torch.tensor(spectrogram).float(), torch.tensor(full_mask).float()), self.files[idx]
        
        return torch.tensor(masked_spec).float(), (torch.tensor(spectrogram).float(), torch.tensor(full_mask).float())

