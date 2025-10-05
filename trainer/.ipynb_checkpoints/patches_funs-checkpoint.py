
import random
import numpy as np

def patchify(spec, patch_size=(16, 16)):
    H, W = spec.shape
    ph, pw = patch_size
    patches = []
    positions = []

    for y in range(0, H - ph + 1, ph):
        for x in range(0, W - pw + 1, pw):
            patch = spec[y:y + ph, x:x + pw]
            patches.append(patch)
            positions.append((y, x))

    return patches, positions


def unpatchify(patches, positions, output_shape):
    H, W = output_shape
    output = np.zeros((H, W))
    for patch, (y, x) in zip(patches, positions):
        h, w = patch.shape
        output[y:y+h, x:x+w] = patch
    return output

### -------- MASKING -------- ###

def generate_random_patch_mask(num_patches, mask_ratio=0.75):
    num_mask = int(mask_ratio * num_patches)
    indices = list(range(num_patches))
    random.shuffle(indices)
    mask = np.zeros(num_patches, dtype=bool)
    mask[indices[:num_mask]] = True
    return mask

def mask_patches(patches, mask):
    masked = []
    for i, patch in enumerate(patches):
        if mask[i] == 1:
            masked.append(np.zeros_like(patch))
        else:
            masked.append(patch)
    return masked

