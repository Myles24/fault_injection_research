from collections import Counter
import numpy as np
from scipy.ndimage import median_filter


# other shape is (64, 128, 128)

def get_free_data(path):
    data = np.loadtxt(path, delimiter=",")
    free = data[0].astype(int)
    scale_factor = 15.875 / free.max()
    scaled_array = free * scale_factor
    reshaped = scaled_array.reshape(1, 128, 256, 64)
    return reshaped
def get_faulty_data(path):
    data = np.loadtxt(path, delimiter=",")
    faulty = data[1].astype(int)
    scale_factor = 15.875 / faulty.max()
    scaled_array = faulty * scale_factor
    reshaped = scaled_array.reshape(1, 128, 256, 64)
    return reshaped

# ERFNET WORKING VERSION (x,y,z bounded)
def get_corrected_data(path):
    data = np.loadtxt(path, delimiter=",")
    faulty = data[1].astype(int)
    faulty_3d = faulty.reshape(128, 256, 64)
    z0 = None
    for z in range(63): 
        if np.any(faulty_3d[:, :, z] == 127) and np.any(faulty_3d[:, :, z + 1] == 127):
            z0 = z
            break
    if z0 is None:
        print("No faulty z-region found.")
        corrected = faulty_3d
    else:
        print(f"Faulty z-region: z={z0} to z={min(z0 + 15, 63)}")
        for z in range(z0, min(z0 + 16, 64)):
            slice_2d = faulty_3d[:, :, z]
            mask_127 = slice_2d == 127
            if not np.any(mask_127):
                continue
            # Get bounding box coordinates of 127-values
            indices = np.argwhere(mask_127)
            x_min, y_min = indices.min(axis=0)
            x_max, y_max = indices.max(axis=0) + 1 
            print(f"[z={z}] Faulty bounding box: x={x_min}:{x_max}, y={y_min}:{y_max}")
            region = slice_2d[x_min:x_max, y_min:y_max]
            filtered = median_filter(region, size=9)
            region_mask = region > 63
            region[region_mask] = filtered[region_mask]
            slice_2d[x_min:x_max, y_min:y_max] = region
            faulty_3d[:, :, z] = slice_2d
        corrected = faulty_3d
    scale_factor = 15.875 / corrected.max()
    scaled_array = corrected * scale_factor
    reshaped = scaled_array.reshape(1, 128, 256, 64)
    return reshaped


