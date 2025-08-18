from collections import Counter
import torch
import numpy as np
import cv2




def get_free_data(path):
    data = np.loadtxt(path, delimiter=",")
    free = data[0, 1:].astype(int)
    scale_factor = 1.4044969081878662 / free.max()
    #scale_factor = 1.1036 / free.max()
    scaled_array = free * scale_factor
    print(scale_factor)
    reshaped = scaled_array.reshape(64, 128, 32)
    outputs = torch.tensor(reshaped, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return outputs
def get_faulty_data(path):
    data = np.loadtxt(path, delimiter=",")
    faulty = data[1, 1:].astype(int)
    scale_factor = 1.4044969081878662 / faulty.max()
    #scale_factor = 1.1036 / faulty.max()
    scaled_array = faulty * scale_factor
    reshaped = scaled_array.reshape(64, 128, 32)
    outputs = torch.tensor(reshaped, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return outputs

#2D MEDIAN FILTERING
def get_corrected_data(path):
    data = np.loadtxt(path, delimiter=",")
    faulty = data[1, 1:].astype(int)
    faulty_3d = faulty.reshape(64, 128, 32)
    z0 = 0
    for z in range(31):
        if np.any(faulty_3d[:, :, z] == 127) and np.any(faulty_3d[:, :, z + 1] == 127):
            z0 = z
            break
    if z0 == 0:
        print("No faulty z-region found.")
        faulty = faulty_3d.flatten()
    else:
        print(f"Faulty z-region: z={0} to z={min(z0+15, 63)}")
        for z in range(32):
            slice_2d = faulty_3d[:, :, z]
            mask_89 = slice_2d > 89
            if not np.any(mask_89):
                continue
            indices = np.argwhere(mask_89)
            x_min, y_min = indices.min(axis=0)
            x_max, y_max = indices.max(axis=0) + 1
            # x_min, y_min = 48, 133
            # x_max, y_max = 64, 158
            outside_mask = np.ones_like(slice_2d, dtype=bool)
            outside_mask[x_min:x_max, y_min:y_max] = False
            outside_values = slice_2d[outside_mask]
            if outside_values.size == 0:
                continue
            low_thresh = np.percentile(outside_values, 4)
            high_thresh = np.percentile(outside_values, 96)
            median_val = int(np.median(outside_values))
            print(f"z={z}: low={low_thresh:.1f}, high={high_thresh:.1f}, median={median_val}, region location x={x_min}:{x_max}, y={y_min}:{y_max}")
            region = slice_2d[x_min:x_max, y_min:y_max]
            region_mask = (region < low_thresh) | (region > high_thresh)
            region[region_mask] = median_val
            slice_2d[x_min:x_max, y_min:y_max] = region
            faulty_3d[:, :, z] = slice_2d
        faulty = faulty_3d.flatten()

    np.savetxt("data/demo_results/corrected.csv", [faulty], delimiter=",", fmt='%d')
    scale_factor = 1.4044969081878662 / faulty.max()
    #scale_factor = 1.1036 / faulty.max()
    scaled_array = faulty * scale_factor
    reshaped = scaled_array.reshape(64, 128, 32)
    outputs = torch.tensor(reshaped, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return outputs
