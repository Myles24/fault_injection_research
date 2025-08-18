from collections import Counter
import torch
import numpy as np

def get_free_data(path):
    data = np.loadtxt(path, delimiter=",")
    free = data[0].astype(int)
    faulty = data[1].astype(int)
    scale_factor = 0.13327007874015748
    scaled_array = free * scale_factor
    reshaped = scaled_array.reshape(64, 128, 19)
    outputs = torch.tensor(reshaped, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return outputs
def get_faulty_data(path):
    data = np.loadtxt(path, delimiter=",")
    faulty = data[1].astype(int)
    #scale_factor = 16.9253 / faulty.max()
    scale_factor = 0.13327007874015748
    scaled_array = faulty * scale_factor
    reshaped = scaled_array.reshape(1, 64, 128, 19)
    outputs = torch.tensor(reshaped, dtype=torch.float32).permute(0, 3, 1, 2)#.unsqueeze(0)
    print(outputs.shape)
    return outputs
#2D MEDIAN FILTERING
def get_corrected_data(path):
    data = np.loadtxt(path, delimiter=",")
    faulty = data[1].astype(int)
    faulty_3d = faulty.reshape(64, 128, 19)
    for z in range(19):
        slice_2d = faulty_3d[:, :, z]
        mask_thresh = slice_2d > 60
        if not np.any(mask_thresh):
            continue
        indices = np.argwhere(mask_thresh)
        x_min, y_min = indices.min(axis=0)
        x_max, y_max = indices.max(axis=0) + 1
        outside_mask = np.ones_like(slice_2d, dtype=bool)
        outside_mask[x_min:x_max, y_min:y_max] = False
        outside_values = slice_2d[outside_mask]
        if outside_values.size == 0:
            continue
        low_thresh = np.percentile(outside_values, 4) # gets the low value
        high_thresh = np.percentile(outside_values, 96) # gets the high value
        median_val = int(np.median(outside_values))
        print(f"z={z}: low={low_thresh:.1f}, high={high_thresh:.1f}, median={median_val}, region location x={x_min}:{x_max}, y={y_min}:{y_max}")
        region = slice_2d[x_min:x_max, y_min:y_max]
        region_mask = (region < low_thresh) | (region > high_thresh)
        region[region_mask] = median_val
        slice_2d[x_min:x_max, y_min:y_max] = region
        faulty_3d[:, :, z] = slice_2d
        faulty = faulty_3d.flatten()

    #np.savetxt("data/corrected.csv", [faulty], delimiter=",", fmt='%d')
    scale_factor = 0.13327007874015748
    scaled_array = faulty * scale_factor
    reshaped = scaled_array.reshape(64, 128, 19)
    outputs = torch.tensor(reshaped, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return outputs