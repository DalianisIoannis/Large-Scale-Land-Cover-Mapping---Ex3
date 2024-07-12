import random
from typing import List
import zarr
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import Dataset

##################################################################################
##################################################################################

class RandomSamplePixels(object):
    """Randomly draw num_pixels from the available pixels in sample.
    If the total number of pixels is less than num_pixels, one arbitrary pixel is repeated.
    The valid_pixels keeps track of true and repeated pixels.

    Args:
        num_pixels (int): Number of pixels to sample.
    """

    def __init__(self, num_pixels):
        self.num_pixels = num_pixels

    def __call__(self, sample):
        
        pixels = sample['pixels']
        
        T, C, S = pixels.shape
        if S > self.num_pixels:
            indices = random.sample(range(S), self.num_pixels)
            x = pixels[:, :, indices]
            valid_pixels = np.ones(self.num_pixels)
        elif S < self.num_pixels:
            x = np.zeros((T, C, self.num_pixels))
            x[..., :S] = pixels
            x[..., S:] = np.stack([x[:, :, 0] for _ in range(S, self.num_pixels)], axis=-1)
            valid_pixels = np.array([1 for _ in range(S)] + [0 for _ in range(S, self.num_pixels)])
        else:
            x = pixels
            valid_pixels = np.ones(self.num_pixels)
        # Repeat valid_pixels across time
        valid_pixels = np.repeat(valid_pixels[np.newaxis].astype(np.float32), x.shape[0], axis=0)
        sample['pixels'] = x
        sample['valid_pixels'] = valid_pixels
        return sample

##################################################################################
##################################################################################

class Normalize(object):
    """Normalize by rescaling pixels to [0, 1]"""

    def __init__(self, max_pixel_value=65535):
        """max_pixel_value (int): Max value of pixels to move pixels to [0, 1]"""
        self.max_pixel_value = max_pixel_value

        # approximate max values
        max_parcel_box_m = 10000
        max_perimeter = max_parcel_box_m * 4
        max_area = max_parcel_box_m ** 2
        max_perimeter_area_ratio = max_perimeter
        max_cover_ratio = 1.0
        self.max_extra_values = np.array([max_perimeter, max_area, max_perimeter_area_ratio, max_cover_ratio])

    def __call__(self, sample):
        sample['pixels'] = np.clip(sample['pixels'], 0, self.max_pixel_value).astype(np.float32) / self.max_pixel_value
        return sample

##################################################################################
##################################################################################

class ToTensor(object):
    def __call__(self, sample):
        sample['pixels'] = torch.from_numpy(sample['pixels'].astype(np.float32))
        sample['valid_pixels'] = torch.from_numpy(sample['valid_pixels'].astype(np.float32))
        if isinstance(sample['label'], int):
            sample['label'] = torch.tensor(sample['label']).long()
        return sample

##################################################################################
##################################################################################

class PixelSetData(Dataset):
    def __init__(
            self,
            data_root,
            class_to_idx,
            true_labels,
            transform=None,
            ):
        
        super(PixelSetData, self).__init__()

        self.folder = os.path.join(data_root)
        self.data_folder = os.path.join(self.folder, "data")
        # self.meta_folder = os.path.join(self.folder, "meta")
        self.data_labels = true_labels
        self.transform = transform
        self.class_to_idx = class_to_idx
        # self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.samples = self._make_dataset()

        for i in range(6):
            print(self.samples[i])
    
    def __getitem__(self, index):
        # get item like timematch
        path, parcel_idx, y = self.samples[index]
        pixels = zarr.load(path)  # (T, C, S)

        sample = {
            "index": index,
            # "parcel_index": parcel_idx,  # mapping to metadata
            "pixels": pixels,
            "valid_pixels": np.ones(
                (pixels.shape[0], pixels.shape[-1]), dtype=np.float32),
            "label": y,
            "label_idx": self.class_to_idx[y],
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return len(self.samples)
    
    def _make_dataset(self):
        # metadata = pkl.load(open(os.path.join(self.meta_folder, "metadata.pkl"), "rb"))
        instances = []
        
        for zarr_file in os.listdir(self.data_folder):
            if zarr_file.endswith(".zarr"):

                zarr_idx = zarr_file.split(".")[-2]
                parcel_path = os.path.join(self.data_folder, f"{zarr_idx}.zarr")

                # is the label in the true_labels dictionary?
                if zarr_idx in self.data_labels:
                    class_name = self.data_labels[zarr_idx]
                    item = (parcel_path, zarr_idx, class_name)
                    instances.append(item)
        return instances

##################################################################################
##################################################################################

def split_dict_train_test(data_dict, test_size=0.2, random_seed=42):
    """
    Splits a dictionary of data into train and test sets based on the
    specified test_size and random_seed. The dictionary is assumed to
    contain integer keys and values. The function returns two dictionaries,
    where all the label categories are divided so that they correspond to
    the train and test size.
    """
    # Set the random seed for reproducibility
    random.seed(random_seed)
    
    # Group indices by class
    class_to_indices = {}
    for idx, label in data_dict.items():
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    # Split indices into train and test sets
    train_indices = []
    test_indices = []
    for indices in class_to_indices.values():
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_seed)
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    
    # Create train and test dictionaries
    train_dict = {idx: data_dict[idx] for idx in train_indices}
    test_dict = {idx: data_dict[idx] for idx in test_indices}
    
    return train_dict, test_dict

##################################################################################
##################################################################################

# def pad_sequences_collate_fn(samples: List[dict]) -> tuple:
#     """
#     Zero-pad (in front) each sample to enable batching. The longest
#     sequence defines the sequence length for the batch.
#     """
#     # Extract labels
#     labels = torch.tensor([v['label_idx'] for v in samples])
    
#     # Find the maximum sequence length in the batch
#     max_len = max(v['pixels'].shape[-1] for v in samples)
    
#     # Pad sequences and create padding masks
#     padded_pixels = []
#     padding_masks = []
#     for sample in samples:
#         pixels = sample['pixels']
#         seq_len = pixels.shape[-1]
#         pad_len = max_len - seq_len
        
#         # Pad with zeros at the beginning
#         padded_pixel = torch.nn.functional.pad(pixels, (pad_len, 0))
#         padded_pixels.append(padded_pixel)
        
#         # Create the padding mask
#         padding_mask = torch.zeros(max_len, dtype=torch.bool)
#         padding_mask[:pad_len] = True
#         padding_masks.append(padding_mask)
    
#     # Stack the padded pixels and masks
#     data = torch.stack(padded_pixels, dim=0)
#     key_mask = torch.stack(padding_masks, dim=0)

#     return data, labels, key_mask

def pad_sequences_collate_fn(samples: List[dict]) -> tuple:
    """
    Zero-pad (in front) each sample to enable batching. The longest
    sequence defines the sequence length for the batch.
    """
    # Extract labels
    labels = torch.tensor([v['label_idx'] for v in samples])
    
    # Find the maximum sequence length in the batch
    max_len = max(v['pixels'].shape[-1] for v in samples)
    
    # Pad sequences and create padding masks
    padded_pixels = []
    padding_masks = []
    for sample in samples:
        pixels = sample['pixels']  # Shape: [52, 10, SIZES]
        seq_len = pixels.shape[-1]
        pad_len = max_len - seq_len
        
        # Pad with zeros at the beginning of the last dimension
        padded_pixel = torch.nn.functional.pad(pixels, (pad_len, 0))
        padded_pixels.append(padded_pixel)
        
        # Create the padding mask
        
        # padding_mask = torch.zeros(max_len, dtype=torch.bool)
        # padding_mask = torch.zeros((pixels.shape[0], pixels.shape[1], max_len), dtype=torch.bool)
        padding_mask = torch.zeros((pixels.shape[0], max_len), dtype=torch.bool)
        
        # padding_mask[:pad_len] = True
        # padding_mask[:, :, :pad_len] = True
        padding_mask[:, :pad_len] = True
        
        padding_masks.append(padding_mask)
    
    # Stack the padded pixels and masks
    data = torch.stack(padded_pixels, dim=0)
    key_mask = torch.stack(padding_masks, dim=0)

    return data, labels, key_mask

##################################################################################
##################################################################################
