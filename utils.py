import random
import copy
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
# from collections import defaultdict
# from typing import List
import math
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

        for i in range(4):
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
    
    # def get_shapes(self):
    #     return [
    #         (_, 10, zarr.load(parcel[0]).shape[-1])
    #         for parcel in self.samples
    #     ]
    
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

# prwto prwto
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

# merikes allages to panw
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
#         pixels = sample['pixels']  # Shape: [52, 10, SIZES]
#         seq_len = pixels.shape[-1]
#         pad_len = max_len - seq_len
        
#         # Pad with zeros at the beginning of the last dimension
#         padded_pixel = torch.nn.functional.pad(pixels, (pad_len, 0))
#         padded_pixels.append(padded_pixel)
        
#         # Create the padding mask
        
#         # padding_mask = torch.zeros(max_len, dtype=torch.bool)
#         # padding_mask = torch.zeros((pixels.shape[0], pixels.shape[1], max_len), dtype=torch.bool)
#         padding_mask = torch.zeros((pixels.shape[0], max_len), dtype=torch.bool)
        
#         # padding_mask[:pad_len] = True
#         # padding_mask[:, :, :pad_len] = True
#         padding_mask[:, :pad_len] = True
        
#         padding_masks.append(padding_mask)
    
#     # Stack the padded pixels and masks
#     data = torch.stack(padded_pixels, dim=0)
#     key_mask = torch.stack(padding_masks, dim=0)

#     return data, labels, key_mask

# apo nikola allagmeno tou tsironi
def pad_sequences_collate_fn(samples) -> tuple:
    """
    Zero-pad (in front) each sample to enable batching.
    The longest sequence defines the sequence length for the batch
    """
    # labels = torch.stack([torch.tensor(v[1]) for v in samples])
    labels = torch.tensor([v['label_idx'] for v in samples])

    # variable dimension must be first
    # data = pad_sequence([v[0].permute((2, 0, 1)) for v in samples], batch_first=True)
    data = pad_sequence([v['pixels'].permute((2, 0, 1)) for v in samples], batch_first=True)

    key_mask = pad_sequence(
        [
            # torch.ones((v[0].shape[-1], v[0].shape[0]), dtype=torch.bool)
            torch.ones((v['pixels'].shape[-1], v['pixels'].shape[0]), dtype=torch.bool)
            for v in samples
        ],
        padding_value=False,
        batch_first=True,
    )

    return data.permute((0, 2, 3, 1)), labels, key_mask.permute((0, 2, 1))

##################################################################################
##################################################################################

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):

        # x = x.permute(0, 2, 1)

        # x = (B, C) or (B, S, C)
        x = self.linear(x)  # linear expect channels last
        if x.dim() == 3:  
            # BatchNorm1d expects channels first, move to (B, C, S)
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:  # (B, C)
            x = self.norm(x)
        return self.activation(x)

##################################################################################
##################################################################################

def masked_mean(x, mask):
    
    # print("input of masked_mean unique", x.unique())
    # print("input of masked_mean shape", x.shape) # torch.Size([416, 32, 32])
    # print("mask of masked_mean shape", mask.shape) # torch.Size([416, 32])
    # print("mask of masked_mean uniques", mask.unique())
    out = x.permute((1, 0, 2)) # torch.Size([32, 416, 32])
    # print("mask.sum(dim=-1)", mask.sum(dim=-1))
    out = out * mask # torch.Size([32, 416, 32])
    # # here every pixel gets to be zero
    # print("out = out * mask shape", out.shape)
    # print("uniques of out = out * mask", out.unique())
    out = out.sum(dim=-1) / mask.sum(dim=-1) # torch.Size([32, 416])
    # print("out = out.sum(dim=-1) / mask.sum(dim=-1) shape", out.shape)
    # print("uniques here", out.unique())
    out = out.permute((1, 0))
    # print("out of masked_mean unique", out.unique())
    # print()
    return out

def masked_std(x, mask):
    # print("input of masked_std unique", x.unique())
    m = masked_mean(x, mask)

    out = x.permute((2, 0, 1))
    out = out - m
    out = out.permute((2, 1, 0))

    out = out * mask
    d = mask.sum(dim=-1)
    d[d == 1] = 2

    out = (out ** 2).sum(dim=-1) / (d - 1)
    out = torch.sqrt(out + 10e-32) # To ensure differentiability
    out = out.permute(1, 0)
    # print("out of masked_std unique", out.unique())
    # print()
    return out

# pooling_methods = {
#     "mean": masked_mean,
#     "std": masked_std}

##################################################################################
##################################################################################

class PixelSetEncoder(nn.Module):
    def __init__(
        self, input_dim,
        mlp1=[10, 32, 64],
        pooling="mean_std",
        mlp2=[64, 128],
    ):
        """
        Pixel-set encoder.
        Args:
            input_dim (int): Number of channels of the input tensors
            mlp1 (list):  Dimensions of the successive feature spaces of MLP1
            pooling (str): Pixel-embedding pooling strategy, can be chosen in ('mean','std','max,'min')
                or any underscore-separated combination thereof.
            mlp2 (list): Dimensions of the successive feature spaces of MLP2
            with_extra (bool): Whether additional pre-computed features are passed between the two MLPs
            extra_size (int, optional): Number of channels of the additional features, if any.
        """

        super(PixelSetEncoder, self).__init__()

        self.input_dim = input_dim
        self.mlp1_dim = copy.deepcopy(mlp1)
        self.mlp2_dim = copy.deepcopy(mlp2)
        self.pooling = pooling

        self.pooling_methods = {
            "mean": masked_mean,
            "std": masked_std}

        self.output_dim = (
            input_dim * len(pooling.split("_"))
            if len(self.mlp2_dim) == 0
            else self.mlp2_dim[-1]
        )

        # inter_dim = self.mlp1_dim[-1] * len(pooling.split("_"))
        # if self.with_extra:
        #     inter_dim += self.extra_size
        # assert input_dim == mlp1[0]
        # assert inter_dim == mlp2[0]

        # Feature extraction
        layers = []
        for i in range(len(self.mlp1_dim) - 1):
            layers.append(LinearLayer(self.mlp1_dim[i], self.mlp1_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        # MLP after pooling
        layers = []
        for i in range(len(self.mlp2_dim) - 1):
            layers.append(LinearLayer(self.mlp2_dim[i], self.mlp2_dim[i + 1]))
        self.mlp2 = nn.Sequential(*layers)

    def forward(self, pixels, mask):
        """
        The input of the PSE is a tuple of tensors as yielded by the PixelSetData class:
          (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
        Pixel-Set : Batch_size x (Sequence length) x Channel x Number of pixels
        Pixel-Mask : Batch_size x (Sequence length) x Number of pixels
        Extra-features : Batch_size x (Sequence length) x Number of features

        If the input tensors have a temporal dimension, it will be combined with the batch dimension so that the
        complete sequences are processed at once. Then the temporal dimension is separated back to produce a tensor of
        shape Batch_size x Sequence length x Embedding dimension
        """
        out = pixels

        # print("pixels in PSE", out.shape)

        batch, temp = out.shape[:2]

        # out = out.view(batch * temp, *out.shape[2:]).transpose(1, 2)  # (B*T, S, C)
        out = out.reshape(batch * temp, *out.shape[2:]).transpose(1, 2)  # (B*T, S, C)
        
        # mask = mask.view(batch * temp, -1)
        mask = mask.reshape(batch * temp, -1)

        out = self.mlp1(out).transpose(1, 2)

        # print("in pse out mask uniques", mask.unique())
        # print("in pse out mask uniques type", type(mask.unique()))
        
        # if not torch.equal(mask.unique(), torch.tensor([False])):
        out = torch.cat(
            # [pooling_methods[n](out, mask) for n in self.pooling.split("_")], dim=1
            [self.pooling_methods[n](out, mask) for n in self.pooling.split("_")], dim=1
        )
        # else:
        #     out = torch.cat(
        #         [out.mean(dim=-1), out.std(dim=-1)], dim=1
        #     )

        out = self.mlp2(out)
        out = out.view(batch, temp, -1)
        return out
    
##################################################################################
##################################################################################

class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                #  dropout: float = 0.1,
                #  max_len: int = 5000
                 max_len: int = 52
                 ):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

##################################################################################
##################################################################################

# from torch.nn import TransformerEncoder, TransformerEncoderLayer
# class TimeSeriesTransformer(nn.Module):
#     def __init__(self,
#                  feature_size,
#                  num_classes: int = 7,
#                  num_layers=3, num_heads=8,
#                 #  hidden_dim=512, dropout=0.1
#                  ):
#         super(TimeSeriesTransformer, self).__init__()
        
#         self.pos_encoder = PositionalEncoding(feature_size)
#         self.encoder_layers = TransformerEncoderLayer(
#             d_model=feature_size, nhead=num_heads,
#             # dim_feedforward=hidden_dim,
#             # dropout=dropout,
#             batch_first=True
#             )
#         self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers)
        
#         self.classification_token = nn.Parameter(torch.zeros(1, 1, feature_size),
#                                                  requires_grad=True
#                                                  )
#         self.fc = nn.Linear(feature_size, num_classes)  # Adjust the output dimension as needed
        
#     def forward(self,
#                 x: torch.Tensor,
#                 ) -> torch.Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
#         """
#         batch_size, seq_len, feature_size = x.size()
        
#         x = self.pos_encoder(x)  # BS x T x d_model

#         # Add classification token
#         cls_tokens = self.classification_token.expand(batch_size, -1, -1)

#         x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, seq_len + 1, feature_size]
        
#         # # Add positional encoding and permute for Transformer [seq_len + 1, batch_size, feature_size]
#         # x = self.pos_encoder(x.permute(1, 0, 2))

#         x = self.transformer_encoder(x)  # Pass through the transformer encoder
#         # x = x[0, :, :]  # Extract the classification token output

#         # from Tsironis
#         # x = x[:, 0, :]  # BS x d_model
        
#         x = self.fc(x)  # Final classification layer
#         # return x.squeeze()  # Adjust based on the output needs
#         return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, batch_first=True
            ),
            num_layers=3,
        )

        self.cls_tkn = nn.Parameter(torch.rand(1, 1, d_model), requires_grad=True)

        self.pos_emb = PositionalEncoding(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # BS x T x 128
        x = self.pos_emb(x)  # BS x T x d_model

        cls_tkn = self.cls_tkn.expand(x.shape[0], -1, -1)  # BS x 1 x d_model
        x = torch.cat([cls_tkn, x], dim=1)  # BS x (T+1) x d_model

        x = self.encoder(x)  # BS x (T+1) x d_model

        x = x[:, 0, :]  # BS x d_model
        
        return x

##################################################################################
##################################################################################

class SimpleMLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim: int = 7):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, x):
        x = self.fc(x)
        return x

##################################################################################
##################################################################################