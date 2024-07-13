import random
import copy
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torchmetrics
import matplotlib.pyplot as plt
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
            # valid_pixels = np.ones(self.num_pixels)
        elif S < self.num_pixels:
            x = np.zeros((T, C, self.num_pixels))
            x[..., :S] = pixels
            x[..., S:] = np.stack([x[:, :, 0] for _ in range(S, self.num_pixels)], axis=-1)
            # valid_pixels = np.array([1 for _ in range(S)] + [0 for _ in range(S, self.num_pixels)])
        else:
            x = pixels
            # valid_pixels = np.ones(self.num_pixels)
        # Repeat valid_pixels across time
        # valid_pixels = np.repeat(valid_pixels[np.newaxis].astype(np.float32), x.shape[0], axis=0)
        sample['pixels'] = x
        # sample['valid_pixels'] = valid_pixels
        return sample

##################################################################################
##################################################################################

class Normalize(object):
    """Normalize by rescaling pixels to [0, 1]"""

    def __init__(self, max_pixel_value=65535):
        """max_pixel_value (int): Max value of pixels to move pixels to [0, 1]"""
        self.max_pixel_value = max_pixel_value

    def __call__(self, sample):
        sample['pixels'] = np.clip(sample['pixels'], 0, self.max_pixel_value).astype(np.float32) / self.max_pixel_value
        return sample

##################################################################################
##################################################################################

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        sample['pixels'] = torch.from_numpy(sample['pixels'].astype(np.float32))
        # sample['valid_pixels'] = torch.from_numpy(sample['valid_pixels'].astype(np.float32))
        if isinstance(sample['label'], int):
            sample['label'] = torch.tensor(sample['label']).long()
        return sample

##################################################################################
##################################################################################

class PixelSetData(Dataset):
    """Dataset class for PixelSet data"""
    def __init__(
            self, data_root,
            class_to_idx, true_labels,
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

        # for i in range(4):
        #     print(self.samples[i])
    
    def __getitem__(self, index):
        # get item like timematch
        path, parcel_idx, y = self.samples[index]
        pixels = zarr.load(path)  # (T, C, S)

        sample = {
            "index": index,
            # "parcel_index": parcel_idx,  # mapping to metadata
            "pixels": pixels,
            # "valid_pixels": np.ones(
            #     (pixels.shape[0], pixels.shape[-1]), dtype=np.float32),
            "label": y,
            "label_idx": self.class_to_idx[y],
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return len(self.samples)
    
    def _make_dataset(self):
        """Returns list of tuples (path, parcel_idx, label)"""
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

def split_dict_train_test(data_dict, test_size=0.05, val_size=0.2, random_seed=42):
    """
    Splits a dictionary of data into train, validation, and test sets based on the
    specified test_size, val_size, and random_seed. The dictionary is assumed to
    contain integer keys and values. The function returns three dictionaries,
    where all the label categories are divided so that they correspond to
    the train, validation, and test sizes.
    """
    # Set the random seed for reproducibility
    random.seed(random_seed)
    
    # Group indices by class
    class_to_indices = {}
    for idx, label in data_dict.items():
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    # Split indices into train, validation, and test sets
    train_indices = []
    val_indices = []
    test_indices = []
    for indices in class_to_indices.values():
        train_idx, temp_idx = train_test_split(indices, test_size=test_size + val_size, random_state=random_seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_size / (test_size + val_size), random_state=random_seed)
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
    
    # Create train, validation, and test dictionaries
    train_dict = {idx: data_dict[idx] for idx in train_indices}
    val_dict = {idx: data_dict[idx] for idx in val_indices}
    test_dict = {idx: data_dict[idx] for idx in test_indices}
    
    return train_dict, val_dict, test_dict

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

# from lesson Lab 4
def pad_sequences_collate_fn(samples) -> tuple:
    """
    Zero-pad (in front) each sample to enable batching.
    The longest sequence defines the sequence length for the batch
    """
    labels = torch.tensor([v['label_idx'] for v in samples])

    # variable dimension must be first
    data = pad_sequence([v['pixels'].permute((2, 0, 1)) for v in samples], batch_first=True)

    key_mask = pad_sequence(
        [
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
    out = x.permute((1, 0, 2)) # torch.Size([32, 416, 32])
    out = out * mask # torch.Size([32, 416, 32])
    out = out.sum(dim=-1) / mask.sum(dim=-1) # torch.Size([32, 416])
    out = out.permute((1, 0))
    return out

def masked_std(x, mask):
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
    return out

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

        batch, temp = out.shape[:2]

        # out = out.view(batch * temp, *out.shape[2:]).transpose(1, 2)  # (B*T, S, C)
        out = out.reshape(batch * temp, *out.shape[2:]).transpose(1, 2)  # (B*T, S, C)
        
        # mask = mask.view(batch * temp, -1)
        mask = mask.reshape(batch * temp, -1)

        out = self.mlp1(out).transpose(1, 2)

        out = torch.cat(
            [self.pooling_methods[n](out, mask) for n in self.pooling.split("_")], dim=1
        )

        out = self.mlp2(out)
        out = out.view(batch, temp, -1)
        return out
    
##################################################################################
##################################################################################

# class PositionalEncoding(nn.Module):
#     def __init__(self,
#                  d_model: int,
#                 #  dropout: float = 0.1,
#                 #  max_len: int = 5000
#                  max_len: int = 52
#                  ):
#         super(PositionalEncoding, self).__init__()
        
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
#             )
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
        
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
    
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return x

# from lesson lab 4
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 52):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        pe = self.pe[:, : x.size(1)].expand(x.shape[0], -1, -1)

        x = x + pe
        return self.dropout(x)

##################################################################################
##################################################################################

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

def trainer_function(model, num_classes, train_dloader, val_dloader, device, num_epochs=9):
    """Train the model: Returns train and validation last losses and accuracies"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Initialize metrics
    f1_micro = torchmetrics.F1Score(task="multiclass", average='micro', num_classes=num_classes).to(device)
    f1_weighted = torchmetrics.F1Score(task="multiclass", average='weighted', num_classes=num_classes).to(device)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    precision = torchmetrics.Precision(task="multiclass", average='micro', num_classes=num_classes).to(device)
    recall = torchmetrics.Recall(task="multiclass", average='micro', num_classes=num_classes).to(device)
    confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    train_f1_micros = []
    train_f1_weighteds = []
    train_precisions = []
    train_recalls = []

    val_losses = []
    val_accuracies = []
    val_f1_micros = []
    val_f1_weighteds = []
    val_precisions = []
    val_recalls = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train(True)

        running_loss = 0.0
        running_corrects = 0

        # Reset metrics
        f1_micro.reset()
        f1_weighted.reset()
        accuracy.reset()
        precision.reset()
        recall.reset()
        confusion_matrix.reset()

        for batch in train_dloader:
            input_tensor = batch[0].to(device)
            labels = batch[1].to(device)
            mask = batch[2].to(device)

            out = model(input_tensor, mask)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_tensor.size(0)
            _, preds = torch.max(out, 1)
            running_corrects += torch.sum(preds == labels.data)

            # Update metrics
            f1_micro.update(preds, labels)
            f1_weighted.update(preds, labels)
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            confusion_matrix.update(preds, labels)

        epoch_loss = running_loss / len(train_dloader.dataset)
        epoch_acc = accuracy.compute().item()
        epoch_f1_micro = f1_micro.compute().item()
        epoch_f1_weighted = f1_weighted.compute().item()

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'F1 Micro: {epoch_f1_micro:.4f} F1 Weighted: {epoch_f1_weighted:.4f}')
        print(f'Precision: {precision.compute().item():.4f} Recall: {recall.compute().item():.4f}')
        print(f'Confusion Matrix:\n{confusion_matrix.compute()}')

        # Store metrics for plotting
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        train_f1_micros.append(epoch_f1_micro)
        train_f1_weighteds.append(epoch_f1_weighted)
        train_precisions.append(precision.compute().item())
        train_recalls.append(recall.compute().item())

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        # Reset metrics
        f1_micro.reset()
        f1_weighted.reset()
        accuracy.reset()
        precision.reset()
        recall.reset()
        confusion_matrix.reset()

        with torch.no_grad():
            for batch in val_dloader:
                input_tensor = batch[0].to(device)
                labels = batch[1].to(device)
                mask = batch[2].to(device)

                out = model(input_tensor, mask)
                
                loss = criterion(out, labels)

                val_running_loss += loss.item() * input_tensor.size(0)
                _, preds = torch.max(out, 1)
                val_running_corrects += torch.sum(preds == labels.data)

                # Update metrics
                f1_micro.update(preds, labels)
                f1_weighted.update(preds, labels)
                accuracy.update(preds, labels)
                precision.update(preds, labels)
                recall.update(preds, labels)
                confusion_matrix.update(preds, labels)

        val_epoch_loss = val_running_loss / len(val_dloader.dataset)
        val_epoch_acc = accuracy.compute().item()
        val_epoch_f1_micro = f1_micro.compute().item()
        val_epoch_f1_weighted = f1_weighted.compute().item()

        print(f'Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        print(f'Validation F1 Micro: {val_epoch_f1_micro:.4f} F1 Weighted: {val_epoch_f1_weighted:.4f}')
        print(f'Validation Precision: {precision.compute().item():.4f} Recall: {recall.compute().item():.4f}')
        print(f'Validation Confusion Matrix:\n{confusion_matrix.compute()}')

        # Store metrics for plotting
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        val_f1_micros.append(val_epoch_f1_micro)
        val_f1_weighteds.append(val_epoch_f1_weighted)
        val_precisions.append(precision.compute().item())
        val_recalls.append(recall.compute().item())

    # Plot metrics
    epochs = range(num_epochs)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_f1_micros, label='Training F1 Micro')
    plt.plot(epochs, val_f1_micros, label='Validation F1 Micro')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Micro')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_f1_weighteds, label='Training F1 Weighted')
    plt.plot(epochs, val_f1_weighteds, label='Validation F1 Weighted')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Weighted')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return train_losses[-1], train_accuracies[-1], train_f1_micros[-1], train_f1_weighteds[-1],\
        val_losses[-1], val_accuracies[-1], val_f1_micros[-1], val_f1_weighteds[-1],\
        train_precisions[-1], train_recalls[-1], val_precisions[-1], val_recalls[-1]
        

##################################################################################
##################################################################################