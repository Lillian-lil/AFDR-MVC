import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import scipy.io as scio
import os

def Get_dataloaders(batch_size=128, path_to_data='./new_datasets/', DATANAME='MNIST-Sobel.mat', val_ratio=0.1):
    """dataloader with (32, 32) images."""
    # Ensure the data path exists
    if not os.path.exists(path_to_data):
        os.makedirs(path_to_data)
        print(f"Created data directory: {path_to_data}")
    
    data_path = os.path.join(path_to_data, DATANAME)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    DATA = scio.loadmat(data_path)
    
    # Automatically detect number of views
    view_keys = [key for key in DATA.keys() if key.startswith('X') and key[1:].isdigit()]
    view_num = len(view_keys)
    print(f"Detected {view_num} views")
    
    # Load all view data (no normalization applied)
    views = []
    for i in range(1, view_num+1):
        view_key = f'X{i}'
        view_data = DATA[view_key]
        print(f"{view_key} shape: {view_data.shape}")
        # Directly convert to torch tensor without normalization
        views.append(torch.from_numpy(view_data.astype(np.float32)).float())
    
    # Load labels
    y = DATA['Y']
    size = y.shape[1] if len(y.shape) > 1 else y.shape[0]
    print(f"Number of samples: {size}")
    
    # Ensure labels are 1D
    if len(y.shape) > 1:
        y = y.flatten()
    
    cluster = np.unique(y)
    print(f'Number of clusters K: {len(cluster)}')
    y = torch.from_numpy(y)
    
    # Add mask tensors (all ones indicating all views are present)
    masks = []
    for i in range(view_num):
        mask = torch.ones(views[0].size(0), dtype=torch.float32)
        masks.append(mask)
    
    # Build dataset
    dataset_items = views + [y] + masks
    X = TensorDataset(*dataset_items)
    
    # Split into train and validation sets
    dataset_size = len(X)
    indices = list(range(dataset_size))
    split = int(np.floor(val_ratio * dataset_size))
    
    if split > 0:
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_dataset = torch.utils.data.Subset(X, train_indices)
        val_dataset = torch.utils.data.Subset(X, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # normalization_methods is no longer used, return empty list for compatibility
        return train_loader, val_loader, view_num, len(cluster), size, []
    else:
        # If no validation set is needed
        train_loader = DataLoader(X, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        return train_loader, None, view_num, len(cluster), size, []