"""
Neural Network Model for QuantumShieldFL

A simple CNN architecture designed for MNIST digit classification.
Kept intentionally lightweight so federated training runs fast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module): 
    """
    Simple Convolutional Neural Network for MNIST classification.
    
    Architecture:
        Conv2d(1, 32, 3) → ReLU → Conv2d(32, 64, 3) → ReLU → MaxPool2d(2)
        → Dropout(0.25) → FC(9216, 128) → ReLU → Dropout(0.5) → FC(128, 10)
    
    This is a standard LeNet-style architecture suitable for federated 
    learning experiments where model simplicity allows focus on the 
    security/communication aspects.
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))            # (B, 32, 28, 28)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 14, 14)
        x = self.pool(x)                      # (B, 64, 7, 7)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)            # (B, 64*7*7) = (B, 3136)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def get_model(num_classes=10, device=None):
    """
    Factory function to create and return a SimpleCNN model.
    
    Args:
        num_classes: Number of output classes (default: 10 for MNIST)
        device: Torch device to place the model on
        
    Returns:
        SimpleCNN model instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)
    return model


def get_model_params(model):
    """Extract model parameters as a list of numpy arrays."""
    return [param.data.cpu().numpy().copy() for param in model.parameters()]


def set_model_params(model, params):
    """Set model parameters from a list of numpy arrays."""
    with torch.no_grad():
        for param, new_val in zip(model.parameters(), params):
            param.copy_(torch.tensor(new_val, dtype=param.dtype, device=param.device))


def serialize_model_params(model):
    """
    Serialize model parameters to bytes for encryption/transmission.
    
    Returns:
        bytes: Pickled and compressed model parameters
    """
    import pickle
    import zlib
    params = get_model_params(model)
    data = pickle.dumps(params)
    compressed = zlib.compress(data)
    return compressed


def deserialize_model_params(data_bytes):
    """
    Deserialize model parameters from bytes.
    
    Args:
        data_bytes: Compressed pickled model parameters
        
    Returns:
        list of numpy arrays: Model parameters
    """
    import pickle
    import zlib
    decompressed = zlib.decompress(data_bytes)
    params = pickle.loads(decompressed)
    return params
