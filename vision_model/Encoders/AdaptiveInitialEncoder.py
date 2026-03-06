import math
from typing import Optional, Tuple
import torch

import torch.nn as nn
import torch.nn.functional as F
from tf.keras.layers import Conv3D,ResBlock3D

class AdaptiveInitialEncoder(nn.Module):
    """
    Learns to encode MRI into a representation that can answer
    most questions without refinement
    """
    def __init__(self, config):
        super().__init__()
        
        # 3D CNN backbone for MRI
        self.feature_extractor = nn.Sequential(
            Conv3D(1, 64, kernel_size=3, stride=2),
            ResBlock3D(64, 64),
            ResBlock3D(64, 128),
            ResBlock3D(128, 128),
            ResBlock3D(128, 256),
        )
        
        # Multi-scale feature aggregation
        # Captures different levels of detail
        self.scale_aggregators = nn.ModuleList([
            nn.AdaptiveAvgPool3d((32, 32, 32)),  # Fine details
            nn.AdaptiveAvgPool3d((16, 16, 16)),  # Medium structures
            nn.AdaptiveAvgPool3d((8, 8, 8)),     # Coarse context
        ])
        
        # Learnable importance weighting
        # Model learns what's generally important across questions
        self.importance_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid()  # Importance weights
        )
        
        # Compress to compact representation
        self.compressor = nn.Sequential(
            nn.Linear(256 * (32**3 + 16**3 + 8**3), 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, config.repr_dim),  # e.g., 512 or 1024
        )
        
    def forward(self, mri):
        """
        Args:
            mri: (B, 1, D, H, W) - 3D MRI volume
        Returns:
            R_0: (B, repr_dim) - Initial representation
        """
        # Extract features
        features = self.feature_extractor(mri)  # (B, 256, d, h, w)
        
        # Multi-scale aggregation
        multi_scale = []
        for aggregator in self.scale_aggregators:
            scale_features = aggregator(features)
            multi_scale.append(scale_features.flatten(2))  # (B, 256, spatial)
        
        # Concatenate scales
        all_features = torch.cat(multi_scale, dim=2)  # (B, 256, total_spatial)
        
        # Global average pooling
        global_features = all_features.mean(dim=2)  # (B, 256)
        
        # Learn importance
        importance = self.importance_net(global_features)  # (B, 256)
        
        # Weight features by importance
        weighted_features = all_features * importance.unsqueeze(2)
        
        # Compress to compact representation
        flat_features = weighted_features.flatten(1)
        R_0 = self.compressor(flat_features)
        
        return R_0, features  # Return both for potential refinement