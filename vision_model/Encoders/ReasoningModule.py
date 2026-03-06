import math
from typing import Optional, Tuple
import torch

import torch.nn as nn
import torch.nn.functional as F
from tf.keras.layers import Conv3D,ResBlock3D

class ReasoningModule(nn.Module):
    """
    Attempts to answer question from current representation
    Outputs answer + confidence score
    """
    def __init__(self, config):
        super().__init__()
        
        # Fuse representation and question
        self.fusion = nn.Sequential(
            nn.Linear(config.repr_dim + config.question_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Transformer for reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Answer head
        self.answer_head = nn.Linear(1024, config.num_classes)
        
        # Confidence head (learns when to refine)
        self.confidence_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Confidence in [0, 1]
        )
        
    def forward(self, R, q):
        """
        Args:
            R: (B, repr_dim) - Current representation
            q: (B, question_dim) - Question embedding
        Returns:
            logits: (B, num_classes)
            confidence: (B, 1)
        """
        # Fuse representation and question
        fused = torch.cat([R, q], dim=1)
        fused = self.fusion(fused)  # (B, 1024)
        
        # Add sequence dimension for transformer
        fused = fused.unsqueeze(0)  # (1, B, 1024)
        
        # Reasoning via transformer
        reasoned = self.transformer(fused)  # (1, B, 1024)
        reasoned = reasoned.squeeze(0)  # (B, 1024)
        
        # Generate answer and confidence
        logits = self.answer_head(reasoned)
        confidence = self.confidence_head(reasoned)
        
        return logits, confidence