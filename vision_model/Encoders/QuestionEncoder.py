import math
from typing import Optional, Tuple
import torch

import torch.nn as nn
import torch.nn.functional as F
from tf.keras.layers import Conv3D,ResBlock3D

class QuestionEncoder(nn.Module):
    """Encodes text questions into embedding space"""
    def __init__(self, config):
        super().__init__()
        
        # Use pretrained language model
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        
        # Project to common dimension
        self.projector = nn.Linear(768, config.question_dim)
        
    def forward(self, questions):
        """
        Args:
            questions: tokenized questions
        Returns:
            q: (B, question_dim) question embeddings
        """
        outputs = self.text_encoder(**questions)
        # Use [CLS] token
        q = outputs.last_hidden_state[:, 0, :]
        q = self.projector(q)
        return q