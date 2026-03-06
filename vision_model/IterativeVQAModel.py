import math
from typing import Optional, Tuple
import torch

import torch.nn as nn
import torch.nn.functional as F

from vision_model.Encoders.AdaptiveInitialEncoder import AdaptiveInitialEncoder
from vision_model.Encoders.QuestionEncoder import QuestionEncoder
from vision_model.Encoders.ReasoningModule import ReasoningModule
from vision_model.Encoders.RefinementModule import RefinementModule

class IterativeVQAModel(nn.Module):
    """
    Complete iterative VQA model with adaptive representation learning
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Components
        self.initial_encoder = AdaptiveInitialEncoder(config)
        self.question_encoder = QuestionEncoder(config)
        self.reasoning = ReasoningModule(config)
        self.refinement = RefinementModule(config)
        
    def forward(self, mri, questions, training=True):
        """
        Args:
            mri: (B, 1, D, H, W) - MRI volumes
            questions: tokenized questions
            training: whether in training mode
        Returns:
            outputs: dict with predictions, confidences, iterations
        """
        B = mri.shape[0]
        
        # Encode question
        q = self.question_encoder(questions)  # (B, question_dim)
        
        # Initial representation (THIS IS WHAT IMPROVES OVER TIME)
        R_0, mri_features = self.initial_encoder(mri)
        
        # Storage for iteration history
        all_logits = []
        all_confidences = []
        all_representations = [R_0]
        
        # Current state
        R_current = R_0
        
        # Iterative refinement
        max_iters = self.config.max_iterations if training else 5
        
        for t in range(max_iters):
            # Try to answer from current representation
            logits, confidence = self.reasoning(R_current, q)
            
            all_logits.append(logits)
            all_confidences.append(confidence)
            
            # Check if confident enough to stop
            if not training:
                # At inference, stop if confident
                if (confidence > self.config.confidence_threshold).all():
                    break
            
            # Refine representation if not last iteration
            if t < max_iters - 1:
                R_current = self.refinement(
                    R_current, 
                    mri_features, 
                    q, 
                    logits.detach()  # Don't backprop through previous answer
                )
                all_representations.append(R_current)
        
        return {
            'all_logits': all_logits,  # List of predictions at each iteration
            'all_confidences': all_confidences,  # Confidence scores
            'all_representations': all_representations,  # R_0, R_1, ...
            'final_logits': all_logits[-1],
            'final_confidence': all_confidences[-1],
            'num_iterations': len(all_logits),
        }