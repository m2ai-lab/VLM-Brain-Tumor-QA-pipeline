import math
from typing import Optional, Tuple
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from vision_model.IterativeVQAModel import IterativeVQAModel



@dataclass
class VQAConfig:
    # Architecture
    repr_dim: int = 1024  # Size of representation
    question_dim: int = 512
    num_classes: int = 100  # Number of possible answers
    
    # Training
    max_iterations: int = 5  # Will be reduced by curriculum
    confidence_threshold: float = 0.7  # Will be increased by curriculum
    
    # Learning rates
    encoder_lr: float = 1e-4  # Higher - we want encoder to improve
    other_lr: float = 5e-5
    
    # Loss weights (will change over training)
    iteration_penalty_schedule: str = "progressive"

class VQATrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Separate optimizers for different learning rates
        self.encoder_optimizer = torch.optim.AdamW(
            model.initial_encoder.parameters(),
            lr=config.encoder_lr  # Higher LR - we want this to improve fast
        )
        
        self.other_optimizer = torch.optim.AdamW(
            list(model.reasoning.parameters()) + 
            list(model.refinement.parameters()) +
            list(model.question_encoder.parameters()),
            lr=config.other_lr
        )
        
    def compute_loss(self, outputs, targets, epoch):
        """
        Loss function that encourages:
        1. Correct answers
        2. High confidence when correct
        3. Fewer iterations over time
        """
        all_logits = outputs['all_logits']
        all_confidences = outputs['all_confidences']
        num_iters = outputs['num_iterations']
        
        # Answer loss for each iteration
        answer_losses = []
        for logits in all_logits:
            loss = F.cross_entropy(logits, targets)
            answer_losses.append(loss)
        
        # Final answer should be correct
        final_answer_loss = answer_losses[-1]
        
        # Intermediate answers should also be reasonable
        intermediate_loss = sum(answer_losses[:-1]) / max(len(answer_losses) - 1, 1)
        
        # Confidence calibration
        # High confidence when correct, low when wrong
        predictions = torch.argmax(all_logits[-1], dim=1)
        correct = (predictions == targets).float()
        
        # Confidence should match correctness
        confidence_target = correct.unsqueeze(1)
        confidence_loss = F.binary_cross_entropy(
            all_confidences[-1], 
            confidence_target
        )
        
        # Iteration penalty (increases over training)
        # Early training: λ small, many iterations OK
        # Late training: λ large, penalize iterations heavily
        iteration_penalty_weight = self.get_iteration_penalty(epoch)
        iteration_penalty = (num_iters - 1) * iteration_penalty_weight
        
        # Representation quality loss
        # Encourage R_0 to be informative
        repr_loss = self.representation_quality_loss(
            outputs['all_representations'][0],  # R_0
            targets
        )
        
        # Total loss
        total_loss = (
            final_answer_loss +
            0.3 * intermediate_loss +
            0.2 * confidence_loss +
            iteration_penalty +
            0.1 * repr_loss
        )
        
        return {
            'total_loss': total_loss,
            'answer_loss': final_answer_loss,
            'confidence_loss': confidence_loss,
            'iteration_penalty': iteration_penalty,
            'num_iterations': num_iters,
        }
    
    def get_iteration_penalty(self, epoch):
        """
        Progressive penalty on iterations
        Epoch 0-50: λ = 0.01 (small penalty, learn to get correct)
        Epoch 50-150: λ grows to 0.5 (moderate penalty)
        Epoch 150+: λ = 1.0 (heavy penalty, force good R_0)
        """
        if epoch < 50:
            return 0.01
        elif epoch < 150:
            # Linear growth
            return 0.01 + (epoch - 50) / 100 * 0.49
        else:
            return 0.5
    
    def representation_quality_loss(self, R_0, targets):
        """
        Encourage R_0 to be maximally informative
        Using information bottleneck principle
        """
        # Simple version: encourage diversity in representations
        # More complex version: mutual information estimation
        
        # Encourage high variance (informative)
        variance = R_0.var(dim=0).mean()
        variance_loss = -torch.log(variance + 1e-8)
        
        return variance_loss
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        
        metrics = {
            'total_loss': [],
            'answer_loss': [],
            'num_iterations': [],
            'first_attempt_accuracy': [],
        }
        
        for batch in train_loader:
            mri, questions, answers = batch
            
            # Forward pass
            outputs = self.model(mri, questions, training=True)
            
            # Compute loss
            loss_dict = self.compute_loss(outputs, answers, epoch)
            
            # Backward pass
            self.encoder_optimizer.zero_grad()
            self.other_optimizer.zero_grad()
            
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.encoder_optimizer.step()
            self.other_optimizer.step()
            
            # Track metrics
            metrics['total_loss'].append(loss_dict['total_loss'].item())
            metrics['answer_loss'].append(loss_dict['answer_loss'].item())
            metrics['num_iterations'].append(loss_dict['num_iterations'])
            
            # First attempt accuracy (from R_0 alone)
            first_preds = torch.argmax(outputs['all_logits'][0], dim=1)
            first_acc = (first_preds == answers).float().mean()
            metrics['first_attempt_accuracy'].append(first_acc.item())
        
        # Aggregate metrics
        return {k: np.mean(v) for k, v in metrics.items()}

class CurriculumScheduler:
    """
    Progressively make training harder to force better R_0
    """
    def __init__(self, config):
        self.config = config
        
    def get_max_iterations(self, epoch):
        """
        Reduce allowed iterations over time
        Epoch 0-50: 5 iterations allowed
        Epoch 50-100: 3 iterations
        Epoch 100+: 2 iterations (force good R_0!)
        """
        if epoch < 50:
            return 5
        elif epoch < 100:
            return 3
        else:
            return 2
    
    def get_confidence_threshold(self, epoch):
        """
        Increase required confidence over time
        """
        if epoch < 50:
            return 0.5  # Low bar initially
        elif epoch < 100:
            return 0.7
        else:
            return 0.85  # High confidence required

def main():
    config = VQAConfig()
    model = IterativeVQAModel(config)
    trainer = VQATrainer(model, config)
    curriculum = CurriculumScheduler(config)
    
    for epoch in range(200):
        # Update curriculum
        model.config.max_iterations = curriculum.get_max_iterations(epoch)
        model.config.confidence_threshold = curriculum.get_confidence_threshold(epoch)
        
        # Train
        metrics = trainer.train_epoch(train_loader, epoch)
        
        print(f"Epoch {epoch}")
        print(f"  Answer Loss: {metrics['answer_loss']:.4f}")
        print(f"  Avg Iterations: {metrics['num_iterations']:.2f}")
        print(f"  First Attempt Acc: {metrics['first_attempt_accuracy']:.2%}")
        print(f"  Iteration Penalty Weight: {trainer.get_iteration_penalty(epoch):.3f}")
        
        # Evaluate
        if epoch % 10 == 0:
            eval_metrics = evaluate(model, val_loader)
            print(f"  Val Accuracy: {eval_metrics['accuracy']:.2%}")
            print(f"  Val Avg Iterations: {eval_metrics['avg_iterations']:.2f}")