class RefinementModule(nn.Module):
    """
    Goes back to MRI and updates representation based on:
    - Current representation R_t
    - Question q
    - Previous answer attempt
    """
    def __init__(self, config):
        super().__init__()
        
        # Attention mechanism: what to look at in MRI?
        self.attention = nn.Sequential(
            nn.Linear(config.repr_dim + config.question_dim + config.num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),  # Attention query
        )
        
        # Spatial attention over MRI features
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature extraction from attended regions
        self.focused_extractor = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
        )
        
        # GRU for recurrent update
        self.gru = nn.GRUCell(
            input_size=256 * 4 * 4 * 4,  # Focused features
            hidden_size=config.repr_dim
        )
        
    def forward(self, R_t, mri_features, q, prev_logits):
        """
        Args:
            R_t: (B, repr_dim) - Current representation
            mri_features: (B, 256, d, h, w) - MRI features from encoder
            q: (B, question_dim) - Question
            prev_logits: (B, num_classes) - Previous answer attempt
        Returns:
            R_{t+1}: (B, repr_dim) - Updated representation
        """
        B = R_t.shape[0]
        
        # Create attention query from current state
        query_input = torch.cat([R_t, q, prev_logits], dim=1)
        query = self.attention(query_input)  # (B, 256)
        
        # Compute spatial attention weights
        # "Where should I look in the MRI?"
        query_expanded = query.view(B, 256, 1, 1, 1)
        attention_weights = self.spatial_attention(mri_features)  # (B, 1, d, h, w)
        
        # Apply attention to features
        attended_features = mri_features * attention_weights  # (B, 256, d, h, w)
        
        # Extract focused features
        focused = self.focused_extractor(attended_features)  # (B, 256, 4, 4, 4)
        focused = focused.flatten(1)  # (B, 256*64)
        
        # Update representation via GRU
        R_next = self.gru(focused, R_t)
        
        return R_next