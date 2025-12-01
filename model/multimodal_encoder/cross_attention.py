"""
Cross Attention module for fusing language and map tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class LanguageMapCrossAttention(nn.Module):
    """
    Cross attention module to fuse language tokens with map tokens
    """
    
    def __init__(
        self,
        language_dim: int = 4096,    # Language model hidden dimension (e.g., LLaMA)
        map_dim: int = 768,          # Map encoder output dimension  
        hidden_dim: int = 768,       # Cross attention hidden dimension
        num_heads: int = 8,         # Number of attention heads
        dropout: float = 0.1,
        max_language_len: int = 2048,
        max_map_len: int = 1024,
    ):
        super().__init__()
        
        self.language_dim = language_dim
        self.map_dim = map_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Project language and map features to common dimension
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.map_proj = nn.Linear(map_dim, hidden_dim)
        
        # Cross attention from language to map (language as query, map as key/value)
        self.lang_to_map_attn = CrossAttentionLayer(
            query_dim=hidden_dim,
            kv_dim=hidden_dim, 
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross attention from map to language (map as query, language as key/value)  
        self.map_to_lang_attn = CrossAttentionLayer(
            query_dim=hidden_dim,
            kv_dim=hidden_dim,
            hidden_dim=hidden_dim, 
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Self attention for joint reasoning
        self.joint_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed forward networks
        self.language_ffn = FeedForward(hidden_dim, hidden_dim * 4, dropout)
        self.map_ffn = FeedForward(hidden_dim, hidden_dim * 4, dropout)
        
        # Layer normalization
        self.lang_norm1 = nn.LayerNorm(hidden_dim)
        self.lang_norm2 = nn.LayerNorm(hidden_dim)
        self.lang_norm3 = nn.LayerNorm(hidden_dim)
        
        self.map_norm1 = nn.LayerNorm(hidden_dim)
        self.map_norm2 = nn.LayerNorm(hidden_dim)
        self.map_norm3 = nn.LayerNorm(hidden_dim)
        
        # Output projections back to original dimensions
        self.language_output_proj = nn.Linear(hidden_dim, language_dim)
        # self.map_output_proj = nn.Linear(hidden_dim, map_dim)
        self.map_output_proj = nn.Linear(hidden_dim, language_dim)

        
        # Learnable tokens for enhanced fusion
        self.fusion_tokens = nn.Parameter(torch.randn(8, hidden_dim))  # 8 fusion tokens
        
    def forward(
        self, 
        language_features: torch.Tensor,  # [B, L, language_dim]
        map_features: torch.Tensor,       # [B, M, map_dim]  
        language_mask: Optional[torch.Tensor] = None,  # [B, L]
        map_mask: Optional[torch.Tensor] = None,       # [B, M]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            language_features: [B, L, language_dim] language token embeddings
            map_features: [B, M, map_dim] map token embeddings
            language_mask: [B, L] mask for language tokens (1 for valid, 0 for padding)
            map_mask: [B, M] mask for map tokens (1 for valid, 0 for padding)
            
        Returns:
            enhanced_language: [B, L, language_dim] language features enhanced with map info
            enhanced_map: [B, M, map_dim] map features enhanced with language info
        """
        B, L, _ = language_features.shape
        B, M, _ = map_features.shape
        
        # Project to common dimension
        lang_proj = self.language_proj(language_features)  # [B, L, hidden_dim]
        map_proj = self.map_proj(map_features)            # [B, M, hidden_dim]
        
        # Cross attention: language attending to map
        lang_attended = self.lang_to_map_attn(
            query=lang_proj,
            key=map_proj, 
            value=map_proj,
            key_mask=map_mask
        )  # [B, L, hidden_dim]
        
        # Residual connection and normalization
        lang_proj = self.lang_norm1(lang_proj + lang_attended)
        
        # Cross attention: map attending to language  
        map_attended = self.map_to_lang_attn(
            query=map_proj,
            key=lang_proj,
            value=lang_proj, 
            key_mask=language_mask
        )  # [B, M, hidden_dim]
        
        # Residual connection and normalization
        map_proj = self.map_norm1(map_proj + map_attended)
        
        # Joint self-attention with fusion tokens
        fusion_tokens = self.fusion_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 8, hidden_dim]
        joint_features = torch.cat([fusion_tokens, lang_proj, map_proj], dim=1)  # [B, 8+L+M, hidden_dim]
        
        # Create attention mask for joint features
        joint_mask = None
        if language_mask is not None or map_mask is not None:
            fusion_mask = torch.ones(B, 8, device=joint_features.device, dtype=torch.bool)
            lang_mask_extended = language_mask if language_mask is not None else torch.ones(B, L, device=joint_features.device, dtype=torch.bool)
            map_mask_extended = map_mask if map_mask is not None else torch.ones(B, M, device=joint_features.device, dtype=torch.bool)
            joint_mask = torch.cat([fusion_mask, lang_mask_extended, map_mask_extended], dim=1)  # [B, 8+L+M]
            joint_mask = ~joint_mask  # Convert to attention mask (True for positions to mask)
        
        # Self attention on joint features
        joint_attended, _ = self.joint_self_attn(
            joint_features, joint_features, joint_features, 
            key_padding_mask=joint_mask
        )  # [B, 8+L+M, hidden_dim]
        
        # Split back into components
        fusion_attended = joint_attended[:, :8, :]          # [B, 8, hidden_dim]
        lang_attended = joint_attended[:, 8:8+L, :]         # [B, L, hidden_dim] 
        map_attended = joint_attended[:, 8+L:, :]           # [B, M, hidden_dim]
        
        # Residual connections and normalization
        lang_proj = self.lang_norm2(lang_proj + lang_attended)
        map_proj = self.map_norm2(map_proj + map_attended)
        
        # Feed forward networks
        lang_proj = self.lang_norm3(lang_proj + self.language_ffn(lang_proj))
        map_proj = self.map_norm3(map_proj + self.map_ffn(map_proj))
        
        # Project back to original dimensions
        enhanced_language = self.language_output_proj(lang_proj)  # [B, L, language_dim]
        enhanced_map = self.map_output_proj(map_proj)            # [B, M, map_dim]
        
        return enhanced_language, enhanced_map


class CrossAttentionLayer(nn.Module):
    """Single cross attention layer"""
    
    def __init__(
        self,
        query_dim: int,
        kv_dim: int, 
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(kv_dim, hidden_dim)
        self.v_proj = nn.Linear(kv_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,     # [B, L_q, query_dim]
        key: torch.Tensor,       # [B, L_kv, kv_dim]
        value: torch.Tensor,     # [B, L_kv, kv_dim]
        key_mask: Optional[torch.Tensor] = None  # [B, L_kv]
    ) -> torch.Tensor:
        """
        Returns:
            attended: [B, L_q, hidden_dim]
        """
        B, L_q, _ = query.shape
        B, L_kv, _ = key.shape
        
        # Linear projections
        q = self.q_proj(query)    # [B, L_q, hidden_dim]
        k = self.k_proj(key)      # [B, L_kv, hidden_dim] 
        v = self.v_proj(value)    # [B, L_kv, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.reshape(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)     # [B, num_heads, L_q, head_dim]
        k = k.reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)    # [B, num_heads, L_kv, head_dim]
        v = v.reshape(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)    # [B, num_heads, L_kv, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, L_q, L_kv]
        
        # Apply key mask if provided
        if key_mask is not None:
            mask = key_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L_kv]
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, L_q, L_kv]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, v)  # [B, num_heads, L_q, head_dim]
        
        # Reshape and project output
        attended = attended.transpose(1, 2).reshape(B, L_q, self.hidden_dim)  # [B, L_q, hidden_dim]
        attended = self.out_proj(attended)
        
        return attended


class FeedForward(nn.Module):
    """Feed forward network with GELU activation"""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MapTokenProcessor(nn.Module):
    """
    Processes raw map data into tokens suitable for cross attention
    """
    
    def __init__(
        self,
        max_map_tokens: int = 512,
        map_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.max_map_tokens = max_map_tokens
        self.map_dim = map_dim
        
        # Token type embeddings for different map elements
        self.lane_type_embedding = nn.Embedding(10, map_dim // 4)  # Lane types
        self.road_type_embedding = nn.Embedding(5, map_dim // 4)   # Road types
        
        # Pooling for reducing sequence length if needed
        self.adaptive_pool = nn.AdaptiveAvgPool1d(max_map_tokens)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        map_embeddings: torch.Tensor,     # [B, N, map_dim] from GNN encoder
        map_metadata: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            map_embeddings: [B, N, map_dim] map token embeddings from GNN
            map_metadata: Optional metadata about map tokens
            
        Returns:
            processed_tokens: [B, max_map_tokens, map_dim] processed map tokens
            map_mask: [B, max_map_tokens] mask for valid tokens
        """
        B, N, D = map_embeddings.shape
        
        # Adaptive pooling to limit sequence length
        if N > self.max_map_tokens:
            # Transpose for adaptive pooling: [B, D, N] -> [B, D, max_map_tokens]
            pooled = self.adaptive_pool(map_embeddings.transpose(1, 2)).transpose(1, 2)
            map_mask = torch.ones(B, self.max_map_tokens, device=map_embeddings.device, dtype=torch.bool)
        else:
            # Pad if necessary
            pad_length = self.max_map_tokens - N
            pooled = F.pad(map_embeddings, (0, 0, 0, pad_length), value=0)
            map_mask = torch.zeros(B, self.max_map_tokens, device=map_embeddings.device, dtype=torch.bool)
            map_mask[:, :N] = True
        
        processed_tokens = self.dropout(pooled)
        
        return processed_tokens, map_mask
