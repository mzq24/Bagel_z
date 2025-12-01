

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class HierarchicalModalityUntiedAttention(torch.nn.Module):
    """
    Two-stage MoT: 
    1. Cross-modal enhancement stage using MoT mechanism
    2. Unified processing stage with enhanced features
    """
    
    def __init__(
        self,
        dim: int,
        head_dim: int, 
        n_heads: int,
        dropout: float,
        norm_eps: float = 1e-5,
        qk_normalization: bool = False,
        n_modalities: int = 2,
    ):
        super().__init__()
        
        self.n_modalities = n_modalities
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dim = dim

        # Stage 1: Cross-modal enhancement MoT
        # 为每个模态创建专门的"询问"其他模态的参数
        self.cross_experts_wq = self._create_experts(dim, n_heads * head_dim)  # 每个模态的query投影
        self.cross_experts_wk = self._create_experts(dim, n_heads * head_dim)  # 跨模态key投影  
        self.cross_experts_wv = self._create_experts(dim, n_heads * head_dim)  # 跨模态value投影
        
        # Stage 2: Unified processing MoT (类似原始MoT)
        self.unified_experts_wq = self._create_experts(dim, n_heads * head_dim)
        self.unified_experts_wk = self._create_experts(dim, n_heads * head_dim) 
        self.unified_experts_wv = self._create_experts(dim, n_heads * head_dim)
        self.unified_experts_wo = self._create_experts(n_heads * head_dim, dim)
        
        # QK normalization (optional)
        if qk_normalization:
            self.cross_experts_q_norm = self._create_norms(head_dim, n_modalities, eps=norm_eps)
            self.cross_experts_k_norm = self._create_norms(head_dim, n_modalities, eps=norm_eps)
            self.unified_experts_q_norm = self._create_norms(head_dim, n_modalities, eps=norm_eps)
            self.unified_experts_k_norm = self._create_norms(head_dim, n_modalities, eps=norm_eps)
            
        # Attention mechanisms
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=n_heads * head_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.unified_attention = torch.nn.MultiheadAttention(
            embed_dim=n_heads * head_dim,
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # new feed-forward network for enhanced features (only map)
        self.map_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.map_ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

        # Normalization layers
        self.cross_output_norm = self._create_norms(dim, n_modalities)
        self.final_output_norm = self._create_norms(dim, n_modalities)
        
    def _create_experts(self, input_dim, output_dim):
        return torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim, bias=False)
            for _ in range(self.n_modalities)
        ])
    
    def _create_norms(self, dim, n_modalities, eps=1e-5):
        return torch.nn.ModuleList([
            torch.nn.LayerNorm(dim, eps=eps) for _ in range(n_modalities)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        modality_masks: list,  # [text_mask, spatial_mask]
    ):
        # Stage 1: Cross-modal Enhancement
        enhanced_features = self._cross_modal_enhancement(x, modality_masks, attn_mask)
        
        # Stage 2: Unified Processing with Enhanced Features
        final_output = self._unified_processing(enhanced_features, modality_masks, attn_mask)
        
        return final_output
    
    def _cross_modal_enhancement(self, x, modality_masks, attn_mask):
        """
        Stage 1: 每个模态主动获取其他模态的信息
        """
        enhanced_outputs = []
        
        for i in range(self.n_modalities):
            # 当前模态作为query
            if modality_masks[i].dim() == 1:  # [seq_len]
                current_tokens = x[:, modality_masks[i], :]
                current_attn_mask = attn_mask[:, modality_masks[i]] if attn_mask is not None else None
            elif modality_masks[i].dim() == 2:  # [batch_size, seq_len]
                current_tokens = self._extract_by_batch_mask(x, modality_masks[i])  # [batch_size, num_valid_tokens, dim]
                current_attn_mask = self._extract_by_batch_mask(attn_mask, modality_masks[i])  # [batch_size, num_valid_tokens]
            #current_modality_tokens = x[modality_masks[i]]
            current_q = self.cross_experts_wq[i](current_tokens)
            
            # 其他模态作为key, value
            other_modalities_k = []
            other_modalities_v = []
            other_attn_masks = []
            
            for j in range(self.n_modalities):
                if j != i:  # 其他模态

                    # other_tokens = x[modality_masks[j]]
                    if modality_masks[j].dim() == 1:
                        other_tokens = x[:, modality_masks[j], :]
                        other_attn_mask = attn_mask[:, modality_masks[j]] if attn_mask is not None else None
                    else:
                        other_tokens = self._extract_by_batch_mask(x, modality_masks[j])
                        other_attn_mask = self._extract_by_batch_mask(attn_mask, modality_masks[j]) if attn_mask is not None else None
                    other_k = self.cross_experts_wk[j](other_tokens)  # 用其他模态的专用投影
                    other_v = self.cross_experts_wv[j](other_tokens)
                    
                    other_modalities_k.append(other_k)
                    other_modalities_v.append(other_v)
                    other_attn_masks.append(other_attn_mask)
            
            # 合并其他模态的K, V
            if other_modalities_k:
                merged_k = torch.cat(other_modalities_k, dim=1)
                merged_v = torch.cat(other_modalities_v, dim=1)
                
                if other_attn_masks[0] is not None:
                    merged_key_mask = torch.cat(other_attn_masks, dim=0)  # [batch_size, total_other_len]
                else:
                    merged_key_mask = None
            
                
                # Apply normalization if enabled
                if hasattr(self, 'cross_experts_q_norm'):
                    current_q = self.cross_experts_q_norm[i](current_q)
                    merged_k = self.cross_experts_k_norm[i](merged_k)  # 用当前模态的norm
                
                batch_attn_mask = self._create_attn_mask_batch(merged_key_mask, current_attn_mask, n_heads=self.n_heads)
                # Cross-modal attention: 当前模态询问其他模态
                cross_modal_info, _ = self.cross_attention(
                    query=current_q,
                    key=merged_k, 
                    value=merged_v,
                    # key_padding_mask=~merged_key_mask,
                    # attn_mask=batch_attn_mask  # 可以设计特殊的cross-modal mask
                )
                if torch.isnan(cross_modal_info).any():
                    raise ValueError(f"NaN detected in cross_modal_info for modality {i}")
                   
                # Residual connection + normalization
                enhanced_current = current_tokens + self.cross_output_norm[i](cross_modal_info)
            else:
                enhanced_current = current_tokens
                
            enhanced_outputs.append(enhanced_current)
            
        return self._merge_enhanced_modalities(enhanced_outputs, modality_masks, x.shape)

    def _extract_by_mask(self, tensor, mask):
        if mask.dim() == 1:  # [seq_len]
            return tensor[:, mask, :]
        elif mask.dim() == 2:  # [batch_size, seq_len]
            return self._extract_by_batch_mask(tensor, mask)

    def _extract_by_batch_mask(self, tensor, batch_mask):
        batch_size = tensor.shape[0]
        extracted_tensors = []
        
        for b in range(batch_size):
            valid_tensor = tensor[b, batch_mask[b], ...]  # [num_valid, ...]
            extracted_tensors.append(valid_tensor)
        
        # Pad到最大长度
        max_len = max(t.shape[0] for t in extracted_tensors)
        if max_len == 0:
            max_len = 1
            
        padded_shape = (batch_size, max_len) + tensor.shape[2:]
        padded_tensor = torch.zeros(padded_shape, device=tensor.device, dtype=tensor.dtype)
        
        for b, t in enumerate(extracted_tensors):
            if t.shape[0] > 0:
                padded_tensor[b, :t.shape[0]] = t
                
        return padded_tensor

    def _create_query_mask(self, query_valid_mask):
        """  create query mask for cross attention """
        query_padding_mask = ~query_valid_mask
        return query_padding_mask

    def _create_attn_mask_batch(self, merged_key_mask, current_attn_mask, n_heads):
        assert merged_key_mask.dim() == 2, "merged_key_mask should be 2D [batch_size, seq_len]"
        assert current_attn_mask.dim() == 2, "current_attn_mask should be 2D [batch_size, seq_len]"
        batch_size, key_len = merged_key_mask.shape
        _, query_len = current_attn_mask.shape
        attention_mask = ~current_attn_mask.unsqueeze(2) | ~merged_key_mask.unsqueeze(1)  # [batch_size, query_len, key_len]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # [batch_size, n_heads, query_len, key_len]
        attention_mask = attention_mask.view(batch_size * n_heads, query_len, key_len)  # [batch_size * n_heads, query_len, key_len]  return attention_mask
        return attention_mask

    def _unified_processing(self, enhanced_x, modality_masks, attn_mask):
        """
        Stage 2: 统一处理增强后的特征，类似原始MoT
        """
        '''
        # Process Q, K, V with modality-specific projections
        expert_outputs_xq, expert_outputs_xk, expert_outputs_xv = [], [], []
        
        for i in range(self.n_modalities):
            
            # expert_input = enhanced_x[modality_masks[i]]
            expert_input = self._extract_by_mask(enhanced_x, modality_masks[i])  # [batch_size, num_valid_tokens, dim]
            xq = self.unified_experts_wq[i](expert_input)
            xk = self.unified_experts_wk[i](expert_input) 
            xv = self.unified_experts_wv[i](expert_input)
            
            if hasattr(self, 'unified_experts_q_norm'):
                xq = self.unified_experts_q_norm[i](xq)
                xk = self.unified_experts_k_norm[i](xk)
                
            expert_outputs_xq.append(xq)
            expert_outputs_xk.append(xk)
            expert_outputs_xv.append(xv)
        
        # Merge all modalities for unified attention
        merged_q = self._merge_modalities(expert_outputs_xq, modality_masks, enhanced_x.shape)
        merged_k = self._merge_modalities(expert_outputs_xk, modality_masks, enhanced_x.shape)
        merged_v = self._merge_modalities(expert_outputs_xv, modality_masks, enhanced_x.shape)
        '''
        # Unified attention computation
        batch_attn_mask = self._create_attn_mask_batch(attn_mask, attn_mask, n_heads=self.n_heads)
        attn_output, _ = self.unified_attention(
            query=enhanced_x,
            key=enhanced_x,
            value=enhanced_x, 
            # attn_mask=batch_attn_mask
        )
        
        if hasattr(self, 'map_ffn'):
            map_mask = modality_masks[1]  # 假设map是第二个模态
            
            if map_mask.dim() == 1:  # [seq_len]
                map_tokens = attn_output[:, map_mask, :]
                enhanced_map = map_tokens + self.map_ffn_norm(self.map_ffn(map_tokens))
                attn_output[:, map_mask, :] = enhanced_map
            else:  # [batch_size, seq_len]
                for b in range(attn_output.shape[0]):
                    map_indices = map_mask[b]
                    map_tokens = attn_output[b, map_indices, :]
                    enhanced_map = map_tokens + self.map_ffn_norm(self.map_ffn(map_tokens))
                    attn_output[b, map_indices, :] = enhanced_map

        return attn_output
    
    def _merge_enhanced_modalities(self, enhanced_outputs, modality_masks, original_shape):
        """合并增强后的模态特征"""
        batch_size, seq_len, dim = original_shape
        device = enhanced_outputs[0].device
        dtype = enhanced_outputs[0].dtype
        merged = torch.zeros(batch_size, seq_len, dim, device=device, dtype=dtype)
        for i, enhanced_output in enumerate(enhanced_outputs):
            for b in range(batch_size):
                merged[b, modality_masks[i][b], :] = enhanced_output[b, :]
            # merged[modality_masks[i]] = enhanced_output
        return merged
        
    def _merge_modalities(self, expert_outputs, modality_masks, shape):
        """合并模态输出用于统一attention"""
        merged = torch.zeros(shape[0], expert_outputs[0].shape[-1])
        for i, expert_output in enumerate(expert_outputs):
            merged[modality_masks[i]] = expert_output
        return merged
        
    def _merge_modalities_final(self, expert_outputs, modality_masks, shape):
        """最终合并输出"""
        merged = torch.zeros_like(torch.empty(shape))
        for i, expert_output in enumerate(expert_outputs):
            merged[modality_masks[i]] = expert_output
        return merged

'''copy from meta
class ModalityUntiedFeedForward(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_modalities = args.n_modalities  # Number of modalities, e.g., 2 (text, image) or 3 (text, image, speech)

        # Initialize feed-forward experts for each modality
        self.local_experts = torch.nn.ModuleList([
            FeedForward(
                dim=args.dim,
                hidden_dim=int(args.ffn_exp * args.dim),
                dropout=args.ffn_dropout,
                ...
            ) for _ in range(self.n_modalities)
        ])

        # Initialize modality-specific normalization layers
        self.local_experts_ffn_norm = torch.nn.ModuleList([
            build_norm_fn(args.norm_type, args.dim, args.norm_eps, args.norm_affine)
            for _ in range(self.n_modalities)
        ])

    def forward(self, x, modality_masks):
        expert_outputs = []

        # Process tokens for each modality separately
        for i in range(self.n_modalities):
            expert_input = x[modality_masks[i]]  # Select tokens for this modality
            expert_output = self.local_experts[i](expert_input)  # Feed-forward processing
            expert_output = self.local_experts_ffn_norm[i](expert_output)  # Normalization
            expert_outputs.append(expert_output)

        # Merge modality-specific outputs into a unified tensor
        merged_output = torch.empty_like(x)
        for i in range(self.n_modalities):
            merged_output[modality_masks[i]] = expert_outputs[i]

        return merged_output
'''

def example_usage():
    """
    使用示例：文本 + 精确空间特征
    """
    batch_size, seq_len, dim = 4, 100, 512
    n_heads, head_dim = 8, 64
    
    # 模拟数据
    x = torch.randn(batch_size, seq_len, dim)
    
    # 模态掩码：前50个token是文本，后50个是空间特征
    text_mask = torch.zeros(seq_len, dtype=torch.bool)
    text_mask[:50] = True
    spatial_mask = torch.zeros(seq_len, dtype=torch.bool) 
    spatial_mask[50:] = True
    modality_masks = [text_mask, spatial_mask]
    
    # 注意力掩码
    attn_mask = torch.zeros(seq_len, seq_len)
    
    # 创建模型
    model = HierarchicalModalityUntiedAttention(
        dim=dim,
        head_dim=head_dim,
        n_heads=n_heads,
        dropout=0.1,
        n_modalities=2
    )
    
    # 前向传播
    output = model(x, attn_mask, modality_masks)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Stage 1: 文本获取精确空间信息，空间获取语义理解")
    print("Stage 2: 增强特征在统一空间中进行复杂交互")

if __name__ == "__main__":
    example_usage()