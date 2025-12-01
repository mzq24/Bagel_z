"""
Extended LLaVA architecture with Map GNN integration
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
from abc import ABC, abstractmethod

from .llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .multimodal_encoder.map_gnn_encoder import SMARTMapDecoder
from .multimodal_encoder.cross_attention import LanguageMapCrossAttention, MapTokenProcessor
#from .multimodal_encoder.builder import build_vision_tower
#from .multimodal_resampler.builder import build_vision_resampler
#from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import PretrainedConfig
from llava.utils import rank0_print
import pickle

# Define new constants for map tokens
DEFAULT_MAP_TOKEN = "<map>"
DEFAULT_MAP_PATCH_TOKEN = "<map_patch>"
DEFAULT_MAP_START_TOKEN = "<map_start>"
DEFAULT_MAP_END_TOKEN = "<map_end>"
MAP_TOKEN_INDEX = -201


class LlavaMapMetaModel(LlavaMetaModel):
    """Extended LLaVA model with map token support"""
    
    def __init__(self, config):
        super(LlavaMapMetaModel, self).__init__(config)
        
        # Initialize map-related components if specified in config
        if hasattr(config, "mm_map_encoder"):
            self.map_encoder = self._build_map_encoder(config)
            map_hidden_dim = getattr(config, "map_hidden_dim", 128)
            language_hidden_dim = getattr(config, "hidden_size", 896)
            
            self.map_projector = nn.Linear(map_hidden_dim, language_hidden_dim)
            # Map newline token for sequence formatting
            if getattr(config, "use_map_newline", False):
                self.map_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
    
    def _build_map_encoder(self, config) -> SMARTMapDecoder:
        """Build map GNN encoder from config"""
        compute_dtype = getattr(config, "map_token_traj_path", None)
        #if getattr(config, "map_token_traj_path", None) is not None:
        #    map_token = pickle.load(open(config.map_token_traj_path, "rb"))
        #    map_token['traj_src'] = torch.from_numpy(map_token['traj_src']).to(torch.bfloat16)
        #    config.map_token = map_token

        map_encoder = SMARTMapDecoder(
            dataset=getattr(config, "dataset", "waymo"),
            input_dim=getattr(config, "map_input_dim", 2),
            hidden_dim=getattr(config, "map_hidden_dim", 128),
            num_historical_steps=getattr(config, "num_historical_steps", 11),
            pl2pl_radius=getattr(config, "map_gnn_radius", 10.0),
            num_freq_bands=getattr(config, "num_freq_bands", 64),
            num_layers=getattr(config, "map_num_layers", 3),
            num_heads=getattr(config, "map_num_heads", 8),
            head_dim=getattr(config, "map_head_dim", 16),
            dropout=getattr(config, "map_dropout", 0.1),
            map_token=getattr(config, "map_token", None),
            map_token_traj_path=getattr(config, "map_token_traj_path", None)
        )
        map_encoder_ckpt_path = getattr(config, "map_encoder_ckpt_path", None)
        if map_encoder_ckpt_path:
            self._load_map_encoder_weights(map_encoder, map_encoder_ckpt_path)
        return map_encoder
    
    def _load_map_encoder_weights(self, map_encoder: SMARTMapDecoder, ckpt_path: str):
        """Load pretrained weights for map encoder"""
        import torch
        
        rank0_print(f"Loading map encoder weights from {ckpt_path}")
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Extract state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter map encoder weights
        map_encoder_state_dict = {}
        map_encoder_prefix = "encoder.map_encoder."  # 根据您的实际checkpoint结构调整
        
        for key, value in state_dict.items():
            if key.startswith(map_encoder_prefix):
                # Remove prefix to match map_encoder module
                new_key = key[len(map_encoder_prefix):]
                map_encoder_state_dict[new_key] = value
            elif "map" in key.lower() and "agent" not in key.lower():
                # 更灵活的匹配方式，根据您的实际情况调整
                map_encoder_state_dict[key] = value
        
        # Load filtered weights
        if map_encoder_state_dict:
            try:
                # 尝试strict=True
                missing_keys, unexpected_keys = map_encoder.load_state_dict(map_encoder_state_dict, strict=True)
                rank0_print(f"Map encoder weights loaded successfully with strict=True")
                if missing_keys:
                    rank0_print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    rank0_print(f"Unexpected keys: {unexpected_keys}")
            except Exception as e:
                rank0_print(f"Strict loading failed: {e}")
                # 如果strict=True失败，尝试strict=False
                try:
                    missing_keys, unexpected_keys = map_encoder.load_state_dict(map_encoder_state_dict, strict=False)
                    rank0_print(f"Map encoder weights loaded with strict=False")
                    if missing_keys:
                        rank0_print(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        rank0_print(f"Unexpected keys: {unexpected_keys}")
                except Exception as e2:
                    rank0_print(f"Failed to load map encoder weights: {e2}")
        else:
            rank0_print("No map encoder weights found in checkpoint")

    def get_map_encoder(self):
        map_encoder = getattr(self, "map_encoder", None)
        return map_encoder
    
    def get_map_projector(self):
        """Get map projector for transforming map embeddings"""
        map_projector = getattr(self, "map_projector", None)
        if map_projector is None:
            raise ValueError("Map projector not initialized. Please set mm_map_encoder=True in config.")
        return map_projector
        
    def encode_maps(self, map_data) -> torch.Tensor:
        # Dict[str, torch.Tensor]
        """
        Encode map data using GNN encoder
        Args:
            map_data: Dictionary with map information
        Returns:
            map_embeddings: Encoded map token embeddings
        """
        if not hasattr(self, 'map_encoder'):
            raise ValueError("Map encoder not initialized. Please set mm_map_encoder=True in config.")
        
        map_output = self.map_encoder(map_data)
        map_embeddings = map_output.get('x_pt', None)

        return map_embeddings


class LlavaMapMetaForCausalLM(LlavaMetaForCausalLM):
    """Extended LLaVA CausalLM with map token support"""
    
    @abstractmethod
    def get_model(self):
        pass
    
    def get_map_encoder(self):
        return self.get_model().get_map_encoder()
    
    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images=None, 
        modalities=["map"], 
        image_sizes=None,
        map_data=None,
        map_labels=None,
        node_idx=None,
        **kwargs
    ):
        # 1. 处理text部分 - 直接处理，不涉及map
        if input_ids is not None:
            # 保存原始文本长度
            original_text_len = input_ids.shape[1]
            
            # 检查是否有MAP_TOKEN_INDEX，如果有就替换掉
            if MAP_TOKEN_INDEX in input_ids:
                input_ids = input_ids.clone()
                labels = labels.clone() if labels is not None else None

                map_token_positions = (input_ids == MAP_TOKEN_INDEX).nonzero(as_tuple=False)  # [num_map_tokens, 2]
                # pad_token_id = getattr(self.config, 'pad_token_id', 151643)   # pad_token_id can be none
                input_ids[input_ids == MAP_TOKEN_INDEX] = 0
                
                if labels is not None:
                    labels[labels == MAP_TOKEN_INDEX] = -100
            
            # 直接转换为embeddings
            inputs_embeds = self.get_model().embed_tokens(input_ids)
        else:
            original_text_len = inputs_embeds.shape[1] if inputs_embeds is not None else 0
        
        # 2. 处理map部分
        map_embeds = None
        # text_map_masks = None

        if map_data is not None and map_token_positions is not None:
            # 获取map embeddings
            raw_map_embeds = self._get_map_embeddings_from_data(map_data, node_idx=node_idx, **kwargs)

            # 投影到language model维度
            if hasattr(self.model, 'map_projector'):
                projected_map_embeds = self.model.map_projector(raw_map_embeds)
            else:
                projected_map_embeds = raw_map_embeds
            inputs_embeds = self._replace_map_tokens_with_embeddings(
                inputs_embeds, projected_map_embeds, map_token_positions
            )

            if hasattr(self, 'hierarchical_mot'):
                total_len = inputs_embeds.shape[1]
                batch_size = inputs_embeds.shape[0]
                text_mask = torch.ones(batch_size, total_len, dtype=torch.bool, device=inputs_embeds.device)
                if map_token_positions is not None:
                    for batch_idx, pos_idx in map_token_positions:
                        if batch_idx < text_mask.shape[0]:  # 确保不越界
                            text_mask[batch_idx, pos_idx] = False  # MAP_TOKEN位置不属于text
                map_mask = ~text_mask
                text_map_masks = [text_mask, map_mask]
        return (
            input_ids, position_ids, attention_mask, past_key_values,
            inputs_embeds, labels, raw_map_embeds, text_map_masks, original_text_len
        )

    def _replace_map_tokens_with_embeddings(
        self, 
        inputs_embeds: torch.Tensor, 
        map_embeds: torch.Tensor, 
        map_token_positions: torch.Tensor
    ) -> torch.Tensor:
        """将MAP_TOKEN位置的embedding替换为map embeddings"""
        if map_token_positions is None or len(map_token_positions) == 0:
            return inputs_embeds
        
        # Clone to avoid in-place modification
        inputs_embeds = inputs_embeds.clone()
        
        # 确保map_embeds的形状正确
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        map_batch_size, map_seq_len, map_hidden_dim = map_embeds.shape
        
        if map_hidden_dim != hidden_dim:
            raise ValueError(f"Map embedding dim {map_hidden_dim} doesn't match input dim {hidden_dim}")
        
        # 按batch处理
        batch_map_counters = {i: 0 for i in range(batch_size)}
        for batch_idx, pos_idx in map_token_positions:
            batch_idx = batch_idx.item()
            pos_idx = pos_idx.item()
            if batch_idx < batch_size and pos_idx < seq_len:
                current_map_idx = batch_map_counters[batch_idx]
                if current_map_idx < map_seq_len:
                    # 将对应位置的embedding替换为map embedding
                    inputs_embeds[batch_idx, pos_idx, :] = map_embeds[batch_idx, current_map_idx, :]
                    batch_map_counters[batch_idx] += 1
                else:
                    # 如果map tokens用完了，用零向量填充
                    inputs_embeds[batch_idx, pos_idx, :] = 0.0
        
        return inputs_embeds

    def _get_map_embeddings_from_data(self, map_data, node_idx=None, **kwargs):
        """从map_data获取map embeddings"""
        # 处理batch中的每个样本
        batch_map_embeds = []
        
        for batch_idx in range(len(map_data)):
            if map_data[batch_idx] is not None:
                # 使用map encoder获取embeddings
                map_embeddings = self.get_model().encode_maps(map_data[batch_idx])
                
                # 如果有node_idx，选择特定节点
                if node_idx is not None:
                    if node_idx.dim() == 1:
                        selected_map_embeddings = map_embeddings[node_idx]
                    else:
                        selected_map_embeddings = map_embeddings[node_idx[batch_idx]]
                else:
                    selected_map_embeddings = map_embeddings
                
                batch_map_embeds.append(selected_map_embeddings)
            else:
                # 如果没有map data，创建dummy embeddings
                dummy_embeds = torch.zeros(1, getattr(self.config, "map_hidden_dim", 128), 
                                         device=self.device, dtype=self.dtype)
                batch_map_embeds.append(dummy_embeds)
        
        # Pad到相同长度
        max_map_len = max(embeds.shape[0] for embeds in batch_map_embeds)
        padded_map_embeds = []
        
        for embeds in batch_map_embeds:
            if embeds.shape[0] < max_map_len:
                padding = torch.zeros(max_map_len - embeds.shape[0], embeds.shape[1], 
                                    device=embeds.device, dtype=embeds.dtype)
                padded_embeds = torch.cat([embeds, padding], dim=0)
            else:
                padded_embeds = embeds[:max_map_len]
            padded_map_embeds.append(padded_embeds)
        
        return torch.stack(padded_map_embeds, dim=0)  # [B, max_map_len, map_dim]

    def _construct_combined_attention_mask(self, text_attention_mask, text_len, map_len):
        """构造合并后的attention mask"""
        batch_size = text_attention_mask.shape[0]
        
        # 为map部分创建全1的mask（假设map tokens都是有效的）
        map_attention_mask = torch.ones(batch_size, map_len, 
                                      device=text_attention_mask.device, 
                                      dtype=text_attention_mask.dtype)
        
        # 合并text和map的attention mask
        combined_mask = torch.cat([text_attention_mask, map_attention_mask], dim=1)
        
        return combined_mask

class LlavaMapConfig(PretrainedConfig):
    """Configuration for LLaVA with map support"""
    
    model_type = "llava_map"
    
    def __init__(self, base_config=None, **kwargs):
        # Initialize with base config if provided
        if base_config is not None:
            # Copy base config attributes
            for key, value in vars(base_config).items():
                if not key.startswith('_'):  # Skip private attributes
                    setattr(self, key, value)
        
        # Map encoder settings
        self.dataset = kwargs.get('dataset', 'waymo')
        self.map_input_dim = kwargs.get('map_input_dim', 2)
        self.map_hidden_dim = kwargs.get('map_hidden_dim', 128)
        self.num_historical_steps = kwargs.get('num_historical_steps', 11)
        self.map_gnn_radius = kwargs.get('map_gnn_radius', 10.0)
        self.num_freq_bands = kwargs.get('num_freq_bands', 64)
        self.map_num_layers = kwargs.get('map_num_layers', 3)
        self.map_num_heads = kwargs.get('map_num_heads', 8)
        self.map_head_dim = kwargs.get('map_head_dim', 16)
        self.map_dropout = kwargs.get('map_dropout', 0.1)
        self.mm_map_encoder = kwargs.get('mm_map_encoder', True)
        
        # Cross attention settings
        self.cross_attention_dim = kwargs.get('cross_attention_dim', 896)
        self.cross_attention_heads = kwargs.get('cross_attention_heads', 12)
        self.cross_attention_dropout = kwargs.get('cross_attention_dropout', 0.1)

        # Map token settings
        self.max_map_tokens = kwargs.get('max_map_tokens', 512)
        self.use_map_newline = kwargs.get('use_map_newline', True)
        
        # Call parent constructor
        super().__init__(**kwargs)
