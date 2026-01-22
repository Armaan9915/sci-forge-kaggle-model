
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import torch.distributed as dist
from dinov2.models.vision_transformer import vit_base

class MLP(nn.Module):
    """3-layer MLP: 768 → 1024 → 768 → 512"""
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        # CRITICAL for SupCon: L2 Normalize embeddings
        x = F.normalize(x, p=2, dim=-1)
        return x

class DINOv2ForgeryDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dino = self._load_dinov2(config)
        self.freeze_dino()
        self.mlp = MLP(
            input_dim=config.DINO_EMBEDDING_DIM,
            hidden_dim=config.MLP_HIDDEN_DIM,
            output_dim=config.FINAL_EMBEDDING_DIM
        )
    
    def _load_dinov2(self, config):
        if os.path.exists(config.DINOV2_PATH):
            try:
                if config.DINOV2_PATH not in sys.path:
                    sys.path.insert(0, config.DINOV2_PATH)
                
                model = vit_base(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
                weights_path = os.path.join(config.DINOV2_PATH, 'dinov2_vitb14_pretrain.pth')
                if os.path.exists(weights_path):
                    state_dict = torch.load(weights_path, map_location='cpu')
                    model.load_state_dict(state_dict, strict=False)
                return model
            except Exception as e:
                print(f"⚠️ Local load failed: {e}")
        
        
        if dist.is_initialized() and dist.get_rank() == 0:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)
        else:
            # Wait or load if not distributed
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)
        return model
        
    def freeze_dino(self):
        for param in self.dino.parameters():
            param.requires_grad = False
    
    def unfreeze_dino(self):
        for param in self.dino.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
            features = self.dino.forward_features(x)
            patch_tokens = features['x_norm_patchtokens']
        
        embeddings = self.mlp(patch_tokens)
        N = patch_tokens.shape[1]
        H = W = int(N ** 0.5)
        embeddings = embeddings.reshape(x.shape[0], H, W, -1)
        return embeddings
    
    def get_learnable_params(self):
        return list(self.mlp.parameters()), list(self.dino.parameters())

def create_model(config):
    return DINOv2ForgeryDetector(config)