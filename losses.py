
import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for Pixel/Patch Embeddings.
    Adapts SupCon to work on the (N_patches, Dim) level per image.
    """
    def __init__(self, temperature=0.07):
        super(PixelSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, masks):
        """
        Args:
            embeddings: (B, 33, 33, 512) - Normalized embeddings
            masks: (B, 6, 33, 33) - Ground truth instance masks
        Returns:
            scalar loss
        """
        device = embeddings.device
        B, H, W, D = embeddings.shape
        
        total_loss = 0.0
        valid_batches = 0

        for b in range(B):
            # 1. Flatten embeddings: (N, 512) where N=1089
            # Embeddings are already L2 normalized in model.py
            features = embeddings[b].reshape(-1, D)
            
            # 2. Construct Labels: (N,)
            # 0 = Background, 1..K = Forgery Instances
            curr_masks = masks[b] # (6, 33, 33)
            labels = torch.zeros((H, W), device=device, dtype=torch.long)
            
            has_forgery = False
            for i in range(curr_masks.shape[0]):
                if curr_masks[i].sum() > 0:
                    # Assign label (i+1) to this instance mask
                    labels[curr_masks[i] > 0.5] = (i + 1)
                    has_forgery = True
            
            labels = labels.flatten() # (N,)

            # If authentic image (only background/label 0), SupCon works but acts 
            # like uniformity (pulling everything together).
            # We process it to ensure background patches are cohesive.
            
            # 3. Compute Similarity Matrix
            # (N, D) @ (D, N) -> (N, N)
            anchor_dot_contrast = torch.div(
                torch.matmul(features, features.T),
                self.temperature
            )

            # 4. Numerical Stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # 5. Create Masks
            # Mask of same class: (N, N) - 1 if same label, 0 else
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
            
            # Mask out self-contrast (diagonal)
            logits_mask = torch.ones_like(mask).scatter_(1, 
                torch.arange(mask.shape[0]).view(-1, 1).to(device), 0)
            
            # Final positive mask (same label, not self)
            mask = mask * logits_mask

            # 6. Compute Log Probabilities
            # Exp(sim) for all NOT self (positives + negatives)
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            # 7. Compute Mean Log-Likelihood over Positives
            mask_sum = mask.sum(1)
            
            # Prevent division by zero for singleton patches (rare/noise)
            # Just set sum to 1 where it's 0 to avoid NaN, result masked out anyway
            mask_sum_safe = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
            
            # Sum of log_prob for positives / num_positives
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum_safe

            # 8. Loss
            # Only count patches that actually have a positive pair
            loss = -mean_log_prob_pos
            
            # Apply mask for valid anchors (those with at least 1 positive)
            loss = loss * (mask_sum > 0).float()
            
            # Average over valid anchors
            num_valid_anchors = (mask_sum > 0).sum()
            
            if num_valid_anchors > 0:
                total_loss += loss.sum() / num_valid_anchors
                valid_batches += 1

        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            # Fallback for empty batch or issues
            return torch.tensor(0.0, device=device, requires_grad=True)

class UniformityLoss(nn.Module):
    """Minimize variance within instances"""
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, embeddings, masks, is_forged):
        B, H, W, D = embeddings.shape
        device = embeddings.device
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(B):
            emb_flat = embeddings[b].reshape(-1, D)
            
            if is_forged[b] < 0.5:
                # Authentic: Global variance
                mean_emb = emb_flat.mean(dim=0)
                variance = ((emb_flat - mean_emb) ** 2).sum(dim=1).mean()
                if variance > self.threshold:
                    total_loss += F.relu(variance - self.threshold)
                    valid_batches += 1
            else:
                # Forged: Per-instance variance
                curr_masks = masks[b]
                label_map = torch.zeros((H, W), device=device, dtype=torch.long)
                for i in range(curr_masks.shape[0]):
                    if curr_masks[i].sum() > 0:
                        label_map[curr_masks[i] > 0.5] = (i + 1)
                
                label_flat = label_map.flatten()
                unique_labels = torch.unique(label_flat)
                
                instance_loss = 0.0
                count = 0
                for label in unique_labels:
                    inst_emb = emb_flat[label_flat == label]
                    if inst_emb.shape[0] < 2: continue
                    
                    mean_inst = inst_emb.mean(dim=0)
                    var = ((inst_emb - mean_inst) ** 2).sum(dim=1).mean()
                    if var > self.threshold:
                        instance_loss += F.relu(var - self.threshold)
                        count += 1
                
                if count > 0:
                    total_loss += instance_loss / count
                    valid_batches += 1
                    
        return total_loss / valid_batches if valid_batches > 0 else torch.tensor(0.0, device=device, requires_grad=True)

class InterInstanceSeparationLoss(nn.Module):
    """Maximize distance between instance centroids"""
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, masks):
        B, H, W, D = embeddings.shape
        device = embeddings.device
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(B):
            emb = embeddings[b].reshape(-1, D)
            curr_masks = masks[b]
            
            label_map = torch.zeros((H, W), device=device, dtype=torch.long)
            for i in range(curr_masks.shape[0]):
                if curr_masks[i].sum() > 0:
                    label_map[curr_masks[i] > 0.5] = (i + 1)
            
            label_flat = label_map.flatten()
            unique_labels = torch.unique(label_flat)
            
            if len(unique_labels) < 2: continue
            
            centroids = []
            for label in unique_labels:
                inst_emb = emb[label_flat == label]
                if inst_emb.shape[0] > 0:
                    centroids.append(inst_emb.mean(dim=0))
            
            if len(centroids) < 2: continue
            
            centroids = torch.stack(centroids) # (K, 512)
            dist_matrix = torch.cdist(centroids, centroids, p=2)
            
            loss = 0.0
            count = 0
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    # Hinge loss: penalize if dist < margin
                    loss += F.relu(self.margin - dist_matrix[i, j])
                    count += 1
            
            if count > 0:
                total_loss += loss / count
                valid_batches += 1
                
        return total_loss / valid_batches if valid_batches > 0 else torch.tensor(0.0, device=device, requires_grad=True)

class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.supcon_loss = PixelSupConLoss(temperature=config.SUPCON_TEMPERATURE)
        self.uniformity_loss = UniformityLoss(threshold=config.UNIFORMITY_THRESHOLD)
        self.separation_loss = InterInstanceSeparationLoss(margin=config.SEPARATION_MARGIN)
    
    def forward(self, embeddings, masks, is_forged):
        """
        Args:
            embeddings: (B, 33, 33, 512)
            masks: (B, 6, 33, 33)
            is_forged: (B,)
        """
        # 1. SupCon (Primary Driver: Clustering)
        supcon = self.supcon_loss(embeddings, masks)
        
        # 2. Separation (Helper: Force centroids apart)
        separation = self.separation_loss(embeddings, masks)
        
        # 3. Uniformity (Helper: Force low variance)
        uniformity = self.uniformity_loss(embeddings, masks, is_forged)
        
        total = (
            self.config.LOSS_WEIGHT_SUPCON * supcon +
            self.config.LOSS_WEIGHT_SEPARATION * separation +
            self.config.LOSS_WEIGHT_UNIFORMITY * uniformity
        )
        
        loss_dict = {
            'supcon': supcon.item() if torch.is_tensor(supcon) else supcon,
            'separation': separation.item() if torch.is_tensor(separation) else separation,
            'uniformity': uniformity.item() if torch.is_tensor(uniformity) else uniformity
        }
        
        return total, loss_dict