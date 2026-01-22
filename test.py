
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import json
import cv2
import random
import albumentations as A
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from config import Config
from model import create_model
from dataset import ForgeryDataset, get_val_transform
from clustering import detect_forgery_cascade
from utils import oF1_score

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(Config.BACKEND, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# =====================================================
# 1. SPECIAL DATASET WRAPPERS
# =====================================================

class AugmentedHQDataset(Dataset):
    """
    Takes the small HQ dataset (48 images) and expands it to 'target_length'.
    Fix: Ensures masks are upscaled before aug and downscaled after to maintain 33x33 size.
    """
    def __init__(self, base_ds, target_length=500):
        self.base_ds = base_ds
        self.target_length = target_length
        self.num_base = len(base_ds)
        self.feature_size = 33
        self.input_size = 462
        
        self.augs = [
            A.NoOp(p=1.0),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Rotate(limit=(90, 90), p=1.0),
            A.Rotate(limit=(180, 180), p=1.0),
            A.Rotate(limit=(270, 270), p=1.0),
            A.Transpose(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=-0.2, contrast_limit=0, p=1.0),
            A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0),
            A.GridDistortion(p=1.0),
            A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(90, 90), p=1.0)])
        ]

    def __len__(self):
        return self.target_length

    def __getitem__(self, idx):
        base_idx = idx % self.num_base
        aug_idx = (idx // self.num_base) % len(self.augs)
        
        sample = self.base_ds[base_idx]
        
        img_t = sample['input'] # (3, 462, 462)
        masks_t = sample['masks'] # (6, 33, 33)
        
        # 1. Prepare Image (H, W, C)
        img_np = img_t.permute(1, 2, 0).numpy() * 255.0
        img_np = img_np.astype(np.uint8)
        
        # 2. Prepare Masks: UPSAMPLE to match Image (462x462)
        # We perform nearest neighbor upsampling so valid pixels don't get blurred during resize
        masks_list = []
        for m in masks_t:
            # m is (33, 33)
            # Resize to (462, 462)
            m_up = cv2.resize(m.numpy(), (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            masks_list.append(m_up)
            
        # 3. Augment
        transform = self.augs[aug_idx]
        augmented = transform(image=img_np, masks=masks_list)
        
        img_aug = augmented['image']
        masks_aug_highres = augmented['masks']
        
        # 4. Prepare Masks: DOWNSAMPLE back to (33x33)
        # We use MaxPool logic simulation or resizing. 
        # Since we upsampled via Nearest, downsampling via Nearest or Area is fine.
        masks_final = []
        for m in masks_aug_highres:
            # Resize back to 33x33
            # We use INTER_AREA for better decimation or INTER_NEAREST to keep binary-ish nature
            m_down = cv2.resize(m, (self.feature_size, self.feature_size), interpolation=cv2.INTER_NEAREST)
            masks_final.append(torch.from_numpy(m_down))
            
        img_ret = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
        masks_ret = torch.stack(masks_final)
        
        return {
            'input': img_ret,
            'masks': masks_ret, # Consistently (6, 33, 33)
            'is_forged': sample['is_forged'],
            'image_name': f"{sample['image_name']}_aug{aug_idx}"
        }

class CombinedDataset(Dataset):
    def __init__(self, mapping_list):
        self.mapping = mapping_list
    def __len__(self): return len(self.mapping)
    def __getitem__(self, idx):
        ds, real_idx = self.mapping[idx]
        return ds[real_idx]

# =====================================================
# 2. MAIN EVALUATION LOGIC
# =====================================================

def evaluate_ddp(rank, world_size):
    setup_ddp(rank, world_size)
    
    # --- CONFIG ---
    if rank == 0:
        print("⚙️  Configuring Inference...")
    
    Config.ENABLE_CASCADE = True
    Config.ANOMALY_PERCENTILE = 95
    Config.ANOMALY_THRESHOLD = 0.05
    Config.LOOSE_MIN_CLUSTER_SIZE = 4
    Config.LOOSE_MIN_SAMPLES = 2
    Config.LOOSE_GEOMETRY_THRESHOLD = 0.15
    Config.EXPANSION_THRESHOLD = 0.20
    
    if rank == 0:
        p_path = os.path.join(Config.OUTPUT_DIR, 'best_hdbscan_params.json')
        if os.path.exists(p_path):
            with open(p_path) as f: p = json.load(f)
            Config.HDBSCAN_MIN_CLUSTER_SIZE = p['min_cluster_size']
            Config.HDBSCAN_MIN_SAMPLES = p['min_samples']
            Config.GEOMETRY_THRESHOLD = p['geometry_threshold']
            print(f"Loaded strict params: {p}")
            
    dist.barrier()
    
    # --- DATASET CONSTRUCTION ---
    ds_std = ForgeryDataset(
        Config.TEST_AUTHENTIC_DIR, 
        Config.TEST_FORGED_DIR, 
        Config.TEST_MASK_DIR, 
        None, Config
    )
    
    ds_hq_base = ForgeryDataset(
        "/dummy/path", 
        Config.HQ_FORGED_DIR, 
        Config.HQ_MASK_DIR, 
        None, Config
    )
    
    ds_hq_aug = AugmentedHQDataset(ds_hq_base, target_length=500)
    
    std_auth_indices = [i for i, s in enumerate(ds_std.samples) if not s['is_forged']]
    std_forg_indices = [i for i, s in enumerate(ds_std.samples) if s['is_forged']]
    hq_aug_indices = list(range(500))
    
    random.seed(42)
    random.shuffle(std_auth_indices)
    random.shuffle(std_forg_indices)
    
    final_auth_idx = std_auth_indices[:300]
    final_std_forg_idx = std_forg_indices[:200]
    
    mapping_list = []
    for idx in hq_aug_indices: mapping_list.append((ds_hq_aug, idx))
    for idx in final_std_forg_idx: mapping_list.append((ds_std, idx))
    for idx in final_auth_idx: mapping_list.append((ds_std, idx))
        
    random.shuffle(mapping_list)
    combined_ds = CombinedDataset(mapping_list)
    
    if rank == 0:
        print(f"\n📊 DATASET COMPOSITION (Target: 1000)")
        print(f"   HQ Forged (Augmented):  {len(hq_aug_indices)}")
        print(f"   Standard Forged:        {len(final_std_forg_idx)}")
        print(f"   Standard Authentic:     {len(final_auth_idx)}")
        print(f"   TOTAL:                  {len(combined_ds)}")
        print("-" * 40)

    sampler = DistributedSampler(combined_ds, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(
        combined_ds, 
        batch_size=Config.BATCH_SIZE, 
        sampler=sampler, 
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # --- MODEL ---
    model = create_model(Config).to(rank)
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, 'best_checkpoint.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, 'current_checkpoint.pth')
    
    ckpt = torch.load(ckpt_path, map_location=f'cuda:{rank}', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = DDP(model, device_ids=[rank])
    model.eval()
    
    metrics = []
    
    with torch.no_grad():
        iterator = tqdm(loader, disable=(rank!=0), desc="Testing") if rank == 0 else loader
        for batch in iterator:
            inputs = batch['input'].to(rank, non_blocking=True)
            masks_33 = batch['masks'].cpu().numpy()
            is_forged_batch = batch['is_forged'].cpu().numpy()
            
            embeddings = model(inputs).cpu().numpy()
            
            for b in range(inputs.shape[0]):
                gt = []
                for k in range(6):
                    if masks_33[b, k].sum() > 0:
                        gt.append(cv2.resize(masks_33[b, k], (224, 224), interpolation=cv2.INTER_NEAREST))
                
                preds = detect_forgery_cascade(embeddings[b], Config)
                
                if not gt:
                    score = 1.0 if not preds else 0.0
                else:
                    score = oF1_score(preds, gt) if preds else 0.0
                
                metrics.append({
                    'is_forged': int(is_forged_batch[b]),
                    'pred_forged': int(len(preds) > 0),
                    'of1': float(score)
                })
    
    gathered = [None] * world_size
    dist.all_gather_object(gathered, metrics)
    
    if rank == 0:
        all_m = [x for sub in gathered for x in sub]
        y_true = [x['is_forged'] for x in all_m]
        y_pred = [x['pred_forged'] for x in all_m]
        of1s = [x['of1'] for x in all_m]
        
        auth_scores = [x['of1'] for x in all_m if x['is_forged'] == 0]
        forg_scores = [x['of1'] for x in all_m if x['is_forged'] == 1]
        
        print("\n" + "="*40)
        print("📈 FINAL RESULTS")
        print("="*40)
        print(f"Total Processed: {len(all_m)}")
        print(f"Authentic: {len(auth_scores)} | Forged: {len(forg_scores)}")
        print("-" * 40)
        print(f"🔹 OVERALL oF1:      {np.mean(of1s):.4f}")
        print(f"🔹 AUTHENTIC oF1:    {np.mean(auth_scores):.4f}")
        print(f"🔹 FORGED oF1:       {np.mean(forg_scores):.4f}")
        print("-" * 40)
        print(f"Detection F1:       {f1_score(y_true, y_pred):.4f}")
        print(f"Accuracy:           {accuracy_score(y_true, y_pred):.4f}")
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f"Confusion Matrix:   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print("="*40)
        
    cleanup_ddp()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.spawn(evaluate_ddp, args=(Config.WORLD_SIZE,), nprocs=Config.WORLD_SIZE, join=True)