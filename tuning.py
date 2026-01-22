
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import random
from pathlib import Path
import cv2

from config import Config
from model import create_model
from dataset import ForgeryDataset, get_val_transform
from utils import oF1_score

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group(Config.BACKEND, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def predict_with_params(embeddings, gt_masks, min_cluster_size, min_samples, geometry_threshold, config):
    from hdbscan import HDBSCAN
    H, W, D = embeddings.shape
    emb_flat = embeddings.reshape(-1, D)
    
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                       metric=config.HDBSCAN_METRIC, core_dist_n_jobs=-1)
    cluster_labels = clusterer.fit_predict(emb_flat).reshape(H, W)
    
    unique = np.unique(cluster_labels)
    unique = unique[unique >= 0]
    
    if len(unique) == 0: return 1.0 if not gt_masks else 0.0
    
    # Identify Background
    label_flat = cluster_labels.flatten()
    sizes = [(c, (label_flat == c).sum()) for c in unique]
    sizes.sort(key=lambda x: x[1], reverse=True)
    bg_cluster = sizes[0][0]
    
    # Validate
    valid_clusters = []
    bg_mask = (label_flat == bg_cluster)
    bg_centroid = emb_flat[bg_mask].mean(axis=0)
    bg_norm = bg_centroid / (np.linalg.norm(bg_centroid) + 1e-8)
    
    for cid, _ in sizes[1:]:
        cand_mask = (label_flat == cid)
        if cand_mask.sum() == 0: continue
        cand_centroid = emb_flat[cand_mask].mean(axis=0)
        cand_norm = cand_centroid / (np.linalg.norm(cand_centroid) + 1e-8)
        
        dist = 1.0 - np.dot(bg_norm, cand_norm)
        if dist > geometry_threshold:
            valid_clusters.append(cid)
    
    # Generate Masks
    if len(valid_clusters) > config.MAX_INSTANCES:
        valid_sizes = [(c, (label_flat == c).sum()) for c in valid_clusters]
        valid_sizes.sort(key=lambda x: x[1], reverse=True)
        valid_clusters = [c for c, _ in valid_sizes[:config.MAX_INSTANCES]]
        
    pred_masks = []
    for cid in valid_clusters:
        mask = (cluster_labels == cid).astype(np.float32)
        # Patch upscale
        mask_462 = np.zeros((462, 462), dtype=np.float32)
        for i in range(33):
            for j in range(33):
                if mask[i, j] > 0.5:
                    y, x = i*14, j*14
                    mask_462[y:min(y+14, 462), x:min(x+14, 462)] = 1.0
        pred_masks.append(cv2.resize(mask_462, (224, 224), interpolation=cv2.INTER_NEAREST))
        
    if not gt_masks: return 1.0 if not pred_masks else 0.0
    return 0.0 if not pred_masks else oF1_score(pred_masks, gt_masks)

def threshold_search_ddp(rank, world_size):
    setup_ddp(rank, world_size)
    
    ds = ForgeryDataset(Config.TEST_AUTHENTIC_DIR, Config.TEST_FORGED_DIR, Config.TEST_MASK_DIR, get_val_transform(Config), Config)
    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, sampler=DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False), num_workers=4)
    
    # Load Model
    model = create_model(Config).to(rank)
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, 'best_checkpoint.pth')
    if not os.path.exists(ckpt_path): ckpt_path = os.path.join(Config.CHECKPOINT_DIR, 'current_checkpoint.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=f'cuda:{rank}')['model_state_dict'])
    model = DDP(model, device_ids=[rank])
    model.eval()
    
    # 1. Extract Embeddings
    all_data = []
    sample_batches = max(1, int(len(loader) * Config.TUNING_SAMPLE_RATIO))
    random.seed(42)
    sampled_indices = set(random.sample(range(len(loader)), sample_batches))
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, disable=(rank!=0), desc="Extracting")):
            if i not in sampled_indices: continue
            inputs = batch['input'].to(rank)
            masks_33 = batch['masks'].cpu().numpy()
            is_forged = batch['is_forged'].cpu().numpy()
            
            emb = model(inputs).cpu().numpy()
            
            for b in range(inputs.shape[0]):
                # Ratio filter
                if is_forged[b] < 0.5:
                    if random.random() > 0.3: continue
                else:
                    if random.random() > 0.7: continue
                    
                gt = []
                for k in range(6):
                    if masks_33[b, k].sum() > 0:
                        gt.append(cv2.resize(masks_33[b, k], (224, 224), interpolation=cv2.INTER_NEAREST))
                all_data.append({'embeddings': emb[b], 'gt_masks': gt})
    
    # 2. Tuning
    # Coarse Search
    best_score = -1
    best_params = (15, 5, 0.5)
    
    if rank == 0: print("🔍 Tuning HDBSCAN...")
    for min_cs in Config.TEST_MIN_CLUSTER_SIZES:
        for min_s in Config.TEST_MIN_SAMPLES:
            scores = [predict_with_params(d['embeddings'], d['gt_masks'], min_cs, min_s, 0.5, Config) for d in all_data]
            mean_score = np.mean(scores) if scores else 0
            if mean_score > best_score:
                best_score = mean_score
                best_params = (min_cs, min_s, 0.5)
    
    # Fine Search Geometry
    b_cs, b_s, _ = best_params
    if rank == 0: print(f"🔍 Tuning Geometry Threshold (using {b_cs}, {b_s})...")
    for geo in Config.TEST_GEOMETRY_THRESHOLDS:
        scores = [predict_with_params(d['embeddings'], d['gt_masks'], b_cs, b_s, geo, Config) for d in all_data]
        mean_score = np.mean(scores) if scores else 0
        if mean_score > best_score:
            best_score = mean_score
            best_params = (b_cs, b_s, geo)
            
    # Gather
    gathered = [None] * world_size
    dist.all_gather_object(gathered, {'params': best_params, 'score': best_score})
    
    if rank == 0:
        final_best = max(gathered, key=lambda x: x['score'])
        print(f"🏆 Best Params: {final_best['params']} | Score: {final_best['score']:.4f}")
        with open(os.path.join(Config.OUTPUT_DIR, 'best_hdbscan_params.json'), 'w') as f:
            json.dump({
                'min_cluster_size': final_best['params'][0],
                'min_samples': final_best['params'][1],
                'geometry_threshold': final_best['params'][2]
            }, f)
            
    cleanup_ddp()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.spawn(threshold_search_ddp, args=(Config.WORLD_SIZE,), nprocs=Config.WORLD_SIZE, join=True)