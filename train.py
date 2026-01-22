
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
import cv2

from config import Config
from model import create_model
from losses import CombinedLoss
from dataset import create_dataloaders
from clustering import cluster_embeddings, generate_masks_from_clusters
from utils import oF1_score

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(Config.BACKEND, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_of1, 
                   checkpoint_dir, is_best=False):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_of1': best_of1
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'current_checkpoint.pth'))
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_checkpoint.pth'))

def compute_of1_score(model, dataloader, config, device, rank):
    model.eval()
    
    authentic_scores, forged_scores = [], []
    total_batches = len(dataloader)
    sample_batches = max(1, int(total_batches * config.VAL_SAMPLE_RATIO))
    
    if rank == 0:
        print(f"📊 Validating on {sample_batches}/{total_batches} batches")
    
    random.seed(42)
    sampled_indices = sorted(random.sample(range(total_batches), sample_batches))
    sampled_set = set(sampled_indices)
    
    with torch.no_grad():
        batch_idx = 0
        pbar = tqdm(dataloader, desc="Validation", disable=(rank != 0), total=len(dataloader))
        
        for batch in pbar:
            if batch_idx not in sampled_set:
                batch_idx += 1
                pbar.update(1)
                continue
            
            inputs = batch['input'].to(device)
            masks_33 = batch['masks'].cpu().numpy()
            is_forged_batch = batch['is_forged'].cpu().numpy()
            
            embeddings = model(inputs)
            embeddings_np = embeddings.cpu().numpy()
            
            for b in range(inputs.shape[0]):
                is_forged = is_forged_batch[b]
                
                # Filter ratio
                if is_forged < 0.5:
                    if random.random() > config.VAL_AUTHENTIC_RATIO / (config.VAL_AUTHENTIC_RATIO + config.VAL_FORGED_RATIO):
                        continue
                else:
                    if random.random() > config.VAL_FORGED_RATIO / (config.VAL_AUTHENTIC_RATIO + config.VAL_FORGED_RATIO):
                        continue
                
                embedding = embeddings_np[b]
                
                gt_masks = []
                for i in range(6):
                    if masks_33[b, i].sum() > 0:
                        gt_mask = cv2.resize(masks_33[b, i],
                                           (config.EVAL_MASK_SIZE, config.EVAL_MASK_SIZE),
                                           interpolation=cv2.INTER_NEAREST)
                        gt_masks.append(gt_mask)
                
                cluster_labels = cluster_embeddings(embedding, config)
                
                pred_masks = generate_masks_from_clusters(cluster_labels, config,
                                                         max_instances=config.MAX_INSTANCES,
                                                         embeddings=embedding)
                
                if not gt_masks:
                    score = 1.0 if not pred_masks else 0.0
                else:
                    score = 0.0 if not pred_masks else oF1_score(pred_masks, gt_masks)
                
                if np.isnan(score): score = 0.0
                
                if is_forged >= 0.5:
                    forged_scores.append(score)
                else:
                    authentic_scores.append(score)
            
            del inputs, embeddings, embeddings_np
            torch.cuda.empty_cache()
            
            batch_idx += 1
            pbar.update(1)
    
    model.train()
    
    if rank == 0:
        print(f"📊 Evaluated {len(authentic_scores)} authentic, {len(forged_scores)} forged")
    
    overall = np.mean(authentic_scores + forged_scores) if (authentic_scores + forged_scores) else 0.0
    auth_of1 = np.mean(authentic_scores) if authentic_scores else 0.0
    forg_of1 = np.mean(forged_scores) if forged_scores else 0.0
    
    return overall, auth_of1, forg_of1

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, 
                   epoch, config, device, rank):
    model.train()
    total_loss = 0.0
    loss_sum = {'supcon': 0.0, 'separation': 0.0, 'uniformity': 0.0}
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
    for i, batch in enumerate(pbar):
        inputs = batch['input'].to(device)
        masks = batch['masks'].to(device)
        is_forged = batch['is_forged'].to(device)
        
        with autocast(enabled=config.USE_AMP):
            embeddings = model(inputs)
            loss, loss_dict = criterion(embeddings, masks, is_forged)
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        for k in loss_dict:
            loss_sum[k] += loss_dict[k]
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f"{loss.item()*config.GRADIENT_ACCUMULATION_STEPS:.4f}",
                'sup': f"{loss_dict['supcon']:.3f}",
                'sep': f"{loss_dict['separation']:.3f}",
                'uni': f"{loss_dict['uniformity']:.3f}"
            })
    
    return total_loss / len(loader), {k: v / len(loader) for k, v in loss_sum.items()}

def train_ddp(rank, world_size, config, smoke_test=False):
    setup_ddp(rank, world_size)
    
    if smoke_test:
        config.set_smoke_test(True)
    
    if rank == 0:
        config.print_config()
    
    train_loader, val_loader, train_sampler, _ = create_dataloaders(config, rank, world_size)
    
    model = create_model(config).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    mlp_params, dino_params = model.module.get_learnable_params()
    
    optimizer = AdamW([
        {'params': mlp_params, 'lr': config.MLP_LEARNING_RATE},
        {'params': dino_params, 'lr': config.DINO_LEARNING_RATE}
    ], weight_decay=config.WEIGHT_DECAY)
    
    criterion = CombinedLoss(config).to(rank)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN)
    scaler = GradScaler(enabled=config.USE_AMP)
    
    best_of1 = -1.0
    patience_counter = 0
    early_stop = False
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        if early_stop:
            if rank == 0: print(f"\n🛑 Early stopping triggered")
            break
        
        train_sampler.set_epoch(epoch)
        
        if epoch == config.ENCODER_FREEZE_EPOCHS + 1:
            model.module.unfreeze_dino()
        
        loss, loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            scaler, epoch, config, rank, rank
        )
        
        scheduler.step()
        
        if epoch % config.VAL_FREQUENCY == 0 or epoch == 1 or epoch == config.NUM_EPOCHS:
            overall, auth, forg = compute_of1_score(model, val_loader, config, rank, rank)
            
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
                print(f"Loss: {loss:.4f} (SupCon: {loss_dict['supcon']:.4f}, "
                      f"Sep: {loss_dict['separation']:.4f}, Uni: {loss_dict['uniformity']:.4f})")
                print(f"OF1: Overall={overall:.4f} | Auth={auth:.4f} | Forged={forg:.4f}")
                
                if overall > best_of1 + config.EARLY_STOPPING_MIN_DELTA:
                    best_of1 = overall
                    patience_counter = 0
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, 
                                  best_of1, config.CHECKPOINT_DIR, is_best=True)
                    print(f"✅ New best OF1: {best_of1:.4f}")
                else:
                    patience_counter += 1
                    print(f"⏳ Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
                    if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                        early_stop = True
                print(f"{'='*60}\n")
        
        if rank == 0 and epoch % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, 
                          best_of1, config.CHECKPOINT_DIR, is_best=False)
        
        if rank == 0:
            early_stop_tensor = torch.tensor([1.0 if early_stop else 0.0], device=rank)
        else:
            early_stop_tensor = torch.tensor([0.0], device=rank)
        dist.broadcast(early_stop_tensor, src=0)
        early_stop = early_stop_tensor.item() > 0.5
    
    if rank == 0:
        print(f"\n🏁 Training completed! Best OF1: {best_of1:.4f}")
    
    cleanup_ddp()

def main():
    import sys
    import torch.multiprocessing as mp
    
    smoke_test = '--smoke-test' in sys.argv
    mp.spawn(train_ddp, args=(Config.WORLD_SIZE, Config, smoke_test), 
             nprocs=Config.WORLD_SIZE, join=True)

if __name__ == '__main__':
    main()