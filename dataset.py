
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
import cv2
import torch.nn.functional as F

class ForgeryDataset(Dataset):
    def __init__(self, authentic_dir, forged_dir, mask_dir, transform=None, config=None):
        self.authentic_dir = authentic_dir
        self.forged_dir = forged_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.config = config
        
        self.samples = []
        if os.path.exists(authentic_dir):
            auth_imgs = sorted([f for f in os.listdir(authentic_dir) 
                              if f.endswith(('.jpg', '.png', '.jpeg'))])
            for img in auth_imgs:
                self.samples.append({'image': img, 'is_forged': False, 'dir': authentic_dir})
                
        if os.path.exists(forged_dir):
            forg_imgs = sorted([f for f in os.listdir(forged_dir) 
                              if f.endswith(('.jpg', '.png', '.jpeg'))])
            for img in forg_imgs:
                self.samples.append({'image': img, 'is_forged': True, 'dir': forged_dir})
        
        print(f"📦 Total samples: {len(self.samples)}")
        
        if config and config.SMOKE_TEST:
            random.shuffle(self.samples)
            self.samples = self.samples[:config.SMOKE_TEST_SAMPLES]
            print(f"🔥 SMOKE TEST: Sampled {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def letterbox_resize(self, image, masks, target_size):
        h, w = image.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_off, x_off = (target_size - new_h) // 2, (target_size - new_w) // 2
        padded_image[y_off:y_off+new_h, x_off:x_off+new_w] = resized_image
        
        padded_masks = []
        for mask in masks:
            m_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            p_mask = np.zeros((target_size, target_size), dtype=np.uint8)
            p_mask[y_off:y_off+new_h, x_off:x_off+new_w] = m_resized
            padded_masks.append(p_mask)
            
        return padded_image, padded_masks
    
    def downsample_masks_maxpool(self, masks, config):
        """Downsample masks: MaxPool(4x4) -> 144x144 -> Interpolate -> 33x33"""
        if not masks: return []
        
        masks_tensor = torch.from_numpy(np.stack(masks)).float().unsqueeze(1)
        pad = (464 - 462) // 2
        if pad > 0:
            masks_tensor = F.pad(masks_tensor, (pad, pad, pad, pad), mode='constant', value=0)
        
        masks_tensor = F.max_pool2d(masks_tensor, kernel_size=4, stride=4)
        masks_tensor = F.interpolate(masks_tensor, size=(144, 144), mode='nearest')
        masks_tensor = F.interpolate(masks_tensor, size=(config.FEATURE_MAP_SIZE, config.FEATURE_MAP_SIZE), 
                                     mode='bilinear', align_corners=False)
        masks_tensor = (masks_tensor > 0.3).float().squeeze(1)
        return [masks_tensor[i].numpy() for i in range(masks_tensor.shape[0])]
    
    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            img_path = os.path.join(sample['dir'], sample['image'])
            image = cv2.imread(img_path)
            if image is None: raise ValueError("Image not found")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            masks = []
            if sample['is_forged']:
                mask_path = os.path.join(self.mask_dir, os.path.splitext(sample['image'])[0] + '.npy')
                if os.path.exists(mask_path):
                    mask_data = np.load(mask_path, allow_pickle=True)
                    h, w = image.shape[:2]
                    if mask_data.ndim == 2:
                        if mask_data.shape != (h, w):
                            mask_data = cv2.resize(mask_data.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        masks.append(mask_data)
                    elif mask_data.ndim == 3:
                        for i in range(mask_data.shape[0]):
                            if np.sum(mask_data[i]) > 0:
                                m = mask_data[i]
                                if m.shape != (h, w):
                                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                                masks.append(m)
            
            image, masks = self.letterbox_resize(image, masks, self.config.INPUT_SIZE)
            
            if self.transform:
                if masks:
                    aug = self.transform(image=image, masks=masks)
                    image, masks = aug['image'], aug['masks']
                else:
                    image = self.transform(image=image)['image']
            
            masks_33 = self.downsample_masks_maxpool(masks, self.config)
            
            # Pad to MAX_INSTANCES + 1 (background)
            # Default to 6 total slots (background + 5 instances)
            while len(masks_33) < 6:
                masks_33.append(np.zeros((self.config.FEATURE_MAP_SIZE, self.config.FEATURE_MAP_SIZE), dtype=np.float32))
            masks_33 = masks_33[:6]
            
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            masks_33 = torch.stack([torch.from_numpy(m.astype(np.float32)) for m in masks_33])
            
            return {
                'input': image,
                'masks': masks_33,
                'is_forged': torch.tensor(sample['is_forged'], dtype=torch.float32),
                'image_name': sample['image']
            }
        except Exception as e:
            print(f"Error loading {sample['image']}: {e}")
            return self.__getitem__(random.randint(0, len(self.samples)-1))

def get_train_transform(config):
    return A.Compose([
        A.HorizontalFlip(p=config.AUG_PROB_FLIP),
        A.VerticalFlip(p=config.AUG_PROB_FLIP),
        A.Rotate(limit=config.AUG_ROTATION_LIMIT, p=config.AUG_PROB_ROTATE),
        A.GaussianBlur(blur_limit=config.AUG_BLUR_LIMIT, p=config.AUG_PROB_BLUR),
        A.GaussNoise(var_limit=config.AUG_NOISE_VAR_LIMIT, p=config.AUG_PROB_NOISE),
        A.ImageCompression(quality_range=config.AUG_JPEG_QUALITY_RANGE, p=config.AUG_PROB_JPEG),
        A.ColorJitter(p=config.AUG_PROB_COLOR)
    ])

def get_val_transform(config):
    return None

def create_dataloaders(config, rank=0, world_size=1):
    train_ds = ForgeryDataset(config.TRAIN_AUTHENTIC_DIR, config.TRAIN_FORGED_DIR, config.TRAIN_MASK_DIR, get_train_transform(config), config)
    val_ds = ForgeryDataset(config.TEST_AUTHENTIC_DIR, config.TEST_FORGED_DIR, config.TEST_MASK_DIR, get_val_transform(config), config)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, train_sampler, val_sampler