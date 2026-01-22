
import numpy as np
from hdbscan import HDBSCAN
import cv2
import torch
import torch.nn.functional as F

# ================= HELPER: ANOMALY SCORE =================
def get_anomaly_score(embeddings, percentile=95):
    """
    Calculates how 'suspicious' the image is based on outlier distance.
    Returns: scalar score (higher = more suspicious)
    """
    H, W, D = embeddings.shape
    emb_flat = embeddings.reshape(-1, D)
    
    # 1. Global Mean (Proxy for background)
    global_mean = emb_flat.mean(axis=0)
    
    # 2. Normalize
    emb_norm = emb_flat / (np.linalg.norm(emb_flat, axis=1, keepdims=True) + 1e-8)
    mean_norm = global_mean / (np.linalg.norm(global_mean) + 1e-8)
    
    # 3. Cosine Distance (1 - dot)
    # Shape: (N,)
    cosine_dist = 1.0 - np.dot(emb_norm, mean_norm)
    
    # 4. Return the distance of the top 5% most distinct patches
    return np.percentile(cosine_dist, percentile)

# ================= HELPER: REGION GROWING =================
def expand_clusters(cluster_labels, embeddings, valid_clusters, threshold=0.2):
    """
    Region Growing: Absorb background pixels similar to valid clusters.
    """
    H, W, D = embeddings.shape
    emb_flat = embeddings.reshape(-1, D)
    labels_flat = cluster_labels.flatten()
    
    # Copy labels to modify
    new_labels = labels_flat.copy()
    
    # Mask of pixels that are currently background (not in any valid cluster)
    # We allow expansion ONLY into the background, not into other clusters.
    is_background = ~np.isin(labels_flat, valid_clusters)
    
    # Normalize all embeddings once
    emb_norm = emb_flat / (np.linalg.norm(emb_flat, axis=1, keepdims=True) + 1e-8)

    for cluster_id in valid_clusters:
        # 1. Get Core Centroid
        mask = (labels_flat == cluster_id)
        if mask.sum() == 0: continue
        
        centroid = emb_flat[mask].mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
        
        # 2. Check Background Pixels
        # We only check pixels that are currently marked as background
        bg_indices = np.where(is_background)[0]
        if len(bg_indices) == 0: continue
        
        bg_embs = emb_norm[bg_indices]
        
        # 3. Calculate Distance
        dists = 1.0 - np.dot(bg_embs, centroid_norm)
        
        # 4. Thresholding
        matches = dists < threshold
        
        # 5. Update Labels
        matched_indices = bg_indices[matches]
        new_labels[matched_indices] = cluster_id
        
        # Update background mask (these pixels are no longer background)
        is_background[matched_indices] = False

    return new_labels.reshape(H, W)

# ================= HELPER: HDBSCAN WRAPPER =================
def run_hdbscan(emb_flat, min_cluster_size, min_samples, metric='euclidean'):
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method='eom',
        core_dist_n_jobs=-1
    )
    return clusterer.fit_predict(emb_flat)

# ================= HELPER: VALIDATION =================
def validate_candidates(cluster_labels, embeddings, threshold):
    """Measure distance between cluster centroid and background centroid."""
    H, W, D = embeddings.shape
    emb_flat = embeddings.reshape(-1, D)
    label_flat = cluster_labels.flatten()
    
    unique_clusters = np.unique(label_flat)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    
    if len(unique_clusters) == 0: return []
    
    # Identify Background (Largest Cluster)
    cluster_sizes = [(c, (label_flat == c).sum()) for c in unique_clusters]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    background_cluster = cluster_sizes[0][0]
    
    bg_mask = (label_flat == background_cluster)
    bg_centroid = emb_flat[bg_mask].mean(axis=0)
    bg_norm = bg_centroid / (np.linalg.norm(bg_centroid) + 1e-8)
    
    valid_clusters = []
    for cluster_id, size in cluster_sizes[1:]:
        cand_mask = (label_flat == cluster_id)
        cand_centroid = emb_flat[cand_mask].mean(axis=0)
        cand_norm = cand_centroid / (np.linalg.norm(cand_centroid) + 1e-8)
        
        dist = 1.0 - np.dot(bg_norm, cand_norm)
        
        if dist > threshold:
            valid_clusters.append(cluster_id)
            
    return valid_clusters

# ================= HELPER: MASK GENERATION =================
def labels_to_masks(cluster_labels, valid_clusters, config):
    masks_224 = []
    
    # Limit Max Instances (Top K by size)
    if len(valid_clusters) > config.MAX_INSTANCES:
        label_flat = cluster_labels.flatten()
        valid_sizes = [(c, (label_flat == c).sum()) for c in valid_clusters]
        valid_sizes.sort(key=lambda x: x[1], reverse=True)
        valid_clusters = [c for c, _ in valid_sizes[:config.MAX_INSTANCES]]

    for cluster_id in valid_clusters:
        mask = (cluster_labels == cluster_id).astype(np.float32)
        
        # Patchwise Upsample (33x33 -> 462x462)
        # Using simple nearest neighbor expansion
        mask_462 = np.zeros((462, 462), dtype=np.float32)
        for i in range(33):
            for j in range(33):
                if mask[i, j] > 0.5:
                    y, x = i*14, j*14
                    mask_462[y:min(y+14, 462), x:min(x+14, 462)] = 1.0
        
        # Final Resize for Evaluation/Output
        mask_224 = cv2.resize(mask_462, (config.EVAL_MASK_SIZE, config.EVAL_MASK_SIZE), 
                             interpolation=cv2.INTER_NEAREST)
        masks_224.append(mask_224)
        
    return masks_224

# ================= MAIN EXPORTED FUNCTION =================
def detect_forgery_cascade(embeddings, config):
    """
    Main Pipeline:
    1. Strict Scan
    2. Region Growing (if found)
    3. Anomaly Check (if not found) -> Loose Scan -> Region Growing
    """
    H, W, D = embeddings.shape
    emb_flat = embeddings.reshape(-1, D)
    
    # --- STEP 1: STRICT INFERENCE ---
    labels_strict = run_hdbscan(
        emb_flat, 
        config.HDBSCAN_MIN_CLUSTER_SIZE, 
        config.HDBSCAN_MIN_SAMPLES,
        config.HDBSCAN_METRIC
    ).reshape(H, W)
    
    valid_strict = validate_candidates(labels_strict, embeddings, config.GEOMETRY_THRESHOLD)
    
    # [PATH A] Found with strict params -> Optimize and Return
    if len(valid_strict) > 0:
        # Apply Region Growing (Expansion)
        labels_expanded = expand_clusters(labels_strict, embeddings, valid_strict, config.EXPANSION_THRESHOLD)
        return labels_to_masks(labels_expanded, valid_strict, config)
    
    # --- STEP 2: CASCADE CHECK ---
    # If strict failed, check if image is suspicious
    if not getattr(config, 'ENABLE_CASCADE', False):
        return []

    anomaly = get_anomaly_score(embeddings, getattr(config, 'ANOMALY_PERCENTILE', 95))
    
    # If anomaly is low, trust the strict result (Authentic)
    if anomaly < getattr(config, 'ANOMALY_THRESHOLD', 0.05):
        return []

    # --- STEP 3: LOOSE INFERENCE (FALLBACK) ---
    labels_loose = run_hdbscan(
        emb_flat, 
        getattr(config, 'LOOSE_MIN_CLUSTER_SIZE', 6), 
        getattr(config, 'LOOSE_MIN_SAMPLES', 3),
        config.HDBSCAN_METRIC
    ).reshape(H, W)
    
    valid_loose = validate_candidates(labels_loose, embeddings, getattr(config, 'LOOSE_GEOMETRY_THRESHOLD', 0.25))
    
    # [PATH B] Found with loose params -> Optimize and Return
    if len(valid_loose) > 0:
        labels_expanded = expand_clusters(labels_loose, embeddings, valid_loose, config.EXPANSION_THRESHOLD)
        return labels_to_masks(labels_expanded, valid_loose, config)
        
    # [PATH C] Nothing found even with loose params -> Authentic
    return []

# Legacy/Compatibility wrapper if needed, but detect_forgery_cascade should be used directly
def cluster_embeddings(embeddings, config):
    # This is just for compatibility with old imports if necessary
    pass 
def generate_masks_from_clusters(labels, config, max_instances=5, embeddings=None):
    # This is just for compatibility with old imports if necessary
    pass