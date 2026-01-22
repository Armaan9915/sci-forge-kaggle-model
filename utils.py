
import json
import numba
import numpy as np
from numba import types
import numpy.typing as npt
import pandas as pd
import scipy.optimize

class ParticipantVisibleError(Exception):
    pass

@numba.jit(nopython=True)
def _rle_encode_jit(x: npt.NDArray, fg_val: int = 1) -> list[int]:
    """Numba-jitted RLE encoder."""
    dots = np.where(x.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def rle_encode(masks: list[npt.NDArray], fg_val: int = 1) -> str:
    return ';'.join([json.dumps(_rle_encode_jit(x, fg_val)) for x in masks])

@numba.njit
def _rle_decode_jit(mask_rle: npt.NDArray, height: int, width: int) -> npt.NDArray:
    if len(mask_rle) % 2 != 0:
        raise ValueError('One or more rows has an odd number of values.')
    starts, lengths = mask_rle[0::2], mask_rle[1::2]
    starts -= 1
    ends = starts + lengths
    for i in range(len(starts) - 1):
        if ends[i] > starts[i + 1]:
            raise ValueError('Pixels must not be overlapping.')
    img = np.zeros(height * width, dtype=np.bool_)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img

def rle_decode(mask_rle: str, shape: tuple[int, int]) -> npt.NDArray:
    mask_rle = json.loads(mask_rle)
    mask_rle = np.asarray(mask_rle, dtype=np.int32)
    try:
        return _rle_decode_jit(mask_rle, shape[0], shape[1]).reshape(shape, order='F')
    except ValueError as e:
        raise ParticipantVisibleError(str(e)) from e

def calculate_f1_score(pred_mask: npt.NDArray, gt_mask: npt.NDArray):
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0

def oF1_score(pred_masks: list[npt.NDArray], gt_masks: list[npt.NDArray]):
    """Calculate optimal F1 score using Hungarian algorithm."""
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)
    f1_matrix = np.zeros((num_pred, num_gt))
    
    for i in range(num_pred):
        for j in range(num_gt):
            f1_matrix[i, j] = calculate_f1_score(pred_masks[i], gt_masks[j])
            
    if num_pred < num_gt:
        f1_matrix = np.vstack((f1_matrix, np.zeros((num_gt - num_pred, num_gt))))
        
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-f1_matrix)
    excess_penalty = num_gt / max(num_pred, num_gt)
    return np.mean(f1_matrix[row_ind, col_ind]) * excess_penalty