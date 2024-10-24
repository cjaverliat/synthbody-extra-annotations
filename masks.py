import os.path as osp
import cv2
import numpy as np

MASKS_NAME = [
    "beard",
    "eyebrows",
    "eyelashes",
    "facewear",
    "glasses",
    "hair",
    "headwear",
]


def _format_segm_mask_path(dataset_dir, identity, frame, mask_name):
    return osp.join(dataset_dir, f"segm_{mask_name}_{identity:07d}_{frame:03d}.png")


def _combine_masks(dataset_dir, identity: int, frame: int):
    parts_mask = (
        cv2.imread(
            _format_segm_mask_path(dataset_dir, identity, frame, "parts"),
            cv2.IMREAD_GRAYSCALE,
        )
        > 0
    )

    final_mask = np.zeros_like(parts_mask, dtype=bool)
    final_mask[parts_mask] = True

    for mask_name in MASKS_NAME:
        mask_path = _format_segm_mask_path(dataset_dir, identity, frame, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 100
        final_mask[mask] = True
    return final_mask


def _compute_bbox_from_mask(mask):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return x, y, w, h


def generate_mask(dataset_dir, identity, frame):
    mask = _combine_masks(dataset_dir, identity, frame)
    return mask


def generate_bbox_annotation(mask: np.ndarray):
    return _compute_bbox_from_mask(mask)
