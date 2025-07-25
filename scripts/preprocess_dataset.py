"""
Predicts segmentation masks via SAM model
for all images in given dataset.
Saves mask as 8-bit grayscale .JPG
"""

from pathlib import Path
import time
from PIL import Image
import tomllib

import numpy as np
import torch
import cv2
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Run pip show SAM-2 & pip install git+https://github.com/facebookresearch/sam2.git if needed
# (has compatibility issues when running locally, works only on jupyter notebooks)
# (see automatic_mask_generator_example.ipynb - taken from facebook)
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# from sam2.build_sam import build_sam2


def make_annotator(weights_path: str, device: str) -> SamAutomaticMaskGenerator:
    model_type = "vit_h"
    print(f"Loading {model_type} on {device} device")
    t1 = time.perf_counter()
    sam = sam_model_registry[model_type](weights_path)
    # if device == "cuda":
    #     # use bfloat16 for the entire notebook
    #     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    #     if torch.cuda.get_device_properties(0).major >= 8:
    #         torch.backends.cuda.matmul.allow_tf32 = True
    #         torch.backends.cudnn.allow_tf32 = True
    # sam2 = build_sam2(
    #     config_file="configs/sam2/sam2_hiera_l.yaml",
    #     checkpoint=weights_path,
    #     device = device,
    #     apply_postprocesssing=False
    # )
    t2 = time.perf_counter()
    sam.to(device)
    t3 = time.perf_counter()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000, # needs opencv for postprocessing
    )
    # mask_generator = SAM2AutomaticMaskGenerator(
    #     model=sam2,
    #     points_per_side=32,
    #     pred_iou_thresh=0.7,
    #     stability_score_thresh=0.92,
    #     stability_score_offset=0.7,
    #     crop_n_layers=1,
    #     box_nms_thresh=0.7,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=1000,
    #     use_m2m=True,
    # )
    print(f"Load weights: {(t2-t1):.3f}s\nMove to {device}: {(t3-t2):.3f}s")
    return mask_generator

def replace_surrounded_masks_opencv(label_img, area_thresh=50):
    label = label_img.copy()
    unique_labels, counts = np.unique(label, return_counts=True)

    # Sort by counts in ascending order
    sorted_indices = np.argsort(counts)
    sorted_labels = unique_labels[sorted_indices]
    unique_labels = sorted_labels[sorted_labels != 0]  # exclude background

    for lbl in unique_labels:
        # Create binary mask for the current label
        mask = (label == lbl).astype(np.uint8)

        # Calculate area
        area = cv2.countNonZero(mask)
        if area >= area_thresh:
            continue  # Skip large masks

        # Find contours (external only)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # For simplicity, just take the first (should be only one for clean labels)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)

        # Check the border pixels around the bounding box (one-pixel border)
        border_labels = []

        # Top and bottom rows
        if y > 0:
            border_labels += list(np.unique(label[y - 1, x:x + w]))
        if y + h < label.shape[0]:
            border_labels += list(np.unique(label[y + h, x:x + w]))

        # Left and right columns
        if x > 0:
            border_labels += list(np.unique(label[y:y + h, x - 1]))
        if x + w < label.shape[1]:
            border_labels += list(np.unique(label[y:y + h, x + w]))

        # Remove current label and background (0)
        border_labels = [b for b in border_labels if b != lbl and b != 0]

        # If all surrounding pixels have the same label, replace
        if len(set(border_labels)) == 1:
            new_label = border_labels[0]
            label[mask == 1] = new_label
            # print(f"Replaced label {lbl} with {new_label}")

    return label

if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    data_path = Path(config["paths"]["data"])
    sam_weights = Path(config["paths"]["sam_weights"])
    images_path = data_path / "images"
    assert (
        images_path.exists()
    ), "Data path must contain 'images' folder with all source data images"
    sam_path = data_path / "sam"
    sam_path.mkdir(exist_ok=True)
    sam = make_annotator(str(sam_weights), config["device"])

    max_masks = 0

    img_stems = [path.stem for path in sorted(images_path.iterdir())]
    for stem in tqdm(img_stems):
        filename = f"{stem}.jpg"
        img_path = images_path / filename
        out_path = sam_path / filename
        img = Image.open(img_path)
        img = np.array(img)
        t1 = time.perf_counter()
        masks = sam.generate(img)
        t2 = time.perf_counter()
        sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        label = np.zeros(sorted_masks[0]["segmentation"].shape, dtype=np.uint8)
        for i, sm in enumerate(sorted_masks):
            m = sm["segmentation"]
            label[m] = i + 1
        max_masks = max(max_masks, np.max(label))
        new_label = replace_surrounded_masks_opencv(label, 7000)
        tqdm.write(f"file: {filename} | SAM: {(t2-t1):.3f}s | labels: {np.max(label)} -> {len(np.unique(new_label))}")
        label_img = Image.fromarray(label, mode="L")
        label_img.save(out_path)
