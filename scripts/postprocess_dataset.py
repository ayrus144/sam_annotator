"""
Postprocess data dir - with classes.json, images & labels (colored seg. masks) sub-dirs
Creates "annotation" dir to save all segmented masks as class_ids instead of colors
"""
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
import tomllib
import json

import numpy as np
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
import cv2
import shutil
from tqdm import tqdm

# Convert hex keys to RGB tuple keys
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def crop_img(img_path, crop_px):
    # Crop bottom pixels
    img = Image.open(img_path).convert('RGB')
    width, height =img.size
    cropped_img = img.crop((0, 0, width, height - crop_px))
    return np.array(cropped_img)  # (H, W, 3)

def correct_label_img(label_rgb, kd_tree):
    """Label correction due to JPEG loss compression"""
    pixels = label_rgb.reshape(-1, 3)

    # Query the KDTree and map for nearest palette color
    _, indices = kd_tree.query(pixels)
    mapped_pixels = rgb_colors[indices]

    # Reshape back to the original image shape
    mapped_img = mapped_pixels.reshape(label_rgb.shape)
    return mapped_img.astype(np.uint8)

# Apply cv2 morphology to boolean class mask
def morph_open_close(channel_img):
    """ Cleans images with cv2 morphology
    Opening: remove noise from image background
    Closing: fill holes on image foreground """

    k_size = 5
    kernel = np.ones((k_size, k_size), np.uint8)
    channel_img = cv2.morphologyEx(channel_img, cv2.MORPH_OPEN, kernel)
    channel_img = cv2.morphologyEx(channel_img, cv2.MORPH_CLOSE, kernel)

    return channel_img

# Convert color mask to class ID mask
def convert_labels_to_anns(label_rgb, color2id):
    """ Color-corrected label_img are cleaned using cv2 followed by
    creating the class_mask of the annotations to use for training """
    h, w, _ = label_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    color_mask = np.zeros_like(label_rgb, dtype=np.uint8)

    for rgb_color, class_id in color2id.items():
        matches = np.all(label_rgb == rgb_color, axis=-1)
        # print(class_id, rgb_color, np.sum(matches))

        cv_mask = matches.astype(np.uint8) * 255 # cv2 compatible mask
        clean_cv_mask = morph_open_close(cv_mask)
        class_mask[clean_cv_mask == 255] = class_id # int
        color_mask[clean_cv_mask == 255] = np.array(rgb_color).astype(np.uint8)

    return class_mask, color_mask


if __name__ == "__main__":
    visualize_ann = True
    crop_px = 135 # crop bottom pixels to remove timestamps

    # Load and check dataset_dir
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    data_path = Path(config["paths"]["data"])
    images_path = data_path / "images"
    assert (
        images_path.exists()
    ), "Data path must contain 'images' folder with all source data images"
    labels_path = data_path / "labels"
    assert (
        labels_path.exists()
    ), "Data path must contain 'labels' folder with annotations for all images"
    used_class_path = data_path / "classes.json"

    # Create img and ann dirs (Optional ann_vis dir)
    dirs = ['img', 'ann']
    if visualize_ann:
        dirs.append('ann_vis')
    for base_dir in dirs:
        for sub_dir in ['train', 'val']:
            path = data_path / 'semanticData' / base_dir / sub_dir
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

    # Create metainfo.yaml
    class_path = Path(__file__).parent.parent / "example_dataset" / "classes.json"
    meta_path = data_path / 'semanticData' / "metainfo.yaml"
    with open(class_path, "r") as f:
        classes = json.loads("".join(f.readlines()))["classes"]
    class_names = [c["name"] for c in classes]
    meta_conf = OmegaConf.create({'class_names' : class_names})
    OmegaConf.save(meta_conf, meta_path)

    # Prepare for postprocessing
    print("Preparing for processing...")
    ids = [c["id"] for c in classes]
    colors = [c["color"] for c in classes]

    if used_class_path.exists():
        print('Using json from data path')
        with open(used_class_path, "r") as f:
            used_classes = json.loads("".join(f.readlines()))["classes"]
        used_class_names = [c["name"] for c in used_classes]
        assert(
            used_class_names == class_names
        ), "Same class names/order were not used during annotations"
        colors = [c["color"] for c in used_classes]

    # Convert hex to rgb
    color2id = {hex_to_rgb(k): v for k, v in zip(colors, ids)}
    color2id[(0, 0, 0)] = 0  # Black → class 0
    rgb_colors = list(color2id.keys())

    # Build KD-Tree for fast nearest neighbor search
    rgb_colors = np.array(rgb_colors)
    kd_tree = KDTree(rgb_colors)

    # Postprocessing labels -> annotations
    def get_corrected_label_img(img_stem):
        filename = f"{img_stem}.png"
        label_path = labels_path / filename

        # Image correction and Check if useful
        corrected_label_img = crop_img(label_path, crop_px)
        if np.any(corrected_label_img):
            is_useful = True  # valid for labels saved in .png, else use below lines
            # corrected_label_img = correct_label_img(corrected_label_img, kd_tree)
            # is_useful = np.any(corrected_label_img)  # check after jpeg corrections
            return is_useful, corrected_label_img
        else:
            # complete image is background, image not useful for training
            return False, None


    def postprocess_img(img_stem, suffix, dataset):
        image_path = images_path / f"{img_stem}{suffix}"
        img_path = data_path / "semanticData" / "img" / dataset / f"{img_stem}.png"
        ann_path = data_path / "semanticData" / "ann" / dataset / f"{img_stem}.png"

        # Original image
        cropped_img = crop_img(image_path, crop_px)
        Image.fromarray(cropped_img, mode = 'RGB').save(img_path)

        # Annotation mask
        _, corrected_label_img = get_corrected_label_img(img_stem)
        class_mask, cleaned_label_img = convert_labels_to_anns(corrected_label_img, color2id)
        Image.fromarray(class_mask, mode='L').save(ann_path)

        # Annotation image
        if visualize_ann:
            ann_vis_path = data_path / "semanticData" / "ann_vis" / dataset / f"{img_stem}.png"
            Image.fromarray(cleaned_label_img, mode='RGB').save(ann_vis_path)

    # Get list of useful images
    print("Filtering useful images...")
    # list(labels) <= list(images)
    img_stems = [path.stem for path in sorted(labels_path.glob("*.png"))]
    img_info = [(path.stem, path.suffix) for path in sorted(images_path.iterdir()) if path.stem in img_stems]
    useful_info = []
    for img_stem, suffix in tqdm(img_info):
        is_useful, _ = get_corrected_label_img(img_stem)
        if is_useful:
            useful_info.append((img_stem, suffix))

    # 80-20 split for train-valid and post process
    print("Creating train-valid splits...")
    train_info, valid_info = train_test_split(useful_info, test_size=0.2, random_state=42)

    print("Processing train split...")
    for img_stem, suffix in tqdm(train_info):
        postprocess_img(img_stem, suffix, 'train')

    print("Processing valid split...")
    for img_stem, suffix in tqdm(valid_info):
        postprocess_img(img_stem, suffix, 'val')

    print("Processing complete.")

