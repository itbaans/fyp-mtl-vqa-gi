import cv2
import os
import glob
import numpy as np

DATA_DIR = r'D:\FYP_MTL_GI_VQA\data'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

def get_original_image_path(mask_path):
    from pathlib import Path
    p = Path(mask_path)
    img_id = None
    
    normalized_path = str(p.as_posix())
    
    if 'polyp_masks' in normalized_path or 'instruments_masks' in normalized_path:
        img_id = p.stem
    elif 'pseudo_masks' in normalized_path:
        img_id = p.name.split('_')[0]
    elif 'gradcam_masks' in normalized_path:
        img_id = p.parent.name
        
    if img_id:
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        if os.path.exists(img_path):
            return img_path
    return None

def process_mask_and_image(mask_path, output_path=None):
    orig_img_path = get_original_image_path(mask_path)
    if not orig_img_path:
        print(f"Original image not found for mask: {mask_path}")
        return
        
    img = cv2.imread(orig_img_path)
    if img is None:
        print(f"Failed to read image: {orig_img_path}")
        return
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to read mask: {mask_path}")
        return
        
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    overlay = img.copy()
    
    for c in contours:
        # cv2 uses BGR. (0,255,0) is Green, (0,0,255) is Red
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), thickness=2)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
        
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay)
        print(f"Saved to {output_path}")

example_polyp = glob.glob(os.path.join(DATA_DIR, "polyp_masks", "*.jpg"))
example_instrument = glob.glob(os.path.join(DATA_DIR, "instruments_masks", "*.jpg"))
example_pseudo = glob.glob(os.path.join(DATA_DIR, "pseudo_masks", "*.jpg"))
example_gradcam = glob.glob(os.path.join(DATA_DIR, "gradcam_masks", "*", "*", "mask.png"))

if example_polyp: process_mask_and_image(example_polyp[0], "test_out_polyp.jpg")
if example_instrument: process_mask_and_image(example_instrument[0], "test_out_instrument.jpg")
if example_pseudo: process_mask_and_image(example_pseudo[0], "test_out_pseudo.jpg")
if example_gradcam: process_mask_and_image(example_gradcam[0], "test_out_gradcam.jpg")
