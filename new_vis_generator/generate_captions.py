import cv2
import os
import glob
import csv
import json
import tempfile
import time
import random
from pathlib import Path
from tqdm import tqdm

import new_vis_gen

DATA_DIR = r'D:\FYP_MTL_GI_VQA\data'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

# Load environment variables from a local .env file if present
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, val = line.strip().split('=', 1)
                os.environ[key.strip()] = val.strip()

def load_valid_instrument_ids():
    valid_ids = set()
    csv_path = r'D:\FYP_MTL_GI_VQA\data\combined\instruments_mask_phrases_v2.csv'
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                valid_ids.add(row['img_id'])
    return valid_ids

VALID_INSTRUMENT_IDS = load_valid_instrument_ids()

def get_image_info(mask_path):
    p = Path(mask_path)
    normalized_path = str(p.as_posix())
    
    img_id = None
    roi_type = None
    mask_id = p.name
    should_process = True
    
    if 'instruments_masks' in normalized_path:
        img_id = p.stem
        roi_type = 'instrument'
        if img_id not in VALID_INSTRUMENT_IDS:
            should_process = False
            
    elif 'pseudo_masks' in normalized_path:
        img_id = p.name.split('_')[0]
        if 'cecum' in p.name:
            roi_type = 'cecum'
        elif 'z-line' in p.name:
            roi_type = 'z-line'
        else:
            should_process = False
            
    elif 'gradcam_masks' in normalized_path:
        class_name = p.parent.parent.name
        img_id = p.parent.name
        mask_id = f"{class_name}_{img_id}_{p.name}" # Use a unique string format since there could be multiple masks across images
        
        if class_name == 'ulcerative_colitis':
            roi_type = 'ulcerative-colitis'
        elif class_name == 'oesophagitis':
            roi_type = 'oesophagitis'
        else:
            should_process = False
            
        if should_process:
            bbox_json_path = p.parent / 'bbox_data.json'
            if bbox_json_path.exists():
                with open(bbox_json_path, 'r') as f:
                    bbox_data = json.load(f)
                    if bbox_data.get('prediction', 0) <= 0.80:
                        should_process = False
            else:
                should_process = False
                
    elif 'polyp_masks' in normalized_path:
        img_id = p.stem
        roi_type = 'polyp'

    if img_id is None or roi_type is None:
        should_process = False

    return should_process, img_id, mask_id, roi_type

def create_overlay(mask_path, orig_img_path, output_path):
    img = cv2.imread(orig_img_path)
    if img is None:
        return False
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
        
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    overlay = img.copy()
    
    for c in contours:
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), thickness=2)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
        
    cv2.imwrite(output_path, overlay)
    return True

def main():
    masks = []
    masks.extend(glob.glob(os.path.join(DATA_DIR, "polyp_masks", "*.jpg")))
    masks.extend(glob.glob(os.path.join(DATA_DIR, "instruments_masks", "*.jpg")))
    masks.extend(glob.glob(os.path.join(DATA_DIR, "pseudo_masks", "*.jpg")))
    masks.extend(glob.glob(os.path.join(DATA_DIR, "gradcam_masks", "*", "*", "mask.png")))
    
    random.shuffle(masks)
    
    csv_out_path = r'D:\FYP_MTL_GI_VQA\generated_captions.csv'
    temp_dir = tempfile.mkdtemp()
    
    # Track statistics
    processed_count = 0
    
    with open(csv_out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['img_id', 'mask_id', 'roi_type', 'vis_des'])
        
        for mask_path in tqdm(masks, desc="Generating Captions"):
            should_process, img_id, mask_id, roi_type = get_image_info(mask_path)
            
            if not should_process:
                continue
                
            orig_img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
            if not os.path.exists(orig_img_path):
                print(f"Skipping {img_id}: Original image not found.")
                continue
                
            temp_overlay_path = os.path.join(temp_dir, f"{img_id}_{roi_type}_overlay.jpg")
            
            if not create_overlay(mask_path, orig_img_path, temp_overlay_path):
                print(f"Skipping {img_id}: Failed to compute CV2 overlay.")
                continue
                
            try:
                caption = new_vis_gen.generate_caption(temp_overlay_path, roi_type)
                print(caption)
                writer.writerow([img_id, mask_id, roi_type, caption])
                f.flush()
                processed_count += 1
                
                # Rate Limiting: Respect maximum 30 RPM limit (3 seconds = 20 Requests Per Minute max target)
                time.sleep(3.0)
                
            except Exception as e:
                print(f"Failed to generate caption for {img_id}_{mask_id}: {e}")
                
            if os.path.exists(temp_overlay_path):
                os.remove(temp_overlay_path)
                
    print(f"\nGenerated {processed_count} total captions.")
    print(f"Dataset successfully appended to: {csv_out_path}")

if __name__ == "__main__":
    main()
