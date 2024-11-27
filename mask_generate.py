import os
import json
import cv2
import numpy as np

def read_json_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    data = []
    for file in json_files:
        with open(os.path.join(directory, file), 'r') as f:
            data.append((file, json.load(f)))
    return data

def create_sep_mask_from_segmentations(image_shape, segmentations):
    mask = np.ones(image_shape[:2], dtype=np.uint8) * 255  # Create a white mask
    for i, segmentation in enumerate(segmentations):
        for polygon in segmentation:
            polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
            color = (i + 1) * 50  # Different color for each item
            cv2.fillPoly(mask, [polygon], color)
    return mask

def create_zero_mask_from_segmentations(image_shape, segmentations):
    mask = np.ones(image_shape[:2], dtype=np.uint8) * 255  # Create a white mask
    for segmentation in segmentations:
        for polygon in segmentation:
            polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [polygon], 0)  # Fill the polygon with 0
    return mask

# /home/work/.daehyeung/IP-Adapter/street_tryon/validation
ROOT_DIR = "/home/work/.daehyeung/IP-Adapter/street_tryon/validation/"
IMAGE_ROOT_DIR = "/home/work/.daehyeung/$DATA/deepfashion2/validation/"

# input file path
JSON_PATH = ROOT_DIR + "anno"
IMAGE_PATH = IMAGE_ROOT_DIR + "image"

# output file path
SEP_MASK_PATH = ROOT_DIR + "sep_mask"
ZERO_MASK_PATH = ROOT_DIR + "zero_mask"
MASKED_IMAGE_PATH = ROOT_DIR + "masked_image"

# Create directories if they don't exist
if not os.path.exists(SEP_MASK_PATH):
    os.makedirs(SEP_MASK_PATH)

if not os.path.exists(ZERO_MASK_PATH):
    os.makedirs(ZERO_MASK_PATH)

if not os.path.exists(MASKED_IMAGE_PATH):
    os.makedirs(MASKED_IMAGE_PATH)

json_data = read_json_files(JSON_PATH)

for json_file, data in json_data:
    segmentations = []
    item_keys = []
    pair_ids = []
    for key, value in data.items():
        if key.startswith('item') and 'segmentation' in value:
            segmentations.append(value['segmentation'])
            item_keys.append(key)
            pair_ids.append(data.get('pair_id', 0))

    if not segmentations:
        print(f"No segmentations found in {json_file}.")
        continue
    
    image_name = json_file.replace('.json', '.jpg')  # Assuming image files are .jpg
    image_path = os.path.join(IMAGE_PATH, image_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Image {image_name} not found.")
        continue
    
    if len(item_keys) == 1:
        item_key = item_keys[0]
        segmentation = segmentations[0]
        pair_id = pair_ids[0]
        
        sep_mask = create_sep_mask_from_segmentations(image.shape, [segmentation])
        sep_mask_path = os.path.join(SEP_MASK_PATH, f"{json_file.replace('.json', '')}_{item_key}.png")
        cv2.imwrite(sep_mask_path, sep_mask)

        zero_mask = create_zero_mask_from_segmentations(image.shape, [segmentation])
        zero_mask_path = os.path.join(ZERO_MASK_PATH, f"{json_file.replace('.json', '')}_{item_key}.png")
        cv2.imwrite(zero_mask_path, zero_mask)

        # Create new image with mask applied
        masked_image = cv2.bitwise_and(image, image, mask=zero_mask)
        masked_image_path = os.path.join(MASKED_IMAGE_PATH, f"{json_file.replace('.json', '')}_{item_key}.png")
        cv2.imwrite(masked_image_path, masked_image)
    else:
        # Create separate masks for each item
        for segmentation, item_key, pair_id in zip(segmentations, item_keys, pair_ids):
            sep_mask = create_sep_mask_from_segmentations(image.shape, [segmentation])
            sep_mask_path = os.path.join(SEP_MASK_PATH, f"{json_file.replace('.json', '')}_{item_key}.png")
            cv2.imwrite(sep_mask_path, sep_mask)

            zero_mask = create_zero_mask_from_segmentations(image.shape, [segmentation])
            zero_mask_path = os.path.join(ZERO_MASK_PATH, f"{json_file.replace('.json', '')}_{item_key}.png")
            cv2.imwrite(zero_mask_path, zero_mask)

            # Create new image with mask applied
            masked_image = cv2.bitwise_and(image, image, mask=zero_mask)
            masked_image_path = os.path.join(MASKED_IMAGE_PATH, f"{json_file.replace('.json', '')}_{item_key}.png")
            cv2.imwrite(masked_image_path, masked_image)
        
        # Create a combined mask for all segmentations
        combined_sep_mask = create_sep_mask_from_segmentations(image.shape, segmentations)
        combined_sep_mask_path = os.path.join(SEP_MASK_PATH, f"{json_file.replace('.json', '')}.png")
        cv2.imwrite(combined_sep_mask_path, combined_sep_mask)

        combined_zero_mask = create_zero_mask_from_segmentations(image.shape, segmentations)
        combined_zero_mask_path = os.path.join(ZERO_MASK_PATH, f"{json_file.replace('.json', '')}.png")
        cv2.imwrite(combined_zero_mask_path, combined_zero_mask)

        # Create new image with combined mask applied
        combined_masked_image = cv2.bitwise_and(image, image, mask=combined_zero_mask)
        combined_masked_image_path = os.path.join(MASKED_IMAGE_PATH, f"{json_file.replace('.json', '')}.png")
        cv2.imwrite(combined_masked_image_path, combined_masked_image)