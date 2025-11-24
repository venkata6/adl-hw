from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info
import json
import random

def get_edge_distances(kart_bbox, ego_bbox):
    """
    Calculate edge-to-edge distances in horizontal and vertical directions.
    
    Args:
        kart_bbox: [x1, y1, x2, y2] bounding box of the kart
        ego_bbox: [x1, y1, x2, y2] bounding box of the ego car
    
    Returns:
        tuple: (horizontal_distance, vertical_distance, dominant_direction)
        where dominant_direction is "horizontal" or "vertical"
    """
    kart_x1, kart_y1, kart_x2, kart_y2 = kart_bbox
    ego_x1, ego_y1, ego_x2, ego_y2 = ego_bbox
    
    # Horizontal distance (gap between boxes on x-axis)
    if kart_x2 < ego_x1:
        # Kart is completely to the left
        h_distance = ego_x1 - kart_x2
    elif kart_x1 > ego_x2:
        # Kart is completely to the right
        h_distance = kart_x1 - ego_x2
    else:
        # Boxes overlap horizontally
        h_distance = 0
    
    # Vertical distance (gap between boxes on y-axis)
    if kart_y2 < ego_y1:
        # Kart is completely above (in front of)
        v_distance = ego_y1 - kart_y2
    elif kart_y1 > ego_y2:
        # Kart is completely below (behind)
        v_distance = kart_y1 - ego_y2
    else:
        # Boxes overlap vertically
        v_distance = 0
    
    # Determine dominant direction
    dominant_direction = "horizontal" if h_distance > v_distance else "vertical"
    
    return h_distance, v_distance, dominant_direction

def get_non_overlapping_area(kart_bbox, ego_bbox):
    """
    Calculate non-overlapping areas in horizontal and vertical directions.
    
    Args:
        kart_bbox: [x1, y1, x2, y2] bounding box of the kart
        ego_bbox: [x1, y1, x2, y2] bounding box of the ego car
    
    Returns:
        tuple: (horizontal_area, vertical_area, dominant_direction)
        where dominant_direction is "horizontal" or "vertical"
    """
    kart_x1, kart_y1, kart_x2, kart_y2 = kart_bbox
    ego_x1, ego_y1, ego_x2, ego_y2 = ego_bbox
    
    # Calculate overlap region
    overlap_x1 = max(kart_x1, ego_x1)
    overlap_x2 = min(kart_x2, ego_x2)
    overlap_y1 = max(kart_y1, ego_y1)
    overlap_y2 = min(kart_y2, ego_y2)
    
    # Check if there's any overlap
    has_horizontal_overlap = overlap_x2 > overlap_x1
    has_vertical_overlap = overlap_y2 > overlap_y1
    #print(f"{has_horizontal_overlap=},{has_vertical_overlap=}")
    if ( has_horizontal_overlap == True and has_vertical_overlap==True):
        bbox_overlap=True
    else:
        bbox_overlap=False
    # Calculate kart dimensions
    kart_width = kart_x2 - kart_x1
    kart_height = kart_y2 - kart_y1
    
    # Calculate non-overlapping horizontal area
    if has_horizontal_overlap:
        # Subtract overlapping width
        overlap_width = overlap_x2 - overlap_x1
        non_overlap_width = kart_width - overlap_width
    else:
        # No horizontal overlap, full width is non-overlapping
        non_overlap_width = kart_width
    
    horizontal_area = non_overlap_width * kart_height
    
    # Calculate non-overlapping vertical area
    if has_vertical_overlap:
        # Subtract overlapping height
        overlap_height = overlap_y2 - overlap_y1
        non_overlap_height = kart_height - overlap_height
    else:
        # No vertical overlap, full height is non-overlapping
        non_overlap_height = kart_height
    
    vertical_area = kart_width * non_overlap_height
    
    # Determine dominant direction
    dominant_direction = "horizontal" if horizontal_area > vertical_area else "vertical"
    
    return horizontal_area, vertical_area, dominant_direction,bbox_overlap


def generate_captions(info_path: str, view_index: int, img_width: int = 600, img_height: int = 400) -> list:
    """
    Generate all captions for a specific view.
    
    Returns:
        List of caption strings
    """
    # Find corresponding image file
    info_path_obj = Path(info_path)
    base_name = info_path_obj.stem.replace("_info", "")
    image_file = list(info_path_obj.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]
    
    # Extract kart objects using the same method as generate_qa
    karts = extract_kart_objects(str(image_file), info_path, view_index, img_width, img_height)
    
    # Extract track name
    track_name = extract_track_info(info_path)
    
    # Find ego car (the one with is_ego=True)
    ego_kart = None
    for kart in karts:
        if kart.get('is_ego', False):
            ego_kart = kart
            break
    
    # Generate all possible captions
    all_captions = []
    
    # 1. Ego car caption
    if ego_kart:
        all_captions.append(f"{ego_kart['kart_name']} is the ego car.")
    
    # 2. Counting caption
    num_karts = len(karts)
    all_captions.append(f"There are {num_karts} karts in the scene.")
    
    # 3. Track name caption
    all_captions.append(f"The track is {track_name}.")
    
    #4. Relative position captions
    if ego_kart:
        ego_x, ego_y = ego_kart['center']
        ego_bbox = ego_kart['bbox']
        ego_back = ego_bbox[3]  # y2 - bottom of ego kart
        
        for kart in karts:
            # Skip the ego car itself
            if kart.get('is_ego', False):
                continue

            kart_bbox = kart['bbox']
            kart_back = kart_bbox[3]  # y2 - bottom of kart
            kart_x, kart_y = kart['center']

            h_area, v_area, dominant, bbox_overlap = get_non_overlapping_area(kart_bbox, ego_bbox)
            he_area, ve_area, edominant = get_edge_distances(kart_bbox, ego_bbox)
            #print(f"{he_area=},{ve_area=},{edominant=}")

             # Calculate signed differences
            dx = abs(kart_x - ego_x)  # Positive = right, Negative = left
            dy = abs(ego_y - kart_y)  # Positive = in front (higher in image)

            #print(f"{dy=},{dx=},{h_area=},{v_area=},{dominant=},{kart['kart_name']=} ")

            # Calculate distances
            h_distance = abs(kart_x - ego_x)
            v_distance = abs(kart_y - ego_y)
            
            # Determine horizontal position
            if kart_x < ego_x:
                horizontal = "left of"
            else:
                horizontal = "right of"
            
            # Determine vertical position (lower y = in front in image space)
            if kart_y < ego_y:
                vertical = "in front of"
            else:
                vertical = "behind"
            
            if kart_back > ego_back:
                vertical = "behind"
                # Recalculate v_distance using back edges
                v_distance = abs(kart_back - ego_back)
            
            # Choose the dimension with greater distance
            SLOPE_RATIO = 0.1 # Start with 0.6 for a moderate cone.
            if (kart_back > ego_back):
                position = "behind"
            else:
                position = horizontal
            if ( v_distance > h_distance * SLOPE_RATIO):
                position = vertical
            elif h_distance > v_distance or (dominant == 'horizontal') \
                or (edominant == 'horizontal') :
                position = horizontal 
            else:
                position = horizontal
            
            if kart_back > ego_back:
                #print(f"hello {position}")
                position = position
            else:
                position = horizontal
            #print(f"{horizontal=},{h_distance=},{v_distance=}")
            #print(f"{kart_back=},{ego_back=}")
                 
            if ( dy < dx and (dominant == 'horizontal')):
                if (h_distance > v_distance ):
                    position = horizontal
                else:
                    position = vertical
            
            if ( ego_back - kart_back >= 15 and (v_area == 0)):
                position = vertical
            # if ( bbox_overlap == False ):
            #     position = vertical
            # if ( he_area == 0 and ve_area == 0 and edominant == 'vertical' ):
            #     if (h_area - v_area > 40):
            #         position = horizontal
            #     else:    
            #         position = vertical
                

# Replace the relative position caption generation loop in generate_captions()
# Find this section and replace it:             

            all_captions.append(f"{kart['kart_name']} is {position} the ego car.")        
    
    return all_captions



def check_caption(info_file: str, view_index: int):
    captions = generate_captions(info_file, view_index)

    print("\nGenerated Captions:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i}. {caption}")
    print("-" * 50)
    print(f"Total captions: {len(captions)}")

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_dataset(
    data_dir: str = "data/train",
    max_samples: int = None
):
    """
    Generate caption dataset from all info files in a directory.
    Each info file gets its own output JSON file.
    
    Args:
        data_dir: Directory containing the info files (relative to project root)
        max_samples: Maximum number of info files to process (None for all)
    """
    import json
    from pathlib import Path
    from tqdm import tqdm
    
    data_path = Path(data_dir)
    info_files = sorted(data_path.glob("*_info.json"))
    
    if max_samples:
        info_files = info_files[:max_samples]
    
    total_samples = 0
    
    for info_file in tqdm(info_files, desc="Generating captions"):
        with open(info_file, 'r') as f:
            info = json.load(f)
        #print(f"{info_file=}")
        num_views = len(info['detections'])
        base_name = info_file.stem.replace("_info", "")
        
        # Collect all samples for this info file
        file_samples = []
        
        for view_index in range(num_views):
            # Find corresponding image file
            image_file = list(data_path.glob(f"{base_name}_{view_index:02d}_im.jpg"))
            if not image_file:
                continue
            
            # Generate all captions for this view
            captions = generate_captions(str(info_file), view_index)
            
            # Clean up the path (remove leading 'data/')
            path = str(image_file[0])
            cleaned_path = path.replace("data/", "", 1)
            
            # Create one sample per caption
            for caption in captions:
                file_samples.append({
                    "image_file": cleaned_path,
                    "caption": caption
                })
        
        # Save to JSON with the same base name as the info file
        output_file = data_path / f"{base_name}_captions.json"
        
        with open(output_file, 'w') as f:
            json.dump(file_samples, f, indent=2)
        
        total_samples += len(file_samples)
    
    print(f"\nGenerated {len(info_files)} caption files")
    print(f"Total samples: {total_samples}")


def generate_all_datasets():
    """
    Generate caption datasets for train, valid, and test splits.
    """
    print("Generating training dataset...")
    generate_dataset(data_dir="data/train")
    
    print("\nGenerating validation dataset...")
    generate_dataset(data_dir="data/valid")
    
    print("\nGenerating test dataset...")
    generate_dataset(data_dir="data/test")
    
    print("\nAll datasets generated!")


def stats(caption_file: str = "data/train/00000_captions.json"):
    """
    Print statistics about a generated caption file.
    """
    import json
    
    with open(caption_file, 'r') as f:
        samples = json.load(f)
    
    print(f"\nDataset Statistics for {caption_file}:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Unique images: {len(set(s['image_file'] for s in samples))}")
    
    # Count caption types
    caption_types = {
        'ego': 0,
        'count': 0,
        'track': 0,
        'position': 0
    }
    
    for sample in samples:
        caption = sample['caption']
        if 'is the ego car' in caption:
            caption_types['ego'] += 1
        elif 'There are' in caption:
            caption_types['count'] += 1
        elif 'The track is' in caption:
            caption_types['track'] += 1
        else:
            caption_types['position'] += 1
    
    print(f"\nCaption type distribution:")
    print(f"  Ego car: {caption_types['ego']}")
    print(f"  Count: {caption_types['count']}")
    print(f"  Track name: {caption_types['track']}")
    print(f"  Position: {caption_types['position']}")
    
    # Show some examples
    print("\nSample entries:")
    for i, sample in enumerate(samples[:5]):
        print(f"\n  Sample {i+1}:")
        print(f"    Image: {sample['image_file']}")
        print(f"    Caption: {sample['caption']}")


"""
Usage Examples:

1. Visualize captions for a specific file and view:
   python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0

2. Generate captions for a single dataset split:
   python -m homework.generate_captions generate --data_dir data/train
   python -m homework.generate_captions generate --data_dir data/train --max_samples 100

3. Generate captions for all dataset splits (train, valid, test):
   python -m homework.generate_captions generate_all

4. View statistics about a generated caption file:
   python -m homework.generate_captions stats --caption_file data/train/00000_captions.json
"""


def main():
    fire.Fire({
        "check": check_caption,
        "generate": generate_dataset,
        "generate_all": generate_all_datasets,
        "stats": stats
    })


if __name__ == "__main__":
    main()