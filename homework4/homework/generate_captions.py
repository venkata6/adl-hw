from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info
import json
import random

# # Helper function to determine position using the Cone of Vision logic
# def get_cone_of_vision_position(kart_center, ego_center, horizontal_threshold=0):
#     #print("in get_cone_of_vision_position ")
#     k_x, k_y = kart_center
#     e_x, e_y = ego_center

#     # A simple way to define the cone is by using a slope/ratio.
#     # SLOPE_RATIO = how many pixels horizontally (dx) equals 1 pixel vertically (dy).
#     # A smaller ratio creates a narrower cone (stricter "Front" definition).
#     # We use 0.5: for every 100 pixels forward (dy), the cone is 50 pixels wide (dx).
#     SLOPE_RATIO = 0.5
    
#     # 1. Calculate relative differences
#     dx = k_x - e_x # Lateral displacement
#     dy = e_y - k_y # Vertical displacement (positive means object is HIGHER than ego car, i.e., "forward")

#     # 2. Horizontal Label (Left/Right)
#     if dx < -horizontal_threshold:
#         horizontal = "left"
#     elif dx > horizontal_threshold:
#         horizontal = "right"
#     else:
#         horizontal = "aligned"
        
#     # 3. Vertical Label (Front/Behind/Side)

#     # Calculate the maximum allowed horizontal distance to still be considered "Front"
#     # This is the width of the cone at the target's vertical level.
#     cone_width_at_y = dy * SLOPE_RATIO
    
#     # Check if the target is significantly behind the ego (should be rare/impossible in FPV)
#     if dy < -100: # Significantly below the ego car center (y-value is greater)
#         vertical = "behind"
#         return f"far behind and to the {horizontal} of"

#     # Check the "Front" cone
#     elif dy > 20: # Must be at least 20 pixels 'in front' (up) to check the cone
#         if abs(dx) < cone_width_at_y:
#             # If the side distance is smaller than the cone width, it's "in front"
#             vertical = "in front of"
#         else:
#             # If it's outside the cone, it's "to the side"
#             vertical = "alongside" 
#     else:
#         # Close to the same Y-level (or too small a difference to tell)
#         vertical = "alongside" 
        
#     # 4. Combine and return the most salient dimension

#     # If it's clearly defined as 'in front of', we prioritize that.
#     if vertical == "in front of":
#         if horizontal != "aligned":
#             return f"is slightly to the {horizontal} and in front of"
#         return "directly in front of"
    
#     # If it's not "in front of", we prioritize the horizontal position (Left/Right).
#     elif horizontal != "aligned":
#         # The answer for the 'Konqi' problem ("left and not front") would fall here.
#         return f"{horizontal} of"
    
#     return "at the same position as"

# def get_cone_of_vision_position(kart_center, ego_center, horizontal_threshold=0):
#     k_x, k_y = kart_center
#     e_x, e_y = ego_center
    
#     # Calculate relative differences
#     dx = k_x - e_x  # Lateral displacement
#     dy = e_y - k_y  # Vertical displacement (positive = kart is higher/forward)
    
#     # Calculate distances
#     h_distance = abs(dx)
#     v_distance = abs(dy)
    
#     # Determine horizontal position
#     if dx < -horizontal_threshold:
#         horizontal = "to the left of"
#     elif dx > horizontal_threshold:
#         horizontal = "to the right of"
#     else:
#         horizontal = "aligned with"
    
#     # Determine vertical position with better thresholds
#     FORWARD_THRESHOLD = 20   # Must be at least 20px up to be "in front"
#     BEHIND_THRESHOLD = 20    # Must be at least 20px down to be "behind"
    
#     if dy > FORWARD_THRESHOLD:
#         vertical = "in front of"
#     elif dy < -BEHIND_THRESHOLD:  # Changed from -100 to -20
#         vertical = "behind"
#     else:
#         vertical = "alongside"
    
#     # Priority logic: choose the most significant dimension
    
#     # If clearly in front or behind, use that
#     if vertical == "in front of":
#         if horizontal != "aligned with" and h_distance > v_distance:
#             # Horizontal distance is greater - prioritize horizontal
#             return horizontal
#         return vertical
    
#     elif vertical == "behind":
#         if horizontal != "aligned with" and h_distance > v_distance:
#             # Horizontal distance is greater - prioritize horizontal
#             return horizontal
#         return vertical
    
#     # Alongside - use horizontal position
#     elif horizontal != "aligned with":
#         return horizontal
    
#     return "at the same position as"


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
    
    # 4. Relative position captions
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
            
            # Calculate distances
            h_distance = abs(kart_x - ego_x)
            v_distance = abs(kart_y - ego_y)
            
            SLOPE_RATIO = 0.1# Start with 0.6 for a moderate cone.
            
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
            
            # Override: if kart's bottom is below ego's bottom, it's behind
            
            if kart_back > ego_back:
                vertical = "behind"
                # Recalculate v_distance using back edges
                v_distance = abs(kart_back - ego_back)
            
            # Choose the dimension with greater distance
            if h_distance > v_distance:
                position = horizontal
            else:
                position = vertical

        # 2. Check for the definitive "Behind" case
        # If the kart's bottom edge is below the ego's bottom edge, it is 100% behind.
            if kart_back > ego_back:
                #print(f"hello {position}")
                position = position
            
            
       
        # 3. Check for the definitive "In Front" case using the Cone Ratio
        # A kart is "In Front" if the vertical distance (forward) is SIGNIFICANTLY
        # greater than the lateral distance (side), relative to the cone slope.
        # Condition: v_distance > h_distance * SLOPE_RATIO
            elif v_distance > h_distance * SLOPE_RATIO:
                #print("slope ratio called {kart['kart_name']}")
                position = vertical # Should be "in front of" or "behind" (but behind is handled above)
            
        # 4. Otherwise, prioritize the horizontal position (Left/Right)
            else:
            # This is the "Alongside" or "Outside the Cone" case.
                #print(f"default {kart['kart_name']} {horizontal} ")
                position = horizontal
            
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