from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info
import json
    

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    #raise NotImplementedError("Not implemented")
    
    # Load the info JSON file
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Get the specific view data
    view_data = info['views'][view_index]
    karts = view_data['karts']
    
    captions = []
    
    # Find ego car (the one with is_ego=True)
    ego_kart = None
    for kart in karts:
        if kart.get('is_ego', False):
            ego_kart = kart
            break
    
    # 1. Ego car caption
    if ego_kart:
        captions.append(f"{ego_kart['kart_name']} is the ego car.")
    
    # 2. Counting caption
    num_karts = len(karts)
    captions.append(f"There are {num_karts} karts in the scenario.")
    
    # 3. Track name caption
    track_name = info.get('track_name', 'unknown')
    captions.append(f"The track is {track_name}.")
    
    # 4. Relative position captions
    if ego_kart:
        ego_bbox = ego_kart['bbox']
        ego_center_x = (ego_bbox[0] + ego_bbox[2]) / 2
        ego_center_y = (ego_bbox[1] + ego_bbox[3]) / 2
        
        for kart in karts:
            # Skip the ego car itself
            if kart.get('is_ego', False):
                continue
            
            kart_bbox = kart['bbox']
            kart_center_x = (kart_bbox[0] + kart_bbox[2]) / 2
            kart_center_y = (kart_bbox[1] + kart_bbox[3]) / 2
            
            # Determine relative position
            # Horizontal position
            if kart_center_x < ego_center_x - 20:  # threshold for "left"
                horizontal = "to the left"
            elif kart_center_x > ego_center_x + 20:  # threshold for "right"
                horizontal = "to the right"
            else:
                horizontal = "aligned with"
            
            # Vertical position
            if kart_center_y < ego_center_y - 20:  # threshold for "above"
                vertical = "above"
            elif kart_center_y > ego_center_y + 20:  # threshold for "below"
                vertical = "below"
            else:
                vertical = "at the same level as"
            
            # Combine position description
            if horizontal == "aligned with" and vertical == "at the same level as":
                position = "at the same position as"
            elif horizontal == "aligned with":
                position = vertical
            elif vertical == "at the same level as":
                position = horizontal
            else:
                position = f"{vertical} and {horizontal}"
            
            captions.append(f"{kart['kart_name']} is {position} the ego car.")
    
    return captions



def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

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
    output_file: str = "data/train_captions.json",
    max_samples: int = None
):
    """
    Generate caption dataset from all info files in a directory.
    
    Args:
        data_dir: Directory containing the info files (relative to project root)
        output_file: Path to save the generated captions (relative to project root)
        max_samples: Maximum number of samples to generate (None for all)
    """
    import json
    from pathlib import Path
    from tqdm import tqdm
    
    data_path = Path(data_dir)
    info_files = sorted(data_path.glob("*_info.json"))
    
    if max_samples:
        info_files = info_files[:max_samples]
    
    all_captions = []
    
    for info_file in tqdm(info_files, desc="Generating captions"):
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        num_views = len(info['views'])
        base_name = info_file.stem.replace("_info", "")
        
        for view_index in range(num_views):
            # Find corresponding image file
            image_file = list(data_path.glob(f"{base_name}_{view_index:02d}_im.jpg"))
            if not image_file:
                continue
            
            # Generate captions for this view
            captions = generate_caption(str(info_file), view_index)
            
            # Store each caption as a separate sample
            for caption in captions:
                all_captions.append({
                    "image_path": str(image_file[0]),
                    "caption": caption,
                    "info_file": str(info_file),
                    "view_index": view_index
                })
    
    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_captions, f, indent=2)
    
    print(f"\nGenerated {len(all_captions)} caption samples")
    print(f"Saved to: {output_path}")
    
    return all_captions


def generate_all_datasets():
    """
    Generate caption datasets for train, valid, and test splits.
    """
    print("Generating training dataset...")
    generate_dataset(
        data_dir="data/train",
        output_file="data/train_captions.json"
    )
    
    print("\nGenerating validation dataset...")
    generate_dataset(
        data_dir="data/valid",
        output_file="data/valid_captions.json"
    )
    
    print("\nGenerating test dataset...")
    generate_dataset(
        data_dir="data/test",
        output_file="data/test_captions.json"
    )
    
    print("\nAll datasets generated!")


def stats(caption_file: str = "data/train_captions.json"):
    """
    Print statistics about a generated caption dataset.
    """
    import json
    
    with open(caption_file, 'r') as f:
        captions = json.load(f)
    
    print(f"\nDataset Statistics for {caption_file}:")
    print(f"  Total samples: {len(captions)}")
    print(f"  Unique images: {len(set(c['image_path'] for c in captions))}")
    
    # Caption length statistics
    caption_lengths = [len(c['caption'].split()) for c in captions]
    print(f"  Avg caption length: {sum(caption_lengths) / len(caption_lengths):.1f} words")
    print(f"  Min caption length: {min(caption_lengths)} words")
    print(f"  Max caption length: {max(caption_lengths)} words")
    
    # Show some examples
    print("\nSample captions:")
    for i, caption in enumerate(captions[:5]):
        print(f"  {i+1}. {caption['caption']}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()
