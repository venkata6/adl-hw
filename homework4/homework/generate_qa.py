import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import json

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 600, img_height: int = 400, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 600)
        img_height: Height of the image (default: 400)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Get kart names and detections for this view
    kart_names = info['karts']
    detections = info['detections'][view_index]
    
    # Calculate scaling factors (same as draw_detections)
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    # Calculate image center
    image_center_x = img_width / 2
    image_center_y = img_height / 2
    
    kart_objects = []
    
    # Process each detection
    for detection in detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        
        # Only process karts (class_id == 1)
        if int(class_id) != 1:
            continue
        
        # Scale coordinates to fit the current image size (same as draw_detections)
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        
        # Skip if bounding box is too small (same as draw_detections)
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        
        # Skip if out of bounds (same as draw_detections)
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue
        
        # Calculate center using scaled coordinates
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2
        
        # track_id == 0 is the ego car
        is_ego = (int(track_id) == 0)
        
        # Get kart name from kart_names array using track_id as index
        kart_name = kart_names[int(track_id)] if int(track_id) < len(kart_names) else f"kart_{track_id}"
        
        kart_objects.append({
            'instance_id': int(track_id),
            'kart_name': kart_name,
            'center': (center_x, center_y),
            'is_ego': is_ego,
            'is_center_kart': False,
            'bbox': [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
        })
    
    # Find kart closest to center
    if kart_objects:
        min_dist = float('inf')
        center_kart_idx = 0
        for idx, kart in enumerate(kart_objects):
            dist = ((kart['center'][0] - image_center_x) ** 2 + 
                   (kart['center'][1] - image_center_y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                center_kart_idx = idx
        
        kart_objects[center_kart_idx]['is_center_kart'] = True
    
    return kart_objects
    #raise NotImplementedError("Not implemented")


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    #raise NotImplementedError("Not implemented")
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    return info.get('track', 'unknown')


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 600, img_height: int = 400) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    #raise NotImplementedError("Not implemented")

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    #print(f"{track_name=}")
    
    qa_pairs = []
    
    # Find ego kart
    ego_kart = None
    for kart in karts:
        if kart.get('is_ego', False):
            ego_kart = kart
            break
    
    # 1. Ego car question
    if ego_kart:
        qa_pairs.append({
            'question': 'What kart is the ego car?',
            'answer': ego_kart['kart_name']
        })
    
    # 2. Total karts question
    qa_pairs.append({
        'question': 'How many karts are there in the scenario?',
        'answer': str(len(karts))
    })
    
    # 3. Track information question
    qa_pairs.append({
        'question': 'What track is this?',
        'answer': track_name
    })
    
    if ego_kart:
        ego_x, ego_y = ego_kart['center']
        
        # Count karts in different positions
        left_count = 0
        right_count = 0
        front_count = 0
        behind_count = 0
        
        # 4. Relative position questions for each non-ego kart
        for kart in karts:
            if kart.get('is_ego', False):
                continue
            
            kart_x, kart_y = kart['center']
            
            # Left/Right
            if kart_x < ego_x - 20:
                horizontal = "left"
                left_count += 1
            elif kart_x > ego_x + 20:
                horizontal = "right"
                right_count += 1
            else:
                horizontal = "aligned"
            
            # Front/Behind (lower y = in front in image space)
            if kart_y < ego_y - 20:
                vertical = "in front of"
                front_count += 1
            elif kart_y > ego_y + 20:
                vertical = "behind"
                behind_count += 1
            else:
                vertical = "at the same level as"
            
            # Generate relative position questions
            qa_pairs.append({
                'question': f'Is {kart["kart_name"]} to the left or right of the ego car?',
                'answer': horizontal if horizontal != "aligned" else "neither"
            })
            
            qa_pairs.append({
                'question': f'Is {kart["kart_name"]} in front of or behind the ego car?',
                'answer': vertical if vertical != "at the same level as" else "neither"
            })
            
            # Combined position
            if horizontal != "aligned" and vertical != "at the same level as":
                position = f"{vertical} and to the {horizontal}"
            elif horizontal != "aligned":
                position = f"to the {horizontal}"
            elif vertical != "at the same level as":
                position = vertical
            else:
                position = "at the same position as"
            
            qa_pairs.append({
                'question': f'Where is {kart["kart_name"]} relative to the ego car?',
                'answer': position
            })
        
        # 5. Counting questions by direction
        qa_pairs.append({
            'question': 'How many karts are to the left of the ego car?',
            'answer': str(left_count)
        })
        
        qa_pairs.append({
            'question': 'How many karts are to the right of the ego car?',
            'answer': str(right_count)
        })
        
        qa_pairs.append({
            'question': 'How many karts are in front of the ego car?',
            'answer': str(front_count)
        })
        
        qa_pairs.append({
            'question': 'How many karts are behind the ego car?',
            'answer': str(behind_count)
        })
    #print(f"{qa_pairs=}")
    return qa_pairs

def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)

def generate_dataset(
    data_dir: str = "./data/train",
    output_file: str = "./data/train/train_qa_pairs.json",
    max_samples: int = None
):
    """Generate QA dataset from all info files."""
    import json
    from tqdm import tqdm
    
    data_path = Path(data_dir)
    info_files = sorted(data_path.glob("*_info.json"))
    print(f"{info_files=}")
    
    if max_samples:
        info_files = info_files[:max_samples]
    
    all_qa_pairs = []
    
    for info_file in tqdm(info_files, desc="Generating QA pairs"):
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        #num_views = len(info['views'])
        num_views = len(info['karts'])
        base_name = info_file.stem.replace("_info", "")
        
        for view_index in range(num_views):
            image_file = list(data_path.glob(f"{base_name}_{view_index:02d}_im.jpg"))
            if not image_file:
                continue
            
            qa_pairs = generate_qa_pairs(str(info_file), view_index)

            path = str(image_file[0])
            cleaned_path = path.replace("data/", "", 1)
            #print(cleaned_path)

            for qa in qa_pairs:
                all_qa_pairs.append({
                    "image_file": cleaned_path,
                    "question": qa['question'],
                    "answer": qa['answer'],
                })
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"\nGenerated {len(all_qa_pairs)} QA pairs")
    print(f"Saved to: {output_path}")

"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate_dataset,
    })


if __name__ == "__main__":
    main()

