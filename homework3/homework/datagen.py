"""
RFT Dataset Generation with Google Drive Checkpointing
Works seamlessly in Google Colab and local environments
"""
import json
from pathlib import Path
import torch
import logging
import os
from typing import List, Tuple, Optional
from .base_llm import BaseLLM
from .cot import CoTModel
from .data import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rft_generation.log'),
        logging.StreamHandler()
    ]
)


def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_google_drive():
    """Mount Google Drive in Colab"""
    if is_colab():
        from google.colab import drive
        drive_path = '/content/drive'
        if not os.path.exists(drive_path):
            print("Mounting Google Drive...")
            drive.mount(drive_path)
            print("✓ Google Drive mounted")
        return True
    return False


def resolve_path(path: str, use_gdrive: bool = None) -> Path:
    """
    Resolve path to use Google Drive if available/requested
    
    Args:
        path: Original path (e.g., "data/rft.json")
        use_gdrive: Force use of Google Drive (None = auto-detect)
    
    Returns:
        Path object pointing to correct location
    """
    # Auto-detect if not specified
    if use_gdrive is None:
        use_gdrive = is_colab()
    
    # If using Google Drive, prepend the drive path
    if use_gdrive:
        if mount_google_drive():
            # Convert to Google Drive path
            gdrive_base = Path('/content/drive/MyDrive/rft_checkpoints')
            gdrive_base.mkdir(parents=True, exist_ok=True)
            return gdrive_base / Path(path).name
    
    # Default to local path
    return Path(path)


def generate_dataset(
    output_json: str = "data/rft.json",
    checkpoint_file: str = "data/rft_checkpoint.json",
    oversample: int = 20,
    temperature: float = 0.7,
    checkpoint_every: int = 50,
    resume: bool = True,
    min_reasoning_length: int = 20,
    selection_strategy: str = "longest",
    use_gdrive: bool = None,  # NEW: Auto-detect or force Google Drive
):
    """
    Generate RFT dataset with Google Drive checkpointing support.
    
    Args:
        output_json: Path to save final dataset
        checkpoint_file: Path for checkpoint saves
        oversample: Number of completions per question (10-20 recommended)
        temperature: Sampling temperature for diversity (> 0)
        checkpoint_every: Save checkpoint every N questions
        resume: Resume from checkpoint if exists
        min_reasoning_length: Minimum chars for valid reasoning
        selection_strategy: "first", "longest", or "shortest"
        use_gdrive: Use Google Drive for checkpoints (None=auto-detect, True=force, False=local only)
    """
    print("="*80)
    print("RFT DATASET GENERATION")
    print("="*80)
    
    # Resolve paths (will use Google Drive if in Colab)
    output_path = resolve_path(output_json, use_gdrive)
    checkpoint_path = resolve_path(checkpoint_file, use_gdrive)
    
    # Show where files will be saved
    print(f"\nStorage location:")
    if is_colab() and use_gdrive != False:
        print(f"  Mode: Google Drive (Colab detected)")
    else:
        print(f"  Mode: Local filesystem")
    print(f"  Output: {output_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load CoT model
    print("\nLoading CoTModel...")
    model = CoTModel()
    
    # Load training data
    print("Loading training dataset...")
    train_data = Dataset("train")
    total_questions = len(train_data)
    
    # Initialize state
    rft_data = []
    start_idx = 0
    successful = 0
    failed = 0
    
    # Load checkpoint if exists
    if resume and checkpoint_path.exists():
        print(f"\n{'='*80}")
        print("RESUMING FROM CHECKPOINT")
        print(f"{'='*80}")
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            rft_data = checkpoint_data['data']
            start_idx = checkpoint_data['last_idx'] + 1
            successful = checkpoint_data['successful']
            failed = checkpoint_data['failed']
            
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"Resuming from question {start_idx}/{total_questions}")
            print(f"Generated: {len(rft_data)} examples")
            print(f"Success: {successful}, Failed: {failed}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
    else:
        print("\nNo checkpoint found. Starting from scratch...")
    
    print(f"\nConfiguration:")
    print(f"  Oversample: {oversample} completions per question")
    print(f"  Temperature: {temperature}")
    print(f"  Min reasoning: {min_reasoning_length} chars")
    print(f"  Selection: {selection_strategy}")
    print(f"  Checkpointing every {checkpoint_every} questions")
    print("="*80)
    
    # Main generation loop
    for idx in range(start_idx, total_questions):
        question, true_answer = train_data[idx]
        true_answer = float(true_answer)
        
        success = process_question(
            model=model,
            question=question,
            true_answer=true_answer,
            idx=idx,
            rft_data=rft_data,
            oversample=oversample,
            temperature=temperature,
            min_reasoning_length=min_reasoning_length,
            selection_strategy=selection_strategy,
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Checkpoint (saves to Google Drive if configured)
        if (idx + 1) % checkpoint_every == 0 or (idx + 1) == total_questions:
            save_checkpoint_simple(
                checkpoint_path,
                rft_data,
                idx,
                successful,
                failed,
                total_questions
            )
        
        # Clear GPU memory
        if (idx + 1) % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Progress
        if (idx + 1) % 10 == 0 or (idx + 1) == total_questions:
            success_rate = 100 * successful / (idx + 1)
            print(f"Progress: {idx+1}/{total_questions} | "
                  f"Success: {successful} | "
                  f"Failed: {failed} | "
                  f"Rate: {success_rate:.1f}%")
    
    # Save final dataset (to Google Drive if configured)
    print("\n" + "="*80)
    print("SAVING FINAL DATASET")
    print("="*80)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(rft_data, f, indent=2)
    
    success_rate = 100 * len(rft_data) / total_questions
    print(f"✓ Saved {len(rft_data)} examples to {output_path}")
    print(f"✓ Final success rate: {success_rate:.1f}%")
    
    # Save failed questions
    if failed > 0:
        failed_path = output_path.parent / "failed_questions.json"
        failed_indices = []
        for i in range(total_questions):
            if i not in [rft_data[j][0] for j in range(len(rft_data))]:
                failed_indices.append(i)
        
        with open(failed_path, 'w') as f:
            json.dump(failed_indices[:failed], f, indent=2)
        print(f"✓ Saved failed question info to {failed_path}")
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"✓ Removed checkpoint file")
    
    # Show statistics
    show_statistics(rft_data, total_questions)
    
    return rft_data


def process_question(
    model: CoTModel,
    question: str,
    true_answer: float,
    idx: int,
    rft_data: List,
    oversample: int,
    temperature: float,
    min_reasoning_length: int,
    selection_strategy: str,
) -> bool:
    """Process a single question using RFT approach"""
    try:
        # Format prompt
        formatted_prompt = model.format_prompt(question)
        
        # Generate multiple completions
        completions = model.batched_generate(
            [formatted_prompt],
            num_return_sequences=oversample,
            temperature=temperature
        )
        
        # Extract completions
        completions = completions[0] if isinstance(completions[0], list) else completions
        
        # Find correct completions
        correct_completions = []
        
        for completion in completions:
            if len(completion.strip()) < min_reasoning_length:
                continue
            
            try:
                parsed_answer = model.parse_answer(completion)
                tolerance = max(0.01 * abs(true_answer), 0.01)
                
                if abs(parsed_answer - true_answer) < tolerance:
                    correct_completions.append(completion)
            except:
                continue
        
        # Select best completion
        if correct_completions:
            if selection_strategy == "first":
                selected = correct_completions[0]
            elif selection_strategy == "longest":
                selected = max(correct_completions, key=len)
            elif selection_strategy == "shortest":
                selected = min(correct_completions, key=len)
            else:
                selected = correct_completions[0]
            
            rft_data.append([question, true_answer, selected])
            return True
        
        return False
        
    except Exception as e:
        logging.error(f"Error processing question {idx}: {e}")
        return False


def save_checkpoint_simple(checkpoint_path: Path, data: List, last_idx: int, 
                          successful: int, failed: int, total: int):
    """Save checkpoint (works with both local and Google Drive paths)"""
    checkpoint_data = {
        'data': data,
        'last_idx': last_idx,
        'successful': successful,
        'failed': failed,
        'total': total,
    }
    
    # Create parent directory
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to temp file first (atomic operation)
    temp_path = checkpoint_path.parent / f"{checkpoint_path.name}.tmp"
    
    with open(temp_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    # Atomic rename
    temp_path.rename(checkpoint_path)
    
    progress = 100 * (last_idx + 1) / total
    print(f"  ✓ Checkpoint saved: {last_idx+1}/{total} ({progress:.1f}%) - {len(data)} examples")


def show_statistics(rft_data: List, total_questions: int):
    """Display dataset statistics"""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    if not rft_data:
        print("⚠ No data generated!")
        return
    
    reasoning_lengths = [len(item[2]) for item in rft_data]
    avg_length = sum(reasoning_lengths) / len(reasoning_lengths)
    
    print(f"Total examples: {len(rft_data)}")
    print(f"Coverage: {len(rft_data)}/{total_questions} ({100*len(rft_data)/total_questions:.1f}%)")
    print(f"Average reasoning length: {avg_length:.0f} characters")
    
    # Sample examples
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES")
    print("="*80)
    for i in range(min(3, len(rft_data))):
        print(f"\nExample {i+1}:")
        print(f"  Question: {rft_data[i][0][:100]}...")
        print(f"  Answer: {rft_data[i][1]}")
        print(f"  Reasoning: {rft_data[i][2][:150]}...")


def clear_checkpoint(checkpoint_file: str = "data/rft_checkpoint.json", use_gdrive: bool = None):
    """Clear checkpoint"""
    checkpoint_path = resolve_path(checkpoint_file, use_gdrive)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"✓ Cleared checkpoint: {checkpoint_path}")
    else:
        print(f"No checkpoint found at: {checkpoint_path}")


def show_checkpoint_status(checkpoint_file: str = "data/rft_checkpoint.json", use_gdrive: bool = None):
    """Show checkpoint status"""
    checkpoint_path = resolve_path(checkpoint_file, use_gdrive)
    
    if not checkpoint_path.exists():
        print(f"No checkpoint found at: {checkpoint_path}")
        return
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        print("="*80)
        print("CHECKPOINT STATUS")
        print("="*80)
        print(f"Location: {checkpoint_path}")
        print(f"Last processed: {checkpoint_data['last_idx']}/{checkpoint_data['total']}")
        progress = 100 * (checkpoint_data['last_idx'] + 1) / checkpoint_data['total']
        print(f"Progress: {progress:.1f}%")
        print(f"Generated examples: {len(checkpoint_data['data'])}")
        print(f"Successful: {checkpoint_data['successful']}")
        print(f"Failed: {checkpoint_data['failed']}")
        success_rate = 100 * checkpoint_data['successful'] / (checkpoint_data['last_idx'] + 1)
        print(f"Success rate: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")


if __name__ == "__main__":
    from fire import Fire
    
    Fire({
        'generate': generate_dataset,
        'clear': clear_checkpoint,
        'status': show_checkpoint_status,
    })