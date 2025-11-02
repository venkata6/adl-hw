"""
Checkpointed RFT data generation - Resume after crashes!
Saves progress periodically so you can continue where you left off.
"""
import json
from pathlib import Path
import torch
from .base_llm import BaseLLM
from .cot import CoTModel
from .data import Dataset


def generate_dataset(
    output_json: str = "data/rft.json",
    checkpoint_file: str = "data/rft_checkpoint.json",
    oversample: int = 5,
    temperature: float = 0.6,
    checkpoint_every: int = 50,  # Save every 50 questions
    resume: bool = True,  # Automatically resume if checkpoint exists
):
    """
    Generate RFT dataset with checkpointing.
    
    Args:
        output_json: Final output path
        checkpoint_file: Checkpoint file to save progress
        oversample: Number of completions per question
        temperature: Sampling temperature
        checkpoint_every: Save checkpoint every N questions
        resume: If True, resume from checkpoint if it exists
    """
    print("="*80)
    print("CHECKPOINTED RFT DATASET GENERATION")
    print("="*80)
    
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
    
    # Try to load checkpoint
    rft_data = []
    start_idx = 0
    successful_generations = 0
    failed_generations = 0
    
    checkpoint_path = Path(checkpoint_file)
    if resume and checkpoint_path.exists():
        print(f"\n{'='*80}")
        print("RESUMING FROM CHECKPOINT")
        print(f"{'='*80}")
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            rft_data = checkpoint_data['data']
            start_idx = checkpoint_data['last_idx'] + 1
            successful_generations = checkpoint_data['successful']
            failed_generations = checkpoint_data['failed']
            
            print(f"Loaded checkpoint from {checkpoint_file}")
            print(f"Resuming from question {start_idx}/{total_questions}")
            print(f"Already generated: {len(rft_data)} examples")
            print(f"Success: {successful_generations}, Failed: {failed_generations}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch...")
            start_idx = 0
            rft_data = []
            successful_generations = 0
            failed_generations = 0
    else:
        print("\nNo checkpoint found. Starting from scratch...")
    
    print(f"\nGenerating {oversample} completions per question with temperature={temperature}")
    print(f"Checkpointing every {checkpoint_every} questions")
    print("="*80)
    
    # Process questions starting from checkpoint
    for idx in range(start_idx, total_questions):
        question, true_answer = train_data[idx]
        true_answer = float(true_answer)
        
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
            
            # Find a correct completion
            found_correct = False
            for completion in completions:
                parsed_answer = model.parse_answer(completion)
                
                # Check if correct
                tolerance = max(0.01 * abs(true_answer), 0.01)
                if abs(parsed_answer - true_answer) < tolerance:
                    rft_data.append([question, true_answer, completion])
                    successful_generations += 1
                    found_correct = True
                    break
            
            if not found_correct:
                failed_generations += 1
        
        except Exception as e:
            print(f"Error on question {idx}: {e}")
            failed_generations += 1
        
        # CHECKPOINT: Save progress periodically
        if (idx + 1) % checkpoint_every == 0 or (idx + 1) == total_questions:
            save_checkpoint(
                checkpoint_path,
                rft_data,
                idx,
                successful_generations,
                failed_generations,
                total_questions
            )
        
        # Clear GPU memory periodically
        if (idx + 1) % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Progress update
        if (idx + 1) % 10 == 0 or (idx + 1) == total_questions:
            success_rate = 100 * successful_generations / (idx + 1)
            print(f"Progress: {idx+1}/{total_questions} | "
                  f"Success: {successful_generations} | "
                  f"Failed: {failed_generations} | "
                  f"Rate: {success_rate:.1f}%")
    
    # Save final dataset
    print("\n" + "="*80)
    print("SAVING FINAL DATASET")
    print("="*80)
    
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(rft_data, f, indent=2)
    
    print(f"Saved {len(rft_data)} examples to {output_json}")
    print(f"Success rate: {100 * successful_generations / total_questions:.1f}%")
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Removed checkpoint file: {checkpoint_file}")
    
    # Show samples
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES")
    print("="*80)
    for i in range(min(3, len(rft_data))):
        print(f"\nExample {i+1}:")
        print(f"  Question: {rft_data[i][0][:80]}...")
        print(f"  Answer: {rft_data[i][1]}")
        print(f"  Reasoning: {rft_data[i][2][:150]}...")
    
    return rft_data


def save_checkpoint(checkpoint_path, data, last_idx, successful, failed, total):
    """Save checkpoint to disk"""
    checkpoint_data = {
        'data': data,
        'last_idx': last_idx,
        'successful': successful,
        'failed': failed,
        'total': total,
        'timestamp': str(Path.cwd())  # Just a marker
    }
    
    # Save to temporary file first, then rename (atomic operation)
    temp_path = checkpoint_path.parent / f"{checkpoint_path.name}.tmp"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    # Atomic rename
    temp_path.rename(checkpoint_path)
    
    progress = 100 * (last_idx + 1) / total
    print(f"  ✓ Checkpoint saved: {last_idx+1}/{total} ({progress:.1f}%) - {len(data)} examples")


def clear_checkpoint(checkpoint_file: str = "data/rft_checkpoint.json"):
    """Clear checkpoint to start fresh"""
    checkpoint_path = Path(checkpoint_file)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Cleared checkpoint: {checkpoint_file}")
    else:
        print(f"No checkpoint found at: {checkpoint_file}")


def show_checkpoint_status(checkpoint_file: str = "data/rft_checkpoint.json"):
    """Show current checkpoint status"""
    checkpoint_path = Path(checkpoint_file)
    
    if not checkpoint_path.exists():
        print("No checkpoint found.")
        return
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        print("="*80)
        print("CHECKPOINT STATUS")
        print("="*80)
        print(f"Last processed index: {checkpoint_data['last_idx']}")
        print(f"Total questions: {checkpoint_data['total']}")
        print(f"Progress: {100 * (checkpoint_data['last_idx'] + 1) / checkpoint_data['total']:.1f}%")
        print(f"Generated examples: {len(checkpoint_data['data'])}")
        print(f"Successful: {checkpoint_data['successful']}")
        print(f"Failed: {checkpoint_data['failed']}")
        print(f"Success rate: {100 * checkpoint_data['successful'] / (checkpoint_data['last_idx'] + 1):.1f}%")
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")


if __name__ == "__main__":
    from fire import Fire
    
    Fire({
        'generate': generate_dataset,
        'clear': clear_checkpoint,
        'status': show_checkpoint_status,
    })