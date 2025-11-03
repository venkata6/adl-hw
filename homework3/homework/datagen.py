def generate_dataset(
    output_json: str = "data/rft.json",
    checkpoint_file: str = "data/rft_checkpoint.json",
    oversample: int = 20,  # Just use more samples upfront
    temperature: float = 0.7,
    checkpoint_every: int = 50,
    resume: bool = True,
    min_reasoning_length: int = 20,
    selection_strategy: str = "longest",
):
    """
    Generate RFT dataset - pure approach without retry queue.
    If we can't get a correct answer in 20 attempts, skip the question.
    """
    print("="*80)
    print("RFT DATASET GENERATION (Pure Approach)")
    print("="*80)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nLoading CoTModel...")
    model = CoTModel()
    
    print("Loading training dataset...")
    train_data = Dataset("train")
    total_questions = len(train_data)
    
    # Simpler state - no retry queue
    rft_data = []
    start_idx = 0
    successful = 0
    failed = 0
    
    # Load checkpoint if exists
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
            successful = checkpoint_data['successful']
            failed = checkpoint_data['failed']
            
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
    print(f"  Strategy: Skip questions if no correct answer found")
    print("="*80)
    
    # Main loop - one pass, no retries
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
            # Just skip - don't retry
        
        # Checkpoint
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
    
    # Save final dataset
    save_final_dataset(rft_data, output_json, successful, failed, total_questions)
    
    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Removed checkpoint file")
    
    return rft_data


def save_checkpoint_simple(checkpoint_path, data, last_idx, successful, failed, total):
    """Simplified checkpoint without retry queue"""
    checkpoint_data = {
        'data': data,
        'last_idx': last_idx,
        'successful': successful,
        'failed': failed,
        'total': total,
    }
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.parent / f"{checkpoint_path.name}.tmp"
    
    with open(temp_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    temp_path.rename(checkpoint_path)
    
    progress = 100 * (last_idx + 1) / total
    print(f"  ✓ Checkpoint saved: {last_idx+1}/{total} ({progress:.1f}%) - {len(data)} examples")