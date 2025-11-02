"""
Memory-efficient RFT data generation.
Processes one question at a time to avoid OOM errors.
"""
import json
from pathlib import Path
import torch
from .base_llm import BaseLLM
from .cot import CoTModel
from .data import Dataset


def generate_dataset(
    output_json: str = "data/rft.json", 
    oversample: int = 5,  # REDUCED from 10 to 5
    temperature: float = 0.6
):
    """
    Generate RFT dataset ONE QUESTION AT A TIME to avoid OOM.
    
    Args:
        output_json: Path to save the generated dataset
        oversample: Number of completions per question (reduced to 5 for memory)
        temperature: Sampling temperature for diversity
    """
    print("="*80)
    print("MEMORY-EFFICIENT RFT DATASET GENERATION")
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
    
    # Statistics
    total_questions = len(train_data)
    successful_generations = 0
    failed_generations = 0
    
    # Generated data
    rft_data = []
    
    print(f"\nGenerating {oversample} completions per question with temperature={temperature}")
    print(f"Processing {total_questions} questions ONE AT A TIME...")
    print("="*80)
    
    # Process ONE question at a time to save memory
    for idx in range(total_questions):
        question, true_answer = train_data[idx]
        true_answer = float(true_answer)
        
        try:
            # Format prompt
            formatted_prompt = model.format_prompt(question)
            
            # Generate multiple completions for THIS SINGLE QUESTION
            completions = model.batched_generate(
                [formatted_prompt],  # Single prompt as a list
                num_return_sequences=oversample,
                temperature=temperature
            )
            
            # completions is a list with one element (which is a list of completions)
            completions = completions[0] if isinstance(completions[0], list) else completions
            
            # Find a correct completion
            found_correct = False
            for completion in completions:
                # Parse the answer
                parsed_answer = model.parse_answer(completion)
                
                # Check if correct (with tolerance)
                tolerance = max(0.01 * abs(true_answer), 0.01)
                if abs(parsed_answer - true_answer) < tolerance:
                    # Found a correct answer!
                    rft_data.append([
                        question,
                        true_answer,
                        completion
                    ])
                    successful_generations += 1
                    found_correct = True
                    break
            
            if not found_correct:
                failed_generations += 1
        
        except Exception as e:
            print(f"Error on question {idx}: {e}")
            failed_generations += 1
            continue
        
        # Clear GPU memory every 10 questions
        if (idx + 1) % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Progress update every 50 questions
        if (idx + 1) % 50 == 0 or (idx + 1) == total_questions:
            success_rate = 100 * successful_generations / (idx + 1)
            print(f"Progress: {idx+1}/{total_questions} | "
                  f"Success: {successful_generations} | "
                  f"Failed: {failed_generations} | "
                  f"Success Rate: {success_rate:.1f}%")
    
    # Save the dataset
    print("\n" + "="*80)
    print("SAVING DATASET")
    print("="*80)
    
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(rft_data, f, indent=2)
    
    print(f"Saved {len(rft_data)} examples to {output_json}")
    print(f"Success rate: {100 * successful_generations / total_questions:.1f}%")
    print(f"Failed: {failed_generations} / {total_questions}")
    
    # Show a few examples
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES")
    print("="*80)
    for i in range(min(3, len(rft_data))):
        print(f"\nExample {i+1}:")
        print(f"  Question: {rft_data[i][0][:80]}...")
        print(f"  Answer: {rft_data[i][1]}")
        print(f"  Reasoning: {rft_data[i][2][:150]}...")
    
    return rft_data


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)