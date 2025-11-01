"""
RFT (Reinforcement Fine-Tuning) implementation.
Generates Chain-of-Thought reasoning data by sampling multiple outputs and selecting correct ones.
"""
import json
from pathlib import Path
from .base_llm import BaseLLM
from .cot import CoTModel
from .data import Dataset


def generate_dataset(output_json: str = "data/rft.json", oversample: int = 10, temperature: float = 0.6):
    """
    Generate RFT dataset by:
    1. Using CoTModel to generate multiple completions per question
    2. Selecting completions with correct answers
    3. Saving question + reasoning + answer tuples
    
    Args:
        output_json: Path to save the generated dataset
        oversample: Number of completions to generate per question
        temperature: Sampling temperature for diversity
    """
    print("="*80)
    print("GENERATING RFT DATASET")
    print("="*80)
    
    # Load CoT model (preferably the larger one for better reasoning)
    print("\nLoading CoTModel...")
    # Try to use the larger model for better rollouts
    try:
        from .base_llm import checkpoint as default_checkpoint
        # Use larger model if available
        larger_checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        print(f"Attempting to use larger model: {larger_checkpoint}")
        model = CoTModel(checkpoint=larger_checkpoint)
    except:
        print("Falling back to default model")
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
    print(f"Processing {total_questions} questions...")
    print("="*80)
    
    # Process in batches for efficiency
    batch_size = 10
    
    for batch_start in range(0, total_questions, batch_size):
        batch_end = min(batch_start + batch_size, total_questions)
        batch_prompts = []
        batch_answers = []
        
        # Prepare batch
        for idx in range(batch_start, batch_end):
            question, answer = train_data[idx]
            formatted_prompt = model.format_prompt(question)
            batch_prompts.append(formatted_prompt)
            batch_answers.append(float(answer))
        
        # Generate multiple completions for each prompt in the batch
        try:
            # Generate oversample completions per prompt
            completions_batch = model.batched_generate(
                batch_prompts,
                num_return_sequences=oversample,
                temperature=temperature
            )
            
            # Process each question in the batch
            for i, (question, true_answer) in enumerate(zip(
                [train_data[batch_start + i][0] for i in range(len(batch_prompts))],
                batch_answers
            )):
                completions = completions_batch[i]
                
                # Find a correct completion
                found_correct = False
                for completion in completions:
                    # Parse the answer from the completion
                    parsed_answer = model.parse_answer(completion)
                    
                    # Check if answer is correct (with small tolerance for floating point)
                    if abs(parsed_answer - true_answer) < 0.01 * abs(true_answer) or abs(parsed_answer - true_answer) < 0.01:
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
            print(f"Error processing batch {batch_start}-{batch_end}: {e}")
            failed_generations += (batch_end - batch_start)
            continue
        
        # Progress update
        if (batch_end) % 100 == 0 or batch_end == total_questions:
            success_rate = 100 * successful_generations / (successful_generations + failed_generations) if (successful_generations + failed_generations) > 0 else 0
            print(f"Progress: {batch_end}/{total_questions} | "
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