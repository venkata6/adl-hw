"""
Data generation for RFT using rejection sampling with Chain-of-Thought reasoning.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional
from .base_llm import BaseLLM
from .data import Dataset


class CoTModel:
    """
    Chain-of-Thought model wrapper for generating reasoning traces.
    Uses a larger model (SmolLM2-1.7B-Instruct) for better reasoning.
    """
    
    def __init__(self, checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct"):
        """Initialize CoT model with larger checkpoint for better reasoning."""
        print(f"Loading CoT model: {checkpoint}")
        self.llm = BaseLLM(checkpoint=checkpoint)
    
    def format_cot_prompt(self, question: str) -> str:
        """
        Format question to encourage chain-of-thought reasoning.
        Uses chat template with few-shot examples for better performance.
        """
        system_message = (
            "You are a helpful math assistant. Solve problems step-by-step and "
            "provide your final answer in <answer>NUMBER</answer> format.\n\n"
            "Examples:\n"
            "Q: How many grams in 2 kg?\n"
            "A: 1 kg = 1000 grams. So 2 kg = 2 × 1000 = <answer>2000</answer>\n\n"
            "Q: Convert 5 meters to centimeters.\n"
            "A: 1 meter = 100 cm. So 5 meters = 5 × 100 = <answer>500</answer>"
        )
        
        user_message = f"{question}\n\nSolve step by step and provide answer in <answer>NUMBER</answer> format:"
        
        # Use chat template for better performance
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        prompt = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def batched_generate(
        self,
        questions: List[str],
        num_return_sequences: int = 10,
        temperature: float = 0.7
    ) -> List[List[str]]:
        """
        Generate multiple diverse reasoning traces for each question.
        
        Args:
            questions: List of questions
            num_return_sequences: Number of diverse samples per question
            temperature: Sampling temperature (>0 for diversity)
        
        Returns:
            List of lists, where each inner list contains reasoning traces
        """
        prompts = [self.format_cot_prompt(q) for q in questions]
        return self.llm.batched_generate(
            prompts,
            num_return_sequences=num_return_sequences,
            temperature=temperature
        )
    
    def parse_answer(self, text: str) -> Optional[float]:
        """Parse answer from generation, same as BaseLLM."""
        return self.llm.parse_answer(text)


def generate_dataset(output_json: str, oversample: int = 20, temperature: float = 0.8):
    """
    Generate RFT training dataset using rejection sampling.
    
    For each question:
    1. Generate 'oversample' diverse reasoning traces
    2. Find traces that produce the correct answer
    3. Select the best trace (first correct one)
    4. Store as (question, answer, reasoning) tuple
    
    Args:
        output_json: Path to save the dataset JSON file
        oversample: Number of diverse samples to generate per question (default: 20)
        temperature: Sampling temperature for diversity (default: 0.8)
    """
    print("Initializing CoT model...")
    cot_model = CoTModel()
    
    print("Loading training dataset...")
    dataset = Dataset("train")
    num_samples = len(dataset)
    
    print(f"Generating RFT dataset with {num_samples} samples...")
    print(f"Using oversample={oversample} attempts per question with temperature={temperature}")
    
    rft_data = []
    success_count = 0
    
    # Process in batches for efficiency
    batch_size = 5
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_questions = []
        batch_answers = []
        
        for j in range(i, batch_end):
            question, answer = dataset[j]
            batch_questions.append(question)
            batch_answers.append(float(answer))
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")
        print(f"Questions {i+1}-{batch_end}/{num_samples}")
        
        # Generate multiple reasoning traces for each question
        all_traces = cot_model.batched_generate(
            batch_questions,
            num_return_sequences=oversample,
            temperature=temperature
        )
        
        # Process each question in the batch
        for q_idx, (question, ground_truth) in enumerate(zip(batch_questions, batch_answers)):
            traces = all_traces[q_idx]
            
            # Find first correct trace
            correct_trace = None
            correct_count = 0
            
            for trace in traces:
                predicted = cot_model.parse_answer(trace)
                if predicted is not None:
                    # Use relative tolerance for better matching
                    tolerance = max(0.01, abs(ground_truth) * 0.001)
                    if abs(predicted - ground_truth) < tolerance:
                        if correct_trace is None:
                            correct_trace = trace
                        correct_count += 1
            
            if correct_trace is not None:
                # Clean up the trace for training
                cleaned_trace = clean_reasoning_trace(correct_trace, ground_truth)
                rft_data.append([question, ground_truth, cleaned_trace])
                success_count += 1
                print(f"  ✓ Question {i + q_idx + 1}: Success ({correct_count}/{oversample} correct)")
            else:
                # Debug: show what was actually generated
                sample_trace = traces[0][:200] if traces else "No traces"
                sample_parsed = cot_model.parse_answer(traces[0]) if traces else None
                print(f"  ✗ Question {i + q_idx + 1}: No correct answer found")
                print(f"     Expected: {ground_truth}, Sample parsed: {sample_parsed}")
                if len(sample_trace) < 200:
                    print(f"     Sample: {sample_trace}")
    
    success_rate = success_count / num_samples * 100
    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Success rate: {success_count}/{num_samples} ({success_rate:.1f}%)")
    print(f"{'='*60}")
    
    # Save to JSON
    output_file = Path(output_json)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(rft_data, f, indent=2)
    
    print(f"\nDataset saved to: {output_json}")
    print(f"Total samples: {len(rft_data)}")


def clean_reasoning_trace(trace: str, answer: float) -> str:
    """
    Clean and format the reasoning trace.
    Ensures it ends with <answer>X</answer> format.
    
    Args:
        trace: Raw reasoning trace from model
        answer: Ground truth answer
    
    Returns:
        Cleaned reasoning trace
    """
    # Remove any leading/trailing whitespace
    trace = trace.strip()
    
    # If trace doesn't end with proper answer tag, add it
    if not trace.endswith("</answer>"):
        # Remove any existing incomplete answer tags
        if "<answer>" in trace:
            trace = trace.split("<answer>")[0].strip()
        
        # Add proper answer tag
        trace = f"{trace} <answer>{answer}</answer>"
    
    return trace


def load_rft_dataset(path: str = "data/rft.json") -> List[Tuple[str, float, str]]:
    """
    Load RFT dataset from JSON file.
    
    Args:
        path: Path to RFT dataset JSON
    
    Returns:
        List of (question, answer, reasoning) tuples
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    return [(item[0], item[1], item[2]) for item in data]


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)