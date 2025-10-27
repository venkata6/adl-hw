from .base_llm import BaseLLM
from .sft import test_model
from .sft import tokenize
from .data import benchmark
import json
from pathlib import Path


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

class RFTDataset:
    """
    Dataset for RFT training.
    Uses Chain-of-Thought reasoning traces generated via rejection sampling.
    """
    
    def __init__(self, tokenizer, data_path: str = "data/rft.json"):
        """
        Initialize RFT dataset.
        
        Args:
            tokenizer: Tokenizer from BaseLLM
            data_path: Path to RFT JSON data
        """
        self.tokenizer = tokenizer
        self.data = self._load_data(data_path)
    
    def _load_data(self, path: str):
        """Load RFT dataset from JSON."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(
                f"RFT dataset not found at {path}. "
                f"Please run: python -m homework.datagen generate"
            )
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} RFT samples from {path}")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get tokenized sample.
        
        RFT data format: [question, answer, reasoning]
        where reasoning includes the chain-of-thought and ends with <answer>X</answer>
        """
        question, answer, reasoning = self.data[idx]
        
        # For RFT, we train on: question -> reasoning (which includes the answer)
        return tokenize(self.tokenizer, question, reasoning)


def data_collator(features):
    """
    Custom data collator that properly batches tokenized samples.
    Same as SFT collator.
    """
    import torch
    
    # Extract fields
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]
    
    # Stack into tensors
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    
    return batch


# def train_model(
#     output_dir: str,
#     **kwargs,
# ):
#     # Reuse much of the SFT code here
#     raise NotImplementedError()

def train_model(
    output_dir: str = "homework/rft_model",
    data_path: str = "data/rft.json",
    **kwargs,
):
    """
    Train RFT model using rejection sampling dataset.
    
    Similar to SFT but:
    1. Uses CoT reasoning traces instead of direct answers
    2. May use slightly larger LoRA rank (still under 50MB)
    3. Trains on question -> reasoning pairs
    
    Args:
        output_dir: Directory to save model
        data_path: Path to RFT dataset JSON
        **kwargs: Additional arguments
    """
    import torch
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    
    # Initialize base model
    print("Initializing BaseLLM for RFT...")
    llm = BaseLLM()
    
    # Create LoRA config with slightly larger rank for better reasoning
    # r=16 should keep us well under 50MB while improving performance
    print("Creating LoRA configuration (r=16 for better reasoning)...")
    lora_config = LoraConfig(
        r=16,  # Increased from 8 for better CoT performance
        lora_alpha=64,  # 4x rank
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Convert model to LoRA
    print("Converting model to LoRA...")
    llm.model = get_peft_model(llm.model, lora_config)
    
    # Print trainable parameters
    llm.model.print_trainable_parameters()
    
    # Load RFT training data
    print(f"Loading RFT data from {data_path}...")
    rft_dataset = RFTDataset(llm.tokenizer, data_path)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=str(output_path),
        logging_dir=str(output_path),
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        gradient_checkpointing=False,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=rft_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting RFT training...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to {output_path}...")
    trainer.save_model(str(output_path))
    
    print("RFT training complete!")
    
    # Test the model
    print("\nTesting trained RFT model...")
    test_rft_model(output_dir)


def test_rft_model(ckpt_path: str):
    """
    Test RFT model performance.
    Uses the same benchmark as SFT but model should show better reasoning.
    """
    from .data import Dataset
    
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")
    
    # Show sample generation to see reasoning
    print("\nSample generation with reasoning:")
    sample_questions = testset[:3]
    for q, a in sample_questions:
        print(f"\nQ: {q}")
        print(f"Expected: {a}")
        generated = llm.generate(q, ) #max_new_tokens=100
        print(f"Generated: {generated}")



if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
