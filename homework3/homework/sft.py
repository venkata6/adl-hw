from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    
    The model should complete: question -> <answer>{answer}</answer>
    """
    # Set padding configuration
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create the full text: question followed by answer with EOS
    full_text = f"{question} {answer}{tokenizer.eos_token}"
    
    # Tokenize the full text
    full = tokenizer(
        full_text, 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )
    
    # Tokenize just the question part to find where the answer starts
    # Use add_special_tokens=False to get just the question tokens
    question_with_space = tokenizer(f"{question} ", add_special_tokens=False)
    question_len = len(question_with_space["input_ids"])
    
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    
    # Create labels by copying input_ids
    labels = list(input_ids)
    
    # Mask the question part - we only supervise the answer
    for i in range(min(question_len, len(labels))):
        labels[i] = -100
    
    # Mask padding tokens
    for i in range(len(labels)):
        if attention_mask[i] == 0:
            labels[i] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    Returns a dict with 'question' and 'answer' keys.
    """
    # Round the answer to 2 decimal places for easier learning
    answer_float = float(answer)
    rounded_answer = round(answer_float, 2)
    
    # Format as: <answer>{answer}</answer>
    formatted_answer = f"<answer>{rounded_answer}</answer>"
    
    return {
        "question": prompt,
        "answer": formatted_answer
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def data_collator(features):
    """
    Custom data collator that properly batches tokenized samples.
    """
    import torch
    
    # Extract fields from features
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]
    
    # Convert to tensors
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    
    return batch


def train_model(
    output_dir: str = "homework/sft_model",
    **kwargs,
):
    import torch
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from pathlib import Path
    
    # Initialize base model
    print("Initializing BaseLLM...")
    llm = BaseLLM()
    
    # Create LoRA configuration
    # Using r=8 to keep model size under 20MB
    # lora_alpha = 4-5 times rank, so 32-40
    print("Creating LoRA configuration...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Convert model to LoRA
    print("Converting model to LoRA...")
    llm.model = get_peft_model(llm.model, lora_config)
    
    # Enable input gradients if using GPU (required for gradient checkpointing)
    if torch.cuda.is_available():
        llm.model.enable_input_require_grads()
    
    # Print trainable parameters
    llm.model.print_trainable_parameters()
    
    # Load training data
    print("Loading training data...")
    train_dataset = Dataset("train")
    tokenized_train = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
    # Debug: Check a sample to ensure labels are correct
    print("\n" + "="*60)
    print("DEBUGGING TOKENIZATION")
    print("="*60)
    sample = tokenized_train[0]
    print(f"Input IDs length: {len(sample['input_ids'])}")
    print(f"Labels length: {len(sample['labels'])}")
    non_masked = sum(1 for l in sample['labels'] if l != -100)
    print(f"Non-masked labels count: {non_masked}")
    
    if non_masked == 0:
        print("\n❌ ERROR: All labels are masked! Training will fail.")
        print("Full text:", llm.tokenizer.decode(sample['input_ids']))
        raise ValueError("All labels are masked. Check tokenization logic.")
    else:
        print(f"\n✓ Labels look good! {non_masked} tokens will be supervised.")
        # Show what's being supervised
        supervised_tokens = [id for id, label in zip(sample['input_ids'], sample['labels']) if label != -100]
        supervised_text = llm.tokenizer.decode(supervised_tokens)
        print(f"Supervised text: {supervised_text}")
        
        # Show full text for comparison
        full_text = llm.tokenizer.decode(sample['input_ids'])
        print(f"Full text: {full_text[:200]}...")
    print("="*60 + "\n")
    
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
        per_device_train_batch_size=32,
        gradient_checkpointing=True,  # Save GPU memory
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Create trainer with custom data collator
    print("Creating trainer...")
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model to the correct directory
    print(f"\nSaving model to {output_path}...")
    trainer.save_model(str(output_path))
    
    print("Training complete!")
    
    # Test the model
    print("\nTesting trained model...")
    test_model(output_dir)


def test_model(ckpt_path: str):
    """
    Test the trained model on the validation set.
    """
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})