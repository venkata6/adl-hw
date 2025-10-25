from .base_llm import BaseLLM
from .data import Dataset, benchmark
import torch
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path


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
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    #raise NotImplementedError()
    answer_float = float(answer)
    rounded_answer = round(answer_float, 2)
    
    # Format as requested: <answer>{answer}</answer>
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


def train_model(
    output_dir: str,
    **kwargs,
):
    #raise NotImplementedError()
    
    # Initialize base model
    print("Initializing BaseLLM...")
    llm = BaseLLM()
    
    # Create LoRA config
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
    
    # Enable input gradients if using GPU (fixes gradient_checkpointing bug)
    if torch.cuda.is_available():
        llm.model.enable_input_require_grads()
    
    # Print trainable parameters
    llm.model.print_trainable_parameters()
    
    # Load training data
    print("Loading training data...")
    train_dataset = Dataset("train")
    tokenized_train = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
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
        gradient_checkpointing=True,
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
        train_dataset=tokenized_train,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to {output_path}...")
    trainer.save_model(str(output_path))
    
    print("Training complete!")
    
    # Test the model
    print("\nTesting trained model...")
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
