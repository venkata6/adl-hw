from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
       
        messages = [
            {"role": "system", "content": (
                "You are a precise and helpful unit conversion assistant. "
                "Always think step-by-step and show intermediate calculations when needed. be concise "
                "Output the final numeric result on a new line enclosed in <answer>...</answer> tags."
            )},
            # metric conversion
            {"role": "user", "content": "How many grams are in 5 kg?"},
            {"role": "assistant", "content": "1 kg = 1000 grams\n5 * 1000 = 5000 grams\n<answer>5000</answer>"},
            # imperial conversion
            {"role": "user", "content": "How many pounds are in 3 kg?"},
            {"role": "assistant",
             "content": "1 kg ≈ 2.20462 pounds\n3 * 2.20462 ≈ 6.61386 pounds\n<answer>6.61386</answer>"},
            # reverse conversion
            {"role": "user", "content": "How many kilograms are in 2200 grams?"},
            {"role": "assistant", "content": "1000 grams = 1 kg\n2200 / 1000 = 2.2 kg\n<answer>2.2</answer>"},
            # conversion with decimals
            {"role": "user", "content": "How many inches are in 2.5 feet?"},
            {"role": "assistant", "content": "1 foot = 12 inches\n2.5 * 12 = 30 inches\n<answer>30</answer>"},
            # temperature (different style)
            {"role": "user", "content": "What is 100 degrees Fahrenheit in Celsius?"},
            {"role": "assistant",
             "content": "C = (F - 32) * 5/9\nC = (100 - 32) * 5/9 = 68 * 5/9 ≈ 37.78\n<answer>37.78</answer>"},
            # time
            {"role": "user", "content": "How many minutes are in 3.5 hours?"},
            {"role": "assistant", "content": "1 hour = 60 minutes\n3.5 * 60 = 210 minutes\n<answer>210</answer>"},
            # area
            {"role": "user", "content": "How many square meters are in 2 hectares?"},
            {"role": "assistant",
             "content": "1 hectare = 10,000 square meters\n2 * 10,000 = 20000 square meters\n<answer>20000</answer>"},
            {"role": "user", "content": question},
        ]
       
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        #raise NotImplementedError()


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
