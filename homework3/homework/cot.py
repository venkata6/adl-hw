from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
       
        messages = [ 
        {
        "role": "system",
        "content": " For unit conversions, always identify: 1) the conversion factor, 2) whether to multiply or divide. NEVER use powers of 2 (1024, 2048, 4096, 1048576). Use decimal: 1000, 2000, 10000."
        },
        {
            "role": "user",
            "content": "Convert 2 kilograms to grams"
        },
        {
            "role": "assistant",
            "content": "1 kg = 1000 g , 2 × 1000 = 2000\n<answer>2000</answer>"
        },
        {
            "role": "user",
            "content": "Express 2 cm as a quantity of in. Show your work and put the final answer in <answer></answer> tags."
        },
        {
            "role": "assistant",
            "content": "1 cm = 0.39370079 in, 2 * 0.39370079\n<answer>0.7874015748031495</answer>"
        },     
        {
            "role": "user",
            "content": "Could you give the conversion from year to day for 7 units? Show your work and put the final answer in <answer></answer> tags."
        },
        {
            "role": "assistant",
            "content": "7 * 365.24219878\n<answer>2556.69539146875</answer>"
        },
        {
            "role": "user",
            "content": "Calculate: 23 * 4. Show your work and put the final answer in <answer></answer> tags."
        },
        {
            "role": "assistant",
            "content": "23 * 4 = 92\n<answer>92</answer>"
        },
        {
            "role": "user",
            "content": f"{question} Show your work and put the final answer in <answer></answer> tags."
        }
        ] # 24/25 benchmark_result.accuracy=0.39  benchmark_result.answer_rate=0.77
        
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
