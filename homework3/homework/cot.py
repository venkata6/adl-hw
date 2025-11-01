from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
       
        messages = [
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
            "content": "Could you give the conversion from gallon to cc for 4 units? Show your work and put the final answer in <answer></answer> tags."
        },
        {
            "role": "assistant",
            "content": "4 * 3785.411784\n<answer>15141.647136000003</answer>"
        },
        {
            "role": "user",
            "content": "What is the measurement of 3 kg when converted into pound? Show your work and put the final answer in <answer></answer> tags."
        },
        {
            "role": "assistant",
            "content": "3 * 2.20462262\n<answer>6.613867865546327</answer>"
        },
        {
            "role": "user",
            "content": "What is the equivalent of 2 kB in bit? Show your work and put the final answer in <answer></answer> tags."
        },
        {
            "role": "assistant",
            "content": "2 * 8000\n<answer>16000</answer>"
        },
        {
            "role": "user",
            "content": "How many MB is 2 G? Show your work and put the final answer in <answer></answer> tags."
        },
        {
            "role": "assistant",
            "content": "2 * 1024\n<answer>2048</answer>"
        },
        {
            "role": "user",
            "content": "How does 4 years measure up in terms of week? Show your work and put the final answer in <answer></answer> tags."
        },
        {
            "role": "assistant",
            "content": "4 * 52\n<answer>208</answer>"
        },
        {
            "role": "user",
            "content": "What is the conversion of 1 month to hour? Show your work and put the final answer in <answer></answer> tags"
        },
        {
            "role": "assistant",
            "content": "30 * 24.34947992\n<answer>730.4843975624999</answer>"
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
        
        message = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        #print(f"format message{message}")
        return message

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
