from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        # messages = [
        #     {
        #         "role": "user",
        #         "content": "What is 15 + 27? Provide your answer in <answer></answer> tags."
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "15 + 27 = 42\n<answer>42</answer>"
        #     },
        #     {
        #         "role": "user",
        #         "content": f"{question} Provide your answer in <answer></answer> tags."
        #     }
        # ] #16/25
        messages = [
        {
            "role": "user",
            "content": "Calculate: 23 * 4. Show your work and put the final answer in <answer></answer> tags."
        },
        {
            "role": "assistant",
            "content": "23 * 4 = 92\n<answer>92</answer>"
        },
        # {
        #     "role": "user",
        #     "content": "Calculate: 100 / 4. Show your work and put the final answer in <answer></answer> tags."
        # },
        # {
        #     "role": "assistant",
        #     "content": "100 / 4 = 25\n<answer>25</answer>"
        # },
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
