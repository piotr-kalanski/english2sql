from abc import ABC, abstractmethod
from typing import Generator

from transformers import pipeline
from llama_index.llms.huggingface import HuggingFaceLLM


class SqlQueryGenerationAdapter(ABC):
    @abstractmethod
    def generate_sql_query(self, prompt: str) -> str:
        ...

    @abstractmethod
    def stream_generate_sql_query(self, prompt: str) -> Generator:
        ...

# TODO - Bedrock models

class HuggingFacePipelineAdapter(SqlQueryGenerationAdapter):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._pipe = pipeline("text-generation", model=model_id)
    
    def generate_sql_query(self, prompt: str) -> str:
        return self._pipe(prompt[-256:])[0]['generated_text']

    def stream_generate_sql_query(self, prompt: str) -> Generator:
        ...


class HuggingFaceLlamaIndex(SqlQueryGenerationAdapter):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._llm = HuggingFaceLLM(
            model_name=model_id,
            tokenizer_name=model_id,
            # context_window=3900,
            # max_new_tokens=256,
            # model_kwargs={"quantization_config": quantization_config},
            # generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            # messages_to_prompt=messages_to_prompt,
            # completion_to_prompt=completion_to_prompt,
            device_map="auto",
        )
    
    def generate_sql_query(self, prompt: str) -> str:
        return self._llm.complete(prompt[-256:]).text

    def stream_generate_sql_query(self, prompt: str) -> Generator:
        return self._llm.stream_complete(prompt[-256:])


def create_sql_generation_adapter_from_env() -> SqlQueryGenerationAdapter:
    return HuggingFaceLlamaIndex(
        model_id="rakeshkiriyath/gpt2Medium_text_to_sql"
    )
