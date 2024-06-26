from abc import ABC, abstractmethod
import os
from typing import Generator, Optional

from transformers import pipeline
from llama_index.core.llms import LLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.bedrock import Bedrock

from english2sql.config import LLMProvider, llm_provider_from_env

class SqlQueryGenerationAdapter(ABC):
    @abstractmethod
    def generate_sql_query(self, prompt: str) -> str:
        ...

    @abstractmethod
    def stream_generate_sql_query(self, prompt: str) -> Generator:
        ...


class HuggingFacePipelineAdapter(SqlQueryGenerationAdapter):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._pipe = pipeline("text-generation", model=model_id)
    
    def generate_sql_query(self, prompt: str) -> str:
        return self._pipe(prompt[-256:])[0]['generated_text'] # type: ignore

    def stream_generate_sql_query(self, prompt: str) -> Generator:
        ...


class LlamaIndexAdapter(SqlQueryGenerationAdapter):
    def __init__(
        self,
        llm: LLM,
        max_context: int,
    ) -> None:
        super().__init__()
        self._llm = llm
        self._max_context = max_context
    
    def generate_sql_query(self, prompt: str) -> str:
        return self._llm.complete(prompt[-self._max_context:]).text

    def stream_generate_sql_query(self, prompt: str) -> Generator:
        return self._llm.stream_complete(prompt[-self._max_context:])


def create_sql_generation_adapter(
    provider: LLMProvider,
    model_id: str,
    aws_profile_name: Optional[str]=None,
) -> LlamaIndexAdapter:
    if provider == LLMProvider.HUGGING_FACE:
        llm = HuggingFaceLLM(
            model_name=model_id,
            tokenizer_name=model_id,
            device_map="auto",
        )
        max_context = llm.context_window // 2
    elif provider == LLMProvider.BEDROCK:
        llm = Bedrock(
            model=model_id,
            profile_name=aws_profile_name,
        )
        max_context = llm.context_size // 2

    return LlamaIndexAdapter(
        llm=llm,
        max_context=max_context,
    )


def create_sql_generation_adapter_from_env() -> SqlQueryGenerationAdapter:
    provider = llm_provider_from_env()
    if provider == LLMProvider.HUGGING_FACE:
        model_id = os.environ.get('SQL_GENERATION_MODEL_ID', 'rakeshkiriyath/gpt2Medium_text_to_sql')
    elif provider == LLMProvider.BEDROCK:
        model_id = os.environ.get('SQL_GENERATION_MODEL_ID', 'amazon.titan-text-express-v1')
    return create_sql_generation_adapter(
        provider,
        model_id,
        aws_profile_name=os.environ.get('AWS_PROFILE', None),
    )
