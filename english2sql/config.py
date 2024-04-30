import os
from enum import Enum


class LLMProvider(Enum):
    HUGGING_FACE = "hf"
    BEDROCK = "bedrock"


def llm_provider_from_env() -> LLMProvider:
    if os.environ.get('LLM_PROVIDER') == LLMProvider.BEDROCK.value:
        return LLMProvider.BEDROCK
    else:
        return LLMProvider.HUGGING_FACE
