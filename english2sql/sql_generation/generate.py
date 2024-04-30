from typing import List

from english2sql.metadata.model import QueryTableResult, QueryColumnResult, QueryMetadata
from english2sql.sql_generation.llm_adapter import create_sql_generation_adapter_from_env
from english2sql.sql_generation.prompt_engineering import generate_query_prompt


def generate_sql_query(
    user_prompt: str,
    related_tables: List[QueryTableResult],
    related_columns: List[QueryColumnResult],
    similar_queries: List[QueryMetadata],
    stream: bool=False
):
    prompt_to_llm = generate_query_prompt(user_prompt, related_tables, related_columns, similar_queries)
    llm = create_sql_generation_adapter_from_env()
    if stream:
        return llm.stream_generate_sql_query(prompt_to_llm)
    else:
        return llm.generate_sql_query(prompt_to_llm)
