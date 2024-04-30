from typing import List

from english2sql.metadata.model import QueryTableResult, QueryColumnResult, QueryMetadata

# prompt based on https://github.com/pinterest/querybook/blob/master/querybook/server/lib/ai_assistant/prompts/text_to_sql_prompt.py
_GENERATE_QUERY_PROMPT = """
You are a {dialect} expert.

Please help to generate a {dialect} query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions.

===Tables
{table_schemas}

===Related columns
{related_columns}

===Similar queries
{similar_queries}

===Response Guidelines
1. If the provided context is sufficient, please generate a valid query without any explanations for the question.
2. If the provided context is insufficient, please explain why it can't be generated.
3. Please use the most relevant table(s).
5. Please format the query before responding.
6. Please always respond with a valid well-formed JSON object with the following format

===Response Format
{{
    "query": "A generated SQL query when context is sufficient.",
    "explanation": "An explanation of failing to generate the query."
}}

===Question
{question}
"""

def generate_query_prompt(
    user_prompt: str,
    related_tables: List[QueryTableResult],
    related_columns: List[QueryColumnResult],
    similar_queries: List[QueryMetadata],
) -> str:
    return _GENERATE_QUERY_PROMPT.format(
        dialect="Redshift",
        table_schemas='\n'.join([
            f'- {t.schema_name}.{t.table_name}({t.columns})'
            for t in related_tables
        ]),
        related_columns='\n'.join([
            f"""- TABLE: {c.schema_name}.{c.table_name}
  COLUMN: {c.column_name}
  TYPE: {c.type}
  {'DESCRIPTION: ' if c.description else ''}{c.description}
  {'ACCEPTED VALUES: ' if c.accepted_values else ''}{c.accepted_values}"""
            for c in related_columns
        ]),
        similar_queries='\n'.join([
            f'- {q.description}: {q.sql}'
            for q in similar_queries
        ]),
        question=user_prompt,
    )
