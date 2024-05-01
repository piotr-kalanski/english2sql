import json
from pathlib import Path
from typing import List, Tuple

import click
from llama_index.embeddings.bedrock import BedrockEmbedding
import pandas as pd

from english2sql.config import LLMProvider
from english2sql.metadata.dbt_docs import load_dbt_metadata
from english2sql.metadata.queries import load_sample_queries
from english2sql.sql_generation.prompt_engineering import generate_query_prompt
from english2sql.sql_generation.llm_adapter import SqlQueryGenerationAdapter, create_sql_generation_adapter_from_env, create_sql_generation_adapter
from english2sql.vector_db.vector_db_adapter import create_vector_db_adapter_from_env, ChromaDbVectorDbAdapter, create_vector_db_adapter


@click.group()
def cli():
    pass


@cli.command()
@click.argument("dbt_metadata_dir")
def ingest_dbt_metadata(dbt_metadata_dir: str):
    """Ingest dbt metadata - manifest.json and catalog.json"""

    click.echo("Loading dbt metadata")
    db = load_dbt_metadata(Path(dbt_metadata_dir))

    vector_db = create_vector_db_adapter_from_env()
    click.echo("Ingesting table metadata")
    vector_db.save_tables_metadata(db)

    click.echo("Ingesting columns metadata")
    vector_db.save_columns_metadata(db)


@cli.command()
@click.argument("sample_queries_path")
def ingest_sample_queries(sample_queries_path: str):
    """Ingest sample queries"""

    click.echo("Loading sample queries")
    sample_queries = load_sample_queries(Path(sample_queries_path))

    click.echo("Ingesting sample queries")
    vector_db = create_vector_db_adapter_from_env()
    vector_db.save_sample_queries(sample_queries)


@cli.command()
@click.argument("query")
def list_related_tables(query: str):
    """List related tables to query"""

    vector_db = create_vector_db_adapter_from_env()
    result = vector_db.find_related_tables(query)
    for r in result:
        print(r.table_name, ' ', r.columns)


@cli.command()
@click.argument("query")
def list_related_columns(query: str):
    """List related columns to query"""

    vector_db = create_vector_db_adapter_from_env()
    result = vector_db.find_related_columns(query)
    for r in result:
        print(r.table_name, ' ', r.column_name, ' ', r.description)


@cli.command()
@click.argument("query")
def list_similar_queries(query: str):
    """List similar queries to query"""

    vector_db = create_vector_db_adapter_from_env()
    result = vector_db.find_similar_queries(query)
    for r in result:
        click.echo('-- ' + r.description)
        click.echo(r.sql)
        click.echo('')


@cli.command()
@click.argument("query")
def generate_prompt(query: str):
    """Generate prompt to LLM from query"""

    click.echo("Retreiving related tables from Vector DB")
    vector_db = create_vector_db_adapter_from_env()
    related_tables = vector_db.find_related_tables(query)
    click.echo("Retreiving related columns from Vector DB")
    related_columns = vector_db.find_related_columns(query)
    click.echo("Retreiving similar queries from Vector DB")
    similar_queries = vector_db.find_similar_queries(query)
    click.echo("Generating prompt to LLM")
    llm_prompt = generate_query_prompt(query, related_tables, related_columns, similar_queries)
    click.echo("Generated prompt:")
    click.echo(llm_prompt)


@cli.command()
@click.argument("query")
def generate_sql_query(query: str):
    """Generate SQL from query"""

    click.echo("Retreiving related tables from Vector DB")
    vector_db = create_vector_db_adapter_from_env()
    related_tables = vector_db.find_related_tables(query)
    click.echo("Retreiving related columns from Vector DB")
    related_columns = vector_db.find_related_columns(query)
    click.echo("Retreiving similar queries from Vector DB")
    similar_queries = vector_db.find_similar_queries(query)
    click.echo("Generating prompt to LLM")
    llm_prompt = generate_query_prompt(query, related_tables, related_columns, similar_queries)
    click.echo("Generated SQL query:")
    llm = create_sql_generation_adapter_from_env()
    sql = llm.generate_sql_query(llm_prompt)
    click.echo(sql)


@cli.command()
def bedrock_embedding_models():
    """Print all supported Bedrock models for embeddings"""

    supported_models = BedrockEmbedding.list_supported_models()
    print(json.dumps(supported_models, indent=2))


@cli.command()
def bedrock_llms():
    """Print all supported Bedrock LLM models"""

    from llama_index.llms.bedrock.utils import BEDROCK_FOUNDATION_LLMS
    print(json.dumps(BEDROCK_FOUNDATION_LLMS, indent=2))


def _get_vector_db_for_each_model(
    hf_models: str, 
    bedrock_models: str
) -> List[Tuple[ChromaDbVectorDbAdapter, str]]:
    providers = [
        LLMProvider.HUGGING_FACE,
        LLMProvider.BEDROCK
    ]
    models = [
        hf_models.split(',') if hf_models else [],
        bedrock_models.split(',') if bedrock_models else []
    ]
    
    return [
        (
           create_vector_db_adapter(
                provider,
                model_id
            ),
            model_id 
        )
        for provider, models_for_provider in zip(providers, models)
        for model_id in models_for_provider    
    ]


def _get_sql_generation_adapter_for_each_model(
    hf_models: str, 
    bedrock_models: str
) -> List[Tuple[SqlQueryGenerationAdapter, str]]:
    providers = [
        LLMProvider.HUGGING_FACE,
        LLMProvider.BEDROCK
    ]
    models = [
        hf_models.split(',') if hf_models else [],
        bedrock_models.split(',') if bedrock_models else []
    ]
    
    return [
        (
           create_sql_generation_adapter(
                provider,
                model_id
            ),
            model_id 
        )
        for provider, models_for_provider in zip(providers, models)
        for model_id in models_for_provider    
    ]


@cli.command()
@click.argument("dbt_metadata_dir")
@click.argument("sample_queries_path")
@click.option('--hf-models') # BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5
@click.option('--bedrock-models')
def ingest_metadata_with_multiple_embedding_models(
    dbt_metadata_dir: str,
    sample_queries_path: str,
    hf_models: str, 
    bedrock_models: str
):
    """Ingest metadata with multiple embedding models for comparison"""

    click.echo("Loading dbt metadata")
    db = load_dbt_metadata(Path(dbt_metadata_dir))

    click.echo("Loading sample queries")
    sample_queries = load_sample_queries(Path(sample_queries_path))

    for vector_db, model_id in _get_vector_db_for_each_model(hf_models, bedrock_models):
        click.echo(f"[{model_id}] Ingesting table metadata")
        vector_db.save_tables_metadata(db)

        click.echo(f"[{model_id}] Ingesting columns metadata")
        vector_db.save_columns_metadata(db)

        click.echo(f"[{model_id}] Ingesting sample queries")
        vector_db.save_sample_queries(sample_queries)


@cli.command()
@click.argument("sample_queries_path")
@click.argument("output_path", default='output/related_items.json')
@click.option('--hf-models')
@click.option('--bedrock-models')
def find_related_items_for_multiple_embedding_models(
    sample_queries_path: str,
    output_path: str,
    hf_models: str, 
    bedrock_models: str
):
    """Find related items for each query and embedding model for comparison"""

    click.echo("Loading sample queries")
    sample_queries = load_sample_queries(Path(sample_queries_path))

    result = {}
    for vector_db, model_id in _get_vector_db_for_each_model(hf_models, bedrock_models):
        result[model_id] = {}
        for query_obj in sample_queries:
            query = query_obj.description
            click.echo(f"[{model_id}] - {query}")
            result[model_id][query] = {
                'tables': [
                    t.schema_name + '.' + t.table_name
                    for t in vector_db.find_related_tables(query)
                ],
                'columns': [
                    c.schema_name + '.' + c.table_name + '.' + c.column_name
                    for c in vector_db.find_related_columns(query)
                ],
                'queries': [
                    q.description
                    for q in vector_db.find_similar_queries(query)
                ],
            }
    
    with open(output_path, 'w') as out:
        out.write(json.dumps(result, indent=2))


@cli.command()
@click.argument("sample_queries_path")
@click.argument("output_path", default='output/generated_sqls.csv')
@click.option('--hf-embed-models')
@click.option('--bedrock-embed-models')
@click.option('--hf-llms')
@click.option('--bedrock-llms')
def generate_sql_for_multiple_models(
    sample_queries_path: str,
    output_path: str,
    hf_embed_models: str, 
    bedrock_embed_models: str,
    hf_llms: str,
    bedrock_llms: str,
):
    """Generate SQL for each model for comparison"""

    click.echo("Loading sample queries")
    sample_queries = load_sample_queries(Path(sample_queries_path))

    vector_dbs = _get_vector_db_for_each_model(hf_embed_models, bedrock_embed_models)
    llms = _get_sql_generation_adapter_for_each_model(hf_llms, bedrock_llms)

    result = []
    for vector_db, embed_model_id in vector_dbs:
        for query in sample_queries:
            click.echo(f"[{embed_model_id}] Retreiving related tables from Vector DB")
            related_tables = vector_db.find_related_tables(query.description)
            click.echo(f"[{embed_model_id}] Retreiving related columns from Vector DB")
            related_columns = vector_db.find_related_columns(query.description)
            click.echo(f"[{embed_model_id}] Retreiving similar queries from Vector DB")
            similar_queries = vector_db.find_similar_queries(query.description)
            click.echo(f"[{embed_model_id}] Generating prompt to LLM")
            llm_prompt = generate_query_prompt(query.description, related_tables, related_columns, similar_queries)
            for llm, llm_model_id in llms:
                click.echo(f"[{embed_model_id}][{llm_model_id}] Generating SQL query for prompt [{query.description}]")
                result.append({
                    "embed_model_id": embed_model_id,
                    "llm_model_id": llm_model_id,
                    "query.description": query.description,
                    "query.sql": query.sql,
                    "generated_sql": llm.generate_sql_query(llm_prompt)
                })

    pd.DataFrame(result).to_csv(output_path)


if __name__ == '__main__':
    cli()
