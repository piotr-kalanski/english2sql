import json
from pathlib import Path

import click
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from english2sql.config import LLMProvider
from english2sql.metadata.dbt_docs import load_dbt_metadata
from english2sql.metadata.queries import load_sample_queries
from english2sql.sql_generation.prompt_engineering import generate_query_prompt
from english2sql.sql_generation.llm_adapter import create_sql_generation_adapter_from_env
from english2sql.vector_db.vector_db_adapter import create_vector_db_adapter_from_env, ChromaDbVectorDbAdapter, get_path_for_model_id


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

    models = {
        LLMProvider.HUGGING_FACE.value: hf_models.split(',') if hf_models else [],
        LLMProvider.BEDROCK.value: bedrock_models.split(',') if bedrock_models else [],
    }

    click.echo(f"All models: {models}")

    for provider in models:
        for model_id in models[provider]:
            if provider == LLMProvider.HUGGING_FACE.value:
                embed_model = HuggingFaceEmbedding(model_name=model_id)
            elif provider == LLMProvider.BEDROCK.value:
                embed_model = BedrockEmbedding(model=model_id)

            vector_db_path = get_path_for_model_id(model_id)
            click.echo(f"[{model_id}] Vector db path: {vector_db_path}")
            vector_db_path.mkdir(parents=True, exist_ok=True)

            vector_db = ChromaDbVectorDbAdapter(
                path=str(vector_db_path),
                embed_model=embed_model
            )

            click.echo(f"[{model_id}] Ingesting table metadata")
            vector_db.save_tables_metadata(db)

            click.echo(f"[{model_id}] Ingesting columns metadata")
            vector_db.save_columns_metadata(db)

            click.echo(f"[{model_id}] Ingesting sample queries")
            vector_db.save_sample_queries(sample_queries)


if __name__ == '__main__':
    cli()
