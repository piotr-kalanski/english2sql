from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from typing import List

from llama_index.core import Document, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding
import chromadb

from english2sql.metadata.model import DatabaseMetadata, QueryTableResult, QueryColumnResult, QueryMetadata
from english2sql.config import LLMProvider, llm_provider_from_env


class VectorDbAdapter(ABC):
    
    @abstractmethod
    def save_tables_metadata(self, db: DatabaseMetadata) -> None:
        ...

    @abstractmethod
    def save_columns_metadata(self, db: DatabaseMetadata) -> None:
        ...

    @abstractmethod
    def save_sample_queries(self, queries: List[QueryMetadata]) -> None:
        ...

    @abstractmethod
    def find_related_tables(self, query: str) -> List[QueryTableResult]:
        ...

    @abstractmethod
    def find_related_columns(self, query: str) -> List[QueryColumnResult]:
        ...

    @abstractmethod
    def find_similar_queries(self, query: str) -> List[QueryMetadata]:
        ...


class ChromaDbVectorDbAdapter(VectorDbAdapter):

    def __init__(
        self,
        path: Path,
        embed_model: BaseEmbedding,
    ):
        self._embed_model = embed_model
        db = chromadb.PersistentClient(path=path)
        self._tables_chroma_collection = db.get_or_create_collection("tables")
        self._tables_vector_store = ChromaVectorStore(chroma_collection=self._tables_chroma_collection)
        self._columns_chroma_collection = db.get_or_create_collection("columns")
        self._columns_vector_store = ChromaVectorStore(chroma_collection=self._columns_chroma_collection)
        self._queries_chroma_collection = db.get_or_create_collection("queries")
        self._queries_vector_store = ChromaVectorStore(chroma_collection=self._queries_chroma_collection)

    def save_tables_metadata(self, db: DatabaseMetadata) -> None:
        documents = [
            Document(
                text=t.table + ' ' + t.description, # TODO include columns and columns descriptions
                metadata={
                    'schema_name': t.schema,
                    'table_name': t.table, # TODO - static variables for table_name, columns
                    'columns': ','.join([c.name for c in t.columns])
                }
            )
            for t in db.tables
        ]
        storage_context = StorageContext.from_defaults(vector_store=self._tables_vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=self._embed_model
        )

    def save_columns_metadata(self, db: DatabaseMetadata) -> None:
        documents = [
            Document(
                text=c.name + ' ' + c.description,
                metadata={
                    'schema_name': t.schema,
                    'table_name': t.table, # TODO - static variables for table_name, columns
                    'column_name': c.name,
                    'type': c.type,
                    'description': c.description,
                    'accepted_values': ','.join(c.accepted_values),
                }
            )
            for t in db.tables
            for c in t.columns
        ]
        storage_context = StorageContext.from_defaults(vector_store=self._columns_vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=self._embed_model
        )

    def save_sample_queries(self, queries: List[QueryMetadata]) -> None:
        documents = [
            Document(
                text=q.description,
                metadata={
                    'sql': q.sql,
                    'description': q.description,
                }
            )
            for q in queries
        ]
        storage_context = StorageContext.from_defaults(vector_store=self._queries_vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=self._embed_model
        )

    def find_related_tables(self, query: str) -> List[QueryTableResult]:
        distance_threshold = 0.8 # TODO - try different values
        query_response = self._tables_chroma_collection.query(
            query_embeddings=[
                self._embed_model.get_query_embedding(query)
            ],
            n_results=20, # TODO - parameter
        )
        result = []
        for metadata, distance in zip(query_response['metadatas'][0], query_response['distances'][0]):
            if distance <= distance_threshold:
                nc = json.loads(metadata['_node_content'])
                node_metadata = nc['metadata']
                result.append(QueryTableResult(
                    schema_name=node_metadata['schema_name'],
                    table_name=node_metadata['table_name'], # TODO - static variables for table_name, columns
                    columns=node_metadata['columns'],
                    distance=distance,
                ))
        return result

    def find_related_columns(self, query: str) -> List[QueryColumnResult]:
        distance_threshold = 0.8 # TODO - try different values
        query_response = self._columns_chroma_collection.query(
            query_embeddings=[
                self._embed_model.get_query_embedding(query)
            ],
            n_results=20, # TODO - parameter
        )
        result = []
        for metadata, distance in zip(query_response['metadatas'][0], query_response['distances'][0]):
            if distance <= distance_threshold:
                nc = json.loads(metadata['_node_content'])
                node_metadata = nc['metadata']
                result.append(QueryColumnResult(
                    schema_name=node_metadata['schema_name'],
                    table_name=node_metadata['table_name'], # TODO - static variables for table_name, columns
                    column_name=node_metadata['column_name'],
                    type=node_metadata['type'],
                    description=node_metadata['description'],
                    accepted_values=node_metadata['accepted_values'],
                    distance=distance,
                ))
        return result

    def find_similar_queries(self, query: str) -> List[QueryMetadata]:
        distance_threshold = 0.7 # TODO - try different values
        query_response = self._queries_chroma_collection.query(
            query_embeddings=[
                self._embed_model.get_query_embedding(query)
            ],
            n_results=3, # TODO - parameter
        )
        result = []
        for metadata, distance in zip(query_response['metadatas'][0], query_response['distances'][0]):
            if distance <= distance_threshold:
                nc = json.loads(metadata['_node_content'])
                node_metadata = nc['metadata']
                result.append(QueryMetadata(
                    description=node_metadata['description'],
                    sql=node_metadata['sql'],
                ))
        return result
        

def create_vector_db_adapter_from_env() -> VectorDbAdapter:
    provider = llm_provider_from_env()
    if provider == LLMProvider.HUGGING_FACE:
        model_id = os.environ.get('EMBEDDING_MODEL_ID', 'BAAI/bge-base-en-v1.5')
        embed_model = HuggingFaceEmbedding(model_name=model_id)
    elif provider == LLMProvider.BEDROCK:
        model_id = os.environ.get('EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v1')
        # "amazon.titan-embed-text-v1",
        # "amazon.titan-embed-g1-text-02",
        # "cohere.embed-english-v3",
        # "cohere.embed-multilingual-v3"
        # "amazon.titan-embed-text-v1",
        # "amazon.titan-embed-g1-text-02",
        # "cohere.embed-english-v3",
        # "cohere.embed-multilingual-v3"
        embed_model = BedrockEmbedding(
            model=model_id
        )

    return ChromaDbVectorDbAdapter(
        path="chroma_db",  # TODO - from static variable
        embed_model=embed_model
    )
