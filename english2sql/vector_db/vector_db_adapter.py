from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import List

from llama_index.core import Document, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding
import chromadb

from english2sql.metadata.model import DatabaseMetadata, TableVectorMetadata, ColumnVectorMetadata, QueryVectorMetadata
from english2sql.config import LLMProvider, llm_provider_from_env
from english2sql.utils import get_cleaned_model_id


class VectorDbAdapter(ABC):
    
    @abstractmethod
    def save_tables_metadata(self, db: DatabaseMetadata) -> None:
        ...

    @abstractmethod
    def save_columns_metadata(self, db: DatabaseMetadata) -> None:
        ...

    @abstractmethod
    def save_sample_queries(self, queries: List[QueryVectorMetadata]) -> None:
        ...

    @abstractmethod
    def find_related_tables(self, query: str) -> List[TableVectorMetadata]:
        ...

    @abstractmethod
    def find_related_columns(self, query: str) -> List[ColumnVectorMetadata]:
        ...

    @abstractmethod
    def find_similar_queries(self, query: str) -> List[QueryVectorMetadata]:
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
        self._tables_index = VectorStoreIndex.from_vector_store(
            self._tables_vector_store,
            embed_model=embed_model,
        )
        self._table_retriever = self._tables_index.as_retriever(similarity_top_k=20)

        self._columns_chroma_collection = db.get_or_create_collection("columns")
        self._columns_vector_store = ChromaVectorStore(chroma_collection=self._columns_chroma_collection)
        self._columns_index = VectorStoreIndex.from_vector_store(
            self._columns_vector_store,
            embed_model=embed_model,
        )
        self._columns_retriever = self._columns_index.as_retriever(similarity_top_k=30)

        self._queries_chroma_collection = db.get_or_create_collection("queries")
        self._queries_vector_store = ChromaVectorStore(chroma_collection=self._queries_chroma_collection)
        self._queries_index = VectorStoreIndex.from_vector_store(
            self._queries_vector_store,
            embed_model=embed_model,
        )
        self._queries_retriever = self._queries_index.as_retriever(similarity_top_k=3)


    def save_tables_metadata(self, db: DatabaseMetadata) -> None:
        self._save_items(
            db.tables,
            self._tables_vector_store,
            lambda t: (
                t.table + ' ' + t.description + ' ' + ','.join([c.name for c in t.columns])
            ),
            lambda t: TableVectorMetadata(
                schema_name=t.schema,
                table_name=t.table,
                columns=','.join([c.name for c in t.columns])
            ).dict(),
        )

    def save_columns_metadata(self, db: DatabaseMetadata) -> None:
        self._save_items(
            [
                (t,c)
                for t in db.tables
                for c in t.columns
            ],
            self._columns_vector_store,
            lambda t_c: t_c[1].name + ' ' + t_c[1].description,
            lambda t_c: ColumnVectorMetadata(
                schema_name=t_c[0].schema,
                table_name=t_c[0].table,
                column_name=t_c[1].name,
                type=t_c[1].type,
                description=t_c[1].description,
                accepted_values=','.join(t_c[1].accepted_values),
            ).dict(),
        )

    def save_sample_queries(self, queries: List[QueryVectorMetadata]) -> None:
        self._save_items(
            queries,
            self._queries_vector_store,
            lambda q: q.description,
            lambda q: QueryVectorMetadata(
                sql=q.sql,
                description=q.description,
            ).dict(),
        )

    def _save_items(
        self,
        items: List,
        vector_store: ChromaVectorStore,
        text_mapper,
        metadata_mapper,
    ):
        documents = [
            Document(
                text=text_mapper(item),
                metadata=metadata_mapper(item),
            )
            for item in items
        ]
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=self._embed_model
        )

    def find_related_tables(self, query: str) -> List[TableVectorMetadata]:
        return self._find_similar_items(
            query,
            self._table_retriever,
            0.4, # TODO - try different values
            TableVectorMetadata,
        )

    def find_related_columns(self, query: str) -> List[ColumnVectorMetadata]:
        return self._find_similar_items(
            query,
            self._columns_retriever,
            0.4, # TODO - try different values
            ColumnVectorMetadata,
        )

    def find_similar_queries(self, query: str) -> List[QueryVectorMetadata]:
        return self._find_similar_items(
            query,
            self._queries_retriever,
            0.4, # TODO - try different values
            QueryVectorMetadata,
        )

    def _find_similar_items(
        self,
        query: str,
        retriever: VectorIndexRetriever,
        similarity_cutoff: float,
        node_metadata_mapper
    ):
        top_n = retriever.retrieve(query)
        filtered = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff).postprocess_nodes(top_n)
        return [
            node_metadata_mapper(**n.metadata)
            for n in filtered
        ]        
        

def get_path_for_model_id(model_id: str) -> Path:
    cleaned_model_id = get_cleaned_model_id(model_id)
    return Path('.') / 'chroma_db' / cleaned_model_id  # TODO - main dir from static variable


def create_vector_db_adapter(
    provider: LLMProvider,
    model_id: str,
) -> VectorDbAdapter:
    if provider == LLMProvider.HUGGING_FACE:
        embed_model = HuggingFaceEmbedding(model_name=model_id)
    elif provider == LLMProvider.BEDROCK:
        embed_model = BedrockEmbedding(
            model=model_id
        )

    vector_db_path = get_path_for_model_id(model_id)
    vector_db_path.mkdir(parents=True, exist_ok=True)

    return ChromaDbVectorDbAdapter(
        path=str(vector_db_path),
        embed_model=embed_model
    )


def create_vector_db_adapter_from_env() -> VectorDbAdapter:
    provider = llm_provider_from_env()
    if provider == LLMProvider.HUGGING_FACE:
        model_id = os.environ.get('EMBEDDING_MODEL_ID', 'BAAI/bge-base-en-v1.5')
    elif provider == LLMProvider.BEDROCK:
        model_id = os.environ.get('EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v1')

    return create_vector_db_adapter(provider, model_id)
