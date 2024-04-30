from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel

# dbt metadata:

@dataclass
class ColumnMetadata:
    name: str
    description: str
    type: str
    accepted_values: List[str]


@dataclass
class TableMetadata:
    database: str
    schema: str
    table: str
    description: str
    columns: List[ColumnMetadata]


@dataclass
class DatabaseMetadata:
    tables: List[TableMetadata]

# models to store metadata in vector db:

class TableVectorMetadata(BaseModel):
    schema_name: str
    table_name: str
    columns: str


class ColumnVectorMetadata(BaseModel):
    schema_name: str
    table_name: str
    column_name: str
    type: Optional[str]
    description: Optional[str]
    accepted_values: Optional[str]


class QueryVectorMetadata(BaseModel):
    sql: str
    description: str
