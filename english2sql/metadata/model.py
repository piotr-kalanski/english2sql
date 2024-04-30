from dataclasses import dataclass
from typing import List


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


@dataclass
class QueryTableResult:
    schema_name: str
    table_name: str
    columns: str
    distance: float


@dataclass
class QueryColumnResult:
    schema_name: str
    table_name: str
    column_name: str
    type: str
    description: str
    accepted_values: str
    distance: float


@dataclass
class QueryMetadata:
    sql: str
    description: str
