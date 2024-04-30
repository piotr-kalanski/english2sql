import json
from typing import List
from pathlib import Path

from english2sql.metadata.model import QueryVectorMetadata


def load_sample_queries(path: Path) -> List[QueryVectorMetadata]:
    with open(path) as f:
        queries = json.load(f)
    return [
        QueryVectorMetadata(**q)
        for q in queries
    ]
