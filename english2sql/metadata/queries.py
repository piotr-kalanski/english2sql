import json
from typing import List
from pathlib import Path

from english2sql.metadata.model import QueryMetadata


def load_sample_queries(path: Path) -> List[QueryMetadata]:
    with open(path) as f:
        queries = json.load(f)
    return [
        QueryMetadata(**q)
        for q in queries
    ]
