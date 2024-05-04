import json
from pathlib import Path
from typing import List

from english2sql.metadata.model import (
    ColumnMetadata,
    TableMetadata,
    DatabaseMetadata,
)


def load_dbt_metadata(dir: Path, accepted_schemas: List[str]) -> DatabaseMetadata:
    catalog_json_file_path = dir / "catalog.json"
    manifest_json_file_path = dir / "manifest.json"

    with open(manifest_json_file_path) as f:
        dbt_manifest = json.load(f)

    with open(catalog_json_file_path) as f:
        dbt_catalog = json.load(f)

    all_accepted_values_tests = {}
    for _, node in dbt_manifest['nodes'].items():
        if 'test_metadata' in node:
            test_metadata = node['test_metadata']
            if test_metadata['name'] == 'accepted_values':
                kwargs = test_metadata['kwargs']
                accepted_value = kwargs['values']
                column_name = kwargs['column_name']
                model_name = node['attached_node']
                if model_name not in all_accepted_values_tests:
                    all_accepted_values_tests[model_name] = {}
                all_accepted_values_tests[model_name][column_name] = accepted_value

    tables_metadata = []
    nodes = dbt_manifest['nodes']
    for model_name in nodes.keys():
        node = nodes[model_name]

        if node['resource_type'] == 'model':
            columns_metadata = []
            columns = node['columns']
            for column_name in columns:
                column_metadata = columns[column_name]

                columns_metadata.append(
                    ColumnMetadata(
                        name=column_name,
                        description=column_metadata['description'],
                        type=dbt_catalog['nodes'].get(model_name, {}).get('columns', {}).get(column_name, {}).get('type'),
                        accepted_values=all_accepted_values_tests.get(model_name, {}).get(column_name, []),
                    )
                )

            if node['schema'] in accepted_schemas:
                tables_metadata.append(
                    TableMetadata(
                        database=node['database'],
                        schema=node['schema'],
                        table=node['name'],
                        description=node['description'],
                        columns=columns_metadata,
                    )
                )

    return DatabaseMetadata(tables_metadata)
