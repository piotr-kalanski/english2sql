import streamlit as st
import pandas as pd
import time

from english2sql.vector_db.vector_db_adapter import create_vector_db_adapter_from_env
from english2sql.sql_generation.prompt_engineering import generate_query_prompt
from english2sql.sql_generation.llm_adapter import create_sql_generation_adapter_from_env


vector_db = create_vector_db_adapter_from_env()
llm = create_sql_generation_adapter_from_env()


def no_results():
    st.text("no results")

st.title('PoC for english to SQL')

prompt = st.chat_input("Insert your query here")
if prompt:
    st.subheader('Your prompt')
    with st.chat_message("user"):
        st.write(prompt)

    tables_tab, columns_tab, queries_tab, prompt_tab, sql_tab = st.tabs([
        "Related tables",
        "Related columns",
        "Similar queries",
        "Generated prompt",
        "Generated SQL"
    ])

    with st.spinner('Fetching table metadata'):
        related_tables = vector_db.find_related_tables(prompt)
        related_tables_df = pd.DataFrame(
            data=[
                {
                    "table": r.schema_name + '.' + r.table_name,
                    "columns": r.columns,
                    "distance": r.distance,
                }
                for r in related_tables
            ],
        )
    with tables_tab:
        if related_tables:
            st.write(related_tables_df)
        else:
            no_results()

    with st.spinner('Fetching related columns metadata'):
        related_columns = vector_db.find_related_columns(prompt)
        related_columns_df = pd.DataFrame(
            data=[
                {
                    "table": c.schema_name + '.' + c.table_name,
                    "column": c.column_name,
                    "description": c.description,
                    "distance": c.distance,
                }
                for c in related_columns
            ],
        )
    with columns_tab:
        if related_columns:
            st.write(related_columns_df)
        else:
            no_results()

    with st.spinner('Fetching similar queries'):
        similar_queries = vector_db.find_similar_queries(prompt)
    queries = '\n'.join([
        f'-- {q.description}\n{q.sql}\n'
        for q in similar_queries
    ])
    with queries_tab:
        if similar_queries:
            st.code(queries, language='sql', line_numbers=True)
        else:
            no_results()

    with st.spinner('Generating prompt to LLM'):
        llm_prompt = generate_query_prompt(prompt, related_tables, related_columns, similar_queries)
    with prompt_tab:
        st.text(llm_prompt)

    with st.spinner('Generating SQL query'):
        sql = llm.generate_sql_query(llm_prompt)
    with sql_tab:
        st.code(sql, language='sql', line_numbers=True)

    try:
        with st.spinner('Fetching data from database'):
            time.sleep(1)
            # TODO - query database
        st.subheader('Query result')
        st.write(pd.DataFrame(data={
            'id': [1,2,3],
            'name': ['n1', 'c2', 'c1']
        }))
    except Exception as e:
        st.exception(e)
