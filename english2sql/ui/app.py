import streamlit as st
import pandas as pd
import time

from english2sql.vector_db.vector_db_adapter import create_vector_db_adapter_from_env
from english2sql.sql_generation.generate import generate_sql_query


def no_results():
    st.text("no results")

st.title('PoC for english to SQL')

prompt = st.chat_input("Insert your query here")
if prompt:
    st.subheader('Your prompt')
    with st.chat_message("user"):
        st.write(prompt)

    #system_message = st.chat_message("system")
    with st.spinner('Fetching table metadata'):
        vector_db = create_vector_db_adapter_from_env()
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
    st.subheader('Related tables')
    if related_tables:
        st.write(related_tables_df)
    else:
        no_results()

    with st.spinner('Fetching related columns metadata'):
        vector_db = create_vector_db_adapter_from_env()
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
    st.subheader('Related columns')
    if related_columns:
        st.write(related_columns_df)
    else:
        no_results()

    with st.spinner('Fetching similar queries'):
        vector_db = create_vector_db_adapter_from_env()
        similar_queries = vector_db.find_similar_queries(prompt)
        # similar_queries_df = pd.DataFrame(
        #     data=[
        #         {
        #             "sql": q.sql,
        #             "decription": q.description,
        #         }
        #         for q in similar_queries
        #     ],
        # )
    st.subheader('Similar queries')
    # st.write(similar_queries_df)
    queries = '\n'.join([
        f'-- {q.description}\n{q.sql}\n'
        for q in similar_queries
    ])
    if similar_queries:
        st.code(queries, language='sql', line_numbers=True)
    else:
        no_results()

    st.subheader('Generated SQL')
    with st.spinner('Generating SQL query'):
        sql = generate_sql_query(prompt, related_tables, related_columns, similar_queries, stream=False)
        #st.write_stream(sql)
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
