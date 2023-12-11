import pandas as pd
from src.utils.conn_manager import create_connection


def insertSQLPandas(df,table,if_exists='append',index=False,chunksize=10000,method='multi'):
    with create_connection() as connection:
        # df.to_sql(table,connection,if_exists=if_exists,index=index,chunksize=chunksize,method=method)
        df.to_sql(table,connection,if_exists=if_exists,index=index)

def selectSQLPandas(sql):
    with create_connection() as connection:
        results = pd.read_sql(sql,connection)
    return results
