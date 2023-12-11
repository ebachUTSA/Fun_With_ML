from src.utils import insertSQLPandas, selectSQLPandas

df = selectSQLPandas('select * from test')
print(df)