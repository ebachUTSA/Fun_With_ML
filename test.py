from src.utils import insertSQLPandas, selectSQLPandas

df = selectSQLPandas('select * from secretsq.test')
print(df)