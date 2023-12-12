from src.utils import insertSQLPandas, selectSQLPandas, getPageSource

url = ''

results = getPageSource(url)

print(results)