from src.utils import insertSQLPandas, selectSQLPandas, getPageSource, getSoup

url = 'https://www.shodan.io/search?query=port+3389'

results = getPageSource(url)

soup = getSoup(results) #assuming it's html and not some other markup language

for a in soup.find_all('a'):
    print(a.get('href'))