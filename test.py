from src.utils import insertSQLPandas, selectSQLPandas, getPageSource, getSoup

baseUrl = 'https://www.shodan.io'
url = 'https://www.shodan.io/search?query=port+3389'

results = getPageSource(url)

soup = getSoup(results) #assuming it's html and not some other markup language

for div in soup.find_all('div',{'class':'result'}):
    a = div.find('a',{'class':['title']})
    ip = a.text
    banner = div.find('pre').text
    print(ip,banner)
    print('\n\n\n')