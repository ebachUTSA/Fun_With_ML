from src.utils import insertSQLPandas, selectSQLPandas, getPageSource, getSoup, getFedBenniesUrl

from src.config import Config
config = Config()

import pandas as pd

df = pd.read_excel(f"{config.base_directory}/data/zipcodes.xlsx")

insertSQLPandas(df,'zipcodes')

# urls = getFedBenniesUrl(78006)

# for url in urls:
#     print(url)

# baseUrl = 'https://www.shodan.io'
# url = 'https://www.shodan.io/search?query=port+3389'

# results = getPageSource(url)

# with open('c:/temp/test','w') as f:
#     f.write(results)

# soup = getSoup(results) #assuming it's html and not some other markup language

# for div in soup.find_all('div',{'class':'result'}):
#     a = div.find('a',{'class':['title']})
#     ip = a.text
#     banner = div.find('pre').text
#     print(ip,banner)
#     print('\n\n\n')

# nextUrl = soup.find('div',{'class':'pagination'})
# print(nextUrl.get('class'))
# nextUrl = nextUrl.find('a')
# nextUrl = nextUrl.text

# print(nextUrl)