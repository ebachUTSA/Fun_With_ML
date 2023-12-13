import requests
from bs4 import BeautifulSoup as bs

def getPageSource(url,justSource=True,userAgent='Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0'):
    headers = {'user-agent':userAgent}
    r = requests.get(url,headers=headers)
    if justSource:
        results = r.text
    else:
        results = r
    return results

def getSoup(content,parser='html.parser'):
    soup = bs(content,parser)
    return soup